//! World — 고정 틱 스텝 오케스트레이션 (설계 §4 파이프라인, M2a: 단일스레드).
//!
//! step(dt): 브로드페이즈(canonical pairs) → 내로우페이즈(manifold) → prepare →
//! substep × N { 속도 적분 → warm start → solve(bias) → 위치 적분 → relax } →
//! restitution → (M2c: 임펄스 캐시 저장).
//! M2a 범위: 바디 제거·아일랜드·슬리핑·조인트는 후속 유닛. 커맨드 버퍼는 M2f와 함께.

use crate::body::{Body, BodyHandle, BodyType};
use crate::command::{Command, CommandReport, PendingCommand};
use crate::joint::{self, Joint};
use crate::solver::{self, CachedImpulse, ContactConstraint, Softness, SolverParams};
use maru_collide::{contact, Aabb, Broadphase, FeatureId, STATIC_BIT};
use maru_math::Vec3;
use std::collections::HashMap;

pub struct WorldParams {
    pub gravity: Vec3,
    pub substeps: u32,
    pub speculative_margin: f32,
    pub solver: SolverParams,
    /// 슬리핑 임계 (설계 §4.6): 이하 속도가 time_to_sleep 지속되면 island 단위 sleep.
    pub sleep_linear_threshold: f32,
    pub sleep_angular_threshold: f32,
    pub time_to_sleep: f32,
    /// 브로드페이즈 페어 예산 계수. 상한 = `max_pairs_per_body × 바디 수 + 1024`.
    /// 같은 자리에 겹친 N개 바디는 N²/2 페어를 만든다 — 입력은 작아도 wasm 메모리를
    /// 삼킬 수 있으므로 예산을 넘으면 `ResourceLimit`으로 스텝을 포기한다(리뷰 BLOCKER).
    pub max_pairs_per_body: u32,
    /// **속도 상한 (물리 파라미터)** — Jolt `mMaxLinearVelocity`·Box2D `b2_maxTranslation`과
    /// 같은 역할이다. 터널링과 폭주를 막는 정상 기능이며, 도달은 오류가 아니다.
    /// 이것이 있어야 파생 상태(속도·임펄스)가 유계가 되어 정책이 step에 대해 닫힌다 —
    /// 겹친 바디를 밀어내는 정상 물리도 이 상한을 향해 가속하기 때문이다.
    pub max_linear_velocity: f32,
    pub max_angular_velocity: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            // **8 서브스텝이 기본이다** (2026-07-26 실측 재보정). 4는 현실적인 스택의
            // 안정성 임계 **아래**였고, 그래서 성능까지 나빴다:
            //   40단 피라미드  substeps=4 → 8 ms/step (붕괴) / substeps=8 → 2 ms/step (안정)
            // 서브스텝 2배 비용은 **수면으로 인한 절감에 완전히 묻힌다** — 정착한 스택은
            // 비용이 0에 수렴하고(브로드페이즈 캐시), 못 자는 스택은 영원히 전부 계산한다.
            // 안정 봉투도 29단 → 60단으로 넓어진다.
            // 대가: **정착하지 않는 장면**(계속 움직이는 씬)에서는 2배 비용이 그대로 남는다.
            substeps: 8,
            speculative_margin: 0.05,
            solver: SolverParams::default(),
            sleep_linear_threshold: 0.03,
            sleep_angular_threshold: 0.05,
            time_to_sleep: 0.5,
            max_pairs_per_body: 64,
            max_linear_velocity: 500.0,
            max_angular_velocity: 100.0,
        }
    }
}

/// 공개 API 오류 (설계 §9·§8: 불변식은 경계에서 강제한다).
#[derive(Debug, PartialEq, Eq)]
pub enum ApiError {
    /// 제거됐거나 세대가 다른 핸들.
    StaleHandle,
    /// 비유한 값 또는 범위를 벗어난 인자.
    InvalidValue,
    /// static 바디는 불변 (설계 §9 — 움직이려면 Kinematic).
    ImmutableStatic,
    /// 자원 예산 초과 (브로드페이즈 페어 등). wasm에서 allocator가 먼저 죽는 것보다 낫다.
    ///
    /// **실패 계약(중요)**: 페어 수는 커맨드 적용 **뒤**의 배치에서만 알 수 있으므로,
    /// 이 오류가 나기 전에 pending 커맨드는 이미 적용되고 속도 클램프도 반영된다.
    /// 즉 "**커맨드는 적용됨, 물리 스텝은 건너뜀**"이 계약이다 — 재시도해도 커맨드가
    /// 사라지지는 않는다(이미 반영됐다). 롤백이 필요한 호스트는 `try_snapshot()`으로
    /// 스텝 전 상태를 잡아두고 실패 시 `restore`해야 한다. 엔진은 스텝마다 스냅샷을
    /// 뜨는 비용을 강제하지 않는다.
    ResourceLimit,
    /// 파생 상태가 상한을 넘어 포화됐다. **그 스텝의 물리는 신뢰할 수 없다**(리뷰 BLOCKER:
    /// 포화를 정상 처리로 삼키면 안 된다). 상태 자체는 안전하고 복원 가능하게 유지되므로
    /// 호스트는 계속 돌릴 수도, 되감을 수도 있다. 지원 범위 밖 입력의 신호다.
    StateOverflow,
}

pub struct World {
    /// **비공개** — 무검증 대입을 막는다(리뷰 MAJOR). 읽기는 `params()`, 변경은 `set_params()`.
    params: WorldParams,
    bodies: Vec<Body>,
    /// 슬롯 생존 여부. 죽은 슬롯은 재사용 대기(세대가 이미 증가한 상태).
    alive: Vec<bool>,
    /// **월드 범위를 벗어나 시뮬레이션에서 제외된** 바디(리뷰 BLOCKER).
    /// `asleep`과 섞으면 아일랜드 불변식이 깨진다 — 조인트로 연결된 둘 중 하나만
    /// 경계를 넘으면 "부분 sleep" 상태가 되고, 다음 스텝의 joint wake 폐쇄가 즉시 깨워
    /// 이동→고정→sleep→wake를 매 프레임 반복한다. OOB는 **자동 wake 대상이 아니며**,
    /// 호스트가 `set_pose`로 월드 안에 되돌려야 해제된다.
    oob: Vec<bool>,
    /// pending `AddBody`가 잡아둔 슬롯. `alive`도 `reserved`도 아닌 슬롯만 재사용한다.
    /// 예약 슬롯은 쿼리에 보이지 않지만 세대는 이미 증가해 옛 핸들이 되살아나지 않는다.
    reserved: Vec<bool>,
    generations: Vec<u32>,
    /// PCM 임펄스 캐시: canonical 페어 → (feature, 임펄스). 스텝 간 warm start (설계 §3).
    contact_cache: HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
    sleep_timer: Vec<f32>,
    asleep: Vec<bool>,
    /// 직전 스텝의 아일랜드 대표(설계 §4.6): wake는 **아일랜드 단위**여야 한다.
    /// 바디 단위 wake는 접촉/조인트 저장 순서에 따라 한 홉만 전파돼, 깨어난 이웃과
    /// 아직 잠든 바디 사이에 제약이 걸리며 속도만 펌핑되는 경로가 생긴다(리뷰 BLOCKER).
    island: Vec<u32>,
    joints: Vec<Joint>,
    /// pending 쓰기 커맨드 (설계 §7). **캐노니컬 상태** — 스냅샷에 포함된다.
    commands: Vec<PendingCommand>,
    last_report: CommandReport,
    diag: Diagnostics,
    /// 직전 스텝이 자원 예산으로 중단됐는가 (비캐노니컬 — 복원 시 초기화).
    resource_limit: bool,
    /// 브로드페이즈 페어 캐시 (비캐노니컬 — 순수 파생). 설계 §4.1이 "페어 집합은 **트리
    /// 비의존**"을 계약으로 두었으므로, 입력(자세·sweep·생존 집합)이 그대로면 재계산 결과가
    /// 비트 동일하다 → **재사용은 근사가 아니라 항등**이다. 전원이 잠든 스택에서 매 스텝
    /// 트리 두 개를 새로 짓던 비용이 사라진다(10단 실측: 브로드페이즈가 스텝의 39%였다).
    bp_pairs: Vec<(u32, u32)>,
    /// 캐시가 유효한가. 구조 변경·변이·적분이 있으면 무효화된다.
    bp_valid: bool,
    /// 캐시를 만들 때의 dt (sweep AABB가 dt에 비례하므로 바뀌면 무효).
    bp_dt: f32,
    /// 솔버 핫패스 스크래치 (비캐노니컬). 매 스텝 재할당하지 않으려고 월드가 보관한다.
    solver_state: solver::SolverState,
    inv_inertia_local: Vec<Vec3>,
    integrate_gate: Vec<bool>,
    gravity_gate: Vec<bool>,
    island_parent: Vec<usize>,
}

/// 읽기 전용 진단 (엔진 캐노니컬 상태가 아니다 — 스냅샷·해시에 포함되지 않는다).
#[derive(Clone, Copy, Default, Debug)]
pub struct Diagnostics {
    pub warm_hits: usize,
    pub warm_total: usize,
    pub max_penetration: f32,
    /// 속도 상한에 걸린 바디 수. **정상 기능**이므로 오류가 아니지만, 지속적으로 크면
    /// 장면 구성이나 파라미터를 의심할 신호다.
    pub velocity_clamped: usize,
    /// 이번 스텝에 **월드 범위를 벗어나 동결된** 바디 수. 정의된 정책이며 오류가 아니다.
    pub out_of_bounds: usize,
    /// 이번 스텝의 브로드페이즈 페어 수와 예산 — 예산이 정상 장면을 거부하지 않는지
    /// 호스트가 직접 확인할 수 있어야 한다.
    pub pairs: usize,
    pub pair_budget: usize,
    /// 이번 스텝이 브로드페이즈 캐시를 재사용했는가 (전원 수면 등 무변화 스텝).
    pub bp_reused: bool,
    /// 단계별 소요 (ns) — 프로파일용. [broadphase, narrowphase, solve, bookkeeping]
    pub phase_ns: [u64; 4],
    /// 이번 스텝에 상한으로 잘린 캐노니컬 값의 개수. **0이 정상이다** —
    /// 0이 아니면 지원 범위 밖 입력이거나 엔진 버그이며, 그 스텝의 물리는 신뢰할 수 없다.
    /// 서로 다른 초과 상태가 같은 포화값으로 합쳐지므로 상태 해시만으로는 구분되지 않는다.
    pub state_saturations: usize,
}

pub use maru_collide::{MAX_DT, MIN_DT};

/// 한 스텝이 잡을 수 있는 **총 스크래치 바이트** 상한.
///
/// 페어 **개수** 예산만으로는 부족하다: 페어 뒤에 constraint·키·HashSet·SoA가 따라붙어
/// 실제 메모리는 개수의 몇 배가 된다(리뷰 BLOCKER). 바이트로 상한을 정하고 그것으로
/// 페어 상한을 되돌려 계산하면, 어떤 입력이 와도 한 스텝의 스크래치가 이 값을 넘지 않는다.
/// 128MiB는 wasm 기본 힙에서 여유 있게 실패를 오류로 표현할 수 있는 크기다.
pub const MAX_STEP_SCRATCH_BYTES: usize = 128 << 20;

/// 파라미터 불변식 — **오류 반환형이 정본**이다(리뷰 PARTIAL BLOCKER: 호스트가 넘긴
/// 파라미터로 wasm 모듈을 trap시키면 안 된다). 패닉 버전은 이것을 감싼 편의 함수다.
///
/// 각 스칼라를 **차원에 맞는 범위**로 검사한다: 속도 임계는 `MAX_SPEED`, 길이·일반
/// 파라미터는 `MAX_INPUT`. NaN 임계는 비교가 항상 false라 게이트를 조용히 무력화하므로
/// 유한성부터 요구한다.
pub(crate) fn check_params(p: &WorldParams) -> Result<(), ApiError> {
    use maru_collide::{in_range, speed_in_range, vec_in_range};
    let ok = vec_in_range(p.gravity)
        && (1..=64).contains(&p.substeps)
        && in_range(p.speculative_margin)
        && p.speculative_margin >= 0.0
        && in_range(p.solver.contact_hertz)
        && p.solver.contact_hertz > 0.0
        && in_range(p.solver.contact_damping)
        && p.solver.contact_damping >= 0.0
        && speed_in_range(p.solver.max_bias_speed)
        && p.solver.max_bias_speed > 0.0
        && speed_in_range(p.solver.restitution_threshold)
        && p.solver.restitution_threshold >= 0.0
        && speed_in_range(p.sleep_linear_threshold)
        && p.sleep_linear_threshold >= 0.0
        && speed_in_range(p.sleep_angular_threshold)
        && p.sleep_angular_threshold >= 0.0
        && in_range(p.time_to_sleep)
        && p.time_to_sleep >= 0.0
        && speed_in_range(p.max_linear_velocity)
        && p.max_linear_velocity > 0.0
        && maru_collide::angular_in_range(p.max_angular_velocity)
        && p.max_angular_velocity > 0.0
        && p.max_pairs_per_body >= 1
        && p.max_pairs_per_body <= 4096;
    if ok {
        Ok(())
    } else {
        Err(ApiError::InvalidValue)
    }
}

fn validate_params(params: &WorldParams) {
    // 메시지는 진단용 — 어떤 필드가 걸렸는지 좁혀준다.
    assert!(
        maru_collide::vec_in_range(params.gravity),
        "gravity must be finite and within MAX_INPUT"
    );
    assert!(
        (1..=64).contains(&params.substeps),
        "substeps must be in 1..=64, got {}",
        params.substeps
    );
    assert!(
        maru_collide::speed_in_range(params.solver.restitution_threshold)
            && params.solver.restitution_threshold >= 0.0,
        "restitution_threshold must be finite, non-negative and within MAX_SPEED"
    );
    assert!(
        maru_collide::in_range(params.time_to_sleep) && params.time_to_sleep >= 0.0,
        "time_to_sleep must be finite and non-negative"
    );
    check_params(params).expect("invalid world params");
}

/// 앵커가 그 바디의 크기에 비례하는가. 마운트 포인트가 셰이프 밖에 있는 것은 정상이므로
/// 바운딩 반경의 배수까지 허용한다.
pub(crate) fn anchor_fits(b: &Body, anchor: Vec3) -> bool {
    const ANCHOR_SCALE: f32 = 4.0;
    let bb = b.shape.local_aabb();
    // **바디 원점에서 가장 먼 AABB 코너까지의 거리**를 쓴다. 앵커는 원점 기준 좌표이므로
    // 척도도 원점 기준이어야 한다(원점에서 떨어진 hull의 정상 표면 앵커 허용).
    //
    // 이 스칼라 한계만으로는 **이방성 바디의 조건수**를 막지 못한다
    // (`Box{half:(1000,1e-3,1e-3)}` + 앵커 `(0,1000,0)`). 그래서 축별로 더 조이는 대신
    // — 축별 결합은 로드형 진자(작은 추가 1m 밖에 매달림) 같은 정상 구성을 막는다 —
    // `try_add_joint`가 **실제 K 행렬의 역행렬 가능성**을 직접 확인한다(`joint_k_is_solvable`).
    let far = Vec3::new(
        bb.min.x.abs().max(bb.max.x.abs()),
        bb.min.y.abs().max(bb.max.y.abs()),
        bb.min.z.abs().max(bb.max.z.abs()),
    );
    let limit = (far.length() * ANCHOR_SCALE).max(maru_collide::MIN_SHAPE_EXTENT);
    anchor.length() <= limit
}

/// 게이트 슬라이스를 인덱스와 함께 순회한다 (clippy: 루프 변수로 인덱싱하지 말 것).
#[inline]
fn gravity_gate_iter(g: &[bool]) -> impl Iterator<Item = (usize, bool)> + '_ {
    g.iter().copied().enumerate()
}

impl World {
    /// 파라미터 불변식을 생성 경계에서 강제한다 (리뷰 MAJOR: 무검증 public 필드가
    /// dt/substeps·Softness 분모를 NaN/Inf로 만들 수 있었다).
    pub fn new(params: WorldParams) -> Self {
        validate_params(&params); // 상세 메시지 유지 (진단용)
        Self::new_unchecked(params)
    }

    /// 월드 생성 (오류 반환형 — **M3 바인딩의 정본 경로**). 호스트가 넘긴 파라미터로
    /// wasm 모듈을 trap시키지 않는다.
    pub fn try_new(params: WorldParams) -> Result<Self, ApiError> {
        check_params(&params)?;
        Ok(Self::new_unchecked(params))
    }

    fn new_unchecked(params: WorldParams) -> Self {
        Self {
            params,
            bodies: Vec::new(),
            alive: Vec::new(),
            oob: Vec::new(),
            reserved: Vec::new(),
            generations: Vec::new(),
            contact_cache: HashMap::new(),
            sleep_timer: Vec::new(),
            asleep: Vec::new(),
            island: Vec::new(),
            joints: Vec::new(),
            commands: Vec::new(),
            last_report: CommandReport::default(),
            diag: Diagnostics::default(),
            resource_limit: false,
            bp_pairs: Vec::new(),
            bp_valid: false,
            bp_dt: 0.0,
            solver_state: solver::SolverState::default(),
            inv_inertia_local: Vec::new(),
            integrate_gate: Vec::new(),
            gravity_gate: Vec::new(),
            island_parent: Vec::new(),
        }
    }

    #[inline]
    pub fn params(&self) -> &WorldParams {
        &self.params
    }

    /// 파라미터 교체 — 생성자와 **같은 불변식**을 통과해야 한다.
    pub fn set_params(&mut self, params: WorldParams) {
        validate_params(&params);
        self.bp_valid = false;
        self.params = params;
    }

    /// 파라미터 교체 (오류 반환형).
    pub fn try_set_params(&mut self, params: WorldParams) -> Result<(), ApiError> {
        check_params(&params)?;
        self.bp_valid = false;
        self.params = params;
        Ok(())
    }

    /// 재사용 가능한 최소 인덱스 슬롯 (살아있지도 예약되지도 않은 것).
    fn free_slot(&self) -> Option<usize> {
        (0..self.bodies.len()).find(|&i| !self.alive[i] && !self.reserved[i])
    }

    fn push_slot(&mut self, body: Body, alive: bool, reserved: bool) -> u32 {
        let index = self.bodies.len() as u32;
        self.bodies.push(body);
        self.alive.push(alive);
        self.oob.push(false);
        self.reserved.push(reserved);
        self.generations.push(0);
        self.sleep_timer.push(0.0);
        self.asleep.push(false);
        self.island.push(u32::MAX);
        index
    }

    /// `AddBody`를 큐에 쌓고 **핸들을 즉시 반환**한다(리뷰 BLOCKER: 커맨드 경로에서
    /// 새 바디를 가리킬 방법이 없었다). 슬롯을 예약하고 세대를 올려 ABA를 차단한다.
    /// 반환된 핸들은 flush 이후에 유효해지며, 그 전에도 같은 배치의 후속 커맨드에 쓸 수 있다.
    pub fn enqueue_add_body(&mut self, body: Body) -> BodyHandle {
        body.validate().expect("invalid body");
        let index = if let Some(index) = self.free_slot() {
            self.reserved[index] = true;
            self.oob[index] = false;
            // 슬롯 재사용 — 예약 시점에 세대를 올려 옛 핸들이 새 바디를 가리키지 못하게 한다
            self.generations[index] = self.generations[index].wrapping_add(1);
            self.sleep_timer[index] = 0.0;
            self.asleep[index] = false;
            self.island[index] = u32::MAX;
            index as u32
        } else {
            self.push_slot(body.clone(), false, true)
        };
        let h = BodyHandle {
            index,
            generation: self.generations[index as usize],
        };
        self.commands.push(PendingCommand::Add(h, Box::new(body)));
        h
    }

    /// 바디 추가. 빈 슬롯이 있으면 **최소 인덱스**를 재사용한다(이력 무관 canonical 배치).
    pub fn add_body(&mut self, body: Body) -> BodyHandle {
        body.validate().expect("invalid body");
        self.bp_valid = false;
        if let Some(index) = self.free_slot() {
            self.bodies[index] = body;
            self.alive[index] = true;
            self.oob[index] = false;
            self.sleep_timer[index] = 0.0;
            self.asleep[index] = false;
            self.island[index] = u32::MAX;
            return BodyHandle {
                index: index as u32,
                generation: self.generations[index],
            };
        }
        let index = self.push_slot(body, true, false);
        BodyHandle {
            index,
            generation: self.generations[index as usize],
        }
    }

    #[inline]
    fn resolve(&self, h: BodyHandle) -> Result<usize, ApiError> {
        let i = h.index as usize;
        if i < self.bodies.len() && self.alive[i] && self.generations[i] == h.generation {
            Ok(i)
        } else {
            Err(ApiError::StaleHandle)
        }
    }

    /// 바디 제거. 설계 §4.6의 **변이 커맨드 3단 원자 절차**를 따른다:
    /// ①변이 전 그래프에서 wake 대상 수집·적용 → ②구조 변경 → ③인접(조인트·PCM) 정리.
    /// 받침을 지웠는데 위 바디가 공중에 잠든 채 남는 것을 막는다.
    pub fn remove_body(&mut self, h: BodyHandle) -> Result<(), ApiError> {
        let i = self.resolve(h)?;
        // ① 변이 전: 이 바디를 참조하는 접촉/조인트로 도달하는 상대를 깨운다(아일랜드 단위)
        let mut neighbors: Vec<usize> = Vec::new();
        for (a, b) in self.contact_cache.keys() {
            let (a, b) = (*a as usize, (*b & !STATIC_BIT) as usize);
            if a == i {
                neighbors.push(b);
            } else if b == i {
                neighbors.push(a);
            }
        }
        for j in &self.joints {
            if j.ia() == i {
                neighbors.push(j.ib());
            } else if j.ib() == i {
                neighbors.push(j.ia());
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        for n in neighbors {
            if n < self.bodies.len() && self.alive[n] && self.bodies[n].is_dynamic() {
                self.wake(n);
            }
        }
        // ② 구조 변경
        self.bp_valid = false;
        self.alive[i] = false;
        self.reserved[i] = false;
        self.oob[i] = false;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.asleep[i] = false;
        self.island[i] = u32::MAX;
        // ③ 인접 정리
        self.joints.retain(|j| j.ia() != i && j.ib() != i);
        let id = i as u32;
        self.contact_cache
            .retain(|(a, b), _| *a != id && (*b & !STATIC_BIT) != id);
        Ok(())
    }

    /// 읽기 접근 (stale 핸들이면 패닉 — 테스트·내부용).
    pub fn body(&self, h: BodyHandle) -> &Body {
        &self.bodies[self.resolve(h).expect("stale handle")]
    }

    /// 읽기 접근 (오류 반환형 — 바인딩용).
    pub fn try_body(&self, h: BodyHandle) -> Result<&Body, ApiError> {
        Ok(&self.bodies[self.resolve(h)?])
    }

    pub fn is_alive(&self, h: BodyHandle) -> bool {
        self.resolve(h).is_ok()
    }

    /// 변이 진입점 — **불변식을 한 곳에서 강제한다**(리뷰 MAJOR: `body_mut`이 불변식 보유
    /// 구조체를 통째로 공개해 friction=-1·inv_mass=∞·NaN pose가 들어갈 수 있었다).
    /// 모든 변이는 유한성 검사 + wake + stale PCM 폐기를 거친다. static은 불변(설계 §9).
    fn mutate(&mut self, h: BodyHandle) -> Result<usize, ApiError> {
        let i = self.resolve(h)?;
        self.bp_valid = false;
        if self.bodies[i].body_type == BodyType::Static {
            return Err(ApiError::ImmutableStatic);
        }
        self.wake(i);
        let id = i as u32;
        self.contact_cache
            .retain(|(a, b), _| *a != id && (*b & !STATIC_BIT) != id);
        Ok(i)
    }

    pub fn set_linear_velocity(&mut self, h: BodyHandle, v: Vec3) -> Result<(), ApiError> {
        // 속도는 **속도 차원 범위**다(MAX_SPEED). 일반 입력 범위를 쓰면 리더가 거부하는
        // 값을 setter가 받아들여 같은 비대칭이 되살아난다.
        if !maru_collide::vec_speed_in_range(v) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].linvel = v;
        Ok(())
    }

    pub fn set_angular_velocity(&mut self, h: BodyHandle, v: Vec3) -> Result<(), ApiError> {
        if !maru_collide::vec_angular_in_range(v) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].angvel = v;
        Ok(())
    }

    /// 텔레포트 (설계 §9: 속도를 만들지 않는다 — kinematic 구동은 별도 API).
    pub fn set_pose(&mut self, h: BodyHandle, pose: maru_collide::Pose) -> Result<(), ApiError> {
        if !maru_collide::vec_in_range(pose.pos) || pose.rot.try_normalize().is_none() {
            return Err(ApiError::InvalidValue);
        }
        let i = self.resolve(h)?;
        if self.bodies[i].body_type == BodyType::Static {
            return Err(ApiError::ImmutableStatic);
        }
        // **자세 변경은 연결 조인트의 K 조건수를 바꾼다**(리뷰 BLOCKER: 삽입 시점에만
        // 검사하면, 두 이방성 바디가 서로 보완되는 자세에서 통과한 뒤 한쪽을 회전시켜
        // 임계 아래로 내려갈 수 있다). 새 자세로 미리 검증하고, 깨지면 **거부**한다 —
        // 조용히 풀리는 조인트보다 거부가 낫다.
        let old = self.bodies[i].pose;
        let new_pose = maru_collide::Pose {
            pos: pose.pos,
            rot: pose.rot.normalize(),
        };
        self.bodies[i].pose = new_pose;
        let ok = self
            .joints
            .iter()
            .filter(|j| j.ia() == i || j.ib() == i)
            .all(|j| crate::joint::k_is_solvable(&self.bodies, j));
        if !ok {
            self.bodies[i].pose = old;
            return Err(ApiError::InvalidValue);
        }
        self.bodies[i].pose = old;
        let i = self.mutate(h)?;
        self.bodies[i].pose = new_pose;
        // **텔레포트는 연속성을 끊는다** — 인접 조인트의 누적 임펄스를 지운다(리뷰 MAJOR).
        // 이전 자세의 월드 임펄스를 새 레버암에 warm start로 적용하면 인공 kick이 된다.
        // (PCM 캐시는 `mutate`가 이미 폐기했다.)
        for j in self.joints.iter_mut() {
            if j.ia() == i || j.ib() == i {
                j.clear_impulses();
            }
        }
        // 월드 안으로 되돌리는 것이 OOB 해제의 **유일한** 경로다(자동 복귀 없음).
        self.oob[i] = false;
        Ok(())
    }

    pub fn set_material(
        &mut self,
        h: BodyHandle,
        friction: f32,
        restitution: f32,
    ) -> Result<(), ApiError> {
        if !maru_collide::in_range(friction)
            || friction < 0.0
            || !(0.0..=1.0).contains(&restitution)
        {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].friction = friction;
        self.bodies[i].restitution = restitution;
        Ok(())
    }

    /// 살아있는 바디 수 (제거된 슬롯 제외).
    pub fn body_count(&self) -> usize {
        self.alive.iter().filter(|a| **a).count()
    }

    /// 슬롯 용량 (인덱스 상한 — 내부·스냅샷용).
    pub fn slot_count(&self) -> usize {
        self.bodies.len()
    }

    /// 커맨드 큐에 쌓는다 (설계 §7 정본 쓰기 경로). 적용은 다음 step 0단계 또는 flush_commands().
    /// **쿼리는 pending을 보지 않는다** — 즉시 반영이 필요하면 flush_commands()를 호출한다.
    ///
    /// **enqueue 시점에 검증한다**(리뷰 PARTIAL MAJOR). 예전에는 아무 커맨드나 받아들이면서
    /// `restore`의 구조 검증은 그런 pending을 거부해, **자기 월드가 만든 스냅샷이 자기에게
    /// 복원되지 않는** 모순이 있었다(제거된 바디의 옛 핸들로 SetVelocity를 enqueue → snapshot →
    /// restore 실패). 이제 큐에 들어간 커맨드는 항상 구조적으로 유효하고, 두 계약이 일치한다.
    ///
    /// 유효성은 **enqueue 시점** 기준이다: 앞선 커맨드가 대상을 지우면 flush에서 거부되어
    /// 리포트에 잡힌다(스텝을 중단시키지 않는다 — 설계 §3 스텝 원자성).
    pub fn enqueue(&mut self, cmd: Command) -> Result<(), ApiError> {
        self.validate_command(&cmd)?;
        self.commands.push(PendingCommand::Public(cmd));
        Ok(())
    }

    /// 커맨드 대상이 지금 큐에 넣기에 유효한가. 대상은 살아있거나(일반 커맨드)
    /// 예약된 슬롯(같은 배치의 AddBody가 채울 예정)이면 된다 — 세대는 어느 쪽이든 맞아야 한다.
    fn validate_command(&self, cmd: &Command) -> Result<(), ApiError> {
        let target = |h: &BodyHandle| -> Result<(), ApiError> {
            let i = h.index as usize;
            if i < self.bodies.len()
                && (self.alive[i] || self.reserved[i])
                && self.generations[i] == h.generation
            {
                Ok(())
            } else {
                Err(ApiError::StaleHandle)
            }
        };
        match cmd {
            Command::RemoveBody(h) => target(h),
            Command::SetLinearVelocity(h, v) => {
                if !maru_collide::vec_speed_in_range(*v) {
                    return Err(ApiError::InvalidValue);
                }
                target(h)
            }
            Command::SetAngularVelocity(h, v) => {
                if !maru_collide::vec_angular_in_range(*v) {
                    return Err(ApiError::InvalidValue);
                }
                target(h)
            }
            Command::SetPose(h, p) => {
                if !maru_collide::vec_in_range(p.pos) || p.rot.try_normalize().is_none() {
                    return Err(ApiError::InvalidValue);
                }
                target(h)
            }
            Command::SetMaterial(h, f, rest) => {
                if !maru_collide::in_range(*f) || *f < 0.0 || !(0.0..=1.0).contains(rest) {
                    return Err(ApiError::InvalidValue);
                }
                target(h)
            }
            Command::AddJoint(j) => {
                j.validate()?;
                if j.body_a.index == j.body_b.index {
                    return Err(ApiError::InvalidValue);
                }
                target(&j.body_a)?;
                target(&j.body_b)
            }
        }
    }

    pub fn pending_commands(&self) -> usize {
        self.commands.len()
    }

    /// 직전 커맨드 적용 결과(적용/거부 수). 거부는 스텝을 중단시키지 않는다.
    pub fn last_command_report(&self) -> CommandReport {
        self.last_report
    }

    /// pending 커맨드를 **enqueue 순서대로** 즉시 적용한다.
    /// 각 커맨드는 자신의 pre-mutation 그래프를 기준으로 3단 원자 절차를 수행한다.
    pub fn flush_commands(&mut self) -> CommandReport {
        let cmds = std::mem::take(&mut self.commands);
        let mut report = CommandReport::default();
        for cmd in cmds {
            let ok = match cmd {
                PendingCommand::Add(h, b) => {
                    let i = h.index as usize;
                    // 예약이 살아있고 세대가 맞을 때만 실현한다(스냅샷에서 온 바이트 방어)
                    if i < self.bodies.len()
                        && self.reserved[i]
                        && self.generations[i] == h.generation
                        && b.validate().is_ok()
                    {
                        // **파생 캐시 무효화는 여기서도 해야 한다**(리뷰 BLOCKER): 직접
                        // `add_body`만 무효화하고 이 경로를 빠뜨려, 빈 페어 캐시를 가진 월드가
                        // 커맨드로 바디를 얻어도 캐시를 재사용했다. 같은 스냅샷을 복원한
                        // 월드는 캐시가 없어 페어를 재빌드하므로 **미래가 갈렸다** — L0 위반.
                        self.bp_valid = false;
                        self.bodies[i] = *b;
                        self.reserved[i] = false;
                        self.alive[i] = true;
                        self.oob[i] = false;
                        self.sleep_timer[i] = 0.0;
                        self.asleep[i] = false;
                        self.island[i] = u32::MAX;
                        true
                    } else {
                        false
                    }
                }
                PendingCommand::Public(c) => match c {
                    Command::RemoveBody(h) => self.remove_body(h).is_ok(),
                    Command::SetLinearVelocity(h, v) => self.set_linear_velocity(h, v).is_ok(),
                    Command::SetAngularVelocity(h, v) => self.set_angular_velocity(h, v).is_ok(),
                    Command::SetPose(h, p) => self.set_pose(h, p).is_ok(),
                    Command::SetMaterial(h, f, r) => self.set_material(h, f, r).is_ok(),
                    Command::AddJoint(j) => self.try_add_joint(*j).is_ok(),
                },
            };
            if ok {
                report.applied += 1;
            } else {
                report.rejected += 1;
            }
        }
        self.last_report = report;
        report
    }

    /// **정책을 닫는 지점**(리뷰 BLOCKER). 입력 정책만으로는 `step`에 대해 닫히지 않는다:
    /// 합법인 `pos=0.9e9`·`v=0.9e9`가 한 스텝 뒤 1.125e9가 되고, 솔버는 큰 질량과 큰
    /// 접근 속도에서 입력 상한을 넘는 임펄스를 정당하게 만든다. 거부(패닉)는 릴리스에서
    /// 답이 아니므로 **캐노니컬 상태 전체를 상태 정책 안으로 클램프**한다 — 바디의 자세·속도,
    /// 조인트 누적 임펄스, PCM 캐시까지. 그 결과
    /// "엔진이 만들 수 있는 상태 = 리더가 받는 상태"가 항상 성립한다.
    ///
    /// 비유한 값도 여기서 정리된다(NaN→0, ±Inf→±MAX_STATE). 경계 검증이 완전하면
    /// 클램프가 실제로 값을 바꿀 일은 없다 — 디버그 빌드는 그것까지 확인한다.
    /// 속도 상한 적용 (물리 파라미터). **스텝 시작과 매 서브스텝 속도 적분 뒤**에 건다:
    /// 시작에 거는 이유는 복원된 스냅샷의 큰 속도가 솔버 산술에 들어가기 전에 잡히도록
    /// 하기 위함이고, 서브스텝마다 거는 이유는 겹침 해소가 스텝 안에서 가속하기 때문이다.
    fn clamp_velocities(&mut self) {
        let (lin, ang) = (
            self.params.max_linear_velocity,
            self.params.max_angular_velocity,
        );
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if !self.alive[i] || b.body_type == BodyType::Static {
                continue;
            }
            let l2 = b.linvel.length_sq();
            // 부정 비교는 **의도**다: NaN은 어떤 비교에도 false이므로 `l2 > lin*lin`으로
            // 쓰면 NaN 속도가 클램프를 그냥 통과한다. `!(l2 <= ...)`라야 NaN이 잡힌다.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(l2 <= lin * lin) {
                let n = b.linvel.length();
                b.linvel = if n.is_finite() && n > 0.0 {
                    b.linvel * (lin / n)
                } else {
                    Vec3::ZERO
                };
                self.diag.velocity_clamped += 1;
            }
            let a2 = b.angvel.length_sq();
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(a2 <= ang * ang) {
                let n = b.angvel.length();
                b.angvel = if n.is_finite() && n > 0.0 {
                    b.angvel * (ang / n)
                } else {
                    Vec3::ZERO
                };
                self.diag.velocity_clamped += 1;
            }
        }
    }

    /// 월드 범위를 벗어난 바디를 **경계에 고정하고 재운다**(리뷰 BLOCKER: 위치 정밀도
    /// 계약이 step에 대해 닫혀 있지 않았다). 10km 밖에서는 f32 간격이 최소 특징 크기를
    /// 넘어 충돌 판정이 의미를 잃으므로, 조용히 계속 계산하는 것이 가장 나쁘다.
    /// 동결은 관측 가능하고(`Diagnostics::out_of_bounds`) 상태는 복원 가능하게 남는다.
    /// 더 넓은 월드가 필요하면 호스트가 floating-origin으로 rebase해야 한다(v1 계약).
    fn enforce_world_bounds(&mut self) {
        let max = maru_collide::MAX_INPUT;
        for i in 0..self.bodies.len() {
            if !self.alive[i] || self.bodies[i].body_type == BodyType::Static {
                continue;
            }
            let p = self.bodies[i].pose.pos;
            if maru_collide::vec_in_range(p) {
                continue;
            }
            let clamp = |v: f32| {
                if v.is_nan() {
                    0.0
                } else {
                    v.clamp(-max, max)
                }
            };
            self.bodies[i].pose.pos = Vec3::new(clamp(p.x), clamp(p.y), clamp(p.z));
            self.bodies[i].linvel = Vec3::ZERO;
            self.bodies[i].angvel = Vec3::ZERO;
            // `asleep`이 아니라 **별도 상태**다 — 자동 wake 경로에 걸리면 안 된다.
            self.bp_valid = false;
            // **인접 조인트의 누적 임펄스를 지운다**(리뷰 MAJOR): OOB 동안 solve만 건너뛰면
            // 이탈 전 자세에서 쌓인 임펄스가 남고, 호스트가 `set_pose`로 복귀시킨 다음 스텝의
            // warm start가 그걸 새 자세에 그대로 적용해 큰 kick을 만든다.
            for j in self.joints.iter_mut() {
                if j.ia() == i || j.ib() == i {
                    j.clear_impulses();
                }
            }
            self.oob[i] = true;
            self.asleep[i] = false;
            self.sleep_timer[i] = 0.0;
            self.island[i] = u32::MAX;
            self.diag.out_of_bounds += 1;
        }
    }

    fn close_state_policy(&mut self) -> usize {
        use maru_collide::{saturate_state, saturate_state_vec};
        let mut hits = 0usize;
        macro_rules! sat {
            ($slot:expr) => {{
                let (v, changed) = saturate_state($slot);
                $slot = v;
                hits += changed as usize;
            }};
        }
        macro_rules! sat_vec {
            ($slot:expr) => {{
                let (v, changed) = saturate_state_vec($slot);
                $slot = v;
                hits += changed as usize;
            }};
        }
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if !self.alive[i] {
                continue;
            }
            sat_vec!(b.pose.pos);
            sat_vec!(b.linvel);
            sat_vec!(b.angvel);
        }
        for j in self.joints.iter_mut() {
            sat_vec!(j.linear_impulse);
            sat!(j.angular_impulse[0]);
            sat!(j.angular_impulse[1]);
        }
        for entry in self.contact_cache.values_mut() {
            for (_, imp) in entry.iter_mut() {
                sat!(imp.normal);
                sat!(imp.tangent[0]);
                sat!(imp.tangent[1]);
                sat_vec!(imp.tangent_world);
                sat_vec!(imp.position);
                // **직렬화되는 필드는 하나도 빠뜨리지 않는다**(리뷰 MAJOR): contact_normal이
                // NaN이 되면 스텝 뒤에도 남아 restore가 실패한다. 단위 벡터라 포화가 아니라
                // 정규화 실패 시 0으로 되돌린다(매칭 게이트가 자연히 끊긴다).
                if imp.contact_normal.try_normalize().is_none() {
                    imp.contact_normal = Vec3::ZERO;
                    hits += 1;
                }
            }
        }
        for t in self.sleep_timer.iter_mut() {
            let (v, changed) = saturate_state(*t);
            *t = v.max(0.0);
            hits += changed as usize;
        }
        hits
    }

    /// 직전 스텝 진단 스냅샷.
    pub fn diagnostics(&self) -> Diagnostics {
        self.diag
    }

    /// 잠들었는지. **stale 핸들은 false**(살아있지 않은 것은 잠든 것이 아니다).
    /// raw 인덱스 접근은 재사용된 슬롯의 남의 상태를 읽으므로 쓰지 않는다(리뷰 MAJOR).
    /// 월드 범위를 벗어나 시뮬레이션에서 제외됐는가. `set_pose`로만 해제된다.
    pub fn is_out_of_bounds(&self, h: BodyHandle) -> bool {
        self.resolve(h).map(|i| self.oob[i]).unwrap_or(false)
    }

    pub fn is_asleep(&self, h: BodyHandle) -> bool {
        self.resolve(h).map(|i| self.asleep[i]).unwrap_or(false)
    }

    /// 조인트 추가. 연결 바디를 깨운다 (edge 생성형 변이 — 설계 §4.6).
    pub fn add_joint(&mut self, joint: Joint) -> usize {
        self.try_add_joint(joint)
            .expect("joint endpoints must be distinct live bodies")
    }

    /// 조인트 추가 (오류 반환형). 엔드포인트는 **세대까지** 검증한다 — 인덱스만 보면
    /// 제거·재사용된 슬롯의 새 바디에 붙는다(리뷰 BLOCKER: ABA).
    pub fn try_add_joint(&mut self, mut joint: Joint) -> Result<usize, ApiError> {
        // 생성자를 통과했더라도 **삽입 경계에서 다시** 검증한다(리뷰 BLOCKER: 공개 필드
        // 대입으로 생성 후 축을 NaN으로 바꿀 수 있었고, `length_sq() < 0.5` 비교는
        // NaN에서 false라 그대로 통과했다).
        joint.validate()?;
        let (a, b) = (self.resolve(joint.body_a)?, self.resolve(joint.body_b)?);
        if a == b {
            return Err(ApiError::InvalidValue);
        }
        // **앵커는 그 바디의 실제 크기에 비례해야 한다**(리뷰 BLOCKER). 전역 상한만 두면
        // 반지름 1mm 구에 1000m 앵커를 붙일 수 있고, K 행렬의 고유 스케일이 2.4e4 : 6e16이
        // 되어 Mat3::inverse의 상대 det가 임계 아래로 떨어져 **0 행렬**을 돌려준다 —
        // 조인트가 오류 없이 조용히 풀린다. 레버암을 바디 크기에 묶으면 `r²·I⁻¹`가
        // `m⁻¹`과 같은 자릿수에 남아 조건수가 형상 스케일과 무관해진다.
        if !anchor_fits(&self.bodies[a], joint.anchor_a)
            || !anchor_fits(&self.bodies[b], joint.anchor_b)
        {
            return Err(ApiError::InvalidValue);
        }
        // **실제 K 행렬이 풀리는지 확인한다**(리뷰 BLOCKER). 형상·앵커 휴리스틱으로는
        // 이방성 조합을 다 막을 수 없다: `Box{half:(1000,1e-3,1e-3)}`에 앵커 `(0,1000,0)`이면
        // K가 `diag(4, 1, 1.5e12)`가 되어 정규화 det가 1e-24로 떨어지고 `Mat3::inverse`가
        // 0 행렬을 돌려준다 — 조인트가 **오류 없이 조용히 풀린다**.
        // 솔버가 쓰는 것과 **같은 함수**로 K를 만들어 같은 판정을 한다(휴리스틱이 아니라 실측).
        if !crate::joint::k_is_solvable(&self.bodies, &joint) {
            return Err(ApiError::InvalidValue);
        }
        self.wake(a);
        self.wake(b);
        joint.linear_impulse = Vec3::ZERO;
        joint.angular_impulse = [0.0, 0.0]; // 재사용 Joint 값의 잔여 임펄스 차단(리뷰 지적)
        self.joints.push(joint);
        Ok(self.joints.len() - 1)
    }

    /// 캐노니컬 상태 접근 (snapshot 모듈 전용).
    #[allow(clippy::type_complexity)]
    pub(crate) fn canonical_parts(
        &self,
    ) -> (
        &WorldParams,
        &Vec<Body>,
        &Vec<bool>,
        &Vec<bool>,
        &Vec<bool>,
        &Vec<u32>,
        &Vec<f32>,
        &Vec<bool>,
        &Vec<u32>,
        &Vec<Joint>,
        &HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
        &Vec<PendingCommand>,
        CommandReport,
    ) {
        (
            &self.params,
            &self.bodies,
            &self.alive,
            &self.oob,
            &self.reserved,
            &self.generations,
            &self.sleep_timer,
            &self.asleep,
            &self.island,
            &self.joints,
            &self.contact_cache,
            &self.commands,
            self.last_report,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_parts(
        &mut self,
        params: WorldParams,
        bodies: Vec<Body>,
        alive: Vec<bool>,
        oob: Vec<bool>,
        reserved: Vec<bool>,
        generations: Vec<u32>,
        sleep_timer: Vec<f32>,
        asleep: Vec<bool>,
        island: Vec<u32>,
        joints: Vec<Joint>,
        cache: HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
        commands: Vec<PendingCommand>,
        last_report: CommandReport,
    ) {
        self.params = params;
        self.alive = alive;
        self.oob = oob;
        self.reserved = reserved;
        self.generations = generations;
        self.bodies = bodies;
        self.sleep_timer = sleep_timer;
        self.asleep = asleep;
        self.island = island;
        self.joints = joints;
        self.contact_cache = cache;
        self.commands = commands;
        self.last_report = last_report;
        // 진단은 캐노니컬 상태가 아니다 — 복원 후 **미래 동작에 영향을 주면 안 된다**
        // (stale `state_saturations`가 다음 `try_step`을 StateOverflow로 만들던 경로).
        self.diag = Diagnostics::default();
        self.resource_limit = false;
        self.bp_valid = false;
    }

    /// 캐노니컬 스냅샷 (스텝 경계에서만 — 설계 §3). 트리는 제외(파생 상태).
    /// 캐노니컬 스냅샷 — **총량 정책 초과 시 명시적으로 실패한다**. M3 바인딩은 이 경로를 쓴다.
    pub fn try_snapshot(&self) -> Result<Vec<u8>, crate::snapshot::SnapshotError> {
        let mut w = crate::snapshot::Writer(Vec::new());
        crate::snapshot::write_world(self, &mut w)?;
        Ok(w.0)
    }

    /// 편의 래퍼 (테스트·진단용). 정책 초과는 패닉 — 실패를 다루려면 `try_snapshot`.
    pub fn snapshot(&self) -> Vec<u8> {
        self.try_snapshot().expect("snapshot exceeds size policy")
    }

    /// 스냅샷 복원. 재생은 원 실행과 같은 타깃에서 L0 비트 동일.
    /// **신뢰하지 않는 바이트를 받는다** — 검증 실패 시 월드는 그대로 두고 오류를 반환한다.
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), crate::snapshot::SnapshotError> {
        crate::snapshot::read_world(self, bytes)
    }

    /// L0 상태 해시 (리플레이 CI 비교).
    /// 캐노니컬 상태 해시 (오류 반환형 — M3는 이 경로만 노출한다).
    pub fn try_state_hash(&self) -> Result<u64, crate::snapshot::SnapshotError> {
        Ok(crate::snapshot::state_hash(&self.try_snapshot()?))
    }

    pub fn state_hash(&self) -> u64 {
        crate::snapshot::state_hash(&self.snapshot())
    }

    /// 아일랜드 단위 wake (설계 §4.6). 직전 스텝의 아일랜드 대표가 같은 동적 바디를 모두 깨운다.
    fn wake(&mut self, i: usize) {
        let island = self.island[i];
        self.asleep[i] = false;
        self.sleep_timer[i] = 0.0;
        if island == u32::MAX {
            return;
        }
        for k in 0..self.bodies.len() {
            if self.alive[k] && self.island[k] == island && self.bodies[k].is_dynamic() {
                self.asleep[k] = false;
                self.sleep_timer[k] = 0.0;
            }
        }
    }

    /// 고정 틱 스텝. 호스트 프레임과 무결합 (설계 §4).
    pub fn step(&mut self, dt: f32) {
        self.try_step(dt).expect("invalid dt");
    }

    /// 스텝 (오류 반환형 — **M3 바인딩의 정본 경로**).
    ///
    /// dt 계약: `0 < dt <= MAX_DT`. 이보다 큰 dt는 speculative margin·CCD·soft 계수가
    /// 전부 의미를 잃는다. **호스트가 accumulator로 나눠 넣을 책임**이다(고정틱 엔진) —
    /// 탭 복귀처럼 큰 dt가 들어오면 여러 스텝으로 쪼개거나 프레임을 버려야 한다.
    /// 엔진이 조용히 클램프하면 시뮬레이션이 실시간과 어긋난 채 진행돼 더 나쁘다.
    /// 위반은 wasm trap이 아니라 `ApiError::InvalidValue`로 표현된다.
    pub fn try_step(&mut self, dt: f32) -> Result<(), ApiError> {
        // 문서와 구현을 하나로(리뷰 MAJOR): 0과 음수는 **합법이 아니다**. 일시정지는
        // 호스트가 step을 부르지 않는 것으로 표현한다 — 0을 무연산으로 받아주면
        // "성공했는데 아무 일도 없음"이 오류와 구분되지 않는다.
        if !(dt.is_finite() && (MIN_DT..=MAX_DT).contains(&dt)) {
            return Err(ApiError::InvalidValue);
        }
        self.step_inner(dt);
        if self.resource_limit {
            return Err(ApiError::ResourceLimit);
        }
        if self.diag.state_saturations > 0 {
            return Err(ApiError::StateOverflow);
        }
        Ok(())
    }

    fn step_inner(&mut self, dt: f32) {
        // 자원 초과 플래그는 **매 스텝 초기화**한다(리뷰 BLOCKER): 생성·복원에서만 지우면
        // 한 번 초과한 뒤 바디를 지우거나 예산을 올려 실제로 성공해도 계속
        // `ResourceLimit`을 돌려주면서 상태만 앞으로 나간다.
        self.resource_limit = false;
        // 진단 초기화와 속도 클램프는 **무엇보다 먼저** 한다(리뷰 BLOCKER ×2):
        // ① stale diagnostics가 남으면 restore 후 `try_step` 반환값이 달라져 L0가 깨진다
        //    (빈 월드 조기 반환 경로 포함).
        // ② 복원된 큰 속도가 브로드페이즈 sweep AABB를 부풀려 O(N²) 페어를 만든다 —
        //    `try_step`이 오류를 돌려주기 전에 wasm OOM이 먼저 온다.
        self.diag = Diagnostics::default();
        // 0) 커맨드 적용 (설계 §4 파이프라인 0단계 — 쓰기의 정본 경로).
        //    **빈 월드 판정보다 먼저** 해야 한다: 빈 월드에 AddBody만 큐잉하면 영원히
        //    적용되지 않아 '다음 step에서 적용' 계약이 깨진다(리뷰 NOT_RESOLVED).
        if !self.commands.is_empty() {
            self.flush_commands();
        }
        // 속도 클램프는 **flush 뒤·브로드페이즈 앞**이어야 한다(리뷰 BLOCKER): 스텝 시작에만
        // 걸면 pending `SetLinearVelocity`가 그 뒤에 적용돼 sweep AABB를 부풀린 채
        // 브로드페이즈로 들어간다(O(N²) 페어 → 오류 반환 전 OOM).
        self.clamp_velocities();
        if self.bodies.is_empty() {
            return;
        }
        let n_sub = self.params.substeps.max(1);
        let h = dt / n_sub as f32;
        let inv_h = 1.0 / h;
        // contact hertz는 서브스텝 주파수의 1/4을 넘지 않게 (Box2D v3 관행)
        let hertz = self.params.solver.contact_hertz.min(0.25 * inv_h);
        let soft = Softness::new(hertz, self.params.solver.contact_damping, h);
        // 정적/키네마틱 상대 접촉은 2배 강성 (Box2D v3 staticSoftness) — 무한 질량 상대라
        // 더 단단히 잡아도 안정하고, 지면 침하가 스택 전체 침하의 시작점이 된다.
        let static_soft = Softness::new(2.0 * hertz, self.params.solver.contact_damping, h);

        // 1) 브로드페이즈 — canonical 페어 (트리 비의존 계약은 maru-collide가 보장)
        let mut dyn_items: Vec<(u32, Aabb)> = Vec::new();
        let mut static_items: Vec<(u32, Aabb)> = Vec::new();
        for (i, b) in self.bodies.iter().enumerate() {
            if !self.alive[i] {
                continue; // 제거된 슬롯
            }
            let local = b.shape.local_aabb();
            // **회전된 AABB를 제대로 계산한다.** 예전에는 로컬 원점에서 가장 먼 코너까지의
            // 거리를 세 축 모두에 썼는데(회전 불변이지만 매우 헐겁다), 반지름 0.5 구가
            // 반크기 0.866 큐브가 되어 축당 1.73배·부피 5.2배로 부풀었다.
            // 실측: 3000구 더미에서 페어 44,240개인데 실제 접점은 3,640개 — 12배 과잉이었다.
            //
            // 정확한 계산: 중심은 `pos + R·c`, 반크기는 `|R|·h` (|R|=성분별 절대값).
            // 원점에서 벗어난 hull도 중심을 함께 옮기므로 누락되지 않는다.
            let c = (local.max + local.min) * 0.5;
            let hl = (local.max - local.min) * 0.5;
            let (ax, ay, az) = (
                b.pose.rot.rotate(Vec3::new(1.0, 0.0, 0.0)),
                b.pose.rot.rotate(Vec3::new(0.0, 1.0, 0.0)),
                b.pose.rot.rotate(Vec3::new(0.0, 0.0, 1.0)),
            );
            let ext = Vec3::new(
                ax.x.abs() * hl.x + ay.x.abs() * hl.y + az.x.abs() * hl.z,
                ax.y.abs() * hl.x + ay.y.abs() * hl.y + az.y.abs() * hl.z,
                ax.z.abs() * hl.x + ay.z.abs() * hl.y + az.z.abs() * hl.z,
            );
            let world = Aabb::from_center(b.pose.pos + b.pose.rot.rotate(c), ext);
            // OOB는 **시뮬레이션에서 완전히 빠진다** — 브로드페이즈에도 넣지 않는다.
            // static tree에 넣으면 PCM 키에 STATIC_BIT가 붙는데 실제 타입은 Dynamic이라
            // 스냅샷 구조 검증(`STATIC_BIT == (type==Static)`)이 **자기 스냅샷을 거부**한다
            // (리뷰 BLOCKER). 월드 밖 물체와의 접촉을 흉내 내는 것보다 계약을 하나로 두는 편이
            // 정직하다: OOB는 없는 것처럼 취급되고, 호스트가 `set_pose`로 되돌려야 살아난다.
            if self.oob[i] {
                continue;
            }
            match b.body_type {
                BodyType::Static => static_items.push((i as u32, world)),
                _ => {
                    let vel_dt = b.linvel * dt;
                    dyn_items.push((
                        i as u32,
                        world.expand_sweep(vel_dt, self.params.speculative_margin),
                    ));
                }
            }
        }
        let _t_bp = std::time::Instant::now();
        // 예산은 **살아있는 바디 수** 기준(슬롯 수가 아니라) + 절대 개수 상한 + **바이트 상한**.
        // 개수만 묶으면 페어 뒤에 따라붙는 constraint·키·HashSet·SoA가 남는다 —
        // 한 스텝이 잡을 수 있는 총 바이트를 정해 그것으로 페어 상한을 되돌려 계산한다.
        let live = self.alive.iter().filter(|a| **a).count();
        let per_pair_bytes = core::mem::size_of::<(u32, u32)>()          // pairs
            + core::mem::size_of::<ContactConstraint>()                  // constraints
            + core::mem::size_of::<(u32, u32)>()                         // constraint_keys
            + 48; // live_keys(HashSet) + PCM 엔트리 여유
        let byte_cap = MAX_STEP_SCRATCH_BYTES / per_pair_bytes.max(1);
        let budget = (self.params.max_pairs_per_body as usize)
            .saturating_mul(live)
            .saturating_add(1024)
            .min(maru_collide::MAX_PAIRS)
            .min(byte_cap);
        self.diag.pair_budget = budget;
        // **아무것도 움직이지 않았으면 트리를 다시 짓지 않는다.** 페어 집합은 트리 비의존
        // 계약(설계 §4.1)이라 입력이 같으면 결과가 비트 동일하다 — 재사용은 항등이다.
        let pairs = if self.bp_valid && self.bp_dt == dt {
            self.diag.bp_reused = true;
            std::mem::take(&mut self.bp_pairs)
        } else {
            let bp = Broadphase::build(dyn_items, static_items);
            let Some(p) = bp.canonical_pairs_bounded(budget) else {
                self.resource_limit = true;
                return;
            };
            p
        };
        self.diag.pairs = pairs.len();
        // 조인트 연결 쌍은 충돌 제외 (collide_connected=false 기본)
        let mut excluded: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        if excluded.try_reserve(self.joints.len()).is_err() {
            self.resource_limit = true;
            return;
        }
        excluded.extend(
            self.joints
                .iter()
                .filter(|j| !j.collide_connected)
                .map(|j| {
                    let (a, b) = (j.ia() as u32, j.ib() as u32);
                    (a.min(b), a.max(b))
                }),
        );

        // 1.5) **wake 폐쇄를 제약 생성보다 먼저 끝낸다** (리뷰 MAJOR).
        // 조인트 edge와 접촉 edge를 **번갈아 닫힐 때까지** 돌린다: 조인트로 깨어난 바디가
        // 접촉 상대를 깨우고, 그 상대가 다시 조인트 상대를 깨우는 연쇄가 한 패스로는
        // 닫히지 않는다. 조인트 wake를 접촉 제약 생성 뒤에 두면 그 프레임은 지지 없이
        // 중력만 받는다(리뷰 MAJOR: 2.5단계 잔존).
        let joint_pairs: Vec<(usize, usize)> = self
            .joints
            .iter()
            // **OOB 끝점은 union하지 않는다**(리뷰 MAJOR): 두 정상 바디가 같은 OOB에
            // 각각 매달리면 비활성 조인트를 통해 한 아일랜드로 묶여 함께 자고 깨며,
            // 정상 바디의 대표가 OOB 인덱스가 될 수 있다.
            .filter(|j| !self.oob[j.ia()] && !self.oob[j.ib()])
            .map(|j| (j.ia(), j.ib()))
            .collect();
        // kinematic은 **항상 활성**이다 — 처방 속도로 움직이므로 잠든 상대를 깨워야 한다.
        let is_active = |i: usize, this: &Self| {
            !this.oob[i]
                && match this.bodies[i].body_type {
                    BodyType::Kinematic => true,
                    BodyType::Dynamic => !this.asleep[i],
                    BodyType::Static => false,
                }
        };
        for _ in 0..=joint_pairs.len() {
            let mut changed = false;
            for &(a, b) in &joint_pairs {
                for (x, y) in [(a, b), (b, a)] {
                    if is_active(x, self)
                        && self.bodies[y].is_dynamic()
                        && self.asleep[y]
                        && !self.oob[y]
                    {
                        self.wake(y);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // 단일 패스로 섞으면 pair 순서에 따라 이미 스킵된 하위 접촉이 이번 스텝에 제약을
        // 얻지 못한다: 잠든 스택 (1,ground)/(1,2)/(2,3)이 먼저 스킵된 뒤 (3,4)가 아일랜드를
        // 깨우면 그 프레임은 지지망 없이 중력만 적분된다.
        // 아일랜드 단위 wake이므로 한 바퀴로 폐쇄된다(전파는 wake() 내부에서 완결).
        for &(a_id, b_raw) in &pairs {
            let is_static_pair = b_raw & STATIC_BIT != 0;
            let b_plain = b_raw & !STATIC_BIT;
            if excluded.contains(&(a_id.min(b_plain), a_id.max(b_plain))) {
                continue;
            }
            let (a, b) = (a_id as usize, b_plain as usize);
            let active = |i: usize, this: &Self| {
                !this.oob[i]
                    && match this.bodies[i].body_type {
                        BodyType::Kinematic => true,
                        BodyType::Dynamic => !this.asleep[i],
                        BodyType::Static => false,
                    }
            };
            let sleeping_dyn = |i: usize, this: &Self| {
                this.bodies[i].is_dynamic() && this.asleep[i] && !this.oob[i]
            };
            let a_sleep = sleeping_dyn(a, self);
            let b_sleep = !is_static_pair && sleeping_dyn(b, self);
            // static은 **비활성** anchor다(설계 §4.6). 정적 상대를 활성으로 치면 바닥에 얹힌
            // 스택이 매 스텝 깨어나 영영 잠들지 못한다.
            let b_active = !is_static_pair && active(b, self);
            let wake_a = a_sleep && b_active;
            let wake_b = b_sleep && active(a, self);
            if !wake_a && !wake_b {
                continue;
            }
            // 실제 접촉이 있을 때만 깨운다 (AABB 근접만으로 깨우면 스택이 영영 못 잔다)
            if contact(
                &self.bodies[a].shape,
                &self.bodies[a].pose,
                &self.bodies[b].shape,
                &self.bodies[b].pose,
                self.params.speculative_margin,
            )
            .is_some()
            {
                if wake_a {
                    self.wake(a);
                }
                if wake_b {
                    self.wake(b);
                }
            }
        }

        self.diag.phase_ns[0] = _t_bp.elapsed().as_nanos() as u64;
        let _t_np = std::time::Instant::now();
        // 2) 내로우페이즈 → 제약 준비 (A=첫 인덱스, B=둘째 — canonical 순서가 곧 해석 순서)
        //    PCM 캐시에서 feature 일치 접점의 임펄스를 warm load (설계 §3)
        // 제약 컨테이너도 페어 수 기준으로 **미리 fallible 예약**한다 — 접촉 페어마다
        // 암묵 성장이 일어나면 그 재할당이 abort가 된다.
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut constraint_keys: Vec<(u32, u32)> = Vec::new();
        if constraints.try_reserve(pairs.len()).is_err()
            || constraint_keys.try_reserve(pairs.len()).is_err()
        {
            self.bp_pairs = pairs;
            self.resource_limit = true;
            return;
        }
        let empty: Vec<(FeatureId, CachedImpulse)> = Vec::new();
        let mut live_keys: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        if live_keys.try_reserve(pairs.len()).is_err() {
            self.bp_pairs = pairs;
            self.resource_limit = true;
            return;
        }
        for &(a_id, b_raw) in &pairs {
            let key = (a_id, b_raw);
            let is_static_pair = b_raw & STATIC_BIT != 0;
            let b_plain = b_raw & !STATIC_BIT;
            if excluded.contains(&(a_id.min(b_plain), a_id.max(b_plain))) {
                continue;
            }
            let (a_id, b_id) = (a_id as usize, b_plain as usize);
            // 양쪽 모두 비활성이면 제약 없음. wake는 위 1.5에서 이미 끝났다.
            let a_asleep = !self.bodies[a_id].is_dynamic() || self.asleep[a_id];
            let b_asleep = is_static_pair || !self.bodies[b_id].is_dynamic() || self.asleep[b_id];
            let a_inactive = self.oob[a_id]
                || match self.bodies[a_id].body_type {
                    BodyType::Kinematic => false,
                    BodyType::Dynamic => self.asleep[a_id],
                    BodyType::Static => true,
                };
            let b_inactive = self.oob[b_id]
                || match self.bodies[b_id].body_type {
                    BodyType::Kinematic => false,
                    BodyType::Dynamic => is_static_pair || self.asleep[b_id],
                    BodyType::Static => true,
                };
            if a_inactive && b_inactive {
                live_keys.insert(key); // 잠든 페어의 PCM은 보존한다(아래 5))
                continue;
            }
            let _ = (a_asleep, b_asleep);
            let (sa, pa) = (&self.bodies[a_id].shape, &self.bodies[a_id].pose);
            let (sb, pb) = (&self.bodies[b_id].shape, &self.bodies[b_id].pose);
            if let Some(m) = contact(sa, pa, sb, pb, self.params.speculative_margin) {
                if !m.points.is_empty() {
                    let cached = self.contact_cache.get(&key).unwrap_or(&empty);
                    for p in &m.points {
                        self.diag.max_penetration = self.diag.max_penetration.max(-p.separation);
                    }
                    let c = solver::prepare(&self.bodies, a_id, b_id, &m, cached, dt);
                    // 진단은 **실제 warm load**를 센다(feature ID 일치만 세면 generic 접점의
                    // 위치·법선 게이트가 매칭을 끊어도 100%로 보인다 — 거짓 지표).
                    self.diag.warm_total += c.points.len();
                    self.diag.warm_hits += c.warm_matched as usize;
                    constraints.push(c);
                    constraint_keys.push(key);
                    live_keys.insert(key);
                }
            }
        }

        self.diag.phase_ns[1] = _t_np.elapsed().as_nanos() as u64;
        let _t_sv = std::time::Instant::now();
        // 3) TGS-soft 서브스텝 루프
        let g = self.params.gravity;
        // 조인트는 접촉보다 강성 (Box2D v3: 2×contact hertz, ζ=2)
        let joint_soft = Softness::new(
            (2.0 * self.params.solver.contact_hertz).min(0.25 * inv_h),
            2.0,
            h,
        );
        // **fallible 선점은 캐노니컬 변이 전에 모두 끝낸다**(리뷰 BLOCKER).
        // 예전에는 `mem::take(&mut self.joints)` 뒤에 예약 실패로 조기 반환해
        // **모든 조인트가 삭제**됐고, 더 늦은 실패는 솔버 write-back·PCM 갱신이 끝난 뒤라
        // 물리는 전진했는데 `ResourceLimit`을 돌려줬다.
        let nb_pre = self.bodies.len();
        if self.solver_state.reserve_for(nb_pre).is_err()
            || self.inv_inertia_local.try_reserve(nb_pre).is_err()
            || self.integrate_gate.try_reserve(nb_pre).is_err()
            || self.gravity_gate.try_reserve(nb_pre).is_err()
            || self.island_parent.try_reserve(nb_pre).is_err()
        {
            self.bp_pairs = pairs;
            self.resource_limit = true;
            return;
        }
        let mut joints = std::mem::take(&mut self.joints);
        // 조인트도 접촉과 같은 비활성 게이트 (설계 §4.6 level-trigger): 양쪽 모두
        // 비활성(sleeping/static/kinematic)이면 스킵 — 잠든 바디에 warm-start 임펄스를
        // 주입하면 위치 적분이 없는 채 속도만 펌핑돼 정착한 체인이 되살아난다(실증:
        // t=6s vmax 0.006 정착 → 슬립 → t=8s 4.5 폭주).
        let active_joints: Vec<usize> = joints
            .iter()
            .enumerate()
            .filter(|(_, j)| {
                // kinematic도 활성 — 구동 중인 kinematic에 매달린 조인트를 스킵하면
                // 체인이 처방 운동을 따라가지 못한다(리뷰 MAJOR).
                let awake = |i: usize| {
                    !self.oob[i]
                        && match self.bodies[i].body_type {
                            BodyType::Kinematic => true,
                            BodyType::Dynamic => !self.asleep[i],
                            BodyType::Static => false,
                        }
                };
                // 한쪽이라도 OOB면 조인트 자체를 비활성화한다 — OOB에 임펄스를 주면
                // "시뮬레이션에서 빠진다"는 계약이 깨지고, 상대만 끌려간다.
                !self.oob[j.ia()] && !self.oob[j.ib()] && (awake(j.ia()) || awake(j.ib()))
            })
            .map(|(i, _)| i)
            .collect();
        // **솔버는 압축 상태 배열 위에서 돈다**(설계 §3 SoA 핫패스). `Body`를 제약 순서로
        // 무작위 접근하면 셰이프·재질까지 딸린 뚱뚱한 구조체가 캐시를 밀어낸다 —
        // 실측 점당 73ns에 산술은 40 flops 남짓이었다. 솔버가 실제로 쓰는 6개 배열만
        // 뽑아 돌리고 루프가 끝난 뒤 한 번만 되쓴다.
        let n_bodies = self.bodies.len();
        let mut st = std::mem::take(&mut self.solver_state);
        // 위에서 선점을 마쳤으므로 여기서는 실패하지 않는다(용량이 이미 확보돼 있다).
        st.fill_from(&self.bodies, &self.alive, &self.oob);
        let inv_i_local = &mut self.inv_inertia_local;
        inv_i_local.clear();
        inv_i_local.extend(self.bodies.iter().map(|b| b.inv_inertia_local));
        // 적분 대상 게이트도 스크래치에 재사용한다(루프에서 매번 판정하지 않는다).
        let integrate = &mut self.integrate_gate;
        let gravity_on = &mut self.gravity_gate;
        integrate.clear();
        gravity_on.clear();
        for i in 0..n_bodies {
            let ok = self.alive[i]
                && self.bodies[i].body_type != BodyType::Static
                && !self.asleep[i]
                && !self.oob[i];
            integrate.push(ok);
            gravity_on.push(ok && self.bodies[i].is_dynamic());
        }
        let inv_i_local = &self.inv_inertia_local;
        let integrate = &self.integrate_gate;
        let gravity_on = &self.gravity_gate;
        let (max_lin, max_ang) = (
            self.params.max_linear_velocity,
            self.params.max_angular_velocity,
        );
        for _ in 0..n_sub {
            for (i, on) in gravity_gate_iter(gravity_on) {
                if on {
                    st.lin[i] += g * h;
                }
            }
            st.refresh_inv_i(inv_i_local);
            for c in constraints.iter_mut() {
                solver::refresh_anchors(&st, c);
            }
            // warm start + bias 솔브 (조인트 먼저 — Box2D 순서)
            for &ji in &active_joints {
                joint::warm_start(&mut st, &joints[ji]);
            }
            for c in constraints.iter_mut() {
                solver::warm_start(&mut st, c);
            }
            for &ji in &active_joints {
                joint::solve(
                    &mut st,
                    &mut joints[ji],
                    Some((
                        &joint_soft,
                        self.params.solver.max_bias_speed,
                        self.params.max_angular_velocity,
                    )),
                );
            }
            for c in constraints.iter_mut() {
                let sf = if st.dynamic[c.a] && st.dynamic[c.b] {
                    &soft
                } else {
                    &static_soft
                };
                solver::solve(
                    &mut st,
                    c,
                    inv_h,
                    Some((sf, self.params.solver.max_bias_speed)),
                );
            }
            // 속도 상한 — 위치 적분 **전에** 걸어야 변위도 유계가 된다
            for (i, on) in gravity_gate_iter(integrate) {
                if !on {
                    continue;
                }
                let l2 = st.lin[i].length_sq();
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(l2 <= max_lin * max_lin) {
                    let n = st.lin[i].length();
                    st.lin[i] = if n.is_finite() && n > 0.0 {
                        st.lin[i] * (max_lin / n)
                    } else {
                        Vec3::ZERO
                    };
                    self.diag.velocity_clamped += 1;
                }
                let a2 = st.ang[i].length_sq();
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(a2 <= max_ang * max_ang) {
                    let n = st.ang[i].length();
                    st.ang[i] = if n.is_finite() && n > 0.0 {
                        st.ang[i] * (max_ang / n)
                    } else {
                        Vec3::ZERO
                    };
                    self.diag.velocity_clamped += 1;
                }
            }
            // 위치 적분 (dynamic + kinematic, sleeping 제외)
            for (i, on) in gravity_gate_iter(integrate) {
                if on {
                    st.pos[i] += st.lin[i] * h;
                    st.rot[i] = st.rot[i].integrate(st.ang[i], h);
                }
            }
            // relax — bias 에너지 제거 (위치 적분으로 자세가 바뀌었으므로 갱신)
            st.refresh_inv_i(inv_i_local);
            for c in constraints.iter_mut() {
                solver::refresh_anchors(&st, c);
            }
            for &ji in &active_joints {
                joint::solve(&mut st, &mut joints[ji], None);
            }
            for c in constraints.iter_mut() {
                solver::solve(&mut st, c, inv_h, None);
            }
        }

        // 4) restitution 패스 (임팩트 게이트)
        for c in constraints.iter_mut() {
            solver::apply_restitution(&mut st, c, self.params.solver.restitution_threshold);
        }

        st.write_back(&mut self.bodies, &self.alive);
        self.solver_state = st;

        // 이번 스텝에 **적분된 바디가 하나도 없으면** 자세가 그대로이므로 페어 집합도 그대로다.
        // (kinematic도 적분 대상이라 구동 중이면 자연히 무효화된다.)
        let moved = self.integrate_gate.iter().any(|x| *x);
        self.bp_pairs = pairs;
        self.bp_valid = !moved;
        self.bp_dt = dt;

        // 5) PCM 캐시 갱신. **잠든 페어의 엔트리는 보존**한다 — 전체 clear는 첫 수면 스텝에
        //    지속 매니폴드를 날려, 나중에 깨어날 때 warm start 없이 거칠게 재정착한다(리뷰 지적).
        //    이번 스텝에 살아있지 않은(페어 소멸) 키만 폐기.
        self.contact_cache.retain(|k, _| live_keys.contains(k));
        for (c, key) in constraints.iter().zip(&constraint_keys) {
            let entry: Vec<(FeatureId, CachedImpulse)> = c
                .points
                .iter()
                .map(|p| {
                    (
                        p.feature,
                        CachedImpulse {
                            normal: p.normal_impulse,
                            tangent: p.tangent_impulse,
                            tangent_world: c.tangents[0] * p.tangent_impulse[0]
                                + c.tangents[1] * p.tangent_impulse[1],
                            position: p.position,
                            contact_normal: c.normal,
                        },
                    )
                })
                .collect();
            self.contact_cache.insert(*key, entry);
        }

        // 6) 아일랜드 union-find (동적-동적 edge만 — static은 anchor 전용, 설계 §4.3)
        //    + island 단위 일괄 sleep (설계 §4.6)
        let n = self.bodies.len();
        // 선점은 스텝 시작에 끝났다 — 여기서 실패할 수 없다.
        let mut parent = std::mem::take(&mut self.island_parent);
        parent.clear();
        parent.extend(0..n);
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // 경로 압축
                x = parent[x];
            }
            x
        }
        let union = |parent: &mut Vec<usize>, x: usize, y: usize| {
            let (rx, ry) = (find(parent, x), find(parent, y));
            if rx != ry {
                let (lo, hi) = (rx.min(ry), rx.max(ry));
                parent[hi] = lo; // canonical: 낮은 인덱스가 대표 (설계 §4.3)
            }
        };
        for c in &constraints {
            if self.bodies[c.a].is_dynamic() && self.bodies[c.b].is_dynamic() {
                union(&mut parent, c.a, c.b);
            }
        }
        // 조인트 edge (설계 §4.3 — hinge 체인이 바디별 아일랜드로 갈라지면 안 됨)
        for (a, b) in &joint_pairs {
            if self.bodies[*a].is_dynamic() && self.bodies[*b].is_dynamic() {
                union(&mut parent, *a, *b);
            }
        }
        // 아일랜드 대표를 다음 스텝의 wake 전파용으로 저장.
        // **잠든 바디는 직전 대표를 보존한다**: 잠든 쌍은 제약을 만들지 않으므로(위 level-trigger)
        // 여기서 재계산하면 잠든 스택이 낱개 대표로 분해되고, 그 상태를 스냅샷하면 복원 후에도
        // 분해된 채 남아 wake가 다시 한 홉만 전파된다(리뷰 BLOCKER). 잠든 동안 멤버십은
        // 변할 수 없으므로(변하려면 깨어나야 한다) 보존이 곧 정답이다.
        for i in 0..n {
            // OOB는 시뮬레이션에서 빠진 상태이므로 아일랜드에도 속하지 않는다
            if !self.alive[i] || !self.bodies[i].is_dynamic() || self.oob[i] {
                self.island[i] = u32::MAX;
            } else if !self.asleep[i] {
                self.island[i] = find(&mut parent, i) as u32;
            }
        }
        // 스크래치를 월드로 돌려놓는다(다음 스텝에서 재할당하지 않도록).
        self.island_parent = std::mem::take(&mut parent);
        let mut parent = std::mem::take(&mut self.island_parent);
        let lin2 = self.params.sleep_linear_threshold * self.params.sleep_linear_threshold;
        let ang2 = self.params.sleep_angular_threshold * self.params.sleep_angular_threshold;
        for i in 0..n {
            if self.alive[i] && self.bodies[i].is_dynamic() && !self.asleep[i] && !self.oob[i] {
                let low = self.bodies[i].linvel.length_sq() < lin2
                    && self.bodies[i].angvel.length_sq() < ang2;
                self.sleep_timer[i] = if low { self.sleep_timer[i] + dt } else { 0.0 };
            }
        }
        // island 전원이 임계 시간을 넘겨야 일괄 sleep (부분 sleep 금지)
        let mut island_min: HashMap<usize, f32> = HashMap::new();
        for i in 0..n {
            if self.alive[i] && self.bodies[i].is_dynamic() && !self.asleep[i] && !self.oob[i] {
                let root = find(&mut parent, i);
                let e = island_min.entry(root).or_insert(f32::INFINITY);
                *e = e.min(self.sleep_timer[i]);
            }
        }
        for i in 0..n {
            // OOB는 sleep 대상이 아니다 — `oob ⇒ !asleep`이 v4 복원 불변식이다
            // (time_to_sleep=0인 합법 월드에서 같은 스텝에 둘 다 참이 되던 경로).
            if self.alive[i] && self.bodies[i].is_dynamic() && !self.asleep[i] && !self.oob[i] {
                let root = find(&mut parent, i);
                if island_min[&root] >= self.params.time_to_sleep {
                    self.asleep[i] = true;
                    self.bodies[i].linvel = Vec3::ZERO;
                    self.bodies[i].angvel = Vec3::ZERO;
                }
            }
        }
        self.island_parent = parent;

        self.joints = joints;

        // 7) denormal 위생 (설계 §8 flush 지점)
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if self.alive[i] && b.is_dynamic() {
                b.linvel = Vec3::new(
                    maru_math::flush_tiny(b.linvel.x),
                    maru_math::flush_tiny(b.linvel.y),
                    maru_math::flush_tiny(b.linvel.z),
                );
                b.angvel = Vec3::new(
                    maru_math::flush_tiny(b.angvel.x),
                    maru_math::flush_tiny(b.angvel.y),
                    maru_math::flush_tiny(b.angvel.z),
                );
            }
        }
        // **스텝 말 최종 클램프**: 서브스텝 루프 뒤의 relax·restitution 패스가 속도를 다시
        // 바꾸므로, 여기서 한 번 더 걸어야 저장되는 상태가 입력 범위 불변식을 만족한다
        // (리더가 `Body::validate`를 그대로 쓰는 근거).
        self.diag.phase_ns[2] = _t_sv.elapsed().as_nanos() as u64;
        let _t_bk = std::time::Instant::now();
        self.clamp_velocities();
        self.enforce_world_bounds();
        self.diag.state_saturations = self.close_state_policy();
        self.diag.phase_ns[3] = _t_bk.elapsed().as_nanos() as u64;
    }
}
