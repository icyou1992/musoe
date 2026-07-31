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
use crate::solver_block;
use maru_collide::{
    any_contact, for_each_contact, try_shape_cast_dispatch, Aabb, Broadphase, FeatureId, STATIC_BIT,
};
use maru_math::Vec3;
use std::collections::HashMap;

/// 정수 키 전용 결정적 해셔 (FxHash 계열 곱셈-시프트).
///
/// 기본 SipHash가 **페어당 5회** 돈다: `excluded.contains` · `contact_cache.get` ·
/// `live_keys.insert` · 스텝 끝의 `retain` · `insert`. 구 더미 3000(페어 33k)에서
/// 이것만으로 narrow의 절반 가까이를 먹었다 — 정작 구-구 접촉 산술은 20 flops다.
/// 키는 (u32,u32) 64비트 정수쌍이라 곱셈 한 번이면 분포가 충분하다.
///
/// HashDoS는 대상이 아니다: 키는 **엔진이 부여한 바디 슬롯 인덱스**이고 호스트가 값을
/// 직접 정하지 못한다(핸들은 slot+generation이고 슬롯은 순차 할당이다).
/// 결정성: 해셔가 바뀌면 이터레이션 순서가 바뀌지만, 순서에 의존하는 두 지점
/// (스냅샷 PCM 직렬화 · `remove_body`의 이웃 수집)은 이미 사용 전에 정렬한다.
#[derive(Default, Clone, Copy)]
pub struct IntHasher(u64);

impl std::hash::Hasher for IntHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u32(b as u32);
        }
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        self.0 = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(K);
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.write_u32(n as u32);
        self.write_u32((n >> 32) as u32);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

pub type IntBuild = std::hash::BuildHasherDefault<IntHasher>;
pub type IntMap<K, V> = HashMap<K, V, IntBuild>;
pub type IntSet<K> = std::collections::HashSet<K, IntBuild>;

/// 질의 필터.
///
/// 물리 충돌과 **같은 양방향 그룹 규칙**을 쓴다 — 총알이 뚫고 지나가야 할 것은
/// 광선에도 안 맞아야 자연스럽고, 규칙이 갈리면 호스트가 같은 개념을 두 벌로 관리한다.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryFilter {
    /// 질의자의 그룹 비트 (membership 겸 filter로 본다).
    pub groups: u32,
    /// 센서(트리거)도 맞힐 것인가. 기본은 **제외** — 총알이 트리거에 막히면 안 된다.
    pub include_sensors: bool,
    /// 이 물체는 건너뛴다 (자기 자신을 맞히는 것 방지).
    pub exclude: Option<BodyHandle>,
}

impl QueryFilter {
    /// 전부 통과 (센서 제외).
    pub const ALL: Self = Self {
        groups: u32::MAX,
        include_sensors: false,
        exclude: None,
    };
    pub fn groups(mut self, g: u32) -> Self {
        self.groups = g;
        self
    }
    pub fn with_sensors(mut self) -> Self {
        self.include_sensors = true;
        self
    }
    pub fn exclude(mut self, h: BodyHandle) -> Self {
        self.exclude = Some(h);
        self
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self::ALL
    }
}

/// 접촉 이벤트 — 두 물체가 닿기 시작했거나 떨어졌다.
///
/// "닿았다"의 기준은 **접촉 매니폴드가 생겼는가**다(스펙큘레이티브 마진 안이면 생긴다).
/// 정확히 침투했을 때만 세면 빠른 물체가 프레임을 건너뛰며 이벤트를 흘린다.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContactEvent {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub kind: ContactEventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactEventKind {
    /// 이번 스텝에 새로 닿았다.
    Started,
    /// 이번 스텝에 떨어졌다 (한쪽이 사라진 경우도 포함).
    Stopped,
}

/// **CCD linear cast가 실제로 절단한 순간** — 스텝 안에서 언제·어디를 쳤는지.
///
/// 접촉 이벤트로는 이 정보를 낼 수 없다: linear cast는 바디를 표면(speculative skin 안)에
/// 세워 두고, 반발·정지는 **다음 스텝**의 일반 접촉 파이프라인이 해석한다. 호스트가
/// "총알이 언제 어디를 맞았나"를 알려면 절단 순간 자체가 필요하다(임팩트 이펙트·데미지 판정).
///
/// 서브스텝 안의 분율이므로 스텝 안의 시각은 `(substep + toi) / substeps`다.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CcdHit {
    /// 절단당한(움직이던) 바디.
    pub body: BodyHandle,
    /// 부딪힌 static 바디 — linear cast 대상은 static 한정이다(설계 §4.5).
    pub other: BodyHandle,
    /// 이 서브스텝 이동분의 분율 [0,1].
    pub toi: f32,
    /// 몇 번째 서브스텝인가 (0부터).
    pub substep: u32,
    /// 표면 법선 (움직이던 바디를 밀어내는 방향).
    pub normal: Vec3,
    /// 표면 위 접점 근사.
    pub point: Vec3,
}

/// 한 스텝이 보고할 수 있는 CCD 히트 상한.
///
/// **상한이 있어야 할당이 없다**: 히트 수는 `CCD 바디 × 서브스텝`이라 입력에 비례해
/// 자라는데, 이벤트 버퍼가 스텝마다 자라면 핫패스에 재할당이 들어온다. 상한을 두고
/// 버퍼를 영속으로 잡으면 어떤 프레임도 할당하지 않는다. 넘치면 **뒤에 온 것부터
/// 버리고**(순회 순서가 결정적이므로 버려지는 집합도 결정적이다)
/// `Diagnostics::ccd_hits_dropped`로 보고한다 — 조용히 사라지지 않는다.
pub const MAX_CCD_HITS: usize = 1024;

/// 광선 질의 히트.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub body: BodyHandle,
    /// 광선 파라미터. 맞은 지점 = `origin + dir * toi`.
    pub toi: f32,
    pub point: Vec3,
    /// 맞은 지점의 바깥쪽 단위 법선.
    pub normal: Vec3,
    /// 광선 시작점이 이 물체 안이었는가 (그 경우 `toi == 0`).
    pub started_inside: bool,
}

#[derive(Clone)]
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
            // 서브스텝 4 (2026-07-27 owner 지시 "속도 우선"으로 8 → 4). **2026-07-30에 2와 3을
            // 실측하고 4로 되돌렸다** — 서브스텝을 더 줄이면 활성 처리량은 벌지만 **실사용에서
            // 더 느리다**:
            //   활성(수면 OFF): 3서브 −8~20%, 2서브 −15~30%
            //   정착(수면 ON): 정착 피라30 0.001 → 3서브 1.324 ms / 정착 더미3000 0.007 →
            //                 2서브 1.497 ms(**214배**), 40단은 2서브에서 UNSTABLE 판정
            // 원인은 이 저장소가 이미 규명한 "수면과의 경주"다(README): 서브스텝이 줄면
            // 잔진동이 커져 **수면 임계를 못 넘고**, 그러면 정착 장면이 활성 비용을 계속 낸다.
            // 실사용은 대부분 정착 상태이므로 이 거래는 손해다.
            //
            // 교훈: 이 트레이드는 **수면을 켠 채로** 재야 한다. 수면을 끄고 순수 처리량만 보면
            // 2서브가 이기는 것처럼 보인다(그렇게 잘못 판단했다가 벤치의 정착 모드가 잡았다).
            substeps: 4,
            speculative_margin: 0.05,
            solver: SolverParams::default(),
            // rapier 0.34 기본은 0.4 m/s · 0.5 rad/s(`time_until_sleep` 2초)다.
            // 우리는 그보다 4배 엄격하되, 예전 0.03(13배 엄격)은 스택이 영원히 못 자게
            // 만들어 서브스텝 8을 강요하고 있었다.
            sleep_linear_threshold: 0.1,
            sleep_angular_threshold: 0.15,
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
    /// **FP 환경이 캐노니컬 상태가 아니다** (설계 §8: RNE + FTZ/DAZ/FZ 전부 off).
    ///
    /// 이 레지스터는 스레드마다 산다 — 호스트나 다른 라이브러리(오디오 등)가 호출 스레드에서
    /// 플러시 모드를 켜 두면 같은 입력이 다른 물리를 낸다. 조용히 진행하면 결정론 계약이
    /// 거짓이 되므로 스텝을 거부한다. `World::canonicalize_fp_environment()`로 되돌린 뒤
    /// 다시 부르면 된다. **상태는 변하지 않는다**(검증이 모든 변이보다 앞이다).
    FpEnvironment,
}

/// 셰이프 스윕 질의 결과.
#[derive(Clone, Copy, Debug)]
pub struct SweepHit {
    pub body: BodyHandle,
    /// 이동 비율 [0,1] — `pose.pos + motion·toi`가 접촉 순간의 위치다.
    pub toi: f32,
    pub point: Vec3,
    /// 맞은 표면의 법선 (스윕하는 쪽을 향한다).
    pub normal: Vec3,
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
    contact_cache: IntMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
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
    /// 증분 병합의 출력 버퍼 — `bp_pairs`와 핑퐁하며 재사용한다(캐노니컬 아님).
    bp_pairs_scratch: Vec<(u32, u32)>,
    /// 캐시가 유효한가. 구조 변경·변이·적분이 있으면 무효화된다.
    bp_valid: bool,
    /// 캐시를 만들 때의 dt (sweep AABB가 dt에 비례하므로 바뀌면 무효).
    bp_dt: f32,
    /// 영속 브로드페이즈 구조(아이템·트리) — **파생 상태**(비캐노니컬, 스냅샷 제외).
    /// `Some`이면 아이템·트리가 "마지막 브로드페이즈 시점 + 그 뒤 엔진 적분"과 일치한다는
    /// 뜻이고, 호스트/구조 변이는 전부 `invalidate_bp`로 `None`을 만든다. 증분 갱신
    /// (R = 직전 스텝 integrate_gate)의 근거이며, §4.1(페어 집합 트리 비의존)이 위상
    /// 재사용을 합법으로 만든다.
    bp: Option<Broadphase>,
    /// 테스트 전용: 증분 브로드페이즈를 끄고 항상 재빌드 (등가성 A/B의 대조군).
    /// 비캐노니컬·비직렬화 — 프로덕션 경로에서는 항상 false다.
    bp_force_full: bool,
    /// 솔버 핫패스 스크래치 (비캐노니컬). 매 스텝 재할당하지 않으려고 월드가 보관한다.
    solver_state: solver::SolverState,
    inv_inertia_local: Vec<Vec3>,
    /// 스텝 스크래치: 슬롯별 누적 (힘, 토크). 솔버 루프가 뚱뚱한 `Body`를 다시 훑지 않게 한다.
    ext_force: Vec<(Vec3, Vec3)>,
    /// 스텝 스크래치: 슬롯별 (선형 감쇠, 각 감쇠).
    damping: Vec<(f32, f32)>,
    /// 이번 스텝의 접촉 이벤트. 스텝 시작에 비우고 스텝 안에서 채운다.
    contact_events: Vec<ContactEvent>,
    /// 이벤트 방출 진입 시점의 `contact_events` 용량 (테스트 관측용).
    #[cfg(test)]
    ev_cap_at_emit: usize,
    /// 이번 스텝의 CCD 절단 히트 (비캐노니컬 — 스냅샷 밖, 스텝 시작에 비운다).
    /// 상한 `MAX_CCD_HITS`, 버퍼는 영속이라 스텝 중 할당이 없다.
    ccd_hits: Vec<CcdHit>,
    /// **직전 스텝에 닿아 있던 페어 집합** — 이벤트의 기준선.
    ///
    /// PCM 캐시 키에서 파생하려다 실패했다: `mutate`(텔레포트·속도 대입)가 그 바디의
    /// 캐시 엔트리를 지우므로, 물체를 치워도 "원래 안 닿아 있었던 것"이 되어
    /// **Stopped가 안 나갔다**. 이벤트 기준선은 캐시 무효화와 수명이 달라야 한다.
    /// 캐노니컬 상태다 — 복원된 월드가 같은 이벤트를 내야 한다.
    touching: IntSet<(u32, u32)>,
    integrate_gate: Vec<bool>,
    gravity_gate: Vec<bool>,
    island_parent: Vec<usize>,
    /// SoA 솔버 블록 풀 (비캐노니컬 스크래치). 블록 1개가 ~1.5KB라 스텝마다 새로 할당하면
    /// 조립 이득이 할당·이동 비용으로 되돌아간다 — 월드가 버퍼를 들고 제자리에서 덮어쓴다.
    contact_blocks: Vec<solver_block::ContactBlock>,
    /// 워커 풀 (설계 §5). **캐노니컬 상태가 아니다** — 스레드 수는 결과를 바꾸지 않으므로
    /// 스냅샷·상태 해시에 들어가지 않는다. 그것이 이 트랙의 계약 자체다.
    pool: crate::threads::Pool,
    /// 내로우페이즈 참여자별 출력 버퍼 (스텝 사이 재사용 — 할당 0).
    narrow_bufs: Vec<NarrowOut>,
    /// wake 폐쇄 **직전**의 sleep 상태 복사본 — 스텝이 자원 오류로 끝날 때 되돌리는 기준선.
    ///
    /// `asleep`·`sleep_timer`는 스냅샷에 실리는 **캐노니컬 상태**다. wake는 narrowphase보다
    /// 먼저 돌므로, 뒤에서 접촉 예산이 터져 스텝을 포기하면 "물리 스텝은 건너뜀"이라 말하면서
    /// 잠들어 있어야 할 아일랜드를 깨워 놓은 채 끝난다 — 호스트가 문제 바디를 지우고 재시도해도
    /// 그 아일랜드는 이미 중력·적분 대상이 되어 실패 전과 다른 미래를 만든다
    /// (확인검수 rev_d5b32e22 BLOCKER). 용량은 wake 이전 통합 fallible 블록에서 잡는다.
    sleep_bak: Vec<bool>,
    sleep_timer_bak: Vec<f32>,
    /// 접촉 솔버 색 병렬 스위치 — **기본 off**. 근거는 `set_parallel_solver` 문서.
    parallel_solver: bool,
}

/// 진단 계측용 시계 — **wasm32-unknown-unknown에는 시계가 없다.**
/// `Instant::now()`가 그 타깃에서 panic(=trap)하므로(브라우저 실측에서 첫 스텝에
/// 발견 — WASI 게이트는 시계가 있어 못 잡는 사각지대), 진단 타이밍은 no-op으로
/// 접는다. 브라우저 프로파일은 호스트의 `performance.now()` 몫이다.
#[derive(Clone, Copy)]
struct Stopwatch(
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))] std::time::Instant,
);

impl Stopwatch {
    #[inline]
    fn start() -> Self {
        Self(
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            std::time::Instant::now(),
        )
    }
    #[inline]
    fn ns(self) -> u64 {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            self.0.elapsed().as_nanos() as u64
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            0
        }
    }
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
    /// 이번 스텝이 증분 브로드페이즈(움직인 바디만 갱신 + 페어 델타)를 탔는가.
    pub bp_incremental: bool,
    /// 이번 스텝이 제한 순회(활성⊆R 증명 → wake/narrow/이벤트를 fresh로 한정)를 탔는가.
    pub narrow_limited: bool,
    /// 이번 스텝 내로우페이즈를 나눠 돈 참여자 수 (1 = 직렬. 설계 §6.3 "티어 강등은
    /// 리포트에 노출" — 스레드가 실제로 쓰였는지를 호스트가 관측할 수 있어야 한다).
    pub narrow_threads: usize,
    /// 하이트필드 페어가 매니폴드 상한(`MAX_HEIGHTFIELD_MANIFOLDS`)에서 끊긴 횟수.
    /// 0이 아니면 그 페어의 접촉 일부가 이번 스텝에서 빠졌다는 뜻이다 — 격자를 성기게
    /// 하거나 바디를 작게 하라는 신호다(조용한 누락이 아니라 관측 가능한 포화).
    pub heightfield_saturated: usize,
    /// 접촉 제약 그래프 컬러링이 쓴 **비어 있지 않은** 색 수 = 솔브 패스당 배리어 수.
    pub colors: usize,
    /// 색 상한(`MARU_COLOR_CAP`)에 밀려 오버플로 버킷에 들어간 제약 수.
    ///
    /// 오버플로는 **한 참여자가 순서대로** 푸는 직렬 잔여다 — 병렬 폭을 유계로 사는 대가이고,
    /// 이 수치가 곧 그 대가의 크기다(Amdahl 직렬 분율). 상한을 정하는 판단은 `colors`(배리어
    /// 수)와 이 값(직렬 분율)의 거래이므로 둘을 함께 보고한다.
    pub color_overflow: usize,
    /// 이번 스텝 솔브 패스를 나눠 돈 참여자 수 (1 = 직렬).
    pub solve_threads: usize,
    /// 솔버 work 항목 수 (블록 + 스칼라 잔여) — 색 병렬의 분배 단위.
    pub solve_work: usize,
    /// 이번 스텝 CCD linear cast가 이동을 절단한 횟수 (서브스텝×바디 누적).
    pub ccd_clamps: usize,
    /// 이번 스텝 frozen(평면 5개째 = 구석에 낌) 상태로 돈 서브스텝×바디 누적.
    pub ccd_frozen: usize,
    /// 상한(`MAX_CCD_HITS`)을 넘겨 버려진 CCD 히트 수. 0이 아니면 호스트가 보는
    /// 히트 목록이 이번 스텝의 전부가 아니다 — 조용한 유실과 구분되어야 한다.
    pub ccd_hits_dropped: usize,
    /// CCD 캐스트가 **판정을 완료하지 못한** 횟수(지형 셀 예산 초과 등).
    ///
    /// 그 바디는 해당 서브스텝의 이동·회전이 0으로 묶인다(보수적 정지 — 관통보다 멈춤).
    /// 스텝 자체는 진행되므로 `try_step`은 `Ok`다: 적분 중에는 스텝을 되돌릴 수 없어
    /// `ResourceLimit`("스텝 건너뜀")을 쓸 수 없다. 이 값이 0이 아니면 그 프레임에
    /// 어떤 바디가 조용히 멈춰 있다는 뜻이므로, 호스트는 지형 해상도나 바디 크기를 봐야 한다.
    pub ccd_incomplete: usize,
    /// 단계별 소요 (ns) — 프로파일용.
    /// `[aabb, broad, narrow, solve, post]`
    /// - `aabb`: 바디 순회 + 회전 월드 AABB 구성 (예전에는 broad에서 **빠져 있었다**)
    /// - `broad`: 트리 구축 + 페어 질의 (캐시 재사용 스텝은 ~0)
    /// - `narrow`: wake 폐쇄 + 접촉 생성 + 제약 준비
    /// - `solve`: 서브스텝 루프 + restitution **만**
    /// - `post`: write-back · PCM 갱신 · 아일랜드/슬립 · 상태 정책
    ///   (예전에는 이 넷이 `solve`에 섞여 있어 최적화 우선순위를 잘못 귀속시켰다 — 리뷰 MAJOR)
    pub phase_ns: [u64; 5],
    /// 하위 단계 소요 (ns) — 정착 레짐 귀속용 계측기.
    /// `[scratch_reserve, wake_walk, narrow_walk, island_post]`
    /// 상위 phase 안에서 **페어 수·바디 수에 비례하는 부분만** 따로 잰다: 전원 수면 씬은
    /// 산출물이 접점 몇 개인데도 이 넷이 페어/바디 전량을 훑어 스텝을 지배할 수 있다.
    pub sub_ns: [u64; 5],
    /// post 단계 내역 (ns): `[write_back+외력클리어, 이벤트 차집합, PCM 갱신, 위생(denormal·클램프·경계·정책)]`.
    /// 아일랜드는 `sub_ns[3]`에 이미 있다. 정착 씬에서 post가 스텝의 40%인데 아일랜드는
    /// 그중 1/9뿐이라 — 나머지가 어디 사는지 귀속하려고 둔다.
    pub post_ns: [u64; 4],
    /// solve 단계 내역 (ns) — **서브스텝 전체의 누적**:
    /// `[inv_i 갱신, refresh_anchors, warm_start(접촉), bias 솔브, relax 솔브, 적분+클램프(CCD 포함), 조인트]`.
    /// 활성 레짐에서 solve가 스텝의 절반인데 그 안의 배분이 안 보이면 SIMD를 어디에 쓸지
    /// 추측으로 정하게 된다 — 실제로 `refresh_anchors`·`inv_i`처럼 **솔버 산술이 아닌**
    /// 준비 패스가 지배할 수 있고, 그것들은 제약 솔브와 최적화 방식이 다르다.
    /// 서브스텝당 `inv_i`·`refresh_anchors`·솔브는 각각 2회 돌므로 합계로 누적한다.
    pub solve_ns: [u64; 7],
    /// 직전 스텝의 접촉 기준선 크기 — 정착 씬에서 페어 수와 함께 비용을 결정한다.
    pub touching: usize,
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
    // 검증 층 — 불변식이 안 선 형상은 거절한다(패닉 금지).
    let Ok(bb) = b.shape.try_local_aabb() else {
        return false;
    };
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
        // **설정 불가 환경은 명시 실패**다(설계 §8) — 조용히 진행하면 결정론 계약이
        // 처음부터 거짓인 월드가 만들어진다.
        if !maru_math::fp_canonicalize() {
            return Err(ApiError::FpEnvironment);
        }
        check_params(&params)?;
        Ok(Self::new_unchecked(params))
    }

    fn new_unchecked(params: WorldParams) -> Self {
        // **FP 환경을 캐노니컬 상태로 맞춘다** (설계 §8). 이 레지스터는 스레드마다 살고
        // 호스트가 이미 플러시 모드를 켜 뒀을 수 있다 — 월드를 만드는 김에 정리한다.
        // 이후 스텝마다 검증하므로, 다른 스레드에서 부르거나 중간에 누가 바꾸면 잡힌다.
        let _ = maru_math::fp_canonicalize();
        Self {
            params,
            bodies: Vec::new(),
            alive: Vec::new(),
            oob: Vec::new(),
            reserved: Vec::new(),
            generations: Vec::new(),
            contact_cache: IntMap::default(),
            sleep_timer: Vec::new(),
            asleep: Vec::new(),
            island: Vec::new(),
            joints: Vec::new(),
            commands: Vec::new(),
            last_report: CommandReport::default(),
            diag: Diagnostics::default(),
            resource_limit: false,
            bp_pairs: Vec::new(),
            bp_pairs_scratch: Vec::new(),
            contact_blocks: Vec::new(),
            bp_valid: false,
            bp_dt: 0.0,
            bp: None,
            bp_force_full: false,
            solver_state: solver::SolverState::default(),
            inv_inertia_local: Vec::new(),
            ext_force: Vec::new(),
            damping: Vec::new(),
            contact_events: Vec::new(),
            #[cfg(test)]
            ev_cap_at_emit: 0,
            ccd_hits: Vec::new(),
            touching: IntSet::default(),
            integrate_gate: Vec::new(),
            gravity_gate: Vec::new(),
            island_parent: Vec::new(),
            // 기본 = 가용 코어(상한 8). 결과는 스레드 수와 무관하게 같으므로 기본 활성이
            // 안전하고, 원하면 `set_threads`로 1(완전 직렬)까지 내릴 수 있다.
            pool: crate::threads::Pool::new(crate::threads::default_threads()),
            narrow_bufs: Vec::new(),
            sleep_bak: Vec::new(),
            sleep_timer_bak: Vec::new(),
            parallel_solver: false,
        }
    }

    #[inline]
    /// 수면 임계 런타임 조정 (설계 §4.6 파라미터의 유일한 가변 노출). 게임은 장면
    /// 성격에 따라 수면 공격성을 고른다 — 톨 스택 정밀 장면은 낮게, 대량 잔해 장면은
    /// 높게(수면이 곧 성능·안정 레버라는 절벽 규명 결론). 검증은 생성과 **같은 정본**
    /// (`check_params`)으로 한다 — 필드별 검사를 따로 적으면 비대칭이 재발한다.
    ///
    /// 잠든 바디는 깨우지 않는다: 임계는 수면 **진입**만 게이트하고, 이미 잠든
    /// 바디의 유지·기상은 wake 경로 소관이다(전 엔진 공통 시맨틱). 파라미터는
    /// 캐노니컬 상태라 스냅샷에 그대로 실린다.
    pub fn try_set_sleep_thresholds(
        &mut self,
        linear: f32,
        angular: f32,
        time_to_sleep: f32,
    ) -> Result<(), ApiError> {
        Self::fp_guard()?;
        let mut p = self.params.clone();
        p.sleep_linear_threshold = linear;
        p.sleep_angular_threshold = angular;
        p.time_to_sleep = time_to_sleep;
        check_params(&p)?;
        self.params = p;
        Ok(())
    }

    pub fn params(&self) -> &WorldParams {
        &self.params
    }

    /// 파라미터 교체 — 생성자와 **같은 불변식**을 통과해야 한다.
    pub fn set_params(&mut self, params: WorldParams) {
        // FP 가드 — 캐노니컬 상태를 바꾸는 모든 진입점에 같은 규칙을 적용한다
        // (비-fallible 판은 관례대로 패닉). 한 곳만 빠지면 "비표준 환경 진입은 상태를
        // 바꾸지 않는다"가 API 선택에 따라 갈린다(검수가 두 번 지적한 계열).
        assert!(
            maru_math::fp_verify(),
            "non-canonical FP environment (see World::canonicalize_fp_environment)"
        );
        validate_params(&params);
        self.invalidate_bp();
        self.params = params;
    }

    /// 파라미터 교체 (오류 반환형).
    pub fn try_set_params(&mut self, params: WorldParams) -> Result<(), ApiError> {
        Self::fp_guard()?;
        check_params(&params)?;
        self.invalidate_bp();
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
    /// 슬롯 예약 + 추가 커맨드 (오류 반환형).
    pub fn try_enqueue_add_body(&mut self, body: Body) -> Result<BodyHandle, ApiError> {
        // **가드는 큐·슬롯을 건드리기 전에** — 실패가 상태를 남기면 안 된다.
        Self::fp_guard()?;
        body.validate()?;
        Ok(self.enqueue_add_body(body))
    }

    pub fn enqueue_add_body(&mut self, body: Body) -> BodyHandle {
        // **여기도 가드가 필요하다** — 이 경로만 검증 없이 예약 슬롯·세대·큐를 바꿨다.
        // 다른 변이는 전부 거부되는데 이것만 통과하면 "비표준 환경 진입은 상태를 바꾸지
        // 않는다"가 API 선택에 따라 갈린다(확인검수 rev_e50670ad MAJOR).
        assert!(
            maru_math::fp_verify(),
            "non-canonical FP environment (see World::canonicalize_fp_environment)"
        );
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
    /// 바디 추가 (오류 반환형).
    ///
    /// 계약 "실패는 trap이 아니라 오류"에서 이 경로만 빠져 있었다 — 브라우저 바인딩은
    /// unwind가 곧 trap이라 `expect`가 호스트에게 진단 불가능한 죽음이 된다.
    pub fn try_add_body(&mut self, body: Body) -> Result<BodyHandle, ApiError> {
        Self::fp_guard()?;
        body.validate()?;
        Ok(self.add_body_validated(body))
    }

    /// 패닉하는 편의 래퍼 (테스트·벤치용). 정본은 [`World::try_add_body`].
    /// 지형/레벨 삼각형 메시를 static으로 넣는다 — **삼각형당 static 바디 1개**(3점 hull +
    /// convex radius). 이것이 이 엔진의 메시 정본 패턴이다(전용 메시 셰이프 없음): 3점은
    /// 두께 0이라 최소 특징 크기를 radius로만 얻고, radius가 얇으면 hull validate가 거부한다.
    /// static 트리는 구조 변경 시에만 빌드되므로 수천 삼각형도 스텝 비용에 영향이 없다.
    ///
    /// **내부 에지 용접(§4.2)**: 정확히 두 삼각형이 공면으로 공유하는 에지는 등록 시
    /// `weld_edges` 마스크로 표시되고, 내로우페이즈가 그 에지 밴드의 접점 법선을 면
    /// 법선으로 스냅한다(Jolt ActiveEdges 동형) — 평평한 메시의 유령 충돌(이음새 킥)을
    /// 제거한다. 볼록 능선·오목 골짜기·경계·비-다양체 에지는 보수적으로 용접하지 않는다
    /// (에지 법선이 물리적으로 옳은 곳이다).
    ///
    /// 반환: 삼각형별 핸들. 퇴화 삼각형(validate 실패)은 `Err`로 전체 거부(부분 삽입 없음).
    /// **격자 하이트필드 지형**을 static 바디 하나로 추가한다 (설계 §4.2, M4).
    ///
    /// `try_add_terrain`(삼각형마다 바디)과 달리 격자는 **바디 하나**다: 256×256이면
    /// 13만 개 대신 1개고, 브로드페이즈 리프도 1개다. 어떤 삼각형이 겹치는지는
    /// 내로우페이즈가 셀 좌표로 바로 구한다.
    ///
    /// 내부 에지 억제는 **격자 구조에서 유도**된다 — 이웃 삼각형이 좌표로 나오므로
    /// 에지 사전이 필요 없고, 공면 판정 임계는 `try_add_terrain`과 같은 값이다
    /// (같은 지형을 두 방식으로 만들어도 물리가 갈리지 않아야 한다).
    ///
    /// 로컬 좌표: `x = ix·cell`, `z = iz·cell`, 높이는 `heights[iz·nx + ix]`.
    /// 원점은 격자의 (0,0) 모서리이므로 타일을 이어 붙일 때 오프셋이 곧 pose다.
    ///
    /// 한 페어가 만드는 접촉은 `MAX_HEIGHTFIELD_MANIFOLDS`에서 끊기고, 끊긴 사실은
    /// `Diagnostics::heightfield_saturated`로 보고된다(조용한 누락 금지).
    pub fn try_add_heightfield(
        &mut self,
        heights: &[f32],
        nx: u32,
        nz: u32,
        cell: f32,
        pose: maru_collide::Pose,
    ) -> Result<BodyHandle, ApiError> {
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(heights.len())
            .map_err(|_| ApiError::ResourceLimit)?;
        owned.extend_from_slice(heights);
        // 구조 불변식과 타일은 **생성자가** 세운다(`HeightfieldGrid::new`) — 위조된 타일이
        // 타입 수준에서 불가능하므로 어느 질의도 재검증하지 않는다.
        let shape =
            maru_collide::Shape::heightfield(owned, nx, nz, cell).ok_or(ApiError::InvalidValue)?;
        // 검증은 라이브·복원 공통 정본(`Shape::validate`)이 한다 — 여기서 따로 적으면
        // 두 경계의 정책이 갈린다.
        shape.validate().map_err(|_| ApiError::InvalidValue)?;
        self.try_add_body(Body::new_static(shape, pose))
    }

    pub fn try_add_terrain(
        &mut self,
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
        radius: f32,
    ) -> Result<Vec<BodyHandle>, ApiError> {
        Self::fp_guard()?;
        // 전체 검증 먼저 — 부분 삽입은 실패 시 호스트가 되돌릴 수 없다.
        let mut bodies = Vec::new();
        bodies
            .try_reserve(triangles.len())
            .map_err(|_| ApiError::ResourceLimit)?;
        for t in triangles {
            let mut pts = Vec::new();
            pts.try_reserve_exact(3)
                .map_err(|_| ApiError::ResourceLimit)?;
            for &i in t {
                let p = *vertices.get(i as usize).ok_or(ApiError::InvalidValue)?;
                pts.push(p);
            }
            // **퇴화 검사 — 스케일 상대 면적**(교차검수 rev_e4075e85 MAJOR): 3점 hull의
            // 회전 불변 폭은 0이라 radius만으로 validate를 통과한다 — 동일/일직선 정점이
            // "둥근 점/선분 장애물"로 조용히 들어가 메시 오류가 유령 충돌이 된다.
            // cross 길이는 L² 차원이므로 최장 에지²에 상대 비교한다(절대 상수 금지).
            let e0 = pts[1] - pts[0];
            let e1 = pts[2] - pts[0];
            let scale = e0.length().max(e1.length()).max((pts[2] - pts[1]).length());
            if e0.cross(e1).length() <= 1e-6 * scale * scale {
                return Err(ApiError::InvalidValue);
            }
            // 삼각형 중심을 바디 원점으로 — hull 점이 원점 근처라 앵커·톨러런스 스케일이 정직해진다.
            let c = (pts[0] + pts[1] + pts[2]) * (1.0 / 3.0);
            let local: Vec<Vec3> = pts.iter().map(|p| *p - c).collect();
            let shape = maru_collide::Shape::Hull {
                points: local,
                radius,
            };
            shape.validate().map_err(|_| ApiError::InvalidValue)?;
            bodies.push(Body::new_static(
                shape,
                maru_collide::Pose {
                    pos: c,
                    rot: maru_math::Quat::IDENTITY,
                },
            ));
        }
        // **내부 에지 판정** (§4.2 ghost edge): 정확히 두 삼각형이 공유하고 **공면**인
        // 에지만 용접 대상이다. 볼록 능선(절벽 모서리)·오목 골짜기는 에지 법선이
        // 물리적으로 옳으므로 제외하고, 3면 이상이 공유하는 비-다양체 에지도 보수적으로
        // 제외한다. 공면 판정은 상대 삼각형 반대 정점의 평면 거리 — 길이 차원이므로
        // 에지 스케일 상대 임계를 쓴다(절대 상수 금지).
        {
            let mut edge_map: HashMap<(u32, u32), Vec<(usize, u8)>> = HashMap::new();
            for (ti, t) in triangles.iter().enumerate() {
                for e in 0..3u8 {
                    let (a, b) = (t[e as usize], t[((e + 1) % 3) as usize]);
                    edge_map
                        .entry((a.min(b), a.max(b)))
                        .or_default()
                        .push((ti, e));
                }
            }
            let unit_normal = |ti: usize| -> Option<Vec3> {
                let p0 = vertices[triangles[ti][0] as usize];
                let p1 = vertices[triangles[ti][1] as usize];
                let p2 = vertices[triangles[ti][2] as usize];
                let n = (p1 - p0).cross(p2 - p0);
                let scale = (p1 - p0)
                    .length()
                    .max((p2 - p0).length())
                    .max((p2 - p1).length());
                let len = n.length();
                // 퇴화는 위에서 이미 거부됐지만 나눗셈 방어.
                (len > maru_collide::area_eps(scale)).then(|| n * (1.0 / len))
            };
            for (&(va, vb), users) in &edge_map {
                let [(t1, e1), (t2, e2)] = users[..] else {
                    continue; // 경계(1) 또는 비-다양체(3+) — 용접하지 않는다
                };
                let opp = |t: &[u32; 3]| {
                    let i = t
                        .iter()
                        .position(|&v| v != va && v != vb)
                        .expect("삼각형은 에지 밖 정점이 정확히 하나");
                    vertices[t[i] as usize]
                };
                let (Some(n1), Some(n2)) = (unit_normal(t1), unit_normal(t2)) else {
                    continue;
                };
                // **공면 판정은 각도(무차원)로** — "반대 정점의 평면 거리 ≤ 한쪽 삼각형
                // 스케일"은 스키니 삼각형에서 임계가 부풀어(에지 1cm·먼 정점 수 km →
                // 수 cm 허용) 진짜 능선을 용접했고, 어느 쪽이 t1이냐에도 비대칭이었다
                // (검수 rev_3398738c MAJOR). 단위 법선의 정렬은 스케일 불변·대칭이다.
                // 와인딩 정합: 다양체 메시는 공유 에지를 서로 반대 방향으로 순회한다 —
                // 같은 방향이면 이웃 와인딩이 뒤집힌 것이므로 법선을 뒤집어 비교한다.
                let same_dir = triangles[t1][e1 as usize] == triangles[t2][e2 as usize];
                let n2o = if same_dir { -n2 } else { n2 };
                // Jolt ActiveEdges 기본 임계와 동급(~5.7°) — 완만한 지형 테셀레이션은
                // 용접하고, 진짜 능선·골짜기는 남긴다.
                const COS_COPLANAR: f32 = 0.995;
                if n1.dot(n2o) < COS_COPLANAR {
                    continue;
                }
                // **반대쪽 검증** — 두 반대 정점이 공유 에지의 서로 반대쪽이어야 진짜
                // 이웃이다. 같은 쪽(중복·겹침면)을 용접하면 실제 외곽 경계의 에지 법선이
                // 사라져 경계 충돌이 소실된다(검수 rev_3398738c MAJOR). 부호 곱이
                // 음수일 때만 용접(0·NaN = 보수적으로 제외).
                let ea = vertices[va as usize];
                let edge = vertices[vb as usize] - ea;
                let side = |o: Vec3| edge.cross(n1).dot(o - ea);
                let s1 = side(opp(&triangles[t1]));
                let s2 = side(opp(&triangles[t2]));
                if s1 * s2 < 0.0 {
                    bodies[t1].weld_edges |= 1 << e1;
                    bodies[t2].weld_edges |= 1 << e2;
                }
            }
        }
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(bodies.len())
            .map_err(|_| ApiError::ResourceLimit)?;
        for b in bodies {
            handles.push(self.add_body(b));
        }
        Ok(handles)
    }

    pub fn add_body(&mut self, body: Body) -> BodyHandle {
        // FP 가드 — 캐노니컬 상태를 바꾸는 모든 진입점에 같은 규칙을 적용한다
        // (비-fallible 판은 관례대로 패닉). 한 곳만 빠지면 "비표준 환경 진입은 상태를
        // 바꾸지 않는다"가 API 선택에 따라 갈린다(검수가 두 번 지적한 계열).
        assert!(
            maru_math::fp_verify(),
            "non-canonical FP environment (see World::canonicalize_fp_environment)"
        );
        body.validate().expect("invalid body");
        self.add_body_validated(body)
    }

    fn add_body_validated(&mut self, body: Body) -> BodyHandle {
        self.invalidate_bp();
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
        Self::fp_guard()?;
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
        self.invalidate_bp();
        self.alive[i] = false;
        self.reserved[i] = false;
        self.oob[i] = false;
        // 접촉 이벤트 기준선에서도 지운다 — 남겨두면 다음 스텝에 죽은 슬롯 키가
        // 종료 이벤트를 만들고, 슬롯이 재사용됐으면 엉뚱한 물체를 가리킨다.
        let id = i as u32;
        self.touching
            .retain(|(a, b)| *a != id && (*b & !STATIC_BIT) != id);
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
        self.invalidate_bp();
        if self.bodies[i].body_type == BodyType::Static {
            return Err(ApiError::ImmutableStatic);
        }
        self.wake(i);
        let id = i as u32;
        self.contact_cache
            .retain(|(a, b), _| *a != id && (*b & !STATIC_BIT) != id);
        Ok(i)
    }

    /// 힘·충격 경로 전용 해석기.
    ///
    /// `mutate`와 달리 **PCM 캐시를 버리지 않는다** — 힘을 준다고 접촉이 사라진 것이
    /// 아니므로 warm start를 끊으면 스택이 근거 없이 흔들린다(`mutate`가 캐시를 버리는
    /// 것은 텔레포트·속도 대입처럼 접촉 이력이 무의미해지는 변경 때문이다).
    /// 동적 바디만 받는다 — static·kinematic에 힘은 의미가 없고, 조용히 무시하면
    /// 호스트가 "왜 안 움직이지"를 진단할 수 없다.
    fn accept_force(&mut self, h: BodyHandle) -> Result<usize, ApiError> {
        let i = self.resolve(h)?;
        if self.bodies[i].body_type != BodyType::Dynamic {
            return Err(ApiError::ImmutableStatic);
        }
        // **브로드페이즈 페어 캐시도 버린다.** sweep AABB는 속도의 함수인데 impulse 계열은
        // 속도를 즉시 바꾼다 — 정착(캐시 유효) 상태에서 impulse를 주면 다음 스텝이 임펄스
        // 이전 속도로 만든 페어를 재사용해, 이번 스텝에 도달할 상대와의 speculative 접촉이
        // 한 스텝 빠진다(얇은 벽 터널링의 씨앗). 캐시 계약("재사용은 항등")의 위반이다.
        // 힘/토크 경로는 속도가 적분 시점에 바뀌므로 원래도 등가지만, 힘이 실리는 장면은
        // 어차피 moved=true로 캐시가 없어 일괄 무효화의 추가 비용이 없다.
        self.invalidate_bp();
        self.wake(i);
        Ok(i)
    }

    /// 누적 힘에 더한다 (월드 좌표, 뉴턴). 스텝이 소비하고 스텝 끝에 지워진다.
    ///
    /// 여러 번 부르면 더해진다. 한 번의 입력은 **입력 범위**로 거르고, 누적분이
    /// 파생 상한을 넘으면 포화시키고 `StateOverflow`를 올린다(다른 파생 상태와 같은 규칙).
    pub fn apply_force(&mut self, h: BodyHandle, force: Vec3) -> Result<(), ApiError> {
        Self::fp_guard()?;
        if !maru_collide::vec_in_range(force) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.accept_force(h)?;
        let (sat, clipped) = Self::saturate_derived(self.bodies[i].force + force);
        self.bodies[i].force = sat;
        if clipped {
            return Err(ApiError::StateOverflow);
        }
        Ok(())
    }

    /// 누적 토크에 더한다 (월드 좌표, N·m).
    pub fn apply_torque(&mut self, h: BodyHandle, torque: Vec3) -> Result<(), ApiError> {
        Self::fp_guard()?;
        if !maru_collide::vec_in_range(torque) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.accept_force(h)?;
        let (sat, clipped) = Self::saturate_derived(self.bodies[i].torque + torque);
        self.bodies[i].torque = sat;
        if clipped {
            return Err(ApiError::StateOverflow);
        }
        Ok(())
    }

    /// 질량중심에 **즉시** 선형 충격을 준다 (N·s). 속도가 그 자리에서 바뀐다.
    pub fn apply_impulse(&mut self, h: BodyHandle, impulse: Vec3) -> Result<(), ApiError> {
        Self::fp_guard()?;
        if !maru_collide::vec_in_range(impulse) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.accept_force(h)?;
        let dv = impulse * self.bodies[i].inv_mass;
        let (sat, clipped) = Self::saturate_derived(self.bodies[i].linvel + dv);
        self.bodies[i].linvel = sat;
        if clipped {
            return Err(ApiError::StateOverflow);
        }
        Ok(())
    }

    /// **즉시** 각충격을 준다 (N·m·s).
    pub fn apply_angular_impulse(&mut self, h: BodyHandle, impulse: Vec3) -> Result<(), ApiError> {
        Self::fp_guard()?;
        if !maru_collide::vec_in_range(impulse) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.accept_force(h)?;
        let dw = self.bodies[i].inv_inertia_world().mul_vec(impulse);
        let (sat, clipped) = Self::saturate_derived(self.bodies[i].angvel + dw);
        self.bodies[i].angvel = sat;
        if clipped {
            return Err(ApiError::StateOverflow);
        }
        Ok(())
    }

    /// 월드 좌표의 한 **점**에 즉시 충격을 준다 — 질량중심에서 벗어난 만큼 회전도 생긴다.
    /// 폭발·타격처럼 "어디를 맞았는가"가 중요한 경우의 정본 경로다.
    pub fn apply_impulse_at_point(
        &mut self,
        h: BodyHandle,
        impulse: Vec3,
        point: Vec3,
    ) -> Result<(), ApiError> {
        Self::fp_guard()?;
        if !maru_collide::vec_in_range(impulse) || !maru_collide::vec_in_range(point) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.accept_force(h)?;
        // 레버암은 **질량중심 기준**이다 (설계 계약: pose.pos = COM).
        let r = point - self.bodies[i].pose.pos;
        let dv = impulse * self.bodies[i].inv_mass;
        let dw = self.bodies[i].inv_inertia_world().mul_vec(r.cross(impulse));
        let (sv, cv) = Self::saturate_derived(self.bodies[i].linvel + dv);
        let (sw, cw) = Self::saturate_derived(self.bodies[i].angvel + dw);
        self.bodies[i].linvel = sv;
        self.bodies[i].angvel = sw;
        if cv || cw {
            return Err(ApiError::StateOverflow);
        }
        Ok(())
    }

    /// 누적 힘·토크를 지운다 (스텝을 돌리지 않고 취소할 때).
    pub fn clear_forces(&mut self, h: BodyHandle) -> Result<(), ApiError> {
        Self::fp_guard()?;
        let i = self.resolve(h)?;
        self.bodies[i].force = Vec3::ZERO;
        self.bodies[i].torque = Vec3::ZERO;
        Ok(())
    }

    /// 파생 상태 포화 — 상한을 넘으면 자른 값과 "잘렸다"를 함께 준다.
    ///
    /// 진단 카운터로 올리지 않는 이유: `apply_*`는 스텝 **전에** 불리는데 진단은
    /// 스텝마다 초기화되므로 그 경로로 올리면 포화가 조용히 사라진다.
    /// 값은 정책대로 포화시키되 `StateOverflow`를 **그 자리에서** 돌려준다.
    fn saturate_derived(v: Vec3) -> (Vec3, bool) {
        maru_collide::saturate_state_vec(v)
    }

    /// 광선 질의 결과.
    ///
    /// `point`는 월드 좌표의 맞은 지점, `normal`은 그 지점의 **바깥쪽** 단위 법선이다.
    /// 광선이 물체 안에서 시작했으면 `toi == 0`이고 `started_inside`가 참이다 —
    /// "이미 안에 있다"를 못 맞은 것으로 처리하면 지면 판정이 조용히 실패한다.
    pub fn cast_ray(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_toi: f32,
    ) -> Result<Option<RayHit>, ApiError> {
        self.cast_ray_filtered(origin, dir, max_toi, QueryFilter::ALL)
    }

    /// 필터를 받는 광선 질의.
    ///
    /// 물리 충돌과 **같은 양방향 그룹 규칙**을 쓴다(`QueryFilter::groups`). 총알이
    /// 뚫고 지나가야 할 것은 광선에도 안 맞아야 자연스럽다 — 두 규칙이 갈리면
    /// 호스트가 같은 개념을 두 벌로 관리하게 된다.
    pub fn cast_ray_filtered(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_toi: f32,
        filter: QueryFilter,
    ) -> Result<Option<RayHit>, ApiError> {
        let (origin, dir, max_toi) = Self::check_ray(origin, dir, max_toi)?;
        let mut best: Option<RayHit> = None;
        for i in 0..self.bodies.len() {
            if !self.passes(i, filter) {
                continue;
            }
            let Some(hit) = self.ray_body(i, origin, dir, max_toi) else {
                continue;
            };
            // 동률은 **낮은 슬롯**이 이긴다 — 결과가 순회 순서에 의존하지 않게 한다.
            if best.as_ref().is_none_or(|b| hit.toi < b.toi) {
                best = Some(hit);
            }
        }
        Ok(best)
    }

    /// 광선이 지나는 **모든** 물체. `toi` 오름차순, 동률은 슬롯 오름차순(전순서).
    pub fn cast_ray_all(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_toi: f32,
        out: &mut Vec<RayHit>,
    ) -> Result<(), ApiError> {
        self.cast_ray_all_filtered(origin, dir, max_toi, QueryFilter::ALL, out)
    }

    /// 필터를 받는 전체 광선 질의.
    pub fn cast_ray_all_filtered(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_toi: f32,
        filter: QueryFilter,
        out: &mut Vec<RayHit>,
    ) -> Result<(), ApiError> {
        let (origin, dir, max_toi) = Self::check_ray(origin, dir, max_toi)?;
        out.clear();
        if out.try_reserve(self.bodies.len()).is_err() {
            return Err(ApiError::ResourceLimit);
        }
        for i in 0..self.bodies.len() {
            if !self.passes(i, filter) {
                continue;
            }
            if let Some(hit) = self.ray_body(i, origin, dir, max_toi) {
                out.push(hit);
            }
        }
        // **안정 정렬을 쓰지 않는다** — 안정 정렬은 임시 버퍼를 할당할 수 있고, 그
        // 할당이 실패하면 `ResourceLimit`이 아니라 trap이 된다(검수 rev_710b92d2 MAJOR).
        // 동률 순서도 "슬롯 순으로 채웠으니까"라는 암묵 의존 대신 **비교자에 명시**한다.
        out.sort_unstable_by(|a, b| {
            a.toi
                .total_cmp(&b.toi)
                .then_with(|| a.body.index.cmp(&b.body.index))
                .then_with(|| a.body.generation.cmp(&b.body.generation))
        });
        Ok(())
    }

    /// AABB와 겹치는 물체들 (슬롯 오름차순). 광역 검색·에디터 박스 선택용.
    pub fn query_aabb(&self, aabb: Aabb, out: &mut Vec<BodyHandle>) -> Result<(), ApiError> {
        self.query_aabb_filtered(aabb, QueryFilter::ALL, out)
    }

    /// 필터를 받는 AABB 질의.
    pub fn query_aabb_filtered(
        &self,
        aabb: Aabb,
        filter: QueryFilter,
        out: &mut Vec<BodyHandle>,
    ) -> Result<(), ApiError> {
        if !maru_collide::vec_in_range(aabb.min) || !maru_collide::vec_in_range(aabb.max) {
            return Err(ApiError::InvalidValue);
        }
        out.clear();
        if out.try_reserve(self.bodies.len()).is_err() {
            return Err(ApiError::ResourceLimit);
        }
        for i in 0..self.bodies.len() {
            if !self.passes(i, filter) {
                continue;
            }
            if aabb.overlaps(self.body_aabb(i)) {
                out.push(self.handle_of(i));
            }
        }
        Ok(())
    }

    /// **셰이프 스윕 질의** — 형상을 `motion`만큼 밀었을 때 처음 닿는 것.
    ///
    /// 캐릭터 컨트롤러(설계 §9 CharacterVirtual)와 호스트의 "이 상자가 여기로 갈 수 있나"
    /// 류 질의의 정본이다. 레이 캐스트와 달리 **두께가 있는** 이동을 본다.
    ///
    /// 순서 규약: 후보를 슬롯 오름차순으로 훑고 최소 TOI를 고른다. 동률은 **낮은 슬롯**이
    /// 이긴다 — 같은 장면이면 같은 답이어야 한다(L0).
    pub fn sweep_shape(
        &self,
        shape: &maru_collide::Shape,
        pose: maru_collide::Pose,
        motion: Vec3,
        filter: QueryFilter,
    ) -> Result<Option<SweepHit>, ApiError> {
        // 변위 상한은 `2*MAX_INPUT`(두 지원 위치의 최대 차)이고, **끝점**은 지원 월드 안이어야
        // 한다 — 예전 검사는 정상 18km 스윕을 거절하면서 x=9km에서 +9km(끝점 18km, 월드 밖)는
        // 통과시켰다(확인검수 rev_7b22e13d MAJOR). 월드 층은 "지원 월드" 개념을 갖고 있으므로
        // 끝점까지 본다(저수준 디스패처는 그 개념이 없어 `Inf` 차단만 한다).
        let max_motion = 2.0 * maru_collide::MAX_INPUT;
        let motion_ok = motion.is_finite()
            && motion.x.abs() <= max_motion
            && motion.y.abs() <= max_motion
            && motion.z.abs() <= max_motion;
        if !maru_collide::vec_in_range(pose.pos)
            || !motion_ok
            || !maru_collide::vec_in_range(pose.pos + motion)
        {
            return Err(ApiError::InvalidValue);
        }
        // **자세를 정규화한다.** 비단위 쿼터니언은 회전이 아니라 회전+축소를 만들어,
        // 아래 AABB 선판정과 GJK가 **찌그러진 형상**으로 조용히 틀린 답을 낸다
        // (확인검수 rev_2e93ebf2 BLOCKER). `set_pose`와 같은 규약이고, 부동소수 드리프트로
        // 정확한 단위는 드물어 거절이 아니라 정규화가 맞다 — 불가능한 것(0·NaN)만 거절한다.
        // 디스패처도 같은 정규화를 하지만 여기서 안 하면 **선판정이 그보다 먼저** 잘못된
        // 자세를 쓴다(그러면 후보가 걸러져 캐스트에 도달하지도 않는다).
        let pose = maru_collide::Pose {
            pos: pose.pos,
            rot: pose.rot.try_normalize().ok_or(ApiError::InvalidValue)?,
        };
        if shape.validate().is_err() {
            return Err(ApiError::InvalidValue);
        }
        // **스윕 주체는 볼록 셰이프만**. 하이트필드·복합을 넣으면 대상이 일반 convex일 때
        // GJK로 내려가 `support_core`의 `unreachable!`에 닿는다(검수 rev_1e81559e BLOCKER).
        // 복합을 스윕하고 싶으면 자식별로 호출하는 것이 호출자의 몫이다.
        if matches!(
            shape,
            maru_collide::Shape::Heightfield { .. }
                | maru_collide::Shape::TriMesh { .. }
                | maru_collide::Shape::Compound { .. }
        ) {
            return Err(ApiError::InvalidValue);
        }
        // 스윕 봉투로 후보를 거른다 — 전수 검사보다 싸고 결과는 같다.
        let a0 = maru_collide::transformed_aabb(
            shape.try_local_aabb().map_err(|_| ApiError::InvalidValue)?,
            &pose,
        );
        let sweep = a0.merge(Aabb::new(a0.min + motion, a0.max + motion));
        let mut best: Option<SweepHit> = None;
        let mut spent_cells = 0usize;
        for i in 0..self.bodies.len() {
            if !self.passes(i, filter) {
                continue;
            }
            if !sweep.overlaps(self.body_aabb(i)) {
                continue;
            }
            // 하이트필드는 셀 예산을 **미리** 본다 — 공개 질의는 보수적 정지(가짜 히트)가
            // 아니라 **오류**로 알려야 호스트가 질의를 좁힐 수 있다(확인검수 rev_7efe4e35).
            if let Some(hf) = self.bodies[i].shape.heightfield_ref() {
                // **디스패치와 정확히 같은 봉투**를 재야 한다 — margin을 빼면 경계에서
                // "질의는 예산 안, 내부는 예산 밖"이 되어 계약이 갈린다(확인검수 rev_d390e73d).
                let a0 = maru_collide::transformed_aabb(
                    shape.try_local_aabb().map_err(|_| ApiError::InvalidValue)?,
                    &pose,
                );
                let sw = a0.merge(Aabb::new(a0.min + motion, a0.max + motion));
                let m = self.params.speculative_margin;
                let pad = Vec3::new(m, m, m);
                let sw = Aabb::new(sw.min - pad, sw.max + pad);
                let local = maru_collide::world_aabb_to_local(sw, &self.bodies[i].pose);
                if let Some(range) = hf.cell_range(local) {
                    // **실제로 볼 셀만 센다.** raw 셀 수를 세면 멀리 있는 봉우리 하나가
                    // 지형 AABB를 높여 후보가 0개인 질의까지 거절한다(확인검수 rev_e50670ad).
                    let cells =
                        maru_collide::overlapping_cell_count(&hf, range, local.min.y, local.max.y);
                    // **질의 하나가 나눠 쓰는 예산.** 바디마다 따로 4096을 주면 같은 자리에
                    // 지형 100개를 겹친 장면에서 한 번의 스윕이 40만 셀을 훑는다
                    // (확인검수 rev_97887d75 MAJOR의 `shape_penetrates`와 같은 결함).
                    spent_cells = spent_cells.saturating_add(cells);
                    if spent_cells > maru_collide::MAX_HEIGHTFIELD_CELLS_PER_QUERY {
                        return Err(ApiError::ResourceLimit);
                    }
                }
            }
            // 디스패처는 **오류를 값으로** 돌려준다 — 예산 초과와 잘못된 입력이 미스로
            // 위장되지 않는다(위 선판정이 같은 예산을 보므로 여기 오류는 이론상 안 나지만,
            // 두 계산이 갈리는 날 조용히 "지나갈 수 있다"가 되는 것을 막는다).
            let hit = maru_collide::try_shape_cast_dispatch(
                shape,
                &pose,
                motion,
                &self.bodies[i].shape,
                &self.bodies[i].pose,
                self.params.speculative_margin,
            )
            .map_err(|e| match e {
                maru_collide::CastReject::BudgetExceeded => ApiError::ResourceLimit,
                maru_collide::CastReject::InvalidInput => ApiError::InvalidValue,
            })?;
            if let Some(h) = hit {
                let better = best.as_ref().is_none_or(|b| h.toi < b.toi);
                if better {
                    best = Some(SweepHit {
                        body: self.handle_of(i),
                        toi: h.toi,
                        point: h.point,
                        normal: h.normal,
                    });
                }
            }
        }
        Ok(best)
    }

    /// **형상이 지금 무언가에 파고들어 있는가** — 캐릭터가 이동 전에 묻는 질문.
    ///
    /// 스윕(`shape_cast`)은 CCD 계약상 "이미 겹친 상태"를 `None`으로 돌려준다(그게 옳다:
    /// 이동 시작점이 이미 접촉이면 TOI가 없다). 그런데 캐릭터가 그 `None`을 "장애물 없음"으로
    /// 읽으면 벽에 살짝 묻힌 채로 **전체 이동을 적용해 벽을 통과한다**(확인검수 rev_7efe4e35
    /// BLOCKER). 그래서 겹침은 스윕이 아니라 이 질의로 따로 본다.
    ///
    /// 판정은 **코어 침투**다 — speculative margin 안의 근접(분리 ≥ 0)은 겹침이 아니다.
    /// 바닥에 서 있는 캐릭터가 매 프레임 "끼었다"가 되면 쓸 수 없다.
    pub fn shape_penetrates(
        &self,
        shape: &maru_collide::Shape,
        pose: maru_collide::Pose,
        filter: QueryFilter,
    ) -> Result<bool, ApiError> {
        if !maru_collide::vec_in_range(pose.pos) || shape.validate().is_err() {
            return Err(ApiError::InvalidValue);
        }
        // 자세 정규화 — 근거는 `sweep_shape`의 같은 자리 주석에.
        let pose = maru_collide::Pose {
            pos: pose.pos,
            rot: pose.rot.try_normalize().ok_or(ApiError::InvalidValue)?,
        };
        let aabb = maru_collide::transformed_aabb(
            shape.try_local_aabb().map_err(|_| ApiError::InvalidValue)?,
            &pose,
        );
        // **예산은 질의 하나가 나눠 쓴다.** 바디마다 새로 4096이면 같은 자리에 지형 100개를
        // 겹친 장면에서 한 번의 질의가 40만 셀을 훑는다(확인검수 rev_97887d75 MAJOR).
        let mut cells = maru_collide::MAX_HEIGHTFIELD_CELLS_PER_QUERY;
        let mut found = false;
        // 순회가 끊긴 바디가 하나라도 있으면 **음성 답을 확정할 수 없다**(양성은 예외).
        let mut incomplete = false;
        for i in 0..self.bodies.len() {
            if found {
                break;
            }
            if !self.passes(i, filter) || !aabb.overlaps(self.body_aabb(i)) {
                continue;
            }
            // **작업량 상한은 순회 안에 있다**(4096셀) — 예전에는 여기서 셀 수를 미리 세어
            // 초과면 즉시 거절했는데, 그러면 첫 셀부터 깊이 박혀 있는 큰 바디도 AABB가 넓으면
            // **증명 가능한 `true`를 영원히 못 받는다**(확인검수 rev_2e93ebf2 MAJOR:
            // 캐릭터의 매 이동이 오류로 고정된다). `any_contact`와 같은 규칙을 쓴다 —
            // 양성은 한 점으로 증명되고, 음성만 순회 완주를 요구한다.
            let walk = maru_collide::for_each_contact_with_budget(
                shape,
                &pose,
                &self.bodies[i].shape,
                &self.bodies[i].pose,
                0.0,
                &mut cells,
                &mut |_, _, m| {
                    if m.points().iter().any(|p| p.separation < -1e-4) {
                        found = true;
                    }
                },
            );
            if found {
                // 하나 찾았으면 끝 — 뒤에 무엇이 있든 "겹친다"는 답은 바뀌지 않는다.
                return Ok(true);
            }
            // 판정 미완료는 **기억만 하고 계속 본다**. 즉시 오류로 나가면 답이 슬롯 순서에
            // 의존한다: 포화하는 지형이 앞 슬롯이면 뒤에 깊게 박힌 구를 못 보고
            // `ResourceLimit`, 구가 앞이면 `Ok(true)`가 된다 — 같은 장면이 삽입 순서에 따라
            // 다른 답을 낸다(확인검수 rev_c73f897c MAJOR). 뒤쪽에 확정 양성이 있으면 그것이
            // 이긴다. 작업량은 예산이 이미 묶었으므로 계속 봐도 유계다.
            // 못 찾았는데 순회가 끊겼다 = **판정하지 못했다**. `false`로 뭉개면 캐릭터가
            // 벽 속에서 출발한다. **포화도 완주가 아니다**(확인검수 rev_97887d75 MAJOR):
            // 64번째 매니폴드에서 끊겼다는 뜻이고, 앞 64개가 침투 임계에 못 미치는 얕은
            // 접촉이었다면 뒤쪽의 깊은 침투를 못 본 채 "안 겹친다"가 된다.
            match walk {
                maru_collide::ContactWalk::Complete => {}
                maru_collide::ContactWalk::Saturated
                | maru_collide::ContactWalk::BudgetExceeded => incomplete = true,
                // 형상·입력 오류는 **즉시** 오류다(순서 의존이 아니라 계약 위반이다).
                maru_collide::ContactWalk::InvalidShape
                | maru_collide::ContactWalk::InvalidInput => {
                    return Err(ApiError::InvalidValue);
                }
            }
        }
        // 여기까지 왔으면 양성은 없었다. 순회가 한 번이라도 끊겼으면 "겹치지 않는다"를
        // 확정할 수 없다 — 그 경우만 자원 오류다.
        debug_assert!(!found, "양성은 루프 안에서 즉시 반환한다");
        if incomplete {
            Err(ApiError::ResourceLimit)
        } else {
            Ok(false)
        }
    }

    /// 점을 품은 물체들 (슬롯 오름차순). 겉면 기준 — convex radius를 포함한다.
    pub fn query_point(&self, p: Vec3, out: &mut Vec<BodyHandle>) -> Result<(), ApiError> {
        self.query_point_filtered(p, QueryFilter::ALL, out)
    }

    /// 필터를 받는 점 질의.
    pub fn query_point_filtered(
        &self,
        p: Vec3,
        filter: QueryFilter,
        out: &mut Vec<BodyHandle>,
    ) -> Result<(), ApiError> {
        if !maru_collide::vec_in_range(p) {
            return Err(ApiError::InvalidValue);
        }
        out.clear();
        if out.try_reserve(self.bodies.len()).is_err() {
            return Err(ApiError::ResourceLimit);
        }
        for i in 0..self.bodies.len() {
            if !self.passes(i, filter) {
                continue;
            }
            let b = &self.bodies[i];
            // 월드 바디의 형상·자세는 삽입·복원 경계에서 검증됐으므로 `try_` 오류는
            // 도달 불가다. 그래도 값으로 받아 **미스로 뭉개지 않는다** — 언젠가 경계가
            // 갈리면 "점이 안 들어 있다"가 아니라 오류로 드러나야 한다.
            if maru_collide::try_contains_point(&b.shape, &b.pose, p)
                .map_err(|_| ApiError::InvalidValue)?
            {
                out.push(self.handle_of(i));
            }
        }
        Ok(())
    }

    fn check_ray(origin: Vec3, dir: Vec3, max_toi: f32) -> Result<(Vec3, Vec3, f32), ApiError> {
        if !maru_collide::vec_in_range(origin)
            || !maru_collide::vec_in_range(dir)
            || !(max_toi.is_finite() && max_toi >= 0.0)
        {
            return Err(ApiError::InvalidValue);
        }
        if dir.length_sq() <= 0.0 {
            return Err(ApiError::InvalidValue);
        }
        Ok((origin, dir, max_toi))
    }

    /// 질의에 보이는 슬롯인가.
    ///
    /// **OOB는 보이지 않는다** — 월드 범위를 벗어난 물체는 "시뮬레이션에서 완전히
    /// 빠진다"가 계약이고(위 `is_out_of_bounds`), 질의만 다르게 취급하면 호스트가
    /// 충돌하지 않는 것을 광선으로 맞히게 된다. **잠든 물체는 보인다** — 자는 것은
    /// 최적화지 부재가 아니다.
    fn query_visible(&self, i: usize) -> bool {
        self.alive[i] && !self.oob[i]
    }

    /// 질의 필터 통과 판정 (가시성 + 그룹 + 센서 + 제외 핸들).
    fn passes(&self, i: usize, f: QueryFilter) -> bool {
        if !self.query_visible(i) {
            return false;
        }
        let b = &self.bodies[i];
        if b.sensor && !f.include_sensors {
            return false;
        }
        // **단방향이다** — 물체의 `membership`만 본다.
        //
        // 처음엔 물리와 같은 양방향 규칙으로 짰는데(검수 rev_710b92d2 반례로 뒤집었다),
        // 그러면 `filter = 0`인 물체(아무것과도 안 부딪히는 유령)가 **필터 없는 광선에도
        // 안 맞는다**. "아무것과도 안 부딪힌다"와 "안 보인다"는 다른 말이고, 그러면
        // 호스트가 그 물체를 클릭할 수도 없다. 물체의 `filter`는 "내가 무엇과 부딪히려
        // 하는가"라서 질의와는 무관하다.
        if (b.membership & f.groups) == 0 {
            return false;
        }
        if let Some(ex) = f.exclude {
            // **세대까지 본다** — 슬롯만 보면 제거 후 재사용된 물체를 죽은 핸들이 계속 가린다.
            if ex.index as usize == i && ex.generation == self.generations[i] {
                return false;
            }
        }
        true
    }

    /// 슬롯의 월드 AABB (sweep 없음 — 질의용 정지 경계).
    ///
    /// 스텝의 브로드페이즈 AABB와 **같은 식**을 쓴다: 중심 `pos + R·c`, 반크기 `|R|·h`.
    /// 두 곳에 따로 적으면 하나만 고쳐지는 비대칭이 생긴다(이 저장소의 반복 결함 계열).
    pub(crate) fn body_aabb(&self, i: usize) -> Aabb {
        let b = &self.bodies[i];
        shape_world_aabb(&b.shape, &b.pose)
    }

    fn handle_of(&self, i: usize) -> BodyHandle {
        BodyHandle {
            index: i as u32,
            generation: self.generations[i],
        }
    }

    fn ray_body(&self, i: usize, origin: Vec3, dir: Vec3, max_toi: f32) -> Option<RayHit> {
        if !self.query_visible(i) {
            return None;
        }
        let b = &self.bodies[i];
        // 월드 바디는 검증됐다(위 `query_point_filtered` 주석과 같은 규약) — 오류는
        // 도달 불가이고, 여기 시그니처는 `Option`이라 오류를 미스로 접는다. 그래서
        // **호출 전에 인자를 검증하는 책임은 공개 진입점**(`cast_ray`·`cast_ray_all`)에 있다.
        let rc = maru_collide::try_cast_ray(&b.shape, &b.pose, origin, dir, max_toi)
            .ok()
            .flatten()?;
        Some(RayHit {
            body: self.handle_of(i),
            toi: rc.toi,
            point: origin + dir * rc.toi,
            normal: rc.normal,
            started_inside: rc.started_inside,
        })
    }

    /// 감쇠 설정 (1/s). 음수·NaN·상한 초과는 거부한다.
    pub fn set_damping(
        &mut self,
        h: BodyHandle,
        linear: f32,
        angular: f32,
    ) -> Result<(), ApiError> {
        Self::fp_guard()?;
        if !(linear >= 0.0 && maru_collide::in_range(linear))
            || !(angular >= 0.0 && maru_collide::in_range(angular))
        {
            return Err(ApiError::InvalidValue);
        }
        let i = self.resolve(h)?;
        self.bodies[i].linear_damping = linear;
        self.bodies[i].angular_damping = angular;
        Ok(())
    }

    pub fn set_linear_velocity(&mut self, h: BodyHandle, v: Vec3) -> Result<(), ApiError> {
        Self::fp_guard()?;
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
        Self::fp_guard()?;
        if !maru_collide::vec_angular_in_range(v) {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].angvel = v;
        Ok(())
    }

    /// 텔레포트 (설계 §9: 속도를 만들지 않는다 — kinematic 구동은 별도 API).
    pub fn set_pose(&mut self, h: BodyHandle, pose: maru_collide::Pose) -> Result<(), ApiError> {
        Self::fp_guard()?;
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
        Self::fp_guard()?;
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
    /// PCM 캐시가 쥐고 있는 총 용량(바이트) — 자리 채우기가 용량을 무한정 쥐지 않는지
    /// 관측하는 창이다(테스트 전용).
    #[cfg(test)]
    pub fn pcm_cache_capacity_bytes(&self) -> usize {
        self.contact_cache
            .values()
            .map(|v| v.capacity() * core::mem::size_of::<(FeatureId, CachedImpulse)>())
            .sum()
    }

    /// 캐시 엔트리 중 **용량/길이 비가 가장 나쁜** 것의 `(len, capacity)` (테스트 전용).
    #[cfg(test)]
    pub fn pcm_cache_worst_capacity_ratio(&self) -> Option<(usize, usize)> {
        self.contact_cache
            .values()
            .map(|v| (v.len(), v.capacity()))
            .max_by_key(|(l, c)| c.saturating_sub(*l))
    }

    /// 접촉 이벤트 출력 버퍼의 현재 보유 용량.
    #[cfg(test)]
    pub fn contact_events_capacity(&self) -> usize {
        self.contact_events.capacity()
    }

    /// **이벤트 방출 구간에 진입한 순간**의 용량 (테스트 전용).
    ///
    /// 이것이 있어야 "이벤트 구간에서 재할당이 없었다"를 관측할 수 있다. 스텝이 끝난 뒤의
    /// 용량과 이 값을 비교하면 된다 — 스텝 시작의 선점 블록에서 자라는 것(정상)과
    /// 방출 중에 자라는 것(계약 위반)이 그제서야 구분된다.
    /// 스텝 후 용량만 보는 검사는 **어떤 Vec에서도 참**이라 아무것도 잠그지 못한다
    /// (교차검수 rev_18633fde MAJOR — 첫 판 테스트가 정확히 그랬다).
    #[cfg(test)]
    pub fn contact_events_capacity_at_emit(&self) -> usize {
        self.ev_cap_at_emit
    }

    /// 바디가 속한 아일랜드 id (`u32::MAX` = 아일랜드 밖 = static·무접촉·수면).
    ///
    /// 테스트 전용 창이다. **병렬 솔버 전략을 아일랜드 위에 세울 수 있는지**가 이 분포에
    /// 달려 있어서 있다(`island_structure_of_target_scenes`가 그것을 측정한다).
    #[cfg(test)]
    pub fn island_of_index(&self, i: usize) -> u32 {
        self.island[i]
    }

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
        Self::fp_guard()?;
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
    /// 커맨드 적용 (오류 반환형).
    ///
    /// **큐를 비우기 전에 FP 환경을 본다** — 가드 없이 drain하면 비표준 환경에서 호출 한 번에
    /// 큐가 사라지고 각 커맨드가 `rejected`로 소비된다(확인검수 rev_7efe4e35 BLOCKER:
    /// 재시도할 커맨드가 없어진다). 실패 시 큐는 **그대로**다.
    pub fn try_flush_commands(&mut self) -> Result<CommandReport, ApiError> {
        Self::fp_guard()?;
        Ok(self.flush_commands_inner())
    }

    /// 커맨드 적용 (관례상 비-fallible 판 — 계약 위반이면 패닉).
    /// 엔진 내부(스텝)는 이미 진입에서 검증했으므로 `flush_commands_inner`를 직접 부른다.
    pub fn flush_commands(&mut self) -> CommandReport {
        self.try_flush_commands()
            .expect("non-canonical FP environment (see World::canonicalize_fp_environment)")
    }

    fn flush_commands_inner(&mut self) -> CommandReport {
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
                        self.invalidate_bp();
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
            // 잠든 바디의 속도는 정확히 0이다(변이 API는 전부 wake 경유, 복원은 asleep ⇒
            // 속도 0을 강제) — 클램프가 손댈 게 없으므로 건너뜀은 항등이다.
            if !self.alive[i] || b.body_type == BodyType::Static || self.asleep[i] {
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

    /// **경계 이탈의 배제는 다음 스텝부터다** — 넘는 스텝은 끝까지 돈다.
    ///
    /// 검수(rev_59dcb940)가 "넘은 바디를 같은 스텝의 relax·조인트 솔버에서도 빼야 한다"며
    /// 남은 서브스텝을 전역 중단(`break 'substeps`)하라고 제안했다. **채택하지 않았다** —
    /// 근거는 정밀도 실측이다:
    ///
    /// - 한 스텝 최대 이동 = `max_linear_velocity × dt` (기본 500 m/s → 8.3m, 호스트 상한
    ///   1e4 m/s → 167m). 즉 이탈은 10km + 167m 이내다.
    /// - f32 간격: 9999m에서 9.766e-4 m, **10167m에서도 9.766e-4 m**(같은 binade). 다음 계단은
    ///   16384m부터다. 즉 넘은 자세의 수치 품질은 경계 안쪽과 **동일**하고, 그 상태로 계산한
    ///   조인트 임펄스는 9995m에서 계산한 것과 같은 정도로 유효하다.
    /// - 반면 전역 중단은 **다른 모든 바디의 남은 서브스텝을 빼먹는다** — 한 바디의 경계 이탈이
    ///   월드 전체의 그 프레임을 under-integrate하는 관측 가능한 인공물이다.
    ///
    /// 없는 인공물을 막으려고 있는 인공물을 만들지 않는다. 대신 이탈은 이 함수가 관측 가능하게
    /// 보고하고(`Diagnostics::out_of_bounds`), CCD는 범위 밖 자세로 **캐스트만** 하지 않는다
    /// (원인 오보 방지 — 그 자리 주석 참조). `is_out_of_bounds` 계약("시뮬레이션에서 제외")은
    /// **플래그가 선 뒤**, 즉 다음 스텝부터 성립한다.
    ///
    /// **"끝까지 돈다"는 병진 적분도 포함한다.** 처음 두 판은 범위 밖에서 `d=0`·`ccd_frozen`으로
    /// 병진을 얼렸고, 그러면 잠깐 넘었다가 중력·조인트가 되돌릴 바디가 복귀하지 못해 여기서
    /// 영구 제거됐다 — 정적 후보가 0개인 장면에서도 **CCD 켬/끔이 궤적을 바꾸는** 결함이다
    /// (확인검수 rev_cccbba53). 위 정밀도 논증을 받아들였다면 적분을 막을 이유가 없고,
    /// 최종 위치가 여전히 밖일 때만 이 함수가 OOB를 세운다.
    /// 회귀: `ccd_does_not_change_the_trajectory_when_there_are_no_static_candidates`.
    ///
    /// 월드 범위를 벗어난 바디를 **경계에 고정하고 재운다**(리뷰 BLOCKER: 위치 정밀도
    /// 계약이 step에 대해 닫혀 있지 않았다). 10km 밖에서는 f32 간격이 최소 특징 크기를
    /// 넘어 충돌 판정이 의미를 잃으므로, 조용히 계속 계산하는 것이 가장 나쁘다.
    /// 동결은 관측 가능하고(`Diagnostics::out_of_bounds`) 상태는 복원 가능하게 남는다.
    /// 더 넓은 월드가 필요하면 호스트가 floating-origin으로 rebase해야 한다(v1 계약).
    fn enforce_world_bounds(&mut self) {
        let max = maru_collide::MAX_INPUT;
        for i in 0..self.bodies.len() {
            // 여기는 **스텝당 1회**라 활성 필터를 넣지 않는다 — 잠든 바디를 건너뛰면
            // `oob ⇒ !asleep` 불변식과 상호작용해 경계 이탈이 기록되지 않는 경로가 생긴다
            // (실측으로 회귀에 걸렸다). 비용은 서브스텝 루프에 비하면 무시할 만하다.
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
            self.invalidate_bp();
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
        // **이번 스텝에 쓰이지 않은 상태는 닫지 않는다** — 포화는 멱등이라, 직전 닫힘 이후
        // 변하지 않은 값은 여전히 닫혀 있다. 잠든(dormant) 바디가 그 경우다: 모든 변이 API는
        // wake를 거치고(`mutate`/`accept_force`), 복원은 "asleep ⇒ 속도 0 + 상태 범위 리더"를
        // 강제하므로, 잠든 바디의 pos·vel은 잠들던 스텝(그때는 integrate_gate=true라 여기서
        // 닫혔다) 이후 불변이다. 전원 수면 pile-3k에서 이 패스가 post의 지배 비용이었다.
        let dormant = |i: usize, asleep: &[bool], gate: &[bool]| {
            asleep[i] && !gate.get(i).copied().unwrap_or(false)
        };
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if !self.alive[i] || dormant(i, &self.asleep, &self.integrate_gate) {
                continue;
            }
            sat_vec!(b.pose.pos);
            sat_vec!(b.linvel);
            sat_vec!(b.angvel);
        }
        for j in self.joints.iter_mut() {
            // 양 끝점이 dormant면 이 조인트는 이번 스텝에 풀리지 않았다 — 임펄스 불변.
            // 잠들던 스텝엔 끝점 gate가 true라 여기 걸리지 않는다(그 스텝의 쓰기를 닫는다).
            if dormant(j.ia(), &self.asleep, &self.integrate_gate)
                && dormant(j.ib(), &self.asleep, &self.integrate_gate)
            {
                continue;
            }
            sat_vec!(j.linear_impulse);
            sat!(j.angular_impulse[0]);
            sat!(j.angular_impulse[1]);
        }
        // PCM 캐시는 여기서 닫지 않는다 — **쓰는 시점(post 5절 insert)에 닫는다**. 캐시 엔트리의
        // 쓰기 경로는 그 insert와 복원(상태 범위 리더가 거부) 둘뿐이라, 삽입 시 닫으면 관측
        // 가능한 모든 시점에 닫혀 있다. 매 스텝 전체 캐시 순회(정착 씬에서 접촉 수천 엔트리 ×
        // 필드 포화 + try_normalize의 sqrt)가 통째로 사라진다.
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

    /// 영속 스크래치가 **목표를 넘겨 쥐고만 있는** 바이트.
    ///
    /// 회계는 "이번 스텝에 필요한 양"으로 세는데 `Vec` 용량은 저절로 줄지 않는다. 큰
    /// 월드를 한 번 돌린 뒤 작은 월드로 복원하면 그 차이가 회계 밖에 남고, 128 MiB
    /// 총상한 주장이 거짓이 된다(확인검수 rev_21c1a97f MAJOR — 참여자 버퍼만 정확히
    /// 세고 나머지를 빠뜨렸다).
    /// `contact_event_target` = 이번 스텝의 이벤트 출력 상한(`ncap + touching.len()`).
    /// 이벤트 버퍼는 **영속**이라 큰 장면을 한 번 돌면 용량이 남는데, 그 초과분을 세지 않으면
    /// 작은 장면으로 돌아온 뒤에도 과거 버퍼가 총상한 밖에 영구 잔류한다
    /// (교차검수 rev_18633fde MAJOR).
    fn persistent_excess_bytes(
        &self,
        nbody: usize,
        constraint_cap: usize,
        contact_event_target: usize,
    ) -> usize {
        let ex = |cap: usize, target: usize, elem: usize| cap.saturating_sub(target) * elem;
        self.solver_state
            .excess_capacity_bytes(nbody)
            .saturating_add(ex(
                self.inv_inertia_local.capacity(),
                nbody,
                core::mem::size_of::<Vec3>(),
            ))
            .saturating_add(ex(
                self.ext_force.capacity(),
                nbody,
                core::mem::size_of::<(Vec3, Vec3)>(),
            ))
            .saturating_add(ex(
                self.damping.capacity(),
                nbody,
                core::mem::size_of::<(f32, f32)>(),
            ))
            .saturating_add(ex(self.integrate_gate.capacity(), nbody, 1))
            .saturating_add(ex(self.gravity_gate.capacity(), nbody, 1))
            .saturating_add(ex(
                self.island_parent.capacity(),
                nbody,
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(ex(
                self.contact_events.capacity(),
                contact_event_target,
                core::mem::size_of::<ContactEvent>(),
            ))
            .saturating_add(ex(self.sleep_bak.capacity(), nbody, 1))
            .saturating_add(ex(
                self.sleep_timer_bak.capacity(),
                nbody,
                core::mem::size_of::<f32>(),
            ))
            .saturating_add(ex(
                self.contact_blocks.capacity(),
                constraint_cap / 4,
                core::mem::size_of::<solver_block::ContactBlock>(),
            ))
    }

    /// 스텝 시작 위생: **지금 필요한 것보다 터무니없이 크게 쥐고 있으면 미리 놓아준다.**
    ///
    /// 총상한 검사와 반납은 브로드페이즈가 새 페어 배열을 만든 **뒤에** 돈다. 그래서 이전
    /// 큰 장면이 남긴 스크래치를 그대로 쥔 채 새 페어 배열을 잡는 순간 상한을 넘길 수 있고,
    /// 그 할당은 fallible이 아니다 — 검사에 도달하기 전에 죽는다(확인검수 rev_51f69fbd
    /// BLOCKER). 여기서 보유량을 "현재 필요의 4배" 안으로 미리 눌러 그 창을 없앤다.
    ///
    /// 4배는 이력 현상(hysteresis)이다 — 장면이 조금 출렁일 때마다 반납·재예약을 반복하면
    /// 그게 더 비싸다. 평상시에는 조건이 거짓이라 아무 일도 하지 않는다.
    ///
    /// **페어 두 벌(`bp_pairs`·`bp_pairs_scratch`)과 `integrate_gate`는 건드리지 않는다** —
    /// 앞의 둘은 이번 스텝 증분 경로의 입력이고, 게이트는 직전 스텝 값으로 읽힌다.
    fn release_stale_scratch(&mut self, nbody: usize) {
        let bloated = |cap: usize, need: usize| cap > 4 * need.max(64);
        if bloated(self.solver_state.lin_capacity_pub(), nbody) {
            self.solver_state.release();
        }
        macro_rules! trim {
            ($v:expr, $need:expr) => {
                if bloated($v.capacity(), $need) {
                    $v = Vec::new();
                }
            };
        }
        trim!(self.inv_inertia_local, nbody);
        trim!(self.ext_force, nbody);
        trim!(self.damping, nbody);
        trim!(self.island_parent, nbody);
        trim!(self.gravity_gate, nbody);
        trim!(self.sleep_bak, nbody);
        trim!(self.sleep_timer_bak, nbody);
        // 블록·참여자 버퍼는 이 시점에 목표를 모른다 — 직전 스텝 사용량(len)을 근사로 쓴다.
        let blocks_need = self.contact_blocks.len();
        trim!(self.contact_blocks, blocks_need);
        for b in self.narrow_bufs.iter_mut() {
            b.trim_if_bloated();
        }
    }

    /// 목표를 넘겨 쥐고 있는 영속 스크래치를 **반납**한다.
    ///
    /// 상한을 넘었다고 스텝을 거절하는 것은 답이 아니다 — 용량은 저절로 줄지 않으므로
    /// 그 월드는 **영원히** 거절된다. 안 쓰는 것을 실제로 돌려주고 다시 잰다.
    /// 평소 경로에서는 초과분이 0이라 이 함수가 불리지 않는다.
    ///
    /// **축소가 아니라 해제다.** `Vec::shrink_to`는 재할당을 시도할 수 있고 실패하면
    /// abort인데, 이 함수는 자원이 빠듯한 순간에만 불린다 — 거기서 abort하면
    /// `ResourceLimit`을 돌려주려던 보호 경로가 프로세스를 죽인다(확인검수 rev_de34bbb5).
    /// `Vec::new()` 대입은 할당 없이 옛 버퍼를 반납하고, 용량은 뒤따르는 통합 fallible
    /// 블록이 wake 폐쇄 이전에 다시 잡는다. 목표치를 남길 이유도 없다 —
    /// 남기면 참여자 버퍼가 과거 청크 용량을 계속 쥔 채 회복하지 못한다(같은 검수 MAJOR).
    ///
    /// 비워도 되는 근거는 전부 "이번 스텝 안에서 clear 후 다시 채운다"이다.
    /// 유일한 예외가 `integrate_gate`인데, 증분 브로드페이즈 경로(`fresh_pairs.is_some()`)
    /// 에서만 이 시점 뒤에 직전 값으로 읽힌다 — 그리고 그 경로는 성립 조건이
    /// `len == bodies.len()`이라 애초에 초과분이 없다. 그래서 그때만 보존한다.
    fn release_oversized_scratch(&mut self, preserve_integrate_gate: bool) {
        self.solver_state.release();
        self.inv_inertia_local = Vec::new();
        self.ext_force = Vec::new();
        self.damping = Vec::new();
        self.island_parent = Vec::new();
        self.contact_blocks = Vec::new();
        // 이벤트 출력도 영속이라 반납 대상이다 — 반납하지 않으면 위 초과 회계가 잡은 바이트를
        // 되돌릴 방법이 없어 "반납 후 재측정" 회복 경로가 작동하지 않는다.
        // 이 시점은 스텝 시작(이미 `clear`됨)이라 버릴 내용이 없다.
        self.contact_events = Vec::new();
        // 페어 핑퐁 두 벌 — 살아있는 목록은 지역 `pairs`가 들고 있다.
        self.bp_pairs = Vec::new();
        self.bp_pairs_scratch = Vec::new();
        self.gravity_gate = Vec::new();
        self.sleep_bak = Vec::new();
        self.sleep_timer_bak = Vec::new();
        if !preserve_integrate_gate {
            self.integrate_gate = Vec::new();
        }
        for b in self.narrow_bufs.iter_mut() {
            b.release();
        }
    }

    /// 테스트 전용 — 반납 경로 직접 호출.
    #[cfg(test)]
    pub(crate) fn release_scratch_for_test(&mut self, _nbody: usize) {
        self.release_oversized_scratch(false);
    }

    /// 테스트 전용 — per-body 영속 스크래치의 용량(회계가 미는 값과 대조하기 위해).
    #[cfg(test)]
    pub(crate) fn body_scratch_capacities(&self) -> [usize; 7] {
        [
            self.inv_inertia_local.capacity(),
            self.integrate_gate.capacity(),
            self.gravity_gate.capacity(),
            self.ext_force.capacity(),
            self.damping.capacity(),
            self.island_parent.capacity(),
            self.solver_state.lin_capacity(),
        ]
    }

    /// 지금 쓰는 참여자 수 (메인 스레드 포함, 1 = 완전 직렬).
    ///
    /// 요청한 수보다 작을 수 있다 — 스레드를 못 띄운 환경(wasm·제한된 호스트)에서는
    /// 실제로 확보한 수를 돌려준다(설계 §6.3 "조용한 폴백 금지").
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// 접촉 솔버의 **색 병렬**을 켠다 (기본 off). 결과는 켜든 끄든 비트 동일하다 —
    /// 이건 성능 스위치일 뿐이다(`parallel_color_solve_matches_serial_bit_for_bit`가 잠금).
    ///
    /// **왜 기본이 off인가 — 측정 결과다.** 4코어(가용 ~2, 부하 2.4) 리눅스 박스에서
    /// 피라미드 40단·60단·80단을 교차 실행 min-of-4로 재보니 솔브 단계가 참여자 수에
    /// 따라 **단조 악화**했다(40단: 3975 → 4451(2) → 4626(3) → 5014us(4)). 원인은 둘이다:
    /// ① 패스 자체의 병렬 효율이 낮다 — 워커가 뜨거운 상태의 격리 계측에서도 3참여자
    ///    1.48배가 상한이고 4에서는 되레 느려진다(색 7개면 배리어 7회에 색당 일감이
    ///    참여자당 수십 us뿐이라 동기화가 이득을 먹는다).
    /// ② 파이프라인 안에서는 패스마다 dispatch를 하는데(서브스텝 4 × 패스 5 = 20회),
    ///    직렬 구간이 스핀 창보다 길면 워커가 잠들어 깨우는 값이 붙는다.
    /// 반면 내로우페이즈는 스텝당 dispatch 1회에 청크가 커서 확실히 이긴다 — 그래서
    /// 그쪽만 기본으로 켠다.
    ///
    /// 코어가 많은 기계(8~16)나 접촉이 훨씬 많은 장면에서는 판단이 달라질 수 있다.
    /// 그 판정은 추측이 아니라 계측으로 한다:
    /// `MARU_THREADS=n cargo run --release -p maru-bench --bin pyramid100 -- 4 150 60 nosleep`이
    /// 단계별 귀속(narrow/solve/refresh/warm/bias/relax)을 찍는다.
    pub fn set_parallel_solver(&mut self, on: bool) {
        self.parallel_solver = on;
    }

    /// 접촉 솔버 색 병렬이 켜져 있는가.
    pub fn parallel_solver(&self) -> bool {
        self.parallel_solver
    }

    /// 참여자 수 변경 (설계 §5). **성능 노브일 뿐 시맨틱이 아니다** — 결과는 스레드 수와
    /// 무관하게 비트 동일하고(그것이 L0 결정론 계약이다), 스냅샷·상태 해시에도 들어가지
    /// 않는다. 그래서 스텝 사이 아무 때나 바꿔도 궤적이 갈리지 않는다.
    ///
    /// 워커는 **처음 병렬로 돌 때** 뜬다(지연 생성) — 병렬 문턱을 못 넘는 작은 월드는
    /// 스레드를 아예 만들지 않는다. 그래도 이 호출은 기존 워커를 정리하고 다시 만들므로
    /// 매 프레임 부를 것은 아니다.
    ///
    /// **가용 코어보다 크게 잡지 마라.** 워커는 짧게 스핀하다 눕는데, 참여자가 코어보다
    /// 많으면 실행 중인 참여자가 대기 중인 참여자에게 밀려 dispatch·배리어 지연이
    /// 수천 배가 된다(4코어에 5명 실측: dispatch 5ms, 배리어 4ms). 기본값
    /// (`available_parallelism`, 상한 8)이 그 선을 넘지 않는다.
    pub fn set_threads(&mut self, n: usize) {
        // 상한을 넘기는 요청은 **자른다** — 참여자가 코어보다 많으면 스핀 대기가 실행 중인
        // 참여자를 밀어내 dispatch·배리어 지연이 수천 배가 되고, 참여자별 스크래치도
        // 입력 크기와 무관하게 늘어난다(교차검수 rev_518b072b MAJOR).
        let n = n.clamp(1, crate::threads::MAX_THREADS);
        if n != self.pool.threads() {
            // 참여자 버퍼는 분할 수에 맞춰 커진 채 영속한다 — 분할이 바뀌면 과거 용량을
            // 새 분할에 얹지 않도록 버린다(확인검수 rev_071f1cd9 MAJOR). 회계는 보유
            // 용량을 세므로 안 버려도 상한은 지켜지지만, 안 쓸 메모리를 계속 쥐고 있으면
            // 그만큼 이번 스텝의 페어·제약 여유가 줄어든다.
            self.narrow_bufs.clear();
            self.pool = crate::threads::Pool::new(n);
        }
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

    /// 슬라이더(prismatic) 조인트의 축 방향 위치 `s` (m). 다른 종류이거나 없는
    /// 인덱스면 `None`. 인덱스는 `try_add_joint`가 돌려준 값이다.
    ///
    /// 호스트가 슬라이더의 현재 위치(문이 얼마나 열렸는지 등)를 읽는 경로다 —
    /// 한계·모터가 쓰는 좌표와 같은 식으로 계산해 표시값과 물리가 어긋나지 않게 한다.
    pub fn prismatic_translation(&self, joint: usize) -> Option<f32> {
        let j = self.joints.get(joint)?;
        // 엔드포인트가 살아 있어야 자세를 읽을 수 있다.
        if !self.alive.get(j.ia()).copied().unwrap_or(false)
            || !self.alive.get(j.ib()).copied().unwrap_or(false)
        {
            return None;
        }
        crate::joint::prismatic_translation(&self.bodies, j)
    }

    /// 조인트 추가 (오류 반환형). 엔드포인트는 **세대까지** 검증한다 — 인덱스만 보면
    /// 제거·재사용된 슬롯의 새 바디에 붙는다(리뷰 BLOCKER: ABA).
    pub fn try_add_joint(&mut self, mut joint: Joint) -> Result<usize, ApiError> {
        Self::fp_guard()?;
        // 생성자를 통과했더라도 **삽입 경계에서 다시** 검증한다(리뷰 BLOCKER: 공개 필드
        // 대입으로 생성 후 축을 NaN으로 바꿀 수 있었고, `length_sq() < 0.5` 비교는
        // NaN에서 false라 그대로 통과했다).
        joint.validate()?;
        // **삽입 경계에서 캐노니컬화한다** — 생성자를 안 거친 경로(공개 필드 대입·
        // `with_*` 체인)도 리더와 같은 표현으로 저장돼야 snapshot 왕복이 항등이다.
        joint.canonicalize()?;
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
        joint.clear_impulses(); // 재사용 Joint 값의 잔여 임펄스 차단(리뷰 지적)
        self.joints.push(joint);
        Ok(self.joints.len() - 1)
    }

    /// 슬롯 인덱스의 수면 여부 (죽은 슬롯 = false). 벌크 스테이징용.
    pub fn is_asleep_at_slot(&self, i: usize) -> bool {
        i < self.asleep.len() && self.alive[i] && self.asleep[i]
    }

    /// 슬롯 인덱스로 바디를 읽는다 (살아있지 않으면 `None`).
    ///
    /// 핸들 경로(`body`/`try_body`)와 달리 **세대를 보지 않는다**. flat buffer 바인딩이
    /// 슬롯 순서로 자세를 훑을 때 쓰라고 둔 것이고, 그 경로는 애초에 "지금 살아있는
    /// 슬롯의 현재 값"만 필요하다. 호스트가 핸들의 유효성을 물으려면 `is_alive`를 쓴다.
    pub fn body_at_slot(&self, i: usize) -> Option<&Body> {
        if i < self.bodies.len() && self.alive[i] {
            Some(&self.bodies[i])
        } else {
            None
        }
    }

    /// 접촉 이벤트 기준선 (정렬 — canonical 직렬화용).
    pub(crate) fn touching_keys(&self) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = self.touching.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// 복원 경로 전용 — 기준선을 통째로 교체한다.
    pub(crate) fn set_touching(&mut self, t: IntSet<(u32, u32)>) {
        self.touching = t;
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
        &IntMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
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
        cache: IntMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
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
        // **스텝 출력은 복원 경계에서 전부 버린다** — 큐 하나만 비우면 안 된다.
        //
        // 접촉 이벤트와 CCD 히트는 둘 다 "직전 스텝의 비캐노니컬 출력"이고 수명이 같다.
        // 복원해도 남아 있으면 폐기한 미래의 사건을 호스트가 다시 읽어 데미지·사운드·
        // 이펙트를 두 번 적용한다. `diag`가 0으로 초기화되어 `ccd_hits_dropped == 0`이라
        // **온전한 현재 목록처럼 보이기까지 한다**.
        // (rev_5e74f64f가 히트 큐를 지적했고, rev_6da1d7ba가 "같은 수명인데 이벤트만
        //  안 비우는 비대칭"을 NEW_DEFECT로 잡았다 — 정책은 **원자적**이어야 한다:
        //  성공 복원이면 둘 다 비우고, 실패면 둘 다 보존한다. 실패는 여기 도달하지 않는다.)
        // 용량은 재사용하고 길이만 비운다.
        self.contact_events.clear();
        self.ccd_hits.clear();
        self.resource_limit = false;
        self.invalidate_bp();
    }

    /// 캐노니컬 스냅샷 (스텝 경계에서만 — 설계 §3). 트리는 제외(파생 상태).
    /// 캐노니컬 스냅샷 — **총량 정책 초과 시 명시적으로 실패한다**. M3 바인딩은 이 경로를 쓴다.
    /// 이 빌드가 읽고 쓰는 스냅샷 포맷 버전. 호스트가 저장된 데이터의 호환성을
    /// **미리** 판정할 수 있도록 노출한다(복원 실패로 알게 되는 것보다 낫다).
    pub fn snapshot_version() -> u32 {
        crate::snapshot::VERSION
    }

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
        // 복원도 산술로 상태를 만든다(정규화·검증) — 비표준 환경에서 확정되면 안 된다.
        // 이 함수의 오류형에는 FP 항목이 없으므로 "내용 불일치"로 보고한다.
        if !maru_math::fp_verify() {
            return Err(crate::snapshot::SnapshotError::FpEnvironment);
        }
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

    /// 브로드페이즈 파생 상태 전체 무효화 — 호스트/구조 변이의 공통 출구.
    /// `bp_valid`(무이동 페어 캐시)와 `bp`(증분용 아이템·트리) 둘 다 버린다:
    /// 아이템 집합·형상·자세·속도가 엔진 밖에서 바뀌면 "R = integrate_gate" 논증이 깨진다.
    fn invalidate_bp(&mut self) {
        self.bp_valid = false;
        self.bp = None;
    }

    /// 테스트 전용 노브: 증분 브로드페이즈 비활성(항상 전체 재빌드).
    /// 증분 궤적 == 재빌드 궤적(스냅샷 바이트) 등가 테스트의 대조군을 만든다.
    #[doc(hidden)]
    pub fn set_force_full_broadphase(&mut self, v: bool) {
        self.bp_force_full = v;
        self.invalidate_bp();
    }

    /// 아일랜드 단위 wake (설계 §4.6). 직전 스텝의 아일랜드 대표가 같은 동적 바디를 모두 깨운다.
    /// wake 폐쇄가 바꾼 캐노니컬 sleep 상태를 기준선으로 되돌린다.
    ///
    /// 스텝을 자원 오류로 포기하는 모든 경로가 이것을 부른다 — 그러지 않으면 "물리 스텝은
    /// 건너뜀"이라 말하면서 sleep 상태만 앞으로 나가고, 그건 스냅샷에 실리는 캐노니컬
    /// 상태라 실패 전과 다른 미래를 만든다(확인검수 rev_d5b32e22 BLOCKER).
    ///
    /// 길이가 다르면(= 이 스텝에 기준선을 못 잡았다) 아무것도 하지 않는다 — wake 이전에
    /// 끝난 경로이므로 되돌릴 것도 없다. **할당하지 않는다**(자원이 빠듯한 순간에 불린다).
    fn rollback_wake(&mut self) {
        if self.sleep_bak.len() != self.asleep.len()
            || self.sleep_timer_bak.len() != self.sleep_timer.len()
        {
            return;
        }
        self.asleep.copy_from_slice(&self.sleep_bak);
        self.sleep_timer.copy_from_slice(&self.sleep_timer_bak);
    }

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
    /// **캐노니컬 상태를 만드는 모든 진입점의 FP 가드** (설계 §8).
    ///
    /// 스텝만 검증하면 부족하다: 월드 생성 뒤 다른 라이브러리가 플러시 모드를 켜면
    /// `set_pose`의 쿼터니언 정규화, 질량·주축 베이킹, 복원 검증이 **비표준 환경에서
    /// 확정된 값**을 캐노니컬 상태에 남긴다. 다음 스텝이 오류를 내도 이미 늦다
    /// (검수 rev_1e81559e MAJOR). 그래서 산술로 상태를 만드는 진입점마다 앞세운다.
    /// 비용은 레지스터 읽기 1회다.
    #[inline]
    fn fp_guard() -> Result<(), ApiError> {
        if maru_math::fp_verify() {
            Ok(())
        } else {
            Err(ApiError::FpEnvironment)
        }
    }

    /// 이 스레드의 FP 환경을 엔진 캐노니컬 상태로 되돌린다 (설계 §8).
    /// `ApiError::FpEnvironment`를 받은 호스트가 재시도 전에 부른다.
    pub fn canonicalize_fp_environment() -> bool {
        maru_math::fp_canonicalize()
    }

    pub fn try_step(&mut self, dt: f32) -> Result<(), ApiError> {
        // 문서와 구현을 하나로(리뷰 MAJOR): 0과 음수는 **합법이 아니다**. 일시정지는
        // 호스트가 step을 부르지 않는 것으로 표현한다 — 0을 무연산으로 받아주면
        // "성공했는데 아무 일도 없음"이 오류와 구분되지 않는다.
        if !(dt.is_finite() && (MIN_DT..=MAX_DT).contains(&dt)) {
            return Err(ApiError::InvalidValue);
        }
        // **FP 환경 검증은 모든 변이보다 앞이다** (설계 §8). 플러시 모드가 켜진 스레드에서
        // 돌면 같은 입력이 다른 물리를 내므로, 조용히 진행하는 대신 **상태를 건드리지 않고**
        // 거부한다. 이 검사는 레지스터 읽기 하나라 핫패스 비용이 사실상 없다.
        if !maru_math::fp_verify() {
            return Err(ApiError::FpEnvironment);
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

    /// 이번 스텝에 발생한 접촉 이벤트 (시작 → 종료, 각각 페어 키 오름차순).
    ///
    /// 다음 `step`이 비우므로 스텝 사이에 읽어야 한다.
    pub fn contact_events(&self) -> &[ContactEvent] {
        &self.contact_events
    }

    /// 이번 스텝의 CCD 절단 히트. 스텝 시작에 비워지므로 **스텝 사이에만 유효**하다
    /// (접촉 이벤트와 같은 수명). `Diagnostics::ccd_hits_dropped`가 0이 아니면
    /// 상한을 넘겨 잘린 목록이다.
    pub fn ccd_hits(&self) -> &[CcdHit] {
        &self.ccd_hits
    }

    fn step_inner(&mut self, dt: f32) {
        // 이벤트는 **스텝 단위 출력**이다 — 매 스텝 비운다.
        self.contact_events.clear();
        self.ccd_hits.clear();
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
            // 스텝 진입에서 이미 검증했다 — 여기서 다시 보면 같은 값을 두 번 읽을 뿐이다.
            self.flush_commands_inner();
        }
        // 속도 클램프는 **flush 뒤·브로드페이즈 앞**이어야 한다(리뷰 BLOCKER): 스텝 시작에만
        // 걸면 pending `SetLinearVelocity`가 그 뒤에 적용돼 sweep AABB를 부풀린 채
        // 브로드페이즈로 들어간다(O(N²) 페어 → 오류 반환 전 OOM).
        self.clamp_velocities();
        if self.bodies.is_empty() {
            return;
        }
        // 브로드페이즈가 새 페어 배열을 잡기 **전에** 과거 장면이 남긴 부풀린 스크래치를
        // 놓아준다(위 함수 주석 — 그 할당은 fallible이 아니라 검사 전에 죽을 수 있다).
        self.release_stale_scratch(self.bodies.len());
        let n_sub = self.params.substeps.max(1);
        let h = dt / n_sub as f32;
        let inv_h = 1.0 / h;
        // contact hertz는 서브스텝 주파수의 1/4을 넘지 않게 (Box2D v3 관행)
        let hertz = self.params.solver.contact_hertz.min(0.25 * inv_h);
        let soft = Softness::new(hertz, self.params.solver.contact_damping, h);
        // 정적/키네마틱 상대 접촉은 2배 강성 (Box2D v3 staticSoftness) — 무한 질량 상대라
        // 더 단단히 잡아도 안정하고, 지면 침하가 스택 전체 침하의 시작점이 된다.
        let static_soft = Softness::new(2.0 * hertz, self.params.solver.contact_damping, h);
        // SIMD solve 배치 파라미터 (Unit 3c) — bias/relax 두 벌. 색 배치 디스패치가 per-lane으로
        // dyn-dyn/dyn-static softness를 고른다. 레인당 스칼라와 비트 동일이라 L0 해시 불변.
        let bias_params = solver::BatchSolveParams {
            inv_h,
            relax: false,
            soft,
            static_soft,
            max_bias: self.params.solver.max_bias_speed,
        };
        let relax_params = solver::BatchSolveParams {
            inv_h,
            relax: true,
            soft,
            static_soft,
            max_bias: self.params.solver.max_bias_speed,
        };

        // 1) 브로드페이즈 — canonical 페어 (트리 비의존 계약은 maru-collide가 보장)
        let _t_aabb = Stopwatch::start();
        let mut dyn_items: Vec<(u32, Aabb)> = Vec::new();
        let mut static_items: Vec<(u32, Aabb)> = Vec::new();
        // **캐시를 쓸 스텝이면 AABB를 아예 만들지 않는다.** 페어 캐시가 유효하면 아래에서
        // `Broadphase::build(dyn_items, static_items)`를 부르지 않으므로 이 목록은 만들어서
        // 버려진다 — 전원 수면 pile-3k에서 스텝의 12%가 그 낭비였다(93us/step).
        // 판정식은 아래 캐시 분기와 **같은 것**이어야 한다(다르면 빈 목록으로 트리를 짓는다).
        let bp_cache_hit = self.bp_valid && self.bp_dt == dt;
        // **전원-휴면 fast path** — 정착한 월드의 스텝은 캐노니컬 상태에 항등이다.
        //
        // 근거 사슬(전부 위·아래 코드의 계약이다):
        // `bp_valid`는 직전 스텝 말 `moved=false`에서만 참이고, moved는 integrate_gate
        // (alive && !static && !asleep && !oob)의 any다 — 즉 캐시가 유효하면 직전 스텝 말에
        // 활성 바디가 없었다. 스텝 사이 상태를 바꾸는 모든 경로는 wake(asleep=false)나
        // `bp_valid=false`를 거친다(`mutate`·`accept_force`·add/remove·set_params·restore).
        // 그래서 지금도 전원이 dormant면 이 스텝의 전 파이프라인이 항등이다:
        // wake 폐쇄는 활성 끝점이 없어 아무도 깨우지 못하고, 제약·이벤트·PCM 갱신은 비며
        // (live_keys == touching — 그룹·제외 입력도 마지막 계산 이후 불변), 아일랜드·수면
        // 타이머는 잠든 값을 보존하고, 위생 패스는 dormant를 건너뛴다. 전원 수면 pile-3k에서
        // 이 스텝들이 스텝의 전부(정착 후 100%)였다 — 항등을 계산으로 재확인하는 비용이
        // 스텝당 ~0.7ms였다.
        //
        // 판정은 dormant의 정의를 바디 배열로 직접 편다(kinematic은 gate=항상 true라
        // 캐시가 애초에 못 살아남으므로 여기 도달하지 않지만, 정의로 다시 거른다).
        if bp_cache_hit
            && (0..self.bodies.len()).all(|i| {
                !self.alive[i]
                    || self.oob[i]
                    || self.asleep[i]
                    || self.bodies[i].body_type == BodyType::Static
            })
        {
            self.diag.bp_reused = true;
            self.diag.pairs = self.bp_pairs.len();
            // 예산은 이번 스텝에 계산하지 않았다 — 절대 상한을 보고한다("예산이 묶지 않았다"는
            // 사실만 전달하면 된다. 0을 보고하면 호스트가 거부로 오독한다).
            self.diag.pair_budget = maru_collide::MAX_PAIRS;
            self.diag.touching = self.touching.len();
            return;
        }
        // **증분 브로드페이즈 판정.** 구조(아이템 집합·형상·자세의 호스트 변이)가 유효하고
        // 직전 스텝의 integrate_gate가 현재 슬롯 수와 맞으면, "직전 스텝에 적분된 바디"가
        // 곧 AABB가 변했을 수 있는 유일한 집합 R이다 — 엔진의 모든 자세·속도 쓰기는
        // gate-true 바디에만 가고(비적분 바디의 write_back은 항등), 호스트 쓰기는 전부
        // `invalidate_bp`를 거친다. R만 갱신하고 페어를 델타로 재생성하면 결과는 §4.1에 의해
        // 전체 재빌드와 비트 동일하다(maru-collide 속성 테스트 + 아래 시뮬 등가 테스트가 잠금).
        // R이 슬롯의 절반을 넘으면 재빌드가 더 싸다.
        let can_incr = !bp_cache_hit
            && !self.bp_force_full
            && self.bp.is_some()
            && self.integrate_gate.len() == self.bodies.len();
        let moved_ids: Vec<u32> = if can_incr {
            (0..self.bodies.len())
                .filter(|&i| self.integrate_gate[i])
                .map(|i| i as u32)
                .collect()
        } else {
            Vec::new()
        };
        let incremental = can_incr && moved_ids.len() <= self.bodies.len() / 2;
        for (i, b) in self
            .bodies
            .iter()
            .enumerate()
            .take(if bp_cache_hit || incremental {
                0
            } else {
                usize::MAX
            })
        {
            if !self.alive[i] {
                continue; // 제거된 슬롯
            }
            // 스텝과 질의가 **같은 AABB 정의**를 쓴다 (헬퍼가 정본).
            let world = self.body_aabb(i);
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
        // 증분 경로의 갱신 목록 — 위 전체 루프와 **같은 AABB 정의**(body_aabb + sweep).
        let mut bp_updates: Vec<(u32, Aabb)> = Vec::new();
        if incremental {
            for &r in &moved_ids {
                let i = r as usize;
                // R 원소는 직전 스텝에 적분됐고(alive·비OOB·비static), 그 뒤의 구조 변경은
                // 전부 invalidate_bp를 거쳐 여기 못 온다.
                debug_assert!(
                    self.alive[i] && !self.oob[i] && self.bodies[i].body_type != BodyType::Static,
                    "R 원소가 브로드페이즈 대상이 아니다 — invalidate_bp 누락"
                );
                let world = self.body_aabb(i);
                let vel_dt = self.bodies[i].linvel * dt;
                bp_updates.push((
                    r,
                    world.expand_sweep(vel_dt, self.params.speculative_margin),
                ));
            }
        }
        self.diag.phase_ns[0] = _t_aabb.ns();
        let _t_bp = Stopwatch::start();
        // 예산은 **살아있는 바디 수** 기준(슬롯 수가 아니라) + 절대 개수 상한 + **바이트 상한**.
        // 개수만 묶으면 페어 뒤에 따라붙는 constraint·키·HashSet·SoA가 남는다 —
        // 한 스텝이 잡을 수 있는 총 바이트를 정해 그것으로 페어 상한을 되돌려 계산한다.
        let live = self.alive.iter().filter(|a| **a).count();
        // per-pair 스크래치. **매니폴드 접점 힙까지** 센다 — `ContactConstraint.points`는 페어당
        // 최대 MAX_MANIFOLD_POINTS 원소를 따로 할당하므로 Vec 헤더만 세면 총상한이 깨진다
        // (교차검수 rev_2ade4bf3 MAJOR).
        // **페어 비례**와 **제약 비례**를 나눠 센다. 둘은 같은 수가 아니다 — 복합 바디는
        // 자식쌍마다 독립 제약을 만들어 페어 하나가 최대 (자식 수)² 개의 제약이 된다
        // (교차검수 rev_518b072b BLOCKER: 페어당 1개로 세면 8자식×8자식 페어에서 회계가
        // 64배 틀린다). 아래에서 실제 페어 목록으로 **정확한** 제약 상한을 구해 다시 잰다.
        let per_pair_bytes = core::mem::size_of::<(u32, u32)>()          // pairs
            + core::mem::size_of::<(u32, u32)>()                         // NarrowOut.live
            + core::mem::size_of::<(u32, u32)>()                         // started (≤ 이번 스텝 페어 수)
            + core::mem::size_of::<ContactEvent>()                       // started 쪽 출력
            + 48; // live_keys(HashSet) + PCM 엔트리 여유
        let per_constraint_bytes = core::mem::size_of::<ContactConstraint>() // constraints
            + maru_collide::MAX_MANIFOLD_POINTS
                * core::mem::size_of::<solver::ContactPointConstraint>() // 접점 힙 원소
            + core::mem::size_of::<(u32, u32)>()                         // constraint_keys
            + core::mem::size_of::<u32>()                                // 컬러 order
            + core::mem::size_of::<(u32, u32)>()                         // 컬러 endpoints
            + core::mem::size_of::<u32>()                                // color_of
            + core::mem::size_of::<u32>()                                // grouped
            + core::mem::size_of::<Option<ContactConstraint>>()          // c_tmp (거의 한 벌 더)
            + core::mem::size_of::<(u32, u32)>()                         // new_keys
            // SoA 솔버 블록 3종. **개수 상한을 제약 수로 되돌려 계산한다**:
            // 블록·block_starts는 제약 4개를 소비하므로 ≤ ⌊n/4⌋, `work` 항목은 하나가
            // 최소 제약 1개를 소비하므로 ≤ n (색마다 4청크 + 꼬리 1개인데, 꼬리도 제약을
            // 최소 1개 쓴다). 나눗셈은 **올림**이다 — 내림으로 깎으면 상한이 상한이 아니다.
            + core::mem::size_of::<solver_block::ContactBlock>().div_ceil(4)
            + core::mem::size_of::<usize>().div_ceil(4)                  // block_starts
            + core::mem::size_of::<solver_block::Work>()                 // work
            ;
        // 내로우페이즈 참여자 버퍼는 **두 번 센다** — 목적이 다르다.
        // ① 아래 페어 예산(byte_cap)에는 "논리 한 벌"을 얹는다: 예산이 이 변경 전보다
        //    관대해지면 안 된다(관대해진 순간 바이트 상한을 재던 스트레스 테스트가
        //    32비트 wasm에서 메모리 한계에 닿았다 — 예산은 조이는 쪽으로만 바꾼다).
        // ② 페어를 받은 뒤의 총상한 검사에는 **실제 보유 용량**을 쓴다(rev_071f1cd9 MAJOR).
        // 접점 힙(`ContactConstraint.points`)은 병합 때 **이동**하므로 중복 계상하지 않는다.
        let per_constraint_budget_bytes = per_constraint_bytes
            + core::mem::size_of::<ContactConstraint>()  // NarrowOut.constraints
            + core::mem::size_of::<(u32, u32)>(); // NarrowOut.keys
                                                  // per-body(슬롯당) 스크래치 전부: SolverState 7배열 + inv_inertia_local + ext_force +
                                                  // damping + 게이트 2 + island_parent + 컬러 body_color/conflict. 총상한에서 먼저 뺀다.
        let per_body_bytes = 3 * core::mem::size_of::<Vec3>()            // lin/ang/pos
            + core::mem::size_of::<maru_math::Quat>()                    // rot
            + core::mem::size_of::<f32>()                                // inv_m
            + core::mem::size_of::<maru_math::Mat3>()                    // inv_i
            + core::mem::size_of::<bool>()                               // dynamic
            + core::mem::size_of::<Vec3>()                               // inv_inertia_local
            + 2 * core::mem::size_of::<Vec3>()                           // ext_force(force,torque)
            + 2 * core::mem::size_of::<f32>()                            // damping(lin,ang)
            + 2 * core::mem::size_of::<bool>()                           // integrate/gravity gates
            + core::mem::size_of::<usize>()                              // island_parent
            + core::mem::size_of::<u64>()
            + core::mem::size_of::<bool>()                                // 컬러 body_color/conflict
            + core::mem::size_of::<bool>()                                // sleep_bak
            + core::mem::size_of::<f32>(); // sleep_timer_bak
        let fixed_body_bytes = self.bodies.len().saturating_mul(per_body_bytes);
        // 조인트 비례 스크래치: joint_pairs·active_joints·excluded. **조인트 수는 pairs·bodies의
        // 상한이 아니므로** 별도로 차감해야 총상한이 성립한다 (교차검수 rev_ba3ca887 MAJOR).
        let per_joint_bytes = core::mem::size_of::<(usize, usize)>()     // joint_pairs
            + core::mem::size_of::<usize>()                             // active_joints
            + 2 * core::mem::size_of::<(u32, u32)>(); // excluded(HashSet 오버헤드 여유)
        let fixed_joint_bytes = self.joints.len().saturating_mul(per_joint_bytes);
        // **고정 스크래치(body+joint)만으로 상한을 넘으면 페어 예약도 시작 전에 종료**한다
        // (교차검수 rev_b86f5aea MAJOR). 페어를 0으로 막아도 아래 통합 블록이 body/joint 버퍼를
        // nbody/njoint 크기로 그대로 예약하므로, 여기서 막지 않으면 소수 바디+대량 조인트가
        // 128MiB를 넘겨 "총 스크래치 ≤ MAX" 계약을 깬다. 이 시점엔 pairs를 아직 안 건드렸다.
        // **CCD 히트 버퍼도 고정 스크래치다.** 상한이 고정(`MAX_CCD_HITS`)이라 페어 비례가
        // 아니고, CCD 바디가 있으면 반드시 선점되므로 페어 예산을 역산하기 **전에** 빼야
        // 한다. 안 빼면 페어가 `byte_cap`까지 찬 월드가 128 MiB를 거의 다 쓴 뒤 이 버퍼를
        // 더 잡아 총상한을 넘긴다 (교차검수 rev_5e74f64f MAJOR).
        // 커맨드는 이미 flush됐으므로 이번 스텝에 CCD 기록이 가능한 바디 집합은 확정돼 있다.
        let ccd_any = self
            .bodies
            .iter()
            .enumerate()
            .any(|(i, b)| self.alive[i] && b.ccd);
        let fixed_ccd_bytes = if ccd_any {
            MAX_CCD_HITS.saturating_mul(core::mem::size_of::<CcdHit>())
        } else {
            0
        };
        // **직전 기준선 비례 스크래치는 페어 비례가 아니다** (교차검수 rev_18633fde BLOCKER).
        // `touch_sorted`·`stopped`·stopped 쪽 출력의 상한은 **직전 스텝**의 `touching` 크기이고,
        // 그것은 이번 스텝의 페어 수와 무관하다. 실제로 어긋난다: 직전 스텝에 N개가 닿은 뒤
        // 전부 멀리 텔레포트하면 `pairs.len() == 0`인데 `touching.len() == N`이라,
        // 페어 비례로만 세면 회계가 0바이트를 보고하면서 N개를 선점한다 — 128 MiB 계약이 깨진다.
        // 그래서 고정 항으로 뺀다(`touch_sorted`는 이 변경 **이전부터** 같은 구멍이었고,
        // 이벤트 버퍼 둘을 같은 상한으로 추가하면서 폭이 넓어졌다).
        let baseline_event_bytes = self.touching.len().saturating_mul(
            2usize
                .saturating_mul(core::mem::size_of::<(u32, u32)>()) // touch_sorted + stopped
                .saturating_add(core::mem::size_of::<ContactEvent>()), // stopped 쪽 출력
        );
        let fixed_bytes = fixed_body_bytes
            .saturating_add(fixed_joint_bytes)
            .saturating_add(fixed_ccd_bytes)
            .saturating_add(baseline_event_bytes);
        if fixed_bytes >= MAX_STEP_SCRATCH_BYTES {
            self.resource_limit = true;
            return;
        }
        // 이 예산은 **페어 개수**의 상한이다 — 제약 수는 페어 목록이 나와야 알 수 있으므로
        // 여기서는 "페어당 제약 1개"로 역산하고, 실제 제약 상한 검사는 페어를 받은 뒤
        // (wake 폐쇄 전에) 아래에서 한다.
        let byte_cap = (MAX_STEP_SCRATCH_BYTES - fixed_bytes)
            / (per_pair_bytes + per_constraint_budget_bytes).max(1);
        let budget = (self.params.max_pairs_per_body as usize)
            .saturating_mul(live)
            .saturating_add(1024)
            .min(maru_collide::MAX_PAIRS)
            .min(byte_cap);
        self.diag.pair_budget = budget;
        // **아무것도 움직이지 않았으면 트리를 다시 짓지 않는다.** 페어 집합은 트리 비의존
        // 계약(설계 §4.1)이라 입력이 같으면 결과가 비트 동일하다 — 재사용은 항등이다.
        // 증분 스텝의 R 접촉 페어(fresh) — 아래 wake/narrow 순회 제한의 근거로 살려 둔다.
        let mut fresh_pairs: Option<Vec<(u32, u32)>> = None;
        let mut moved_mask: Vec<bool> = Vec::new();
        let pairs = if bp_cache_hit {
            self.diag.bp_reused = true;
            std::mem::take(&mut self.bp_pairs)
        } else if incremental {
            // 증분: R의 AABB만 교체 + 트리 리핏(위상 유지) + 페어 델타 병합.
            // 결과는 전체 재빌드와 비트 동일하다 — 등가 논증은 incremental_pairs_bounded 문서에.
            self.diag.bp_incremental = true;
            let bp = self.bp.as_mut().expect("incremental ⇒ bp 존재");
            if !bp.refresh_dynamic(&bp_updates) {
                // 슬롯 캐시 할당 실패(첫 증분 스텝) — 구조를 버려 재빌드 폴백으로.
                self.invalidate_bp();
                self.resource_limit = true;
                return;
            }
            moved_mask = vec![false; self.bodies.len()];
            for &r in &moved_ids {
                moved_mask[r as usize] = true;
            }
            let prev = std::mem::take(&mut self.bp_pairs);
            // 병합 출력은 영속 스크래치 — 직전 prev 버퍼와 핑퐁해 워밍업 후 재할당이 없다.
            let mut out = std::mem::take(&mut self.bp_pairs_scratch);
            match bp.incremental_pairs_bounded(&moved_ids, &moved_mask, &prev, budget, &mut out) {
                Some(fresh) => {
                    fresh_pairs = Some(fresh);
                    self.bp_pairs_scratch = prev;
                    out
                }
                None => {
                    // 아이템은 이미 갱신됐는데 페어는 옛것 — 구조를 통째로 버려 다음 스텝이
                    // 재빌드하게 한다(스텝은 상태 불변으로 종료: 여기는 wake 폐쇄 이전이다).
                    self.bp_pairs = prev;
                    self.bp_pairs_scratch = out;
                    self.invalidate_bp();
                    self.resource_limit = true;
                    return;
                }
            }
        } else {
            let bp = Broadphase::build(dyn_items, static_items);
            let Some(p) = bp.canonical_pairs_bounded(budget) else {
                self.invalidate_bp();
                self.resource_limit = true;
                return;
            };
            // 다음 스텝의 증분 갱신을 위해 구조를 보존한다.
            self.bp = Some(bp);
            p
        };
        self.diag.pairs = pairs.len();
        // === 스텝 스크래치 전량 선점 (교차검수 rev_2ade4bf3 MAJOR) ===
        // wake 폐쇄(아래 1.5)가 canonical sleep 상태(asleep·sleep_timer)를 바꾸기 전에,
        // known-bound 스크래치를 **전부** fallible 예약한다. wake 이후 예약이 실패하면 깨워둔 채
        // ResourceLimit을 돌려줘 "Err면 상태 불변" 계약이 깨진다(직전 검수가 컬러링 버퍼만 옮긴 걸
        // 지적). 여기서 용량만 잡고(실패=wake 전 종료), 내용은 원래 자리에서 채운다(무재할당).
        // 상한: 페어당=pairs.len(), 바디당=bodies.len(), 조인트당=joints.len(). 전부 위 byte 예산에 포함.
        let ncap = pairs.len();
        // **제약 상한을 실제 페어 목록에서 정확히 구한다.** 복합 바디는 자식쌍마다 독립
        // 제약을 만들므로(narrow의 `for_each_contact` 디스패치) 페어 하나의 상한은
        // parts(a)×parts(b)이고, 비복합끼리면 1이라 기존 장면의 회계는 변하지 않는다.
        // 이 합이 곧 constraints·키·컬러링·블록 버퍼의 상한이다.
        let parts = |id: u32| -> usize {
            match &self.bodies[(id & !STATIC_BIT) as usize].shape {
                maru_collide::Shape::Compound { children } => children.len(),
                // 하이트필드는 겹치는 셀마다 매니폴드를 낸다 — 상한이 곧 페어당 제약 상한이다
                // (그 상한을 내로우페이즈가 실제로 강제하고 포화를 진단으로 보고한다).
                // 메시도 겹치는 삼각형마다 매니폴드를 내고 **같은 상한**에서 끊는다
                // (`trimesh_contacts`가 `MAX_HEIGHTFIELD_MANIFOLDS`를 그대로 쓴다).
                // 이 회계를 빠뜨리면 제약 버퍼가 모자라 스텝이 자원 오류로 떨어진다.
                maru_collide::Shape::Heightfield { .. } | maru_collide::Shape::TriMesh { .. } => {
                    maru_collide::MAX_HEIGHTFIELD_MANIFOLDS
                }
                _ => 1,
            }
        };
        let mut constraint_cap = 0usize;
        for &(a, b) in pairs.iter() {
            constraint_cap = constraint_cap.saturating_add(parts(a).saturating_mul(parts(b)));
        }
        // 내로우페이즈 참여자 수와 참여자별 버퍼 용량은 **wake 폐쇄 전에** 확정해야 한다
        // (아래 통합 fallible 블록의 존재 이유와 같다: wake 뒤의 예약 실패는 깨워둔 채
        // ResourceLimit을 돌려줘 "Err면 상태 불변" 계약을 깬다).
        let nbody = self.bodies.len();
        let n_par = narrow_participants(&self.pool, ncap);
        // 참여자별 상한은 **자기 청크의 정확한 제약 합**이다(복합 페어는 1개가 아니다).
        // 이 스텝이 전체 목록을 걸을지 fresh만 걸을지는 wake 뒤에 정해지므로, 두 목록의
        // 같은 번호 청크 중 **큰 쪽**을 잡는다 — 어느 쪽으로 가도 모자라지 않는다.
        let mut chunk_caps = [(0usize, 0usize); crate::threads::MAX_THREADS];
        for (w, slot) in chunk_caps.iter_mut().enumerate().take(n_par) {
            let mut worst = (0usize, 0usize);
            for list in [Some(&pairs), fresh_pairs.as_ref()].into_iter().flatten() {
                let n = list.len();
                let (lo, hi) = (n * w / n_par, n * (w + 1) / n_par);
                let mut c = 0usize;
                for &(a, b) in &list[lo..hi] {
                    c = c.saturating_add(parts(a).saturating_mul(parts(b)));
                }
                worst = (worst.0.max(c), worst.1.max(hi - lo));
            }
            *slot = worst;
        }
        // 바이트 총상한 재검사 — 페어는 예산 안이어도 **제약**이 넘칠 수 있다.
        // 여기는 wake 폐쇄 **이전**이라 실패해도 상태는 그대로다("Err면 상태 불변").
        //
        // 참여자 버퍼는 **논리적 상한이 아니라 실제로 들고 있게 될 용량**으로 센다
        // (확인검수 rev_071f1cd9 MAJOR): 버퍼는 영속이라 줄지 않으므로, 1참여자로 큰
        // 장면을 한 번 돌린 뒤 8참여자로 바꾸면 0번 버퍼가 **전체 한 벌**을 쥔 채 나머지
        // 7개가 청크만큼 더 잡는다. 청크 합 한 벌만 세면 그 초과분이 회계 밖으로 샌다.
        let narrow_held: usize = self
            .narrow_bufs
            .iter()
            .enumerate()
            .map(|(w, b)| {
                let (c_cap, p_cap) = chunk_caps.get(w).copied().unwrap_or((0, 0));
                b.projected_bytes(
                    if w < n_par { c_cap } else { 0 },
                    if w < n_par { p_cap } else { 0 },
                )
            })
            .sum();
        // 아직 안 만들어진 참여자 버퍼(첫 병렬 스텝)도 예약될 것이므로 함께 센다.
        let narrow_new: usize = chunk_caps
            .iter()
            .take(n_par)
            .skip(self.narrow_bufs.len())
            .map(|&(c, p)| NarrowOut::bytes_for(c, p))
            .sum();
        // **페어 배열은 한 벌이 아니다.** 논리식은 `ncap` 한 벌만 세는데, 증분 경로는
        // 지역 `pairs`(=병합 출력)와 `bp_pairs_scratch`(=직전 목록)를 동시에 들고 있고,
        // 전체 재빌드 경로는 거기에 옛 `bp_pairs`까지 남는다 — 최대 세 벌이다
        // (확인검수 rev_de34bbb5 BLOCKER). 지역 배열의 **초과분**과 self가 쥔 두 벌을
        // 실제 용량으로 센다(지역의 첫 ncap은 논리식이 이미 셌다).
        let pair_bytes = core::mem::size_of::<(u32, u32)>();
        let pair_held_extra = pairs
            .capacity()
            .saturating_sub(ncap)
            .saturating_mul(pair_bytes)
            .saturating_add(self.bp_pairs.capacity().saturating_mul(pair_bytes))
            .saturating_add(self.bp_pairs_scratch.capacity().saturating_mul(pair_bytes));
        let contact_event_target = ncap.saturating_add(self.touching.len());
        let step_bytes = |this: &Self, narrow_held: usize| {
            fixed_bytes
                .saturating_add(ncap.saturating_mul(per_pair_bytes))
                .saturating_add(constraint_cap.saturating_mul(per_constraint_bytes))
                .saturating_add(narrow_held)
                .saturating_add(narrow_new)
                .saturating_add(pair_held_extra)
                .saturating_add(this.persistent_excess_bytes(
                    nbody,
                    constraint_cap,
                    contact_event_target,
                ))
        };
        if step_bytes(self, narrow_held) > MAX_STEP_SCRATCH_BYTES {
            // 먼저 **반납**하고 다시 잰다(위 `release_oversized_scratch` 주석 — 거절만 하면
            // 줄어든 월드가 영원히 못 돈다). 여기는 wake 폐쇄 이전이고 용량만 건드리므로
            // 캐노니컬 상태는 그대로다.
            self.release_oversized_scratch(fresh_pairs.is_some());
            let narrow_held: usize = self
                .narrow_bufs
                .iter()
                .enumerate()
                .map(|(w, b)| {
                    let (c_cap, p_cap) = chunk_caps.get(w).copied().unwrap_or((0, 0));
                    let (c, p) = if w < n_par { (c_cap, p_cap) } else { (0, 0) };
                    b.projected_bytes(c, p)
                })
                .sum();
            // 반납 뒤에는 self 쪽 페어 두 벌도 사라졌다 — 지역 배열의 초과분만 남는다.
            let pair_held_extra = pairs.capacity().saturating_sub(ncap) * pair_bytes;
            let after = fixed_bytes
                .saturating_add(ncap.saturating_mul(per_pair_bytes))
                .saturating_add(constraint_cap.saturating_mul(per_constraint_bytes))
                .saturating_add(narrow_held)
                .saturating_add(narrow_new)
                .saturating_add(pair_held_extra)
                .saturating_add(self.persistent_excess_bytes(
                    nbody,
                    constraint_cap,
                    contact_event_target,
                ));
            if after > MAX_STEP_SCRATCH_BYTES {
                // 참여자 버퍼는 아직 self가 들고 있다(아래에서 take한다) — 되돌릴 것이 없다.
                self.bp_pairs = pairs;
                self.resource_limit = true;
                return;
            }
        }
        let njoint = self.joints.len();
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut constraint_keys: Vec<(u32, u32)> = Vec::new();
        let mut live_keys: IntSet<(u32, u32)> = IntSet::default();
        let mut joint_pairs: Vec<(usize, usize)> = Vec::new();
        let mut active_joints: Vec<usize> = Vec::new();
        let mut excluded: IntSet<(u32, u32)> = IntSet::default();
        let mut order: Vec<u32> = Vec::new();
        let mut conflict: Vec<bool> = Vec::new();
        let mut endpoints: Vec<(u32, u32)> = Vec::new();
        let mut body_color: Vec<u64> = Vec::new();
        let mut color_of: Vec<u32> = Vec::new();
        let mut grouped: Vec<u32> = Vec::new();
        let mut c_tmp: Vec<Option<ContactConstraint>> = Vec::new();
        let mut new_keys: Vec<(u32, u32)> = Vec::new();
        // 접촉 이벤트 델타 버퍼. **스텝 시작의 통합 fallible 블록에서 선점한다** — 예전에는
        // 이벤트 구간에서 `collect()`로 잡았고(무한 실패 불가), 그 지점은 wake·솔브가 이미
        // 끝난 뒤라 할당이 실패하면 "Err면 상태 불변"도 "스텝 중 할당 0"도 성립하지 않았다
        // (README '알려진 한계', 교차검수 rev_51f69fbd). 상한은 wake 이전에 확정된다:
        // started ≤ 이번 스텝 페어 수, stopped ≤ 직전 기준선 크기.
        let mut started: Vec<(u32, u32)> = Vec::new();
        let mut stopped: Vec<(u32, u32)> = Vec::new();
        // 직전 접촉 기준선을 **정렬된 벡터**로 — dormant 페어 보존을 페어 순회와 병합하려고 둔다
        // (아래 narrow 루프 주석). 크기는 직전 스텝 접촉 수이므로 wake 이전에 확정된다.
        let mut touch_sorted: Vec<(u32, u32)> = Vec::new();
        // 색 범위 [start,end) — Unit 3b SIMD warm_start 배치 디스패치용. 색은 최대 64개.
        let mut color_ranges: Vec<(u32, u32)> = Vec::new();
        // SoA 솔버 블록과 그 잔여 스칼라 구간 — 스텝당 1회 조립(설계 §3, solver_block 모듈 주석).
        // 블록은 제약 4개당 1개라 상한은 ncap/4 + 색 수(색마다 꼬리가 생길 수 있다).
        let mut block_starts: Vec<usize> = Vec::new();
        let mut work: Vec<solver_block::Work> = Vec::new();
        // 색별 work 구간 — 병렬 솔브의 배리어 경계. 색은 최대 64개라 고정 상한이다.
        let mut color_work: Vec<(u32, u32)> = Vec::new();
        let _t_res = Stopwatch::start();
        let mut narrow_bufs = std::mem::take(&mut self.narrow_bufs);
        if (ccd_any
            && self
                .ccd_hits
                .try_reserve(MAX_CCD_HITS.saturating_sub(self.ccd_hits.capacity()))
                .is_err())
            // **참여자가 1명이면 이 지역 버퍼는 예약하지 않는다.** 병합이 없으므로 아래에서
            // 참여자 버퍼와 **스왑**만 하고, 스왑 뒤 이 지역 변수가 그 용량을 물려받는다.
            // 예약하면 같은 크기 배열을 두 벌 쥐게 되어 스텝 최대 메모리가 두 배가 된다
            // (32비트 wasm 테스트 인스턴스가 이것 때문에 메모리 한계에 닿았다 —
            // 선형 메모리는 줄지 않아 한 스텝의 최대치가 그대로 상한을 민다).
            || (n_par > 1 && constraints.try_reserve(constraint_cap).is_err())
            // 여유 상수(`+1`·`+64`)를 붙이지 않는다: 페어가 0이면 이 버퍼들도 0이어야
            // 총 바이트 회계(위 `per_pair_bytes`)가 실제 선점과 일치한다.
            // 예산이 0페어로 잘린 월드에서 상수만큼 더 잡으면 128 MiB 계약을 넘긴다
            // (교차검수 rev_d4f505e6 MAJOR).
            || self
                .contact_blocks
                .try_reserve((constraint_cap / 4).saturating_sub(self.contact_blocks.len()))
                .is_err()
            || block_starts.try_reserve(constraint_cap / 4).is_err()
            || work.try_reserve(constraint_cap).is_err()
            || color_ranges
                .try_reserve(solver::COLOR_OVERFLOW as usize + 1)
                .is_err()
            || color_work
                .try_reserve(solver::COLOR_OVERFLOW as usize + 1)
                .is_err()
            || (n_par > 1 && constraint_keys.try_reserve(constraint_cap).is_err())
            || live_keys.try_reserve(ncap).is_err()
            || joint_pairs.try_reserve(njoint).is_err()
            || active_joints.try_reserve(njoint).is_err()
            || excluded.try_reserve(njoint).is_err()
            || order.try_reserve(constraint_cap).is_err()
            || conflict.try_reserve(nbody).is_err()
            || endpoints.try_reserve(constraint_cap).is_err()
            || body_color.try_reserve(nbody).is_err()
            || color_of.try_reserve(constraint_cap).is_err()
            || grouped.try_reserve(constraint_cap).is_err()
            || c_tmp.try_reserve(constraint_cap).is_err()
            || new_keys.try_reserve(constraint_cap).is_err()
            || touch_sorted.try_reserve(self.touching.len()).is_err()
            // 이벤트 델타와 출력 버퍼 — 상한은 wake 이전 값으로 정한다.
            // 출력은 두 델타의 합이 상한이다(같은 키가 양쪽에 동시에 들지 않는다).
            || started.try_reserve(ncap).is_err()
            || stopped.try_reserve(self.touching.len()).is_err()
            // `try_reserve(additional)`는 **`len + additional`**을 보장한다 — 목표에서
            // `capacity`를 빼서 넘기면 용량이 목표에 못 미친다(첫 스텝에만 우연히 맞는다).
            // 이벤트 버퍼는 스텝 시작에 비워지므로 `len`은 0이고, 목표를 그대로 넘기면
            // 이미 충분할 때 무연산이다. (이 실수를 아래 `debug_assert!`가 즉시 잡았다.)
            || self
                .contact_events
                .try_reserve(
                    ncap.saturating_add(self.touching.len())
                        .saturating_sub(self.contact_events.len()),
                )
                .is_err()
            || self.solver_state.reserve_for(nbody).is_err()
            || self.inv_inertia_local
                .try_reserve(nbody.saturating_sub(self.inv_inertia_local.len()))
                .is_err()
            || self.integrate_gate
                .try_reserve(nbody.saturating_sub(self.integrate_gate.len()))
                .is_err()
            || self.gravity_gate
                .try_reserve(nbody.saturating_sub(self.gravity_gate.len()))
                .is_err()
            || self.ext_force
                .try_reserve(nbody.saturating_sub(self.ext_force.len()))
                .is_err()
            || self.damping
                .try_reserve(nbody.saturating_sub(self.damping.len()))
                .is_err()
            || self.island_parent
                .try_reserve(nbody.saturating_sub(self.island_parent.len()))
                .is_err()
            // sleep 롤백 기준선 — **wake 이전에** 용량을 잡아야 의미가 있다(예약 실패가
            // wake 뒤로 밀리면 되돌릴 수 없는 상태에서 오류를 내게 된다).
            || self.sleep_bak
                .try_reserve(nbody.saturating_sub(self.sleep_bak.len()))
                .is_err()
            || self.sleep_timer_bak
                .try_reserve(nbody.saturating_sub(self.sleep_timer_bak.len()))
                .is_err()
            || reserve_narrow_bufs(&mut narrow_bufs, n_par, &chunk_caps).is_err()
        {
            self.bp_pairs = pairs;
            self.narrow_bufs = narrow_bufs;
            self.resource_limit = true;
            return;
        }
        // === sleep 롤백 기준선 (wake 폐쇄 직전) ===
        // 용량은 위 블록이 이미 잡았다 — `clear` 뒤 `extend`는 재할당하지 않는다(스텝 중 할당 0).
        self.sleep_bak.clear();
        self.sleep_bak.extend_from_slice(&self.asleep);
        self.sleep_timer_bak.clear();
        self.sleep_timer_bak.extend_from_slice(&self.sleep_timer);
        self.diag.sub_ns[0] = _t_res.ns();
        // 조인트 연결 쌍은 충돌 제외 (collide_connected=false 기본).
        // excluded는 위 통합 블록에서 선점됨(실패 bail이 bp_pairs를 복원 — 캐시 재사용 스텝에
        // 빈 페어 집합이 유효 캐시로 남던 BLOCKER 차단, 교차검수 rev_ba3ca887). 여기선 채운다.
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
        // joint_pairs는 위에서 선점됨 — 여기선 채운다(무재할당). wake 루프 입력이라 wake 전에 채운다.
        joint_pairs.extend(
            self.joints
                .iter()
                // **OOB 끝점은 union하지 않는다**(리뷰 MAJOR): 두 정상 바디가 같은 OOB에
                // 각각 매달리면 비활성 조인트를 통해 한 아일랜드로 묶여 함께 자고 깨며,
                // 정상 바디의 대표가 OOB 인덱스가 될 수 있다.
                .filter(|j| !self.oob[j.ia()] && !self.oob[j.ib()])
                .map(|j| (j.ia(), j.ib())),
        );
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
        //
        // `active`의 **초집합** 술어 — 뚱뚱한 `Body`를 안 건드리고 Vec<bool> 두 개만 읽는다.
        // static은 여기 쓰지 않는다(페어의 STATIC_BIT가 이미 구분한다). 근거는 호출부 주석.
        let maybe_active = |i: usize, this: &Self| !this.asleep[i] && !this.oob[i];
        let _t_wake = Stopwatch::start();
        // **순회 제한 판정** — 활성 집합이 R에 포함되는가. 전체 스윕의 "첫 wake"는 스윕 시작
        // 상태에서 활성인 끝점을 가지므로, 활성 ⊆ R이면 그 페어는 반드시 fresh(R 접촉)에 있다.
        // 따라서 fresh **드라이런**(무변이)이 후보를 못 찾으면 전체 스윕은 no-op이 증명되고,
        // 하나라도 찾으면 원래 전체 스윕을 그대로 돌린다 — 스윕은 순서-민감(진행 중 wake가
        // 뒤 페어의 판정을 바꾼다)이라 제한 목록으로 스윕 자체를 대체하면 갈릴 수 있다.
        // R 밖 활성(W₀)은 add_joint처럼 wake만 하고 구조를 안 버리는 경로에서 생긴다 —
        // 직전의 조인트 wake 폐쇄가 깨운 바디도 gate=false라 이 스캔에 걸린다.
        let active_subset_of_r = fresh_pairs.is_some()
            && (0..self.bodies.len()).all(|i| {
                self.integrate_gate[i]
                    || !self.alive[i]
                    || self.oob[i]
                    || self.asleep[i]
                    || self.bodies[i].body_type == BodyType::Static
            });
        let mut need_full_sweep = !active_subset_of_r;
        if active_subset_of_r {
            for &(a_id, b_raw) in fresh_pairs.as_ref().expect("checked").iter() {
                let is_static_pair = b_raw & STATIC_BIT != 0;
                let b_plain = b_raw & !STATIC_BIT;
                let (a, b) = (a_id as usize, b_plain as usize);
                let sleeping_dyn = |i: usize, this: &Self| {
                    this.bodies[i].is_dynamic() && this.asleep[i] && !this.oob[i]
                };
                let active = |i: usize, this: &Self| {
                    !this.oob[i]
                        && match this.bodies[i].body_type {
                            BodyType::Kinematic => true,
                            BodyType::Dynamic => !this.asleep[i],
                            BodyType::Static => false,
                        }
                };
                let a_sleep = sleeping_dyn(a, self);
                let b_sleep = !is_static_pair && sleeping_dyn(b, self);
                let b_active = !is_static_pair && active(b, self);
                if !(a_sleep && b_active) && !(b_sleep && active(a, self)) {
                    continue;
                }
                // 제외 쌍은 접촉 판정 **전에** 걸러낸다(공짜 검사를 먼저).
                if !excluded.is_empty()
                    && excluded.contains(&(a_id.min(b_plain), a_id.max(b_plain)))
                {
                    continue;
                }
                // 판정 불가(`None`)는 **보수적으로 "후보 있음"**으로 본다 — 이 드라이런의
                // 목적은 "전체 스윕을 생략해도 되는가"이고, 모르면 생략하면 안 된다.
                let probe = any_contact(
                    &self.bodies[a].shape,
                    &self.bodies[a].pose,
                    &self.bodies[b].shape,
                    &self.bodies[b].pose,
                    self.params.speculative_margin,
                );
                if probe == Some(false) {
                    continue;
                }
                need_full_sweep = true;
                break;
            }
        }
        for &(a_id, b_raw) in pairs
            .iter()
            .take(if need_full_sweep { usize::MAX } else { 0 })
        {
            let is_static_pair = b_raw & STATIC_BIT != 0;
            let b_plain = b_raw & !STATIC_BIT;
            // **양쪽 다 깨어날 수 없으면 즉시 버린다** — 뚱뚱한 `Body`를 건드리기 전에.
            // `maybe_active`는 `active`의 초집합이라 활성 끝점이 있는 페어를 버리지 않는다:
            // `asleep[i]=true`는 dynamic에만 쓰이고(6절 island sleep) 복원도 `asleep ⇒ live
            // dynamic`을 거부 검증하므로, kinematic은 항상 `!asleep`이고 static은 페어의
            // STATIC_BIT로 이미 구분된다. 전원 수면 씬에서 이 한 줄이 24810 페어를
            // Vec<bool> 두 번 읽고 끝낸다(원래는 페어마다 `Body` 4~6회 랜덤 접근).
            if !maybe_active(a_id as usize, self)
                && (is_static_pair || !maybe_active(b_plain as usize, self))
            {
                continue;
            }
            // 조인트가 없으면 집합이 비어 있다 — 빈 집합에도 해시는 도는다.
            if !excluded.is_empty() && excluded.contains(&(a_id.min(b_plain), a_id.max(b_plain))) {
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
            match any_contact(
                &self.bodies[a].shape,
                &self.bodies[a].pose,
                &self.bodies[b].shape,
                &self.bodies[b].pose,
                self.params.speculative_margin,
            ) {
                Some(true) => {
                    if wake_a {
                        self.wake(a);
                    }
                    if wake_b {
                        self.wake(b);
                    }
                }
                Some(false) => {}
                // **판정 불가면 스텝을 포기한다.** `false`로 넘기면 실제로 닿은 잠든 바디를
                // 깨우지 않아 지지 접촉이 빠진 채 적분된다. 이미 깨운 바디가 남는 것은
                // 보수적으로 안전하고(여분 기상은 성능 문제일 뿐), 물리는 적용되지 않는다.
                None => {
                    self.rollback_wake();
                    self.invalidate_bp();
                    self.resource_limit = true;
                    self.narrow_bufs = narrow_bufs;
                    return;
                }
            }
        }
        // 제한 narrow의 조건: 활성 ⊆ R **이고 이번 스텝에 아무도 깨지 않았다** —
        // 그러면 활성 끝점을 가진 페어의 정확한 상집합이 여전히 fresh다.
        let narrow_restricted = active_subset_of_r && !need_full_sweep;
        self.diag.narrow_limited = narrow_restricted;

        self.diag.sub_ns[1] = _t_wake.ns();
        self.diag.phase_ns[1] = _t_bp.ns();
        let _t_np = Stopwatch::start();
        // 2) 내로우페이즈 → 제약 준비 (A=첫 인덱스, B=둘째 — canonical 순서가 곧 해석 순서)
        //    PCM 캐시에서 feature 일치 접점의 임펄스를 warm load (설계 §3)
        // 제약 컨테이너도 페어 수 기준으로 **미리 fallible 예약**한다 — 접촉 페어마다
        // 암묵 성장이 일어나면 그 재할당이 abort가 된다.
        // constraints·constraint_keys·live_keys는 wake 이전에 선점됨 (위) — 여기선 채워 쓴다.
        let _t_nw = Stopwatch::start();
        // 직전 기준선을 페어와 **같은 전순서**로 정렬해 둔다(둘 다 (a, b_raw) 오름차순).
        // 아래 루프가 단조 포인터 하나로 훑으므로 페어당 비용은 비교 1회다.
        // 제한 모드에서는 병합을 안 쓰므로(작은 fresh만 돌고 기준선은 touching 쪽에서 처리)
        // 정렬 자체를 건너뛴다.
        let _t_sort = Stopwatch::start();
        if !narrow_restricted {
            touch_sorted.extend(self.touching.iter().copied());
            touch_sorted.sort_unstable();
        }
        self.diag.sub_ns[4] = _t_sort.ns();
        self.diag.touching = self.touching.len();
        // **제한 모드의 dormant 기준선** — 예전에는 touching 전량(수천)을 필터 재검사 후
        // live_keys에 재삽입하고, 이벤트 패스가 그 전량을 양방향 대조한 뒤 기준선을 통째로
        // 재구축했다(스텝당 해시 연산 ~4×|touching|). 지금은 **잔류를 기본값**으로 한다:
        //
        // 비-fresh(¬R접촉) 기준선 키는 필터 입력이 전 스텝과 동일하다 —
        // ① 그룹(membership/filter)은 생성 후 불변(런타임 setter 없음)
        // ② excluded는 add_joint로만 늘고, add_joint는 양끝을 깨워 그 스텝은 활성⊄R
        //    (제거 API 없음). 필터가 거부한 키는 그 스텝에 이미 touching에서 떨어졌다.
        // ③ 그 밖의 변이는 전부 wake 또는 invalidate_bp 경유 — 제한 모드 전제가 깨진다.
        // 따라서 touching에 있었다 = 전 스텝 필터 통과 = 이번에도 통과. 재검사 없이 잔류하고,
        // 그 논증을 debug_assert로 잠근다. **live_keys는 제한 모드에서 fresh 전용**이 되고,
        // 이벤트·기준선 갱신·PCM retain이 전부 fresh 델타로 내려간다(아래 5-a).
        //
        // R접촉 기준선 키는 여기서 touch_sorted(제한 모드에서 병합 정렬이 안 쓰는 예약
        // 스크래치 — 용량 = |touching| 선점 완료)에 모아 이벤트 패스의 Stopped 후보로 쓴다.
        // 판정 자체는 아래 fresh 순회의 dormant 분기가 담당한다(rev_7f7bb07c 필터 순서 유지).
        if narrow_restricted {
            for &key in self.touching.iter() {
                let (a_id, b_raw) = key;
                let b_plain = b_raw & !STATIC_BIT;
                let r_touch = moved_mask[a_id as usize]
                    || (b_raw & STATIC_BIT == 0 && moved_mask[b_plain as usize]);
                if r_touch {
                    touch_sorted.push(key);
                    continue;
                }
                debug_assert!(
                    excluded.is_empty()
                        || !excluded.contains(&(a_id.min(b_plain), a_id.max(b_plain))),
                    "dormant 기준선 키가 excluded에 있다 — 잔류 논증 붕괴(조인트 변이 경로가 wake를 안 함?)"
                );
                debug_assert!(
                    groups_collide(&self.bodies[a_id as usize], &self.bodies[b_plain as usize]),
                    "dormant 기준선 키가 그룹 필터에 거부됐다 — 그룹이 wake 없이 변이됐다"
                );
            }
        }
        let narrow_pairs: &[(u32, u32)] = if narrow_restricted {
            fresh_pairs.as_ref().expect("narrow_restricted ⇒ 증분 스텝")
        } else {
            &pairs
        };
        // === 내로우페이즈 실행 — 페어 청크를 참여자들이 나눠 돈다 (설계 §5) ===
        //
        // 페어 하나의 처리는 **순수**하다: 읽기는 `NarrowCtx`(전부 불변 참조), 쓰기는
        // 자기 `NarrowOut` 하나뿐이다. 그래서 청크를 나눠 돌리고 청크 순서대로 이어 붙이면
        // 직렬 결과와 **비트 동일**하다 — 각 제약은 자기 페어만의 함수이고 부동소수점
        // 누산이 페어 사이를 넘지 않는다.
        //
        // 청크는 **연속 구간**으로 나눈다(스트라이드 인터리브가 아니라). 이어 붙인 순서가
        // 곧 페어 순서여야 하고, 그 순서가 아래 정렬·컬러링·Gauss-Seidel 순서의 입력이기
        // 때문이다. 참여자 수는 위(wake 폐쇄 전)에서 확정했다.
        let n_np = narrow_pairs.len();
        let n_par = n_par.min(n_np.max(1));
        {
            let cx = NarrowCtx {
                bodies: &self.bodies,
                asleep: &self.asleep,
                oob: &self.oob,
                cache: &self.contact_cache,
                touching: &self.touching,
                touch_sorted: &touch_sorted,
                excluded: &excluded,
                restricted: narrow_restricted,
                margin: self.params.speculative_margin,
                dt,
            };
            // 청크 경계 — 참여자 w는 [n_np·w/n_par, n_np·(w+1)/n_par)를 맡는다.
            let bounds = |w: usize| (n_np * w / n_par, n_np * (w + 1) / n_par);
            // 기준선 병합 포인터의 시작값. 직렬 루프는 단조 포인터 하나로 훑지만 청크는
            // 중간에서 시작하므로, 자기 첫 키까지 **이분 탐색**으로 전진시킨 값이 필요하다
            // (둘 다 오름차순이라 결과가 같다). 제한 모드는 이 포인터를 안 쓴다.
            let ti_of = |lo: usize| {
                if narrow_restricted || lo >= n_np {
                    0
                } else {
                    touch_sorted.partition_point(|k| *k < narrow_pairs[lo])
                }
            };
            if n_par > 1 {
                let view = crate::threads::DisjointSlice::new(&mut narrow_bufs[..n_par]);
                self.pool.dispatch(|ctx| {
                    let mut w = ctx.index();
                    while w < n_par {
                        // SAFETY: 참여자 w가 만지는 버퍼는 인덱스 w 하나뿐이고, w는
                        // 참여자마다 서로 다르다(`w = index + k·threads`).
                        let out = unsafe { view.get(w) };
                        out.reset();
                        let (lo, hi) = bounds(w);
                        narrow_chunk(&cx, &narrow_pairs[lo..hi], ti_of(lo), out);
                        w += ctx.threads();
                    }
                });
            } else {
                let out = &mut narrow_bufs[0];
                out.reset();
                narrow_chunk(&cx, narrow_pairs, 0, out);
            }
        }
        // 병합 — 청크(=페어) 순서 그대로. 실패(bail)는 어느 청크에서 났든 같은 결말이다.
        if narrow_bufs[..n_par].iter().any(|b| b.bail) {
            // 접점 힙 할당 실패(극단 OOM). narrowphase는 wake 이후이고 pairs를 대여 중이라
            // bp_pairs를 되돌릴 수 없다 — 구조를 버려 다음 스텝이 재빌드하게 한다.
            // sleep 상태는 캐노니컬이므로 되돌린다.
            self.rollback_wake();
            self.narrow_bufs = narrow_bufs;
            self.invalidate_bp();
            self.resource_limit = true;
            return;
        }
        if n_par == 1 {
            // 참여자가 하나면 **옮기지 않고 스왑**한다(제약 하나가 ~100B라 이동도 공짜가
            // 아니다). 결과는 append와 같고, 버퍼는 다음 스텝을 위해 용량째 되돌아온다.
            core::mem::swap(&mut constraints, &mut narrow_bufs[0].constraints);
            core::mem::swap(&mut constraint_keys, &mut narrow_bufs[0].keys);
        } else {
            for b in narrow_bufs[..n_par].iter_mut() {
                constraints.append(&mut b.constraints);
                constraint_keys.append(&mut b.keys);
            }
        }
        for b in narrow_bufs[..n_par].iter() {
            for &k in &b.live {
                live_keys.insert(k);
            }
            // max는 결합·교환법칙이 성립하고(NaN이 있어도 f32::max는 비-NaN을 고른다)
            // 정수 합도 순서 무관이라, 진단 누적은 청크 순서에 의존하지 않는다.
            self.diag.max_penetration = self.diag.max_penetration.max(b.max_penetration);
            self.diag.warm_total += b.warm_total;
            self.diag.warm_hits += b.warm_hits;
            self.diag.heightfield_saturated += b.heightfield_saturated;
        }
        // **접촉 판정이 불완전하면 여기서 끝낸다.** 플래그만 세우고 계속 풀면 앞부분
        // 제약만으로 한 스텝을 적용한 뒤 오류를 돌려주게 되는데, `ResourceLimit` 계약은
        // "커맨드는 적용됨, **물리 스텝은 건너뜀**"이다 — 호스트가 재시도하면 물리가 두 번
        // 나아간다(확인검수 rev_1a31d83f BLOCKER). 솔브·적분·이벤트·PCM 전부 앞에서 끊는다.
        if narrow_bufs[..n_par].iter().any(|b| b.budget_exceeded) {
            self.rollback_wake();
            self.narrow_bufs = narrow_bufs;
            self.invalidate_bp();
            self.resource_limit = true;
            return;
        }
        self.narrow_bufs = narrow_bufs;
        // 진단은 **dispatch 뒤에** 읽는다 — 워커는 지연 생성이라 그 전에는 "요청한 수"만
        // 알 수 있고, 스레드를 못 띄운 환경에서 실제와 갈린다(설계 §6.3 조용한 폴백 금지).
        self.diag.narrow_threads = if n_par > 1 {
            self.pool.threads().min(n_par)
        } else {
            1
        };

        self.diag.sub_ns[2] = _t_nw.ns();

        // **중력 방향 정렬** — Gauss-Seidel은 제약 순서에 민감하다. 스택에서 지지력은
        // 바닥에서 위로 전파되는데, 브로드페이즈가 내놓은 순서로 풀면 한 스윕에 한 층씩만
        // 전파돼 수렴이 스택 깊이에 비례해 느려진다. 중력 방향 아래쪽 접촉부터 풀면
        // 한 스윕으로 위까지 전파된다.
        // 정렬 키 = 중력 단위벡터로의 투영(아래일수록 큼) → 내림차순. 무중력이면 전부 0이라
        // 페어 키만 남는다. 동률은 페어 키(전순서·유일)로 끊어 **결정성**을 유지한다.
        if constraints.len() > 1 {
            let nb = self.bodies.len();
            let nc = constraints.len();
            // 컬러링·재그룹 스크래치(order·conflict·endpoints·body_color·color_of·grouped·
            // c_tmp·new_keys)는 **wake 폐쇄 이전에 이미 fallible 선점**했다(위 1.5 앞) — 여기선
            // 채워 쓰기만 하므로 어떤 push/extend/resize도 재할당하지 않는다(전부 pairs.len()·
            // bodies.len() 상한 안). nc ≤ pairs.len(), nb == 선점 시 bodies.len().
            let gdir = self.params.gravity.normalize_or_zero();
            let depth = |c: &ContactConstraint| -> f32 {
                let pa = self.bodies[c.a].pose.pos;
                let pb = self.bodies[c.b].pose.pos;
                pa.dot(gdir).max(pb.dot(gdir))
            };
            // ① canonical 전순서 (중력깊이 주 + pair-key 동률) — GS 수렴·결정성의 근거.
            order.extend(0..nc as u32);
            order.sort_unstable_by(|&i, &j| {
                let (i, j) = (i as usize, j as usize);
                depth(&constraints[j])
                    .total_cmp(&depth(&constraints[i]))
                    .then_with(|| constraint_keys[i].cmp(&constraint_keys[j]))
            });
            // ② 그래프 컬러링 (설계 §5): canonical 순서 기반 그리디. 같은 색 = 공유 dynamic
            //    바디 없음 → 레인 병렬 가능. static/kinematic은 솔버가 안 쓰므로 conflict 아님.
            //    이 순서로 **콜러드-스칼라** 솔브를 돌려(아래 서브스텝 루프는 그대로) 리오더링
            //    비용·수렴·안정성을 먼저 재는 게 Unit 1 게이트다 — SIMD는 그 다음.
            conflict.extend(
                (0..nb).map(|i| self.alive[i] && !self.oob[i] && self.bodies[i].is_dynamic()),
            );
            endpoints.extend(order.iter().map(|&o| {
                let c = &constraints[o as usize];
                (c.a as u32, c.b as u32)
            }));
            body_color.resize(nb, 0u64);
            solver::color_constraints(
                &endpoints,
                &conflict,
                &mut body_color,
                &mut color_of,
                color_cap(),
            );
            // ③ 색으로 **안정** 재그룹: 최종 순서 = 색 오름차순, 색 내 canonical 순서 보존.
            //    제약'간' 순서만 red-black으로 바뀌고 색 안의 GS 순서는 canonical 그대로다.
            //    색은 0..=COLOR_OVERFLOW(63) 유계라 offs는 **고정 배열**(할당 없음).
            let ncol = color_of.iter().copied().max().unwrap_or(0) as usize + 1;
            let mut offs = [0u32; solver::COLOR_OVERFLOW as usize + 1];
            for &c in &color_of {
                offs[c as usize] += 1;
            }
            let mut acc = 0u32;
            for o in offs.iter_mut().take(ncol) {
                let cnt = *o;
                *o = acc;
                acc += cnt;
            }
            // 색 범위 [start,end) 캡처 — 이 시점 offs[c]=색 c 시작(placement 루프가 곧 변형).
            for c in 0..ncol {
                let start = offs[c];
                let end = if c + 1 < ncol { offs[c + 1] } else { nc as u32 };
                color_ranges.push((start, end));
            }
            grouped.resize(nc, 0u32);
            for r in 0..order.len() {
                let col = color_of[r] as usize;
                grouped[offs[col] as usize] = order[r];
                offs[col] += 1;
            }
            // ④ 최종 순열을 constraints/constraint_keys에 락스텝으로 적용 (PCM 갱신이
            //    `constraints.iter().zip(&constraint_keys)`로 정렬 정합을 요구, 5)절).
            //    constraints는 drain으로 용량을 보존하고, keys는 선점한 new_keys로 재구성한다.
            c_tmp.extend(constraints.drain(..).map(Some));
            for &o in &grouped {
                constraints.push(
                    c_tmp[o as usize]
                        .take()
                        .expect("permutation is a bijection"),
                );
                new_keys.push(constraint_keys[o as usize]);
            }
            constraint_keys = new_keys;
        }
        // 제약 ≤1이면 컬러링 블록을 건너뛴다 — 색 분리가 불필요하다. warm_start 배치 디스패치가
        // 비지 않도록 전체를 한 범위(스칼라)로 둔다.
        if color_ranges.is_empty() && !constraints.is_empty() {
            color_ranges.push((0, constraints.len() as u32));
        }
        // === SoA 블록 조립 (스텝당 1회) ===
        // 같은 색의 연속 4제약은 공유 dynamic 바디가 없으므로 레인 독립이다. 그 4개를
        // SoA 블록으로 미리 실어 두면 서브스텝 루프(스텝당 12패스)가 게더 없이 돈다.
        // 분할 규칙은 `solve_batched`의 chunk(4)와 동일하므로 두 경로의 처리 순서가 같다.
        // 블록 풀을 꺼내 온다 — 여기까지 오면 이 스텝의 조기 반환은 없다(bail은 위에서 끝났다).
        let mut blocks = std::mem::take(&mut self.contact_blocks);
        solver_block::plan(
            &constraints,
            &color_ranges,
            &mut work,
            &mut block_starts,
            &mut color_work,
        );
        // 풀을 필요한 길이까지만 늘리고(용량은 위에서 확보), 기존 원소는 제자리에서 덮어쓴다.
        if blocks.len() < block_starts.len() {
            blocks.resize_with(block_starts.len(), Default::default);
        }
        for (bi, &s) in block_starts.iter().enumerate() {
            blocks[bi].assemble_into([
                &constraints[s],
                &constraints[s + 1],
                &constraints[s + 2],
                &constraints[s + 3],
            ]);
        }

        self.diag.colors = color_work.len();
        // 오버플로 제약 수 — 색 상한의 대가(직렬 잔여)를 관측 가능하게 남긴다.
        self.diag.color_overflow = color_ranges
            .get(solver::COLOR_OVERFLOW as usize)
            .map(|&(s, e)| (e - s) as usize)
            .unwrap_or(0);
        self.diag.solve_work = work.len();
        self.diag.solve_threads =
            solver_block::solve_participants(&self.pool, self.parallel_solver, &work);
        self.diag.phase_ns[2] = _t_np.ns();
        let _t_sv = Stopwatch::start();
        // 3) TGS-soft 서브스텝 루프
        let g = self.params.gravity;
        // 조인트는 접촉보다 강성 (Box2D v3: 2×contact hertz, ζ=2)
        let joint_soft = Softness::new(
            (2.0 * self.params.solver.contact_hertz).min(0.25 * inv_h),
            2.0,
            h,
        );
        // 솔버 스크래치(solver_state·inv_inertia_local·게이트·ext_force·damping·island_parent)는
        // **wake 이전에 전부 선점됨** (위 '스텝 스크래치 전량 선점'). 캐노니컬 변이(joints take)보다
        // 앞서 예약이 끝났으므로 여기선 take·채우기만 하고 실패하지 않는다.
        let mut joints = std::mem::take(&mut self.joints);
        // 조인트도 접촉과 같은 비활성 게이트 (설계 §4.6 level-trigger): 양쪽 모두
        // 비활성(sleeping/static/kinematic)이면 스킵 — 잠든 바디에 warm-start 임펄스를
        // 주입하면 위치 적분이 없는 채 속도만 펌핑돼 정착한 체인이 되살아난다(실증:
        // t=6s vmax 0.006 정착 → 슬립 → t=8s 4.5 폭주).
        // active_joints는 위에서 선점됨 — 여기선 채운다(post-wake asleep를 읽으므로 wake 이후).
        active_joints.extend(
            joints
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
                .map(|(i, _)| i),
        );
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
        // 누적 힘·토크도 압축 배열로 뽑는다 — 솔버 루프에서 뚱뚱한 Body를 다시 훑지 않는다.
        let ext_force = &mut self.ext_force;
        ext_force.clear();
        ext_force.extend(self.bodies.iter().map(|b| (b.force, b.torque)));
        let damping = &mut self.damping;
        damping.clear();
        damping.extend(
            self.bodies
                .iter()
                .map(|b| (b.linear_damping, b.angular_damping)),
        );
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
        let ext_force = &self.ext_force;
        let damping = &self.damping;
        let integrate = &self.integrate_gate;
        let gravity_on = &self.gravity_gate;
        let (max_lin, max_ang) = (
            self.params.max_linear_velocity,
            self.params.max_angular_velocity,
        );
        // **활성 바디만 훑는다.** 전원이 잠든 40단에서도 서브스텝 루프가 8회 × O(바디)
        // 패스를 그대로 돌고 있었다(중력·역관성 갱신·속도 클램프·위치 적분) — 820 바디면
        // 수만 번의 무의미한 순회다. 압축 인덱스 목록을 만들어 그것만 순회한다.
        let active: Vec<u32> = (0..n_bodies)
            .filter(|&i| integrate[i])
            .map(|i| i as u32)
            .collect();
        let grav_active: Vec<u32> = active
            .iter()
            .copied()
            .filter(|&i| gravity_on[i as usize])
            .collect();
        // === CCD linear cast 준비 (설계 §4 CCD 계약 v0.5 — 대상 static 한정, opt-in) ===
        // 활성 CCD 바디마다: 서브스텝 적분에서 이동 경로를 static 트리에 캐스트해 TOI에서
        // 절단하고, 이번 스텝의 잔여 서브스텝 동안 **단방향 평면 클램프**(침투 방향 이동
        // 성분만 사영 제거 — 슬라이드 허용)를 유지한다. 평면은 바디당 최대 4개, 초과 hit는
        // 이동 종료(구석 퇴화 — 결정적·안전). 속도는 절대 건드리지 않는다: 정지 위치가
        // 표면(speculative skin 안)이라 다음 스텝의 일반 접촉 파이프라인이 실제 접근
        // 속도로 반발·정지를 해석한다.
        let ccd_list: Vec<u32> = active
            .iter()
            .copied()
            .filter(|&i| {
                let b = &self.bodies[i as usize];
                // 센서는 절대 위치 절단 대상이 아니다(교차검수 rev_e4075e85 MAJOR) —
                // "이벤트만 내고 밀지 않는다" 계약과 정반대가 된다. swept trigger는 미래 확장.
                b.ccd && b.body_type == BodyType::Dynamic && !b.sensor
            })
            .collect();
        let mut ccd_index: Vec<u32> = Vec::new();
        let mut ccd_planes: Vec<[Vec3; 4]> = Vec::new();
        let mut ccd_plane_count: Vec<u8> = Vec::new();
        let mut ccd_frozen: Vec<bool> = Vec::new();
        let mut ccd_cand: Vec<u32> = Vec::new();
        let mut ccd_stack: Vec<u32> = Vec::new();
        if !ccd_list.is_empty() {
            ccd_index = vec![u32::MAX; n_bodies];
            for (k, &i) in ccd_list.iter().enumerate() {
                ccd_index[i as usize] = k as u32;
            }
            ccd_planes = vec![[Vec3::ZERO; 4]; ccd_list.len()];
            ccd_plane_count = vec![0u8; ccd_list.len()];
            ccd_frozen = vec![false; ccd_list.len()];
        }
        for sub_i in 0..n_sub {
            // 역관성 갱신을 **외력보다 먼저** — 토크는 현재 자세의 월드 역관성을 쓴다
            // (뒤에 두면 첫 서브스텝에서 `inv_i`가 아직 0이라 토크가 통째로 사라진다).
            let _t_ii = Stopwatch::start();
            st.refresh_inv_i_active(inv_i_local, &active);
            self.diag.solve_ns[0] += _t_ii.ns();
            for &i in &grav_active {
                let i = i as usize;
                st.lin[i] += g * h;
                // 외력은 **중력과 같은 자리**에서 가속도로 들어간다: dv = F/m · h.
                // 토크는 현재 자세의 월드 역관성을 쓴다(자세가 서브스텝마다 바뀐다).
                let (f, t) = ext_force[i];
                if f != Vec3::ZERO {
                    st.lin[i] += f * (st.inv_m[i] * h);
                }
                if t != Vec3::ZERO {
                    st.ang[i] += st.inv_i[i].mul_vec(t) * h;
                }
            }

            // **암시적 감쇠**: v ← v / (1 + h·d). 명시적(`1 − h·d`)이면 `h·d > 1`에서
            // 부호가 뒤집혀 발산한다 — 큰 값을 넣으면 폭발하는 대신 더 세게 멈춘다.
            //
            // 중력 루프에 얹지 않고 **여기서** 따로 돈다. 오늘은 두 게이트의 집합이 같아
            // 결과가 맞지만, 바디별 중력 배율·비활성이 중력 게이트에 들어오는 날
            // 감쇠가 조용히 같이 꺼진다(검수 MINOR). 감쇠는 활성 dynamic의 성질이다.
            for &i in &active {
                let i = i as usize;
                if !st.dynamic[i] {
                    continue;
                }
                let (dl, da) = damping[i];
                if dl > 0.0 {
                    st.lin[i] = st.lin[i] * (1.0 / (1.0 + h * dl));
                }
                if da > 0.0 {
                    st.ang[i] = st.ang[i] * (1.0 / (1.0 + h * da));
                }
            }

            let _t_ra = Stopwatch::start();
            solver_block::refresh_anchors_all(
                &self.pool,
                self.parallel_solver,
                &solver::SolveView::new(&mut st),
                &mut blocks,
                &mut constraints,
                &work,
                &color_work,
            );
            self.diag.solve_ns[1] += _t_ra.ns();
            // warm start + bias 솔브 (조인트 먼저 — Box2D 순서)
            let _t_jw = Stopwatch::start();
            for &ji in &active_joints {
                joint::warm_start(&mut st, &joints[ji]);
            }
            self.diag.solve_ns[6] += _t_jw.ns();
            // 접촉 warm_start — 색 배치 SIMD 디스패치(Unit 3b). 색 내 4제약은 4-wide, 나머지는
            // 스칼라. 콜러드-스칼라와 비트 동일이라 L0 해시 불변(아래 state_hash 테스트가 잠금).
            let _t_ws = Stopwatch::start();
            solver_block::warm_start_all(
                &self.pool,
                self.parallel_solver,
                &solver::SolveView::new(&mut st),
                &blocks,
                &constraints,
                &work,
                &color_work,
            );
            self.diag.solve_ns[2] += _t_ws.ns();
            let _t_js = Stopwatch::start();
            for &ji in &active_joints {
                joint::solve(
                    &mut st,
                    &mut joints[ji],
                    h,
                    Some((
                        &joint_soft,
                        self.params.solver.max_bias_speed,
                        self.params.max_angular_velocity,
                    )),
                );
            }
            self.diag.solve_ns[6] += _t_js.ns();
            // bias 솔브 — 색 배치 SIMD 디스패치(Unit 3c). 콜러드-스칼라와 비트 동일.
            let _t_bs = Stopwatch::start();
            solver_block::solve_all(
                &self.pool,
                self.parallel_solver,
                &solver::SolveView::new(&mut st),
                &mut blocks,
                &mut constraints,
                &work,
                &color_work,
                &bias_params,
            );
            self.diag.solve_ns[3] += _t_bs.ns();
            // 속도 상한 — 위치 적분 **전에** 걸어야 변위도 유계가 된다
            let _t_int = Stopwatch::start();
            for &i in &active {
                let i = i as usize;
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
            for &i in &active {
                let i = i as usize;
                let ci = if ccd_list.is_empty() {
                    u32::MAX
                } else {
                    ccd_index[i]
                };
                if ci == u32::MAX {
                    st.pos[i] += st.lin[i] * h;
                } else {
                    // CCD 절단 적분: 평면 사영 → 잔여 이동 재캐스트 → TOI 절단.
                    let ci = ci as usize;
                    let mut d = st.lin[i] * h;
                    for &n in ccd_planes[ci].iter().take(ccd_plane_count[ci] as usize) {
                        let pen = d.dot(n);
                        if pen < 0.0 {
                            d = d - n * pen; // 침투 성분만 제거 — 슬라이드 허용
                        }
                    }
                    // **회전 스윕**(§4 예약 확장): 이번 서브스텝의 총각 × 외접 반지름이
                    // skin의 1/4을 넘으면 회전 인지 캐스트를 쓴다 — 그 아래는 표면 이동이
                    // skin 봉투 안이라 선형 캐스트(기존 경로, 비트 보존)로 충분하다.
                    // 회전 히트 시 이 서브스텝의 회전도 같은 TOI에서 절단한다(속도는 불변 —
                    // 반발은 다음 스텝 일반 파이프라인 소관, 선형 계약과 동일).
                    let skin = self.params.speculative_margin;
                    let omega = st.ang[i];
                    let sub_angle = omega.length() * h;
                    // **게이트는 스텝 전체 회전 봉투로** — 서브스텝 arc로 판정하면
                    // substeps가 클수록 서브스텝당 이동이 skin/4 아래로 숨어, 스텝
                    // 전체로는 수십 cm를 도는 블레이드가 캐스트 없이 회전한다(서브스텝
                    // 사이에는 내로우페이즈가 없어 침투가 무검사로 누적된다 — 확인 검수
                    // rev_8672a3eb MAJOR). 캐스트 자체는 서브스텝 각(sub_angle)으로 돈다.
                    // 반지름은 **각 스윕 반지름**(코어의 축 수직 거리) — 구·자기 축
                    // 캡슐처럼 회전이 점유를 안 바꾸는 형상은 0이라 게이트가 꺼진다.
                    // 굴러가는 CCD 탄환 구가 매 서브스텝 회전 캐스트를 도는 낭비와
                    // 보수적 절단을 막는다(확인 검수 rev_477d7366 MAJOR).
                    let sweep_r = if sub_angle > 0.0 {
                        let axis_local =
                            st.rot[i].conjugate().rotate(omega * (1.0 / omega.length()));
                        self.bodies[i].shape.angular_sweep_radius(axis_local)
                    } else {
                        0.0
                    };
                    let arc = sub_angle * (n_sub as f32) * sweep_r;
                    let rot_sweep = arc > 0.25 * skin;
                    let mut rot_toi = 1.0f32;
                    if ccd_frozen[ci] {
                        // frozen = 이번 스텝 **이동** 종료. 회전 스윕은 캐스트를 계속
                        // 돌려 TOI 절단한다 — 이동만 얼리고 회전을 무검사로 적분하면
                        // 구석에 낀 블레이드의 팁이 남은 서브스텝 동안 벽을 쓸고
                        // 지나간다(검수 rev_3398738c MAJOR).
                        d = Vec3::ZERO;
                        self.diag.ccd_frozen += 1;
                    }
                    // **월드 범위를 벗어난 자세로는 캐스트하지 않는다** (자기 검수 2026-07-30).
                    // 경계 고정(`enforce_world_bounds`)은 스텝 **말**에 도는데, 빠른 바디는 스텝
                    // 중간에 10km를 넘을 수 있다. 그 자세를 디스패처에 넘기면 이제
                    // `InvalidInput`이 되고(범위 검사) 그것이 `cast_incomplete`로 접혀
                    // **원인이 잘못 보고된다**(`ccd_incomplete` 문서는 "지형 해상도·바디 크기를
                    // 보라"고 말한다). 범위 이탈은 스텝 말 OOB 패스가 `out_of_bounds`로 센다.
                    //
                    // **캐스트는 상대 좌표로 돈다** — 이 자리를 네 판 틀린 끝에 나온 답이다.
                    //
                    // ① `continue`로 회전 적분까지 건너뜀 → 감지 서브스텝만 회전이 빠지는
                    //    비대칭(rev_59dcb940). ② `d=0`+`ccd_frozen`으로 병진 동결 → 잠깐 넘었다
                    //    돌아올 바디가 복귀 불가·영구 OOB, 정적 후보 0개인데도 CCD 켬/끔이 궤적을
                    //    바꿈(rev_cccbba53). ③ 범위 밖이면 캐스트만 생략 → **"밖에는 부딪힐 것이
                    //    없다"는 내 전제가 틀렸다**: 정적 바디는 **부모 자세만** `MAX_INPUT` 안이면
                    //    되고 복합 자식 오프셋·격자는 `MAX_SHAPE_EXTENT`(1km)까지 더 뻗을 수 있다.
                    //    경계 밖의 유효한 장애물을 조용히 관통했다(rev_5ca93cb1).
                    //
                    // ④ 그래서 **절대 좌표를 쓰지 않는다.** 충돌 기하는 병진 불변이므로 두 자세에서
                    //    공통 원점을 빼면 결과가 같다. 원점은 **두 부모의 중점**이다(아래 캐스트
                    //    자리 주석: 움직이는 바디에 두면 상대 자세 하나가 간격 전체를 떠안아
                    //    유효한 후보가 범위 검사에 걸린다 — rev_6f54ca63). 덤으로 10km 좌표의
                    //    자리수 손실도 사라진다 — 상대 좌표는 0 근처라 f32 정밀도를 온전히 쓴다.
                    if (!ccd_frozen[ci] && (d != Vec3::ZERO || rot_sweep))
                        || (ccd_frozen[ci] && rot_sweep)
                    {
                        let body = &self.bodies[i];
                        let pose_now = maru_collide::Pose {
                            pos: st.pos[i],
                            rot: st.rot[i],
                        };
                        let a0 = shape_world_aabb(&body.shape, &pose_now);
                        // 회전 스윕이면 외접 반지름 상자로 봉투를 넓힌다 — 스윕 AABB가
                        // 후보를 놓치면 캐스트는 옳아도 관통이 조용히 지나간다.
                        let a0 = if rot_sweep {
                            let r = body.shape.bounding_radius();
                            let e = Vec3::new(r, r, r);
                            Aabb::new(pose_now.pos - e, pose_now.pos + e).merge(a0)
                        } else {
                            a0
                        };
                        let sweep = a0.merge(Aabb::new(a0.min + d, a0.max + d));
                        // static 후보 — bp 구조는 브로드페이즈 단계가 항상 남긴다(무효화
                        // 경로는 전부 그 전에 return). 방어적으로 없으면 캐스트 생략.
                        let mut best: Option<(f32, Vec3, Vec3, u32)> = None;
                        // 캐스트가 판정을 완료하지 못한 경우(지형 셀 예산 초과 등).
                        // **여기는 이미 솔브·적분 뒤라 스텝을 되돌릴 수 없다** — 그래서
                        // 소비자 정책으로 처리한다: 이 서브스텝의 이동·회전을 0으로 묶어
                        // 관통을 막고(보수적 정지), `resource_limit`으로 호스트에 알린다.
                        // 캐스트가 가짜 히트를 만들어 내려보내지 않는 이유는 그쪽 주석에 있다.
                        let mut cast_incomplete = false;
                        if let Some(bp) = &self.bp {
                            ccd_cand.clear();
                            bp.query_static(sweep, &mut ccd_cand, &mut ccd_stack);
                            for &sid in &ccd_cand {
                                let sb = &self.bodies[sid as usize];
                                // 접촉 규칙과 동일한 필터: 그룹·센서·조인트 제외 쌍.
                                if sb.sensor || !groups_collide(body, sb) {
                                    continue;
                                }
                                if !excluded.is_empty() {
                                    let (x, y) = (i as u32, sid);
                                    if excluded.contains(&(x.min(y), x.max(y))) {
                                        continue;
                                    }
                                }
                                // **상대 좌표 — 원점은 두 부모 위치의 중점**이다. 반환값 중
                                // 좌표계에 의존하는 것은 `point`뿐이라 아래에서 원점을 다시
                                // 더한다(`toi`·`normal`은 병진 불변).
                                //
                                // 처음에는 원점을 움직이는 바디에 뒀는데, 그러면 상대 자세
                                // 하나(`cast_pb`)가 두 부모 간격 전체를 떠안아 **유효한 후보가
                                // 디스패처의 `MAX_INPUT` 검사에 걸린다**. 내가 적었던
                                // "|Δ| ≤ 4.2km"는 과소평가였다(확인검수 rev_6f54ca63):
                                //   hull 점 (±1e3)³ → |p| = √3·1e3 = 1732m, + convex radius 1e3
                                //   → 자식 2732m, + 자식 오프셋 √3·1e3 → 복합 반경 4464m.
                                //   두 바디 8928m(축별), 벡터로 √3배 = 15464m, + 이동 2500m
                                //   → 최악 17964m > MAX_INPUT.
                                // 중점을 쓰면 각 상대 자세가 그 절반(≤ 8982m)이라 범위 안이고,
                                // 기하는 공통 병진이라 그대로다.
                                let cast_origin = ccd_cast_origin(pose_now.pos, sb.pose.pos);
                                let cast_pa = maru_collide::Pose {
                                    pos: pose_now.pos - cast_origin,
                                    rot: pose_now.rot,
                                };
                                let cast_pb = maru_collide::Pose {
                                    pos: sb.pose.pos - cast_origin,
                                    rot: sb.pose.rot,
                                };
                                let cast = if rot_sweep {
                                    maru_collide::try_shape_cast_dispatch_rotational(
                                        &body.shape,
                                        &cast_pa,
                                        d,
                                        omega * (1.0 / omega.length()),
                                        sub_angle,
                                        &sb.shape,
                                        &cast_pb,
                                        skin,
                                    )
                                } else {
                                    try_shape_cast_dispatch(
                                        &body.shape,
                                        &cast_pa,
                                        d,
                                        &sb.shape,
                                        &cast_pb,
                                        skin,
                                    )
                                };
                                let cast = match cast {
                                    Ok(c) => c,
                                    Err(_) => {
                                        // **첫 실패에서 후보 순회를 끝낸다.** 결말(보수적 정지)이
                                        // 이미 확정됐으므로 나머지 후보 검사는 전부 낭비다 —
                                        // 같은 자리에 지형 100개를 겹쳐 두면 서브스텝당 40만 셀이
                                        // 된다(확인검수 rev_2e93ebf2 MAJOR).
                                        cast_incomplete = true;
                                        break;
                                    }
                                };
                                // 접점만 월드 좌표로 되돌린다.
                                let cast = cast.map(|h| maru_collide::ShapeCastHit {
                                    point: h.point + cast_origin,
                                    ..h
                                });
                                if let Some(hit) = cast {
                                    // 결정적 최소: (toi, static id) 사전식 — 동시 hit 동률 절단.
                                    let better = match best {
                                        None => true,
                                        Some((bt, _, _, bid)) => {
                                            hit.toi < bt || (hit.toi == bt && sid < bid)
                                        }
                                    };
                                    if better {
                                        best = Some((hit.toi, hit.normal, hit.point, sid));
                                    }
                                }
                            }
                        }
                        if cast_incomplete {
                            // 판정 미완료 = "막혔다"로 취급한다(보수적 정지). 미스로 넘기면
                            // 지형을 통과한다.
                            //
                            // **`resource_limit`은 세우지 않는다.** 그 플래그의 계약은
                            // "커맨드는 적용, 물리 스텝은 건너뜀"이고, 여기는 이미 솔브·적분
                            // 중이라 스텝을 건너뛸 수 없다 — 세우면 한 플래그가 두 가지를
                            // 뜻하게 되고 호스트는 "안 움직였다"고 오해한다. 보고 채널은
                            // `Diagnostics::ccd_incomplete`다(`ccd_hits_dropped`·
                            // `heightfield_saturated`와 같은 규약: 조용한 강등 금지).
                            d = Vec3::ZERO;
                            rot_toi = 0.0;
                            // **속도도 멈춘다.** 위치만 얼리면 중력·솔버가 매 서브스텝 속도를
                            // 계속 키워 `max_linear_velocity`까지 쌓이고, 호스트가 지형을 줄여
                            // 판정이 가능해지는 순간 그 바디가 최대 속도로 발사된다
                            // (확인검수 rev_2e93ebf2 MAJOR — 위치만 비교한 테스트가 놓쳤다).
                            // 보수적 정지의 물리적 해석은 "막혔다"이므로 정지가 일관된다.
                            st.lin[i] = Vec3::ZERO;
                            st.ang[i] = Vec3::ZERO;
                            ccd_frozen[ci] = true;
                            self.diag.ccd_incomplete += 1;
                        } else if let Some((toi, normal, point, sid)) = best {
                            // **절단 순간을 호스트에게 보고한다** (설계 §4.5 TOI-hit 큐).
                            // 상한을 넘으면 뒤에 온 것부터 버린다 — 바디 순회(활성 인덱스
                            // 오름차순)와 서브스텝 순서가 결정적이라 버려지는 집합도 결정적이다.
                            if self.ccd_hits.len() < MAX_CCD_HITS {
                                self.ccd_hits.push(CcdHit {
                                    body: self.handle_of(i),
                                    other: self.handle_of(sid as usize),
                                    toi,
                                    substep: sub_i,
                                    normal,
                                    point,
                                });
                            } else {
                                self.diag.ccd_hits_dropped += 1;
                            }
                            d = d * toi;
                            if rot_sweep {
                                // 회전 스윕 히트: 이 서브스텝의 회전도 TOI에서 절단 —
                                // 아니면 캐스트가 멈춘 자리를 회전이 마저 뚫는다.
                                rot_toi = toi;
                            }
                            self.diag.ccd_clamps += 1;
                            let cnt = ccd_plane_count[ci] as usize;
                            if cnt < 4 {
                                ccd_planes[ci][cnt] = normal;
                                ccd_plane_count[ci] = (cnt + 1) as u8;
                            } else {
                                // 평면 5개째 = 구석에 낀 상태 — 이번 스텝 이동 종료 (설계).
                                ccd_frozen[ci] = true;
                            }
                        }
                    }
                    st.pos[i] += d;
                    if rot_toi < 1.0 {
                        st.rot[i] = st.rot[i].integrate(st.ang[i], h * rot_toi);
                        continue; // 아래 전체-h 회전 적분을 건너뛴다
                    }
                }
                st.rot[i] = st.rot[i].integrate(st.ang[i], h);
            }
            self.diag.solve_ns[5] += _t_int.ns();
            // relax — bias 에너지 제거 (위치 적분으로 자세가 바뀌었으므로 갱신)
            let _t_ii2 = Stopwatch::start();
            st.refresh_inv_i_active(inv_i_local, &active);
            self.diag.solve_ns[0] += _t_ii2.ns();
            let _t_ra2 = Stopwatch::start();
            solver_block::refresh_anchors_all(
                &self.pool,
                self.parallel_solver,
                &solver::SolveView::new(&mut st),
                &mut blocks,
                &mut constraints,
                &work,
                &color_work,
            );
            self.diag.solve_ns[1] += _t_ra2.ns();
            let _t_js2 = Stopwatch::start();
            for &ji in &active_joints {
                joint::solve(&mut st, &mut joints[ji], h, None);
            }
            self.diag.solve_ns[6] += _t_js2.ns();
            // relax 솔브 — 색 배치 SIMD 디스패치(Unit 3c).
            let _t_rs = Stopwatch::start();
            solver_block::solve_all(
                &self.pool,
                self.parallel_solver,
                &solver::SolveView::new(&mut st),
                &mut blocks,
                &mut constraints,
                &work,
                &color_work,
                &relax_params,
            );
            self.diag.solve_ns[4] += _t_rs.ns();
        }

        // 블록의 누적 임펄스를 스칼라 제약으로 되쓴다 — restitution 패스와 PCM 갱신이
        // 스칼라 제약을 읽는다. 스텝당 1회면 충분하다(서브스텝 안에서는 블록이 정본).
        // **풀은 직전 스텝의 길이를 남길 수 있다** — 이번 스텝의 블록 수만 훑는다.
        for (bi, blk) in blocks.iter().enumerate().take(block_starts.len()) {
            let s = block_starts[bi];
            let (l, r) = constraints.split_at_mut(s + 1);
            let (c1, c2) = r.split_at_mut(1);
            let (c3, c4) = c2.split_at_mut(1);
            blk.scatter_impulses([&mut l[s], &mut c1[0], &mut c3[0], &mut c4[0]]);
        }

        self.contact_blocks = blocks;

        // 4) restitution 패스 (임팩트 게이트)
        for c in constraints.iter_mut() {
            solver::apply_restitution(&mut st, c, self.params.solver.restitution_threshold);
        }

        self.diag.phase_ns[3] = _t_sv.ns();
        let _t_post = Stopwatch::start();
        let _t_wb = Stopwatch::start();
        st.write_back(&mut self.bodies, &self.alive);
        self.solver_state = st;

        // **외력은 이번 스텝이 소비했다** — 지운다. 힘은 프레임 단위 입력이고,
        // 남겨두면 호스트가 매 프레임 다시 주는 순간 두 배로 쌓인다(모든 엔진의 관례).
        for b in self.bodies.iter_mut() {
            b.force = Vec3::ZERO;
            b.torque = Vec3::ZERO;
        }

        // 이번 스텝에 **적분된 바디가 하나도 없으면** 자세가 그대로이므로 페어 집합도 그대로다.
        // (kinematic도 적분 대상이라 구동 중이면 자연히 무효화된다.)
        let moved = self.integrate_gate.iter().any(|x| *x);
        self.bp_pairs = pairs;
        self.bp_valid = !moved;
        self.bp_dt = dt;
        self.diag.post_ns[0] = _t_wb.ns();
        let _t_ev = Stopwatch::start();

        // 5-a) **접촉 이벤트** = 직전 기준선(`touching`)과 이번 라이브 집합의 차집합.
        //
        // 잠든 페어도 라이브에 남으므로 수면이 종료 이벤트를 거짓 발생시키지 않는다.
        // 순서는 전순서다(키 오름차순) — 이벤트 순서가 해시 순회에 의존하면 안 된다.
        //
        // 제한 모드: 라이브 집합은 "잔류한 비-fresh 기준선(위 잔류 논증) ∪ live_keys(fresh
        // 전용)"이고, 차집합은 **fresh 안에서만** 생길 수 있다 — Started 후보는 live_keys,
        // Stopped 후보는 touch_sorted(R접촉 기준선 키)뿐이다. 기준선도 통째 재구축 대신
        // 그 델타만 반영한다. 전체 경로와의 등가는 락스텝 궤적 테스트가 이벤트까지 잠근다.
        {
            // 용량은 스텝 시작에 선점됐다 — `clear` 뒤 `extend`는 재할당하지 않는다.
            started.clear();
            stopped.clear();
            started.extend(
                live_keys
                    .iter()
                    .filter(|k| !self.touching.contains(*k))
                    .copied(),
            );
            if narrow_restricted {
                stopped.extend(
                    touch_sorted
                        .iter()
                        .filter(|k| !live_keys.contains(*k))
                        .copied(),
                );
            } else {
                stopped.extend(
                    self.touching
                        .iter()
                        .filter(|k| !live_keys.contains(*k))
                        .copied(),
                );
            }
            started.sort_unstable();
            stopped.sort_unstable();
            // 기준선 갱신 — 다음 스텝의 이벤트가 이 집합과의 차집합이다.
            if narrow_restricted {
                for k in &stopped {
                    self.touching.remove(k);
                }
                for k in &started {
                    self.touching.insert(*k);
                }
            } else {
                // 전체 경로: live_keys가 곧 새 기준선 — 수천 키 재삽입 대신 스왑.
                std::mem::swap(&mut self.touching, &mut live_keys);
            }
            // **이 루프는 절대 재할당하지 않는다** — 용량은 스텝 시작의 fallible 블록에서
            // `ncap + touching.len()`으로 잡았고, 두 델타의 합은 그 상한을 넘을 수 없다
            // (같은 키가 started와 stopped에 동시에 들어가지 않는다: 하나는 live∖touching,
            // 다른 하나는 touching∖live다). 이 단언이 그 논증을 검사로 바꾼다 —
            // 넘으면 이벤트 구간에서 할당이 나고, 그건 wake 뒤라 되돌릴 수 없다.
            #[cfg(test)]
            {
                self.ev_cap_at_emit = self.contact_events.capacity();
            }
            debug_assert!(
                started.len() + stopped.len() <= self.contact_events.capacity(),
                "이벤트 출력 용량 부족: started {} + stopped {} > cap {}",
                started.len(),
                stopped.len(),
                self.contact_events.capacity()
            );
            for (k, kind) in started
                .drain(..)
                .map(|k| (k, ContactEventKind::Started))
                .chain(stopped.drain(..).map(|k| (k, ContactEventKind::Stopped)))
            {
                let (ai, bi) = (k.0 as usize, (k.1 & !STATIC_BIT) as usize);
                if ai >= self.bodies.len() || bi >= self.bodies.len() {
                    continue;
                }
                // **죽은 슬롯은 이벤트를 내지 않는다.** 슬롯이 재사용되면 `handle_of`가
                // 새 물체의 핸들을 만들어, 닿은 적 없는 물체에 Stopped가 나간다.
                // 제거는 호스트가 이미 아는 사실이므로 침묵이 옳다.
                if !self.alive[ai] || !self.alive[bi] {
                    continue;
                }
                self.contact_events.push(ContactEvent {
                    a: self.handle_of(ai),
                    b: self.handle_of(bi),
                    kind,
                });
            }
        }
        self.diag.post_ns[1] = _t_ev.ns();
        let _t_pcm = Stopwatch::start();

        // 5) PCM 캐시 갱신. **잠든 페어의 엔트리는 보존**한다 — 전체 clear는 첫 수면 스텝에
        //    지속 매니폴드를 날려, 나중에 깨어날 때 warm start 없이 거칠게 재정착한다(리뷰 지적).
        //    이번 스텝에 살아있지 않은(페어 소멸) 키만 폐기.
        //    제한 모드: live_keys는 fresh 전용이라 그대로 쓰면 dormant 엔트리가 전멸한다 —
        //    비-fresh 키는 잔류(캐시 ⊆ touching이고 비-fresh touching은 위에서 불변),
        //    fresh 키만 live_keys로 판정한다. 전체 경로는 스왑 후 touching == 새 라이브 집합.
        if narrow_restricted {
            self.contact_cache.retain(|k, _| {
                let r_touch = moved_mask[k.0 as usize]
                    || (k.1 & STATIC_BIT == 0 && moved_mask[(k.1 & !STATIC_BIT) as usize]);
                !r_touch || live_keys.contains(k)
            });
        } else {
            let touching = &self.touching;
            self.contact_cache.retain(|k, _| touching.contains(k));
        }
        // **캐시는 삽입 시점에 상태 정책으로 닫는다** — `close_state_policy`가 더는 캐시를
        // 순회하지 않는 근거다(그쪽 주석 참조). 포화는 멱등이라 예전처럼 "스텝 말에 닫는" 것과
        // 저장되는 값·`state_saturations` 카운트 모두 동일하다(예전에도 fixpoint 도달 후엔
        // hit이 나지 않았다). retain은 값을 바꾸지 않으므로 삽입만 닫으면 충분하다.
        let mut cache_sat_hits = 0usize;
        // **복합 페어는 같은 키의 제약이 여러 개다** — 첫 제약이 엔트리를 새로 만들고
        // 나머지는 append 병합한다(insert로 덮으면 앞 자식의 warm start가 소실된다).
        // feature가 자식쌍 네임스페이스로 유일하므로 병합 목록 안에서 충돌하지 않는다.
        let mut cache_written: IntSet<(u32, u32)> = IntSet::default();
        for (c, key) in constraints.iter().zip(&constraint_keys) {
            let first = cache_written.insert(*key);
            // 접점 → 캐시 엔트리 변환. **자리에서 채우기 위해** 클로저로 뽑아 뒀다:
            // 예전에는 `collect()`로 새 `Vec`을 만들어 `insert`했는데, 그러면 직전 스텝의
            // 엔트리가 통째로 해제되고 다음 스텝이 다시 할당한다 — 접촉 하나당 한 번,
            // 40단 피라미드에서 스텝당 2,387회다(설계 §6 "스텝 중 할당 0" 위반의 잔여분,
            // 할당 계측기가 ≤256B 버킷으로 지목했다). 살아남은 키의 `Vec`을 `clear` 후
            // 다시 채우면 정상 상태에서 할당이 사라진다.
            let mut make = |p: &solver::ContactPointConstraint| {
                use maru_collide::{saturate_state, saturate_state_vec};
                let mut sat = |v: f32| -> f32 {
                    let (v, changed) = saturate_state(v);
                    cache_sat_hits += changed as usize;
                    v
                };
                let normal = sat(p.normal_impulse);
                let tangent = [sat(p.tangent_impulse[0]), sat(p.tangent_impulse[1])];
                let tw =
                    c.tangents[0] * p.tangent_impulse[0] + c.tangents[1] * p.tangent_impulse[1];
                let (tangent_world, tw_hit) = saturate_state_vec(tw);
                let (position, pos_hit) = saturate_state_vec(p.position);
                cache_sat_hits += tw_hit as usize + pos_hit as usize;
                // 직렬화되는 필드는 하나도 빠뜨리지 않는다(리뷰 MAJOR): contact_normal이
                // NaN이면 restore가 실패한다. 단위 벡터라 포화가 아니라 정규화 실패 시
                // 0으로 되돌린다(매칭 게이트가 자연히 끊긴다).
                let contact_normal = if c.normal.try_normalize().is_none() {
                    cache_sat_hits += 1;
                    Vec3::ZERO
                } else {
                    c.normal
                };
                (
                    p.feature,
                    CachedImpulse {
                        normal,
                        tangent,
                        tangent_world,
                        position,
                        contact_normal,
                    },
                )
            };
            match self.contact_cache.get_mut(key) {
                // 살아남은 키 — 용량을 그대로 재사용한다. 같은 키의 첫 제약만 비우고,
                // 복합 페어의 나머지 제약은 **덧붙인다**(insert로 덮으면 앞 자식의
                // warm start가 소실된다 — 위 주석).
                Some(existing) => {
                    if first {
                        existing.clear();
                    }
                    existing.extend(c.points.iter().map(&mut make));
                    // **재사용 용량에 상한을 둔다** (교차검수 rev_8f75295a MAJOR).
                    // 자리에서 채우면 `Vec` 용량이 스텝 간 살아남는데, 예전에는 매 스텝
                    // 교체로 해제됐다. 복합 페어가 한 프레임에 자식 접점을 잔뜩 만들어
                    // 크게 자란 뒤 접점 하나만 계속 유지되면, 논리 길이는 작아져도
                    // **과거 최대치가 영구 잔류**한다 — 그것도 128 MiB 회계 밖에서.
                    // 여유 4배(+바닥 4)를 넘으면 되돌린다: 정상 상태에서는 절대 발화하지
                    // 않고(할당 0 유지), 한 번 부푼 키만 제자리로 돌아온다.
                    let cap_limit = existing.len().max(4).saturating_mul(4);
                    if existing.capacity() > cap_limit {
                        existing.shrink_to_fit();
                    }
                }
                // 새 키에서만 할당한다(이번 스텝에 새로 생긴 접촉).
                None => {
                    // `first == false`면 이번 스텝에 이미 이 키를 썼다는 뜻이라 맵에 있어야
                    // 한다 — 여기 오면 앞 제약의 엔트리를 **덮어써 warm start를 잃는다**.
                    // 도달 불가 논증을 검사로 바꿔 둔다(복합 페어의 병합 순서가 이 전제다).
                    debug_assert!(
                        first,
                        "같은 키를 이번 스텝에 이미 썼는데 캐시에 없다 — 병합이 깨진다"
                    );
                    let entry: Vec<(FeatureId, CachedImpulse)> =
                        c.points.iter().map(&mut make).collect();
                    self.contact_cache.insert(*key, entry);
                }
            }
        }
        self.diag.post_ns[2] = _t_pcm.ns();

        // 6) 아일랜드 union-find (동적-동적 edge만 — static은 anchor 전용, 설계 §4.3)
        //    + island 단위 일괄 sleep (설계 §4.6)
        let _t_isl = Stopwatch::start();
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
        let mut island_min: IntMap<usize, f32> = IntMap::default();
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
        self.diag.sub_ns[3] = _t_isl.ns();

        self.joints = joints;
        let _t_hyg = Stopwatch::start();

        // 7) denormal 위생 (설계 §8 flush 지점)
        // 잠든 바디는 건너뛴다 — 속도가 정확히 0이다(이번 스텝에 잠든 바디는 위 6절이 방금
        // 0으로 만들었고, 이전에 잠든 바디는 모든 변이 API가 wake를 거치므로 0인 채 불변,
        // 복원도 asleep ⇒ 속도 0을 강제한다). flush_tiny(0)=0이라 건너뜀은 항등이다.
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if self.alive[i] && b.is_dynamic() && !self.asleep[i] {
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
        self.clamp_velocities();
        self.enforce_world_bounds();
        // **접점 버퍼를 풀로 되돌린다.** 병합이 제약을 이 지역 벡터로 **옮겼기** 때문에,
        // 여기서 회수하지 않으면 스텝 끝 드롭에서 버퍼가 통째로 해제되고 다음 스텝이 다시
        // 수천 번 할당한다(설계 §6 "스텝 중 할당 0"). 참여자별로 나눠 담아 병렬 청크도
        // 자기 풀을 갖게 한다.
        let n_bufs = self.narrow_bufs.len().max(1);
        for (i, c) in constraints.drain(..).enumerate() {
            let buf = c.into_points_buf();
            // 풀은 제약 상한을 넘지 않는다 — 보유 용량은 아래 회계가 세고, 무한 성장은 막는다.
            if let Some(b) = self.narrow_bufs.get_mut(i % n_bufs) {
                if b.point_pool.len() < constraint_cap {
                    b.point_pool.push(buf);
                }
            }
        }
        // 캐시는 insert 시점에 닫았다(위 5절) — 그 hit도 이 스텝의 포화 카운트다.
        self.diag.state_saturations = cache_sat_hits + self.close_state_policy();
        self.diag.post_ns[3] = _t_hyg.ns();
        self.diag.phase_ns[4] = _t_post.ns();
    }
}

/// 그래프 컬러링의 색 상한 = **병렬 솔브의 패스당 배리어 상한**
/// (근거는 `solver::COLOR_CAP_DEFAULT` 문서).
///
/// `MARU_COLOR_CAP`으로 바꾼다. 프로세스당 1회만 읽는다 — 스텝 중간에 값이 바뀌면 제약
/// 순서가 스텝마다 달라져 결정론이 깨진다. **스레드 수에 의존해서는 안 된다**(같은 이유).
///
/// 이것이 측정 노브인 이유: 상한은 궤적을 바꾸므로(제약 순서가 바뀐다) 성능만 보고 정할 수
/// 없고 안정 봉투와 함께 재야 한다. 기본값은 오늘 동작(64 = 실질적 무제한)이다.
fn color_cap() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CACHED: AtomicU32 = AtomicU32::new(u32::MAX);
    let v = CACHED.load(Ordering::Relaxed);
    if v != u32::MAX {
        return v;
    }
    let cap = std::env::var("MARU_COLOR_CAP")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .map(|c| c.clamp(1, 64))
        .unwrap_or(solver::COLOR_CAP_DEFAULT);
    CACHED.store(cap, Ordering::Relaxed);
    cap
}

/// CCD 캐스트의 **공통 원점** — 두 부모 위치의 중점.
///
/// 캐스트는 이 원점을 뺀 상대 좌표로 돈다(충돌 기하는 병진 불변). 원점을 움직이는 바디에 두면
/// 상대 자세 하나가 두 부모 간격 **전체**를 떠안아, 유효한 후보가 디스패처의 `MAX_INPUT` 검사에
/// 걸린다(확인검수 rev_6f54ca63). 중점이면 각 상대 자세가 간격의 절반이다.
///
/// 간격의 상한: 후보는 스윕 AABB가 겹쳐야 하므로 축별 `|Δ| ≤ ha + hb + |motion|`이고,
/// 회전된 AABB의 반크기는 바운딩 반경(복합 최대 4464m = √3·1e3 오프셋 + √3·1e3 점 + 1e3 반경)을
/// 넘지 않는다. 즉 축별 ≤ 4464+4464+2500 = 11428m이고 벡터로는 최악 √3배지만, **중점을 쓰면
/// 각 상대 자세가 그 절반**이라 `MAX_INPUT`(10km) 안에 남는다.
fn ccd_cast_origin(a: Vec3, b: Vec3) -> Vec3 {
    a + (b - a) * 0.5
}

/// 테스트 전용 노출 — 원점 선택은 산술 성질이라 씬 대신 직접 고정한다(그 근거는 테스트 문서에).
/// `world` 모듈은 비공개라 `pub`만으로는 도달 불가 = dead code다(게이트가 잡았다).
#[cfg(test)]
#[doc(hidden)]
pub fn ccd_cast_origin_pub_for_test(a: Vec3, b: Vec3) -> Vec3 {
    ccd_cast_origin(a, b)
}

/// 셰이프의 월드 AABB — 자세를 명시 인자로 받는 정본 (body_aabb와 CCD 중간-서브스텝
/// 캐스트가 같은 정의를 공유한다: 후자는 솔버 스크래치의 현재 자세를 써야 한다).
fn shape_world_aabb(shape: &maru_collide::Shape, pose: &maru_collide::Pose) -> Aabb {
    // 월드 바디의 형상은 삽입·복원 경계에서 검증됐다 — 거절할 것이 없다.
    let local = shape
        .try_local_aabb()
        .expect("월드 바디는 삽입·복원 경계에서 검증된다");
    let c = (local.max + local.min) * 0.5;
    let hl = (local.max - local.min) * 0.5;
    let (ax, ay, az) = (
        pose.rot.rotate(Vec3::new(1.0, 0.0, 0.0)),
        pose.rot.rotate(Vec3::new(0.0, 1.0, 0.0)),
        pose.rot.rotate(Vec3::new(0.0, 0.0, 1.0)),
    );
    let ext = Vec3::new(
        ax.x.abs() * hl.x + ay.x.abs() * hl.y + az.x.abs() * hl.z,
        ax.y.abs() * hl.x + ay.y.abs() * hl.y + az.y.abs() * hl.z,
        ax.z.abs() * hl.x + ay.z.abs() * hl.y + az.z.abs() * hl.z,
    );
    Aabb::from_center(pose.pos + pose.rot.rotate(c), ext)
}

/// 두 물체가 충돌 그룹상 부딪히는가.
///
/// **양방향 조건**이다: 한 방향만 보면 "A는 B를 무시하는데 B는 A를 민다"가 되어
/// 운동량이 새고, 호스트가 설정한 적 없는 비대칭이 생긴다.
#[inline]
fn groups_collide(a: &Body, b: &Body) -> bool {
    (a.membership & b.filter) != 0 && (b.membership & a.filter) != 0
}

/// 내로우페이즈를 병렬로 돌리기 시작하는 페어 수.
///
/// dispatch 한 번은 워커를 깨우는 비용(뮤텍스+condvar broadcast, ~us)이 있고 페어 하나는
/// ~0.1-1us다. 페어가 적으면 깨우는 값이 일보다 비싸므로 그냥 직렬로 돈다 — 어느 쪽이든
/// 결과는 같으므로 이 문턱은 **성능 노브일 뿐 시맨틱이 아니다**.
const NARROW_PARALLEL_MIN: usize = 512;

/// 이번 스텝 내로우페이즈에 쓸 참여자 수.
fn narrow_participants(pool: &crate::threads::Pool, pairs: usize) -> usize {
    if pairs >= NARROW_PARALLEL_MIN {
        pool.threads().max(1)
    } else {
        1
    }
}

/// 참여자별 버퍼를 **wake 폐쇄 전에** 확보한다(실패는 상태 변경 없이 ResourceLimit).
/// `caps[w] = (제약 상한, 페어 상한)`.
fn reserve_narrow_bufs(
    bufs: &mut Vec<NarrowOut>,
    n_par: usize,
    caps: &[(usize, usize)],
) -> Result<(), ApiError> {
    if bufs.len() < n_par {
        bufs.try_reserve(n_par - bufs.len())
            .map_err(|_| ApiError::ResourceLimit)?;
        bufs.resize_with(n_par, NarrowOut::default);
    }
    for (b, &(c_cap, p_cap)) in bufs[..n_par].iter_mut().zip(caps) {
        b.reserve(c_cap, p_cap)?;
    }
    Ok(())
}

/// 내로우페이즈 한 청크가 읽는 것 전부 — 전부 **불변 참조**다.
///
/// 페어 하나의 처리가 순수하다는 사실(읽기는 이 구조체, 쓰기는 자기 `NarrowOut`)이
/// 병렬화의 근거이자 결정론의 근거다. 새 필드를 추가할 때 **가변 상태를 끼워 넣으면
/// 그 근거가 무너진다** — 누적이 필요하면 `NarrowOut`으로 간다.
struct NarrowCtx<'a> {
    bodies: &'a [Body],
    asleep: &'a [bool],
    oob: &'a [bool],
    cache: &'a IntMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
    touching: &'a IntSet<(u32, u32)>,
    touch_sorted: &'a [(u32, u32)],
    excluded: &'a IntSet<(u32, u32)>,
    restricted: bool,
    margin: f32,
    dt: f32,
}

/// 내로우페이즈 참여자 하나의 출력 버퍼. 스텝 사이에 **재사용**한다(할당 0).
#[derive(Default)]
pub(crate) struct NarrowOut {
    constraints: Vec<ContactConstraint>,
    keys: Vec<(u32, u32)>,
    /// 이번 청크에서 살아있는 접촉 키. 집합 삽입은 병합 때 직렬로 한다
    /// (해시 집합을 병렬로 만들면 순회 순서가 스레드 수에 의존한다).
    live: Vec<(u32, u32)>,
    max_penetration: f32,
    warm_total: usize,
    warm_hits: usize,
    /// 하이트필드 페어가 매니폴드 상한에서 끊긴 횟수 (진단 — 조용한 누락 금지).
    heightfield_saturated: usize,
    /// 셀 예산을 넘어 접촉 순회를 포기했다 — 이 스텝은 자원 오류다.
    budget_exceeded: bool,
    /// 접점 힙 할당 실패(극단 OOM) — 직렬 경로의 `bail`과 같은 뜻이다.
    bail: bool,
    /// **접점 버퍼 풀** — 스텝당 수천 회 할당을 없앤다(설계 §6 "스텝 중 할당 0").
    ///
    /// `reset`이 직전 스텝 제약의 버퍼를 여기 담고, `narrow_chunk`가 꺼내 쓴다. 스텝 말에
    /// 병합된 제약의 버퍼도 되돌려준다(`recycle_constraint_points`) — 병합이 제약을 지역
    /// 벡터로 **옮기기** 때문에, 그러지 않으면 스텝 끝 드롭에서 통째로 해제된다.
    /// 보유 용량은 `persistent_excess_bytes` 회계에 들어간다(상한 계약 유지).
    point_pool: Vec<Vec<solver::ContactPointConstraint>>,
}

impl NarrowOut {
    /// 용량 `c`(제약)·`p`(페어)를 가진 버퍼가 잡는 바이트.
    pub(crate) fn bytes_for(c: usize, p: usize) -> usize {
        c.saturating_mul(core::mem::size_of::<ContactConstraint>())
            .saturating_add(c.saturating_mul(core::mem::size_of::<(u32, u32)>()))
            .saturating_add(p.saturating_mul(core::mem::size_of::<(u32, u32)>()))
    }

    /// 이 버퍼가 예약 뒤에 **실제로 들고 있게 될** 바이트 — 영속 버퍼는 줄지 않으므로
    /// 현재 용량과 이번 목표 중 큰 쪽이다.
    pub(crate) fn projected_bytes(&self, c_cap: usize, p_cap: usize) -> usize {
        Self::bytes_for(
            self.constraints
                .capacity()
                .max(c_cap)
                .max(self.keys.capacity()),
            self.live.capacity().max(p_cap),
        )
    }

    /// 테스트 전용 — 예약이 실제로 목표 용량에 닿았는지 본다.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> (usize, usize, usize) {
        (
            self.constraints.capacity(),
            self.keys.capacity(),
            self.live.capacity(),
        )
    }

    /// 버퍼를 **해제**한다(축소가 아니라). 내용은 매 청크 시작에 `reset`되므로 버려도 되고,
    /// 용량은 `reserve_narrow_bufs`가 wake 이전에 fallible하게 다시 잡는다.
    /// 목표치를 남기지 않는 이유: 남기면 과거에 큰 청크를 맡았던 버퍼들이 각자 한 벌씩
    /// 쥔 채 회복하지 못한다(확인검수 rev_de34bbb5 MAJOR).
    fn release(&mut self) {
        self.constraints = Vec::new();
        self.keys = Vec::new();
        self.live = Vec::new();
    }

    /// 직전 사용량(len)의 4배를 넘게 쥐고 있으면 놓아준다 (스텝 시작 위생).
    fn trim_if_bloated(&mut self) {
        let bloated = |cap: usize, need: usize| cap > 4 * need.max(64);
        if bloated(self.constraints.capacity(), self.constraints.len())
            || bloated(self.keys.capacity(), self.keys.len())
            || bloated(self.live.capacity(), self.live.len())
        {
            self.release();
        }
    }

    fn reset(&mut self) {
        // 제약을 버리기 전에 접점 버퍼를 풀로 회수한다(용량 보존 — 다음 스텝이 재사용한다).
        for c in self.constraints.drain(..) {
            self.point_pool.push(c.into_points_buf());
        }
        self.constraints.clear();
        self.keys.clear();
        self.live.clear();
        self.max_penetration = 0.0;
        self.warm_total = 0;
        self.warm_hits = 0;
        self.heightfield_saturated = 0;
        self.budget_exceeded = false;
        self.bail = false;
    }

    /// `constraints`/`keys`는 제약 수, `live`는 페어 수 상한으로 잡는다.
    ///
    /// **`try_reserve`의 인자는 목표 용량이 아니라 `len`에 더할 여분이다** — 목표에서
    /// `capacity`를 빼서 넘기면, 비워진(len=0) 버퍼가 목표보다 작은 용량으로도 요청을
    /// "이미 충분"으로 통과시킨다(교차검수 rev_518b072b MAJOR: len=0·cap=150에 목표 200이면
    /// `try_reserve(50)`이 그냥 성공하고 151번째 push에서 wake 뒤 재할당이 난다).
    /// 그래서 **`len`을 뺀다**.
    pub(crate) fn reserve(&mut self, constraints: usize, pairs: usize) -> Result<(), ApiError> {
        self.constraints
            .try_reserve(constraints.saturating_sub(self.constraints.len()))
            .map_err(|_| ApiError::ResourceLimit)?;
        self.keys
            .try_reserve(constraints.saturating_sub(self.keys.len()))
            .map_err(|_| ApiError::ResourceLimit)?;
        self.live
            .try_reserve(pairs.saturating_sub(self.live.len()))
            .map_err(|_| ApiError::ResourceLimit)?;
        Ok(())
    }
}

/// 페어 청크 하나를 처리한다 — **직렬 경로와 병렬 경로가 공유하는 단일 정본**이다.
///
/// 두 경로가 각자 루프를 갖고 있으면 "스레드 1개면 예전과 같다"가 검증이 아니라 희망이
/// 된다. 하나만 두고, 병렬은 이 함수를 청크마다 부르는 것으로 정의한다.
///
/// `ti`는 기준선(`touch_sorted`) 병합 포인터의 **시작 위치**다. 청크의 첫 키까지
/// 미리 전진시켜 넣어야 직렬과 같은 판정이 나온다(둘 다 오름차순이라 이분 탐색으로 구한다).
fn narrow_chunk(cx: &NarrowCtx, pairs: &[(u32, u32)], mut ti: usize, out: &mut NarrowOut) {
    let empty: Vec<(FeatureId, CachedImpulse)> = Vec::new();
    // `active`의 초집합 술어 — 뚱뚱한 `Body`를 안 건드리고 Vec<bool> 두 개만 읽는다.
    let maybe_active = |i: usize| !cx.asleep[i] && !cx.oob[i];
    for &(a_id, b_raw) in pairs {
        if out.bail {
            return;
        }
        let key = (a_id, b_raw);
        let is_static_pair = b_raw & STATIC_BIT != 0;
        let b_plain = b_raw & !STATIC_BIT;
        // 기준선 병합 포인터를 이 페어까지 전진시킨다(전 페어 공통, 총 전진 = |touching|).
        // 제한 모드는 fresh만 걸으므로 병합 대신 아래 dormant 분기에서 해시 1회를 쓴다
        // (그 모드의 `touch_sorted`는 정렬돼 있지도 않다 — 전진시키면 안 된다).
        if !cx.restricted {
            while ti < cx.touch_sorted.len() && cx.touch_sorted[ti] < key {
                ti += 1;
            }
        }
        // **양쪽 다 비활성인 페어는 여기서 끝난다** (wake 폐쇄는 이미 위에서 닫혔다).
        // 움직이지 않은 페어는 새 접촉을 만들 수 없고, 브로드페이즈 AABB 겹침은 매니폴드의
        // 충분조건이 아니므로 **직전 기준선에 실제 접촉으로 있던 키만** 살린다 — 아니면
        // 잠들면서 거짓 Started가 난다. 원래는 페어마다 `touching`을 해시했는데(전원 수면
        // pile-3k에서 24810회/스텝), 두 정렬 배열의 병합이면 페어당 비교 1회로 같은 교집합이
        // 나온다. 술어는 wake 루프와 동일한 초집합이다.
        if !maybe_active(a_id as usize) && (is_static_pair || !maybe_active(b_plain as usize)) {
            let in_baseline = if cx.restricted {
                cx.touching.contains(&key)
            } else {
                ti < cx.touch_sorted.len() && cx.touch_sorted[ti] == key
            };
            if in_baseline {
                // 필터 순서는 활성 경로와 같다(제외 → 그룹 → 비활성). 특히 **그룹 판정이
                // 수면보다 먼저**여야 잠든 사이 그룹이 바뀐 페어가 접촉을 승계하지 않고
                // Stopped를 낸다(검수 rev_7f7bb07c MAJOR). 여기 도달하는 키는 ≤ 접촉 수라
                // 뚱뚱한 `Body` 접근도 그만큼만 일어난다.
                let excluded_pair = !cx.excluded.is_empty()
                    && cx
                        .excluded
                        .contains(&(a_id.min(b_plain), a_id.max(b_plain)));
                if !excluded_pair
                    && groups_collide(&cx.bodies[a_id as usize], &cx.bodies[b_plain as usize])
                {
                    push_live(out, key);
                }
            }
            continue;
        }
        // 조인트가 없으면 집합이 비어 있다 — 빈 집합에도 해시는 도는다.
        if !cx.excluded.is_empty()
            && cx
                .excluded
                .contains(&(a_id.min(b_plain), a_id.max(b_plain)))
        {
            continue;
        }
        let (a_id, b_id) = (a_id as usize, b_plain as usize);
        let a_inactive = cx.oob[a_id]
            || match cx.bodies[a_id].body_type {
                BodyType::Kinematic => false,
                BodyType::Dynamic => cx.asleep[a_id],
                BodyType::Static => true,
            };
        let b_inactive = cx.oob[b_id]
            || match cx.bodies[b_id].body_type {
                BodyType::Kinematic => false,
                BodyType::Dynamic => is_static_pair || cx.asleep[b_id],
                BodyType::Static => true,
            };
        // **그룹 판정이 수면 분기보다 먼저다.** 반대면 그룹이 명시적으로 충돌을
        // 거부한 페어도 잠든 동안 접촉으로 승격된다(검수 rev_7f7bb07c MAJOR).
        if !groups_collide(&cx.bodies[a_id], &cx.bodies[b_id]) {
            continue;
        }
        // 양쪽 비활성 페어는 위 조기 탈출에서 이미 걸러졌다. 두 술어는 **동치**다:
        // a는 항상 동적 트리 원소(Dynamic|Kinematic)라 `a_inactive == !maybe_active(a)`이고,
        // b는 static이면 `b_inactive == true == is_static_pair` 아니면 마찬가지다.
        // 동치가 깨지면 접촉 기준선 보존이 두 경로로 갈라지므로 여기서 잠근다.
        debug_assert!(
            !(a_inactive && b_inactive),
            "dormant 페어는 조기 탈출이 이미 걸렀어야 한다 — 술어 동치 붕괴"
        );
        let sensor_pair = cx.bodies[a_id].sensor || cx.bodies[b_id].sensor;
        // **복합 디스패치**: 자식쌍마다 독립 매니폴드 → 페어당 제약 여러 개(법선이 달라
        // 병합 불가). 같은 pair 키를 keys에 반복 push하는 것이 규약이고,
        // PCM 갱신(post 5절)이 같은 키의 엔트리를 append 병합한다. feature는 자식쌍 키
        // (1..=64, 비복합=0 미변형)를 bits 9..15에 접어 자식 간 warm start 혼선을 막는다.
        let is_compound = matches!(cx.bodies[a_id].shape, maru_collide::Shape::Compound { .. })
            || matches!(cx.bodies[b_id].shape, maru_collide::Shape::Compound { .. });
        let mut touched = false;
        // 지형 삼각형(내부 에지 마스크 보유)이면 월드 좌표 삼각형을 미리 풀어 둔다 —
        // 접점 법선 용접(§4.2 ghost edge)은 매니폴드 생성 직후, 소비(prepare) 전이다.
        // 지형은 항상 static = B 쪽이다(validate가 static 3점 hull에만 마스크 허용).
        let weld = match &cx.bodies[b_id].shape {
            maru_collide::Shape::Hull { points, radius }
                if cx.bodies[b_id].weld_edges != 0 && points.len() == 3 =>
            {
                let pb = &cx.bodies[b_id].pose;
                let t = |p: Vec3| pb.pos + pb.rot.rotate(p);
                Some((
                    [t(points[0]), t(points[1]), t(points[2])],
                    *radius,
                    cx.bodies[b_id].weld_edges,
                ))
            }
            _ => None,
        };
        let saturated = for_each_contact(
            &cx.bodies[a_id].shape,
            &cx.bodies[a_id].pose,
            &cx.bodies[b_id].shape,
            &cx.bodies[b_id].pose,
            cx.margin,
            &mut |ca, cb, mut m| {
                if out.bail || m.is_empty() {
                    return;
                }
                if let Some((tri, r, mask)) = weld {
                    maru_collide::weld_internal_edges(&mut m, tri, r, mask);
                }
                touched = true;
                if sensor_pair {
                    return; // 센서: 이벤트만 — 아래 touched 처리에서 live로
                }
                if is_compound {
                    m.namespace_features(ca as u32 * 8 + cb as u32 + 1);
                }
                let cached = cx.cache.get(&key).unwrap_or(&empty);
                for p in m.points() {
                    out.max_penetration = out.max_penetration.max(-p.separation);
                }
                // 접점 Vec 할당(≤4점) 실패는 극단 OOM — bail로 모아 호출부에서 처리.
                // 페어당 계측은 두지 않는다 — `Instant::now()` 두 번이 페어당
                // 실제 작업의 몇 %가 되어 계측기가 대상을 왜곡한다(2026-07-29 실측:
                // 40단에서 스텝당 ~0.2ms).
                //
                // **내역은 `boxsplit`과의 차로 귀속하지 마라 — 42% 틀린다**(2026-07-31 실측).
                // `boxsplit`은 미리 뽑아 둔 배열을 도는 마이크로벤치라 캐시가 계속 데워져 있고,
                // 실제 루프는 호출 사이에 뚱뚱한 `Body`·PCM 캐시·제약 버퍼를 훑어 그 캐시를
                // 밀어낸다. 같은 세션에서 boxsplit 1,402 us vs **실루프 1,988 us**였다.
                // 올바른 방법은 **한계비용**이다: 같은 호출을 한 번 더 돌리고 결과를 버린 뒤
                // `narrow` 증가분을 읽는다(대조군 solve가 ±1% 안일 때만 유효).
                let buf = out.point_pool.pop().unwrap_or_default();
                let Ok(c) = solver::prepare_into(cx.bodies, a_id, b_id, &m, cached, cx.dt, buf)
                else {
                    out.bail = true;
                    return;
                };
                // 진단은 **실제 warm load**를 센다(feature ID 일치만 세면 generic 접점의
                // 위치·법선 게이트가 매칭을 끊어도 100%로 보인다 — 거짓 지표).
                out.warm_total += c.points.len();
                out.warm_hits += c.warm_matched as usize;
                out.constraints.push(c);
                out.keys.push(key);
                push_live(out, key);
            },
        );
        match saturated {
            maru_collide::ContactWalk::Complete => {}
            maru_collide::ContactWalk::Saturated => out.heightfield_saturated += 1,
            // **예산 초과는 포화와 다르다** — 판정 자체를 못 했으므로 이 스텝의 접촉 집합이
            // 불완전하다. 포화로 뭉개면 "접촉 없음"으로 계속 진행해 뒤쪽 접촉이 사라진다.
            // 자원 오류로 올려 호스트가 장면을 줄이거나 격자를 성기게 하도록 한다.
            maru_collide::ContactWalk::BudgetExceeded => out.budget_exceeded = true,
            // 월드에 들어오는 형상은 삽입·복원에서 `validate()`를 통과했으므로 여기는
            // 도달 불가다. 그래도 `Complete`로 뭉개지 않는다 — 언젠가 검증 경로가 갈리면
            // "접촉 없음"으로 조용히 진행하는 대신 예산 초과와 같은 결말(스텝 포기)을 준다.
            maru_collide::ContactWalk::InvalidShape | maru_collide::ContactWalk::InvalidInput => {
                out.budget_exceeded = true
            }
        }
        // 센서 페어(또는 접촉만 있고 제약이 없는 경우의 이벤트 기준선).
        if touched && sensor_pair {
            push_live(out, key);
        }
    }
}

/// live 키는 **연속 중복만 접어** 넣는다 — 복합 페어가 제약을 여러 개 밀어도 키는 같으므로
/// 그대로 쌓으면 버퍼가 페어 수를 넘어 선점 용량을 깬다. 집합 삽입은 병합 때 하므로
/// 중복이 남아도 결과는 같지만, **용량 상한**이 근거를 잃는다.
#[inline]
fn push_live(out: &mut NarrowOut, key: (u32, u32)) {
    if out.live.last() != Some(&key) {
        out.live.push(key);
    }
}
