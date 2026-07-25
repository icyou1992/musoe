//! World — 고정 틱 스텝 오케스트레이션 (설계 §4 파이프라인, M2a: 단일스레드).
//!
//! step(dt): 브로드페이즈(canonical pairs) → 내로우페이즈(manifold) → prepare →
//! substep × N { 속도 적분 → warm start → solve(bias) → 위치 적분 → relax } →
//! restitution → (M2c: 임펄스 캐시 저장).
//! M2a 범위: 바디 제거·아일랜드·슬리핑·조인트는 후속 유닛. 커맨드 버퍼는 M2f와 함께.

use crate::body::{Body, BodyHandle, BodyType};
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
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            substeps: 4,
            speculative_margin: 0.05,
            solver: SolverParams::default(),
            sleep_linear_threshold: 0.03,
            sleep_angular_threshold: 0.05,
            time_to_sleep: 0.5,
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
}

pub struct World {
    pub params: WorldParams,
    bodies: Vec<Body>,
    /// 슬롯 생존 여부. 죽은 슬롯은 재사용 대기(세대가 이미 증가한 상태).
    alive: Vec<bool>,
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
    diag: Diagnostics,
}

/// 읽기 전용 진단 (엔진 캐노니컬 상태가 아니다 — 스냅샷·해시에 포함되지 않는다).
#[derive(Clone, Copy, Default, Debug)]
pub struct Diagnostics {
    pub warm_hits: usize,
    pub warm_total: usize,
    pub max_penetration: f32,
}

impl World {
    /// 파라미터 불변식을 생성 경계에서 강제한다 (리뷰 MAJOR: 무검증 public 필드가
    /// dt/substeps·Softness 분모를 NaN/Inf로 만들 수 있었다).
    pub fn new(params: WorldParams) -> Self {
        assert!(params.gravity.is_finite(), "gravity must be finite");
        assert!(
            (1..=64).contains(&params.substeps),
            "substeps must be in 1..=64, got {}",
            params.substeps
        );
        assert!(
            params.speculative_margin.is_finite() && params.speculative_margin >= 0.0,
            "speculative_margin must be finite and non-negative"
        );
        assert!(
            params.solver.contact_hertz.is_finite() && params.solver.contact_hertz > 0.0,
            "contact_hertz must be finite and positive"
        );
        assert!(
            params.solver.contact_damping.is_finite() && params.solver.contact_damping >= 0.0,
            "contact_damping must be finite and non-negative"
        );
        assert!(
            params.solver.max_bias_speed.is_finite() && params.solver.max_bias_speed > 0.0,
            "max_bias_speed must be finite and positive"
        );
        assert!(
            params.sleep_linear_threshold >= 0.0 && params.sleep_angular_threshold >= 0.0,
            "sleep thresholds must be non-negative"
        );
        Self {
            params,
            bodies: Vec::new(),
            alive: Vec::new(),
            generations: Vec::new(),
            contact_cache: HashMap::new(),
            sleep_timer: Vec::new(),
            asleep: Vec::new(),
            island: Vec::new(),
            joints: Vec::new(),
            diag: Diagnostics::default(),
        }
    }

    /// 바디 추가. 빈 슬롯이 있으면 **최소 인덱스**를 재사용한다(이력 무관 canonical 배치).
    pub fn add_body(&mut self, body: Body) -> BodyHandle {
        if let Some(index) = (0..self.bodies.len()).find(|&i| !self.alive[i]) {
            self.bodies[index] = body;
            self.alive[index] = true;
            self.sleep_timer[index] = 0.0;
            self.asleep[index] = false;
            self.island[index] = u32::MAX;
            return BodyHandle {
                index: index as u32,
                generation: self.generations[index],
            };
        }
        let index = self.bodies.len() as u32;
        self.bodies.push(body);
        self.alive.push(true);
        self.generations.push(0);
        self.sleep_timer.push(0.0);
        self.asleep.push(false);
        self.island.push(u32::MAX);
        BodyHandle {
            index,
            generation: 0,
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
            if j.body_a == i {
                neighbors.push(j.body_b);
            } else if j.body_b == i {
                neighbors.push(j.body_a);
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
        self.alive[i] = false;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.asleep[i] = false;
        self.island[i] = u32::MAX;
        // ③ 인접 정리
        self.joints.retain(|j| j.body_a != i && j.body_b != i);
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
        if !v.is_finite() {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].linvel = v;
        Ok(())
    }

    pub fn set_angular_velocity(&mut self, h: BodyHandle, v: Vec3) -> Result<(), ApiError> {
        if !v.is_finite() {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].angvel = v;
        Ok(())
    }

    /// 텔레포트 (설계 §9: 속도를 만들지 않는다 — kinematic 구동은 별도 API).
    pub fn set_pose(&mut self, h: BodyHandle, pose: maru_collide::Pose) -> Result<(), ApiError> {
        if !pose.pos.is_finite() || pose.rot.try_normalize().is_none() {
            return Err(ApiError::InvalidValue);
        }
        let i = self.mutate(h)?;
        self.bodies[i].pose = maru_collide::Pose {
            pos: pose.pos,
            rot: pose.rot.normalize(),
        };
        Ok(())
    }

    pub fn set_material(
        &mut self,
        h: BodyHandle,
        friction: f32,
        restitution: f32,
    ) -> Result<(), ApiError> {
        if !friction.is_finite()
            || friction < 0.0
            || !restitution.is_finite()
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

    /// 직전 스텝 진단 스냅샷.
    pub fn diagnostics(&self) -> Diagnostics {
        self.diag
    }

    pub fn is_asleep(&self, h: BodyHandle) -> bool {
        self.asleep[h.index as usize]
    }

    /// 조인트 추가. 연결 바디를 깨운다 (edge 생성형 변이 — 설계 §4.6).
    pub fn add_joint(&mut self, mut joint: Joint) -> usize {
        assert!(
            joint.body_a < self.bodies.len()
                && joint.body_b < self.bodies.len()
                && self.alive[joint.body_a]
                && self.alive[joint.body_b]
                && joint.body_a != joint.body_b,
            "joint endpoints must be distinct live bodies"
        );
        self.wake(joint.body_a);
        self.wake(joint.body_b);
        joint.linear_impulse = Vec3::ZERO;
        joint.angular_impulse = [0.0, 0.0]; // 재사용 Joint 값의 잔여 임펄스 차단(리뷰 지적)
        self.joints.push(joint);
        self.joints.len() - 1
    }

    /// 캐노니컬 상태 접근 (snapshot 모듈 전용).
    #[allow(clippy::type_complexity)]
    pub(crate) fn canonical_parts(
        &self,
    ) -> (
        &Vec<Body>,
        &Vec<bool>,
        &Vec<u32>,
        &Vec<f32>,
        &Vec<bool>,
        &Vec<u32>,
        &Vec<Joint>,
        &HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
    ) {
        (
            &self.bodies,
            &self.alive,
            &self.generations,
            &self.sleep_timer,
            &self.asleep,
            &self.island,
            &self.joints,
            &self.contact_cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_parts(
        &mut self,
        bodies: Vec<Body>,
        alive: Vec<bool>,
        generations: Vec<u32>,
        sleep_timer: Vec<f32>,
        asleep: Vec<bool>,
        island: Vec<u32>,
        joints: Vec<Joint>,
        cache: HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
    ) {
        self.alive = alive;
        self.generations = generations;
        self.bodies = bodies;
        self.sleep_timer = sleep_timer;
        self.asleep = asleep;
        self.island = island;
        self.joints = joints;
        self.contact_cache = cache;
    }

    /// 캐노니컬 스냅샷 (스텝 경계에서만 — 설계 §3). 트리는 제외(파생 상태).
    pub fn snapshot(&self) -> Vec<u8> {
        let mut w = crate::snapshot::Writer(Vec::new());
        crate::snapshot::write_world(self, &mut w);
        w.0
    }

    /// 스냅샷 복원. 재생은 원 실행과 같은 타깃에서 L0 비트 동일.
    /// **신뢰하지 않는 바이트를 받는다** — 검증 실패 시 월드는 그대로 두고 오류를 반환한다.
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), crate::snapshot::SnapshotError> {
        crate::snapshot::read_world(self, bytes)
    }

    /// L0 상태 해시 (리플레이 CI 비교).
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
        assert!(
            dt.is_finite(),
            "dt must be finite (NaN dt would poison all state)"
        );
        if dt <= 0.0 || self.bodies.is_empty() {
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
            // 회전 보수 AABB: pose 원점 중심이므로 반경은 **로컬 원점에서** 가장 먼 코너까지.
            // (max-min)/2로 잡으면 원점에서 벗어난 hull이 통째로 누락된다(리뷰 지적).
            let far = Vec3::new(
                local.min.x.abs().max(local.max.x.abs()),
                local.min.y.abs().max(local.max.y.abs()),
                local.min.z.abs().max(local.max.z.abs()),
            );
            let r = far.length();
            let world = Aabb::from_center(b.pose.pos, Vec3::new(r, r, r));
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
        self.diag = Diagnostics::default();
        let bp = Broadphase::build(dyn_items, static_items);
        let pairs = bp.canonical_pairs();
        // 조인트 연결 쌍은 충돌 제외 (collide_connected=false 기본)
        let excluded: std::collections::HashSet<(u32, u32)> = self
            .joints
            .iter()
            .filter(|j| !j.collide_connected)
            .map(|j| {
                let (a, b) = (j.body_a as u32, j.body_b as u32);
                (a.min(b), a.max(b))
            })
            .collect();

        // 1.5) **wake 폐쇄를 제약 생성보다 먼저 끝낸다** (리뷰 MAJOR).
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
            let active = |i: usize, this: &Self| match this.bodies[i].body_type {
                BodyType::Kinematic => true,
                BodyType::Dynamic => !this.asleep[i],
                BodyType::Static => false,
            };
            let sleeping_dyn =
                |i: usize, this: &Self| this.bodies[i].is_dynamic() && this.asleep[i];
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

        // 2) 내로우페이즈 → 제약 준비 (A=첫 인덱스, B=둘째 — canonical 순서가 곧 해석 순서)
        //    PCM 캐시에서 feature 일치 접점의 임펄스를 warm load (설계 §3)
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut constraint_keys: Vec<(u32, u32)> = Vec::new();
        let empty: Vec<(FeatureId, CachedImpulse)> = Vec::new();
        let mut live_keys: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
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
            let a_inactive = match self.bodies[a_id].body_type {
                BodyType::Kinematic => false,
                BodyType::Dynamic => self.asleep[a_id],
                BodyType::Static => true,
            };
            let b_inactive = match self.bodies[b_id].body_type {
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
                        self.diag.warm_total += 1;
                        self.diag.max_penetration = self.diag.max_penetration.max(-p.separation);
                        if cached.iter().any(|(f, _)| *f == p.feature) {
                            self.diag.warm_hits += 1;
                        }
                    }
                    constraints.push(solver::prepare(&self.bodies, a_id, b_id, &m, cached));
                    constraint_keys.push(key);
                    live_keys.insert(key);
                }
            }
        }

        // 2.5) 조인트: 한쪽이 깨어있으면 상대도 wake (설계 §4.3 조인트 edge)
        let joint_pairs: Vec<(usize, usize)> =
            self.joints.iter().map(|j| (j.body_a, j.body_b)).collect();
        for (a, b) in &joint_pairs {
            let a_awake = self.bodies[*a].is_dynamic() && !self.asleep[*a];
            let b_awake = self.bodies[*b].is_dynamic() && !self.asleep[*b];
            if a_awake && self.bodies[*b].is_dynamic() && self.asleep[*b] {
                self.wake(*b);
            }
            if b_awake && self.bodies[*a].is_dynamic() && self.asleep[*a] {
                self.wake(*a);
            }
        }

        // 3) TGS-soft 서브스텝 루프
        let g = self.params.gravity;
        // 조인트는 접촉보다 강성 (Box2D v3: 2×contact hertz, ζ=2)
        let joint_soft = Softness::new(
            (2.0 * self.params.solver.contact_hertz).min(0.25 * inv_h),
            2.0,
            h,
        );
        let mut joints = std::mem::take(&mut self.joints);
        // 조인트도 접촉과 같은 비활성 게이트 (설계 §4.6 level-trigger): 양쪽 모두
        // 비활성(sleeping/static/kinematic)이면 스킵 — 잠든 바디에 warm-start 임펄스를
        // 주입하면 위치 적분이 없는 채 속도만 펌핑돼 정착한 체인이 되살아난다(실증:
        // t=6s vmax 0.006 정착 → 슬립 → t=8s 4.5 폭주).
        let active_joints: Vec<usize> = joints
            .iter()
            .enumerate()
            .filter(|(_, j)| {
                let awake = |i: usize| self.bodies[i].is_dynamic() && !self.asleep[i];
                awake(j.body_a) || awake(j.body_b)
            })
            .map(|(i, _)| i)
            .collect();
        for _ in 0..n_sub {
            // 속도 적분 (sleeping 제외)
            for (i, b) in self.bodies.iter_mut().enumerate() {
                if self.alive[i] && b.is_dynamic() && !self.asleep[i] {
                    b.linvel += g * h;
                }
            }
            // warm start + bias 솔브 (조인트 먼저 — Box2D 순서)
            for &ji in &active_joints {
                joint::warm_start(&mut self.bodies, &joints[ji]);
            }
            for c in constraints.iter_mut() {
                solver::warm_start(&mut self.bodies, c);
            }
            for &ji in &active_joints {
                joint::solve(&mut self.bodies, &mut joints[ji], Some(&joint_soft));
            }
            for c in constraints.iter_mut() {
                let sf = if self.bodies[c.a].is_dynamic() && self.bodies[c.b].is_dynamic() {
                    &soft
                } else {
                    &static_soft
                };
                solver::solve(
                    &mut self.bodies,
                    c,
                    inv_h,
                    Some((sf, self.params.solver.max_bias_speed)),
                );
            }
            // 위치 적분 (dynamic + kinematic, sleeping 제외)
            for (i, b) in self.bodies.iter_mut().enumerate() {
                if self.alive[i] && b.body_type != BodyType::Static && !self.asleep[i] {
                    b.pose.pos += b.linvel * h;
                    b.pose.rot = b.pose.rot.integrate(b.angvel, h);
                }
            }
            // relax — bias 에너지 제거
            for &ji in &active_joints {
                joint::solve(&mut self.bodies, &mut joints[ji], None);
            }
            for c in constraints.iter_mut() {
                solver::solve(&mut self.bodies, c, inv_h, None);
            }
        }

        // 4) restitution 패스 (임팩트 게이트)
        for c in constraints.iter_mut() {
            solver::apply_restitution(
                &mut self.bodies,
                c,
                self.params.solver.restitution_threshold,
            );
        }

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
                        },
                    )
                })
                .collect();
            self.contact_cache.insert(*key, entry);
        }

        // 6) 아일랜드 union-find (동적-동적 edge만 — static은 anchor 전용, 설계 §4.3)
        //    + island 단위 일괄 sleep (설계 §4.6)
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();
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
            if !self.alive[i] || !self.bodies[i].is_dynamic() {
                self.island[i] = u32::MAX;
            } else if !self.asleep[i] {
                self.island[i] = find(&mut parent, i) as u32;
            }
        }
        let lin2 = self.params.sleep_linear_threshold * self.params.sleep_linear_threshold;
        let ang2 = self.params.sleep_angular_threshold * self.params.sleep_angular_threshold;
        for i in 0..n {
            if self.alive[i] && self.bodies[i].is_dynamic() && !self.asleep[i] {
                let low = self.bodies[i].linvel.length_sq() < lin2
                    && self.bodies[i].angvel.length_sq() < ang2;
                self.sleep_timer[i] = if low { self.sleep_timer[i] + dt } else { 0.0 };
            }
        }
        // island 전원이 임계 시간을 넘겨야 일괄 sleep (부분 sleep 금지)
        let mut island_min: HashMap<usize, f32> = HashMap::new();
        for i in 0..n {
            if self.alive[i] && self.bodies[i].is_dynamic() && !self.asleep[i] {
                let root = find(&mut parent, i);
                let e = island_min.entry(root).or_insert(f32::INFINITY);
                *e = e.min(self.sleep_timer[i]);
            }
        }
        for i in 0..n {
            if self.alive[i] && self.bodies[i].is_dynamic() && !self.asleep[i] {
                let root = find(&mut parent, i);
                if island_min[&root] >= self.params.time_to_sleep {
                    self.asleep[i] = true;
                    self.bodies[i].linvel = Vec3::ZERO;
                    self.bodies[i].angvel = Vec3::ZERO;
                }
            }
        }

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
    }
}
