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

pub struct World {
    pub params: WorldParams,
    bodies: Vec<Body>,
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
    /// [진단] 직전 스텝의 warm-start 적중/전체 접점 수, 최대 침투(음의 separation).
    pub warm_hits: usize,
    pub warm_total: usize,
    pub max_penetration: f32,
}

impl World {
    pub fn new(params: WorldParams) -> Self {
        Self {
            params,
            bodies: Vec::new(),
            generations: Vec::new(),
            contact_cache: HashMap::new(),
            sleep_timer: Vec::new(),
            asleep: Vec::new(),
            island: Vec::new(),
            joints: Vec::new(),
            warm_hits: 0,
            warm_total: 0,
            max_penetration: 0.0,
        }
    }

    pub fn add_body(&mut self, body: Body) -> BodyHandle {
        let index = self.bodies.len() as u32;
        self.bodies.push(body);
        self.generations.push(0);
        self.sleep_timer.push(0.0);
        self.asleep.push(false);
        self.island.push(u32::MAX);
        BodyHandle {
            index,
            generation: 0,
        }
    }

    pub fn body(&self, h: BodyHandle) -> &Body {
        assert_eq!(
            self.generations[h.index as usize], h.generation,
            "stale handle"
        );
        &self.bodies[h.index as usize]
    }

    /// 가변 접근. **공개 변이는 edge/접촉 상태를 바꾸므로 반드시 깨우고 stale PCM을 버린다**
    /// (리뷰 지적: 잠든 바디에 속도를 써도 계속 잠들어 적분이 생략됐다).
    /// static은 설계 §9상 불변 — kinematic을 쓸 것.
    pub fn body_mut(&mut self, h: BodyHandle) -> &mut Body {
        let i = h.index as usize;
        assert_eq!(self.generations[i], h.generation, "stale handle");
        assert!(
            self.bodies[i].body_type != BodyType::Static,
            "static bodies are immutable (design §9) — use Kinematic"
        );
        self.wake(i);
        let id = i as u32;
        self.contact_cache
            .retain(|(a, b), _| *a != id && (*b & !STATIC_BIT) != id);
        &mut self.bodies[i]
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_asleep(&self, h: BodyHandle) -> bool {
        self.asleep[h.index as usize]
    }

    /// 조인트 추가. 연결 바디를 깨운다 (edge 생성형 변이 — 설계 §4.6).
    pub fn add_joint(&mut self, mut joint: Joint) -> usize {
        // 핸들 인덱스 검증은 v1 단순화: usize 인덱스 직접 사용 (BodyHandle API는 M3 바인딩에서)
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
        &Vec<f32>,
        &Vec<bool>,
        &Vec<u32>,
        &Vec<Joint>,
        &HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
    ) {
        (
            &self.bodies,
            &self.sleep_timer,
            &self.asleep,
            &self.island,
            &self.joints,
            &self.contact_cache,
        )
    }

    pub(crate) fn restore_parts(
        &mut self,
        bodies: Vec<Body>,
        sleep_timer: Vec<f32>,
        asleep: Vec<bool>,
        island: Vec<u32>,
        joints: Vec<Joint>,
        cache: HashMap<(u32, u32), Vec<(FeatureId, CachedImpulse)>>,
    ) {
        self.generations = vec![0; bodies.len()];
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
    pub fn restore(&mut self, bytes: &[u8]) {
        crate::snapshot::read_world(self, bytes);
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
            if self.island[k] == island && self.bodies[k].is_dynamic() {
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
        self.warm_hits = 0;
        self.warm_total = 0;
        self.max_penetration = 0.0;
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

        // 2) 내로우페이즈 → 제약 준비 (A=첫 인덱스, B=둘째 — canonical 순서가 곧 해석 순서)
        //    PCM 캐시에서 feature 일치 접점의 임펄스를 warm load (설계 §3)
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut constraint_keys: Vec<(u32, u32)> = Vec::new();
        let empty: Vec<(FeatureId, CachedImpulse)> = Vec::new();
        for (a_id, b_raw) in pairs {
            let key = (a_id, b_raw);
            let is_static_pair = b_raw & STATIC_BIT != 0;
            // 조인트 연결 쌍 충돌 제외 — 정적 상대 포함 (앵커 바디와의 접촉이 조인트와 맞서며
            // 영구 에너지 주입 → link0 발산 실증). 비교는 STATIC_BIT를 벗긴 실제 인덱스로.
            let b_plain = b_raw & !STATIC_BIT;
            if excluded.contains(&(a_id.min(b_plain), a_id.max(b_plain))) {
                continue;
            }
            let b_id = (b_raw & !STATIC_BIT) as usize;
            let a_id = a_id as usize;
            // level-trigger(설계 §4.6): 양쪽 모두 비활성(sleeping/static)이면 스킵.
            // 한쪽이 깨어있는 동적이면 잠든 상대를 깨워 정상 해석.
            let a_asleep = self.asleep[a_id];
            let b_asleep = if is_static_pair {
                true
            } else {
                self.asleep[b_id]
            };
            if a_asleep && b_asleep {
                continue;
            }
            let (sa, pa) = (&self.bodies[a_id].shape, &self.bodies[a_id].pose);
            let (sb, pb) = (&self.bodies[b_id].shape, &self.bodies[b_id].pose);
            if let Some(m) = contact(sa, pa, sb, pb, self.params.speculative_margin) {
                if !m.points.is_empty() {
                    if a_asleep {
                        self.wake(a_id);
                    }
                    if !is_static_pair && self.asleep[b_id] {
                        self.wake(b_id);
                    }
                    // 방어선: wake 후에도 잠든 동적 바디가 남아 있으면 제약을 만들지 않는다
                    // (위치 적분 없이 속도만 펌핑되는 경로 차단 — 리뷰 BLOCKER)
                    let a_still_asleep = self.bodies[a_id].is_dynamic() && self.asleep[a_id];
                    let b_still_asleep =
                        !is_static_pair && self.bodies[b_id].is_dynamic() && self.asleep[b_id];
                    if a_still_asleep || b_still_asleep {
                        continue;
                    }
                    let cached = self.contact_cache.get(&key).unwrap_or(&empty);
                    for p in &m.points {
                        self.warm_total += 1;
                        if -p.separation > 1.0 && self.max_penetration <= 1.0 {
                            let (qa, qb) = (self.bodies[a_id].pose.rot, self.bodies[b_id].pose.rot);
                            let g = maru_collide::gjk(
                                &self.bodies[a_id].shape,
                                &self.bodies[a_id].pose,
                                &self.bodies[b_id].shape,
                                &self.bodies[b_id].pose,
                            );
                            eprintln!(
                                "   GJK={}  shapes a={:?} b={:?}",
                                match g {
                                    maru_collide::GjkResult::Separated {
                                        distance, normal, ..
                                    } => format!("Sep d={distance:e} n={normal:?}"),
                                    maru_collide::GjkResult::Intersecting(_) =>
                                        "Intersecting".to_string(),
                                },
                                self.bodies[a_id].shape,
                                self.bodies[b_id].shape
                            );
                            eprintln!(
                                "DEEP pa=({:?},{},{},{},{}) pb=({:?},{},{},{},{}) n={:?} sep={}",
                                self.bodies[a_id].pose.pos,
                                qa.x,
                                qa.y,
                                qa.z,
                                qa.w,
                                self.bodies[b_id].pose.pos,
                                qb.x,
                                qb.y,
                                qb.z,
                                qb.w,
                                m.normal,
                                p.separation
                            );
                        }
                        self.max_penetration = self.max_penetration.max(-p.separation);
                        if cached.iter().any(|(f, _)| *f == p.feature) {
                            self.warm_hits += 1;
                        }
                    }
                    constraints.push(solver::prepare(&self.bodies, a_id, b_id, &m, cached));
                    constraint_keys.push(key);
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
                if b.is_dynamic() && !self.asleep[i] {
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
                if b.body_type != BodyType::Static && !self.asleep[i] {
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

        // 5) PCM 캐시 갱신 — 이번 스텝의 최종 임펄스를 feature 키로 저장 (사라진 페어는 폐기)
        self.contact_cache.clear();
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
        // 아일랜드 대표를 다음 스텝의 wake 전파용으로 저장 (동적 바디만)
        for i in 0..n {
            self.island[i] = if self.bodies[i].is_dynamic() {
                find(&mut parent, i) as u32
            } else {
                u32::MAX
            };
        }
        let lin2 = self.params.sleep_linear_threshold * self.params.sleep_linear_threshold;
        let ang2 = self.params.sleep_angular_threshold * self.params.sleep_angular_threshold;
        for i in 0..n {
            if self.bodies[i].is_dynamic() && !self.asleep[i] {
                let low = self.bodies[i].linvel.length_sq() < lin2
                    && self.bodies[i].angvel.length_sq() < ang2;
                self.sleep_timer[i] = if low { self.sleep_timer[i] + dt } else { 0.0 };
            }
        }
        // island 전원이 임계 시간을 넘겨야 일괄 sleep (부분 sleep 금지)
        let mut island_min: HashMap<usize, f32> = HashMap::new();
        for i in 0..n {
            if self.bodies[i].is_dynamic() && !self.asleep[i] {
                let root = find(&mut parent, i);
                let e = island_min.entry(root).or_insert(f32::INFINITY);
                *e = e.min(self.sleep_timer[i]);
            }
        }
        for i in 0..n {
            if self.bodies[i].is_dynamic() && !self.asleep[i] {
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
        for b in self.bodies.iter_mut() {
            if b.is_dynamic() {
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
