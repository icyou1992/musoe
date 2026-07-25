//! World — 고정 틱 스텝 오케스트레이션 (설계 §4 파이프라인, M2a: 단일스레드).
//!
//! step(dt): 브로드페이즈(canonical pairs) → 내로우페이즈(manifold) → prepare →
//! substep × N { 속도 적분 → warm start → solve(bias) → 위치 적분 → relax } →
//! restitution → (M2c: 임펄스 캐시 저장).
//! M2a 범위: 바디 제거·아일랜드·슬리핑·조인트는 후속 유닛. 커맨드 버퍼는 M2f와 함께.

use crate::body::{Body, BodyHandle, BodyType};
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
        }
    }

    pub fn add_body(&mut self, body: Body) -> BodyHandle {
        let index = self.bodies.len() as u32;
        self.bodies.push(body);
        self.generations.push(0);
        self.sleep_timer.push(0.0);
        self.asleep.push(false);
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

    pub fn body_mut(&mut self, h: BodyHandle) -> &mut Body {
        assert_eq!(
            self.generations[h.index as usize], h.generation,
            "stale handle"
        );
        &mut self.bodies[h.index as usize]
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_asleep(&self, h: BodyHandle) -> bool {
        self.asleep[h.index as usize]
    }

    fn wake(&mut self, i: usize) {
        self.asleep[i] = false;
        self.sleep_timer[i] = 0.0;
    }

    /// 고정 틱 스텝. 호스트 프레임과 무결합 (설계 §4).
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 || self.bodies.is_empty() {
            return;
        }
        let n_sub = self.params.substeps.max(1);
        let h = dt / n_sub as f32;
        let inv_h = 1.0 / h;
        let soft = Softness::new(
            // contact hertz는 서브스텝 주파수의 1/4을 넘지 않게 (Box2D v3 관행)
            self.params.solver.contact_hertz.min(0.25 * inv_h),
            self.params.solver.contact_damping,
            h,
        );

        // 1) 브로드페이즈 — canonical 페어 (트리 비의존 계약은 maru-collide가 보장)
        let mut dyn_items: Vec<(u32, Aabb)> = Vec::new();
        let mut static_items: Vec<(u32, Aabb)> = Vec::new();
        for (i, b) in self.bodies.iter().enumerate() {
            let local = b.shape.local_aabb();
            // [가정] 회전 보수 AABB: 로컬 AABB의 외접구 반경으로 팽창 (타이트화는 성능 유닛에서)
            let r = (local.max - local.min).length() * 0.5;
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
        let bp = Broadphase::build(dyn_items, static_items);
        let pairs = bp.canonical_pairs();

        // 2) 내로우페이즈 → 제약 준비 (A=첫 인덱스, B=둘째 — canonical 순서가 곧 해석 순서)
        //    PCM 캐시에서 feature 일치 접점의 임펄스를 warm load (설계 §3)
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut constraint_keys: Vec<(u32, u32)> = Vec::new();
        let empty: Vec<(FeatureId, CachedImpulse)> = Vec::new();
        for (a_id, b_raw) in pairs {
            let key = (a_id, b_raw);
            let is_static_pair = b_raw & STATIC_BIT != 0;
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
                    let cached = self.contact_cache.get(&key).unwrap_or(&empty);
                    constraints.push(solver::prepare(&self.bodies, a_id, b_id, &m, cached));
                    constraint_keys.push(key);
                }
            }
        }

        // 3) TGS-soft 서브스텝 루프
        let g = self.params.gravity;
        for _ in 0..n_sub {
            // 속도 적분 (sleeping 제외)
            for (i, b) in self.bodies.iter_mut().enumerate() {
                if b.is_dynamic() && !self.asleep[i] {
                    b.linvel += g * h;
                }
            }
            // warm start + bias 솔브
            for c in constraints.iter_mut() {
                solver::warm_start(&mut self.bodies, c);
            }
            for c in constraints.iter_mut() {
                solver::solve(
                    &mut self.bodies,
                    c,
                    inv_h,
                    Some((&soft, self.params.solver.max_bias_speed)),
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
        for c in &constraints {
            if self.bodies[c.a].is_dynamic() && self.bodies[c.b].is_dynamic() {
                let (ra, rb) = (find(&mut parent, c.a), find(&mut parent, c.b));
                // canonical: 낮은 인덱스가 대표 (설계 §4.3 결정론)
                if ra != rb {
                    let (lo, hi) = (ra.min(rb), ra.max(rb));
                    parent[hi] = lo;
                }
            }
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
