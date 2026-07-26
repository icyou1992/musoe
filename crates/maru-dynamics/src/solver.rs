//! TGS-soft 접촉 솔버 (설계 §4.4 — Box2D v3 "Soft Step" 정식화).
//!
//! 서브스텝마다: 속도 적분 → warm start → soft bias 임펄스 해석 → 위치 적분 → relax(무bias).
//! 전 서브스텝 후 restitution 패스 1회 — **임팩트 게이트**(설계 §4.4): 접촉 생성 시점의
//! 접근 속도(approach_vn)가 임계 이상일 때만, 실제 임펄스가 발생한 접점에만 발화.
//! [근거: Solver2D 실증 + Box2D v3 releasing-3.0]

use crate::body::Body;
use maru_collide::Manifold;
use maru_math::{Mat3, Quat, Vec3};

pub struct SolverParams {
    pub contact_hertz: f32,
    pub contact_damping: f32,
    /// 침투 해소 최대 속도 (m/s) — 폭발 방지.
    pub max_bias_speed: f32,
    pub restitution_threshold: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            contact_hertz: 30.0,
            contact_damping: 10.0,
            max_bias_speed: 3.0,
            restitution_threshold: 1.0,
        }
    }
}

/// soft constraint 계수 (Box2D v3 공식).
#[derive(Clone, Copy)]
pub struct Softness {
    pub bias_rate: f32,
    pub mass_scale: f32,
    pub impulse_scale: f32,
}

impl Softness {
    pub fn new(hertz: f32, zeta: f32, h: f32) -> Self {
        if hertz <= 0.0 {
            return Self {
                bias_rate: 0.0,
                mass_scale: 1.0,
                impulse_scale: 0.0,
            };
        }
        let omega = 2.0 * core::f32::consts::PI * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

pub struct ContactPointConstraint {
    /// PCM 캐시 매칭 키 (설계 §4.2 canonical feature ID — M1 클리핑 계보).
    pub feature: maru_collide::FeatureId,
    /// prepare 시점 접점 월드 위치 (generic 접점 근접 매칭용).
    pub position: Vec3,
    /// 질량중심 기준 로컬 앵커 (서브스텝마다 현재 자세로 재회전 — TGS의 위치 추적).
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// prepare 시점 separation (앵커 동일 월드점 기준 오프셋).
    pub base_separation: f32,
    pub normal_mass: f32,
    pub tangent_mass: [f32; 2],
    pub normal_impulse: f32,
    pub tangent_impulse: [f32; 2],
    pub max_normal_impulse: f32,
    /// 이번 서브스텝의 월드 레버암 — `refresh_anchors`가 채운다.
    /// warm start와 bias 솔브는 같은 자세를 보므로 회전을 한 번만 한다
    /// (전에는 warm_start·solve·relax가 각각 다시 돌려 서브스텝당 3중 계산이었다).
    pub ra: Vec3,
    pub rb: Vec3,
    /// 접촉 생성 시점 법선 접근 속도 (restitution 게이트 입력).
    pub approach_vn: f32,
}

pub struct ContactConstraint {
    pub a: usize,
    pub b: usize,
    /// A를 +normal로 밀면 분리 (maru-collide 관례).
    pub normal: Vec3,
    pub tangents: [Vec3; 2],
    pub friction: f32,
    pub restitution: f32,
    pub points: Vec<ContactPointConstraint>,
    /// 이번 스텝에 **실제로 warm load된** 접점 수 (진단 전용, 캐노니컬 상태 아님).
    pub warm_matched: u32,
}

/// 법선에서 결정적 접선 기저.
fn tangent_basis(n: Vec3) -> [Vec3; 2] {
    let t1 = if n.x.abs() > 0.707 {
        Vec3::new(n.y, -n.x, 0.0).normalize_or_zero()
    } else {
        Vec3::new(0.0, n.z, -n.y).normalize_or_zero()
    };
    [t1, n.cross(t1)]
}

fn effective_mass(
    inv_m_a: f32,
    inv_i_a: &Mat3,
    ra: Vec3,
    inv_m_b: f32,
    inv_i_b: &Mat3,
    rb: Vec3,
    dir: Vec3,
) -> f32 {
    let rna = ra.cross(dir);
    let rnb = rb.cross(dir);
    let k = inv_m_a + inv_m_b + rna.dot(inv_i_a.mul_vec(rna)) + rnb.dot(inv_i_b.mul_vec(rnb));
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// 솔버 핫패스의 **압축 상태**(설계 §3: SoA 핫패스).
///
/// 솔버가 `Body` 배열을 직접 훑으면 셰이프·재질까지 딸린 뚱뚱한 구조체를 제약 순서대로
/// 무작위 접근하게 된다. 실제로 점당 73ns가 나왔는데 산술은 40 flops 남짓이었다 —
/// 비용은 계산이 아니라 **메모리**였다. 솔버가 실제로 읽고 쓰는 6개 배열만 뽑아
/// 연속 메모리에 두고, 서브스텝 루프가 끝난 뒤 한 번만 바디로 되쓴다.
#[derive(Default)]
pub struct SolverState {
    pub lin: Vec<Vec3>,
    pub ang: Vec<Vec3>,
    pub pos: Vec<Vec3>,
    pub rot: Vec<Quat>,
    pub inv_m: Vec<f32>,
    pub inv_i: Vec<Mat3>,
    /// 동적 바디만 임펄스를 받는다 (static/kinematic은 inv_m=0이지만 명시 게이트가 더 싸다).
    pub dynamic: Vec<bool>,
}

impl SolverState {
    /// 바디에서 상태를 **제자리로** 채운다. 매 스텝 새 Vec을 만들면 할당 비용이
    /// 솔버 이득을 먹는다(실측: SoA 전환이 되레 1.3ms 느려졌다) — 버퍼는 월드가 보관한다.
    /// 용량만 미리 확보한다 — **캐노니컬 변이 전에** 호출해 실패를 여기서 끝낸다.
    pub fn reserve_for(&mut self, n: usize) -> Result<(), crate::world::ApiError> {
        macro_rules! res {
            ($v:expr) => {
                $v.try_reserve(n.saturating_sub($v.capacity()))
                    .map_err(|_| crate::world::ApiError::ResourceLimit)?;
            };
        }
        res!(self.lin);
        res!(self.ang);
        res!(self.pos);
        res!(self.rot);
        res!(self.inv_m);
        res!(self.inv_i);
        res!(self.dynamic);
        Ok(())
    }

    /// 용량이 확보된 상태에서 값을 채운다 (실패하지 않는다).
    pub fn fill_from(&mut self, bodies: &[Body], alive: &[bool], oob: &[bool]) {
        let n = bodies.len();
        macro_rules! fit {
            ($v:expr, $d:expr) => {
                $v.clear();
                $v.resize(n, $d);
            };
        }
        fit!(self.lin, Vec3::ZERO);
        fit!(self.ang, Vec3::ZERO);
        fit!(self.pos, Vec3::ZERO);
        fit!(self.rot, Quat::IDENTITY);
        fit!(self.inv_m, 0.0);
        fit!(self.inv_i, Mat3::from_diagonal(Vec3::ZERO));
        fit!(self.dynamic, false);
        for i in 0..n {
            let b = &bodies[i];
            // **OOB는 솔버에서도 비활성**이다(리뷰 BLOCKER: `dynamic`을 타입만 보고 정해서
            // OOB에 임펄스가 들어가고 그 속도가 바디로 되쓰였다). 적분 게이트만으로는 부족하다.
            let active = alive[i] && !oob[i];
            self.lin[i] = if active { b.linvel } else { Vec3::ZERO };
            self.ang[i] = if active { b.angvel } else { Vec3::ZERO };
            self.pos[i] = b.pose.pos;
            self.rot[i] = b.pose.rot;
            self.inv_m[i] = if active { b.inv_mass } else { 0.0 };
            self.dynamic[i] = active && b.is_dynamic();
        }
    }

    /// 솔버 결과를 바디로 되쓴다 (스텝당 1회 — 핫패스 밖).
    pub fn write_back(&self, bodies: &mut [Body], alive: &[bool]) {
        for i in 0..bodies.len() {
            if alive[i] {
                bodies[i].linvel = self.lin[i];
                bodies[i].angvel = self.ang[i];
                bodies[i].pose.pos = self.pos[i];
                bodies[i].pose.rot = self.rot[i];
            }
        }
    }

    #[inline]
    fn vel_at(&self, i: usize, r: Vec3) -> Vec3 {
        self.lin[i] + self.ang[i].cross(r)
    }
    #[inline]
    fn apply(&mut self, i: usize, imp: Vec3, r: Vec3) {
        if self.dynamic[i] {
            self.lin[i] += imp * self.inv_m[i];
            self.ang[i] += self.inv_i[i].mul_vec(r.cross(imp));
        }
    }
    #[inline]
    fn apply_ang(&mut self, i: usize, ang: Vec3) {
        if self.dynamic[i] {
            self.ang[i] += self.inv_i[i].mul_vec(ang);
        }
    }
    #[inline]
    pub fn vel_at_pub(&self, i: usize, r: Vec3) -> Vec3 {
        self.vel_at(i, r)
    }
    #[inline]
    pub fn apply_pub(&mut self, i: usize, imp: Vec3, r: Vec3) {
        self.apply(i, imp, r)
    }
    #[inline]
    pub fn apply_ang_pub(&mut self, i: usize, ang: Vec3) {
        self.apply_ang(i, ang)
    }
    /// **활성 바디만** 갱신한다 — 잠든 바디는 자세가 변하지 않아 값이 그대로다.
    pub fn refresh_inv_i_active(&mut self, local: &[Vec3], active: &[u32]) {
        for &i in active {
            let i = i as usize;
            if self.dynamic[i] {
                self.inv_i[i] = Mat3::rotate_inertia(local[i], self.rot[i]);
            }
        }
    }
}

/// 캐시된 접점 임펄스 (스텝 간 warm start — 설계 §3 PCM).
/// generic witness 접점의 warm start 매칭 규칙 (리뷰 PARTIAL MAJOR).
///
/// 위치 근접 **하나만** 보면 양방향으로 틀린다: 기하 척도만 쓰면 반두께 1cm 판의 허용치가
/// 2.5mm라 60Hz·1m/s 정상 슬라이딩(프레임당 16.7mm)에서 매 프레임 warm start가 끊기고,
/// 반대로 큰 반지름 곡면에서는 같은 비율이 다른 feature로의 점프를 통과시킨다.
/// 그래서 세 가지를 함께 쓴다:
/// - **기하 바닥** = 최소 특징 크기의 이 비율 (정지 접촉의 수치 흔들림 흡수)
const WARM_MATCH_GEO: f32 = 0.25;
/// - **운동 항** = 접점이 이번 스텝에 **정당하게 움직일 수 있는 상한** × 여유.
///   상대 슬립 속도를 쓰면 안 된다: 구가 구르면 슬립은 0인데 기하 접점은 구의 선속도만큼
///   월드에서 이동한다(실측: 슬립 기준으로는 계승률 0.01). 상한은 두 바디의 선속도와
///   회전이 접점에 만드는 이동량의 합이다.
const WARM_MATCH_MOTION: f32 = 1.0;
/// - **법선 게이트** = 캐시된 법선과의 정렬. 위치가 가까워도 접촉 면이 바뀌었으면
///   다른 접촉이다 — 곡면에서 feature가 점프해도 법선은 크게 돈다.
const WARM_MATCH_COS: f32 = 0.9;

#[derive(Clone, Copy, Default)]
pub struct CachedImpulse {
    pub normal: f32,
    /// 이전 기저 성분 (진단·직렬화용).
    pub tangent: [f32; 2],
    /// **월드 공간 마찰 임펄스**. 접선 기저는 법선에서 유도되므로(0.707 분기) 법선이 회전하면
    /// 같은 성분이 다른 월드 방향으로 재적용된다 — 월드 벡터를 저장해 새 접평면에 투영한다.
    pub tangent_world: Vec3,
    /// 접점의 월드 위치 (generic witness 접점의 근접 매칭용).
    pub position: Vec3,
    /// 캐시 시점의 접촉 **법선 방향**(`normal`은 법선 임펄스 크기라 이름을 구분한다).
    /// 위치가 가까워도 면이 바뀌었으면 다른 접촉이다.
    pub contact_normal: Vec3,
}

/// 매니폴드 → 제약 준비. `cache`는 전 스텝 (feature → 임펄스) — feature ID 일치 접점만 warm load.
pub fn prepare(
    bodies: &[Body],
    a: usize,
    b: usize,
    m: &Manifold,
    cache: &[(maru_collide::FeatureId, CachedImpulse)],
    dt: f32,
) -> ContactConstraint {
    // 기하 바닥은 형상 크기에 비례해야 한다(절대 상수 금지 — 톨러런스 정책과 동일 원칙).
    let geo_tol =
        WARM_MATCH_GEO * maru_collide::min_feature_size(&bodies[a].shape, &bodies[b].shape);
    let (ba, bb) = (&bodies[a], &bodies[b]);
    let n = m.normal;
    let tangents = tangent_basis(n);
    // **실제로 warm load된 접점 수**. 진단이 feature ID 일치만 세면 generic 접점의 위치·법선
    // 게이트가 얼마나 자주 매칭을 끊는지 보이지 않는다 — 지표가 거짓이면 회귀 테스트도 거짓이다.
    let mut warm_matched = 0u32;
    let (ia, ib) = (ba.inv_inertia_world(), bb.inv_inertia_world());
    let points = m
        .points
        .iter()
        .map(|p| {
            let ra = p.position - ba.pose.pos;
            let rb = p.position - bb.pose.pos;
            let vn = n.dot(ba.velocity_at(ra) - bb.velocity_at(rb));
            // warm start 매칭 규칙 (리뷰 MAJOR):
            // - 클리핑 접점: feature ID가 계보를 담아 안정적 → ID 일치
            // - generic witness 접점: pair 안에서 ID를 공유하므로(구-박스 접점이 면→모서리로
            //   이동해도 같은 ID) **위치 근접**까지 요구한다. 점프하면 임펄스를 물려주지 않는다.
            let generic = p.feature.0 == maru_collide::FEATURE_GENERIC;
            // 허용치 = 기하 바닥 + 접점의 정당한 이동 상한. 정지 접촉은 기하 바닥만 쓴다.
            let drift = ba.linvel.length()
                + ba.angvel.length() * ra.length()
                + bb.linvel.length()
                + bb.angvel.length() * rb.length();
            let match_tol = geo_tol + WARM_MATCH_MOTION * drift * dt;
            let match_tol2 = match_tol * match_tol;
            let hit = cache
                .iter()
                .find(|(f, c)| {
                    *f == p.feature
                        && (!generic
                            || ((c.position - p.position).length_sq() <= match_tol2
                                // 면이 바뀌었으면 다른 접촉 (곡면 feature 점프 차단)
                                && c.contact_normal.dot(n) >= WARM_MATCH_COS))
                })
                .map(|(_, c)| *c);
            if hit.is_some() {
                warm_matched += 1;
            }
            let warm = hit.unwrap_or_default();
            // 월드 마찰 임펄스를 현재 접평면에 재투영 (기저 회전 대응)
            let warm_t = [
                warm.tangent_world.dot(tangents[0]),
                warm.tangent_world.dot(tangents[1]),
            ];
            ContactPointConstraint {
                feature: p.feature,
                position: p.position,
                anchor_a: ba.pose.rot.conjugate().rotate(ra),
                anchor_b: bb.pose.rot.conjugate().rotate(rb),
                base_separation: p.separation,
                normal_mass: effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, n),
                tangent_mass: [
                    effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[0]),
                    effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[1]),
                ],
                normal_impulse: warm.normal,
                tangent_impulse: warm_t,
                max_normal_impulse: 0.0,
                ra,
                rb,
                approach_vn: vn,
            }
        })
        .collect();
    ContactConstraint {
        warm_matched,
        a,
        b,
        normal: n,
        tangents,
        friction: (ba.friction * bb.friction).sqrt(),
        restitution: ba.restitution.max(bb.restitution),
        points,
    }
}

/// 현재 자세로 레버암을 갱신한다 (서브스텝당 1회, 위치 적분 뒤 1회).
pub fn refresh_anchors(st: &SolverState, c: &mut ContactConstraint) {
    let (qa, qb) = (st.rot[c.a], st.rot[c.b]);
    for p in &mut c.points {
        p.ra = qa.rotate(p.anchor_a);
        p.rb = qb.rotate(p.anchor_b);
    }
}

/// warm start: 누적 임펄스를 서브스텝 시작 추정으로 재적용 (Box2D v3 동형).
pub fn warm_start(st: &mut SolverState, c: &ContactConstraint) {
    for p in &c.points {
        let (ra, rb) = (p.ra, p.rb);
        let imp = c.normal * p.normal_impulse
            + c.tangents[0] * p.tangent_impulse[0]
            + c.tangents[1] * p.tangent_impulse[1];
        st.apply(c.a, imp, ra);
        st.apply(c.b, -imp, rb);
    }
}

/// 속도 임펄스 해석. `soft`=bias 페이즈, `relax`(soft=None)=bias 제거 페이즈.
pub fn solve(
    st: &mut SolverState,
    c: &mut ContactConstraint,
    inv_h: f32,
    soft: Option<(&Softness, f32)>, // (계수, max_bias_speed)
) {
    for p in &mut c.points {
        let (ra, rb) = (p.ra, p.rb);
        // TGS: 현재 자세로 separation 추적. A가 +n로 이동하면 s 증가해야 하므로
        // s = n·(pa − pb) + base (prepare 시점엔 앵커가 같은 월드점 → s = base).
        let pa = st.pos[c.a] + ra;
        let pb = st.pos[c.b] + rb;
        let s = c.normal.dot(pa - pb) + p.base_separation;

        // --- 법선 ---
        let vn = c.normal.dot(st.vel_at(c.a, ra) - st.vel_at(c.b, rb));
        let (bias, mass_scale, impulse_scale) = match soft {
            Some((sf, max_bias)) => {
                if s > 0.0 {
                    // speculative: 이번 서브스텝에 닿을 만큼만 감속 (임팩트 게이트 유지)
                    (s * inv_h, 1.0, 0.0)
                } else {
                    (
                        (sf.bias_rate * s).max(-max_bias),
                        sf.mass_scale,
                        sf.impulse_scale,
                    )
                }
            }
            None => {
                if s > 0.0 {
                    (s * inv_h, 1.0, 0.0)
                } else {
                    (0.0, 1.0, 0.0)
                }
            }
        };
        // 부호 계약: vn = n·(va−vb) = ds/dt — 분리=양수, 접근=음수.
        // s<0(침투): bias<0 → impulse>0 (밀어냄). s>0(speculative): bias=s/h → 초과 접근분만 감속.
        let impulse = -p.normal_mass * mass_scale * (vn + bias) - impulse_scale * p.normal_impulse;
        let new_total = (p.normal_impulse + impulse).max(0.0);
        let applied = new_total - p.normal_impulse;
        p.normal_impulse = new_total;
        p.max_normal_impulse = p.max_normal_impulse.max(new_total);
        let imp = c.normal * applied;
        st.apply(c.a, imp, ra);
        st.apply(c.b, -imp, rb);

        // --- 마찰: 축별 **순차** 해석(GS) 후 누적 2D 임펄스를 Coulomb **원판**에 투영 ---
        // 사각형 클램프는 대각 슬라이드에서 √2·μN까지 허용해 41% 과도 감속하고 기저 의존적이다.
        // 단, 두 축을 동시(Jacobi)로 풀면 유효질량이 큰 접점에서 과잉 보정이 겹쳐 발산한다
        // (낙하 체인 실증) — 순차 갱신을 유지한 채 마지막에 원판으로 되돌린다.
        let max_f = c.friction * p.normal_impulse;
        for axis in 0..2 {
            let t = c.tangents[axis];
            let vt = t.dot(st.vel_at(c.a, ra) - st.vel_at(c.b, rb));
            let delta = -p.tangent_mass[axis] * vt;
            p.tangent_impulse[axis] += delta;
            let imp = t * delta;
            st.apply(c.a, imp, ra);
            st.apply(c.b, -imp, rb);
        }
        let mag2 = p.tangent_impulse[0] * p.tangent_impulse[0]
            + p.tangent_impulse[1] * p.tangent_impulse[1];
        if mag2 > max_f * max_f && mag2 > 0.0 {
            let scale = max_f / mag2.sqrt();
            let corr0 = p.tangent_impulse[0] * (scale - 1.0);
            let corr1 = p.tangent_impulse[1] * (scale - 1.0);
            p.tangent_impulse[0] += corr0;
            p.tangent_impulse[1] += corr1;
            let imp = c.tangents[0] * corr0 + c.tangents[1] * corr1;
            st.apply(c.a, imp, ra);
            st.apply(c.b, -imp, rb);
        }
    }
    let _ = inv_h;
}

/// restitution 패스 — 전 서브스텝 종료 후 1회 (설계 §4.4 임팩트 게이트).
pub fn apply_restitution(st: &mut SolverState, c: &mut ContactConstraint, threshold: f32) {
    if c.restitution == 0.0 {
        return;
    }
    for p in &mut c.points {
        // 게이트(설계 §4.4): 실제 임펄스가 발생했고, 접근 속도(음수)가 임계보다 빨랐던 접점만
        if p.approach_vn > -threshold || p.max_normal_impulse == 0.0 {
            continue;
        }
        let ra = st.rot[c.a].rotate(p.anchor_a);
        let rb = st.rot[c.b].rotate(p.anchor_b);
        let vn = c.normal.dot(st.vel_at(c.a, ra) - st.vel_at(c.b, rb));
        // 목표: vn_out = −e·approach_vn (approach_vn<0 → 양수 반발)
        let impulse = -p.normal_mass * (vn + c.restitution * p.approach_vn);
        let new_total = (p.normal_impulse + impulse).max(0.0);
        let applied = new_total - p.normal_impulse;
        p.normal_impulse = new_total;
        let imp = c.normal * applied;
        st.apply(c.a, imp, ra);
        st.apply(c.b, -imp, rb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Body;
    use maru_collide::{ContactPoint, FeatureId, Manifold, Pose, Shape, FEATURE_GENERIC};
    use maru_math::Quat;

    fn world_bodies() -> Vec<Body> {
        let pose = |y: f32| Pose {
            pos: Vec3::new(0.0, y, 0.0),
            rot: Quat::IDENTITY,
        };
        vec![
            Body::new_static(
                Shape::Box {
                    half: Vec3::new(5.0, 0.5, 5.0),
                    radius: 0.0,
                },
                pose(0.0),
            ),
            Body::new_dynamic(Shape::Sphere { radius: 0.5 }, pose(1.0), 1.0),
        ]
    }

    fn manifold_at(pos: Vec3) -> Manifold {
        Manifold {
            normal: Vec3::new(0.0, 1.0, 0.0),
            points: vec![ContactPoint {
                position: pos,
                separation: -0.001,
                feature: FeatureId(FEATURE_GENERIC),
            }],
        }
    }

    /// 계약(리뷰 MAJOR): 계보 없는 witness 접점은 ID를 공유하므로 **위치 근접**까지 맞을 때만
    /// 이전 임펄스를 물려받는다. 접점이 점프하면(면→모서리) 물려받지 않는다.
    #[test]
    fn generic_contacts_warm_start_only_when_close() {
        let bodies = world_bodies();
        let cached_at = Vec3::new(0.0, 0.5, 0.0);
        let cache = vec![(
            FeatureId(FEATURE_GENERIC),
            CachedImpulse {
                normal: 7.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: cached_at,
                contact_normal: Vec3::new(0.0, 1.0, 0.0),
            },
        )];
        // 정지 상태이므로 허용치 = 기하 바닥만 (최소 특징 크기 = 구 반지름 0.5 → 0.125)
        const DT: f32 = 1.0 / 60.0;
        // ① 근접(1cm 이동): 계승
        let near = prepare(
            &bodies,
            1,
            0,
            &manifold_at(cached_at + Vec3::new(0.01, 0.0, 0.0)),
            &cache,
            DT,
        );
        assert_eq!(
            near.points[0].normal_impulse, 7.0,
            "근접 접점은 warm start 계승"
        );
        // ② 점프(1m 이동): 차단
        let far = prepare(
            &bodies,
            1,
            0,
            &manifold_at(cached_at + Vec3::new(1.0, 0.0, 0.0)),
            &cache,
            DT,
        );
        assert_eq!(
            far.points[0].normal_impulse, 0.0,
            "점프한 접점에 이전 임펄스가 적용됨"
        );
        // ③ 법선이 뒤집힌 접점: 위치가 같아도 다른 접촉이다 (곡면 feature 점프 차단)
        let flipped = vec![(
            FeatureId(FEATURE_GENERIC),
            CachedImpulse {
                normal: 7.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: cached_at,
                contact_normal: Vec3::new(1.0, 0.0, 0.0),
            },
        )];
        let turned = prepare(&bodies, 1, 0, &manifold_at(cached_at), &flipped, DT);
        assert_eq!(
            turned.points[0].normal_impulse, 0.0,
            "법선이 바뀐 접촉에 이전 임펄스가 계승됨"
        );
    }

    /// 대조: 클리핑 접점(계보 있는 ID)은 위치와 무관하게 ID로 매칭된다.
    #[test]
    fn clipped_contacts_match_by_feature_id() {
        let bodies = world_bodies();
        let fid = FeatureId(0x8123_4567);
        let cache = vec![(
            fid,
            CachedImpulse {
                normal: 3.0,
                tangent: [0.0, 0.0],
                tangent_world: Vec3::ZERO,
                position: Vec3::new(0.0, 0.5, 0.0),
                contact_normal: Vec3::new(0.0, 1.0, 0.0),
            },
        )];
        let mut m = manifold_at(Vec3::new(2.0, 0.5, 0.0));
        m.points[0].feature = fid;
        let c = prepare(&bodies, 1, 0, &m, &cache, 1.0 / 60.0);
        assert_eq!(c.points[0].normal_impulse, 3.0);
    }
}
