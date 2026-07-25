//! TGS-soft 접촉 솔버 (설계 §4.4 — Box2D v3 "Soft Step" 정식화).
//!
//! 서브스텝마다: 속도 적분 → warm start → soft bias 임펄스 해석 → 위치 적분 → relax(무bias).
//! 전 서브스텝 후 restitution 패스 1회 — **임팩트 게이트**(설계 §4.4): 접촉 생성 시점의
//! 접근 속도(approach_vn)가 임계 이상일 때만, 실제 임펄스가 발생한 접점에만 발화.
//! [근거: Solver2D 실증 + Box2D v3 releasing-3.0]

use crate::body::{Body, BodyType};
use maru_collide::Manifold;
use maru_math::{Mat3, Vec3};

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

/// 캐시된 접점 임펄스 (스텝 간 warm start — 설계 §3 PCM).
#[derive(Clone, Copy, Default)]
pub struct CachedImpulse {
    pub normal: f32,
    pub tangent: [f32; 2],
}

/// 매니폴드 → 제약 준비. `cache`는 전 스텝 (feature → 임펄스) — feature ID 일치 접점만 warm load.
pub fn prepare(
    bodies: &[Body],
    a: usize,
    b: usize,
    m: &Manifold,
    cache: &[(maru_collide::FeatureId, CachedImpulse)],
) -> ContactConstraint {
    let (ba, bb) = (&bodies[a], &bodies[b]);
    let n = m.normal;
    let tangents = tangent_basis(n);
    let (ia, ib) = (ba.inv_inertia_world(), bb.inv_inertia_world());
    let points = m
        .points
        .iter()
        .map(|p| {
            let ra = p.position - ba.pose.pos;
            let rb = p.position - bb.pose.pos;
            let vn = n.dot(ba.velocity_at(ra) - bb.velocity_at(rb));
            let warm = cache
                .iter()
                .find(|(f, _)| *f == p.feature)
                .map(|(_, c)| *c)
                .unwrap_or_default();
            ContactPointConstraint {
                feature: p.feature,
                anchor_a: ba.pose.rot.conjugate().rotate(ra),
                anchor_b: bb.pose.rot.conjugate().rotate(rb),
                base_separation: p.separation,
                normal_mass: effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, n),
                tangent_mass: [
                    effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[0]),
                    effective_mass(ba.inv_mass, &ia, ra, bb.inv_mass, &ib, rb, tangents[1]),
                ],
                normal_impulse: warm.normal,
                tangent_impulse: warm.tangent,
                max_normal_impulse: 0.0,
                approach_vn: vn,
            }
        })
        .collect();
    ContactConstraint {
        a,
        b,
        normal: n,
        tangents,
        friction: (ba.friction * bb.friction).sqrt(),
        restitution: ba.restitution.max(bb.restitution),
        points,
    }
}

#[inline]
fn apply_impulse(body: &mut Body, imp: Vec3, r: Vec3, inv_i: &Mat3) {
    if body.body_type == BodyType::Dynamic {
        body.linvel += imp * body.inv_mass;
        body.angvel += inv_i.mul_vec(r.cross(imp));
    }
}

/// warm start: 누적 임펄스를 서브스텝 시작 추정으로 재적용 (Box2D v3 동형).
pub fn warm_start(bodies: &mut [Body], c: &ContactConstraint) {
    let ia = bodies[c.a].inv_inertia_world();
    let ib = bodies[c.b].inv_inertia_world();
    for p in &c.points {
        let ra = bodies[c.a].pose.rot.rotate(p.anchor_a);
        let rb = bodies[c.b].pose.rot.rotate(p.anchor_b);
        let imp = c.normal * p.normal_impulse
            + c.tangents[0] * p.tangent_impulse[0]
            + c.tangents[1] * p.tangent_impulse[1];
        apply_impulse(&mut bodies[c.a], imp, ra, &ia);
        apply_impulse(&mut bodies[c.b], -imp, rb, &ib);
    }
}

/// 속도 임펄스 해석. `soft`=bias 페이즈, `relax`(soft=None)=bias 제거 페이즈.
pub fn solve(
    bodies: &mut [Body],
    c: &mut ContactConstraint,
    inv_h: f32,
    soft: Option<(&Softness, f32)>, // (계수, max_bias_speed)
) {
    let ia = bodies[c.a].inv_inertia_world();
    let ib = bodies[c.b].inv_inertia_world();
    for p in &mut c.points {
        let ra = bodies[c.a].pose.rot.rotate(p.anchor_a);
        let rb = bodies[c.b].pose.rot.rotate(p.anchor_b);
        // TGS: 현재 자세로 separation 추적. A가 +n로 이동하면 s 증가해야 하므로
        // s = n·(pa − pb) + base (prepare 시점엔 앵커가 같은 월드점 → s = base).
        let pa = bodies[c.a].pose.pos + ra;
        let pb = bodies[c.b].pose.pos + rb;
        let s = c.normal.dot(pa - pb) + p.base_separation;

        // --- 법선 ---
        let vn = c
            .normal
            .dot(bodies[c.a].velocity_at(ra) - bodies[c.b].velocity_at(rb));
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
        apply_impulse(&mut bodies[c.a], imp, ra, &ia);
        apply_impulse(&mut bodies[c.b], -imp, rb, &ib);

        // --- 마찰: 축별 **순차** 해석(GS) 후 누적 2D 임펄스를 Coulomb **원판**에 투영 ---
        // 사각형 클램프는 대각 슬라이드에서 √2·μN까지 허용해 41% 과도 감속하고 기저 의존적이다.
        // 단, 두 축을 동시(Jacobi)로 풀면 유효질량이 큰 접점에서 과잉 보정이 겹쳐 발산한다
        // (낙하 체인 실증) — 순차 갱신을 유지한 채 마지막에 원판으로 되돌린다.
        let max_f = c.friction * p.normal_impulse;
        for axis in 0..2 {
            let t = c.tangents[axis];
            let vt = t.dot(bodies[c.a].velocity_at(ra) - bodies[c.b].velocity_at(rb));
            let delta = -p.tangent_mass[axis] * vt;
            p.tangent_impulse[axis] += delta;
            let imp = t * delta;
            apply_impulse(&mut bodies[c.a], imp, ra, &ia);
            apply_impulse(&mut bodies[c.b], -imp, rb, &ib);
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
            apply_impulse(&mut bodies[c.a], imp, ra, &ia);
            apply_impulse(&mut bodies[c.b], -imp, rb, &ib);
        }
    }
    let _ = inv_h;
}

/// restitution 패스 — 전 서브스텝 종료 후 1회 (설계 §4.4 임팩트 게이트).
pub fn apply_restitution(bodies: &mut [Body], c: &mut ContactConstraint, threshold: f32) {
    if c.restitution == 0.0 {
        return;
    }
    let ia = bodies[c.a].inv_inertia_world();
    let ib = bodies[c.b].inv_inertia_world();
    for p in &mut c.points {
        // 게이트(설계 §4.4): 실제 임펄스가 발생했고, 접근 속도(음수)가 임계보다 빨랐던 접점만
        if p.approach_vn > -threshold || p.max_normal_impulse == 0.0 {
            continue;
        }
        let ra = bodies[c.a].pose.rot.rotate(p.anchor_a);
        let rb = bodies[c.b].pose.rot.rotate(p.anchor_b);
        let vn = c
            .normal
            .dot(bodies[c.a].velocity_at(ra) - bodies[c.b].velocity_at(rb));
        // 목표: vn_out = −e·approach_vn (approach_vn<0 → 양수 반발)
        let impulse = -p.normal_mass * (vn + c.restitution * p.approach_vn);
        let new_total = (p.normal_impulse + impulse).max(0.0);
        let applied = new_total - p.normal_impulse;
        p.normal_impulse = new_total;
        let imp = c.normal * applied;
        apply_impulse(&mut bodies[c.a], imp, ra, &ia);
        apply_impulse(&mut bodies[c.b], -imp, rb, &ib);
    }
}
