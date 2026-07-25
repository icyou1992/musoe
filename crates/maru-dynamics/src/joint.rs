//! 최소 조인트 — point(ball) · hinge (설계 §4.4 M2: 접촉과 동일 soft 정식화, 별도 솔버 없음).
//!
//! 서브스텝마다 TGS 방식으로 현재 자세에서 위치 오차 C를 재계산하고 soft bias로 해석.
//! [근거: Box2D v3가 조인트까지 soft-step 단일 솔버로 처리 — 01 §2 정정 반영]

use crate::body::{Body, BodyHandle};
use crate::solver::Softness;
use maru_math::{Mat3, Vec3};

#[derive(Clone, Debug)]
pub enum JointKind {
    /// ball-socket: 앵커 일치 (3-DOF 위치 제약).
    Point,
    /// point + 축 정렬 (2-DOF 각 제약 추가). axis는 각 바디 로컬.
    Hinge { axis_a: Vec3, axis_b: Vec3 },
}

/// **엔드포인트는 세대 보유 핸들**이다(리뷰 BLOCKER). raw 인덱스면 같은 배치 안에서
/// 슬롯이 제거·재사용될 때 조인트가 엉뚱한 새 바디에 붙는 ABA가 성립한다.
/// 핫패스는 `ia()`/`ib()`로 인덱스만 쓰되, 정본 상태·직렬화·검증은 핸들로 한다.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Joint {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// 질량중심 기준 로컬 앵커.
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub kind: JointKind,
    /// 연결 바디 간 충돌 허용 여부 (기본 false — Box2D·Jolt 관행. 조인트 당김과 접촉 밀어냄이
    /// warm start로 맞서면 공진 발산한다: hinge 체인 접지 실증, step 390→420 inf).
    pub collide_connected: bool,
    /// 누적 임펄스 (warm start — 스텝 간 유지).
    pub linear_impulse: Vec3,
    pub angular_impulse: [f32; 2],
}

impl Joint {
    /// 슬롯 인덱스 (월드가 생존·세대를 이미 검증한 뒤의 핫패스 전용).
    #[inline]
    pub(crate) fn ia(&self) -> usize {
        self.body_a.index as usize
    }
    #[inline]
    pub(crate) fn ib(&self) -> usize {
        self.body_b.index as usize
    }

    pub fn point(body_a: BodyHandle, body_b: BodyHandle, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            kind: JointKind::Point,
            collide_connected: false,
            linear_impulse: Vec3::ZERO,
            angular_impulse: [0.0, 0.0],
        }
    }

    pub fn hinge(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
    ) -> Self {
        Self {
            kind: JointKind::Hinge {
                axis_a: axis_a.normalize_or_zero(),
                axis_b: axis_b.normalize_or_zero(),
            },
            ..Self::point(body_a, body_b, anchor_a, anchor_b)
        }
    }
}

#[inline]
fn apply(body: &mut Body, imp: Vec3, r: Vec3, inv_i: &Mat3) {
    if body.is_dynamic() {
        body.linvel += imp * body.inv_mass;
        body.angvel += inv_i.mul_vec(r.cross(imp));
    }
}

#[inline]
fn apply_angular(body: &mut Body, imp: Vec3, inv_i: &Mat3) {
    if body.is_dynamic() {
        body.angvel += inv_i.mul_vec(imp);
    }
}

/// warm start (서브스텝 시작마다 — 접촉과 동일 패턴).
pub fn warm_start(bodies: &mut [Body], j: &Joint) {
    let ia = bodies[j.ia()].inv_inertia_world();
    let ib = bodies[j.ib()].inv_inertia_world();
    let ra = bodies[j.ia()].pose.rot.rotate(j.anchor_a);
    let rb = bodies[j.ib()].pose.rot.rotate(j.anchor_b);
    apply(&mut bodies[j.ia()], j.linear_impulse, ra, &ia);
    apply(&mut bodies[j.ib()], -j.linear_impulse, rb, &ib);
    if let JointKind::Hinge { axis_a, .. } = &j.kind {
        let wa = bodies[j.ia()].pose.rot.rotate(*axis_a);
        let (u, v) = perp_basis(wa);
        let ang = u * j.angular_impulse[0] + v * j.angular_impulse[1];
        apply_angular(&mut bodies[j.ia()], ang, &ia);
        apply_angular(&mut bodies[j.ib()], -ang, &ib);
    }
}

fn perp_basis(a: Vec3) -> (Vec3, Vec3) {
    let u = if a.x.abs() > 0.707 {
        Vec3::new(a.y, -a.x, 0.0).normalize_or_zero()
    } else {
        Vec3::new(0.0, a.z, -a.y).normalize_or_zero()
    };
    (u, a.cross(u))
}

/// 서브스텝 해석. `soft`=bias 페이즈, None=relax.
pub fn solve(bodies: &mut [Body], j: &mut Joint, soft: Option<&Softness>) {
    let ia = bodies[j.ia()].inv_inertia_world();
    let ib = bodies[j.ib()].inv_inertia_world();
    let (inv_ma, inv_mb) = (bodies[j.ia()].inv_mass, bodies[j.ib()].inv_mass);
    let ra = bodies[j.ia()].pose.rot.rotate(j.anchor_a);
    let rb = bodies[j.ib()].pose.rot.rotate(j.anchor_b);

    // --- 선형 3-DOF: K = (1/ma+1/mb)I + skew(ra)ᵀIa⁻¹skew(ra) + skew(rb)ᵀIb⁻¹skew(rb) ---
    let k = {
        let mut m =
            Mat3::from_diagonal(Vec3::new(inv_ma + inv_mb, inv_ma + inv_mb, inv_ma + inv_mb));
        // skew(r)ᵀ I⁻¹ skew(r) 를 열 단위로 구성: col_j = r × (I⁻¹ (e_j × r))
        for (mat, r) in [(&ia, ra), (&ib, rb)] {
            let e = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ];
            for (ji, ej) in e.iter().enumerate() {
                let col = r.cross(mat.mul_vec(ej.cross(r)));
                // K += -skew(r) I⁻¹ skew(r) — 부호: skew(r)ᵀ = -skew(r)
                m.c[ji] += col;
            }
        }
        m
    };
    let c_pos = (bodies[j.ia()].pose.pos + ra) - (bodies[j.ib()].pose.pos + rb);
    let cdot = bodies[j.ia()].velocity_at(ra) - bodies[j.ib()].velocity_at(rb);
    let (bias, mass_scale, impulse_scale) = match soft {
        Some(sf) => (c_pos * sf.bias_rate, sf.mass_scale, sf.impulse_scale),
        None => (Vec3::ZERO, 1.0, 0.0),
    };
    let impulse =
        k.inverse().mul_vec(-(cdot + bias)) * mass_scale - j.linear_impulse * impulse_scale;
    j.linear_impulse += impulse;
    apply(&mut bodies[j.ia()], impulse, ra, &ia);
    apply(&mut bodies[j.ib()], -impulse, rb, &ib);

    // --- hinge 각 2-DOF: 축 수직 성분 상대 각속도 = 0, 위치 bias = 축 어긋남 ---
    if let JointKind::Hinge { axis_a, axis_b } = j.kind {
        let wa_axis = bodies[j.ia()].pose.rot.rotate(axis_a);
        let wb_axis = bodies[j.ib()].pose.rot.rotate(axis_b);
        let (u, v) = perp_basis(wa_axis);
        let wrel = bodies[j.ia()].angvel - bodies[j.ib()].angvel;
        // 축 오차. 부호는 임펄스 적용 규약(A += imp, B -= imp)과 맞아야 한다 —
        // wa×wb를 쓰면 20° 어긋난 축이 정렬되는 대신 반대축(-1)으로 넘어간다(실증).
        let err = wb_axis.cross(wa_axis);
        for (axis_i, dir) in [(0usize, u), (1usize, v)] {
            let k_ang = dir.dot(ia.mul_vec(dir)) + dir.dot(ib.mul_vec(dir));
            if k_ang <= 0.0 {
                continue;
            }
            let cdot = dir.dot(wrel);
            // (ωa−ωb)를 −bias_rate·err 로 구동하면 두 축이 서로를 향해 회전한다.
            let c = dir.dot(err);
            let (bias, ms, is_) = match soft {
                Some(sf) => (c * sf.bias_rate, sf.mass_scale, sf.impulse_scale),
                None => (0.0, 1.0, 0.0),
            };
            let imp = -(cdot + bias) / k_ang * ms - j.angular_impulse[axis_i] * is_;
            j.angular_impulse[axis_i] += imp;
            let ang = dir * imp;
            apply_angular(&mut bodies[j.ia()], ang, &ia);
            apply_angular(&mut bodies[j.ib()], -ang, &ib);
        }
    }
}
