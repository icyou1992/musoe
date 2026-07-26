//! 최소 조인트 — point(ball) · hinge (설계 §4.4 M2: 접촉과 동일 soft 정식화, 별도 솔버 없음).
//!
//! 서브스텝마다 TGS 방식으로 현재 자세에서 위치 오차 C를 재계산하고 soft bias로 해석.
//! [근거: Box2D v3가 조인트까지 soft-step 단일 솔버로 처리 — 01 §2 정정 반영]

use crate::body::BodyHandle;
use crate::solver::Softness;
use maru_collide::{in_state_range, vec_anchor_in_range, vec_in_state_range};
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
    pub(crate) body_a: BodyHandle,
    pub(crate) body_b: BodyHandle,
    /// 질량중심 기준 로컬 앵커.
    pub(crate) anchor_a: Vec3,
    pub(crate) anchor_b: Vec3,
    pub(crate) kind: JointKind,
    /// 연결 바디 간 충돌 허용 여부 (기본 false — Box2D·Jolt 관행. 조인트 당김과 접촉 밀어냄이
    /// warm start로 맞서면 공진 발산한다: hinge 체인 접지 실증, step 390→420 inf).
    pub(crate) collide_connected: bool,
    /// 누적 임펄스 (warm start — 스텝 간 유지).
    pub(crate) linear_impulse: Vec3,
    pub(crate) angular_impulse: [f32; 2],
}

impl Joint {
    #[inline]
    pub fn body_a(&self) -> BodyHandle {
        self.body_a
    }
    #[inline]
    pub fn body_b(&self) -> BodyHandle {
        self.body_b
    }
    #[inline]
    pub fn kind(&self) -> &JointKind {
        &self.kind
    }
    #[inline]
    pub fn collide_connected(&self) -> bool {
        self.collide_connected
    }

    /// 조인트 자체의 불변식 — 삽입·역직렬화 경계에서 강제한다.
    /// 비교는 **양성 조건**으로 쓴다: `!(x < 0.5)`는 NaN에서 참이 되어 통과시킨다.
    pub(crate) fn validate(&self) -> Result<(), crate::world::ApiError> {
        use crate::world::ApiError;
        // 앵커는 **레버암**이다: `ω×r` 점 속도를 유계로 만들려면 셰이프 범위로 묶어야 한다
        // (리뷰 BLOCKER: 앵커 1e6 × 각속도 100rad/s = 점 속도 1e8 m/s).
        if !vec_anchor_in_range(self.anchor_a) || !vec_anchor_in_range(self.anchor_b) {
            return Err(ApiError::InvalidValue);
        }
        if !vec_in_state_range(self.linear_impulse)
            || !in_state_range(self.angular_impulse[0])
            || !in_state_range(self.angular_impulse[1])
        {
            return Err(ApiError::InvalidValue);
        }
        if let JointKind::Hinge { axis_a, axis_b } = self.kind {
            for a in [axis_a, axis_b] {
                let l2 = a.length_sq();
                if !(0.9..=1.1).contains(&l2) {
                    return Err(ApiError::InvalidValue);
                }
            }
        }
        Ok(())
    }

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
        Self::try_point(body_a, body_b, anchor_a, anchor_b).expect("invalid point joint")
    }

    /// 볼조인트 생성 (오류 반환형).
    pub fn try_point(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> Result<Self, crate::world::ApiError> {
        let j = Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            kind: JointKind::Point,
            collide_connected: false,
            linear_impulse: Vec3::ZERO,
            angular_impulse: [0.0, 0.0],
        };
        j.validate()?;
        Ok(j)
    }

    pub fn hinge(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
    ) -> Self {
        Self::try_hinge(body_a, body_b, anchor_a, anchor_b, axis_a, axis_b)
            .expect("invalid hinge joint")
    }

    /// 힌지 생성 (오류 반환형 — **M3 바인딩의 정본 경로**). 0/NaN 축은 여기서 거부된다.
    /// `try_add_joint`는 Joint가 만들어진 **뒤에** 불리므로 생성 단계의 trap을 잡지 못한다.
    pub fn try_hinge(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
    ) -> Result<Self, crate::world::ApiError> {
        use crate::world::ApiError;
        // 0/NaN 축을 조용히 0으로 남기면 각 제약이 사라진다 — 생성 경계에서 거부
        let axis_a = axis_a.try_normalize().ok_or(ApiError::InvalidValue)?;
        let axis_b = axis_b.try_normalize().ok_or(ApiError::InvalidValue)?;
        let j = Self {
            kind: JointKind::Hinge { axis_a, axis_b },
            ..Self::try_point(body_a, body_b, anchor_a, anchor_b)?
        };
        j.validate()?;
        Ok(j)
    }
}

/// warm start (서브스텝 시작마다 — 접촉과 동일 패턴).
pub fn warm_start(st: &mut crate::solver::SolverState, j: &Joint) {
    let ra = st.rot[j.ia()].rotate(j.anchor_a);
    let rb = st.rot[j.ib()].rotate(j.anchor_b);
    st.apply_pub(j.ia(), j.linear_impulse, ra);
    st.apply_pub(j.ib(), -j.linear_impulse, rb);
    if let JointKind::Hinge { axis_a, .. } = &j.kind {
        let wa = st.rot[j.ia()].rotate(*axis_a);
        let (u, v) = perp_basis(wa);
        let ang = u * j.angular_impulse[0] + v * j.angular_impulse[1];
        st.apply_ang_pub(j.ia(), ang);
        st.apply_ang_pub(j.ib(), -ang);
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

/// 벡터 길이를 상한으로 자른다 (방향 보존).
#[inline]
fn clamp_len(v: Vec3, max: f32) -> Vec3 {
    let l2 = v.length_sq();
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(l2 <= max * max) {
        let l = v.length();
        if l.is_finite() && l > 0.0 {
            return v * (max / l);
        }
        return Vec3::ZERO;
    }
    v
}

/// 서브스텝 해석. `soft`=(bias 파라미터, 최대 bias 속도), None=relax.
///
/// **bias를 차원별로 클램프한다**(리뷰 BLOCKER ×2): 선형 bias는 `max_bias_speed`(m/s),
/// **각 bias는 각속도 상한**(rad/s). `c × bias_rate`는 오차가 크고 h가 작을 때 1e12까지 가고,
/// 유효(관성)질량을 곱하면 파생 상태 상한을 넘긴다. 접촉이 이미 선형 방어를 하고 있었는데
/// 조인트는 선형·각 모두 빠져 있었다.
pub fn solve(
    st: &mut crate::solver::SolverState,
    j: &mut Joint,
    soft: Option<(&Softness, f32, f32)>,
) {
    let (ia, ib) = (st.inv_i[j.ia()], st.inv_i[j.ib()]);
    let (inv_ma, inv_mb) = (st.inv_m[j.ia()], st.inv_m[j.ib()]);
    let ra = st.rot[j.ia()].rotate(j.anchor_a);
    let rb = st.rot[j.ib()].rotate(j.anchor_b);

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
    let c_pos = (st.pos[j.ia()] + ra) - (st.pos[j.ib()] + rb);
    let cdot = st.vel_at_pub(j.ia(), ra) - st.vel_at_pub(j.ib(), rb);
    let (bias, mass_scale, impulse_scale) = match soft {
        Some((sf, max_bias, _)) => (
            clamp_len(c_pos * sf.bias_rate, max_bias),
            sf.mass_scale,
            sf.impulse_scale,
        ),
        None => (Vec3::ZERO, 1.0, 0.0),
    };
    let impulse =
        k.inverse().mul_vec(-(cdot + bias)) * mass_scale - j.linear_impulse * impulse_scale;
    j.linear_impulse += impulse;
    st.apply_pub(j.ia(), impulse, ra);
    st.apply_pub(j.ib(), -impulse, rb);

    // --- hinge 각 2-DOF: 축 수직 성분 상대 각속도 = 0, 위치 bias = 축 어긋남 ---
    if let JointKind::Hinge { axis_a, axis_b } = j.kind {
        let wa_axis = st.rot[j.ia()].rotate(axis_a);
        let wb_axis = st.rot[j.ib()].rotate(axis_b);
        let (u, v) = perp_basis(wa_axis);
        let wrel = st.ang[j.ia()] - st.ang[j.ib()];
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
                // **각 bias는 각속도**다 — 선형 상한(m/s)이 아니라 각속도 상한(rad/s)으로
                // 묶어야 `imp = bias/k_ang`이 유계가 된다(리뷰 BLOCKER: 큰 관성에서
                // 1/k_ang이 1e10까지 가므로 선형 상한 1e4로는 1e14를 넘긴다).
                Some((sf, _, max_ang_bias)) => (
                    (c * sf.bias_rate).clamp(-max_ang_bias, max_ang_bias),
                    sf.mass_scale,
                    sf.impulse_scale,
                ),
                None => (0.0, 1.0, 0.0),
            };
            let imp = -(cdot + bias) / k_ang * ms - j.angular_impulse[axis_i] * is_;
            j.angular_impulse[axis_i] += imp;
            let ang = dir * imp;
            st.apply_ang_pub(j.ia(), ang);
            st.apply_ang_pub(j.ib(), -ang);
        }
    }
}
