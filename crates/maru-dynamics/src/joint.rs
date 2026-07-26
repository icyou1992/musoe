//! 최소 조인트 — point(ball) · hinge (설계 §4.4 M2: 접촉과 동일 soft 정식화, 별도 솔버 없음).
//!
//! 서브스텝마다 TGS 방식으로 현재 자세에서 위치 오차 C를 재계산하고 soft bias로 해석.
//! [근거: Box2D v3가 조인트까지 soft-step 단일 솔버로 처리 — 01 §2 정정 반영]

use crate::body::{Body, BodyHandle};
use crate::solver::Softness;
use maru_collide::{in_state_range, vec_anchor_in_range, vec_in_state_range};
use maru_math::{Mat3, Quat, Vec3};

#[derive(Clone, Debug)]
pub enum JointKind {
    /// ball-socket: 앵커 일치 (3-DOF 위치 제약).
    Point,
    /// point + 축 정렬 (2-DOF 각 제약 추가). axis는 각 바디 로컬.
    ///
    /// `limit`이 있으면 축 둘레 각도를 `[min, max]`(라디안)로 묶고,
    /// `motor`가 있으면 목표 각속도(rad/s)를 최대 토크까지 추종한다.
    Hinge {
        axis_a: Vec3,
        axis_b: Vec3,
        limit: Option<[f32; 2]>,
        motor: Option<Motor>,
    },
    /// 완전 고정 — point + 3-DOF 각 제약. `rel_rot` = A 기준 B의 목표 상대 자세.
    Fixed { rel_rot: Quat },
    /// 두 앵커 사이 거리를 `[min, max]`로 묶는다 (로프·스프링·막대).
    ///
    /// `min == max`면 막대(양방향), 아니면 **부등식 제약**이라 구간 안에서는 자유롭다.
    Distance { min: f32, max: f32 },
}

/// 조인트 모터 — 목표 속도를 최대 힘/토크까지 추종한다.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motor {
    /// 목표 상대 각속도 (rad/s).
    pub target_speed: f32,
    /// 이 모터가 낼 수 있는 **최대 토크** (N·m).
    ///
    /// 임펄스가 아니라 토크인 이유: 누적 임펄스는 서브스텝마다 warm start로 다시
    /// 실리므로, 임펄스에 상한을 두면 **전달량이 서브스텝 수에 비례**한다
    /// (검수: substeps 1 vs 8에서 결과가 갈렸다). 서브스텝 간격 `h`로 스케일한
    /// `max_torque · h`를 누적 상한으로 쓰면 한 스텝 총량이 `max_torque · dt`로 고정된다.
    pub max_torque: f32,
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
    /// 각 제약 누적 임펄스. Hinge는 [0],[1]만, Fixed는 셋 다 쓴다.
    pub(crate) angular_impulse: [f32; 3],
    /// 한계(limit) 누적 임펄스 — 부등식 제약이라 부호가 한쪽으로 묶인다.
    pub(crate) limit_impulse: f32,
    /// 모터 누적 임펄스 — `max_impulse`로 클램프된다.
    pub(crate) motor_impulse: f32,
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
            || !self.angular_impulse.iter().all(|x| in_state_range(*x))
            || !in_state_range(self.limit_impulse)
            || !in_state_range(self.motor_impulse)
        {
            return Err(ApiError::InvalidValue);
        }
        match self.kind {
            JointKind::Hinge {
                axis_a,
                axis_b,
                limit,
                motor,
            } => {
                for a in [axis_a, axis_b] {
                    let l2 = a.length_sq();
                    if !(0.9..=1.1).contains(&l2) {
                        return Err(ApiError::InvalidValue);
                    }
                }
                if let Some([lo, hi]) = limit {
                    // 양성 조건 — NaN은 어느 비교도 통과하지 못한다.
                    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
                        return Err(ApiError::InvalidValue);
                    }
                    // 각도 한계는 ±2π 안에서만 의미가 있다(축 둘레 각은 atan2 치역).
                    if !(-core::f32::consts::TAU..=core::f32::consts::TAU).contains(&lo)
                        || !(-core::f32::consts::TAU..=core::f32::consts::TAU).contains(&hi)
                    {
                        return Err(ApiError::InvalidValue);
                    }
                }
                if let Some(m) = motor {
                    if !(m.max_torque >= 0.0 && in_state_range(m.max_torque))
                        || !maru_collide::angular_in_range(m.target_speed)
                    {
                        return Err(ApiError::InvalidValue);
                    }
                }
            }
            JointKind::Fixed { rel_rot } => {
                let l2 = rel_rot.x * rel_rot.x
                    + rel_rot.y * rel_rot.y
                    + rel_rot.z * rel_rot.z
                    + rel_rot.w * rel_rot.w;
                if !(0.9..=1.1).contains(&l2) {
                    return Err(ApiError::InvalidValue);
                }
            }
            JointKind::Distance { min, max } => {
                // 거리는 **앵커와 같은 차원**이라 같은 범위를 쓴다.
                if !(min >= 0.0 && min <= max && maru_collide::anchor_in_range(max)) {
                    return Err(ApiError::InvalidValue);
                }
            }
            JointKind::Point => {}
        }
        Ok(())
    }

    /// 누적 임펄스 초기화 (OOB 전이 등 연속성이 끊기는 지점).
    pub(crate) fn clear_impulses(&mut self) {
        self.linear_impulse = Vec3::ZERO;
        self.angular_impulse = [0.0; 3];
        self.limit_impulse = 0.0;
        self.motor_impulse = 0.0;
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
            angular_impulse: [0.0; 3],
            limit_impulse: 0.0,
            motor_impulse: 0.0,
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
            kind: JointKind::Hinge {
                axis_a,
                axis_b,
                limit: None,
                motor: None,
            },
            ..Self::try_point(body_a, body_b, anchor_a, anchor_b)?
        };
        j.validate()?;
        Ok(j)
    }
}

/// 선형 3-DOF의 K 행렬. 솔버와 삽입 검증이 **같은 식**을 쓴다.
impl Joint {
    /// 완전 고정 조인트 — 두 앵커를 붙이고 상대 자세를 `rel_rot`으로 묶는다.
    pub fn try_fixed(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rel_rot: Quat,
    ) -> Result<Self, crate::world::ApiError> {
        let mut j = Self::try_point(body_a, body_b, anchor_a, anchor_b)?;
        j.kind = JointKind::Fixed { rel_rot };
        j.validate()?;
        Ok(j)
    }

    /// 거리 조인트 — 두 앵커 사이 거리를 `[min, max]`로 묶는다.
    /// `min == max`면 막대, `min = 0`이면 로프(당기기만 한다).
    pub fn try_distance(
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        min: f32,
        max: f32,
    ) -> Result<Self, crate::world::ApiError> {
        let mut j = Self::try_point(body_a, body_b, anchor_a, anchor_b)?;
        j.kind = JointKind::Distance { min, max };
        j.validate()?;
        Ok(j)
    }

    /// hinge에 각도 한계를 단다 (라디안, 축 둘레).
    pub fn with_limit(mut self, min: f32, max: f32) -> Result<Self, crate::world::ApiError> {
        match &mut self.kind {
            JointKind::Hinge { limit, .. } => *limit = Some([min, max]),
            _ => return Err(crate::world::ApiError::InvalidValue),
        }
        self.validate()?;
        Ok(self)
    }

    /// hinge에 모터를 단다.
    pub fn with_motor(
        mut self,
        target_speed: f32,
        max_torque: f32,
    ) -> Result<Self, crate::world::ApiError> {
        match &mut self.kind {
            JointKind::Hinge { motor, .. } => {
                *motor = Some(Motor {
                    target_speed,
                    max_torque,
                })
            }
            _ => return Err(crate::world::ApiError::InvalidValue),
        }
        self.validate()?;
        Ok(self)
    }
}

pub(crate) fn linear_k(inv_ma: f32, inv_mb: f32, ia: &Mat3, ib: &Mat3, ra: Vec3, rb: Vec3) -> Mat3 {
    let mut m = Mat3::from_diagonal(Vec3::new(inv_ma + inv_mb, inv_ma + inv_mb, inv_ma + inv_mb));
    let e = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    for (mat, r) in [(ia, ra), (ib, rb)] {
        for (ji, ej) in e.iter().enumerate() {
            // K += -skew(r) I⁻¹ skew(r) — skew(r)ᵀ = -skew(r)이므로 **더한다**.
            // (함수로 뽑으며 부호를 뒤집어, 검증이 솔버와 다른 행렬을 보고 있었다 — 리뷰 BLOCKER)
            let col = r.cross(mat.mul_vec(ej.cross(r)));
            m.c[ji] += col;
        }
    }
    m
}

/// K의 역행렬. 특이하면 **정규화(Tikhonov)** 해서 반드시 풀리게 만든다.
///
/// 삽입·복원·`set_pose`에서 조건수를 검사해도 **정상 적분 중 자유 회전**으로 조건수가
/// 임계 아래로 내려갈 수 있다(리뷰 BLOCKER: point joint는 회전을 구속하지 않는다).
/// 그때 `Mat3::inverse`가 0 행렬을 돌려주면 제약이 **조용히 풀린다** — 가장 나쁜 실패다.
/// 대각에 미세 항을 더하면 조건수가 유계가 되고, 물리적으로는 그 축만 아주 약간
/// 무르게 만드는 것이라 잘 조건화된 K에서는 측정 가능한 차이가 없다.
pub(crate) fn solve_k(k: &Mat3) -> Mat3 {
    let inv = k.inverse();
    let nonzero = [inv.c[0], inv.c[1], inv.c[2]]
        .iter()
        .any(|c| c.x != 0.0 || c.y != 0.0 || c.z != 0.0);
    if nonzero {
        return inv;
    }
    // 대각 규모에 비례한 정규화 — 스케일 무관(절대 상수 금지, 톨러런스 정책과 동일 원칙)
    let scale = (k.c[0].x.abs() + k.c[1].y.abs() + k.c[2].z.abs()) / 3.0;
    let eps = if scale > 0.0 { scale * 1.0e-4 } else { 1.0e-12 };
    let mut r = *k;
    r.c[0].x += eps;
    r.c[1].y += eps;
    r.c[2].z += eps;
    r.inverse()
}

/// 삽입 시점의 K가 실제로 역행렬을 가지는가. `Mat3::inverse`는 실패하면 **0 행렬**을
/// 돌려주므로(조용한 실패), 그 값을 직접 확인한다.
pub(crate) fn k_is_solvable(bodies: &[Body], j: &Joint) -> bool {
    let (a, b) = (j.ia(), j.ib());
    let (ia, ib) = (bodies[a].inv_inertia_world(), bodies[b].inv_inertia_world());
    let ra = bodies[a].pose.rot.rotate(j.anchor_a);
    let rb = bodies[b].pose.rot.rotate(j.anchor_b);
    let k = linear_k(bodies[a].inv_mass, bodies[b].inv_mass, &ia, &ib, ra, rb);
    let inv = k.inverse();
    let any = [inv.c[0], inv.c[1], inv.c[2]]
        .iter()
        .any(|c| c.x != 0.0 || c.y != 0.0 || c.z != 0.0);
    any && inv.c[0].is_finite() && inv.c[1].is_finite() && inv.c[2].is_finite()
}

/// warm start (서브스텝 시작마다 — 접촉과 동일 패턴).
pub fn warm_start(st: &mut crate::solver::SolverState, j: &Joint) {
    let ra = st.rot[j.ia()].rotate(j.anchor_a);
    let rb = st.rot[j.ib()].rotate(j.anchor_b);
    match j.kind {
        // Distance는 **1축**이라 누적 임펄스가 스칼라다(.x에 담는다).
        JointKind::Distance { .. } => {
            if let Some(n) = distance_axis(st, j) {
                let imp = n * j.linear_impulse.x;
                st.apply_pub(j.ia(), imp, ra);
                st.apply_pub(j.ib(), -imp, rb);
            }
        }
        _ => {
            st.apply_pub(j.ia(), j.linear_impulse, ra);
            st.apply_pub(j.ib(), -j.linear_impulse, rb);
        }
    }
    match j.kind {
        JointKind::Hinge { axis_a, .. } => {
            let (u, v) = perp_basis(st.rot[j.ia()].rotate(axis_a));
            let ang = u * j.angular_impulse[0] + v * j.angular_impulse[1];
            st.apply_ang_pub(j.ia(), ang);
            st.apply_ang_pub(j.ib(), -ang);
            // 한계·모터는 축 둘레 1축이다.
            let axis = st.rot[j.ia()].rotate(axis_a);
            let ang = axis * (j.limit_impulse + j.motor_impulse);
            st.apply_ang_pub(j.ia(), ang);
            st.apply_ang_pub(j.ib(), -ang);
        }
        JointKind::Fixed { .. } => {
            // 3축 전부 — 월드 기저(x,y,z)에 그대로 담아둔다.
            let ang = Vec3::new(
                j.angular_impulse[0],
                j.angular_impulse[1],
                j.angular_impulse[2],
            );
            st.apply_ang_pub(j.ia(), ang);
            st.apply_ang_pub(j.ib(), -ang);
        }
        _ => {}
    }
}

/// 두 앵커를 잇는 단위 방향(A→B의 반대 = B에서 A로). 퇴화하면 `None`.
fn distance_axis(st: &crate::solver::SolverState, j: &Joint) -> Option<Vec3> {
    let ra = st.rot[j.ia()].rotate(j.anchor_a);
    let rb = st.rot[j.ib()].rotate(j.anchor_b);
    let d = (st.pos[j.ia()] + ra) - (st.pos[j.ib()] + rb);
    let len = d.length();
    if len > 1e-6 {
        Some(d * (1.0 / len))
    } else {
        None
    }
}

/// 축에 수직인 결정적 2축 기저.
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

/// 서브스텝 해석. `soft`=(bias 파라미터, 최대 bias 속도, 최대 각 bias), None=relax.
///
/// **bias를 차원별로 클램프한다**(리뷰 BLOCKER ×2): 선형 bias는 `max_bias_speed`(m/s),
/// **각 bias는 각속도 상한**(rad/s). `c × bias_rate`는 오차가 크고 h가 작을 때 1e12까지 가고,
/// 유효(관성)질량을 곱하면 파생 상태 상한을 넘긴다.
pub fn solve(
    st: &mut crate::solver::SolverState,
    j: &mut Joint,
    h: f32,
    soft: Option<(&Softness, f32, f32)>,
) {
    let (ia, ib) = (st.inv_i[j.ia()], st.inv_i[j.ib()]);
    let (inv_ma, inv_mb) = (st.inv_m[j.ia()], st.inv_m[j.ib()]);
    let ra = st.rot[j.ia()].rotate(j.anchor_a);
    let rb = st.rot[j.ib()].rotate(j.anchor_b);

    // --- Distance: 1축 **부등식** 제약이라 위 3-DOF 경로를 타지 않는다 ---
    if let JointKind::Distance { min, max } = j.kind {
        solve_distance(st, j, ra, rb, min, max, &ia, &ib, inv_ma, inv_mb, soft);
        let _ = h;
        return;
    }

    // --- 선형 3-DOF: K = (1/ma+1/mb)I + skew(ra)ᵀIa⁻¹skew(ra) + skew(rb)ᵀIb⁻¹skew(rb) ---
    // **삽입 검증과 같은 함수**를 쓴다. 중복 구현이 있으면 둘이 갈라져 검증이 무의미해진다.
    let k = linear_k(inv_ma, inv_mb, &ia, &ib, ra, rb);
    let k_inv = solve_k(&k);
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
    let impulse = k_inv.mul_vec(-(cdot + bias)) * mass_scale - j.linear_impulse * impulse_scale;
    j.linear_impulse += impulse;
    st.apply_pub(j.ia(), impulse, ra);
    st.apply_pub(j.ib(), -impulse, rb);

    // --- hinge 각 2-DOF: 축 수직 성분 상대 각속도 = 0, 위치 bias = 축 어긋남 ---
    if let JointKind::Hinge { axis_a, axis_b, .. } = j.kind {
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

    // --- hinge 한계·모터: 축 둘레 1-DOF ---
    if let JointKind::Hinge {
        axis_a,
        axis_b,
        limit,
        motor,
    } = j.kind
    {
        let axis = st.rot[j.ia()].rotate(axis_a);
        let k_ang = axis.dot(ia.mul_vec(axis)) + axis.dot(ib.mul_vec(axis));
        if k_ang > 0.0 {
            let wrel = axis.dot(st.ang[j.ia()] - st.ang[j.ib()]);

            // 모터: 목표 상대 각속도를 최대 토크까지 추종한다(등식 제약 + 클램프).
            if let Some(m) = motor {
                let want = -(wrel - m.target_speed) / k_ang;
                let old = j.motor_impulse;
                // **누적을 서브스텝 간격으로 스케일한 상한에 클램프한다.**
                // 증분만 자르면 상한을 넘고, 상한을 임펄스로 두면 전달량이 서브스텝
                // 수에 비례한다 — warm start가 누적분을 매 서브스텝 다시 싣기 때문이다
                // (검수: substeps 1 vs 8에서 결과가 갈렸다).
                let cap = m.max_torque * h;
                j.motor_impulse = (old + want).clamp(-cap, cap);
                let ang = axis * (j.motor_impulse - old);
                st.apply_ang_pub(j.ia(), ang);
                st.apply_ang_pub(j.ib(), -ang);
            }

            // 한계: 부등식이라 **한쪽 부호로만** 임펄스가 산다.
            //
            // 부호 유도(이 저장소가 두 번 틀린 지점이라 적어둔다):
            // 임펄스 `imp`를 축 방향으로 A += / B -= 적용하면 `wrel(=ω_a−ω_b)`이
            // `k_ang·imp` 만큼 는다. twist의 변화율은 `ω_b−ω_a = −wrel`이므로,
            // 오차 `c`를 줄이려면 `d(twist)/dt = −β·c`, 즉 `wrel → +β·c`가 목표다.
            // 따라서 `imp = (β·c − wrel)/k_ang = −(wrel + bias)/k_ang`,  **bias = −β·c**.
            // 그리고 c > 0(위 한계 초과)이면 목표 wrel이 양수라 **임펄스도 양수**다 —
            // 게이트는 `c`의 부호를 그대로 따른다.
            if let Some([lo, hi]) = limit {
                let angle = hinge_angle(st, j, axis_a, axis_b);
                let c = if angle < lo {
                    angle - lo
                } else if angle > hi {
                    angle - hi
                } else {
                    0.0
                };
                if c != 0.0 {
                    let wrel = axis.dot(st.ang[j.ia()] - st.ang[j.ib()]);
                    let bias = match soft {
                        Some((sf, _, max_ang_bias)) => {
                            (-(c * sf.bias_rate)).clamp(-max_ang_bias, max_ang_bias)
                        }
                        None => 0.0,
                    };
                    let want = -(wrel + bias) / k_ang;
                    let old = j.limit_impulse;
                    let acc = if c > 0.0 {
                        (old + want).max(0.0)
                    } else {
                        (old + want).min(0.0)
                    };
                    j.limit_impulse = acc;
                    let ang = axis * (acc - old);
                    st.apply_ang_pub(j.ia(), ang);
                    st.apply_ang_pub(j.ib(), -ang);
                } else if j.limit_impulse != 0.0 {
                    // 한계 안으로 돌아왔다 — warm start가 실은 임펄스를 되돌린다.
                    let ang = axis * (-j.limit_impulse);
                    st.apply_ang_pub(j.ia(), ang);
                    st.apply_ang_pub(j.ib(), -ang);
                    j.limit_impulse = 0.0;
                }
            }
        }
    }

    // --- Fixed: 3-DOF 각 제약 (상대 자세를 목표로 되돌린다) ---
    if let JointKind::Fixed { rel_rot } = j.kind {
        // 오차 = 현재 상대 자세와 목표의 차. 소각 근사로 축각 벡터를 만든다.
        let qa = st.rot[j.ia()];
        let qb = st.rot[j.ib()];
        let want = qa * rel_rot; // B가 있어야 할 자세
                                 // **부호 방향에 주의**: `qb → want`(B를 되돌리는 방향)여야 한다.
                                 // 반대로 두면 아래 임펄스 규약(`imp = -(cdot+bias)/k`, A += / B -=)이
                                 // `dC/dt = +β·C`를 만들어 **오차가 지수적으로 커진다**(검수 실측 0.0999 → 1.98).
        let err_q = want * qb.conjugate();
        // 쿼터니언 부호 정규화 — q와 −q는 같은 자세지만 오차 벡터의 부호를 뒤집는다.
        let s = if err_q.w < 0.0 { -1.0 } else { 1.0 };
        let err = Vec3::new(err_q.x, err_q.y, err_q.z) * (2.0 * s);
        let wrel = st.ang[j.ia()] - st.ang[j.ib()];
        for (axis_i, dir) in [
            (0usize, Vec3::new(1.0, 0.0, 0.0)),
            (1, Vec3::new(0.0, 1.0, 0.0)),
            (2, Vec3::new(0.0, 0.0, 1.0)),
        ] {
            let k_ang = dir.dot(ia.mul_vec(dir)) + dir.dot(ib.mul_vec(dir));
            if k_ang <= 0.0 {
                continue;
            }
            let cdot = dir.dot(wrel);
            let c = dir.dot(err);
            let (bias, ms, is_) = match soft {
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

/// hinge 축 둘레 상대 각도 (라디안) — **상대 자세의 twist 성분**.
///
/// 기준(0 rad)은 **두 바디의 상대 자세가 항등일 때**다. 문·팔을 정렬된 자세로 만들었으면
/// 그 자세가 0이 된다.
///
/// 처음엔 `perp_basis(축)`으로 각 바디의 수직 기준을 만들어 각을 쟀는데 **항상 0이었다**:
/// `perp_basis`는 축만의 순수 함수이고 경첩 제약이 두 축을 평행하게 유지하므로 두 기저가
/// 늘 같아진다. 기준 벡터는 **바디에 붙어 있어야** 한다 — 그래서 축이 아니라 상대 자세에서
/// 뽑는다(swing-twist 분해의 twist 성분).
fn hinge_angle(st: &crate::solver::SolverState, j: &Joint, axis_a: Vec3, _axis_b: Vec3) -> f32 {
    // `q_rel` = A 프레임에서 본 B의 자세. `axis_a`도 A 로컬이라 그대로 투영한다.
    let q_rel = st.rot[j.ia()].conjugate() * st.rot[j.ib()];
    let s = q_rel.x * axis_a.x + q_rel.y * axis_a.y + q_rel.z * axis_a.z;
    2.0 * s.atan2(q_rel.w)
}

/// Distance: 두 앵커 거리를 `[min, max]`로 묶는다.
///
/// **부등식**이라 구간 안에서는 임펄스가 0이어야 한다 — 등식으로 풀면 로프가 막대가 된다.
#[allow(clippy::too_many_arguments)]
fn solve_distance(
    st: &mut crate::solver::SolverState,
    j: &mut Joint,
    ra: Vec3,
    rb: Vec3,
    min: f32,
    max: f32,
    ia: &Mat3,
    ib: &Mat3,
    inv_ma: f32,
    inv_mb: f32,
    soft: Option<(&Softness, f32, f32)>,
) {
    let d = (st.pos[j.ia()] + ra) - (st.pos[j.ib()] + rb);
    let len = d.length();
    if len <= 1e-6 {
        return; // 축이 퇴화 — 이번 서브스텝은 건너뛴다
    }
    let n = d * (1.0 / len);
    let (c, gate) = if len < min {
        (len - min, 1.0f32)
    } else if len > max {
        (len - max, -1.0f32)
    } else {
        // **구간 안으로 돌아왔다 — 그냥 0으로 두면 안 된다.** 이 서브스텝의 warm start가
        // 이미 옛 누적 임펄스를 속도에 실었으므로, 되돌리는 델타(0 − old)를 적용해야
        // 그 운동량이 남지 않는다 (검수: 부등식 게이트 × warm start).
        let old = j.linear_impulse.x;
        if old != 0.0 {
            let imp = n * (-old);
            st.apply_pub(j.ia(), imp, ra);
            st.apply_pub(j.ib(), -imp, rb);
        }
        j.limit_impulse = 0.0;
        j.linear_impulse = Vec3::ZERO;
        return;
    };
    // 1축 유효질량: nᵀ K n
    let k = linear_k(inv_ma, inv_mb, ia, ib, ra, rb);
    let k_axis = n.dot(k.mul_vec(n));
    if k_axis <= 0.0 {
        return;
    }
    let cdot = n.dot(st.vel_at_pub(j.ia(), ra) - st.vel_at_pub(j.ib(), rb));
    let (bias, ms, is_) = match soft {
        Some((sf, max_bias, _)) => (
            (c * sf.bias_rate).clamp(-max_bias, max_bias),
            sf.mass_scale,
            sf.impulse_scale,
        ),
        None => (0.0, 1.0, 0.0),
    };
    let want = -(cdot + bias) / k_axis * ms - j.linear_impulse.x * is_;
    let old = j.linear_impulse.x;
    // 부호 게이트: 벗어난 방향으로 되돌리는 성분만 남긴다(로프는 당기기만 한다).
    let acc = if gate > 0.0 {
        (old + want).max(0.0)
    } else {
        (old + want).min(0.0)
    };
    j.linear_impulse = Vec3::new(acc, 0.0, 0.0);
    let imp = n * (acc - old);
    st.apply_pub(j.ia(), imp, ra);
    st.apply_pub(j.ib(), -imp, rb);
}
