//! 강체 — 타입 3종 시맨틱 (설계 §9).

use crate::world::ApiError;
use maru_collide::{
    in_range, inv_inertia_in_range, inv_mass_in_range, vec_angular_in_range, vec_extent_in_range,
    vec_in_range, vec_speed_in_range, Pose, Shape,
};
use maru_math::{Mat3, Vec3};

/// 기본 충돌 그룹 — 모든 비트가 켜져 있어 아무 설정 없이도 전부 부딪힌다.
pub const ALL_GROUPS: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyType {
    /// 불변. 아일랜드 미참여(anchor 전용 — 설계 §4.3).
    Static,
    Dynamic,
    /// 처방 속도로 구동, inv_mass=0이지만 접촉 상대속도에 기여 (설계 §9).
    Kinematic,
}

/// 크레이트 밖에서는 필드 대입이 **컴파일되지 않는다** (인코딩된 회귀 테스트):
///
/// ```compile_fail
/// use maru_collide::{Pose, Shape};
/// use maru_dynamics::Body;
/// use maru_math::{Quat, Vec3};
/// let mut b = Body::new_dynamic(
///     Shape::Sphere { radius: 1.0 },
///     Pose { pos: Vec3::ZERO, rot: Quat::IDENTITY },
///     1.0,
/// );
/// b.inv_mass = 0.0; // ← private field: 컴파일 오류
/// ```
///
/// **불변식 보유 필드는 전부 비공개**다(리뷰 BLOCKER). `#[non_exhaustive]`는 구조체 리터럴만
/// 막을 뿐, 생성자로 얻은 값의 공개 필드는 밖에서 얼마든지 대입할 수 있다 —
/// `Body::new_dynamic(...)` 뒤에 `b.shape = Shape::Hull { points: vec![], .. }`로 바꿔
/// `add_body`하면 첫 `local_aabb()`에서 패닉한다. 읽기는 동명 게터로 제공한다
/// (필드와 메서드는 네임스페이스가 달라 크레이트 안 코드는 그대로 필드를 쓴다).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Body {
    pub(crate) body_type: BodyType,
    pub(crate) pose: Pose,
    pub(crate) linvel: Vec3,
    pub(crate) angvel: Vec3,
    pub(crate) shape: Shape,
    pub(crate) inv_mass: f32,
    /// 로컬 역관성 대각 (월드 역관성은 매 서브스텝 회전).
    pub(crate) inv_inertia_local: Vec3,
    pub(crate) friction: f32,
    pub(crate) restitution: f32,
    /// 재중심화 보정: **원래 셰이프 원점**이 질량중심 로컬 좌표에서 어디였는가.
    /// 호스트는 자기 정점 버퍼를 `pose.pos + pose.rot * shape_origin_offset`에 그린다.
    /// hull이 아니거나 이미 중심이 맞으면 0 (설계: pose.pos = 질량중심).
    pub(crate) shape_origin_offset: Vec3,
    /// 다음 스텝에 적용할 **월드 좌표 힘** 누적. 스텝 끝에 0으로 지워진다.
    ///
    /// 캐노니컬 상태다 — `apply_force` 뒤 `step` 전에 스냅샷을 뜨면 그 힘까지 담겨야
    /// 복원된 월드가 같은 미래를 만든다(빠지면 해시는 같은데 다음 스텝이 갈린다).
    pub(crate) force: Vec3,
    /// 다음 스텝에 적용할 **월드 좌표 토크** 누적. 스텝 끝에 0으로 지워진다.
    pub(crate) torque: Vec3,
    /// 이 물체가 **속한** 충돌 그룹 비트.
    pub(crate) membership: u32,
    /// 이 물체가 **부딪히려는** 그룹 비트.
    ///
    /// 충돌 조건은 **양방향**이다: `a.membership & b.filter != 0 && b.membership & a.filter != 0`.
    /// 한 방향만 보면 "A는 B를 무시하는데 B는 A를 민다"가 되어 운동량이 새고,
    /// 호스트 입장에서는 설정한 적 없는 비대칭이 생긴다.
    pub(crate) filter: u32,
    /// 감지 전용(트리거). 접촉 이벤트는 내지만 임펄스는 주고받지 않는다.
    pub(crate) sensor: bool,
    /// 선형 감쇠 (1/s). 매 서브스텝 `v ← v / (1 + h·d)`로 **암시적**으로 적용한다.
    ///
    /// 명시적(`v *= 1 - h·d`)이면 `h·d > 1`에서 부호가 뒤집혀 발산한다 —
    /// 호스트가 큰 값을 넣었을 때 조용히 폭발하는 대신 그냥 더 강하게 멈춘다.
    pub(crate) linear_damping: f32,
    /// 각 감쇠 (1/s). 같은 암시적 형태.
    pub(crate) angular_damping: f32,
}

impl Body {
    #[inline]
    pub fn body_type(&self) -> BodyType {
        self.body_type
    }
    #[inline]
    pub fn pose(&self) -> Pose {
        self.pose
    }
    #[inline]
    pub fn linvel(&self) -> Vec3 {
        self.linvel
    }
    #[inline]
    pub fn angvel(&self) -> Vec3 {
        self.angvel
    }
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    #[inline]
    pub fn inv_mass(&self) -> f32 {
        self.inv_mass
    }
    #[inline]
    pub fn inv_inertia_local(&self) -> Vec3 {
        self.inv_inertia_local
    }
    #[inline]
    pub fn friction(&self) -> f32 {
        self.friction
    }
    /// 이 물체가 속한 충돌 그룹 비트.
    #[inline]
    pub fn membership(&self) -> u32 {
        self.membership
    }
    /// 이 물체가 부딪히려는 그룹 비트.
    #[inline]
    pub fn filter(&self) -> u32 {
        self.filter
    }
    /// 감지 전용(트리거)인가.
    #[inline]
    pub fn is_sensor(&self) -> bool {
        self.sensor
    }
    /// 충돌 그룹을 바꾼 사본.
    pub fn with_groups(mut self, membership: u32, filter: u32) -> Self {
        self.membership = membership;
        self.filter = filter;
        self
    }
    /// 감지 전용으로 바꾼 사본.
    pub fn as_sensor(mut self, sensor: bool) -> Self {
        self.sensor = sensor;
        self
    }
    /// 선형 감쇠 (1/s).
    #[inline]
    pub fn linear_damping(&self) -> f32 {
        self.linear_damping
    }
    /// 각 감쇠 (1/s).
    #[inline]
    pub fn angular_damping(&self) -> f32 {
        self.angular_damping
    }
    /// 감쇠를 바꾼 사본. 음수·NaN·상한 초과는 오류다.
    pub fn with_damping(mut self, linear: f32, angular: f32) -> Result<Self, ApiError> {
        self.linear_damping = linear;
        self.angular_damping = angular;
        self.validate()?;
        Ok(self)
    }
    /// 다음 스텝에 적용될 누적 힘 (월드 좌표).
    #[inline]
    pub fn force(&self) -> Vec3 {
        self.force
    }
    /// 다음 스텝에 적용될 누적 토크 (월드 좌표).
    #[inline]
    pub fn torque(&self) -> Vec3 {
        self.torque
    }
    pub fn restitution(&self) -> f32 {
        self.restitution
    }
    /// 바디 불변식 — **삽입 경계에서 다시** 강제한다. 생성자를 통과한 값이라도
    /// 공개 API 밖에서 필드가 바뀌었을 수 있고, 역직렬화 경로는 생성자를 거치지 않는다.
    pub(crate) fn validate(&self) -> Result<(), crate::world::ApiError> {
        use crate::world::ApiError;
        self.shape.validate().map_err(|_| ApiError::InvalidValue)?;
        // 크기 상한(MAX_INPUT)까지 여기서 강제한다 — 스냅샷 리더와 **같은 정책**이라야
        // `restore(snapshot())`가 항상 성립한다(리뷰 BLOCKER: 비대칭).
        if !vec_in_range(self.pose.pos) || self.pose.rot.try_normalize().is_none() {
            return Err(ApiError::InvalidValue);
        }
        if !vec_speed_in_range(self.linvel) || !vec_angular_in_range(self.angvel) {
            return Err(ApiError::InvalidValue);
        }
        if !vec_extent_in_range(self.shape_origin_offset) {
            return Err(ApiError::InvalidValue);
        }
        // 비교는 양성 조건으로 — `!(x < 0.0)`은 NaN에서 참이라 통과시킨다
        if !(self.friction >= 0.0 && in_range(self.friction))
            || !(0.0..=1.0).contains(&self.restitution)
        {
            return Err(ApiError::InvalidValue);
        }
        // 힘·토크는 여러 번 더해져 자라므로 **파생 상태 범위**로 본다
        // (입력 한 건은 setter가 입력 범위로 거르고, 누적분은 포화시킨다).
        if !maru_collide::vec_in_state_range(self.force)
            || !maru_collide::vec_in_state_range(self.torque)
        {
            return Err(ApiError::InvalidValue);
        }
        // 감쇠는 비음수 유한값 (양성 조건으로 비교 — `!(x < 0)`은 NaN을 통과시킨다)
        if !(self.linear_damping >= 0.0 && in_range(self.linear_damping))
            || !(self.angular_damping >= 0.0 && in_range(self.angular_damping))
        {
            return Err(ApiError::InvalidValue);
        }
        let inv_i = self.inv_inertia_local;
        match self.body_type {
            BodyType::Dynamic => {
                // 질량·관성은 **양쪽**으로 묶는다: 너무 가벼우면 역수가 폭발하고,
                // 너무 무거우면(1e9 kg 초과) 그 임펄스가 파생 상한을 위협한다.
                if !(inv_mass_in_range(self.inv_mass)
                    && inv_inertia_in_range(inv_i.x)
                    && inv_inertia_in_range(inv_i.y)
                    && inv_inertia_in_range(inv_i.z))
                {
                    return Err(ApiError::InvalidValue);
                }
                // **질량·관성이 셰이프와 물리적으로 정합**해야 한다(리뷰 BLOCKER).
                // 범위만 따로 보면 반지름 1mm 구에 inv_mass=1e-6·inv_I=1e12 같은 조합이
                // 통과하고, 그 조인트 K는 diag(1e-6, 1.6e7, 1.6e7)이 되어 정규화 det가
                // 임계 아래로 떨어져 `Mat3::inverse`가 0 행렬을 돌려준다 —
                // 조인트가 오류 없이 조용히 풀린다.
                //
                // 균질 강체는 축별로 `I ≈ k·m·(수직 두 방향 반크기의 제곱합)`이고
                // 우리 생성자의 k는 1/3(박스)·0.4(구) 수준이다. 넉넉한 [0.1, 1000] 띠로
                // 형상 다양성(막대·판)은 허용하면서 비물리적 주입만 막는다.
                if !inertia_matches_shape(self) {
                    return Err(ApiError::InvalidValue);
                }
            }
            BodyType::Static | BodyType::Kinematic => {
                if self.inv_mass != 0.0 || inv_i != Vec3::ZERO {
                    return Err(ApiError::InvalidValue);
                }
            }
        }
        Ok(())
    }

    /// 엔진이 쓰는 셰이프의 원점이 **호스트가 준 원점과 다를 수 있다**(hull 재중심화).
    /// 렌더러는 이 오프셋을 적용해야 시각 형상과 물리 형상이 일치한다.
    #[inline]
    pub fn shape_origin_offset(&self) -> Vec3 {
        self.shape_origin_offset
    }
}

impl Body {
    /// 동적 바디 생성. 질량·관성이 유한 양수가 아니면 **생성 경계에서 거부**한다
    /// (설계 §8 위생: inv_mass=∞는 0·∞=NaN으로 속도를 오염시킨다 — 리뷰 지적).
    /// 동적 바디 생성 (오류 반환형 — **M3 바인딩의 정본 경로**).
    /// 생성자가 자기 삽입 정책을 위반하는 값을 만들 수 있었다(리뷰 NEW MAJOR):
    /// 반지름 1m 구에 `density=1e-10`이면 질량·역수는 유한 양수라 통과하지만
    /// `inv_mass ≈ 2.39e9`가 입력 상한을 넘어 뒤따르는 `add_body`가 패닉했다.
    /// **최종 Body에 `validate()`를 걸어** 생성 경계에서 끝낸다.
    pub fn try_new_dynamic(shape: Shape, pose: Pose, density: f32) -> Result<Self, ApiError> {
        if !(in_range(density) && density > 0.0) {
            return Err(ApiError::InvalidValue);
        }
        shape.validate().map_err(|_| ApiError::InvalidValue)?;
        if !vec_in_range(pose.pos) || pose.rot.try_normalize().is_none() {
            return Err(ApiError::InvalidValue);
        }
        let pose = Pose {
            pos: pose.pos,
            rot: pose.rot.normalize(),
        };
        let (shape, pose, shape_origin_offset) = recenter_hull(shape, pose);
        let (mass, inertia) = mass_properties(&shape, density);
        if !(mass.is_finite() && mass > 0.0) {
            return Err(ApiError::InvalidValue);
        }
        let inv_mass = 1.0 / mass;
        let inv_inertia = Vec3::new(1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z);
        let b = Self {
            body_type: BodyType::Dynamic,
            pose,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass,
            inv_inertia_local: inv_inertia,
            friction: 0.6,
            restitution: 0.0,
            shape_origin_offset,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            membership: ALL_GROUPS,
            filter: ALL_GROUPS,
            sensor: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
        };
        b.validate()?;
        Ok(b)
    }

    /// 마찰 계수를 바꾼 사본. 범위 밖(음수·NaN·`MAX_INPUT` 초과)이면 오류다.
    ///
    /// 재질은 생성 시점에 0.6/0.0으로 고정돼 있었고 호스트가 바꿀 방법이 없었다 —
    /// 물리엔진의 API 결손이다. 다른 쓰기 경로와 같은 규율을 따른다:
    /// **바꾼 뒤 `validate`를 다시 통과해야 값이 확정된다.**
    pub fn with_friction(mut self, friction: f32) -> Result<Self, ApiError> {
        self.friction = friction;
        self.validate()?;
        Ok(self)
    }

    /// 반발 계수를 바꾼 사본. `0.0..=1.0` 밖이면 오류다.
    pub fn with_restitution(mut self, restitution: f32) -> Result<Self, ApiError> {
        self.restitution = restitution;
        self.validate()?;
        Ok(self)
    }

    /// 정적 바디 생성 (오류 반환형).
    pub fn try_new_static(shape: Shape, pose: Pose) -> Result<Self, ApiError> {
        shape.validate().map_err(|_| ApiError::InvalidValue)?;
        if !vec_in_range(pose.pos) || pose.rot.try_normalize().is_none() {
            return Err(ApiError::InvalidValue);
        }
        let b = Self {
            body_type: BodyType::Static,
            pose: Pose {
                pos: pose.pos,
                rot: pose.rot.normalize(),
            },
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass: 0.0,
            inv_inertia_local: Vec3::ZERO,
            friction: 0.6,
            restitution: 0.0,
            shape_origin_offset: Vec3::ZERO,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            membership: ALL_GROUPS,
            filter: ALL_GROUPS,
            sensor: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
        };
        b.validate()?;
        Ok(b)
    }

    pub fn new_dynamic(shape: Shape, pose: Pose, density: f32) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "density must be finite and positive, got {density}"
        );
        shape.validate().expect("invalid shape");
        let pose = normalized_pose(pose);
        // **hull 재중심화**: 솔버는 pose.pos를 질량중심으로 가정한다(r = 접점 − pose.pos).
        // 중심이 어긋난 hull은 평행축 항으로 대각 관성을 키워도 **관성곱(products of
        // inertia)**이 빠져 2축 이상 어긋나면 축이 틀린 텐서가 된다(리뷰 MAJOR).
        // 대각 텐서로 정확해지는 유일한 방법은 실제로 중심을 맞추는 것 — 점을 중심으로
        // 옮기고 pose를 같은 양만큼 월드에서 밀어 **같은 물체**를 유지한다.
        let (shape, pose, shape_origin_offset) = recenter_hull(shape, pose);
        let (mass, inertia) = mass_properties(&shape, density);
        assert!(
            mass.is_finite()
                && mass > 0.0
                && inertia.x.is_finite()
                && inertia.y.is_finite()
                && inertia.z.is_finite()
                && inertia.x > 0.0
                && inertia.y > 0.0
                && inertia.z > 0.0,
            "dynamic shape must have finite positive mass/inertia (mass={mass}, inertia={inertia:?})"
        );
        // 역수까지 검사해야 한다: subnormal 질량은 양수·유한이지만 1/mass가 inf가 되고,
        // 임펄스 적용의 0·inf가 다시 NaN을 만든다(리뷰 지적).
        let inv_mass = 1.0 / mass;
        let inv_inertia = Vec3::new(1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z);
        assert!(
            inv_mass.is_finite()
                && inv_mass > 0.0
                && inv_inertia.x.is_finite()
                && inv_inertia.y.is_finite()
                && inv_inertia.z.is_finite(),
            "mass/inertia too small: reciprocals must stay finite (mass={mass}, inertia={inertia:?})"
        );
        let b = Self {
            body_type: BodyType::Dynamic,
            pose,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass,
            inv_inertia_local: inv_inertia,
            friction: 0.6,
            restitution: 0.0,
            shape_origin_offset,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            membership: ALL_GROUPS,
            filter: ALL_GROUPS,
            sensor: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
        };
        // **생성자는 자기 삽입 정책을 위반한 값을 내보내지 않는다**(리뷰 NEW MAJOR):
        // density=1e-10인 반지름 1m 구는 위 검사를 다 통과하지만 inv_mass≈2.39e9로
        // 입력 상한을 넘어, 패닉이 `add_body`까지 미뤄졌었다.
        b.validate()
            .expect("constructed body violates the size policy (density/shape out of range)");
        b
    }

    pub fn new_static(shape: Shape, pose: Pose) -> Self {
        shape.validate().expect("invalid shape");
        let pose = normalized_pose(pose);
        let b = Self {
            body_type: BodyType::Static,
            pose,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass: 0.0,
            inv_inertia_local: Vec3::ZERO,
            friction: 0.6,
            restitution: 0.0,
            shape_origin_offset: Vec3::ZERO,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            membership: ALL_GROUPS,
            filter: ALL_GROUPS,
            sensor: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
        };
        b.validate()
            .expect("constructed body violates the size policy");
        b
    }

    pub fn new_kinematic(shape: Shape, pose: Pose) -> Self {
        Self::try_new_kinematic(shape, pose).expect("invalid kinematic body")
    }

    /// 기구학 바디 생성 (오류 반환형).
    pub fn try_new_kinematic(shape: Shape, pose: Pose) -> Result<Self, ApiError> {
        Ok(Self {
            body_type: BodyType::Kinematic,
            ..Self::try_new_static(shape, pose)?
        })
    }

    #[inline]
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }

    /// 월드 역관성 텐서 (현재 자세 기준).
    #[inline]
    pub fn inv_inertia_world(&self) -> Mat3 {
        if self.is_dynamic() {
            Mat3::rotate_inertia(self.inv_inertia_local, self.pose.rot)
        } else {
            Mat3::from_diagonal(Vec3::ZERO)
        }
    }

    /// 점 r(질량중심 기준 월드 오프셋)에서의 속도.
    #[inline]
    pub fn velocity_at(&self, r: Vec3) -> Vec3 {
        self.linvel + self.angvel.cross(r)
    }
}

/// 축별 `inv_I · (수직 반크기 제곱합) / inv_m`이 물리적으로 그럴듯한 띠 안인가.
fn inertia_matches_shape(b: &Body) -> bool {
    const LO: f32 = 0.1;
    const HI: f32 = 1000.0;
    let bb = b.shape.local_aabb();
    let h = Vec3::new(
        bb.min.x.abs().max(bb.max.x.abs()),
        bb.min.y.abs().max(bb.max.y.abs()),
        bb.min.z.abs().max(bb.max.z.abs()),
    );
    let perp = [
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ];
    let inv_i = [
        b.inv_inertia_local.x,
        b.inv_inertia_local.y,
        b.inv_inertia_local.z,
    ];
    for (ii, p) in inv_i.iter().zip(perp.iter()) {
        if *p <= 0.0 {
            return false;
        }
        let ratio = ii * p / b.inv_mass;
        if !(ratio.is_finite() && (LO..=HI).contains(&ratio)) {
            return false;
        }
    }
    true
}

/// 유한하고 **정규화 가능한** 자세만 통과시키고, 정규화된 값을 저장한다.
/// `is_finite`만 보면 `Quat{0,0,0,0}`이 통과해 support 방향·관성 회전이 무효화되고
/// dynamic은 첫 적분에서 갑자기 identity로 튄다(리뷰 NEW MAJOR).
fn normalized_pose(pose: Pose) -> Pose {
    assert!(
        vec_in_range(pose.pos),
        "pose.pos must be finite and within MAX_INPUT"
    );
    let rot = pose
        .rot
        .try_normalize()
        .expect("pose.rot must be finite and normalizable (zero/NaN quaternion rejected)");
    Pose { pos: pose.pos, rot }
}

/// hull의 AABB 중심을 원점으로 옮기고, 같은 양을 월드 자세에 반영한다(물체는 불변).
/// 결과: 질량중심이 pose.pos와 일치 → 평행축 항 0 → 관성곱 0 → 대각 텐서가 정확해진다.
/// 셋째 반환값 = **원래 셰이프 원점**의 질량중심 로컬 좌표(렌더 오프셋, 리뷰 MAJOR).
fn recenter_hull(shape: Shape, pose: Pose) -> (Shape, Pose, Vec3) {
    let Shape::Hull { points, radius } = shape else {
        return (shape, pose, Vec3::ZERO);
    };
    let (mut min, mut max) = (points[0], points[0]);
    for p in &points[1..] {
        min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
        max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
    }
    let c = (max + min) * 0.5;
    if c == Vec3::ZERO {
        return (Shape::Hull { points, radius }, pose, Vec3::ZERO);
    }
    let points = points.iter().map(|p| *p - c).collect();
    let pose = Pose {
        pos: pose.pos + pose.rot.rotate(c),
        rot: pose.rot,
    };
    (Shape::Hull { points, radius }, pose, -c)
}

/// (질량, 관성 대각). 캡슐은 원통+반구 근사 대신 v1은 동체 실린더 근사(문서화된 가정).
fn mass_properties(shape: &Shape, density: f32) -> (f32, Vec3) {
    match *shape {
        Shape::Sphere { radius } => {
            let m = density * 4.0 / 3.0 * core::f32::consts::PI * radius.powi(3);
            let i = 0.4 * m * radius * radius;
            (m, Vec3::new(i, i, i))
        }
        Shape::Box { half, .. } => {
            let m = density * 8.0 * half.x * half.y * half.z;
            let f = m / 3.0;
            (
                m,
                Vec3::new(
                    f * (half.y * half.y + half.z * half.z),
                    f * (half.x * half.x + half.z * half.z),
                    f * (half.x * half.x + half.y * half.y),
                ),
            )
        }
        Shape::Capsule {
            half_height,
            radius,
        } => {
            // [가정] v1: 실린더 근사 (h=2*(hh+r), r)
            let h = 2.0 * (half_height + radius);
            let m = density * core::f32::consts::PI * radius * radius * h;
            let ixz = m * (3.0 * radius * radius + h * h) / 12.0;
            let iy = 0.5 * m * radius * radius;
            (m, Vec3::new(ixz, iy, ixz))
        }
        Shape::Hull { ref points, .. } => {
            // [가정] v1: AABB 근사 — hull 정밀 적분은 hull 빌더와 함께.
            // 동적 바디는 `recenter_hull`을 이미 거쳐 d≈0이다(평행축 항은 방어적 잔존).
            // d≠0 경로로 들어오면 관성곱이 빠진 근사임을 알고 쓴다.
            let mut min = points[0];
            let mut max = points[0];
            for p in &points[1..] {
                min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
                max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
            }
            let half = (max - min) * 0.5;
            let center = (max + min) * 0.5;
            let (m, i_center) = mass_properties(&Shape::Box { half, radius: 0.0 }, density);
            let d = center;
            (
                m,
                Vec3::new(
                    i_center.x + m * (d.y * d.y + d.z * d.z),
                    i_center.y + m * (d.x * d.x + d.z * d.z),
                    i_center.z + m * (d.x * d.x + d.y * d.y),
                ),
            )
        }
    }
}

/// 핸들: index + 세대 (설계 §3 generational arena).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyHandle {
    pub index: u32,
    pub generation: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use maru_math::Quat;

    /// 리뷰 회귀: subnormal 질량은 검사를 통과하지만 역수가 inf → 0·inf = NaN
    #[test]
    #[should_panic(expected = "reciprocals must stay finite")]
    fn tiny_density_rejected_at_construction() {
        Body::new_dynamic(
            Shape::Sphere { radius: 1.0 },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            1.0e-40,
        );
    }

    /// 리뷰 회귀: 중심이 2축 이상 어긋난 hull은 대각 텐서로 표현 불가 →
    /// 생성 경계에서 재중심화해 pose.pos = 질량중심 계약을 참으로 만든다.
    #[test]
    fn offset_hull_is_recentered_to_com() {
        // 3D 두께가 있어야 한다 (회전 불변 퇴화 검사)
        let pts = vec![
            Vec3::new(9.5, 19.5, 29.5),
            Vec3::new(10.5, 19.5, 29.5),
            Vec3::new(9.5, 20.5, 29.5),
            Vec3::new(9.5, 19.5, 30.5),
            Vec3::new(10.5, 20.5, 30.5),
        ];
        let b = Body::new_dynamic(
            Shape::Hull {
                points: pts,
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            1.0,
        );
        // 자세가 질량중심으로 이동했다 (같은 월드 위치의 같은 물체)
        assert!((b.pose.pos - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-3);
        // 셰이프 점은 원점 중심
        let Shape::Hull { ref points, .. } = b.shape else {
            panic!()
        };
        let (mut mn, mut mx) = (points[0], points[0]);
        for p in &points[1..] {
            mn = Vec3::new(mn.x.min(p.x), mn.y.min(p.y), mn.z.min(p.z));
            mx = Vec3::new(mx.x.max(p.x), mx.y.max(p.y), mx.z.max(p.z));
        }
        let c = (mn + mx) * 0.5;
        assert!(c.length() < 1e-4, "재중심화 실패 {c:?}");
        // 관성은 1×1×1 중심 박스와 동일 — 평행축 항으로 부풀지 않는다
        let expect = 1.0f32 / 6.0;
        for v in [
            b.inv_inertia_local.x,
            b.inv_inertia_local.y,
            b.inv_inertia_local.z,
        ] {
            assert!(
                (1.0 / v - expect).abs() < 1e-3,
                "I={} expect {expect}",
                1.0 / v
            );
        }
    }

    #[test]
    fn box_inertia_analytic() {
        let b = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1.0, 2.0, 3.0),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            1.0,
        );
        let m = 8.0 * 6.0; // 2*4*6 * 1.0
        assert!((1.0 / b.inv_mass - m).abs() < 1e-3);
        let ix = m / 3.0 * (4.0 + 9.0);
        assert!((1.0 / b.inv_inertia_local.x - ix).abs() < 1e-2);
    }
}
