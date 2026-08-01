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
    /// **회전 잠금 마스크** — 비트 0/1/2 = 월드 X/Y/Z 축 회전을 막는다.
    ///
    /// rapier의 `LockedAxes` 회전 절반에 해당한다. 직립 캐릭터(전부 잠금)나 문·바퀴처럼
    /// 한 축만 도는 물체(둘 잠금)에 쓴다. 월드 축 기준이라 몸체가 돌아도 잠긴 축은 그대로다 —
    /// 그래서 로컬 관성에 구울 수 없고 월드 역관성에 투영으로 건다.
    pub(crate) locked_rot: u8,
    /// 지형 삼각형의 **내부 에지 비트마스크**(비트 i = 에지 (pᵢ, p₍ᵢ₊₁₎%₃)이 이웃
    /// 삼각형과 공면 공유 — 접점 법선을 면 법선으로 용접한다, §4.2 ghost edge).
    /// 0 = 용접 없음(일반 바디 전부). 비영은 **static 3점 hull에만** 허용(validate).
    /// 캐노니컬 상태다 — 법선을 바꾸므로 스냅샷에 빠지면 복원 궤적이 갈린다.
    pub(crate) weld_edges: u8,
    /// **접촉력 이벤트 임계**(뉴턴). 이 물체가 낀 접촉의 총 힘이 임계를 넘으면
    /// `World::contact_force_events()`에 실린다.
    ///
    /// 기본은 `f32::INFINITY` = **꺼짐**이다. rapier는 0을 기본으로 두고 별도 opt-in
    /// 플래그를 요구하는데, 여기서는 임계 자체가 opt-in이라 플래그가 하나 준다.
    /// 꺼져 있으면 합산 자체를 건너뛰므로 안 쓰는 호스트는 비용이 0이다.
    ///
    /// 페어 임계는 두 물체 임계의 **min**이다 — 더 민감한 쪽이 이긴다. 한쪽만 알고 싶어
    /// 켰는데 상대가 꺼져 있어 못 듣는 것이 더 놀랍기 때문이다.
    pub(crate) contact_force_threshold: f32,
    /// CCD linear cast opt-in (설계 §4 CCD 계약 v0.5, 대상 static 한정).
    /// 고속 소형 바디(총알)가 얇은 static을 관통하지 않도록 서브스텝 적분마다
    /// 이동 경로를 static 트리에 캐스트해 TOI에서 위치를 절단한다. dynamic 전용 의미 —
    /// 다른 타입에서는 무시된다(kinematic은 처방 이동이라 절단이 계약 위반).
    pub(crate) ccd: bool,
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
    /// CCD 켜기 — 고속 dynamic 바디용. static 대상 관통만 막는다.
    ///
    /// 선형 이동은 linear cast(설계 §4 CCD v0.5), **회전 스윕은 회전 보수적 전진**
    /// (예약 확장 구현): 서브스텝 총각 × 외접 반지름이 speculative margin의 1/4을 넘으면
    /// 회전 인지 캐스트로 전환하고, 히트 시 그 서브스텝의 회전도 같은 TOI에서 절단한다
    /// (속도 불변 — 반발은 다음 스텝 일반 파이프라인 소관). 각 기여 상한이 보수적이라
    /// 극고속 스핀에서는 반복 상한의 보수적 히트로 전진이 일찍 잘릴 수 있다(관통 대신
    /// 감속 방향의 실패 — 의도된 방어).
    pub fn with_ccd(mut self) -> Self {
        self.ccd = true;
        self
    }

    pub fn ccd(&self) -> bool {
        self.ccd
    }

    /// **월드 축 회전 잠금** (rapier `LockedAxes`의 회전 절반).
    ///
    /// 잠근 축으로는 각속도가 생기지 않는다 — 유효 역관성을 그 축에서 0으로 투영하므로
    /// 솔버가 "그 방향으로는 무한 관성"으로 보고 임펄스를 계산한다. 속도를 사후에 0으로
    /// 지우는 방식과 다르다: 그러면 솔버가 매 반복 헛되이 밀어 수렴이 나빠진다.
    ///
    /// 정적·kinematic 바디는 애초에 역관성이 0이라 영향이 없다.
    pub fn with_locked_rotation(mut self, x: bool, y: bool, z: bool) -> Self {
        self.locked_rot = (x as u8) | ((y as u8) << 1) | ((z as u8) << 2);
        // 이미 들어 있던 각속도도 즉시 투영한다 — 회전 중인 바디를 clone해 잠금을 붙여
        // 재삽입하는 경로가 잠긴 축 각속도를 들여오지 못하게.
        self.angvel = self.project_locked_angvel(self.angvel);
        self
    }

    /// 잠긴 축 성분을 **0으로 투영한** 각속도.
    ///
    /// 유효 역관성을 0으로 만드는 것만으로는 부족하다 — 호스트가 `set_angular_velocity`나
    /// 스냅샷으로 잠긴 축에 값을 **직접** 넣으면 솔버는 그걸 없앨 수단이 없고(역관성이 0이라
    /// 임펄스가 안 먹는다) 위치 적분이 그대로 그 축으로 돌린다. 게다가 그 가짜 각속도가
    /// 접점 상대속도에도 들어간다(교차검수 rev_bb593ab5 BLOCKER).
    /// 그래서 캐노니컬 진입점마다 이 투영을 건다.
    pub(crate) fn project_locked_angvel(&self, w: Vec3) -> Vec3 {
        if self.locked_rot == 0 {
            return w;
        }
        Vec3::new(
            if self.locked_rot & 1 != 0 { 0.0 } else { w.x },
            if self.locked_rot & 2 != 0 { 0.0 } else { w.y },
            if self.locked_rot & 4 != 0 { 0.0 } else { w.z },
        )
    }

    /// **접촉력 이벤트 임계**(뉴턴) — 이 값을 넘는 접촉만 이벤트가 된다.
    ///
    /// `f32::INFINITY`(기본)면 꺼짐. 페어 임계는 두 물체 임계의 min이다.
    /// 0 이상 유한값이나 무한대만 받는다(음수·NaN은 `validate`가 거절한다).
    pub fn with_contact_force_threshold(mut self, newtons: f32) -> Self {
        self.contact_force_threshold = newtons;
        self
    }

    /// 접촉력 이벤트 임계(뉴턴). `f32::INFINITY`면 꺼짐.
    #[inline]
    pub fn contact_force_threshold(&self) -> f32 {
        self.contact_force_threshold
    }

    /// 회전 잠금 마스크 (비트 0/1/2 = 월드 X/Y/Z).
    pub fn locked_rotation(&self) -> (bool, bool, bool) {
        (
            self.locked_rot & 1 != 0,
            self.locked_rot & 2 != 0,
            self.locked_rot & 4 != 0,
        )
    }

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
        // **회전 잠금의 예약 비트는 0이어야 한다** (교차검수 rev_bb593ab5 MINOR).
        // 안 막으면 상위 비트가 켜진 스냅샷이 복원·재직렬화되면서 물리적으로 같은
        // 무잠금 상태가 서로 다른 바이트와 상태 해시를 갖는다(비정규 상태 고착).
        // 임계는 음수·NaN이면 안 된다 — 음수면 모든 접촉이 이벤트가 되어 버퍼가 넘치고,
        // NaN이면 비교가 항상 거짓이라 **조용히 꺼진다**(꺼짐은 무한대로 명시해야 한다).
        // NaN은 `>= 0.0`이 거짓이라 여기서 걸린다 — 그게 목적이다(NaN 임계는 비교가
        // 항상 거짓이라 이벤트가 **조용히 꺼진다**; 꺼짐은 무한대로 명시해야 한다).
        if self.contact_force_threshold.is_nan() || self.contact_force_threshold < 0.0 {
            return Err(ApiError::InvalidValue);
        }
        if self.locked_rot & !0b111 != 0 {
            return Err(ApiError::InvalidValue);
        }
        // 잠긴 축 각속도는 0이어야 한다 — 복원 경계에서도 같은 규칙이다(입력·복원 대칭).
        if self.project_locked_angvel(self.angvel) != self.angvel {
            return Err(ApiError::InvalidValue);
        }
        // **하이트필드는 static 전용**이다. 복원 경계(`validate_structure`)에서만 막고
        // 라이브 생성에서 빠뜨리면 `restore(snapshot())`가 자기 스냅샷을 거부한다 —
        // 입력·복원 정책 비대칭은 이 저장소가 네 번 재발로 배운 계열이다
        // (검수 rev_1e81559e BLOCKER). 질량 경로가 없어 kinematic도 성립하지 않는다.
        if matches!(
            self.shape,
            Shape::Heightfield { .. } | Shape::TriMesh { .. }
        ) && self.body_type != BodyType::Static
        {
            return Err(ApiError::InvalidValue);
        }
        // 용접 마스크: 3비트뿐이고, 비영이면 static 3점 hull(지형 삼각형)이어야 한다 —
        // 다른 셰이프에 물리면 면 법선 정의가 없어 보정이 무의미하거나 틀린다.
        if self.weld_edges >= 8 {
            return Err(ApiError::InvalidValue);
        }
        if self.weld_edges != 0 {
            let is_tri_hull =
                matches!(&self.shape, Shape::Hull { data: hull, .. } if hull.points().len() == 3);
            if self.body_type != BodyType::Static || !is_tri_hull {
                return Err(ApiError::InvalidValue);
            }
        }
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
        if !maru_math::fp_verify() {
            return Err(ApiError::FpEnvironment);
        }
        if !maru_math::fp_verify() {
            return Err(ApiError::FpEnvironment);
        }
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
        // **복합 베이킹**: 자식 합성 텐서는 평행축 항의 관성곱(비대각) 때문에 대각 로컬
        // 관성 가정과 충돌한다 — COM 재중심화 + 주축(고유분해) 회전을 pose에 접고 자식
        // pose를 역보정해 같은 물체를 유지한다(hull 재중심화와 같은 계열의 생성 경계 처리).
        let (shape, pose, shape_origin_offset, baked) = match shape {
            Shape::Compound { .. } => {
                let (s2, p2, off, m, i) = bake_compound(shape, pose, density);
                (s2, p2, off, Some((m, i)))
            }
            other => {
                let (s2, p2, off) = recenter_hull(other, pose);
                (s2, p2, off, None)
            }
        };
        let (mass, inertia) = match baked {
            Some(mi) => mi,
            None => mass_properties(&shape, density),
        };
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
            ccd: false,
            weld_edges: 0,
            locked_rot: 0,
            contact_force_threshold: f32::INFINITY,
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
        // **생성도 산술이다** — 자세 정규화·질량·주축 베이킹이 여기서 확정된다.
        // 월드 진입점에만 가드를 두면, 비표준 환경에서 만든 Body를 들고 있다가 환경을
        // 되돌린 뒤 넣는 우회가 남는다(확인검수 rev_7efe4e35 BLOCKER).
        if !maru_math::fp_verify() {
            return Err(ApiError::FpEnvironment);
        }
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
            ccd: false,
            weld_edges: 0,
            locked_rot: 0,
            contact_force_threshold: f32::INFINITY,
            linear_damping: 0.0,
            angular_damping: 0.0,
        };
        b.validate()?;
        Ok(b)
    }

    pub fn new_dynamic(shape: Shape, pose: Pose, density: f32) -> Self {
        // **비-fallible 판에도 가드가 필요하다.** 본문이 fallible 판과 복제돼 있어서
        // 가드가 한쪽에만 붙어 있었다 — 비표준 FP 환경에서 만든 Body를 들고 있다가
        // 환경을 되돌린 뒤 월드에 넣는 우회가 열려 있었다(확인검수 rev_d390e73d BLOCKER).
        // 여기서는 오류형이 없으므로 관례대로 패닉이다(기존 메시지 계약은 아래가 유지).
        assert!(
            maru_math::fp_verify(),
            "non-canonical FP environment (see World::canonicalize_fp_environment)"
        );
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
        // **복합 베이킹**: 자식 합성 텐서는 평행축 항의 관성곱(비대각) 때문에 대각 로컬
        // 관성 가정과 충돌한다 — COM 재중심화 + 주축(고유분해) 회전을 pose에 접고 자식
        // pose를 역보정해 같은 물체를 유지한다(hull 재중심화와 같은 계열의 생성 경계 처리).
        let (shape, pose, shape_origin_offset, baked) = match shape {
            Shape::Compound { .. } => {
                let (s2, p2, off, m, i) = bake_compound(shape, pose, density);
                (s2, p2, off, Some((m, i)))
            }
            other => {
                let (s2, p2, off) = recenter_hull(other, pose);
                (s2, p2, off, None)
            }
        };
        let (mass, inertia) = match baked {
            Some(mi) => mi,
            None => mass_properties(&shape, density),
        };
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
            ccd: false,
            weld_edges: 0,
            locked_rot: 0,
            contact_force_threshold: f32::INFINITY,
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
        // **비-fallible 판에도 가드가 필요하다.** 본문이 fallible 판과 복제돼 있어서
        // 가드가 한쪽에만 붙어 있었다 — 비표준 FP 환경에서 만든 Body를 들고 있다가
        // 환경을 되돌린 뒤 월드에 넣는 우회가 열려 있었다(확인검수 rev_d390e73d BLOCKER).
        // 여기서는 오류형이 없으므로 관례대로 패닉이다(기존 메시지 계약은 아래가 유지).
        assert!(
            maru_math::fp_verify(),
            "non-canonical FP environment (see World::canonicalize_fp_environment)"
        );
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
            ccd: false,
            weld_edges: 0,
            locked_rot: 0,
            contact_force_threshold: f32::INFINITY,
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
        let b = Self {
            body_type: BodyType::Kinematic,
            ..Self::try_new_static(shape, pose)?
        };
        // **타입을 바꿨으면 다시 검증한다** — static으로는 합법인 셰이프가 kinematic으로는
        // 아닐 수 있다(하이트필드). 이 한 줄이 빠져 있어 라이브 생성이 통과하고 복원만
        // 거부하는 비대칭이 생겼다(검수 rev_1e81559e BLOCKER).
        b.validate()?;
        Ok(b)
    }

    #[inline]
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }

    /// 월드 역관성 텐서 (현재 자세 기준).
    #[inline]
    pub fn inv_inertia_world(&self) -> Mat3 {
        if self.is_dynamic() {
            crate::solver::locked_inv_inertia_world(
                self.inv_inertia_local,
                self.pose.rot,
                self.locked_rot,
            )
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
    // 이 함수는 **검증 층**이라 아직 불변식이 안 선 형상을 볼 수 있다 — 패닉 대신 거절.
    let Ok(bb) = b.shape.try_local_aabb() else {
        return false;
    };
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
    let Shape::Hull { data: hull, radius } = shape else {
        return (shape, pose, Vec3::ZERO);
    };
    let points = hull.points();
    let (mut min, mut max) = (points[0], points[0]);
    for p in &points[1..] {
        min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
        max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
    }
    let c = (max + min) * 0.5;
    if c == Vec3::ZERO {
        return (Shape::Hull { data: hull, radius }, pose, Vec3::ZERO);
    }
    // 점을 옮겼으므로 면 구조도 다시 만든다 — `HullData::new`가 둘을 함께 잡는다.
    let moved = maru_collide::HullData::new(points.iter().map(|p| *p - c).collect());
    let pose = Pose {
        pos: pose.pos + pose.rot.rotate(c),
        rot: pose.rot,
    };
    (
        Shape::Hull {
            data: moved,
            radius,
        },
        pose,
        -c,
    )
}

/// 복합 셰이프의 질량 베이킹: (재표현된 셰이프, 새 pose, 원점 오프셋, 질량, 주축 관성 대각).
///
/// 절차: 자식별 (질량, COM, COM 기준 대각 관성) → 합성 COM → COM 기준 합성 텐서
/// (자식 회전 R·diag·Rᵀ + 평행축 m(|d|²E − ddᵀ)) → Jacobi 주축 분해 → 주축 회전 q_p를
/// 바디 pose에 접고 자식 pose를 주축 프레임으로 재표현. 결과적으로 솔버의 두 가정
/// (pose.pos = COM, 로컬 관성 = 대각)이 **정확하게** 참이 된다.
/// **전제: 자식 자세가 단위에 가깝다**(`Shape::validate`가 길이 제곱 1e-3 안으로 강제하고,
/// 호출자는 그 검증 뒤에 이 함수를 부른다). 자식 회전은 정규화 없이 관성 텐서 회전에 쓰이므로
/// 그 전제가 풀리면 질량 특성이 그만큼 찌그러진다 — 검증을 느슨하게 하려면 여기서 정규화부터
/// 해야 한다(확인검수 rev_c73f897c BLOCKER의 두 번째 절).
fn bake_compound(shape: Shape, pose: Pose, density: f32) -> (Shape, Pose, Vec3, f32, Vec3) {
    use maru_math::{Mat3, Quat};
    let Shape::Compound { children } = shape else {
        unreachable!("bake_compound은 Compound 전용")
    };
    // 자식별 질량·COM(바디 프레임)·COM 기준 대각 관성.
    // primitive는 로컬 원점이 곧 COM이다(구·박스·캡슐 — 생성 규약; hull은 AABB 중심 근사).
    let mut parts: Vec<(f32, Vec3, Vec3, Pose)> = Vec::with_capacity(children.len());
    let mut total_m = 0.0f32;
    let mut com = Vec3::ZERO;
    for (cp, child) in &children {
        // **자식 회전을 정규화해서 관성을 돈다.** 아래 텐서 회전은 `Mat3::from_quat(cp.rot)`을
        // 쓰는데, 비단위 쿼터니언은 비직교 행렬을 만들어 질량 특성이 그만큼 찌그러진다
        // (확인검수 rev_201e3d20 BLOCKER). 수용 기준(1e-3)은 입력 관용이고 계산은 정규화로 한다.
        let cp = &Pose {
            pos: cp.pos,
            rot: cp.rot.normalize(),
        };
        let (m, i_diag, local_com) = match child {
            Shape::Hull { data: hull, .. } => {
                let points = hull.points();
                let (mut min, mut max) = (points[0], points[0]);
                for p in &points[1..] {
                    min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
                    max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
                }
                let half = (max - min) * 0.5;
                let (m, i) = mass_properties(&Shape::Box { half, radius: 0.0 }, density);
                (m, i, (max + min) * 0.5)
            }
            other => {
                let (m, i) = mass_properties(other, density);
                (m, i, Vec3::ZERO)
            }
        };
        let c_world = cp.pos + cp.rot.rotate(local_com);
        total_m += m;
        com += c_world * m;
        parts.push((m, i_diag, c_world, *cp));
    }
    com = com * (1.0 / total_m);
    // COM 기준 합성 텐서 (대칭 3×3).
    let mut i_total = Mat3 {
        c: [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO],
    };
    for (m, i_diag, c_world, cp) in &parts {
        let r = Mat3::from_quat(cp.rot);
        let rot_i = r
            .mul_mat(&Mat3::from_diagonal(*i_diag))
            .mul_mat(&r.transpose());
        let d = *c_world - com;
        let d2 = d.dot(d);
        // 평행축: m(|d|²E − ddᵀ)
        let pa = Mat3 {
            c: [
                Vec3::new(m * (d2 - d.x * d.x), m * (-d.x * d.y), m * (-d.x * d.z)),
                Vec3::new(m * (-d.y * d.x), m * (d2 - d.y * d.y), m * (-d.y * d.z)),
                Vec3::new(m * (-d.z * d.x), m * (-d.z * d.y), m * (d2 - d.z * d.z)),
            ],
        };
        for k in 0..3 {
            i_total.c[k] = i_total.c[k] + rot_i.c[k] + pa.c[k];
        }
    }
    let (lambda, mut ev) = i_total.eigen_symmetric();
    // 고유벡터 행렬이 반사(det<0)면 한 열을 뒤집어 회전으로 만든다.
    let det = ev.c[0].dot(ev.c[1].cross(ev.c[2]));
    if det < 0.0 {
        ev.c[2] = -ev.c[2];
    }
    let q_p = Quat::from_mat3(&ev);
    // 새 바디 프레임: 원점=COM, 축=주축. 자식을 재표현해 같은 물체를 유지한다.
    let inv_q = q_p.conjugate();
    let children: Vec<(Pose, Shape)> = children
        .into_iter()
        .map(|(cp, child)| {
            (
                Pose {
                    pos: inv_q.rotate(cp.pos - com),
                    rot: (inv_q * cp.rot).normalize(),
                },
                child,
            )
        })
        .collect();
    let new_pose = Pose {
        pos: pose.pos + pose.rot.rotate(com),
        rot: (pose.rot * q_p).normalize(),
    };
    // 원래 셰이프 원점의 새 로컬 좌표 (렌더 오프셋 — recenter_hull의 -c와 같은 의미).
    let origin_offset = inv_q.rotate(-com);
    (
        Shape::Compound { children },
        new_pose,
        origin_offset,
        total_m,
        lambda,
    )
}

/// (질량, 관성 대각). 캡슐은 원통+반구 근사 대신 v1은 동체 실린더 근사(문서화된 가정).
fn mass_properties(shape: &Shape, density: f32) -> (f32, Vec3) {
    match *shape {
        // 하이트필드·삼각형은 **static 전용**이라 질량이 없다(월드가 강제). 0을 돌려
        // 두면 dynamic으로 만들려는 시도가 상위 검증에서 걸린다 — 패닉이 아니라 거부다.
        Shape::Heightfield { .. } | Shape::TriMesh { .. } | Shape::Triangle { .. } => {
            (0.0, Vec3::ZERO)
        }
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
        Shape::Cylinder {
            half_height,
            radius,
        } => {
            // 균질 원기둥의 닫힌 해다(캡슐과 달리 근사가 아니다).
            // m = ρ·π·r²·h,  Ixx = Izz = m(3r² + h²)/12,  Iyy = m·r²/2.
            let h = 2.0 * half_height;
            let m = density * core::f32::consts::PI * radius * radius * h;
            let ixz = m * (3.0 * radius * radius + h * h) / 12.0;
            let iy = 0.5 * m * radius * radius;
            (m, Vec3::new(ixz, iy, ixz))
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
        Shape::Compound { .. } => {
            unreachable!("compound 질량은 bake_compound가 계산한다 — new_dynamic 디스패치 확인")
        }
        Shape::Hull { data: ref hull, .. } => {
            let points = hull.points();
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
                data: maru_collide::HullData::new(pts),
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
        let Shape::Hull { data: ref hull, .. } = b.shape else {
            panic!()
        };
        let points = hull.points();
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
