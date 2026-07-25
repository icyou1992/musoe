//! 강체 — 타입 3종 시맨틱 (설계 §9).

use maru_collide::{Pose, Shape};
use maru_math::{Mat3, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyType {
    /// 불변. 아일랜드 미참여(anchor 전용 — 설계 §4.3).
    Static,
    Dynamic,
    /// 처방 속도로 구동, inv_mass=0이지만 접촉 상대속도에 기여 (설계 §9).
    Kinematic,
}

/// `#[non_exhaustive]`: **크레이트 밖에서 구조체 리터럴로 만들 수 없다**(리뷰 MAJOR).
/// 생성자가 강제하는 불변식(유한 pose·양수 역질량·검증된 셰이프)을 우회하는 경로를
/// 타입 수준에서 막는다. 읽기는 그대로 공개 필드로 한다(월드는 `&mut Body`를 밖에 주지 않는다).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Body {
    pub body_type: BodyType,
    pub pose: Pose,
    pub linvel: Vec3,
    pub angvel: Vec3,
    pub shape: Shape,
    pub inv_mass: f32,
    /// 로컬 역관성 대각 (월드 역관성은 매 서브스텝 회전).
    pub inv_inertia_local: Vec3,
    pub friction: f32,
    pub restitution: f32,
}

impl Body {
    /// 동적 바디 생성. 질량·관성이 유한 양수가 아니면 **생성 경계에서 거부**한다
    /// (설계 §8 위생: inv_mass=∞는 0·∞=NaN으로 속도를 오염시킨다 — 리뷰 지적).
    pub fn new_dynamic(shape: Shape, pose: Pose, density: f32) -> Self {
        assert!(
            density.is_finite() && density > 0.0,
            "density must be finite and positive, got {density}"
        );
        shape.validate().expect("invalid shape");
        assert!(
            pose.pos.is_finite() && pose.rot.is_finite(),
            "pose must be finite"
        );
        // **hull 재중심화**: 솔버는 pose.pos를 질량중심으로 가정한다(r = 접점 − pose.pos).
        // 중심이 어긋난 hull은 평행축 항으로 대각 관성을 키워도 **관성곱(products of
        // inertia)**이 빠져 2축 이상 어긋나면 축이 틀린 텐서가 된다(리뷰 MAJOR).
        // 대각 텐서로 정확해지는 유일한 방법은 실제로 중심을 맞추는 것 — 점을 중심으로
        // 옮기고 pose를 같은 양만큼 월드에서 밀어 **같은 물체**를 유지한다.
        let (shape, pose) = recenter_hull(shape, pose);
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
        Self {
            body_type: BodyType::Dynamic,
            pose,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass,
            inv_inertia_local: inv_inertia,
            friction: 0.6,
            restitution: 0.0,
        }
    }

    pub fn new_static(shape: Shape, pose: Pose) -> Self {
        shape.validate().expect("invalid shape");
        assert!(
            pose.pos.is_finite() && pose.rot.is_finite(),
            "pose must be finite"
        );
        Self {
            body_type: BodyType::Static,
            pose,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass: 0.0,
            inv_inertia_local: Vec3::ZERO,
            friction: 0.6,
            restitution: 0.0,
        }
    }

    pub fn new_kinematic(shape: Shape, pose: Pose) -> Self {
        Self {
            body_type: BodyType::Kinematic,
            ..Self::new_static(shape, pose)
        }
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

/// hull의 AABB 중심을 원점으로 옮기고, 같은 양을 월드 자세에 반영한다(물체는 불변).
/// 결과: 질량중심이 pose.pos와 일치 → 평행축 항 0 → 관성곱 0 → 대각 텐서가 정확해진다.
fn recenter_hull(shape: Shape, pose: Pose) -> (Shape, Pose) {
    let Shape::Hull { points, radius } = shape else {
        return (shape, pose);
    };
    let (mut min, mut max) = (points[0], points[0]);
    for p in &points[1..] {
        min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
        max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
    }
    let c = (max + min) * 0.5;
    if c == Vec3::ZERO {
        return (Shape::Hull { points, radius }, pose);
    }
    let points = points.iter().map(|p| *p - c).collect();
    let pose = Pose {
        pos: pose.pos + pose.rot.rotate(c),
        rot: pose.rot,
    };
    (Shape::Hull { points, radius }, pose)
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
        let pts = vec![Vec3::new(9.5, 19.5, 29.5), Vec3::new(10.5, 20.5, 30.5)];
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
        let c = (points[0] + points[1]) * 0.5;
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
