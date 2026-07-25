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

#[derive(Clone, Debug)]
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
        if let Shape::Hull { points, .. } = &shape {
            assert!(!points.is_empty(), "dynamic hull must contain points");
        }
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
        Self {
            body_type: BodyType::Dynamic,
            pose,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            shape,
            inv_mass: 1.0 / mass,
            inv_inertia_local: Vec3::new(1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z),
            friction: 0.6,
            restitution: 0.0,
        }
    }

    pub fn new_static(shape: Shape, pose: Pose) -> Self {
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
            // [가정] v1: AABB 근사 — hull 정밀 질량은 hull 빌더와 함께
            let mut min = points[0];
            let mut max = points[0];
            for p in &points[1..] {
                min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
                max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
            }
            let half = (max - min) * 0.5;
            mass_properties(&Shape::Box { half, radius: 0.0 }, density)
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
