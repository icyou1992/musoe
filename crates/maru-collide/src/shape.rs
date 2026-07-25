//! Convex 셰이프 — support 함수 기반 (설계 §4.2).
//!
//! **convex radius 계약(shrink-then-reinflate)**: `support_core()`는 radius를 뺀 코어 형상의
//! support point를 반환하고, 접촉 생성이 radius를 다시 더한다. GJK/EPA는 코어에서 돌므로
//! 침투 깊이가 얕게 유지되어 EPA 진입이 최소화된다. [근거: Jolt convex radius]
//! Hull의 코어 포인트는 빌더가 이미 수축된 점을 준다는 계약(v1 — 문서화된 가정).

use crate::limits::{in_range, vec_in_range, MAX_HULL_POINTS};
use crate::Aabb;
use maru_math::Vec3;

#[derive(Clone, Debug)]
pub enum Shape {
    /// 코어=중심점, radius=반지름 (구 전체가 convex radius).
    Sphere { radius: f32 },
    /// Y축 정렬 세그먼트 코어 [-half_height, +half_height] + radius.
    Capsule { half_height: f32, radius: f32 },
    /// 코어=half에서 radius를 뺀 박스. half는 겉면 기준(사용자 관점 치수).
    Box { half: Vec3, radius: f32 },
    /// 코어 포인트 집합(이미 수축됨) + radius.
    Hull { points: Vec<Vec3>, radius: f32 },
}

impl Shape {
    /// 형상 불변식 검증. **모든 바디 타입의 생성 경계에서 호출된다** — static/kinematic도
    /// 빈 hull이면 support/local_aabb가 `points[0]`에서 패닉한다(리뷰 지적).
    pub fn validate(&self) -> Result<(), &'static str> {
        // 크기 상한까지 여기서 강제한다 — 라이브 생성과 역직렬화가 같은 정책을 쓴다.
        let r = self.radius();
        if !in_range(r) || r < 0.0 {
            return Err("shape radius must be finite, non-negative and within MAX_MAGNITUDE");
        }
        match self {
            Shape::Sphere { radius } => {
                if !in_range(*radius) || *radius <= 0.0 {
                    return Err("sphere radius must be finite and positive");
                }
            }
            Shape::Capsule { half_height, .. } => {
                if !in_range(*half_height) || *half_height < 0.0 {
                    return Err("capsule half_height must be finite and non-negative");
                }
                if r <= 0.0 {
                    return Err("capsule radius must be positive");
                }
            }
            Shape::Box { half, .. } => {
                if !vec_in_range(*half) || half.x <= 0.0 || half.y <= 0.0 || half.z <= 0.0 {
                    return Err("box half extents must be finite and positive");
                }
                if r > half.x.min(half.y).min(half.z) {
                    return Err("box convex radius must not exceed the smallest half extent");
                }
            }
            Shape::Hull { points, .. } => {
                if points.is_empty() {
                    return Err("hull must contain at least one point");
                }
                if points.len() > MAX_HULL_POINTS {
                    return Err("hull point count exceeds MAX_HULL_POINTS");
                }
                if !points.iter().all(|p| vec_in_range(*p)) {
                    return Err("hull points must be finite and within MAX_MAGNITUDE");
                }
            }
        }
        Ok(())
    }

    /// 겉면과 코어를 가르는 convex radius.
    #[inline]
    pub fn radius(&self) -> f32 {
        match *self {
            Shape::Sphere { radius }
            | Shape::Capsule { radius, .. }
            | Shape::Box { radius, .. }
            | Shape::Hull { radius, .. } => radius,
        }
    }

    /// 방향 d(비정규화 허용, 0이면 임의점)의 **코어** support point (로컬 공간).
    pub fn support_core(&self, d: Vec3) -> Vec3 {
        match self {
            Shape::Sphere { .. } => Vec3::ZERO,
            Shape::Capsule { half_height, .. } => Vec3::new(
                0.0,
                if d.y >= 0.0 {
                    *half_height
                } else {
                    -*half_height
                },
                0.0,
            ),
            Shape::Box { half, radius } => {
                let c = Vec3::new(
                    (half.x - radius).max(0.0),
                    (half.y - radius).max(0.0),
                    (half.z - radius).max(0.0),
                );
                Vec3::new(
                    if d.x >= 0.0 { c.x } else { -c.x },
                    if d.y >= 0.0 { c.y } else { -c.y },
                    if d.z >= 0.0 { c.z } else { -c.z },
                )
            }
            Shape::Hull { points, .. } => {
                // 선형 스캔 (v1 — hill climbing은 벤치 증명 시 도입). 동률은 낮은 인덱스 우선(결정론).
                let mut best = points[0];
                let mut best_d = best.dot(d);
                for p in &points[1..] {
                    let pd = p.dot(d);
                    if pd > best_d {
                        best_d = pd;
                        best = *p;
                    }
                }
                best
            }
        }
    }

    /// 겉면 기준 로컬 AABB (코어 + radius 재팽창 포함).
    pub fn local_aabb(&self) -> Aabb {
        let r = self.radius();
        match self {
            Shape::Sphere { .. } => Aabb::from_center(Vec3::ZERO, Vec3::new(r, r, r)),
            Shape::Capsule { half_height, .. } => {
                Aabb::from_center(Vec3::ZERO, Vec3::new(r, half_height + r, r))
            }
            Shape::Box { half, .. } => Aabb::from_center(Vec3::ZERO, *half),
            Shape::Hull { points, .. } => {
                let mut min = points[0];
                let mut max = points[0];
                for p in &points[1..] {
                    min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
                    max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
                }
                let rv = Vec3::new(r, r, r);
                Aabb::new(min - rv, max + rv)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 리뷰 회귀: 잘못된 형상은 생성 경계에서 거부되어야 한다(빈 hull은 첫 step에서 패닉했다).
    #[test]
    fn invalid_shapes_are_rejected() {
        assert!(Shape::Hull {
            points: vec![],
            radius: 0.0
        }
        .validate()
        .is_err());
        assert!(Shape::Sphere { radius: 0.0 }.validate().is_err());
        assert!(Shape::Sphere { radius: f32::NAN }.validate().is_err());
        assert!(Shape::Box {
            half: Vec3::new(1.0, -1.0, 1.0),
            radius: 0.0
        }
        .validate()
        .is_err());
        // radius가 half를 잠식하면 평면 패치가 사라진다
        assert!(Shape::Box {
            half: Vec3::new(0.1, 1.0, 1.0),
            radius: 0.5
        }
        .validate()
        .is_err());
        assert!(Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.1
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn box_core_shrinks_by_radius() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.05,
        };
        let p = s.support_core(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(p, Vec3::new(0.95, 1.95, 2.95));
        // 겉면 AABB는 사용자 치수 그대로
        assert_eq!(s.local_aabb().max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn capsule_support_and_aabb() {
        let s = Shape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        assert_eq!(
            s.support_core(Vec3::new(0.3, -2.0, 0.1)),
            Vec3::new(0.0, -1.0, 0.0)
        );
        assert_eq!(s.local_aabb().max, Vec3::new(0.5, 1.5, 0.5));
    }

    /// 리뷰 지적: 구 버전은 **좌표가 같은 두 점**을 넣어 반환값만 비교했으므로 어느 쪽이
    /// 선택됐는지 관측할 수 없었다(동률 규칙을 검증하지 못함). 좌표는 다르되 dot이 같은
    /// 두 점을 써서 '낮은 인덱스가 이긴다'를 실제로 관측한다.
    #[test]
    fn hull_support_tie_prefers_lower_index() {
        // d=(1,0,0)에 대해 dot이 모두 1.0인 서로 다른 점들
        let s = Shape::Hull {
            points: vec![
                Vec3::new(1.0, 0.5, 0.0),  // idx 0 (선택되어야 함)
                Vec3::new(1.0, -0.5, 0.0), // idx 1 동률
                Vec3::new(1.0, 0.0, 0.7),  // idx 2 동률
                Vec3::new(-1.0, 0.0, 0.0),
            ],
            radius: 0.0,
        };
        let d = Vec3::new(1.0, 0.0, 0.0);
        let p = s.support_core(d);
        assert_eq!(
            p,
            Vec3::new(1.0, 0.5, 0.0),
            "동률에서 낮은 인덱스가 이겨야 한다"
        );
        // 순서를 바꾸면 선택도 바뀌어야 관측이 유효함을 증명
        let s2 = Shape::Hull {
            points: vec![
                Vec3::new(1.0, -0.5, 0.0),
                Vec3::new(1.0, 0.5, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
            ],
            radius: 0.0,
        };
        assert_eq!(s2.support_core(d), Vec3::new(1.0, -0.5, 0.0));
    }

    #[test]
    fn support_is_extreme_point() {
        // support의 정의 검증: 모든 점에 대해 dot(d, support) >= dot(d, p)
        let s = Shape::Hull {
            points: vec![
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, 2.0, 0.5),
                Vec3::new(0.0, -1.0, 2.0),
                Vec3::new(0.5, 0.5, 0.5),
            ],
            radius: 0.0,
        };
        let d = Vec3::new(0.3, -0.7, 0.9);
        let sup = s.support_core(d);
        if let Shape::Hull { points, .. } = &s {
            for p in points {
                assert!(sup.dot(d) >= p.dot(d) - 1e-6);
            }
        }
    }
}
