//! Convex 셰이프 — support 함수 기반 (설계 §4.2).
//!
//! **convex radius 계약(shrink-then-reinflate)**: `support_core()`는 radius를 뺀 코어 형상의
//! support point를 반환하고, 접촉 생성이 radius를 다시 더한다. GJK/EPA는 코어에서 돌므로
//! 침투 깊이가 얕게 유지되어 EPA 진입이 최소화된다. [근거: Jolt convex radius]
//! Hull의 코어 포인트는 빌더가 이미 수축된 점을 준다는 계약(v1 — 문서화된 가정).

use crate::limits::{extent_in_range, vec_extent_in_range, MAX_HULL_POINTS, MIN_SHAPE_EXTENT};
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

/// 점 집합의 **회전 불변** 최소 두께 (평면에서 가장 먼 점까지의 거리).
/// 세 단계 극점 절차 — 최장 축 → 직선에서 최원점 → 평면에서 최원점.
fn hull_thickness(points: &[Vec3]) -> f32 {
    if points.len() < 4 {
        // 3점 이하는 항상 한 평면 위 → 두께 0 (convex radius로만 두꺼워진다)
        return 0.0;
    }
    // ① 최장 축
    let (mut a, mut b) = (points[0], points[0]);
    let mut best = 0.0f32;
    for (i, p) in points.iter().enumerate() {
        for q in &points[i + 1..] {
            let d = (*q - *p).length_sq();
            if d > best {
                best = d;
                a = *p;
                b = *q;
            }
        }
    }
    let ab = b - a;
    if ab.length_sq() <= 0.0 {
        return 0.0;
    }
    // ② 직선 ab에서 가장 먼 점
    let inv = 1.0 / ab.length();
    let mut c = a;
    let mut wmax = 0.0f32;
    for p in points {
        let w = ab.cross(*p - a).length() * inv;
        if w > wmax {
            wmax = w;
            c = *p;
        }
    }
    if wmax <= 0.0 {
        return 0.0;
    }
    // ③ 평면 abc에서 가장 먼 점 = 두께
    let n = ab.cross(c - a);
    let nl = n.length();
    if nl <= 0.0 {
        return 0.0;
    }
    let inv_n = 1.0 / nl;
    let mut tmax = 0.0f32;
    for p in points {
        let t = n.dot(*p - a).abs() * inv_n;
        if t > tmax {
            tmax = t;
        }
    }
    tmax
}

impl Shape {
    /// 형상 불변식 검증. **모든 바디 타입의 생성 경계에서 호출된다** — static/kinematic도
    /// 빈 hull이면 support/local_aabb가 `points[0]`에서 패닉한다(리뷰 지적).
    pub fn validate(&self) -> Result<(), &'static str> {
        // 크기 상한까지 여기서 강제한다 — 라이브 생성과 역직렬화가 같은 정책을 쓴다.
        let r = self.radius();
        if !extent_in_range(r) || r < 0.0 {
            return Err("shape radius must be finite, non-negative and within MAX_SHAPE_EXTENT");
        }
        match self {
            Shape::Sphere { radius } => {
                if !extent_in_range(*radius) || *radius < MIN_SHAPE_EXTENT {
                    return Err(
                        "sphere radius must be within [MIN_SHAPE_EXTENT, MAX_SHAPE_EXTENT]",
                    );
                }
            }
            Shape::Capsule { half_height, .. } => {
                if !extent_in_range(*half_height) || *half_height < 0.0 {
                    return Err("capsule half_height must be finite and non-negative");
                }
                if r < MIN_SHAPE_EXTENT {
                    return Err("capsule radius must be at least MIN_SHAPE_EXTENT");
                }
            }
            Shape::Box { half, .. } => {
                if !vec_extent_in_range(*half)
                    || half.x < MIN_SHAPE_EXTENT
                    || half.y < MIN_SHAPE_EXTENT
                    || half.z < MIN_SHAPE_EXTENT
                {
                    return Err(
                        "box half extents must be within [MIN_SHAPE_EXTENT, MAX_SHAPE_EXTENT]",
                    );
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
                if !points.iter().all(|p| vec_extent_in_range(*p)) {
                    return Err("hull points must be finite and within MAX_SHAPE_EXTENT");
                }
                // **최소 특징 크기는 모든 셰이프에 적용된다**(리뷰 MAJOR: hull·capsule이
                // 빠져 있었다). 축 정렬 AABB span만 보면 **회전된 퇴화 hull**이 통과한다 —
                // 한 평면 위의 점들도 x/y/z span은 모두 크다. 회전 불변 두께를 재야 한다:
                // 최장 축 → 그 직선에서 가장 먼 점(폭) → 그 평면에서 가장 먼 점(두께).
                let thick = 2.0 * r;
                if hull_thickness(points) + thick < MIN_SHAPE_EXTENT {
                    return Err(
                        "hull is degenerate (thinner than MIN_SHAPE_EXTENT in some direction)",
                    );
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
