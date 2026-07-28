//! Convex 셰이프 — support 함수 기반 (설계 §4.2).
//!
//! **convex radius 계약(shrink-then-reinflate)**: `support_core()`는 radius를 뺀 코어 형상의
//! support point를 반환하고, 접촉 생성이 radius를 다시 더한다. GJK/EPA는 코어에서 돌므로
//! 침투 깊이가 얕게 유지되어 EPA 진입이 최소화된다. [근거: Jolt convex radius]
//! Hull의 코어 포인트는 빌더가 이미 수축된 점을 준다는 계약(v1 — 문서화된 가정).

use crate::limits::{extent_in_range, vec_extent_in_range, MAX_HULL_POINTS, MIN_SHAPE_EXTENT};
use crate::Aabb;
use crate::Pose;
use maru_math::Vec3;

/// 복합 바디 자식 수 상한 — feature ID 네임스페이스(자식 3비트×2)와 예산 유도의 근거.
pub const MAX_COMPOUND_CHILDREN: usize = 8;

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
    /// 복합 — 자식(로컬 pose + **primitive**) 목록. **볼록이 아니다**: GJK/EPA/support
    /// 경로에 단독으로 들어가면 안 되고, 모든 진입점(내로우·캐스트·레이·질량)이 자식으로
    /// 디스패치한다(`support_core`가 unreachable로 잠근다). 중첩 금지·자식 1..=8은
    /// validate가 강제한다. 질량 생성 시 COM 재중심화+주축 베이킹은 body 계층 소관.
    Compound { children: Vec<(Pose, Shape)> },
}

/// 점 집합의 **회전 불변** 최소 폭 **하한 추정**.
///
/// 정확한 최소 폭은 볼록 껍질의 모든 면 법선에 대해 재야 한다(껍질 빌더가 필요하다 —
/// v1 계약: hull 점은 빌더가 준다). 여기서는 생성·복원 경계의 **퇴화 스크린**으로,
/// 후보 방향 집합에 대한 폭의 최소값을 쓴다. 한 평면의 높이만 재면(직경 축 → 그 직선의
/// 최원점 → 그 평면의 최원점) 다른 면 방향의 더 얇은 폭을 놓치므로(리뷰 MAJOR),
/// 극점들의 모든 삼중조합 법선까지 후보에 넣어 추정을 낮춘다.
/// 이 값은 실제 최소 폭보다 **크지 않다고 보장되지 않는다** — 스크린이지 증명이 아니다.
fn hull_min_width(points: &[Vec3]) -> f32 {
    if points.len() < 4 {
        return 0.0; // 3점 이하는 항상 한 평면 위 (convex radius로만 두꺼워진다)
    }
    // 후보 극점: 13방향 support + 직경 쌍
    const DIRS: [Vec3; 13] = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];
    // 점이 적으면 **전수 열거**가 정확하다(사면체의 최소 폭 = 최소 높이).
    // 극점 집합만 쓰면 어떤 support 방향에서도 극점이 아닌 정점이 빠져 그 정점이 만드는
    // 얇은 면을 놓친다(리뷰가 제시한 sliver가 정확히 그 경우 — 0.575mm를 1.1mm로 오판).
    // 지원 상한(MAX_HULL_POINTS=32) 안에서는 **항상 전수 열거**다 — 사면체의 최소 폭은
    // 최소 높이이므로 삼중조합 전수가 정확한 답을 준다. 스크린이 아니라 증명이다.
    let mut ext: Vec<Vec3> = points.to_vec();
    let use_support = false;
    let push_unique = |ext: &mut Vec<Vec3>, p: Vec3| {
        if !ext.iter().any(|q| (*q - p).length_sq() == 0.0) {
            ext.push(p);
        }
    };
    for d in DIRS.iter().filter(|_| use_support) {
        let (mut lo, mut hi) = (points[0], points[0]);
        let (mut lo_v, mut hi_v) = (d.dot(points[0]), d.dot(points[0]));
        for p in &points[1..] {
            let v = d.dot(*p);
            if v < lo_v {
                lo_v = v;
                lo = *p;
            }
            if v > hi_v {
                hi_v = v;
                hi = *p;
            }
        }
        push_unique(&mut ext, lo);
        push_unique(&mut ext, hi);
    }
    // 후보 법선: 극점 삼중조합 + 좌표축
    let width_along = |n: Vec3| -> f32 {
        let l = n.length();
        if l <= 0.0 {
            return f32::INFINITY;
        }
        let inv = 1.0 / l;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in points {
            let v = n.dot(*p) * inv;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    };
    let mut w = f32::INFINITY;
    for d in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ] {
        w = w.min(width_along(d));
    }
    for i in 0..ext.len() {
        for j in (i + 1)..ext.len() {
            for k in (j + 1)..ext.len() {
                let n = (ext[j] - ext[i]).cross(ext[k] - ext[i]);
                if n.length_sq() > 0.0 {
                    w = w.min(width_along(n));
                }
            }
        }
    }
    if w.is_finite() {
        w
    } else {
        0.0
    }
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
            Shape::Compound { children } => {
                if children.is_empty() || children.len() > MAX_COMPOUND_CHILDREN {
                    return Err("compound must have 1..=8 children");
                }
                for (pose, child) in children {
                    if matches!(child, Shape::Compound { .. }) {
                        return Err("compound nesting is not allowed");
                    }
                    child.validate()?;
                    if !vec_extent_in_range(pose.pos) {
                        return Err("compound child offset must be finite and within extent range");
                    }
                    let q = pose.rot;
                    let l2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
                    if !l2.is_finite() || !(0.9..=1.1).contains(&l2) {
                        return Err("compound child rotation must be a unit quaternion");
                    }
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
                if hull_min_width(points) + thick < MIN_SHAPE_EXTENT {
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
            // 복합은 볼록 질의 대상이 아니다 — radius는 convex 경로 전용 개념.
            Shape::Compound { .. } => 0.0,
        }
    }

    /// 셰이프 원점 기준 **외접 반지름**(겉면 포함) — 회전 캐스트의 각 기여 상한
    /// `|angle| · bounding_radius`의 근거. 원점에서 가장 먼 표면점까지의 거리 상한이다.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Shape::Sphere { radius } => *radius,
            Shape::Capsule {
                half_height,
                radius,
            } => half_height + radius,
            Shape::Box { half, .. } => half.length(), // half는 겉면 기준(radius 포함)
            Shape::Hull { points, radius } => {
                points.iter().fold(0.0f32, |m, p| m.max(p.length())) + radius
            }
            Shape::Compound { children } => children.iter().fold(0.0f32, |m, (cp, cs)| {
                m.max(cp.pos.length() + cs.bounding_radius())
            }),
        }
    }

    /// **각 스윕 반지름** — 단위 회전축 `axis`(로컬)에 대한 회전이 점유를 바꾸는
    /// 정도의 상한: 코어 점들의 축 수직 거리 최댓값. 표면 = 코어 ⊕ 반지름 볼이고
    /// 볼은 자기 중심 기준 회전 불변이므로, 점유 변화는 코어 점의 호 `angle × r⊥`에
    /// 유계다. 구는 0(회전 불변), 캡슐은 자기 축 회전에서 0 — 회전 CCD 게이트가
    /// 회전 불변 형상을 공짜로 걸러내는 근거(확인 검수 rev_477d7366 MAJOR).
    /// 축 대비 로컬 기하는 그 축 회전 아래 불변이므로 t=0에 잰 값이 캐스트 내내 유효하다.
    pub fn angular_sweep_radius(&self, axis: Vec3) -> f32 {
        let perp = |p: Vec3| (p - axis * p.dot(axis)).length();
        match self {
            Shape::Sphere { .. } => 0.0,
            Shape::Capsule { half_height, .. } => perp(Vec3::new(0.0, *half_height, 0.0)),
            Shape::Box { half, radius } => {
                // 코어 꼭짓점 8개(부호 대칭으로 4개만) 중 축 수직 거리 최대.
                let h = Vec3::new(
                    (half.x - radius).max(0.0),
                    (half.y - radius).max(0.0),
                    (half.z - radius).max(0.0),
                );
                let mut m = 0.0f32;
                for sx in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        m = m.max(perp(Vec3::new(sx * h.x, sy * h.y, h.z)));
                    }
                }
                m
            }
            Shape::Hull { points, .. } => points.iter().fold(0.0f32, |m, p| m.max(perp(*p))),
            Shape::Compound { children } => children.iter().fold(0.0f32, |m, (cp, cs)| {
                // 자식 원점의 축 수직 거리 + 자식 자체의 스윕 반지름(자식 로컬 축).
                let child_axis = cp.rot.conjugate().rotate(axis);
                m.max(perp(cp.pos) + cs.angular_sweep_radius(child_axis))
            }),
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
            Shape::Compound { .. } => {
                unreachable!("compound is not convex — dispatch to children before GJK/support")
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
            Shape::Compound { children } => {
                let mut acc: Option<Aabb> = None;
                for (pose, child) in children {
                    let t = transformed_aabb(child.local_aabb(), pose);
                    acc = Some(match acc {
                        None => t,
                        Some(a) => a.merge(t),
                    });
                }
                acc.expect("validate가 자식 ≥1을 강제한다")
            }
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

/// 로컬 AABB를 자식 pose로 변환한 보수적 AABB (회전-절댓값 확장 — world.rs body_aabb와 동형).
pub fn transformed_aabb(local: Aabb, pose: &Pose) -> Aabb {
    let c = (local.max + local.min) * 0.5;
    let hl = (local.max - local.min) * 0.5;
    let (ax, ay, az) = (
        pose.rot.rotate(Vec3::new(1.0, 0.0, 0.0)),
        pose.rot.rotate(Vec3::new(0.0, 1.0, 0.0)),
        pose.rot.rotate(Vec3::new(0.0, 0.0, 1.0)),
    );
    let ext = Vec3::new(
        ax.x.abs() * hl.x + ay.x.abs() * hl.y + az.x.abs() * hl.z,
        ax.y.abs() * hl.x + ay.y.abs() * hl.y + az.y.abs() * hl.z,
        ax.z.abs() * hl.x + ay.z.abs() * hl.y + az.z.abs() * hl.z,
    );
    Aabb::from_center(pose.pos + pose.rot.rotate(c), ext)
}

#[cfg(test)]
mod sweep_tests {
    use super::*;

    /// 회전 불변·축 대칭 형상의 각 스윕 반지름 — 게이트가 이 값으로 꺼진다.
    #[test]
    fn angular_sweep_radius_respects_rotational_symmetry() {
        let y = Vec3::new(0.0, 1.0, 0.0);
        let x = Vec3::new(1.0, 0.0, 0.0);
        // 구: 어느 축이든 0.
        assert_eq!(Shape::Sphere { radius: 0.5 }.angular_sweep_radius(y), 0.0);
        assert_eq!(Shape::Sphere { radius: 0.5 }.angular_sweep_radius(x), 0.0);
        // 캡슐: 자기 축(Y) 회전은 0, 수직 축은 half_height.
        let cap = Shape::Capsule {
            half_height: 0.7,
            radius: 0.2,
        };
        assert!(cap.angular_sweep_radius(y).abs() < 1e-6);
        assert!((cap.angular_sweep_radius(x) - 0.7).abs() < 1e-6);
        // 박스: Y축 회전이면 코어 꼭짓점의 XZ 평면 거리 — half=(1.5,0.05,0.05), r=0이면
        // 코어 (1.5,·,0.05)의 수직 거리 √(1.5²+0.05²).
        let blade = Shape::Box {
            half: Vec3::new(1.5, 0.05, 0.05),
            radius: 0.0,
        };
        let want = (1.5f32 * 1.5 + 0.05 * 0.05).sqrt();
        assert!((blade.angular_sweep_radius(y) - want).abs() < 1e-5);
        // 외접 반지름보다 크지 않다 (상한 정합).
        assert!(blade.angular_sweep_radius(x) <= blade.bounding_radius() + 1e-6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **삼각형은 전용 셰이프가 아니라 3점 hull이다.**
    ///
    /// 지형·레벨 메시를 삼각형 단위로 넣을 때 쓰는 정본 패턴이다. 3점은 항상 한 평면
    /// 위이므로 두께가 0이고, 최소 특징 크기는 **convex radius로만** 얻는다 —
    /// `hull_min_width`가 3점 이하에 0을 돌려주는 이유가 그것이다.
    /// radius가 너무 얇으면 생성 경계에서 거부된다(퇴화 형상이 GJK/EPA를 깨뜨리지 않게).
    #[test]
    fn a_triangle_is_a_three_point_hull() {
        let tri = |r: f32| Shape::Hull {
            points: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
            radius: r,
        };
        // 두께 = 2r 이 최소 특징 크기를 넘어야 통과한다
        assert!(
            tri(0.0).validate().is_err(),
            "두께 0인 삼각형은 거부돼야 한다"
        );
        let ok = tri(0.02);
        ok.validate().expect("두꺼운 삼각형은 유효하다");

        // support: 각 꼭짓점이 자기 방향에서 극점이어야 한다
        assert_eq!(
            ok.support_core(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(2.0, 0.0, 0.0)
        );
        assert_eq!(
            ok.support_core(Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(0.0, 0.0, 2.0)
        );
        assert_eq!(
            ok.support_core(Vec3::new(-1.0, 0.0, -1.0)),
            Vec3::new(0.0, 0.0, 0.0)
        );

        // AABB는 평면 방향으로도 radius만큼 부풀어야 한다 (겉면 기준)
        let bb = ok.local_aabb();
        assert!((bb.min.y - -0.02).abs() < 1e-6, "min.y={}", bb.min.y);
        assert!((bb.max.y - 0.02).abs() < 1e-6, "max.y={}", bb.max.y);
        assert!((bb.max.x - 2.02).abs() < 1e-6);
    }

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
