//! GJK — convex 코어 쌍의 거리/교차 판정 (설계 §4.2).
//!
//! 코어(수축 형상) 기준으로 돌며, 호출자가 convex radius를 빼서 실거리로 환산한다.
//! Separated면 witness 점·법선까지 반환(스페큘러티브 접촉 생성 입력).
//! Intersecting이면 원점을 감싸는 사면체 심플렉스를 반환(EPA 시드).
//! 결정론: 반복 상한 고정, 탐색 방향·리덕션이 입력만의 함수.

use crate::Shape;
use maru_math::{Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub pos: Vec3,
    pub rot: Quat,
}

impl Pose {
    #[inline]
    pub fn support_core(&self, shape: &Shape, dir_world: Vec3) -> Vec3 {
        let local = shape.support_core(self.rot.conjugate().rotate(dir_world));
        self.pos + self.rot.rotate(local)
    }
}

/// 민코프스키 차 A−B의 support 정점 (witness용 원본 좌표 동반).
#[derive(Clone, Copy, Debug)]
pub struct SupportPoint {
    pub w: Vec3,
    pub on_a: Vec3,
    pub on_b: Vec3,
}

#[derive(Debug)]
pub enum GjkResult {
    /// 코어 간 분리. `normal`은 B→A 단위 방향(= A witness − B witness 정규화).
    Separated {
        distance: f32,
        on_a: Vec3,
        on_b: Vec3,
        normal: Vec3,
    },
    /// 코어 교차. 원점을 포함하는 사면체(EPA 시드).
    Intersecting([SupportPoint; 4]),
}

const MAX_ITER: usize = 64;
const REL_EPS: f32 = 1.0e-6;

#[inline]
fn support(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose, d: Vec3) -> SupportPoint {
    let on_a = pa.support_core(sa, d);
    let on_b = pb.support_core(sb, -d);
    SupportPoint {
        w: on_a - on_b,
        on_a,
        on_b,
    }
}

pub fn gjk(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose) -> GjkResult {
    let mut dir = pa.pos - pb.pos;
    if dir.length_sq() < 1.0e-12 {
        dir = Vec3::new(1.0, 0.0, 0.0);
    }
    let mut simplex: Vec<SupportPoint> = Vec::with_capacity(4);
    let mut v = dir; // 현재 심플렉스에서 원점까지의 최근접 벡터(초기값은 임의)

    for _ in 0..MAX_ITER {
        let w = support(sa, pa, sb, pb, -v);
        // 진전 없음 → 분리 확정: 현재 v가 최근접 벡터.
        // 첫 반복은 v가 초기 추정일 뿐이므로 심플렉스가 비어 있으면 판정하지 않는다
        // (빈 심플렉스로 witness를 만들면 오답 — sphere-sphere 정확 일치 케이스에서 실증).
        let vv = v.length_sq();
        if !simplex.is_empty() && vv - v.dot(w.w) <= REL_EPS * vv {
            return separated(&simplex, v);
        }
        simplex.push(w);
        let (new_v, contains) = closest_and_reduce(&mut simplex);
        if contains {
            debug_assert_eq!(simplex.len(), 4);
            return GjkResult::Intersecting([simplex[0], simplex[1], simplex[2], simplex[3]]);
        }
        v = new_v;
        if v.length_sq() <= 1.0e-12 {
            // 원점이 심플렉스 경계 위 — 접촉(깊이 0 교차)으로 처리하려면 사면체가 필요하므로
            // 미세 분리로 간주해 witness 반환(호출자에서 radius 재팽창 시 접촉이 됨).
            return separated(&simplex, v);
        }
    }
    separated(&simplex, v)
}

fn separated(simplex: &[SupportPoint], v: Vec3) -> GjkResult {
    let (on_a, on_b) = witness(simplex, v);
    let distance = v.length();
    let normal = if distance > 0.0 {
        v * (1.0 / distance)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    GjkResult::Separated {
        distance,
        on_a,
        on_b,
        normal,
    }
}

/// 현재 심플렉스(리덕션 후)에서 v = Σλw 를 만족하는 λ로 witness 복원.
fn witness(simplex: &[SupportPoint], v: Vec3) -> (Vec3, Vec3) {
    match simplex.len() {
        0 => (Vec3::ZERO, Vec3::ZERO),
        1 => (simplex[0].on_a, simplex[0].on_b),
        2 => {
            let (a, b) = (simplex[0].w, simplex[1].w);
            let ab = b - a;
            let t = if ab.length_sq() > 0.0 {
                ((v - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (
                simplex[0].on_a + (simplex[1].on_a - simplex[0].on_a) * t,
                simplex[0].on_b + (simplex[1].on_b - simplex[0].on_b) * t,
            )
        }
        _ => {
            // 삼각형: 바리센트릭 (원점 투영 v 기준)
            let (a, b, c) = (simplex[0].w, simplex[1].w, simplex[2].w);
            let (v0, v1, v2) = (b - a, c - a, v - a);
            let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
            let (d20, d21) = (v2.dot(v0), v2.dot(v1));
            let denom = d00 * d11 - d01 * d01;
            let (bv, cv) = if denom.abs() > 1.0e-12 {
                (
                    ((d11 * d20 - d01 * d21) / denom),
                    ((d00 * d21 - d01 * d20) / denom),
                )
            } else {
                (0.0, 0.0)
            };
            let av = 1.0 - bv - cv;
            (
                simplex[0].on_a * av + simplex[1].on_a * bv + simplex[2].on_a * cv,
                simplex[0].on_b * av + simplex[1].on_b * bv + simplex[2].on_b * cv,
            )
        }
    }
}

/// 심플렉스에서 원점 최근접점을 구하고, 지지 feature만 남긴다. (반환: 최근접 벡터, 원점 포함 여부)
fn closest_and_reduce(simplex: &mut Vec<SupportPoint>) -> (Vec3, bool) {
    match simplex.len() {
        1 => (simplex[0].w, false),
        2 => {
            let (a, b) = (simplex[1].w, simplex[0].w); // a=최신
            let ab = b - a;
            if ab.dot(-a) <= 0.0 {
                simplex.swap_remove(0); // a만 지지
                (a, false)
            } else {
                let t = (-a).dot(ab) / ab.length_sq();
                if t >= 1.0 {
                    simplex.swap_remove(1);
                    (b, false)
                } else {
                    (a + ab * t, false)
                }
            }
        }
        3 => closest_triangle(simplex),
        4 => closest_tetra(simplex),
        _ => unreachable!(),
    }
}

/// Ericson RTCD 5.1.5 — 삼각형 보로노이 영역 판정 + 리덕션.
fn closest_triangle(simplex: &mut Vec<SupportPoint>) -> (Vec3, bool) {
    let (pa, pb, pc) = (simplex[2], simplex[1], simplex[0]); // pa=최신
    let (a, b, c) = (pa.w, pb.w, pc.w);
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        simplex.clear();
        simplex.push(pa);
        return (a, false);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        simplex.clear();
        simplex.push(pb);
        return (b, false);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        simplex.clear();
        simplex.extend([pb, pa]);
        return (a + ab * t, false);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        simplex.clear();
        simplex.push(pc);
        return (c, false);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        simplex.clear();
        simplex.extend([pc, pa]);
        return (a + ac * t, false);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        simplex.clear();
        simplex.extend([pc, pb]);
        return (b + (c - b) * t, false);
    }
    // 면 내부
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    (a + ab * v + ac * w, false)
}

/// 사면체: 원점이 내부면 contains, 아니면 원점이 밖에 있는 면들 중 최근접 삼각형으로 리덕션.
fn closest_tetra(simplex: &mut Vec<SupportPoint>) -> (Vec3, bool) {
    let pts = [simplex[0], simplex[1], simplex[2], simplex[3]]; // [3]=최신
                                                                // 면 (i,j,k) — 남는 정점 l 반대편이 바깥
    const FACES: [[usize; 4]; 4] = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]];
    let mut best: Option<(f32, Vec<SupportPoint>, Vec3)> = None;
    let mut inside = true;
    for f in FACES {
        let (a, b, c, l) = (pts[f[0]].w, pts[f[1]].w, pts[f[2]].w, pts[f[3]].w);
        let mut n = (b - a).cross(c - a);
        if n.dot(l - a) > 0.0 {
            n = -n; // l 반대 방향 = 바깥
        }
        if n.dot(-a) > 0.0 {
            inside = false;
            let mut tri = vec![pts[f[0]], pts[f[1]], pts[f[2]]];
            let (cp, _) = closest_triangle(&mut tri);
            let d = cp.length_sq();
            if best.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
                best = Some((d, tri, cp));
            }
        }
    }
    if inside {
        return (Vec3::ZERO, true);
    }
    let (_, tri, cp) = best.unwrap();
    *simplex = tri;
    (cp, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(x: f32, y: f32, z: f32) -> Pose {
        Pose {
            pos: Vec3::new(x, y, z),
            rot: Quat::IDENTITY,
        }
    }

    #[test]
    fn sphere_sphere_distance_analytic() {
        // 코어=점이므로 코어 거리 = 중심 거리. 반지름 1짜리 두 구, 중심距 5 → 코어 5.
        let s = Shape::Sphere { radius: 1.0 };
        match gjk(&s, &pose(0.0, 0.0, 0.0), &s, &pose(5.0, 0.0, 0.0)) {
            GjkResult::Separated {
                distance,
                normal,
                on_a,
                on_b,
            } => {
                assert!((distance - 5.0).abs() < 1e-4, "{distance}");
                assert!((normal.x + 1.0).abs() < 1e-4); // v = A−B 방향 = −x
                assert!(on_a.length() < 1e-4 && (on_b.x - 5.0).abs() < 1e-4);
            }
            r => panic!("expected separated: {r:?}"),
        }
    }

    #[test]
    fn box_box_face_distance() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        match gjk(&s, &pose(0.0, 0.0, 0.0), &s, &pose(3.5, 0.0, 0.0)) {
            GjkResult::Separated { distance, .. } => {
                assert!((distance - 1.5).abs() < 1e-3, "{distance}")
            }
            r => panic!("expected separated: {r:?}"),
        }
    }

    #[test]
    fn box_box_deep_overlap_intersects() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        match gjk(&s, &pose(0.0, 0.0, 0.0), &s, &pose(0.5, 0.3, -0.2)) {
            GjkResult::Intersecting(_) => {}
            r => panic!("expected intersecting: {r:?}"),
        }
    }

    #[test]
    fn rotated_box_corner_distance() {
        // 45° 회전한 박스: 꼭짓점이 x축을 향함 — 코어 최근접 거리 = 3.5 − 1 − √2
        let a = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let b = a.clone();
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_4);
        let pb = Pose {
            pos: Vec3::new(3.5, 0.0, 0.0),
            rot,
        };
        match gjk(&a, &pose(0.0, 0.0, 0.0), &b, &pb) {
            GjkResult::Separated { distance, .. } => {
                let expect = 3.5 - 1.0 - 2.0f32.sqrt();
                assert!((distance - expect).abs() < 1e-3, "{distance} vs {expect}");
            }
            r => panic!("expected separated: {r:?}"),
        }
    }

    #[test]
    fn capsule_box_distance() {
        // 캡슐(y축, hh=1) 코어=세그먼트. 박스 코어와의 거리 = 4 − 1 = 3 (x축 분리)
        let cap = Shape::Capsule {
            half_height: 1.0,
            radius: 0.25,
        };
        let bx = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        match gjk(&cap, &pose(4.0, 0.0, 0.0), &bx, &pose(0.0, 0.0, 0.0)) {
            GjkResult::Separated { distance, .. } => {
                assert!((distance - 3.0).abs() < 1e-3, "{distance}")
            }
            r => panic!("expected separated: {r:?}"),
        }
    }

    #[test]
    fn deterministic_repeat() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 0.5, 2.0),
            radius: 0.0,
        };
        let pb = Pose {
            pos: Vec3::new(2.7, 1.1, -0.4),
            rot: Quat::from_axis_angle(Vec3::new(0.3, 1.0, 0.2), 0.9),
        };
        let d1 = match gjk(&s, &pose(0.0, 0.0, 0.0), &s, &pb) {
            GjkResult::Separated { distance, .. } => distance,
            _ => panic!(),
        };
        let d2 = match gjk(&s, &pose(0.0, 0.0, 0.0), &s, &pb) {
            GjkResult::Separated { distance, .. } => distance,
            _ => panic!(),
        };
        assert_eq!(d1.to_bits(), d2.to_bits(), "비트 단위 재현 실패");
    }
}
