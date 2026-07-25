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

/// 사면체가 EPA 시드로 쓸 수 있는 부피를 갖는가.
/// **상대 임계**: 절대 임계는 스케일에 취약하다 — 1m 박스 쌍의 vol=5e-10 시드가 절대 1e-10을
/// 통과해 EPA 폴리토프를 폭주시킨 실사고(피라미드 100단, 인접 박스 면접촉)로 도입.
#[inline]
fn tetra_has_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    let vol = (b - a).cross(c - a).dot(d - a);
    let max_e2 = [
        (b - a).length_sq(),
        (c - a).length_sq(),
        (d - a).length_sq(),
        (c - b).length_sq(),
        (d - b).length_sq(),
        (d - c).length_sq(),
    ]
    .iter()
    .fold(0.0f32, |m, v| m.max(*v));
    let scale_cubed = max_e2 * max_e2.sqrt(); // L³
    vol.abs() > 1.0e-6 * scale_cubed.max(1.0e-20)
}

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
    let mut v = dir; // 현재 심플렉스에서 원점까지의 최근접 벡터(초기값은 임의 추정)

    for _ in 0..MAX_ITER {
        // 깊이 0 접촉(v=0)에서 법선을 잃지 않도록 마지막 유효 탐색 방향을 승계한다
        // (리뷰 BLOCKER: 동일 박스 정확 면접촉에서 첫 support가 w=0이면 +X 오답 법선).
        let previous_v = v;
        let w = support(sa, pa, sb, pb, -v);
        // 진전 없음 → 분리 확정. 첫 반복(빈 심플렉스)은 v가 추정일 뿐이라 판정하지 않는다.
        let vv = v.length_sq();
        if !simplex.is_empty() && vv - v.dot(w.w) <= REL_EPS * vv {
            return separated(&simplex, v);
        }
        simplex.push(w);
        let (new_v, contains) = closest_and_reduce(&mut simplex);
        if contains {
            debug_assert_eq!(simplex.len(), 4);
            // 퇴화(평면) 사면체는 EPA 시드가 될 수 없음 → 접촉(깊이≈0)으로 강등
            let (a, b, c, d) = (simplex[0].w, simplex[1].w, simplex[2].w, simplex[3].w);
            if !tetra_has_volume(a, b, c, d) {
                return separated_with_normal(&simplex[..3], Vec3::ZERO, previous_v);
            }
            return GjkResult::Intersecting([simplex[0], simplex[1], simplex[2], simplex[3]]);
        }
        if new_v.length_sq() <= 1.0e-12 {
            // 원점이 심플렉스 위. 리덕션 루프를 계속 돌면 수직 확장이 매번 같은 심플렉스로
            // 되돌아와 순환한다(축 정렬 깊은 교차에서 실증) — 리덕션 없이 즉시 사면체를 완성.
            return complete_on_boundary(&simplex, sa, pa, sb, pb, previous_v);
        }
        v = new_v;
    }
    separated(&simplex, v)
}

/// 원점이 현재 심플렉스 위에 있을 때: 리덕션 없이 결정적 방향들로 사면체를 완성해
/// Intersecting(EPA 시드)으로 승격하거나, 폴리토프가 평면(진짜 접촉)이면 깊이 0 분리로 마감.
fn complete_on_boundary(
    simplex: &[SupportPoint],
    sa: &Shape,
    pa: &Pose,
    sb: &Shape,
    pb: &Pose,
    fallback_normal: Vec3,
) -> GjkResult {
    let sup = |d: Vec3| support(sa, pa, sb, pb, d);
    match simplex.len() {
        // 원점이 support 정점과 일치 — 표면 접촉 (법선은 마지막 유효 탐색 방향 승계)
        1 => separated_with_normal(simplex, Vec3::ZERO, fallback_normal),
        2 => {
            let (a, b) = (simplex[0], simplex[1]);
            let ab = b.w - a.w;
            let p0 = ab.cross(Vec3::new(1.0, 0.0, 0.0));
            let p = if p0.length_sq() > 1.0e-12 {
                p0
            } else {
                ab.cross(Vec3::new(0.0, 1.0, 0.0))
            };
            let q = ab.cross(p);
            // 같은 probe의 ±support는 대칭 형상에서 세그먼트와 공면이 된다(축 정렬 박스 실증)
            // — 서로 다른 직교 probe의 support를 섞어야 3차원을 스팬한다.
            let cands = [sup(p), sup(-p), sup(q), sup(-q)];
            for i in 0..cands.len() {
                for s2 in cands.iter().skip(i + 1) {
                    let s1 = cands[i];
                    if tetra_has_volume(a.w, b.w, s1.w, s2.w) {
                        return GjkResult::Intersecting([a, b, s1, *s2]);
                    }
                }
            }
            separated_with_normal(simplex, Vec3::ZERO, fallback_normal)
        }
        _ => {
            let (a, b, c) = (simplex[0], simplex[1], simplex[2]);
            let n = (b.w - a.w).cross(c.w - a.w);
            if n.length_sq() <= 1.0e-12 {
                return separated_with_normal(simplex, Vec3::ZERO, fallback_normal);
            }
            let s1 = sup(n);
            if tetra_has_volume(a.w, b.w, c.w, s1.w) {
                return GjkResult::Intersecting([a, b, c, s1]);
            }
            let s2 = sup(-n);
            if tetra_has_volume(a.w, b.w, c.w, s2.w) {
                return GjkResult::Intersecting([a, b, c, s2]);
            }
            separated_with_normal(simplex, Vec3::ZERO, fallback_normal)
        }
    }
}

fn separated(simplex: &[SupportPoint], v: Vec3) -> GjkResult {
    separated_with_normal(simplex, v, Vec3::new(1.0, 0.0, 0.0))
}

fn separated_with_normal(simplex: &[SupportPoint], v: Vec3, fallback: Vec3) -> GjkResult {
    let (on_a, on_b) = witness(simplex, v);
    let distance = v.length();
    let normal = if distance > 0.0 {
        v * (1.0 / distance)
    } else {
        let l2 = fallback.length_sq();
        if l2 > 0.0 {
            fallback * (1.0 / l2.sqrt())
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        }
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
    fn identical_boxes_exact_face_touch_keeps_separating_axis() {
        // 리뷰 BLOCKER 회귀: 동일 박스 정확 면접촉 — 법선이 +X로 무너지면 안 됨
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        match gjk(&s, &pose(0.0, 2.0, 0.0), &s, &pose(0.0, 0.0, 0.0)) {
            GjkResult::Separated {
                distance, normal, ..
            } => {
                assert!(distance < 1.0e-6, "distance={distance}");
                assert!((normal.y - 1.0).abs() < 1.0e-6, "normal={normal:?}");
            }
            r => panic!("expected touching separation: {r:?}"),
        }
    }

    /// 회귀(피라미드 100단 실사고): 인접 박스가 거의 정확히 면접촉하면 GJK가 부피 5e-10짜리
    /// 사면체를 EPA 시드로 넘겨 폴리토프가 폭주했다(스텝 16에서 무한 확장). 상대 퇴화 판정으로
    /// 깊이 0 접촉으로 강등되어야 한다.
    #[test]
    fn near_flat_intersection_degrades_to_zero_depth_contact() {
        let s = Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        };
        let pa = Pose {
            pos: Vec3::new(-19.046108, 55.40236, 0.0023186088),
            rot: Quat {
                x: 0.00012426352,
                y: -0.0014365312,
                z: -0.001989146,
                w: 0.99999696,
            },
        };
        let pb = Pose {
            pos: Vec3::new(-18.045982, 55.3985, 0.0027412514),
            rot: Quat {
                x: 0.0006017331,
                y: -0.0013692725,
                z: -0.001860802,
                w: 0.9999972,
            },
        };
        match gjk(&s, &pa, &s, &pb) {
            GjkResult::Separated {
                distance, normal, ..
            } => {
                assert!(distance < 1.0e-4, "distance={distance}");
                // A는 B의 왼쪽 → −x로 밀어야 분리
                assert!(normal.x < -0.99, "normal={normal:?}");
            }
            GjkResult::Intersecting(seed) => {
                // 시드를 넘긴다면 EPA가 쓸 수 있는 부피여야 한다
                let v: Vec<Vec3> = seed.iter().map(|p| p.w).collect();
                let vol = (v[1] - v[0]).cross(v[2] - v[0]).dot(v[3] - v[0]).abs();
                assert!(vol > 1.0e-7, "퇴화 시드가 EPA로 새어나감 vol={vol:e}");
            }
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
