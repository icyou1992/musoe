//! 광선·점 질의 — 셰이프 단위 primitive (월드 순회는 maru-dynamics).
//!
//! 계약:
//! - `toi`는 **광선 파라미터**다: 맞은 지점 = `origin + dir * toi`. `dir`은 정규화하지
//!   않아도 되지만, 정규화하면 `toi`가 곧 거리다.
//! - 광선이 셰이프 **안에서 시작**하면 `toi = 0`, 법선은 `-dir` 방향(관례: Bullet·rapier와 동일).
//!   "이미 안에 있다"를 못 맞은 것으로 처리하면 지면 판정이 조용히 실패한다.
//! - 반환 법선은 **바깥쪽 단위 벡터**다.
//! - convex radius가 있는 셰이프(둥근 박스·hull)는 코어에서 반지름만큼 부푼 형상으로 잰다.

use crate::{Pose, Shape};
use maru_math::Vec3;

/// 셰이프 하나와의 교차.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCast {
    /// 광선 파라미터. 맞은 지점 = `origin + dir * toi`.
    pub toi: f32,
    /// 맞은 지점의 바깥쪽 단위 법선.
    pub normal: Vec3,
    /// 광선 시작점이 셰이프 내부였는가 (그 경우 `toi == 0`).
    pub started_inside: bool,
}

/// 광선 × 셰이프. `max_toi`를 넘는 교차는 없는 것으로 본다.
///
/// 셰이프별 해석해가 있으면 그것을 쓰고(정확하고 빠르다), 없으면 support 함수 기반
/// 보수적 전진(GJK ray cast)으로 떨어진다 — hull과 둥근 박스가 그 경로다.
pub fn cast_ray(
    shape: &Shape,
    pose: &Pose,
    origin: Vec3,
    dir: Vec3,
    max_toi: f32,
) -> Option<RayCast> {
    // 셰이프 로컬로 옮겨 푼다 (자세는 단위 쿼터니언 — 입력 검증이 보장한다).
    let inv = pose.rot.conjugate();
    let o = inv.rotate(origin - pose.pos);
    let d = inv.rotate(dir);
    let local = match *shape {
        Shape::Sphere { radius } => ray_sphere(o, d, Vec3::ZERO, radius, max_toi),
        Shape::Capsule {
            half_height,
            radius,
        } => ray_capsule(o, d, half_height, radius, max_toi),
        Shape::Box { half, radius } => {
            if radius <= 0.0 {
                ray_box(o, d, half, max_toi)
            } else {
                ray_convex_ca(shape, o, d, max_toi)
            }
        }
        Shape::Hull { .. } => ray_convex_ca(shape, o, d, max_toi),
        Shape::Triangle { points, .. } => crate::heightfield::ray_triangle(o, d, points, max_toi)
            .map(|(toi, n)| Local {
                toi,
                normal: n,
                started_inside: false,
            }),
        // 하이트필드: 격자 DDA — 지나는 셀만 삼각형으로 시험한다.
        Shape::Heightfield { .. } => {
            let hf = shape.heightfield_ref().expect("Heightfield 변종");
            crate::heightfield::ray_heightfield(&hf, o, d, max_toi).map(|(toi, n)| Local {
                toi,
                normal: n,
                // 지형은 표면이다 — "안에서 시작"은 표면 아래를 뜻하고, 그건 아래
                // `contains_point`가 판정한다. 레이 자체는 면을 뚫는 순간을 보고한다.
                started_inside: false,
            })
        }
        // 복합: 자식별 최소 TOI. 동률은 앞선 자식(낮은 인덱스) — 결정론.
        Shape::Compound { ref children } => {
            let mut best: Option<Local> = None;
            for (cp, child) in children {
                // 자식 로컬로 다시 변환해 재귀 대신 직접 호출 (중첩은 validate가 금지).
                let world_child = Pose {
                    pos: pose.pos + pose.rot.rotate(cp.pos),
                    rot: pose.rot * cp.rot,
                };
                if let Some(hit) = cast_ray(child, &world_child, origin, dir, max_toi) {
                    let better = best.as_ref().is_none_or(|b| hit.toi < b.toi);
                    if better {
                        // 자식 히트는 이미 월드 법선 — 아래 공통 회전을 건너뛰기 위해
                        // 로컬 형식으로 역변환한다.
                        best = Some(Local {
                            toi: hit.toi,
                            normal: pose.rot.conjugate().rotate(hit.normal),
                            started_inside: hit.started_inside,
                        });
                    }
                }
            }
            best
        }
    }?;
    Some(RayCast {
        toi: local.toi,
        normal: pose.rot.rotate(local.normal),
        started_inside: local.started_inside,
    })
}

/// 점이 셰이프 안에 있는가 (겉면 기준 — convex radius 포함).
pub fn contains_point(shape: &Shape, pose: &Pose, p: Vec3) -> bool {
    let local = pose.rot.conjugate().rotate(p - pose.pos);
    let r = shape.radius();
    match *shape {
        Shape::Sphere { radius } => local.length_sq() <= radius * radius,
        Shape::Capsule {
            half_height,
            radius,
        } => {
            let y = local.y.clamp(-half_height, half_height);
            (local - Vec3::new(0.0, y, 0.0)).length_sq() <= radius * radius
        }
        Shape::Box { half, .. } => {
            // 겉면 기준 half가 곧 사용자 관점 치수다 (코어+radius = half).
            local.x.abs() <= half.x && local.y.abs() <= half.y && local.z.abs() <= half.z
        }
        Shape::Hull { .. } => {
            // 코어까지의 거리가 radius 이하인가 — support 기반 최근접점으로 판정한다.
            closest_core_dist(shape, local).0 <= r
        }
        // 삼각형은 부피가 없다 — radius가 있을 때만 코어까지 거리로 판정(hull과 같은 경로).
        Shape::Triangle { .. } => r > 0.0 && closest_core_dist(shape, local).0 <= r,
        // 하이트필드는 **표면 아래를 속이 찬 것으로** 본다(Jolt 동형): 격자 범위 안이고
        // 보간 높이 아래면 포함이다. 그래야 `query_point`가 지형에 묻힌 물체를 찾는다.
        Shape::Heightfield { .. } => {
            let hf = shape.heightfield_ref().expect("Heightfield 변종");
            match hf.height_at(local.x, local.z) {
                Some(h) => local.y <= h,
                None => false,
            }
        }
        // 복합: 어느 자식이든 포함하면 포함.
        Shape::Compound { ref children } => children.iter().any(|(cp, child)| {
            let world_child = Pose {
                pos: pose.pos + pose.rot.rotate(cp.pos),
                rot: pose.rot * cp.rot,
            };
            contains_point(child, &world_child, p)
        }),
    }
}

// ---------- 셰이프별 해석해 (로컬 좌표) ----------

struct Local {
    toi: f32,
    normal: Vec3,
    started_inside: bool,
}

fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f32, max_toi: f32) -> Option<Local> {
    let m = o - c;
    if m.length_sq() <= r * r {
        return Some(Local {
            toi: 0.0,
            normal: -d.normalize_or_zero(),
            started_inside: true,
        });
    }
    let a = d.length_sq();
    if a <= 0.0 {
        return None;
    }
    let b = m.dot(d);
    if b >= 0.0 {
        return None; // 구에서 멀어지는 방향
    }
    let cc = m.length_sq() - r * r;
    let disc = b * b - a * cc;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    if t < 0.0 || t > max_toi {
        return None;
    }
    let hit = o + d * t;
    Some(Local {
        toi: t,
        normal: (hit - c).normalize_or_zero(),
        started_inside: false,
    })
}

/// 축 정렬 박스 슬랩 판정 (radius 0 전용 — 둥근 박스는 보수적 전진으로 간다).
fn ray_box(o: Vec3, d: Vec3, half: Vec3, max_toi: f32) -> Option<Local> {
    if o.x.abs() <= half.x && o.y.abs() <= half.y && o.z.abs() <= half.z {
        return Some(Local {
            toi: 0.0,
            normal: -d.normalize_or_zero(),
            started_inside: true,
        });
    }
    let (mut tmin, mut tmax) = (0.0f32, max_toi);
    let mut axis = 0usize;
    let mut sign = 1.0f32;
    let (od, dd, hd) = ([o.x, o.y, o.z], [d.x, d.y, d.z], [half.x, half.y, half.z]);
    for i in 0..3 {
        if dd[i].abs() < 1e-20 {
            // 축에 평행 — 그 축의 슬랩 밖이면 절대 못 맞는다
            if od[i].abs() > hd[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dd[i];
        let mut t1 = (-hd[i] - od[i]) * inv;
        let mut t2 = (hd[i] - od[i]) * inv;
        let mut s = -1.0f32;
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
            s = 1.0;
        }
        if t1 > tmin {
            tmin = t1;
            axis = i;
            sign = s;
        }
        tmax = tmax.min(t2);
        if tmin > tmax {
            return None;
        }
    }
    let mut n = Vec3::ZERO;
    match axis {
        0 => n.x = sign,
        1 => n.y = sign,
        _ => n.z = sign,
    }
    Some(Local {
        toi: tmin,
        normal: n,
        started_inside: false,
    })
}

/// 광선 × 캡슐 (Y축 세그먼트 + radius).
fn ray_capsule(o: Vec3, d: Vec3, half_height: f32, r: f32, max_toi: f32) -> Option<Local> {
    // 내부에서 시작?
    let y0 = o.y.clamp(-half_height, half_height);
    if (o - Vec3::new(0.0, y0, 0.0)).length_sq() <= r * r {
        return Some(Local {
            toi: 0.0,
            normal: -d.normalize_or_zero(),
            started_inside: true,
        });
    }
    // 원통부: xz 평면에서 원과의 교차 → y가 세그먼트 안이면 채택
    let mut best: Option<Local> = None;
    let mut consider = |t: f32, n: Vec3| {
        if t >= 0.0 && t <= max_toi && best.as_ref().is_none_or(|b| t < b.toi) {
            best = Some(Local {
                toi: t,
                normal: n,
                started_inside: false,
            });
        }
    };
    let a = d.x * d.x + d.z * d.z;
    if a > 1e-20 {
        let b = o.x * d.x + o.z * d.z;
        let c = o.x * o.x + o.z * o.z - r * r;
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let t = (-b - disc.sqrt()) / a;
            let y = o.y + d.y * t;
            if y >= -half_height && y <= half_height {
                let p = o + d * t;
                consider(t, Vec3::new(p.x, 0.0, p.z).normalize_or_zero());
            }
        }
    }
    // 양 끝 반구
    for cy in [-half_height, half_height] {
        let c = Vec3::new(0.0, cy, 0.0);
        if let Some(l) = ray_sphere(o, d, c, r, max_toi) {
            // 반구 쪽만 채택 (원통부와 겹치는 절반은 위에서 이미 봤다)
            let p = o + d * l.toi;
            let keep = if cy > 0.0 { p.y >= cy } else { p.y <= cy };
            if keep {
                consider(l.toi, l.normal);
            }
        }
    }
    best
}

// ---------- 일반 convex: support 기반 보수적 전진 ----------

/// 로컬 점에서 **코어**까지의 거리와 바깥 법선.
///
/// 새로 GJK를 쓰지 않고 **크레이트의 검증된 GJK를 그대로 재사용한다** — 점을 반지름 0
/// 구로 모델링하면 점-볼록체 거리는 셰이프-셰이프 거리의 특수한 경우다.
/// (처음엔 점 전용 GJK를 따로 썼는데 심플렉스 축약을 대충 만들어 면 위 최근접점을
///  놓쳤고, 보수적 전진이 통째로 실패했다. 검증된 코드를 부르는 편이 짧고 옳다.)
fn closest_core_dist(shape: &Shape, p: Vec3) -> (f32, Vec3) {
    let point = Shape::Sphere { radius: 0.0 };
    let at = Pose {
        pos: p,
        rot: maru_math::Quat::IDENTITY,
    };
    let ident = Pose {
        pos: Vec3::ZERO,
        rot: maru_math::Quat::IDENTITY,
    };
    match crate::gjk::gjk(&point, &at, shape, &ident) {
        crate::gjk::GjkResult::Separated {
            distance, normal, ..
        } => (distance, normal),
        // 코어 안 — 거리 0
        crate::gjk::GjkResult::Intersecting(_) => (0.0, Vec3::ZERO),
    }
}

/// 보수적 전진 광선 캐스트 — 임의 convex(코어 + radius)에 쓴다.
///
/// 매 반복에서 현재 점의 표면까지 거리 `d`를 구하고, 광선이 그만큼은 안전하게 나아갈 수
/// 있으므로 `t += d / |dir 성분|`으로 전진한다. 표면에 닿으면 종료한다.
fn ray_convex_ca(shape: &Shape, o: Vec3, d: Vec3, max_toi: f32) -> Option<Local> {
    let r = shape.radius();
    let dlen = d.length();
    if dlen <= 0.0 {
        return None;
    }
    // 스케일 의존 종료 임계 — 절대 상수는 장면 스케일에서 깨진다(설계 §8).
    let scale = crate::characteristic_length(shape, shape).max(1e-6);
    let eps = 1e-4 * scale;

    let (d0, n0) = closest_core_dist(shape, o);
    if d0 <= r {
        return Some(Local {
            toi: 0.0,
            normal: -d * (1.0 / dlen),
            started_inside: true,
        });
    }
    let mut t = 0.0f32;
    let mut normal = n0;
    for _ in 0..64 {
        let p = o + d * t;
        let (dist, n) = closest_core_dist(shape, p);
        let gap = dist - r;
        if gap <= eps {
            // **0 법선을 내보내지 않는다.** 전진이 반올림으로 코어 안으로 한 발 넘어가면
            // GJK가 `Intersecting`을 주고 `closest_core_dist`가 방향을 0으로 돌려준다.
            // 그대로 반환하면 "법선은 바깥쪽 단위 벡터"라는 공개 계약이 깨지고
            // 호스트의 반사·슬라이딩이 0/NaN이 된다 (검수 rev_710b92d2 MAJOR).
            // 직전 반복의 법선이 여전히 유효한 바깥 방향이다.
            let hit_normal = if n.length_sq() > 0.0 { n } else { normal };
            return Some(Local {
                toi: t,
                normal: hit_normal,
                started_inside: false,
            });
        }
        if n.length_sq() > 0.0 {
            normal = n;
        }
        // 광선이 표면 쪽으로 다가가는 속도. 멀어지면 절대 못 맞는다.
        let approach = -d.dot(n);
        if approach <= 1e-20 {
            return None;
        }
        t += gap / approach;
        if t > max_toi {
            return None;
        }
    }
    // 반복 상한 — 수렴 못 했으면 못 맞은 것으로 본다(거짓 히트보다 낫다).
    let _ = normal;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use maru_math::Quat;

    fn ident() -> Pose {
        Pose {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }

    #[test]
    fn sphere_hit_and_miss() {
        let s = Shape::Sphere { radius: 1.0 };
        let h = cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
        assert!((h.toi - 4.0).abs() < 1e-4, "toi={}", h.toi);
        assert!((h.normal - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4);
        assert!(!h.started_inside);
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 3.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0
        )
        .is_none());
        // 뒤로 쏘면 안 맞는다
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            100.0
        )
        .is_none());
    }

    #[test]
    fn ray_starting_inside_reports_zero_toi() {
        for s in [
            Shape::Sphere { radius: 1.0 },
            Shape::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            Shape::Box {
                half: Vec3::new(1.0, 1.0, 1.0),
                radius: 0.0,
            },
        ] {
            let h = cast_ray(&s, &ident(), Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 10.0)
                .unwrap_or_else(|| panic!("내부 시작인데 못 맞았다: {s:?}"));
            assert_eq!(h.toi, 0.0);
            assert!(h.started_inside);
        }
    }

    #[test]
    fn box_face_normals_are_axis_aligned() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.0,
        };
        let cases = [
            (
                Vec3::new(-9.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                8.0,
                Vec3::new(-1.0, 0.0, 0.0),
            ),
            (
                Vec3::new(0.0, 9.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                7.0,
                Vec3::new(0.0, 1.0, 0.0),
            ),
            (
                Vec3::new(0.0, 0.0, -9.0),
                Vec3::new(0.0, 0.0, 1.0),
                6.0,
                Vec3::new(0.0, 0.0, -1.0),
            ),
        ];
        for (o, d, want_t, want_n) in cases {
            let h = cast_ray(&s, &ident(), o, d, 100.0).unwrap();
            assert!((h.toi - want_t).abs() < 1e-4, "toi={} want {want_t}", h.toi);
            assert!((h.normal - want_n).length() < 1e-4, "n={:?}", h.normal);
        }
    }

    /// 회전한 자세에서도 맞아야 한다 — 로컬 변환이 맞는지 가른다.
    #[test]
    fn rotated_pose_is_handled() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let p = Pose {
            pos: Vec3::new(5.0, 0.0, 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_4),
        };
        let h = cast_ray(&s, &p, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 100.0).unwrap();
        // 45° 돌린 정육면체의 x쪽 최근접점은 모서리 → 중심에서 √2 떨어져 있다
        let want = 5.0 - 2.0f32.sqrt();
        assert!((h.toi - want).abs() < 1e-3, "toi={} want {want}", h.toi);
    }

    /// 보수적 전진 경로(둥근 박스·hull)가 해석해와 어긋나지 않아야 한다.
    #[test]
    fn conservative_advancement_matches_analytic_on_a_sphere_like_hull() {
        // 정팔면체 hull + radius → 코어가 작아 구에 가까운 형상
        let pts = vec![
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, -0.5),
        ];
        let hull = Shape::Hull {
            points: pts,
            radius: 0.25,
        };
        // +x로 쏘면 코어 최원점 0.5 + radius 0.25 = 0.75 에서 맞는다
        let h = cast_ray(
            &hull,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
        assert!((h.toi - 4.25).abs() < 5e-3, "toi={}", h.toi);
        assert!(h.normal.x < -0.99, "n={:?}", h.normal);
    }

    /// 둥근 박스: 면 중앙은 half까지, 모서리 방향은 더 가깝다.
    #[test]
    fn rounded_box_face_distance() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.25,
        };
        let h = cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
        assert!(
            (h.toi - 4.0).abs() < 5e-3,
            "면 중앙은 half=1.0에서 맞아야 한다: {}",
            h.toi
        );
    }

    /// **성공 결과의 법선은 항상 단위 벡터**여야 한다 (공개 계약).
    ///
    /// 보수적 전진이 코어 안으로 한 발 넘어가면 GJK가 `Intersecting`을 주고 방향이
    /// 0이 된다 — 그대로 내보내면 호스트의 반사 계산이 0/NaN이 된다.
    #[test]
    fn conservative_advancement_never_returns_a_zero_normal() {
        let mut rng = 0x1234_5678u32;
        let mut f = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32
        };
        let shapes = [
            Shape::Box {
                half: Vec3::new(1.0, 0.05, 1.0), // 얇은 판 — 스치는 각이 잘 나온다
                radius: 0.02,
            },
            Shape::Hull {
                points: vec![
                    Vec3::new(0.6, 0.0, 0.0),
                    Vec3::new(-0.6, 0.0, 0.0),
                    Vec3::new(0.0, 0.03, 0.5),
                    Vec3::new(0.0, -0.03, -0.5),
                ],
                radius: 0.05,
            },
        ];
        let mut hits = 0;
        for s in &shapes {
            for _ in 0..4000 {
                let o = Vec3::new((f() - 0.5) * 8.0, (f() - 0.5) * 8.0, (f() - 0.5) * 8.0);
                // 셰이프 근처를 스치도록 목표점을 형상 경계 부근으로 잡는다
                let target = Vec3::new((f() - 0.5) * 2.2, (f() - 0.5) * 0.3, (f() - 0.5) * 2.2);
                let d = target - o;
                if d.length_sq() < 1e-6 {
                    continue;
                }
                let Some(h) = cast_ray(s, &ident(), o, d, 100.0) else {
                    continue;
                };
                hits += 1;
                let l2 = h.normal.length_sq();
                assert!(
                    (0.9..=1.1).contains(&l2),
                    "법선이 단위가 아니다: {:?} (|n|²={l2}) shape={s:?} o={o:?} d={d:?}",
                    h.normal
                );
            }
        }
        assert!(hits > 500, "표본이 너무 적다: {hits}");
    }

    #[test]
    fn max_toi_is_respected() {
        let s = Shape::Sphere { radius: 1.0 };
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            3.0
        )
        .is_none());
        assert!(cast_ray(
            &s,
            &ident(),
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            4.5
        )
        .is_some());
    }

    #[test]
    fn point_containment() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.0,
        };
        assert!(contains_point(&s, &ident(), Vec3::new(0.9, 1.9, 2.9)));
        assert!(!contains_point(&s, &ident(), Vec3::new(1.1, 0.0, 0.0)));
        let sp = Shape::Sphere { radius: 1.0 };
        assert!(contains_point(&sp, &ident(), Vec3::new(0.5, 0.5, 0.5)));
        assert!(!contains_point(&sp, &ident(), Vec3::new(0.7, 0.7, 0.7)));
    }
}
