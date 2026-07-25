//! 접촉 매니폴드 생성 — face clipping + canonical feature ID (설계 §4.2).
//!
//! - GJK/EPA가 준 접촉 법선으로 reference/incident 면을 고르고(동률=낮은 면 인덱스,
//!   ref 소속 동률=A — 결정론), incident 면을 ref 측면 평면들로 클리핑해 최대 4점 생성.
//! - **feature ID는 단일 정본 함수**: 클리핑 계보(원본 정점/에지×평면)에서 결정적으로 부여.
//!   PCM 캐시 매칭·중복 판정 키(설계 §4)가 이 ID를 소비한다.
//! - box-box 면 접촉이 4점을 내는 것이 M1 완료 기준. 면 지배적이 아닌 접촉(에지-에지 등)과
//!   비박스 쌍은 witness 단일점(v1 명시 범위 — hull 면 클리핑은 hull 토폴로지 빌더와 함께).

use crate::epa::epa;
use crate::gjk::{gjk, GjkResult};
use crate::{Pose, Shape};
use maru_math::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeatureId(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct ContactPoint {
    /// 월드 좌표 (B 표면 기준).
    pub position: Vec3,
    /// 부호 있는 간격: 음수 = 침투. (겉면 기준 — box는 half가 겉면)
    pub separation: f32,
    pub feature: FeatureId,
}

#[derive(Clone, Debug, Default)]
pub struct Manifold {
    /// A를 +normal로 밀면 분리 (GJK/EPA 관례 승계).
    pub normal: Vec3,
    pub points: Vec<ContactPoint>,
}

/// 면 지배 판정 임계: |면법선·접촉법선| 이 이 값 이상이면 face contact로 취급.
const FACE_DOT_MIN: f32 = 0.95;

/// 접촉 생성 진입점. `margin` = speculative margin (양의 separation 허용 한계).
pub fn contact(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose, margin: f32) -> Option<Manifold> {
    let (ra, rb) = (sa.radius(), sb.radius());
    let (normal, core_sep, on_a, on_b) = match gjk(sa, pa, sb, pb) {
        GjkResult::Separated {
            distance,
            on_a,
            on_b,
            normal,
        } => {
            let sep = distance - ra - rb;
            if sep > margin {
                return None;
            }
            (normal, sep, on_a, on_b)
        }
        GjkResult::Intersecting(seed) => {
            let p = epa(seed, sa, pa, sb, pb);
            (p.normal, -(p.depth + ra + rb), p.on_a, p.on_b)
        }
    };

    // box-box + 면 지배 → 클리핑 매니폴드
    if let (Shape::Box { half: ha, .. }, Shape::Box { half: hb, .. }) = (sa, sb) {
        if let Some(m) = clip_box_box(*ha, pa, *hb, pb, normal, margin) {
            return Some(m);
        }
    }

    // 폴백: witness 단일점 (radius 재팽창 — 설계 shrink-then-reinflate의 reinflate 지점)
    let surface_a = on_a - normal * ra; // A 표면: 코어점에서 −normal(=B쪽)로 radius만큼
    let surface_b = on_b + normal * rb;
    Some(Manifold {
        normal,
        points: vec![ContactPoint {
            position: (surface_a + surface_b) * 0.5,
            separation: core_sep,
            feature: FeatureId(0xFFFF_0000), // witness 단일점 feature (에지-에지/곡면)
        }],
    })
}

// ---------- box face 유틸 ----------

/// 면 f = axis*2 + (양의 방향이면 1). 월드 법선.
fn box_face_normal(f: usize, rot: Quat) -> Vec3 {
    let axis = f / 2;
    let sign = if f % 2 == 1 { 1.0 } else { -1.0 };
    let local = match axis {
        0 => Vec3::new(sign, 0.0, 0.0),
        1 => Vec3::new(0.0, sign, 0.0),
        _ => Vec3::new(0.0, 0.0, sign),
    };
    rot.rotate(local)
}

/// 면 f의 4개 꼭짓점 (로컬, 바깥에서 볼 때 CCW — 결정적 순서).
fn box_face_verts(f: usize, half: Vec3) -> [Vec3; 4] {
    let axis = f / 2;
    let s = if f % 2 == 1 { 1.0 } else { -1.0 };
    let (h, u, v) = match axis {
        0 => (
            half.x,
            Vec3::new(0.0, half.y, 0.0),
            Vec3::new(0.0, 0.0, half.z),
        ),
        1 => (
            half.y,
            Vec3::new(0.0, 0.0, half.z),
            Vec3::new(half.x, 0.0, 0.0),
        ),
        _ => (
            half.z,
            Vec3::new(half.x, 0.0, 0.0),
            Vec3::new(0.0, half.y, 0.0),
        ),
    };
    let n = match axis {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    };
    let c = n * h;
    // s<0이면 u,v 순서를 바꿔 CCW 유지
    if s > 0.0 {
        [c + u + v, c - u + v, c - u - v, c + u - v]
    } else {
        [c + u + v, c + u - v, c - u - v, c - u + v]
    }
}

/// 법선과 가장 정렬된 면 선택 (동률: 낮은 면 인덱스 — 결정론). 반환 (face, dot).
fn best_face(rot: Quat, want: Vec3) -> (usize, f32) {
    let mut best = 0;
    let mut best_dot = -f32::INFINITY;
    for f in 0..6 {
        let d = box_face_normal(f, rot).dot(want);
        if d > best_dot + 1.0e-6 {
            best_dot = d;
            best = f;
        }
    }
    (best, best_dot)
}

/// box-box 면 클리핑. 법선이 면 지배적이 아니면 None(→ witness 폴백).
fn clip_box_box(
    ha: Vec3,
    pa: &Pose,
    hb: Vec3,
    pb: &Pose,
    normal: Vec3,
    margin: f32,
) -> Option<Manifold> {
    // A에서 B를 향하는 면(법선 ≈ −normal), B에서 A를 향하는 면(≈ +normal)
    let (fa, da) = best_face(pa.rot, -normal);
    let (fb, db) = best_face(pb.rot, normal);
    if da.max(db) < FACE_DOT_MIN {
        return None; // 에지-에지 지배 — witness 단일점 경로
    }
    // reference = 더 정렬된 쪽 (동률 = A — 결정론)
    let ref_is_a = da >= db;
    let (rf, rh, rp, inf, ih, ip) = if ref_is_a {
        (fa, ha, pa, fb, hb, pb)
    } else {
        (fb, hb, pb, fa, ha, pa)
    };

    let rn = box_face_normal(rf, rp.rot); // reference 바깥 법선 (월드)
    let rverts: Vec<Vec3> = box_face_verts(rf, rh)
        .iter()
        .map(|v| rp.pos + rp.rot.rotate(*v))
        .collect();
    // incident = 상대 박스에서 rn과 가장 반대인 면
    let (inc_f, _) = best_face(ip.rot, -rn);
    let iverts: Vec<Vec3> = box_face_verts(inc_f, ih)
        .iter()
        .map(|v| ip.pos + ip.rot.rotate(*v))
        .collect();
    let _ = inf;

    // Sutherland–Hodgman: incident 폴리곤을 reference 4 측면 평면으로 클리핑 (계보 추적)
    let mut poly: Vec<(Vec3, u32)> = iverts
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, 0x100 | i as u32))
        .collect();
    for side in 0..4 {
        let e0 = rverts[side];
        let e1 = rverts[(side + 1) % 4];
        // 측면 평면: 법선 = rn × edge (안쪽을 향하도록)
        let mut pn = rn.cross(e1 - e0);
        let center = (rverts[0] + rverts[1] + rverts[2] + rverts[3]) * 0.25;
        if pn.dot(center - e0) < 0.0 {
            pn = -pn;
        }
        let mut out: Vec<(Vec3, u32)> = Vec::with_capacity(8);
        for i in 0..poly.len() {
            let (v0, t0) = poly[i];
            let (v1, _t1) = poly[(i + 1) % poly.len()];
            let d0 = pn.dot(v0 - e0);
            let d1 = pn.dot(v1 - e0);
            if d0 >= 0.0 {
                out.push((v0, t0));
            }
            if (d0 >= 0.0) != (d1 >= 0.0) {
                let t = d0 / (d0 - d1);
                let x = v0 + (v1 - v0) * t;
                // 교점 feature = (원본 에지 시작 태그, 클리핑 측면) — 결정적
                out.push((x, 0x200 | ((t0 & 0xFF) << 4) | side as u32));
            }
        }
        poly = out;
        if poly.is_empty() {
            return None;
        }
    }

    // reference 평면 아래(margin 이내)만 채택, separation = 평면까지 부호 거리
    let mut pts: Vec<ContactPoint> = Vec::with_capacity(8);
    let ref_face_bits = ((ref_is_a as u32) << 31) | ((rf as u32) << 24);
    for (v, tag) in poly {
        let sep = rn.dot(v - rverts[0]);
        if sep <= margin {
            // 접점 위치는 reference 면 위로 투영한 점과 incident 점의 중간
            let proj = v - rn * sep;
            pts.push(ContactPoint {
                position: (v + proj) * 0.5,
                separation: sep,
                feature: FeatureId(ref_face_bits | tag),
            });
        }
    }
    if pts.is_empty() {
        return None;
    }

    // 4점 축소 (설계 §4.2): 최심점 고정 + 나머지는 면적 최대 조합(그리디, 결정적)
    if pts.len() > 4 {
        pts = reduce_to_4(pts, rn);
    }
    // manifold 법선은 push-A 관례로: reference가 A면 rn은 A→B → 뒤집기
    let m_normal = if ref_is_a { -rn } else { rn };
    Some(Manifold {
        normal: m_normal,
        points: pts,
    })
}

/// 최심점 고정 → 최원점 → 남은 점 중 삼각형 면적 최대 2점 (동률=낮은 feature — 결정론).
fn reduce_to_4(pts: Vec<ContactPoint>, n: Vec3) -> Vec<ContactPoint> {
    let deepest = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.separation
                .partial_cmp(&b.separation)
                .unwrap()
                .then(a.feature.cmp(&b.feature))
        })
        .map(|(i, _)| i)
        .unwrap();
    let p0 = pts[deepest];
    let farthest = pts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != deepest)
        .max_by(|(_, a), (_, b)| {
            let da = (a.position - p0.position).length_sq();
            let db = (b.position - p0.position).length_sq();
            da.partial_cmp(&db).unwrap().then(b.feature.cmp(&a.feature))
        })
        .map(|(i, _)| i)
        .unwrap();
    let p1 = pts[farthest];
    let mut best2 = (0.0f32, None::<usize>, None::<usize>);
    for (i, ci) in pts.iter().enumerate() {
        if i == deepest || i == farthest {
            continue;
        }
        let area1 = (p1.position - p0.position)
            .cross(ci.position - p0.position)
            .dot(n)
            .abs();
        for (j, cj) in pts.iter().enumerate().skip(i + 1) {
            if j == deepest || j == farthest {
                continue;
            }
            let area2 = (ci.position - p0.position)
                .cross(cj.position - p0.position)
                .dot(n)
                .abs();
            let total = area1 + area2;
            if total > best2.0 {
                best2 = (total, Some(i), Some(j));
            }
        }
    }
    let mut out = vec![p0, p1];
    if let (Some(i), Some(j)) = (best2.1, best2.2) {
        out.push(pts[i]);
        out.push(pts[j]);
    }
    out.sort_by_key(|p| p.feature);
    out
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

    /// M1 완료 기준: box 면 접촉이 4점 매니폴드를 낸다.
    #[test]
    fn box_on_box_face_gives_4_points() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        // top이 ground 위에 딱 얹힘 (y: 1+1=2)
        let m = contact(
            &top,
            &pose(0.0, 2.0, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .expect("contact expected");
        assert_eq!(m.points.len(), 4, "{:?}", m.points);
        for p in &m.points {
            assert!(p.separation.abs() < 1e-4, "{:?}", p);
        }
        assert!(
            (m.normal.y - 1.0).abs() < 1e-4,
            "top을 +y로 밀면 분리: {:?}",
            m.normal
        );
        // 4점의 feature는 서로 달라야 함
        let mut feats: Vec<_> = m.points.iter().map(|p| p.feature).collect();
        feats.dedup();
        assert_eq!(feats.len(), 4);
    }

    /// M1 완료 기준: feature ID 안정성 — 미세 섭동에도 같은 feature 집합.
    #[test]
    fn feature_ids_stable_under_perturbation() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let base = contact(
            &top,
            &pose(0.0, 1.999, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let perturbed = contact(
            &top,
            &pose(1.0e-4, 1.999, -2.0e-4),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let f0: Vec<_> = base.points.iter().map(|p| p.feature).collect();
        let f1: Vec<_> = perturbed.points.iter().map(|p| p.feature).collect();
        assert_eq!(f0, f1, "feature 집합이 섭동에 흔들림");
    }

    /// 회전 슬라이드 케이스: yaw 회전한 박스가 얹혀도 ≤4점 + 전부 접촉권.
    #[test]
    fn rotated_box_on_ground_clipped() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.6);
        let pa = Pose {
            pos: Vec3::new(0.0, 1.995, 0.0),
            rot,
        };
        let m = contact(&top, &pa, &ground, &pose(0.0, 0.0, 0.0), 0.05).unwrap();
        assert!(
            !m.points.is_empty() && m.points.len() <= 4,
            "{:?}",
            m.points.len()
        );
        for p in &m.points {
            assert!(p.separation < 0.05);
        }
    }

    /// 침투 케이스: EPA 경로에서도 클리핑 4점 + 음의 separation.
    #[test]
    fn penetrating_face_contact() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &top,
            &pose(0.0, 1.9, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        assert_eq!(m.points.len(), 4);
        for p in &m.points {
            assert!((p.separation + 0.1).abs() < 1e-3, "sep={}", p.separation);
        }
    }

    /// 구-박스: witness 단일점 + radius 재팽창 separation.
    #[test]
    fn sphere_box_single_point() {
        let ball = Shape::Sphere { radius: 0.5 };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &ball,
            &pose(0.0, 1.52, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.1,
        )
        .unwrap();
        assert_eq!(m.points.len(), 1);
        assert!(
            (m.points[0].separation - 0.02).abs() < 1e-3,
            "{:?}",
            m.points[0]
        );
        assert!((m.normal.y - 1.0).abs() < 1e-3);
    }

    /// speculative: margin 밖이면 None, 안이면 양의 separation 접촉.
    #[test]
    fn speculative_margin_gate() {
        let s = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        assert!(contact(&s, &pose(0.0, 2.3, 0.0), &s, &pose(0.0, 0.0, 0.0), 0.05).is_none());
        let m = contact(&s, &pose(0.0, 2.3, 0.0), &s, &pose(0.0, 0.0, 0.0), 0.5).unwrap();
        assert!(m.points.iter().all(|p| (p.separation - 0.3).abs() < 1e-3));
    }
}
