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
use crate::tolerance::{characteristic_length, length_sq_eps};
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

/// 계보 없는 witness 접점 표식. 클리핑 접점의 feature 인코딩(상위 비트=ref_is_a, 면·계보)과
/// 겹치지 않는 예약값이며, 솔버는 이 값을 보면 **위치 근접**으로 캐시를 매칭한다.
pub const FEATURE_GENERIC: u32 = 0xFFFF_0000;

/// 클리핑 정점의 계보 (리뷰 MAJOR 반영 — 구 태그 재사용 인코딩은 다단 클리핑에서 서로 다른
/// 계보가 같은 태그로 충돌했다): incident 원본 정점 비트합 ∪ 관여한 reference 측면 비트합.
#[derive(Clone, Copy)]
struct ClipVertex {
    position: Vec3,
    incident_mask: u8,
    reference_mask: u8,
}

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
    if let (
        Shape::Box {
            half: ha,
            radius: ra_box,
        },
        Shape::Box {
            half: hb,
            radius: rb_box,
        },
    ) = (sa, sb)
    {
        let eps_l2 = length_sq_eps(characteristic_length(sa, sb));
        if let Some(m) = clip_box_box(eps_l2, *ha, *ra_box, pa, *hb, *rb_box, pb, normal, margin) {
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
            // witness 단일점: 계보가 없다(곡면·에지-에지). 상위 비트로 **generic 표식**을 달아
            // 솔버가 ID 대신 위치 근접으로 warm start를 매칭하게 한다(리뷰 MAJOR).
            feature: FeatureId(FEATURE_GENERIC),
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
/// rounded box(리뷰 MAJOR 반영): 실제 평면 패치는 접선 방향으로 radius만큼 수축되고
/// 법선 방향은 겉면(half)에 놓인다 — 겉 코너를 접점으로 쓰면 곡면 밖 점이 솔버로 간다.
fn box_face_verts(f: usize, half: Vec3, radius: f32) -> [Vec3; 4] {
    let axis = f / 2;
    let s = if f % 2 == 1 { 1.0 } else { -1.0 };
    let core = Vec3::new(
        (half.x - radius).max(0.0),
        (half.y - radius).max(0.0),
        (half.z - radius).max(0.0),
    );
    let (h, u, v) = match axis {
        0 => (
            core.x,
            Vec3::new(0.0, core.y, 0.0),
            Vec3::new(0.0, 0.0, core.z),
        ),
        1 => (
            core.y,
            Vec3::new(0.0, 0.0, core.z),
            Vec3::new(core.x, 0.0, 0.0),
        ),
        _ => (
            core.z,
            Vec3::new(core.x, 0.0, 0.0),
            Vec3::new(0.0, core.y, 0.0),
        ),
    };
    let n = match axis {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    };
    let c = n * (h + radius);
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
#[allow(clippy::too_many_arguments)]
fn clip_box_box(
    eps_l2: f32,
    ha: Vec3,
    ra: f32,
    pa: &Pose,
    hb: Vec3,
    rb: f32,
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
    let (rf, rh, rr, rp, ih, ir, ip) = if ref_is_a {
        (fa, ha, ra, pa, hb, rb, pb)
    } else {
        (fb, hb, rb, pb, ha, ra, pa)
    };

    let rn = box_face_normal(rf, rp.rot); // reference 바깥 법선 (월드)
    let rverts: Vec<Vec3> = box_face_verts(rf, rh, rr)
        .iter()
        .map(|v| rp.pos + rp.rot.rotate(*v))
        .collect();
    // incident = 상대 박스에서 rn과 가장 반대인 면
    let (inc_f, _) = best_face(ip.rot, -rn);
    let iverts: Vec<Vec3> = box_face_verts(inc_f, ih, ir)
        .iter()
        .map(|v| ip.pos + ip.rot.rotate(*v))
        .collect();
    // radius가 half를 잠식해 평면 패치가 선/점으로 퇴화하면 클리핑 불가 → witness 폴백
    if (rverts[1] - rverts[0])
        .cross(rverts[2] - rverts[1])
        .length_sq()
        <= eps_l2
        || (iverts[1] - iverts[0])
            .cross(iverts[2] - iverts[1])
            .length_sq()
            <= eps_l2
    {
        return None;
    }

    // Sutherland–Hodgman: incident 폴리곤을 reference 4 측면 평면으로 클리핑 (계보 추적)
    let mut poly: Vec<ClipVertex> = iverts
        .iter()
        .enumerate()
        .map(|(i, v)| ClipVertex {
            position: *v,
            incident_mask: 1u8 << i,
            reference_mask: 0,
        })
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
        let mut out: Vec<ClipVertex> = Vec::with_capacity(8);
        for i in 0..poly.len() {
            let v0 = poly[i];
            let v1 = poly[(i + 1) % poly.len()];
            let d0 = pn.dot(v0.position - e0);
            let d1 = pn.dot(v1.position - e0);
            if d0 >= 0.0 {
                out.push(v0);
            }
            // 교점은 **엄격한 부호 반대**일 때만 생성 (확인 라운드 MAJOR 반영 — d0==0인 경계
            // 정점은 위에서 이미 보존되므로 t=0 교점까지 만들면 같은 위치가 서로 다른 계보
            // 마스크로 중복 진입해 솔버가 동일 지점에 제약을 중복 적용한다).
            if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
                let t = d0 / (d0 - d1);
                let x = v0.position + (v1.position - v0.position) * t;
                out.push(ClipVertex {
                    position: x,
                    incident_mask: v0.incident_mask | v1.incident_mask,
                    reference_mask: v0.reference_mask | v1.reference_mask | (1u8 << side),
                });
            }
        }
        poly = out;
        if poly.is_empty() {
            return None;
        }
    }

    // reference 평면 아래(margin 이내)만 채택, separation = 평면까지 부호 거리
    let mut pts: Vec<ContactPoint> = Vec::with_capacity(8);
    for v in poly {
        let sep = rn.dot(v.position - rverts[0]);
        if sep <= margin {
            // 접점 위치는 reference 면 위로 투영한 점과 incident 점의 중간
            let proj = v.position - rn * sep;
            let feature = ((ref_is_a as u32) << 31)
                | ((rf as u32) << 28)
                | ((inc_f as u32) << 25)
                | ((v.reference_mask as u32) << 4)
                | v.incident_mask as u32;
            pts.push(ContactPoint {
                position: (v.position + proj) * 0.5,
                separation: sep,
                feature: FeatureId(feature),
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

/// 최심점 고정 → 나머지 3점 전수 조합으로 접촉 사각형 면적 최대 선택 (설계 §4.2 원문,
/// 리뷰 MAJOR 반영 — 구 그리디(최원점 선고정)는 면적 최대 조합을 보장하지 않았다).
/// pts는 클리핑 다각형의 순환 순서 — 인덱스 정렬로 부분집합의 순환 순서를 보존해 면적 계산.
fn reduce_to_4(pts: Vec<ContactPoint>, n: Vec3) -> Vec<ContactPoint> {
    use core::cmp::Ordering;
    let deepest = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.separation
                .total_cmp(&b.separation)
                .then(a.feature.cmp(&b.feature))
        })
        .map(|(i, _)| i)
        .unwrap();
    let candidates: Vec<usize> = (0..pts.len()).filter(|i| *i != deepest).collect();
    let mut best_indices = [deepest, candidates[0], candidates[1], candidates[2]];
    let mut best_area = -1.0f32;
    let mut best_key = [FeatureId(u32::MAX); 4];

    for a in 0..(candidates.len() - 2) {
        for b in (a + 1)..(candidates.len() - 1) {
            for c in (b + 1)..candidates.len() {
                let mut indices = [deepest, candidates[a], candidates[b], candidates[c]];
                indices.sort_unstable(); // 순환 순서 보존
                let origin = pts[indices[0]].position;
                let twice_area = (pts[indices[1]].position - origin)
                    .cross(pts[indices[2]].position - origin)
                    .dot(n)
                    + (pts[indices[2]].position - origin)
                        .cross(pts[indices[3]].position - origin)
                        .dot(n);
                let area = twice_area.abs();
                let mut key = [
                    pts[indices[0]].feature,
                    pts[indices[1]].feature,
                    pts[indices[2]].feature,
                    pts[indices[3]].feature,
                ];
                key.sort_unstable();
                let better = match area.total_cmp(&best_area) {
                    Ordering::Greater => true,
                    Ordering::Equal => key < best_key,
                    Ordering::Less => false,
                };
                if better {
                    best_area = area;
                    best_indices = indices;
                    best_key = key;
                }
            }
        }
    }

    let mut out: Vec<ContactPoint> = best_indices.iter().map(|&i| pts[i]).collect();
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
        feats.sort_unstable();
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

    /// 확인 라운드 MAJOR 회귀: 경계 정점이 클리핑 평면 위에 정확히 놓여도 중복 접점 금지.
    #[test]
    fn boundary_vertex_no_duplicate_points() {
        // incident 꼭짓점이 reference 측면 평면(x=1) 위에 정확히 놓이는 45° 구성
        let ground = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        };
        let top = Shape::Box {
            half: Vec3::new(
                core::f32::consts::FRAC_1_SQRT_2,
                1.0,
                core::f32::consts::FRAC_1_SQRT_2,
            ),
            radius: 0.0,
        };
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_4);
        let pa = Pose {
            pos: Vec3::new(1.0, 2.0, 0.0),
            rot,
        };
        let m = contact(&top, &pa, &ground, &pose(0.0, 0.0, 0.0), 0.05).unwrap();
        assert!(!m.points.is_empty() && m.points.len() <= 4);
        for (i, p) in m.points.iter().enumerate() {
            for q in m.points.iter().skip(i + 1) {
                let d = (p.position - q.position).length_sq();
                assert!(d > 1.0e-8, "중복 접점: {p:?} vs {q:?}");
            }
        }
    }

    /// 리뷰 요구: **metamorphic 스케일 테스트**. 같은 씬을 1e-3~1e2배로 변환해도
    /// 접촉 판정·법선·상대 침투가 유지되어야 한다(절대 임계가 남아 있으면 여기서 깨진다).
    #[test]
    fn contact_is_scale_invariant() {
        for &k in &[1.0e-3f32, 1.0e-1, 1.0, 1.0e1, 1.0e2] {
            let top = Shape::Box {
                half: Vec3::new(k, k, k),
                radius: 0.0,
            };
            let ground = Shape::Box {
                half: Vec3::new(4.0 * k, k, 4.0 * k),
                radius: 0.0,
            };
            // 상대 침투 1% 로 동일하게 구성
            let m = contact(
                &top,
                &pose(0.0, 1.99 * k, 0.0),
                &ground,
                &pose(0.0, 0.0, 0.0),
                0.05 * k,
            )
            .unwrap_or_else(|| panic!("scale {k}: 접촉 소실"));
            assert_eq!(
                m.points.len(),
                4,
                "scale {k}: 매니폴드 점수 {}",
                m.points.len()
            );
            assert!(
                (m.normal.y - 1.0).abs() < 1.0e-3,
                "scale {k}: 법선 {:?}",
                m.normal
            );
            for p in &m.points {
                let rel = p.separation / k;
                assert!(
                    (rel + 0.01).abs() < 0.05,
                    "scale {k}: 상대 침투 {rel} (절대 {})",
                    p.separation
                );
            }
        }
    }

    /// 리뷰 MAJOR 회귀: rounded box의 접점은 평면 패치(접선 ±(half−radius)) 안에 있어야 한다.
    #[test]
    fn rounded_box_face_points_stay_on_flat_patch() {
        let top = Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.25,
        };
        let ground = Shape::Box {
            half: Vec3::new(4.0, 1.0, 4.0),
            radius: 0.0,
        };
        let m = contact(
            &top,
            &pose(0.0, 2.0, 0.0),
            &ground,
            &pose(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        assert_eq!(m.points.len(), 4);
        for p in &m.points {
            assert!(p.position.x.abs() <= 0.7501, "point={p:?}");
            assert!(p.position.z.abs() <= 0.7501, "point={p:?}");
            assert!(p.separation.abs() < 1.0e-4, "point={p:?}");
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
