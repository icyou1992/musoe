//! 변수 정점 볼록 면 클리퍼 — 일반 볼록(hull·원기둥) 면 접촉용.
//!
//! `clip.rs`의 클리퍼는 **박스 전용**이다: 참조면·입사면 모두 정점 4개 고정(`[Vec3; 4]`),
//! 측면 루프 `0..4`, 계보를 4비트 마스크 둘로 담는다. hull 면은 정점이 최대 32개라
//! 그 어느 것도 성립하지 않는다. 여기서는 정점 수를 변수로 두고, 계보를 인덱스로 담는다
//! (`feature.rs` 코덱).
//!
//! **박스 경로는 건드리지 않는다** — 박스-박스 접점의 비트 동일성이 절대 조건이라
//! 새 경로를 따로 둔다(같은 바디 페어가 두 경로를 동시에 타지 않는다: 디스패치가
//! 셰이프 쌍으로 갈린다).
//!
//! ## 계보 추적이 정점이 아니라 **에지**에 붙는 이유
//! Sutherland–Hodgman은 다단이다. 2번째 측면을 자를 때 만나는 정점 v0·v1은 이미
//! 1번째 측면이 만든 교점일 수 있다. 그래서 "v0과 v1의 계보를 합친다"(박스 경로의
//! 비트마스크 합집합)로는 무엇을 잘랐는지 알 수 없다. 대신 **폴리곤의 각 에지가
//! 어디서 왔는지**(입사면의 원본 에지 j, 또는 참조면 측면 t)를 들고 다닌다. 볼록
//! 폴리곤을 반평면으로 자르면 이 불변식이 유지된다 — 새로 생기는 에지는 자른 평면
//! 자신뿐이다. 교점의 계보는 곧 **잘린 에지의 출처**로 결정된다:
//!
//! - 입사 에지 j를 측면 s로 자름 → `EdgeCross { a: j, b: s }`
//! - 측면 t를 측면 s로 자름 → 두 측면이 만나는 **참조면 정점** → `ReferenceVertex { a: v }`
//!
//! 뒤엣것은 t·s가 인접할 때만 참조면 정점이다. 부동소수 퇴화로 비인접 측면이 만나면
//! 계보를 특정할 수 없으므로 **모른다고 표시한다**(`UNKNOWN_INDEX`) — 코덱이 범위 밖으로
//! 거절하고 호출자가 `FEATURE_GENERIC`으로 떨어뜨린다. 지어내면 서로 다른 접점이 같은
//! ID를 받아 warm start가 엉뚱한 곳에 붙는다.

use super::feature::ClipKind;
use maru_math::Vec3;

/// 면 하나의 정점 상한. `MAX_HULL_POINTS`와 같다 — hull 면 링이 그보다 길 수 없다.
pub(super) const MAX_FACE_VERTS: usize = crate::MAX_HULL_POINTS;

/// 클리핑 결과 상한. 볼록 m각형을 반평면 k개로 자르면 정점은 최대 m + k다
/// (측면마다 최대 +1). 힙을 쓰지 않으므로 상한이 곧 스택 배열 크기다.
pub(super) const GEN_CLIP_CAP: usize = 2 * MAX_FACE_VERTS;

/// 계보를 특정할 수 없다는 표시. 코덱 인덱스 상한(31)을 넘으므로 `pack()`이 거절한다 —
/// 조용히 0번으로 지어내면 서로 다른 접점이 같은 ID를 받는다.
pub(super) const UNKNOWN_INDEX: u8 = u8::MAX;

/// 폴리곤 에지의 출처. `Incident(j)` = 입사면 원본 에지 j, `RefSide(t)` = 참조면 측면 t.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeSrc {
    Incident(u8),
    RefSide(u8),
}

/// 클리핑된 폴리곤의 정점 하나.
#[derive(Clone, Copy, Debug)]
pub(super) struct GenVertex {
    pub position: Vec3,
    pub kind: ClipKind,
    /// 계보 첫째 인덱스. `UNKNOWN_INDEX`면 특정 실패(→ GENERIC 폴백).
    pub a: u8,
    /// 계보 둘째 인덱스. `EdgeCross`에서만 의미가 있고 나머지는 0.
    pub b: u8,
    /// 이 정점에서 **시작하는** 에지의 출처 (다음 단계 클리핑이 소비한다).
    edge: EdgeSrc,
}

/// 클리핑 결과 버퍼. 반환값으로 넘기면 1.5KiB를 복사하므로 out-param으로 채운다.
pub(super) struct GenClip {
    verts: [GenVertex; GEN_CLIP_CAP],
    len: usize,
}

impl GenClip {
    pub(super) fn new() -> Self {
        Self {
            verts: [GenVertex {
                position: Vec3::ZERO,
                kind: ClipKind::IncidentVertex,
                a: 0,
                b: 0,
                edge: EdgeSrc::Incident(0),
            }; GEN_CLIP_CAP],
            len: 0,
        }
    }
    pub(super) fn as_slice(&self) -> &[GenVertex] {
        &self.verts[..self.len]
    }
}

/// 측면 t와 측면 s가 공유하는 참조면 정점. 측면 s는 `rverts[s] -> rverts[(s+1)%k]`이므로
/// 측면 t와 t+1이 정점 t+1을 공유한다. 인접하지 않으면 `None`.
fn shared_ref_vertex(t: usize, s: usize, k: usize) -> Option<usize> {
    if (t + 1) % k == s {
        Some(s)
    } else if (s + 1) % k == t {
        Some(t)
    } else {
        None
    }
}

/// 입사면 폴리곤을 참조면의 측면 평면들로 클리핑한다.
///
/// - `rverts`: 참조면 정점(월드, 순환 순서). `rn`: 참조면 바깥 법선.
/// - `iverts`: 입사면 정점(월드, 순환 순서).
/// - 성공하면 `out`에 계보가 붙은 폴리곤을 채우고 `true`. 폴리곤이 비거나 입력이
///   퇴화(정점 3 미만 / 상한 초과)면 `false`(→ 호출자가 witness 단일점으로 폴백).
///
/// 힙을 쓰지 않는다. 감기 방향을 가정하지 않는다 — 측면 평면의 안쪽 방향을 참조면
/// 중심으로 맞춘다(박스 경로와 같은 규칙).
pub(super) fn clip_convex_face(
    rverts: &[Vec3],
    rn: Vec3,
    iverts: &[Vec3],
    out: &mut GenClip,
) -> bool {
    let k = rverts.len();
    let m = iverts.len();
    if !(3..=MAX_FACE_VERTS).contains(&k) || !(3..=MAX_FACE_VERTS).contains(&m) {
        return false;
    }

    let mut poly = [GenVertex {
        position: Vec3::ZERO,
        kind: ClipKind::IncidentVertex,
        a: 0,
        b: 0,
        edge: EdgeSrc::Incident(0),
    }; GEN_CLIP_CAP];
    for (i, v) in iverts.iter().enumerate() {
        poly[i] = GenVertex {
            position: *v,
            kind: ClipKind::IncidentVertex,
            a: i as u8,
            b: 0,
            // 정점 i에서 시작하는 에지 = 입사면 원본 에지 i (i -> i+1)
            edge: EdgeSrc::Incident(i as u8),
        };
    }
    let mut poly_len = m;

    // 박스 경로와 같이 두 버퍼의 **역할만 바꾼다** — 측면마다 배열을 통째로 복사하지 않는다.
    let mut buf = poly;
    let mut flip = false;

    // 참조면 중심은 측면 전체에 대해 같은 값이라 루프 밖에서 한 번만 만든다.
    let mut center = Vec3::ZERO;
    for v in rverts {
        center += *v;
    }
    center = center * (1.0 / k as f32);

    for side in 0..k {
        let e0 = rverts[side];
        let e1 = rverts[(side + 1) % k];
        let mut pn = rn.cross(e1 - e0);
        if pn.dot(center - e0) < 0.0 {
            pn = -pn;
        }
        let (src, dst) = if flip {
            (&buf, &mut poly)
        } else {
            (&poly, &mut buf)
        };
        let mut out_len = 0usize;
        for i in 0..poly_len {
            let v0 = src[i];
            let v1 = src[(i + 1) % poly_len];
            let d0 = pn.dot(v0.position - e0);
            let d1 = pn.dot(v1.position - e0);
            if d0 >= 0.0 && out_len < GEN_CLIP_CAP {
                // v0는 계보도 출발 에지도 그대로다. 에지 v0->x는 여전히 v0.edge의 일부다.
                dst[out_len] = v0;
                out_len += 1;
            }
            // **엄격한 부호 반대**일 때만 교점을 만든다 — d0 == 0인 경계 정점은 위에서
            // 이미 보존됐고, t=0 교점까지 만들면 같은 자리가 서로 다른 계보로 중복 진입한다
            // (박스 경로가 같은 이유로 같은 규칙을 쓴다).
            if ((d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0)) && out_len < GEN_CLIP_CAP {
                let t = d0 / (d0 - d1);
                let x = v0.position + (v1.position - v0.position) * t;
                // 계보는 **잘린 에지**(v0에서 출발하는 에지)의 출처가 결정한다.
                let (kind, a, b) = match v0.edge {
                    EdgeSrc::Incident(j) => (ClipKind::EdgeCross, j, side as u8),
                    EdgeSrc::RefSide(t_side) => match shared_ref_vertex(t_side as usize, side, k) {
                        Some(v) => (ClipKind::ReferenceVertex, v as u8, 0),
                        // 비인접 측면끼리 만났다 = 볼록 가정이 부동소수로 깨졌다.
                        // 지어내지 않고 모른다고 표시한다.
                        None => (ClipKind::ReferenceVertex, UNKNOWN_INDEX, 0),
                    },
                };
                // 나가는 교차(d0>0)면 x 다음은 이 측면을 따라 간다 → 출발 에지 = 측면 s.
                // 들어오는 교차(d0<0)면 x 다음은 v1이다 → 출발 에지 = 잘린 에지 그대로.
                let edge = if d0 > 0.0 {
                    EdgeSrc::RefSide(side as u8)
                } else {
                    v0.edge
                };
                dst[out_len] = GenVertex {
                    position: x,
                    kind,
                    a,
                    b,
                    edge,
                };
                out_len += 1;
            }
        }
        flip = !flip;
        poly_len = out_len;
        if poly_len == 0 {
            return false;
        }
    }

    let poly = if flip { &buf } else { &poly };
    out.verts[..poly_len].copy_from_slice(&poly[..poly_len]);
    out.len = poly_len;
    true
}

/// 셰이프의 `dir`(월드) 방향 지지면을 월드 좌표 정점 배열로 채운다.
///
/// 성공하면 `(정점 수, 면 코드, 면 바깥 법선(월드))`. 면을 낼 수 없으면 `None`:
/// 구·캡슐처럼 평면 패치가 없는 셰이프, 퇴화 hull(면 0개), 그리고 **둥근 hull**
/// (`radius > 0`)이 그렇다.
///
/// 둥근 hull을 제외하는 이유: 실제 접촉 패치는 면을 접선 방향으로 radius만큼 **안쪽으로
/// 오프셋한** 다각형이다. 박스는 `box_face_verts`가 축 정렬이라 성분별 축소로 정확히
/// 해내지만, 일반 다각형의 내향 오프셋은 에지 직선을 각각 밀어 다시 교차시켜야 하고
/// 폭이 2*radius보다 좁으면 퇴화한다. 근사(중심 방향 축소)는 비정다각형에서 틀린다.
/// 그래서 지금은 둥근 hull을 기존 witness 단일점 경로에 그대로 둔다(회귀 없음).
/// 알려진 한계이며 README '매니폴드 생성'에 적어 두었다.
fn support_face(
    shape: &crate::Shape,
    pose: &crate::Pose,
    m: &maru_math::Mat3,
    dir: Vec3,
    out: &mut [Vec3; MAX_FACE_VERTS],
) -> Option<(usize, u32, Vec3)> {
    match shape {
        crate::Shape::Box { half, radius } => {
            // 박스는 기존 헬퍼를 그대로 쓴다 — 면 인덱스 0..5, 정점 4개.
            let (f, _) = super::clip::best_face(m, dir);
            let vs = super::clip::box_face_verts(f, *half, *radius);
            for (i, v) in vs.iter().enumerate() {
                out[i] = pose.pos + m.mul_vec(*v);
            }
            Some((4, f as u32, super::clip::box_face_normal(f, m)))
        }
        crate::Shape::Hull { data, radius } => {
            if *radius > 0.0 {
                return None;
            }
            // 면 법선은 로컬이다 — 방향을 로컬로 돌리는 편이 면마다 법선을 돌리는 것보다 싸다.
            let local_dir = m.transpose().mul_vec(dir);
            let (f, _) = data.faces().best_face(local_dir)?;
            let face = &data.faces().faces()[f];
            let ring = &face.verts;
            if !(3..=MAX_FACE_VERTS).contains(&ring.len()) {
                return None;
            }
            let pts = data.points();
            for (i, vi) in ring.iter().enumerate() {
                out[i] = pose.pos + m.mul_vec(pts[*vi as usize]);
            }
            Some((ring.len(), f as u32, m.mul_vec(face.normal)))
        }
        _ => None,
    }
}

/// 일반 볼록 면 접촉 매니폴드. 실패하면 `None`(호출자가 witness 단일점으로 폴백).
///
/// 박스-박스는 **여기 오지 않는다** — 전용 경로가 비트 동일을 보장해야 하므로 호출자가
/// 먼저 걸러 낸다. 여기서도 한 번 더 막아 둔다(호출 순서가 바뀌어도 계약이 유지되게).
pub(super) fn convex_face_manifold(
    sa: &crate::Shape,
    pa: &crate::Pose,
    sb: &crate::Shape,
    pb: &crate::Pose,
    normal: Vec3,
    margin: f32,
) -> Option<super::Manifold> {
    use maru_math::Mat3;
    if matches!(sa, crate::Shape::Box { .. }) && matches!(sb, crate::Shape::Box { .. }) {
        return None;
    }
    let ma = Mat3::from_quat(pa.rot);
    let mb = Mat3::from_quat(pb.rot);
    let mut va = [Vec3::ZERO; MAX_FACE_VERTS];
    let mut vb = [Vec3::ZERO; MAX_FACE_VERTS];
    // A에서 B를 향하는 면(법선 ≈ −normal), B에서 A를 향하는 면(≈ +normal)
    let (na, fa, nrm_a) = support_face(sa, pa, &ma, -normal, &mut va)?;
    let (nb, fb, nrm_b) = support_face(sb, pb, &mb, normal, &mut vb)?;
    let da = nrm_a.dot(-normal);
    let db = nrm_b.dot(normal);
    if da.max(db) < super::FACE_DOT_MIN {
        return None; // 에지-에지 지배 — witness 단일점 경로
    }
    // reference = 더 정렬된 쪽 (동률 = A — 결정론)
    let ref_is_a = da >= db;
    let (rverts, rn, ref_face, iverts, inc_face) = if ref_is_a {
        (&va[..na], nrm_a, fa, &vb[..nb], fb)
    } else {
        (&vb[..nb], nrm_b, fb, &va[..na], fa)
    };

    let mut clipped = GenClip::new();
    if !clip_convex_face(rverts, rn, iverts, &mut clipped) {
        return None;
    }

    // reference 평면 아래(margin 이내)만 채택. 같은 자리 중복은 접촉 국소 척도로 친다
    // (박스 경로와 같은 이유·같은 척도 — AABB 대각을 쓰면 고종횡비에서 패치가 붕괴한다).
    let dup_eps_sq = crate::tolerance::length_sq_eps(crate::tolerance::min_feature_size(sa, sb));
    let mut pts = [super::ContactPoint::default(); GEN_CLIP_CAP];
    let mut pts_len = 0usize;
    for v in clipped.as_slice() {
        let sep = rn.dot(v.position - rverts[0]);
        if sep > margin {
            continue;
        }
        let proj = v.position - rn * sep;
        let position = (v.position + proj) * 0.5;
        if pts[..pts_len]
            .iter()
            .any(|q| (q.position - position).length_sq() <= dup_eps_sq)
        {
            continue;
        }
        // 계보를 특정 못 했으면(퇴화) 지어내지 않고 generic으로 떨어뜨린다 —
        // 솔버가 ID 대신 위치 근접으로 warm start를 매칭한다.
        let feature = super::feature::pack(super::feature::FaceFeature {
            ref_is_a,
            ref_face,
            inc_face,
            kind: v.kind,
            a: v.a as u32,
            b: v.b as u32,
        })
        .unwrap_or(super::FeatureId(super::FEATURE_GENERIC));
        pts[pts_len] = super::ContactPoint {
            position,
            separation: sep,
            feature,
        };
        pts_len += 1;
    }
    if pts_len == 0 {
        return None;
    }

    let reduced;
    let final_pts: &[super::ContactPoint] = if pts_len > 4 {
        reduced = reduce_to_4_general(&pts[..pts_len], rn);
        &reduced
    } else {
        &pts[..pts_len]
    };
    let m_normal = if ref_is_a { -rn } else { rn };
    Some(super::Manifold::from_points(m_normal, final_pts))
}

/// 접점 다수를 4개로 줄인다 — 박스 경로와 **같은 규칙**(최심점 고정 + 면적 최대 조합).
///
/// 후보가 많으면 그 규칙이 O(n³)이라 정점 64개에서 조합이 39,711개가 된다. 그래서 8개를
/// 넘으면 **순환 순서에서 균등 간격으로 8개를 먼저 고르고**(최심점은 반드시 포함) 같은
/// 규칙을 돌린다. 표본이 순환 순서를 보존하므로 면적 계산의 전제가 유지되고, 균등 간격이라
/// 퍼짐도 지켜진다. 탐욕 선택으로 바꾸지 않는 이유: 탐욕은 면적 최대를 보장하지 못해
/// 박스 경로에서 이미 한 번 리뷰로 기각된 규칙이다.
pub(super) fn reduce_to_4_general(
    pts: &[super::ContactPoint],
    n: Vec3,
) -> [super::ContactPoint; 4] {
    const SAMPLE: usize = 8;
    if pts.len() <= SAMPLE {
        return super::clip::reduce_to_4(pts, n);
    }
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
    let mut sel = [0usize; SAMPLE];
    for (j, s) in sel.iter_mut().enumerate() {
        *s = (deepest + (j * pts.len()) / SAMPLE) % pts.len();
    }
    sel.sort_unstable();
    debug_assert!(
        sel.windows(2).all(|w| w[0] != w[1]),
        "균등 간격 표본이 겹쳤다"
    );
    let sample: [super::ContactPoint; SAMPLE] = core::array::from_fn(|j| pts[sel[j]]);
    super::clip::reduce_to_4(&sample, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::feature::{pack, FaceFeature};
    use std::collections::HashSet;

    fn square(cx: f32, cy: f32, h: f32, z: f32) -> Vec<Vec3> {
        vec![
            Vec3::new(cx - h, cy - h, z),
            Vec3::new(cx + h, cy - h, z),
            Vec3::new(cx + h, cy + h, z),
            Vec3::new(cx - h, cy + h, z),
        ]
    }

    /// **엄밀 볼록** n각형(n은 짝수). 삼각함수를 쓰지 않는다(결정론 정책 L1: 상태에
    /// 닿는 경로에서 libm 금지 — 테스트 픽스처도 같은 규율로 만든다).
    ///
    /// 포물선 두 개로 렌즈를 만든다: 아래 사슬 `y = x²`, 위 사슬 `y = 2 − x²`, x ∈ [−1,1].
    /// 포물선 위 세 점은 절대 일직선이 아니므로 **좌표가 딱 떨어지지 않는다** — 정다각형
    /// 근사를 정사각형 둘레 균등분할로 만들면 변마다 점이 일직선이 되어 측면 평면이
    /// 겹치고, 클리퍼가 "비인접 측면끼리 만나는" 퇴화에 빠진다(실측으로 겪었다 —
    /// `collinear_reference_edges_mark_lineage_unknown_instead_of_guessing`가 그쪽을 잠근다).
    fn convex_lens(n: usize, r: f32, z: f32) -> Vec<Vec3> {
        assert!(n >= 4 && n.is_multiple_of(2));
        let h = n / 2;
        let x_at = |j: usize| -1.0 + 2.0 * (j as f32) / (h as f32);
        let mut out: Vec<Vec3> = Vec::with_capacity(n);
        for j in 0..=h {
            let x = x_at(j);
            out.push(Vec3::new(x * r, (x * x - 1.0) * r, z));
        }
        for j in (1..h).rev() {
            let x = x_at(j);
            out.push(Vec3::new(x * r, (1.0 - x * x) * r, z));
        }
        assert_eq!(out.len(), n);
        // 픽스처가 진짜 엄밀 볼록인지 스스로 판정한다 — 아니면 시험이 겨누는 것을 못 겨눈다.
        for i in 0..n {
            let a = out[i];
            let b = out[(i + 1) % n];
            let c = out[(i + 2) % n];
            let cross = (b - a).cross(c - b).z;
            assert!(cross > 1e-6, "{i}번 코너가 일직선이거나 오목하다: {cross}");
        }
        out
    }

    fn lineage_ids(c: &GenClip) -> Vec<Option<u32>> {
        c.as_slice()
            .iter()
            .map(|v| {
                pack(FaceFeature {
                    ref_is_a: true,
                    ref_face: 1,
                    inc_face: 2,
                    kind: v.kind,
                    a: v.a as u32,
                    b: v.b as u32,
                })
                .map(|f| f.0)
            })
            .collect()
    }

    /// 입사면이 참조면 안에 완전히 들면 그대로 살아남는다 — 정점 수·계보가 원본이다.
    #[test]
    fn fully_contained_incident_face_survives_unclipped() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let i = square(0.0, 0.0, 0.4, 0.1);
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        assert_eq!(out.as_slice().len(), 4);
        for (n, v) in out.as_slice().iter().enumerate() {
            assert_eq!(
                v.kind,
                ClipKind::IncidentVertex,
                "{n}번 정점 계보가 바뀌었다"
            );
            assert_eq!(v.a as usize, n);
        }
    }

    /// 어긋난 사각형끼리는 겹친 사각형이 남는다: 원본 정점 1 + 교점 2 + 참조면 정점 1.
    #[test]
    fn offset_squares_clip_to_the_overlap_with_mixed_lineage() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let i = square(0.6, 0.6, 1.0, 0.1); // 우상단으로 밀린 같은 크기 사각형
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        assert_eq!(pts.len(), 4, "겹침은 사각형이어야 한다: {pts:?}");
        let kinds: HashSet<_> = pts.iter().map(|v| v.kind).collect();
        assert!(
            kinds.contains(&ClipKind::IncidentVertex),
            "입사면 원본 정점이 하나 남아야 한다"
        );
        assert!(
            kinds.contains(&ClipKind::EdgeCross),
            "입사 에지 × 참조 측면 교점이 있어야 한다"
        );
        assert!(
            kinds.contains(&ClipKind::ReferenceVertex),
            "참조면 코너가 겹침 안에 들어야 한다"
        );
        // 참조면 [-1,1]² ∩ 입사면 [-0.4,1.6]² = [-0.4,1]² 사각형이다.
        for v in pts {
            assert!(
                v.position.x >= -0.4 - 1e-5 && v.position.x <= 1.0 + 1e-5,
                "{v:?}"
            );
            assert!(
                v.position.y >= -0.4 - 1e-5 && v.position.y <= 1.0 + 1e-5,
                "{v:?}"
            );
        }
    }

    /// 참조면 코너의 계보는 **정확히 그 코너의 정점 인덱스**여야 한다.
    /// (0.4,0.4)가 겹침의 코너이고 그것은 참조면 정점 2번 (1,1)이 아니라... 를 좌표로 판정.
    #[test]
    fn reference_corner_lineage_names_the_actual_corner() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0); // 정점 0=(-1,-1) 1=(1,-1) 2=(1,1) 3=(-1,1)
        let i = square(0.0, 0.0, 2.0, 0.1); // 참조면을 완전히 덮는다 → 참조면 코너 4개가 남는다
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        assert_eq!(pts.len(), 4, "참조면 사각형이 그대로 남아야 한다: {pts:?}");
        for v in pts {
            assert_eq!(v.kind, ClipKind::ReferenceVertex, "{v:?}");
            assert_ne!(v.a, UNKNOWN_INDEX, "코너 계보를 특정 못 했다: {v:?}");
            // 클리핑 결과는 **입사면 평면 위**에 있으므로 법선 방향 성분은 다르다.
            // 계보 판정은 면 안(xy) 좌표로 한다.
            let named = r[v.a as usize];
            let d = named - v.position;
            assert!(
                (d.x * d.x + d.y * d.y).sqrt() < 1e-5,
                "계보가 가리키는 참조면 정점 {named:?}가 실제 위치 {:?}와 다르다",
                v.position
            );
        }
    }

    /// 계보는 접점마다 달라야 한다 — 같으면 warm start가 엉뚱한 접점에 붙는다.
    #[test]
    fn every_clipped_vertex_gets_a_distinct_feature_id() {
        let mut out = GenClip::new();
        for (ox, oy) in [(0.0, 0.0), (0.6, 0.6), (0.9, 0.1), (0.5, 0.0)] {
            let r = square(0.0, 0.0, 1.0, 0.0);
            let i = square(ox, oy, 1.0, 0.1);
            assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
            let ids = lineage_ids(&out);
            assert!(
                ids.iter().all(|x| x.is_some()),
                "계보 특정 실패: ({ox},{oy}) {:?}",
                out.as_slice()
            );
            let set: HashSet<_> = ids.iter().collect();
            assert_eq!(
                set.len(),
                ids.len(),
                "({ox},{oy})에서 접점 둘이 같은 feature ID를 받았다: {:?}",
                out.as_slice()
            );
        }
    }

    /// 많은 정점 면끼리도 상한을 넘지 않고 계보가 단사다.
    #[test]
    fn many_sided_faces_stay_within_the_cap_and_stay_injective() {
        let mut out = GenClip::new();
        let r = convex_lens(MAX_FACE_VERTS, 1.0, 0.0);
        let i = convex_lens(MAX_FACE_VERTS, 1.0, 0.1);
        // 살짝 어긋나게 밀어 교점이 생기게 한다
        let i: Vec<Vec3> = i
            .iter()
            .map(|v| Vec3::new(v.x + 0.37, v.y + 0.11, v.z))
            .collect();
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        assert!(
            pts.len() <= GEN_CLIP_CAP,
            "상한 {GEN_CLIP_CAP}을 넘었다: {}",
            pts.len()
        );
        let ids = lineage_ids(&out);
        let set: HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), ids.len(), "계보가 충돌했다");
        assert!(ids.iter().all(|x| x.is_some()), "계보 특정 실패");
    }

    /// **일직선 점이 섞인 참조면**은 측면 평면이 겹쳐 비인접 측면끼리 만난다. 그럴 때
    /// 계보를 지어내지 않고 모른다고 표시해야 한다 — 지어내면 서로 다른 접점이 같은 ID를
    /// 받아 warm start가 엉뚱한 곳에 붙는다. 실제 hull 면 링은 monotone chain이 일직선
    /// 점을 걸러 주지만, 부동소수 퇴화로 여기 도달할 수 있다.
    #[test]
    fn collinear_reference_edges_mark_lineage_unknown_instead_of_guessing() {
        // 정사각형 각 변을 4등분한 16각형 — 변마다 점 5개가 일직선이다.
        let mut r: Vec<Vec3> = Vec::new();
        for side in 0..4 {
            for j in 0..4 {
                let t = j as f32 / 4.0;
                let (u, v) = match side {
                    0 => (-1.0 + 2.0 * t, -1.0),
                    1 => (1.0, -1.0 + 2.0 * t),
                    2 => (1.0 - 2.0 * t, 1.0),
                    _ => (-1.0, 1.0 - 2.0 * t),
                };
                r.push(Vec3::new(u, v, 0.0));
            }
        }
        let mut out = GenClip::new();
        let i = square(0.0, 0.0, 2.0, 0.1); // 참조면을 완전히 덮는다
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let unknown = out
            .as_slice()
            .iter()
            .filter(|v| v.a == UNKNOWN_INDEX)
            .count();
        assert!(
            unknown > 0,
            "퇴화 참조면인데 계보를 전부 특정했다 — 지어낸 것 아닌가: {:?}",
            out.as_slice()
        );
        // 모른다고 표시한 정점은 코덱이 거절해 호출자가 GENERIC으로 떨어뜨린다.
        for v in out.as_slice().iter().filter(|v| v.a == UNKNOWN_INDEX) {
            assert_eq!(
                pack(FaceFeature {
                    ref_is_a: true,
                    ref_face: 1,
                    inc_face: 2,
                    kind: v.kind,
                    a: v.a as u32,
                    b: v.b as u32,
                }),
                None,
                "미특정 계보가 코덱을 통과했다: {v:?}"
            );
        }
    }

    /// **정점이 측면 평면 위에 정확히 놓인** 배치에서 같은 자리에 점이 둘 생기면 안 된다.
    /// 솔버는 접점마다 제약을 걸므로 겹친 점은 그 지점에 제약을 두 번 거는 것과 같다.
    /// 계보는 서로 달라서(원본 정점 vs t=0 교점) 단사성 시험으로는 안 잡힌다 —
    /// **위치**로 판정해야 한다(뮤테이션 실측: 부호 규칙을 `>=`로 풀어도 단사성은 통과).
    #[test]
    fn a_vertex_exactly_on_a_side_plane_does_not_spawn_a_duplicate_point() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        // x = 1(참조면 오른쪽 측면) **위에** 정점 둘이 정확히 놓인 삼각형.
        let i = vec![
            Vec3::new(1.0, -0.5, 0.1),
            Vec3::new(2.0, 0.0, 0.1),
            Vec3::new(1.0, 0.5, 0.1),
        ];
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let pts = out.as_slice();
        for (n, v) in pts.iter().enumerate() {
            for w in &pts[n + 1..] {
                assert!(
                    (w.position - v.position).length() > 1e-6,
                    "같은 자리에 접점이 둘이다: {v:?} / {w:?}"
                );
            }
        }
    }

    /// **입사 에지 하나가 참조면 측면 둘을 가로지르는** 배치. 볼록 영역을 지나는 직선은
    /// 반드시 한 측면으로 들어와 다른 측면으로 나가므로 교점 둘이 같은 에지에서 나온다 —
    /// 계보에 측면 인덱스가 없으면 둘이 같은 ID를 받는다(뮤테이션 실측: `b`를 0으로
    /// 고정해도 다른 시험은 전부 통과했다).
    #[test]
    fn one_incident_edge_crossing_two_reference_sides_gets_two_ids() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        // 에지 (2,-2) -> (-2,0.5)는 아래 측면(y=-1)을 (0.4,-1)에서 통과해
        // 왼쪽 측면(x=-1)을 (-1,-0.125)에서 빠져나간다. 좌표가 딱 떨어지지 않는다.
        let i = vec![
            Vec3::new(2.0, -2.0, 0.1),
            Vec3::new(-2.0, 0.5, 0.1),
            Vec3::new(3.0, 3.0, 0.1),
        ];
        assert!(clip_convex_face(&r, Vec3::new(0.0, 0.0, 1.0), &i, &mut out));
        let crossings: Vec<_> = out
            .as_slice()
            .iter()
            .filter(|v| v.kind == ClipKind::EdgeCross && v.a == 0)
            .collect();
        assert!(
            crossings.len() >= 2,
            "입사 에지 0이 측면 둘을 가로질러야 한다: {:?}",
            out.as_slice()
        );
        let sides: HashSet<u8> = crossings.iter().map(|v| v.b).collect();
        assert_eq!(
            sides.len(),
            crossings.len(),
            "같은 에지의 교점들이 같은 측면 인덱스를 받았다: {crossings:?}"
        );
        let ids = lineage_ids(&out);
        let set: HashSet<_> = ids.iter().collect();
        assert_eq!(
            set.len(),
            ids.len(),
            "계보가 충돌했다: {:?}",
            out.as_slice()
        );
    }

    /// 분리된 면끼리는 폴리곤이 비고 `false` — 호출자가 witness로 폴백한다.
    #[test]
    fn disjoint_faces_produce_no_polygon() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let i = square(5.0, 5.0, 0.5, 0.1);
        assert!(!clip_convex_face(
            &r,
            Vec3::new(0.0, 0.0, 1.0),
            &i,
            &mut out
        ));
    }

    /// 퇴화 입력(정점 2개 / 상한 초과)은 거절한다. 경계 바로 안쪽은 통과해야 한다.
    #[test]
    fn degenerate_vertex_counts_are_rejected_at_the_boundary() {
        let mut out = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let two = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.1)];
        assert!(
            !clip_convex_face(&r, n, &two, &mut out),
            "정점 2개가 통과했다"
        );
        assert!(
            !clip_convex_face(&two, n, &r, &mut out),
            "참조면 2개가 통과했다"
        );
        let over = convex_lens(MAX_FACE_VERTS + 2, 1.0, 0.1);
        assert!(
            !clip_convex_face(&r, n, &over, &mut out),
            "상한 초과가 통과했다"
        );
        let at_cap = convex_lens(MAX_FACE_VERTS, 0.5, 0.1);
        assert!(
            clip_convex_face(&r, n, &at_cap, &mut out),
            "경계값(상한 정확히)이 거절됐다"
        );
    }

    /// 참조면 감기 방향이 반대여도 같은 겹침이 나온다 — 측면 안쪽 방향을 중심으로 맞추므로.
    #[test]
    fn reversed_reference_winding_gives_the_same_overlap() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let i = square(0.6, 0.6, 1.0, 0.1);
        let mut ccw = GenClip::new();
        let r = square(0.0, 0.0, 1.0, 0.0);
        assert!(clip_convex_face(&r, n, &i, &mut ccw));
        let mut cw = GenClip::new();
        let mut rr = r.clone();
        rr.reverse();
        assert!(clip_convex_face(&rr, n, &i, &mut cw));
        assert_eq!(ccw.as_slice().len(), cw.as_slice().len());
        for v in cw.as_slice() {
            assert!(
                ccw.as_slice()
                    .iter()
                    .any(|w| (w.position - v.position).length() < 1e-5),
                "감기를 뒤집었더니 없던 점이 생겼다: {v:?}"
            );
        }
    }
}
