//! 볼록 껍질의 **면 구조**(canonical face topology) — 생성 시 한 번 만드는 파생물.
//!
//! ## 왜 필요한가
//!
//! 면 클리핑 매니폴드는 "접촉 법선과 가장 잘 맞는 **면**"을 알아야 한다. 지금 엔진은
//! 박스-박스만 그 면을 알고(박스는 면이 6개로 고정) 나머지 볼록 쌍은 GJK/EPA witness
//! **단일점**으로 떨어져 평면 접촉이 불안정하다(실측: 박스 모양 hull이 지면에 가라앉는다).
//!
//! `Shape::Hull`은 점 집합만 들고 있어서, 런타임에 면을 뽑으려면 "`p·dir`가 최대인 점들"을
//! 임계로 모아야 한다. **그 임계가 위험하다** — 이 저장소에서 임계 기반 기하 판정이 반복해서
//! 결함의 원인이었다(퇴화 코어의 법선·저차원 판정·동률 증명). 그래서 면 구조를 **생성 시**
//! 한 번 정확히 만들고 런타임은 그걸 읽기만 한다(교차검수 rev_ab7232ed 승인 설계).
//!
//! ## 파생물이지 캐노니컬 상태가 아니다
//!
//! 스냅샷에는 **점만** 저장하고 복원에서 다시 만든다(`TriMeshData`의 BVH·`HeightfieldGrid`의
//! 타일과 같은 정책). 그래서 빌더는 **결정적**이어야 한다 — 같은 점 배열이면 같은 면 목록,
//! 같은 순서, 같은 정점 순서. 테스트가 그걸 잠근다.
//!
//! ## 알고리즘: 삼중조합 전수 + 지지 검사
//!
//! 점이 **볼록 위치에 있다고 가정**한다(엔진 계약: hull 점은 빌더가 준다). 그러면 면 평면은
//! "세 점이 만든 평면 중 나머지 점이 전부 한쪽에 있는 것"이다. 상한이 32점이라 삼중조합이
//! 4,960개뿐이고 생성은 스텝 중이 아니므로 전수로 돌린다 — **근사가 아니라 열거**다.
//!
//! 같은 면을 여러 삼중조합이 만들므로 평면으로 중복을 없애고, 각 평면 위 점을 모아
//! 법선 기준 CCW로 정렬한다.

use crate::limits::MAX_HULL_POINTS;
use maru_math::Vec3;

/// 면 하나 — 바깥 법선과 그 위 정점(원본 인덱스, 법선 기준 CCW).
#[derive(Clone, Debug, PartialEq)]
pub struct HullFace {
    /// 바깥을 향하는 단위 법선.
    pub normal: Vec3,
    /// 평면 오프셋 — 면 위 점 `p`는 `normal·p == offset`.
    pub offset: f32,
    /// 이 면 위의 정점 인덱스, **법선에서 봤을 때 CCW**. 3개 이상이다.
    pub verts: Vec<u8>,
}

/// hull의 면 목록. 인덱스가 곧 **면 id**이고, feature 계보가 이 id를 쓴다.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HullFaces {
    faces: Vec<HullFace>,
}

impl HullFaces {
    pub fn faces(&self) -> &[HullFace] {
        &self.faces
    }
    pub fn len(&self) -> usize {
        self.faces.len()
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// 방향 `dir`에 가장 잘 맞는 면의 인덱스와 그 정렬도(`normal·dir`).
    ///
    /// **임계가 없다** — 최대를 고르고 동률은 낮은 인덱스가 이긴다(면 목록이 결정적이므로
    /// 답도 결정적이다). 면이 없으면 `None`(퇴화 hull).
    pub fn best_face(&self, dir: Vec3) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, f) in self.faces.iter().enumerate() {
            let d = f.normal.dot(dir);
            if best.is_none_or(|(_, bd)| d > bd) {
                best = Some((i, d));
            }
        }
        best
    }

    /// 점 집합에서 면 구조를 만든다. 점이 **볼록 위치**에 있다고 가정한다.
    ///
    /// 3차원이 아니면(모두 한 평면·한 직선 위) 면이 0개인 빈 구조를 돌려준다 — 오류가
    /// 아니다. 그런 hull도 convex radius가 두께를 주면 합법이고, 면 클리핑을 못 할 뿐이다
    /// (호출자는 `is_empty()`를 보고 기존 witness 경로로 간다).
    pub fn build(points: &[Vec3]) -> Self {
        let n = points.len();
        if !(4..=MAX_HULL_POINTS).contains(&n) {
            return Self::default();
        }
        // **임계는 점 집합의 지름에 상대적**이다 — 절대 상수는 장면 스케일에서 깨진다.
        let mut span = 0.0f32;
        for p in points {
            for q in points {
                span = span.max((*p - *q).length());
            }
        }
        if span <= 0.0 || span.is_nan() {
            return Self::default();
        }
        let eps = crate::length_eps(span);

        let mut faces: Vec<HullFace> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let Some(nrm) = (points[j] - points[i])
                        .cross(points[k] - points[i])
                        .try_normalize()
                    else {
                        continue; // 세 점이 한 직선 위 — 평면을 못 만든다
                    };
                    // 두 방향 모두 후보다: 어느 쪽이 바깥인지는 나머지 점이 정한다.
                    for cand in [nrm, nrm * -1.0] {
                        let off = cand.dot(points[i]);
                        // **지지 검사**: 모든 점이 `cand` 반대쪽(또는 평면 위)이면 바깥 면이다.
                        let mut hi = f32::NEG_INFINITY;
                        for p in points {
                            hi = hi.max(cand.dot(*p));
                        }
                        if hi > off + eps {
                            continue; // 평면 밖에 점이 있다 — 면이 아니다
                        }
                        Self::insert_face(&mut faces, points, cand, off, eps);
                    }
                }
            }
        }
        // 면 순서를 **입력만의 함수**로 고정한다. 삼중조합 순회 순서가 이미 결정적이지만,
        // 정렬을 명시해 두면 나중에 순회를 바꿔도 id가 안 흔들린다(feature 계보가 이 id다).
        faces.sort_by(|a, b| {
            a.verts
                .cmp(&b.verts)
                .then_with(|| a.normal.x.total_cmp(&b.normal.x))
                .then_with(|| a.normal.y.total_cmp(&b.normal.y))
                .then_with(|| a.normal.z.total_cmp(&b.normal.z))
        });
        Self { faces }
    }

    /// 이미 있는 면이면 버리고, 새 면이면 그 평면 위 정점을 모아 CCW로 정렬해 넣는다.
    fn insert_face(
        faces: &mut Vec<HullFace>,
        points: &[Vec3],
        normal: Vec3,
        offset: f32,
        eps: f32,
    ) {
        // 같은 평면인가 — 법선이 거의 같고 오프셋이 거의 같으면 같은 면이다.
        if faces
            .iter()
            .any(|f| f.normal.dot(normal) > 0.999_9 && (f.offset - offset).abs() <= eps)
        {
            return;
        }
        // 이 평면 위 정점 전부.
        let mut on: Vec<u8> = Vec::new();
        let mut centroid = Vec3::ZERO;
        for (idx, p) in points.iter().enumerate() {
            if (normal.dot(*p) - offset).abs() <= eps {
                on.push(idx as u8);
                centroid += *p;
            }
        }
        if on.len() < 3 {
            return; // 면이 되려면 정점 3개 이상
        }
        centroid = centroid * (1.0 / on.len() as f32);
        // 면 평면 안의 기준축 — **첫 정점**으로 잡아 입력만의 함수로 만든다.
        let Some(u) = (points[on[0] as usize] - centroid).try_normalize() else {
            return; // 첫 정점이 중심과 겹친다(퇴화)
        };
        let v = normal.cross(u);
        // 법선에서 봤을 때 CCW. 각도 동률(같은 자리 중복점)은 인덱스가 가른다.
        on.sort_by(|a, b| {
            let pa = points[*a as usize] - centroid;
            let pb = points[*b as usize] - centroid;
            let aa = pa.dot(v).atan2(pa.dot(u));
            let ab = pb.dot(v).atan2(pb.dot(u));
            aa.total_cmp(&ab).then_with(|| a.cmp(b))
        });
        faces.push(HullFace {
            normal,
            offset,
            verts: on,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn box_points(hx: f32, hy: f32, hz: f32) -> Vec<Vec3> {
        let mut pts = Vec::new();
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    pts.push(v(hx * sx, hy * sy, hz * sz));
                }
            }
        }
        pts
    }

    #[test]
    fn box_hull_has_six_quad_faces() {
        let f = HullFaces::build(&box_points(1.0, 2.0, 3.0));
        assert_eq!(f.len(), 6, "박스는 면이 6개다");
        for face in f.faces() {
            assert_eq!(face.verts.len(), 4, "박스 면은 사각형이다: {face:?}");
            // 법선은 축 하나에만 성분이 있다.
            let n = face.normal;
            let axes = [n.x.abs(), n.y.abs(), n.z.abs()];
            let big = axes.iter().filter(|a| **a > 0.5).count();
            assert_eq!(big, 1, "박스 면 법선이 축 정렬이 아니다: {n:?}");
        }
    }

    /// **법선은 바깥을 향해야 한다** — 안쪽이면 클리핑의 reference/incident가 뒤집힌다.
    #[test]
    fn face_normals_point_outward() {
        for pts in [box_points(1.0, 1.0, 1.0), box_points(0.1, 5.0, 0.3)] {
            let f = HullFaces::build(&pts);
            for face in f.faces() {
                // 중심(원점)은 모든 면의 안쪽에 있어야 한다.
                assert!(
                    face.normal.dot(Vec3::ZERO) < face.offset,
                    "법선이 안쪽을 본다: {face:?}"
                );
                // 그리고 어떤 점도 면 밖에 없다.
                for p in &pts {
                    assert!(
                        face.normal.dot(*p) <= face.offset + 1e-4,
                        "면 밖에 점이 있다: {p:?} vs {face:?}"
                    );
                }
            }
        }
    }

    /// 면 정점은 **법선 기준 CCW**여야 한다 — 클리핑의 측면 평면 방향이 여기서 나온다.
    #[test]
    fn face_verts_are_ccw_around_the_normal() {
        let pts = box_points(1.0, 2.0, 3.0);
        let f = HullFaces::build(&pts);
        for face in f.faces() {
            let mut area = Vec3::ZERO;
            let m = face.verts.len();
            for i in 0..m {
                let a = pts[face.verts[i] as usize];
                let b = pts[face.verts[(i + 1) % m] as usize];
                area += a.cross(b);
            }
            assert!(
                area.dot(face.normal) > 0.0,
                "정점 순서가 CW다: {face:?} (area·n = {})",
                area.dot(face.normal)
            );
        }
    }

    /// 사면체 — 면 4개, 전부 삼각형.
    #[test]
    fn tetrahedron_has_four_triangular_faces() {
        let pts = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let f = HullFaces::build(&pts);
        assert_eq!(f.len(), 4);
        for face in f.faces() {
            assert_eq!(face.verts.len(), 3);
        }
    }

    /// **면 위에 여분 정점이 있어도 하나의 면**이어야 한다 — 삼각형으로 쪼개지면
    /// 클리핑이 같은 면을 여러 번 처리하고 feature 계보가 흔들린다.
    #[test]
    fn coplanar_extra_vertex_joins_one_face() {
        let mut pts = box_points(1.0, 1.0, 1.0);
        // +Z 면 한가운데에 점을 하나 더 얹는다(볼록 위치를 깨지 않는다).
        pts.push(v(0.0, 0.0, 1.0));
        let f = HullFaces::build(&pts);
        assert_eq!(f.len(), 6, "면이 쪼개졌다: {}", f.len());
        let zface = f.faces().iter().find(|f| f.normal.z > 0.9).expect("+Z 면");
        assert_eq!(zface.verts.len(), 5, "여분 정점이 면에 안 들어갔다");
    }

    /// **평면 점 집합은 앞뒤 두 면**을 만든다 — 얇은 판의 양면이고, convex radius가 두께를
    /// 주므로 실제로 클리핑할 수 있는 면이다. 지지 검사가 ±n 양쪽에서 통과하기 때문이고,
    /// 그게 기하학적으로 맞다(모든 점이 두 반공간 경계 위에 있다).
    #[test]
    fn planar_point_set_has_two_opposite_faces() {
        let planar = vec![
            v(-1.0, 0.0, -1.0),
            v(1.0, 0.0, -1.0),
            v(1.0, 0.0, 1.0),
            v(-1.0, 0.0, 1.0),
        ];
        let f = HullFaces::build(&planar);
        assert_eq!(f.len(), 2, "평면 판은 앞뒤 두 면이다");
        let (a, b) = (&f.faces()[0], &f.faces()[1]);
        assert!(
            (a.normal + b.normal).length() < 1e-5,
            "두 면이 서로 반대가 아니다: {:?} / {:?}",
            a.normal,
            b.normal
        );
        assert_eq!(a.verts.len(), 4);
        assert_eq!(b.verts.len(), 4);
        // 두 면 다 CCW 계약을 지킨다(방향이 반대이므로 정점 순서도 반대여야 한다).
        for face in f.faces() {
            let mut area = Vec3::ZERO;
            let m = face.verts.len();
            for i in 0..m {
                area +=
                    planar[face.verts[i] as usize].cross(planar[face.verts[(i + 1) % m] as usize]);
            }
            assert!(area.dot(face.normal) > 0.0, "CW다: {face:?}");
        }
    }

    /// **좌표가 딱 떨어지지 않는 면**에서도 공면 정점이 모두 한 면에 모여야 한다.
    ///
    /// 축 정렬 박스는 평면 산술이 정확해서 임계가 0이어도 통과한다 — 그걸로는 임계가
    /// 일을 하는지 알 수 없다. 육각기둥은 뚜껑 6점의 법선·오프셋이 마지막 비트에서
    /// 흔들리므로 임계가 있어야 6각형 하나가 된다.
    #[test]
    fn hexagonal_prism_caps_are_single_faces() {
        let mut pts = Vec::new();
        for k in 0..6 {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            for sy in [-1.0f32, 1.0] {
                pts.push(v(a.cos() * 0.73, sy * 1.37, a.sin() * 0.73));
            }
        }
        let f = HullFaces::build(&pts);
        // 뚜껑 2개 + 옆면 6개.
        assert_eq!(f.len(), 8, "면 수가 다르다: {}", f.len());
        let caps: Vec<&HullFace> = f
            .faces()
            .iter()
            .filter(|f| f.normal.y.abs() > 0.9)
            .collect();
        assert_eq!(caps.len(), 2, "뚜껑이 2개가 아니다");
        for c in caps {
            assert_eq!(c.verts.len(), 6, "뚜껑이 6각형이 아니다: {c:?}");
        }
        for face in f.faces().iter().filter(|f| f.normal.y.abs() <= 0.9) {
            assert_eq!(face.verts.len(), 4, "옆면이 사각형이 아니다: {face:?}");
        }
    }

    /// **면 id는 발견 순서가 아니라 면 내용으로 정해진다.**
    ///
    /// feature 계보가 이 id를 쓰므로, 나중에 삼중조합 순회 순서를 바꿔도 id가 흔들리면
    /// warm start가 통째로 끊긴다. 그래서 빌더가 정해진 키로 정렬해 둔다 — 그 불변식을
    /// 여기서 잠근다(정렬을 빼면 발견 순서가 그대로 id가 된다).
    #[test]
    fn face_ids_follow_a_canonical_order() {
        for pts in [
            box_points(1.0, 2.0, 3.0),
            vec![
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(0.0, 0.0, 1.0),
            ],
        ] {
            let f = HullFaces::build(&pts);
            let key = |x: &HullFace| {
                (
                    x.verts.clone(),
                    x.normal.x.to_bits(),
                    x.normal.y.to_bits(),
                    x.normal.z.to_bits(),
                )
            };
            let mut sorted: Vec<_> = f.faces().iter().map(key).collect();
            let asis = sorted.clone();
            sorted.sort();
            assert_eq!(asis, sorted, "면이 캐노니컬 순서가 아니다");
        }
    }

    /// 1차원 이하(선·점)와 점이 너무 적은 경우는 **빈 구조**다 — 오류가 아니라
    /// "면 클리핑 불가"이고, 호출자는 기존 witness 경로로 간다.
    #[test]
    fn degenerate_point_sets_have_no_faces() {
        let planar = [
            v(-1.0, 0.0, -1.0),
            v(1.0, 0.0, -1.0),
            v(1.0, 0.0, 1.0),
            v(-1.0, 0.0, 1.0),
        ];
        // 한 직선 위
        let line: Vec<Vec3> = (0..5).map(|i| v(i as f32, 0.0, 0.0)).collect();
        assert!(HullFaces::build(&line).is_empty());
        // 점이 너무 적다
        assert!(HullFaces::build(&planar[..3]).is_empty());
        // 모든 점이 한 자리
        assert!(HullFaces::build(&[Vec3::ZERO; 5]).is_empty());
    }

    /// **결정적이어야 한다** — 파생물이라 복원에서 다시 만들고, feature 계보가 면 id를 쓴다.
    /// 같은 입력이면 면 목록·순서·정점 순서가 모두 같아야 한다.
    #[test]
    fn build_is_deterministic() {
        let pts = box_points(1.0, 2.0, 3.0);
        let a = HullFaces::build(&pts);
        for _ in 0..8 {
            assert_eq!(a, HullFaces::build(&pts), "같은 입력에 다른 면 구조");
        }
    }

    /// `best_face`는 임계 없이 최대를 고르고, 동률은 낮은 인덱스가 이긴다.
    #[test]
    fn best_face_picks_the_aligned_face() {
        let pts = box_points(1.0, 1.0, 1.0);
        let f = HullFaces::build(&pts);
        for (dir, want) in [
            (v(0.0, 1.0, 0.0), v(0.0, 1.0, 0.0)),
            (v(0.0, -1.0, 0.0), v(0.0, -1.0, 0.0)),
            (v(0.9, 0.1, 0.0), v(1.0, 0.0, 0.0)),
        ] {
            let (i, d) = f.best_face(dir).expect("면이 있다");
            assert!(
                (f.faces()[i].normal - want).length() < 1e-5,
                "{dir:?} → {:?} (기대 {want:?})",
                f.faces()[i].normal
            );
            assert!(d > 0.0);
        }
        // 동률(정육면체를 대각으로 보면 세 면이 같다) — 두 번 물어도 같은 답.
        let diag = v(1.0, 1.0, 1.0).try_normalize().unwrap();
        assert_eq!(f.best_face(diag), f.best_face(diag));
    }

    /// 상한(32점)에서도 만들어지고 면 수가 오일러 상한(2m−4) 안이다.
    #[test]
    fn respects_the_point_cap() {
        // 구 표면 샘플 32점 — 볼록 위치.
        let mut pts = Vec::new();
        let mut rng = 12345u32;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        while pts.len() < MAX_HULL_POINTS {
            if let Some(u) = v(next(), next(), next()).try_normalize() {
                pts.push(u);
            }
        }
        let f = HullFaces::build(&pts);
        assert!(!f.is_empty(), "32점 구 샘플에 면이 없다");
        assert!(
            f.len() <= 2 * MAX_HULL_POINTS - 4,
            "면 수가 오일러 상한을 넘었다: {}",
            f.len()
        );
        // 상한을 넘는 입력은 빈 구조(정책값 초과는 거절이다).
        pts.push(v(2.0, 0.0, 0.0));
        assert!(HullFaces::build(&pts).is_empty());
    }
}
