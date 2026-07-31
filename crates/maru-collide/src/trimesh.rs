//! **삼각형 메시** — 정점·인덱스 + 내부 BVH (설계: 하이트필드와 같은 "비볼록 컨테이너" 계약).
//!
//! # 왜 바디 N개가 아니라 셰이프 하나인가
//!
//! 삼각형을 바디로 두면 브로드페이즈 트리에 N개가 들어가고 바디당 부기도 N배가 된다.
//! 실측(`maru-bench --bin meshcost`): 같은 삼각형 수를 바디 N개 vs 내부 구조 하나로 놓으면
//! 1,152개에서 1.08배, 4,608개에서 1.27배, **12,800개에서 1.79배** 차이가 난다 —
//! 이득이 메시 크기에 비례한다. 대신 narrow는 비싸진다(접촉 순간에 삼각형을 펼친다):
//! 같은 실측에서 broad 31→10 us, post 54→20 us인데 narrow는 149→214 us였다.
//! 이 타입은 그 거래를 사는 쪽이고, 삼각형이 많을수록 유리하다.
//!
//! # 계약 (하이트필드와 동일)
//!
//! - **볼록이 아니다.** GJK/EPA·support 경로에 단독으로 들어가면 안 된다 —
//!   모든 진입점이 삼각형(`Shape::Triangle`)으로 디스패치하고 `support_core`는 unreachable이다.
//! - **static 전용.** 질량·관성 경로가 없다(월드가 강제한다).
//! - **데이터는 이 타입이 소유하고 필드는 비공개다.** 파생물(BVH·용접 마스크)을 호스트가
//!   위조하면 가지치기가 진짜 접촉을 조용히 건너뛴다 — 하이트필드의 `tiles`가 같은 이유로
//!   비공개다(확인검수 rev_97887d75). **검사 대신 구성으로 막으면** 질의 경계의 재검증이
//!   런타임 비용 0이 된다.
//!
//! # 용접(내부 에지) 마스크를 왜 빌드 시점에 만드는가
//!
//! 격자는 이웃이 **좌표에서 나오지만**(하이트필드의 `neighbor`), 메시는 그렇지 않다 —
//! 인접은 에지 공유로만 안다. 그래서 빌드 때 한 번 계산해 삼각형마다 8비트로 들고 있는다.
//! 비트 `i` = 에지 `(i, (i+1)%3)`가 내부 에지(= 이웃과 공유하고 법선이 `COS_COPLANAR` 안).
//! 이 규약은 `manifold::weld_internal_edges`가 그대로 읽는다.

use crate::Aabb;
use maru_math::Vec3;
use std::collections::HashMap;

/// 메시 삼각형 수 상한. 순회 예산과 스냅샷 크기를 유계로 만드는 정책값이다.
pub const MAX_TRIMESH_TRIS: usize = 1 << 20;
/// 메시 정점 수 상한.
pub const MAX_TRIMESH_VERTS: usize = 1 << 20;

/// 삼각형 메시 — 정점·인덱스 + 내부 BVH + 용접 마스크.
///
/// 파생물(BVH·마스크·AABB)은 **캐노니컬 상태가 아니다**: 스냅샷에는 정점·인덱스만 넣고
/// 복원에서 다시 만든다(하이트필드 `tiles`와 같은 정책).
// **다음 슬라이스가 지울 표식이다.** `tree`·`weld`와 그 접근자는 접촉·레이·캐스트 디스패치가
// 쓸 것인데, 그 디스패치는 아직 없다. 지금 `-D warnings`를 통과시키려고 접근자를 공개 API로
// 올리면 "공개 표면은 셰이프 무관 디스패처만"이라는 계약이 깨지므로(README), 대신 **이 표식을
// 남긴다** — 조용한 강등이 아니라 눈에 띄는 미완 표시다. 디스패치가 들어오면 이 줄은 사라지고,
// 사라지지 않으면 그 자체가 "아직 안 붙였다"는 신호다.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct TriMeshData {
    verts: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    /// 삼각형 인덱스를 리프 id로 담은 4-ary BVH — 브로드페이즈와 **같은 타입**을 쓴다.
    tree: crate::tree::Tree,
    /// 삼각형별 내부 에지 비트(빌드 시점 계산 — 위 모듈 주석의 규약).
    weld: Vec<u8>,
    local_aabb: Aabb,
}

/// `Tree`는 `PartialEq`가 없고 파생물이라 비교 대상이 아니다 — **캐노니컬 데이터만** 비교한다.
/// (스냅샷 왕복 검증이 이 비교를 쓴다: 파생물이 달라도 같은 입력이면 같은 메시다.)
impl PartialEq for TriMeshData {
    fn eq(&self, other: &Self) -> bool {
        self.verts == other.verts && self.tris == other.tris
    }
}

#[allow(dead_code)]
impl TriMeshData {
    /// 메시를 만든다 — 구조 불변식을 검사하고 BVH·용접 마스크를 계산한다. 어기면 `None`.
    ///
    /// 검사 항목(전부 뒤쪽 인덱싱·기하 계산의 전제다):
    /// 비지 않음 · 개수 상한 · **인덱스가 정점 범위 안** · 정점 유한하고 크기 정책 안 ·
    /// 삼각형이 퇴화가 아님(면적 > 차원별 epsilon).
    ///
    /// **크기 정책까지 여기서 세운다** — 하이트필드가 정책을 `Shape::validate`에만 뒀다가
    /// "생성 경계는 거절하는데 질의 경계는 통과시키는" 구멍을 냈다(rev_7b22e13d BLOCKER).
    /// 필드가 비공개이므로 여기서 한 번 세우면 질의 경계는 O(1)로 참을 유지한다.
    pub fn new(verts: Vec<Vec3>, tris: Vec<[u32; 3]>) -> Option<Self> {
        if verts.is_empty()
            || tris.is_empty()
            || verts.len() > MAX_TRIMESH_VERTS
            || tris.len() > MAX_TRIMESH_TRIS
        {
            return None;
        }
        // 정점: 유한 + 크기 정책. (`extent_in_range`가 NaN·Inf·과대값을 함께 거른다.)
        if !verts.iter().all(|v| {
            crate::extent_in_range(v.x)
                && crate::extent_in_range(v.y)
                && crate::extent_in_range(v.z)
        }) {
            return None;
        }
        // 인덱스 범위 — 이 검사가 없으면 뒤의 모든 `verts[i]`가 패닉 후보다(wasm에선 trap).
        let nv = verts.len() as u32;
        if !tris.iter().all(|t| t.iter().all(|&i| i < nv)) {
            return None;
        }
        // 퇴화 삼각형 거부: 면적이 0이면 법선이 없고, 법선 없는 삼각형은 GJK/EPA를 깨뜨린다.
        // 임계는 **차원별로 유도**한다 — cross의 길이는 L²이므로 `area_eps`(L²)와 비교한다
        // (`length_sq`는 L⁴이라 비교하면 스케일에서 깨진다, tolerance 모듈 주석).
        for t in &tris {
            let (a, b, c) = (
                verts[t[0] as usize],
                verts[t[1] as usize],
                verts[t[2] as usize],
            );
            let scale = (b - a).length().max((c - a).length()).max((c - b).length());
            if (b - a).cross(c - a).length() <= crate::tolerance::area_eps(scale) {
                return None;
            }
        }
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for v in &verts {
            min = Vec3::new(min.x.min(v.x), min.y.min(v.y), min.z.min(v.z));
            max = Vec3::new(max.x.max(v.x), max.y.max(v.y), max.z.max(v.z));
        }
        let local_aabb = Aabb::new(min, max);
        // BVH: 리프 id = 삼각형 인덱스.
        let items: Vec<(u32, Aabb)> = tris
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let (a, b, c) = (
                    verts[t[0] as usize],
                    verts[t[1] as usize],
                    verts[t[2] as usize],
                );
                let lo = Vec3::new(
                    a.x.min(b.x).min(c.x),
                    a.y.min(b.y).min(c.y),
                    a.z.min(b.z).min(c.z),
                );
                let hi = Vec3::new(
                    a.x.max(b.x).max(c.x),
                    a.y.max(b.y).max(c.y),
                    a.z.max(b.z).max(c.z),
                );
                (i as u32, Aabb::new(lo, hi))
            })
            .collect();
        let tree = crate::tree::Tree::build(&items);
        let weld = build_weld_masks(&verts, &tris);
        Some(Self {
            verts,
            tris,
            tree,
            weld,
            local_aabb,
        })
    }

    pub fn verts(&self) -> &[Vec3] {
        &self.verts
    }

    pub fn tris(&self) -> &[[u32; 3]] {
        &self.tris
    }

    pub fn tri_count(&self) -> usize {
        self.tris.len()
    }

    pub fn local_aabb(&self) -> Aabb {
        self.local_aabb
    }

    /// 삼각형 `i`의 로컬 정점 3개. **범위 미검사** — 호출자는 BVH가 돌려준 id만 넘긴다
    /// (크레이트 전용, 하이트필드 `triangle`과 같은 규약).
    pub(crate) fn triangle(&self, i: u32) -> [Vec3; 3] {
        let t = self.tris[i as usize];
        [
            self.verts[t[0] as usize],
            self.verts[t[1] as usize],
            self.verts[t[2] as usize],
        ]
    }

    /// 삼각형 `i`의 내부 에지 마스크 (비트 `e` = 에지 `(e, (e+1)%3)`).
    pub(crate) fn weld_mask(&self, i: u32) -> u8 {
        self.weld[i as usize]
    }

    /// 로컬 AABB와 겹치는 삼각형 후보를 모은다 (BVH 순회, 상위집합).
    pub(crate) fn query_in(&self, local: Aabb, out: &mut Vec<u32>, stack: &mut Vec<u32>) {
        self.tree.query_candidates_in(local, out, stack);
    }
}

/// 에지 공유 + 법선 정렬로 내부 에지를 표시한다.
///
/// 격자와 달리 메시는 인접이 좌표에서 나오지 않으므로 **에지 → 삼각형** 사전을 한 번 만든다.
/// 에지 키는 정점 인덱스 쌍을 **정렬**해 만든다 — 와인딩이 반대인 이웃(정상적인 다양체)도
/// 같은 키를 갖게 하려는 것이다.
///
/// 한 에지를 셋 이상이 공유하면(비다양체) 그 에지는 내부로 보지 않는다: 어느 이웃과
/// 정렬됐다고 판단해야 할지 정할 수 없고, 잘못 용접하면 **진짜 충돌 법선을 지워** 물체가
/// 지형을 통과한다. 모르면 용접하지 않는 쪽이 안전하다(용접은 품질 개선이지 정확성이 아니다).
fn build_weld_masks(verts: &[Vec3], tris: &[[u32; 3]]) -> Vec<u8> {
    let normal = |t: &[u32; 3]| -> Option<Vec3> {
        let (a, b, c) = (
            verts[t[0] as usize],
            verts[t[1] as usize],
            verts[t[2] as usize],
        );
        let n = (b - a).cross(c - a);
        let len = n.length();
        let scale = (b - a).length().max((c - a).length()).max((c - b).length());
        if len <= crate::tolerance::area_eps(scale) {
            None
        } else {
            Some(n * (1.0 / len))
        }
    };
    // 에지 → 그 에지를 쓰는 삼각형들. 다양체면 최대 2개다.
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3usize {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(ti as u32);
        }
    }
    let mut out = vec![0u8; tris.len()];
    for (ti, t) in tris.iter().enumerate() {
        let Some(n) = normal(t) else {
            continue; // 퇴화는 생성자가 이미 거절한다 — 방어적 분기.
        };
        let mut mask = 0u8;
        for e in 0..3usize {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            let Some(users) = edges.get(&(a.min(b), a.max(b))) else {
                continue;
            };
            if users.len() != 2 {
                continue; // 경계 에지(1) 또는 비다양체(3+) — 위 주석의 이유로 용접하지 않는다.
            }
            let other = if users[0] == ti as u32 {
                users[1]
            } else {
                users[0]
            };
            let Some(n2) = normal(&tris[other as usize]) else {
                continue;
            };
            // 와인딩이 반대인 이웃은 법선이 뒤집혀 보인다 — **절대값**으로 판정해
            // 같은 평면이면 방향과 무관하게 내부로 본다(격자는 와인딩이 일관돼 이 문제가 없다).
            if n.dot(n2).abs() >= crate::COS_COPLANAR {
                mask |= 1 << e;
            }
        }
        out[ti] = mask;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 두 삼각형이 대각 에지를 공유하는 평면 사각형.
    fn quad() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        (
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        )
    }

    #[test]
    fn rejects_structurally_broken_meshes() {
        // 빈 입력
        assert!(TriMeshData::new(vec![], vec![]).is_none());
        let (v, _) = quad();
        assert!(TriMeshData::new(v.clone(), vec![]).is_none());
        // **범위 밖 인덱스** — 이 검사가 없으면 뒤의 모든 인덱싱이 패닉(wasm에선 trap)이다.
        assert!(TriMeshData::new(v.clone(), vec![[0, 1, 99]]).is_none());
        // 퇴화 삼각형(세 점이 한 직선)
        let line = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        assert!(TriMeshData::new(line, vec![[0, 1, 2]]).is_none());
        // 같은 점 두 개(면적 0)
        let dup = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        assert!(TriMeshData::new(dup, vec![[0, 1, 2]]).is_none());
        // 비유한 정점
        let nan = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(TriMeshData::new(nan, vec![[0, 1, 2]]).is_none());
        // 크기 정책 밖(MAX_INPUT 초과)
        let huge = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(crate::MAX_INPUT * 10.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(TriMeshData::new(huge, vec![[0, 1, 2]]).is_none());
        // 정상 입력은 만들어진다(음성 대조 — 위 거절들이 "전부 거절"이 아님을 보인다)
        let (v, t) = quad();
        assert!(TriMeshData::new(v, t).is_some());
    }

    #[test]
    fn shared_coplanar_edge_is_internal_and_boundary_is_not() {
        let (v, t) = quad();
        let m = TriMeshData::new(v, t).expect("정상 메시");
        // 삼각형 0 = [0,1,2]: 에지 1 = (1→2)가 공유 대각. 나머지 둘은 경계.
        assert_eq!(m.weld_mask(0), 0b010, "공유 대각만 내부여야 한다");
        // 삼각형 1 = [1,3,2]: 에지 2 = (2→1)이 같은 대각.
        assert_eq!(m.weld_mask(1), 0b100);
    }

    #[test]
    fn sharp_fold_is_not_welded() {
        // 90° 접힘 — 공유 에지지만 법선이 어긋나므로 내부가 아니다.
        // 여기를 용접하면 모서리의 진짜 충돌 법선이 지워져 물체가 지형을 통과한다.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let m = TriMeshData::new(verts, vec![[0, 1, 2], [0, 3, 1]]).expect("정상 메시");
        assert_eq!(m.weld_mask(0), 0, "90도 접힘은 용접 대상이 아니다");
        assert_eq!(m.weld_mask(1), 0);
    }

    #[test]
    fn non_manifold_edge_is_not_welded() {
        // 한 에지를 삼각형 셋이 공유(T자 벽) — 어느 이웃과 맞출지 정할 수 없으므로 용접 금지.
        // 셋 중 둘은 동일 평면이라 "정렬만" 보면 용접해 버린다(그러면 벽이 사라진다).
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let m = TriMeshData::new(verts, vec![[0, 1, 2], [0, 3, 1], [0, 1, 4]]).expect("정상 메시");
        for i in 0..3u32 {
            assert_eq!(
                m.weld_mask(i) & 0b001,
                0,
                "비다양체 에지(삼각형 {i}의 에지 0)는 용접하지 않는다"
            );
        }
    }

    #[test]
    fn bvh_query_returns_a_superset_of_overlapping_triangles() {
        let (v, t) = quad();
        let m = TriMeshData::new(v, t).expect("정상 메시");
        let (mut out, mut stack) = (Vec::new(), Vec::new());
        // 사각형 전체를 덮는 질의 — 둘 다 후보다.
        m.query_in(
            Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(2.0, 1.0, 2.0)),
            &mut out,
            &mut stack,
        );
        out.sort_unstable();
        assert_eq!(out, vec![0, 1]);
        // 완전히 벗어난 질의 — 후보 없음(가지치기가 실제로 자른다는 증거).
        out.clear();
        m.query_in(
            Aabb::new(Vec3::new(10.0, 10.0, 10.0), Vec3::new(11.0, 11.0, 11.0)),
            &mut out,
            &mut stack,
        );
        assert!(out.is_empty(), "떨어진 질의가 후보를 냈다: {out:?}");
    }

    /// `triangle(i)`가 **인덱스를 통해** 올바른 정점 3개를 낸다.
    ///
    /// 이 접근자는 다음 슬라이스(접촉 디스패치)가 BVH 후보 id로 호출한다. 인덱스 해석이
    /// 어긋나면 **엉뚱한 삼각형이 펼쳐져** 물체가 허공에서 튕기거나 지형을 통과하는데,
    /// 그건 상위 씬 테스트에서 원인 추적이 어렵다 — 여기서 직접 잠근다.
    #[test]
    fn triangle_accessor_resolves_indices() {
        let (v, t) = quad();
        let m = TriMeshData::new(v.clone(), t.clone()).expect("정상 메시");
        for (i, tri) in t.iter().enumerate() {
            let got = m.triangle(i as u32);
            for k in 0..3 {
                assert_eq!(
                    got[k], v[tri[k] as usize],
                    "삼각형 {i}의 정점 {k}가 인덱스와 어긋난다"
                );
            }
        }
    }

    #[test]
    fn local_aabb_covers_every_vertex() {
        let (v, t) = quad();
        let m = TriMeshData::new(v.clone(), t).expect("정상 메시");
        let bb = m.local_aabb();
        for p in &v {
            assert!(
                p.x >= bb.min.x && p.x <= bb.max.x && p.y >= bb.min.y && p.y <= bb.max.y,
                "AABB가 정점 {p:?}를 안 덮는다"
            );
        }
    }
}
