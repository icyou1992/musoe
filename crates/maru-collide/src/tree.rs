//! 4-ary AABB 트리 — 브로드페이즈 가속 구조 (설계 §4.1).
//!
//! 계약: **트리는 성능 전용이다.** 후보는 superset이면 되고, 최종 페어 집합·쿼리 결과는
//! 정밀 AABB 판정 + canonical 정렬로 확정된다(pairs 모듈). 노드 바운드는 SoA `Aabbx4` —
//! 순회 1스텝 = SIMD128 비교 1회. [근거: Jolt 쿼드트리]
//! v1 범위: 결정적 top-down 빌드 + overlap 쿼리. widen-only 증분·백그라운드 재빌드는 M2+.

use crate::Aabb;
use maru_math::{Aabbx4, Vec3, Vec3x4};
use wide::f32x4;

// leaf/node/empty 인코딩 (리뷰 MAJOR 반영 — 구 i32 보수 인코딩은 id 0x7fff_ffff가
// EMPTY(i32::MIN)와 충돌해 해당 리프가 질의에서 소실됐다):
// 최상위 비트 = leaf, EMPTY = LEAF_BIT-1 (node 인덱스 용량 상한으로 예약).
const LEAF_BIT: u32 = 1 << 31;
const EMPTY: u32 = LEAF_BIT - 1;

#[derive(Clone, Copy, Debug)]
enum Child {
    Node(u32),
    Leaf(u32),
    None,
}

pub struct Tree {
    /// 노드별 자식 4개의 바운드(SoA). 빈 레인은 반전 AABB(항상 불일치).
    bounds: Vec<soa_bounds::AabbLanes>,
    children: Vec<[u32; 4]>,
    root: u32,
}

fn decode(c: u32) -> Child {
    if c == EMPTY {
        Child::None
    } else if c & LEAF_BIT != 0 {
        Child::Leaf(c & !LEAF_BIT)
    } else {
        Child::Node(c)
    }
}

impl Tree {
    /// (id, aabb) 목록에서 top-down 4분할 빌드.
    ///
    /// **트리 위상은 결과에 영향을 주지 않는다** — 페어 집합은 트리 후보에 정밀 AABB
    /// 판정과 canonical 정렬을 적용한 결과로 정의된다(설계 §4.1,
    /// `pair_set_is_tree_topology_independent`가 이를 잠근다). 그래서 분할 방식은
    /// 자유롭게 고를 수 있고, 여기서는 **중앙값 선택**(O(n))을 쓴다 — 전에는 중앙값
    /// 하나를 얻으려고 전체 정렬(O(n log n))을 돌렸다.
    pub fn build(items: &[(u32, Aabb)]) -> Self {
        let mut t = Tree {
            bounds: Vec::new(),
            children: Vec::new(),
            root: EMPTY,
        };
        if items.is_empty() {
            return t;
        }
        assert!(
            items.iter().all(|(id, _)| id & LEAF_BIT == 0),
            "tree leaf id 최상위 비트는 예약됨"
        );
        // centroid를 **한 번만** 계산해 둔다. 전에는 비교 함수 안에서 계산해
        // 비교마다 두 번씩 다시 구했다(정렬 비교 횟수 × 2회).
        let centroids: Vec<Vec3> = items.iter().map(|(_, b)| (b.min + b.max) * 0.5).collect();
        let mut idx: Vec<usize> = (0..items.len()).collect();
        t.root = t.build_rec(items, &centroids, &mut idx);
        t
    }

    fn build_rec(&mut self, items: &[(u32, Aabb)], cen: &[Vec3], idx: &mut [usize]) -> u32 {
        if idx.len() == 1 {
            return items[idx[0]].0 | LEAF_BIT;
        }
        let node = self.children.len();
        assert!(node < EMPTY as usize, "tree node index capacity exceeded");
        self.children.push([EMPTY; 4]);
        self.bounds.push(soa_bounds::AabbLanes::empty());

        // 4그룹: 최장축 중앙값 분할 → 각 절반을 그 절반의 최장축으로 재분할.
        // **제자리 분할**이라 그룹당 Vec을 만들지 않는다 — 전에는 내부 노드마다 힙 할당이
        // 4번이었고, 3000 바디면 스텝당 4천 번이었다.
        let half = split2(cen, idx);
        let (lo, hi) = idx.split_at_mut(half);
        let lo_half = split2(cen, lo);
        let hi_half = split2(cen, hi);
        let (a, b) = lo.split_at_mut(lo_half);
        let (c, d) = hi.split_at_mut(hi_half);
        for (lane, g) in [a, b, c, d].into_iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            // 바운드는 **재귀 전에** 구한다 — 재귀가 g를 재배열해도 합집합은 불변이지만,
            // 빌림을 겹치지 않게 하려면 여기서 끝내는 편이 단순하다.
            let bnd = g
                .iter()
                .map(|&i| items[i].1)
                .reduce(|x, y| x.merge(y))
                .unwrap();
            let child = self.build_rec(items, cen, g);
            self.children[node][lane] = child;
            self.bounds[node].set_lane(lane, bnd);
        }
        node as u32
    }

    /// 질의 AABB와 겹칠 **가능성이 있는** 리프 id들(superset). 정렬·정밀 판정은 호출자 몫.
    /// 스택을 매번 할당하지 않도록 스크래치를 받는 버전은 [`Tree::query_candidates_in`].
    pub fn query_candidates(&self, q: Aabb, out: &mut Vec<u32>) {
        let mut stack = Vec::new();
        self.query_candidates_in(q, out, &mut stack);
    }

    /// 순회 스택을 호출자가 보관하는 버전. 브로드페이즈는 바디마다 질의하므로
    /// `vec![root]` 한 줄이 스텝당 수천 번의 힙 할당이 된다.
    pub fn query_candidates_in(&self, q: Aabb, out: &mut Vec<u32>, stack: &mut Vec<u32>) {
        let root = self.root;
        if root == EMPTY {
            return;
        }
        if let Child::Leaf(id) = decode(root) {
            out.push(id);
            return;
        }
        let qmin = Vec3x4::splat(q.min);
        let qmax = Vec3x4::splat(q.max);
        stack.clear();
        stack.push(root);
        while let Some(n) = stack.pop() {
            let n = n as usize;
            let mask = self.bounds[n].overlaps_mask(qmin, qmax).to_array();
            for (hit, &child) in mask.iter().zip(&self.children[n]) {
                if *hit != 0.0 {
                    match decode(child) {
                        Child::Node(c) => stack.push(c),
                        Child::Leaf(id) => out.push(id),
                        Child::None => {}
                    }
                }
            }
        }
    }
}

/// centroid 최장축 중앙값으로 제자리 분할하고 분할점을 돌려준다.
///
/// `select_nth_unstable_by`는 O(n)이고, 분할 양쪽의 내부 순서는 규정되지 않는다 —
/// 트리 위상이 결과에 영향을 주지 않으므로(위 `build` 주석) 그래도 된다.
fn split2(cen: &[Vec3], idx: &mut [usize]) -> usize {
    if idx.len() <= 1 {
        return idx.len();
    }
    let mut lo = cen[idx[0]];
    let mut hi = lo;
    for &i in idx.iter().skip(1) {
        let c = cen[i];
        lo = Vec3::new(lo.x.min(c.x), lo.y.min(c.y), lo.z.min(c.z));
        hi = Vec3::new(hi.x.max(c.x), hi.y.max(c.y), hi.z.max(c.z));
    }
    let d = hi - lo;
    let axis = if d.x >= d.y && d.x >= d.z {
        0
    } else if d.y >= d.z {
        1
    } else {
        2
    };
    let key = |i: usize| {
        let c = cen[i];
        match axis {
            0 => c.x,
            1 => c.y,
            _ => c.z,
        }
    };
    let mid = idx.len() / 2;
    // total_cmp: NaN에도 전순서 유지 (설계 §8 — 비교/분기에 NaN 유입 차단, 패닉 금지)
    idx.select_nth_unstable_by(mid, |&a, &b| key(a).total_cmp(&key(b)));
    mid
}

/// SoA 노드 바운드 저장 (crate 내부용).
pub(crate) mod soa_bounds {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct AabbLanes {
        min: [Vec3; 4],
        max: [Vec3; 4],
    }

    impl AabbLanes {
        pub(crate) fn empty() -> Self {
            let inf = f32::INFINITY;
            Self {
                min: [Vec3::new(inf, inf, inf); 4],
                max: [Vec3::new(-inf, -inf, -inf); 4],
            }
        }

        pub(crate) fn set_lane(&mut self, lane: usize, b: Aabb) {
            self.min[lane] = b.min;
            self.max[lane] = b.max;
        }

        #[inline]
        pub(crate) fn overlaps_mask(&self, qmin: Vec3x4, qmax: Vec3x4) -> f32x4 {
            let bmin = Vec3x4::from_array(self.min);
            let bmax = Vec3x4::from_array(self.max);
            Aabbx4 {
                min: bmin,
                max: bmax,
            }
            .overlaps(qmin, qmax)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 리뷰 MAJOR 회귀: 최대 31비트 id가 EMPTY와 앨리어싱되면 리프가 소실된다.
    #[test]
    fn max_31_bit_leaf_id_does_not_alias_empty() {
        let id = 0x7fff_ffff;
        let aabb = Aabb::from_center(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let tree = Tree::build(&[(id, aabb)]);
        let mut out = Vec::new();
        tree.query_candidates(aabb, &mut out);
        assert_eq!(out, vec![id]);
    }

    /// 내부 노드 인코딩까지 통과하는지: 다수 아이템 속의 최대 id.
    #[test]
    fn max_id_survives_inner_node_encoding() {
        let mut items: Vec<(u32, Aabb)> = (0..16)
            .map(|i| {
                let c = Vec3::new(i as f32 * 3.0, 0.0, 0.0);
                (i, Aabb::from_center(c, Vec3::new(1.0, 1.0, 1.0)))
            })
            .collect();
        items.push((
            0x7fff_ffff,
            Aabb::from_center(Vec3::new(24.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        ));
        let tree = Tree::build(&items);
        let mut out = Vec::new();
        tree.query_candidates(
            Aabb::from_center(Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)),
            &mut out,
        );
        assert!(out.contains(&0x7fff_ffff), "{out:?}");
    }
}
