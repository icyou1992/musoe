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
    /// 결정적 빌드: (id, aabb) 목록에서 top-down 4분할. 같은 입력 → 같은 트리.
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
        let mut idx: Vec<usize> = (0..items.len()).collect();
        t.root = t.build_rec(items, &mut idx);
        t
    }

    fn build_rec(&mut self, items: &[(u32, Aabb)], idx: &mut [usize]) -> u32 {
        if idx.len() == 1 {
            return items[idx[0]].0 | LEAF_BIT;
        }
        // 4그룹: 최장축 2분할 → 각 절반을 그 절반의 최장축으로 재분할 (centroid 기준, id 동률 정렬)
        let groups = split4(items, idx);
        let node = self.children.len();
        assert!(node < EMPTY as usize, "tree node index capacity exceeded");
        self.children.push([EMPTY; 4]);
        self.bounds.push(soa_bounds::AabbLanes::empty());
        for (lane, g) in groups.into_iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            let mut g = g;
            let child = self.build_rec(items, &mut g);
            let b = g
                .iter()
                .map(|&i| items[i].1)
                .reduce(|a, x| a.merge(x))
                .unwrap();
            self.children[node][lane] = child;
            self.bounds[node].set_lane(lane, b);
        }
        node as u32
    }

    /// 질의 AABB와 겹칠 **가능성이 있는** 리프 id들(superset). 정렬·정밀 판정은 호출자 몫.
    pub fn query_candidates(&self, q: Aabb, out: &mut Vec<u32>) {
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
        let mut stack: Vec<u32> = vec![root];
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

/// centroid 최장축 기준 4분할(결정적: 좌표 동률은 id 오름차순).
fn split4(items: &[(u32, Aabb)], idx: &mut [usize]) -> [Vec<usize>; 4] {
    let half = split2(items, idx);
    let (lo, hi) = idx.split_at_mut(half);
    let lo_half = split2(items, lo);
    let hi_half = split2(items, hi);
    let (a, b) = lo.split_at(lo_half);
    let (c, d) = hi.split_at(hi_half);
    [a.to_vec(), b.to_vec(), c.to_vec(), d.to_vec()]
}

fn split2(items: &[(u32, Aabb)], idx: &mut [usize]) -> usize {
    if idx.len() <= 1 {
        return idx.len();
    }
    let centroid = |i: usize| {
        let b = items[i].1;
        (b.min + b.max) * 0.5
    };
    let mut lo = centroid(idx[0]);
    let mut hi = lo;
    for &i in idx.iter().skip(1) {
        let c = centroid(i);
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
        let c = centroid(i);
        match axis {
            0 => c.x,
            1 => c.y,
            _ => c.z,
        }
    };
    idx.sort_by(|&a, &b| {
        key(a)
            .partial_cmp(&key(b))
            .unwrap()
            .then_with(|| items[a].0.cmp(&items[b].0))
    });
    idx.len() / 2
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
