//! 4-ary AABB 트리 — 브로드페이즈 가속 구조 (설계 §4.1).
//!
//! 계약: **트리는 성능 전용이다.** 후보는 superset이면 되고, 최종 페어 집합·쿼리 결과는
//! 정밀 AABB 판정 + canonical 정렬로 확정된다(pairs 모듈). 노드 바운드는 SoA `Aabbx4` —
//! 순회 1스텝 = SIMD128 비교 1회. [근거: Jolt 쿼드트리]
//! v1 범위: 결정적 top-down 빌드 + overlap 쿼리. widen-only 증분·백그라운드 재빌드는 M2+.

use crate::Aabb;
use musoe_math::{Aabbx4, Vec3, Vec3x4};
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

/// `Clone`/`Debug`는 **메시가 트리를 소유하기 때문에** 필요하다(`TriMeshData` → `Shape` →
/// 바디 삽입에서 셰이프가 복제된다). 트리는 파생물이지만 복제 때 다시 빌드하면 큰 메시에서
/// 삽입이 O(n log n)이 되므로 그냥 복사한다.
#[derive(Clone, Debug)]
pub struct Tree {
    /// 노드별 자식 4개의 바운드(SoA). 빈 레인은 반전 AABB(항상 불일치).
    bounds: Vec<soa_bounds::AabbLanes>,
    children: Vec<[u32; 4]>,
    /// 자식 서브트리의 **최대 리프 id**. 빈 자식은 0.
    ///
    /// 동적-동적 페어를 뽑을 때 `(i, j)`를 i의 질의와 j의 질의에서 두 번 만나 절반을
    /// `other <= id`로 버린다 — 버릴 후보를 **순회 단계에서** 잘라내려고 둔다.
    /// 서브트리 최대 id가 질의 id 이하면 그 아래에는 채택될 후보가 하나도 없다.
    /// 페어 집합은 그대로다(버릴 것만 안 만든다).
    child_max_id: Vec<[u32; 4]>,
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
            child_max_id: Vec::new(),
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
        self.child_max_id.push([0; 4]);
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
            let max_id = g.iter().map(|&i| items[i].0).max().unwrap();
            let child = self.build_rec(items, cen, g);
            self.children[node][lane] = child;
            self.child_max_id[node][lane] = max_id;
            self.bounds[node].set_lane(lane, bnd);
        }
        node as u32
    }

    /// **바운드만 다시 계산한다 — 위상은 그대로.** 아이템 AABB가 바뀐 뒤 호출하면
    /// 모든 노드 바운드가 현재 아이템의 정확한(느슨하지 않은) 합집합이 된다.
    ///
    /// 위상을 안 바꾸는 것이 합법인 근거가 §4.1이다: 페어 집합은 트리 후보에 정밀 AABB
    /// 판정을 적용한 결과로 정의되므로, 위상은 성능에만 영향을 준다. 낡은 위상은 후보를
    /// 더 만들 수는 있어도 놓칠 수는 없다 — 바운드가 정확한 한.
    ///
    /// `build_rec`는 부모 노드를 자식보다 먼저 push하므로(자식 인덱스 > 부모) 역순 순회
    /// 한 번이면 자식 바운드가 항상 부모보다 먼저 확정된다. O(노드) = O(n/3).
    /// `aabb_of`는 리프 id → 현재 AABB (호출자가 슬롯 색인으로 O(1) 제공).
    pub fn refit(&mut self, aabb_of: impl Fn(u32) -> Aabb) {
        for node in (0..self.children.len()).rev() {
            for lane in 0..4 {
                let b = match decode(self.children[node][lane]) {
                    Child::None => continue, // empty() 초기값(역전 AABB) 유지 — 항상 불일치
                    Child::Leaf(id) => aabb_of(id),
                    Child::Node(c) => self.bounds[c as usize].union_all(),
                };
                self.bounds[node].set_lane(lane, b);
            }
        }
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
        self.query_above_in(q, None, out, stack)
    }

    /// **힙 없이** 도는 순회 — 후보를 모으지 않고 콜백으로 흘린다.
    ///
    /// 스텝 안에서 도는 경로(메시 접촉·CCD 캐스트)는 "스텝 중 힙 할당 0" 계약 아래 있어서
    /// `Vec` 스택도 후보 벡터도 쓸 수 없다. 스택은 **고정 배열**이고, 넘치면 `false`를
    /// 돌려준다 — 조용히 후보를 빠뜨리는 대신 호출자가 포화로 처리하게 한다
    /// (조용한 누락은 물체가 지형을 통과하는 결함이 된다).
    ///
    /// 깊이 상한: 4-ary 트리라 리프 n개면 깊이 ≈ log4(n)이고, DFS 스택에는 형제 노드가
    /// 최대 3개씩 쌓이므로 3·log4(n) + 1이 상한이다. n = 2^20이면 31이라 64면 충분하다.
    /// 콜백이 `false`를 돌려주면 순회를 즉시 멈춘다(예산 소진).
    pub(crate) fn for_each_candidate(&self, q: Aabb, mut f: impl FnMut(u32) -> bool) -> bool {
        const STACK_CAP: usize = 64;
        let root = self.root;
        if root == EMPTY {
            return true;
        }
        if let Child::Leaf(id) = decode(root) {
            f(id);
            return true;
        }
        let qmin = Vec3x4::splat(q.min);
        let qmax = Vec3x4::splat(q.max);
        let mut stack = [0u32; STACK_CAP];
        let mut top = 0usize;
        stack[top] = root;
        top += 1;
        while top > 0 {
            top -= 1;
            let n = stack[top] as usize;
            let mask = self.bounds[n].overlaps_mask(qmin, qmax).to_array();
            for (hit, &child) in mask.iter().zip(&self.children[n]) {
                if *hit == 0.0 {
                    continue;
                }
                match decode(child) {
                    Child::Node(c) => {
                        if top == STACK_CAP {
                            return false; // 포화 — 호출자가 오류로 처리한다
                        }
                        stack[top] = c;
                        top += 1;
                    }
                    Child::Leaf(id) => {
                        if !f(id) {
                            return true; // 호출자가 멈추라고 했다(예산)
                        }
                    }
                    Child::None => {}
                }
            }
        }
        true
    }

    /// `above`가 있으면 **id가 그 값 이하인 후보는 만들지 않는다**.
    ///
    /// 동적-동적 페어는 `(i, j)`를 i의 질의와 j의 질의에서 두 번 만나 절반을 버린다.
    /// 서브트리의 최대 리프 id가 `above` 이하면 그 아래에는 채택될 후보가 없으므로
    /// 순회 자체를 건너뛴다 — **페어 집합은 동일하고**(버릴 것만 안 만든다)
    /// 순회와 후보 벡터가 대략 절반이 된다.
    /// 동적-정적 질의는 id 공간이 달라 이 가지치기를 쓰면 안 된다(`None`을 넘긴다).
    pub fn query_above_in(
        &self,
        q: Aabb,
        above: Option<u32>,
        out: &mut Vec<u32>,
        stack: &mut Vec<u32>,
    ) {
        let root = self.root;
        if root == EMPTY {
            return;
        }
        if let Child::Leaf(id) = decode(root) {
            if above.is_none_or(|a| id > a) {
                out.push(id);
            }
            return;
        }
        let qmin = Vec3x4::splat(q.min);
        let qmax = Vec3x4::splat(q.max);
        stack.clear();
        stack.push(root);
        while let Some(n) = stack.pop() {
            let n = n as usize;
            let mask = self.bounds[n].overlaps_mask(qmin, qmax).to_array();
            for (lane, (hit, &child)) in mask.iter().zip(&self.children[n]).enumerate() {
                if *hit == 0.0 {
                    continue;
                }
                if let Some(a) = above {
                    if self.child_max_id[n][lane] <= a {
                        continue;
                    }
                }
                match decode(child) {
                    Child::Node(c) => stack.push(c),
                    Child::Leaf(id) => out.push(id),
                    Child::None => {}
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

        /// 네 레인 바운드의 합집합 (빈 레인은 역전 AABB라 merge에 중립).
        pub(crate) fn union_all(&self) -> Aabb {
            let mut b = Aabb {
                min: self.min[0],
                max: self.max[0],
            };
            for lane in 1..4 {
                b = b.merge(Aabb {
                    min: self.min[lane],
                    max: self.max[lane],
                });
            }
            b
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
