//! Canonical 페어 집합 — 브로드페이즈의 정본 출력 (설계 §4.1).
//!
//! 계약: 최종 페어 집합 = 트리 후보(superset)에 **정밀 AABB 겹침 판정**을 적용하고
//! (min(a,b), max(a,b))로 정규화해 **정렬·중복 제거**한 결과. 트리 토폴로지·재빌드 타이밍은
//! 이 집합에 영향을 줄 수 없다 — L0 결정론의 초석.

use crate::{Aabb, Tree};

/// 듀얼 트리 브로드페이즈: 동적/정적 아이템과 각 트리에서 canonical 페어 집합 산출.
/// 반환: 정렬된 (동적 id, 상대 id) — 동적-동적은 (min,max), 동적-정적은 (dyn, static|FLAG).
pub struct Broadphase {
    pub dynamic_items: Vec<(u32, Aabb)>,
    pub static_items: Vec<(u32, Aabb)>,
    dyn_tree: Tree,
    static_tree: Tree,
}

/// 페어의 상대가 정적임을 표시하는 비트 (id 공간 분리 — canonical 키의 일부).
pub const STATIC_BIT: u32 = 1 << 31;

impl Broadphase {
    pub fn build(dynamic_items: Vec<(u32, Aabb)>, static_items: Vec<(u32, Aabb)>) -> Self {
        assert!(
            dynamic_items
                .iter()
                .chain(&static_items)
                .all(|(id, _)| id & STATIC_BIT == 0),
            "id 최상위 비트는 예약됨"
        );
        let dyn_tree = Tree::build(&dynamic_items);
        let static_tree = Tree::build(&static_items);
        Self {
            dynamic_items,
            static_items,
            dyn_tree,
            static_tree,
        }
    }

    /// canonical 페어 집합: 동적-동적 + 동적-정적. 정밀 판정 + 전순서 정렬 + dedup.
    pub fn canonical_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut cand: Vec<u32> = Vec::new();

        let static_lookup: std::collections::HashMap<u32, Aabb> =
            self.static_items.iter().copied().collect();
        let dyn_lookup: std::collections::HashMap<u32, Aabb> =
            self.dynamic_items.iter().copied().collect();

        for &(id, aabb) in &self.dynamic_items {
            // 동적-동적 (self-pair 제거, canonical (min,max)로 절반만 채택해 중복 원천 차단)
            cand.clear();
            self.dyn_tree.query_candidates(aabb, &mut cand);
            for &other in &cand {
                if other <= id {
                    continue; // self + 역방향 중복 제거
                }
                if aabb.overlaps(dyn_lookup[&other]) {
                    pairs.push((id, other));
                }
            }
            // 동적-정적
            cand.clear();
            self.static_tree.query_candidates(aabb, &mut cand);
            for &s in &cand {
                if aabb.overlaps(static_lookup[&s]) {
                    pairs.push((id, s | STATIC_BIT));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maru_math::Vec3;

    /// 고정 시드 xorshift — 재현 가능한 무작위 박스.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn aabb(&mut self, world: f32, size: f32) -> Aabb {
            let c = Vec3::new(
                (self.f() - 0.5) * world,
                (self.f() - 0.5) * world,
                (self.f() - 0.5) * world,
            );
            let h = Vec3::new(
                self.f() * size + 0.01,
                self.f() * size + 0.01,
                self.f() * size + 0.01,
            );
            Aabb::from_center(c, h)
        }
    }

    fn brute_force(dynamic: &[(u32, Aabb)], statics: &[(u32, Aabb)]) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        for i in 0..dynamic.len() {
            for j in (i + 1)..dynamic.len() {
                if dynamic[i].1.overlaps(dynamic[j].1) {
                    let (a, b) = (
                        dynamic[i].0.min(dynamic[j].0),
                        dynamic[i].0.max(dynamic[j].0),
                    );
                    pairs.push((a, b));
                }
            }
            for s in statics {
                if dynamic[i].1.overlaps(s.1) {
                    pairs.push((dynamic[i].0, s.0 | STATIC_BIT));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// M1 정확성 스위트의 핵심: 트리 경유 페어 집합 == 브루트포스 전수 비교.
    #[test]
    fn canonical_pairs_match_brute_force() {
        let mut rng = Rng(0xC0FFEE);
        for scale in [10.0f32, 30.0] {
            let dynamic: Vec<(u32, Aabb)> = (0..120).map(|i| (i, rng.aabb(scale, 1.5))).collect();
            let statics: Vec<(u32, Aabb)> = (0..60).map(|i| (i, rng.aabb(scale, 3.0))).collect();
            let bp = Broadphase::build(dynamic.clone(), statics.clone());
            let got = bp.canonical_pairs();
            let want = brute_force(&dynamic, &statics);
            assert_eq!(
                got, want,
                "scale={scale}: 트리 페어 집합이 브루트포스와 불일치"
            );
            assert!(!want.is_empty(), "테스트 씬이 비어 있으면 무의미");
        }
    }

    /// 트리 비의존 계약: 입력 순서를 섞어 다른 토폴로지의 트리를 만들어도 페어 집합 동일.
    #[test]
    fn pair_set_is_tree_topology_independent() {
        let mut rng = Rng(0xBADA55);
        let dynamic: Vec<(u32, Aabb)> = (0..80).map(|i| (i, rng.aabb(15.0, 2.0))).collect();
        let statics: Vec<(u32, Aabb)> = (0..40).map(|i| (i, rng.aabb(15.0, 2.0))).collect();

        let mut shuffled_dyn = dynamic.clone();
        shuffled_dyn.reverse();
        let mut shuffled_static = statics.clone();
        shuffled_static.reverse();

        let a = Broadphase::build(dynamic, statics).canonical_pairs();
        let b = Broadphase::build(shuffled_dyn, shuffled_static).canonical_pairs();
        assert_eq!(a, b);
    }

    #[test]
    fn query_candidates_is_superset() {
        let mut rng = Rng(0x5EED);
        let items: Vec<(u32, Aabb)> = (0..100).map(|i| (i, rng.aabb(20.0, 1.0))).collect();
        let tree = Tree::build(&items);
        let q = rng.aabb(20.0, 5.0);
        let mut cand = Vec::new();
        tree.query_candidates(q, &mut cand);
        // 실제 겹치는 모든 아이템이 후보에 포함돼야 함 (superset 계약)
        for (id, aabb) in &items {
            if q.overlaps(*aabb) {
                assert!(cand.contains(id), "id={id} 누락 — superset 계약 위반");
            }
        }
    }
}
