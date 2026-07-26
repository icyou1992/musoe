//! Canonical 페어 집합 — 브로드페이즈의 정본 출력 (설계 §4.1).
//!
//! 계약: 최종 페어 집합 = 트리 후보(superset)에 **정밀 AABB 겹침 판정**을 적용하고
//! (min(a,b), max(a,b))로 정규화해 **정렬·중복 제거**한 결과. 트리 토폴로지·재빌드 타이밍은
//! 이 집합에 영향을 줄 수 없다 — L0 결정론의 초석.

use crate::{Aabb, Tree};
use maru_math::Vec3;

/// 듀얼 트리 브로드페이즈: 동적/정적 아이템과 각 트리에서 canonical 페어 집합 산출.
/// 반환: 정렬된 (동적 id, 상대 id) — 동적-동적은 (min,max), 동적-정적은 (dyn, static|FLAG).
pub struct Broadphase {
    pub dynamic_items: Vec<(u32, Aabb)>,
    pub static_items: Vec<(u32, Aabb)>,
    dyn_tree: Tree,
    static_tree: Tree,
}

/// 페어의 상대가 정적임을 표시하는 비트 (id 공간 분리 — canonical 키의 일부).
/// 브로드페이즈 페어의 **절대 상한**. 1M 페어 = 8MB 벡터 + 후속 constraints/키/HashSet.
/// wasm 선형 메모리에서 통제된 오류가 allocator abort보다 먼저 오게 하는 마지막 방어선이다.
pub const MAX_PAIRS: usize = 1 << 20;

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
        self.canonical_pairs_bounded(usize::MAX)
            .expect("pair budget exceeded")
    }

    /// 페어 예산이 있는 버전 — 초과하면 **더 쌓지 않고** None을 돌려준다(리뷰 BLOCKER).
    /// 같은 자리에 겹친 1만 개 바디는 5천만 페어를 만든다: 입력은 수 MB로 자원 정책을
    /// 통과하지만 wasm에서는 `try_step`이 오류를 돌려주기 전에 allocator가 먼저 죽는다.
    pub fn canonical_pairs_bounded(&self, budget: usize) -> Option<Vec<(u32, u32)>> {
        // 예산 자체를 **절대 상한**으로 한 번 더 자른다: 계수형 예산은 메모리 예산이 아니다
        // (계수 4096 × 5000 바디 = 2천만 페어가 "예산 안"이지만 벡터만 160MB다).
        let budget = budget.min(MAX_PAIRS);
        // **예산 전체를 미리 fallible로 잡는다**. 일부만 잡고 나머지를 `push`의 암묵
        // 재할당에 맡기면 그 재할당이 abort/trap이 된다 — 오류 ABI가 거기서 깨진다.
        // 예산은 이미 MAX_PAIRS(1M = 8MB)로 묶여 있으므로 선점 비용도 유계다.
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        pairs.try_reserve_exact(budget).ok()?;
        let mut cand: Vec<u32> = Vec::new();

        // **슬롯 색인 배열**로 상대 AABB를 찾는다. 전에는 `HashMap<u32, Aabb>` 두 개를
        // 매 스텝 새로 채우고 **후보마다** 조회했다 — 구 더미 3000에서 스텝당 10만 회를
        // 넘는 SipHash다. id는 호출부가 부여하는 조밀한 인덱스이고 아이템 수로 유계이므로
        // (max_id+1) 크기 배열이 곧 완전한 색인이다.
        //
        // 동적과 정적은 **별개 id 공간**이다(같은 정수가 양쪽에서 서로 다른 물체를 가리킬 수
        // 있다 — STATIC_BIT이 존재하는 이유가 그것이다). 배열을 하나로 합치면 안 된다.
        let slots = |items: &[(u32, Aabb)]| -> Option<Vec<Aabb>> {
            let n = items
                .iter()
                .map(|(id, _)| *id)
                .max()
                .map_or(0, |m| m as usize + 1);
            let mut v: Vec<Aabb> = Vec::new();
            v.try_reserve_exact(n).ok()?;
            // 빈 슬롯은 어떤 질의와도 겹치지 않는 역전 AABB — 조밀하지 않은 id를 받아도
            // 페어를 만들지 않는다.
            v.resize(
                n,
                Aabb::new(
                    Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
                    Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
                ),
            );
            for &(id, a) in items {
                v[id as usize] = a;
            }
            Some(v)
        };
        let dyn_by_slot = slots(&self.dynamic_items)?;
        let static_by_slot = slots(&self.static_items)?;

        for &(id, aabb) in &self.dynamic_items {
            // 동적-동적 (self-pair 제거, canonical (min,max)로 절반만 채택해 중복 원천 차단)
            cand.clear();
            self.dyn_tree.query_candidates(aabb, &mut cand);
            for &other in &cand {
                if other <= id {
                    continue; // self + 역방향 중복 제거
                }
                if aabb.overlaps(dyn_by_slot[other as usize]) {
                    if pairs.len() >= budget {
                        return None;
                    }
                    pairs.push((id, other));
                }
            }
            // 동적-정적
            cand.clear();
            self.static_tree.query_candidates(aabb, &mut cand);
            for &s in &cand {
                if aabb.overlaps(static_by_slot[s as usize]) {
                    if pairs.len() >= budget {
                        return None;
                    }
                    pairs.push((id, s | STATIC_BIT));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        Some(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
