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
    /// 슬롯 색인 캐시(id → AABB, 빈 슬롯 = 역전 AABB). 예전에는 증분 스텝마다 이 배열을
    /// **세 번 새로 할당**했다(refit용 1 + 페어 판정용 2 — 3000바디에서 스텝당 ~216KB
    /// 할당·초기화). 정적 캐시는 재빌드까지 불변이고 동적 캐시는 refresh가 변경분만
    /// 덮으므로, 한 번 만들어 유지한다. `refresh_dynamic`이 (재)구축·갱신의 정본이다.
    dyn_by_slot: Vec<Aabb>,
    static_by_slot: Vec<Aabb>,
}

/// 페어의 상대가 정적임을 표시하는 비트 (id 공간 분리 — canonical 키의 일부).
/// 브로드페이즈 페어의 **절대 상한**. 1M 페어 = 8MB 벡터 + 후속 constraints/키/HashSet.
/// wasm 선형 메모리에서 통제된 오류가 allocator abort보다 먼저 오게 하는 마지막 방어선이다.
pub const MAX_PAIRS: usize = 1 << 20;

pub const STATIC_BIT: u32 = 1 << 31;

/// 증분 갱신 결과: (병합된 canonical 페어 전체, fresh = R 접촉 페어만).
/// 증분 병합 성공 시 반환 — fresh(R 접촉 페어)만. 병합 전체는 호출자의 `out` 버퍼에 채운다.
pub type IncrementalPairs = Vec<(u32, u32)>;

/// `incremental_pairs_bounded`의 결과.
pub struct IncrementalOut {
    /// R이 만든 새 페어 (활성 끝점을 가질 수 있는 페어의 정확한 상집합).
    pub fresh: IncrementalPairs,
    /// **페어 목록이 직전과 완전히 같다** — 병합을 아예 돌리지 않았고 `out`은 비어 있다.
    /// 호출자는 `prev`를 그대로 이번 스텝의 목록으로 써야 한다.
    pub unchanged: bool,
}

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
            // 캐시는 증분 경로 첫 사용 시(fallible) 구축한다 — build는 무할당 계약 유지.
            dyn_by_slot: Vec::new(),
            static_by_slot: Vec::new(),
        }
    }

    fn empty_aabb() -> Aabb {
        Aabb::new(
            Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        )
    }

    /// 슬롯 색인 캐시를 fallible하게 (재)구축한다. 이미 크기가 맞으면 무연산.
    fn ensure_slot_caches(&mut self) -> bool {
        let need = |items: &[(u32, Aabb)]| {
            items
                .iter()
                .map(|(id, _)| *id)
                .max()
                .map_or(0, |m| m as usize + 1)
        };
        let fill = |cache: &mut Vec<Aabb>, items: &[(u32, Aabb)]| -> bool {
            let n = need(items);
            if cache.len() != n {
                cache.clear();
                if cache.try_reserve_exact(n).is_err() {
                    return false;
                }
                cache.resize(n, Self::empty_aabb());
                for &(id, a) in items {
                    cache[id as usize] = a;
                }
            }
            true
        };
        // 정적은 재빌드까지 불변 — 크기 일치면 내용도 최신이다. 동적은 refresh가 변경분을
        // 즉시 덮으므로 마찬가지다(추가/제거는 재빌드 → 새 Broadphase → 빈 캐시부터).
        fill(&mut self.dyn_by_slot, &self.dynamic_items)
            && fill(&mut self.static_by_slot, &self.static_items)
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
        // 트리 순회 스택도 스텝 전체에서 재사용한다(질의마다 할당하지 않는다).
        let mut stack: Vec<u32> = Vec::new();

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

        // **입력이 id 오름차순이면 전역 정렬이 필요 없다.** 바깥 루프가 id 순으로 돌므로
        // 페어는 이미 id 블록 단위로 오름차순이고, 블록 안만 정렬하면 전체가 정렬된다.
        // 33k 페어 전역 정렬(≈50만 비교)이 블록당 ~11개 정렬(≈11만 비교)로 줄어든다.
        // 정렬돼 있지 않으면(공개 API는 임의 순서를 허용한다) 기존대로 전역 정렬한다.
        let blocked = self.dynamic_items.windows(2).all(|w| w[0].0 < w[1].0);

        for &(id, aabb) in &self.dynamic_items {
            let block_start = pairs.len();
            // 동적-동적 (self-pair 제거, canonical (min,max)로 절반만 채택해 중복 원천 차단)
            cand.clear();
            // `above=id`로 순회 단계에서 잘라낸다 — 아래 `other <= id` 판정과 같은 결과지만
            // 버릴 후보를 애초에 만들지 않는다(페어 집합 불변, 순회·후보 벡터가 절반).
            self.dyn_tree
                .query_above_in(aabb, Some(id), &mut cand, &mut stack);
            for &other in &cand {
                debug_assert!(other > id, "가지치기가 채택 가능한 후보만 남겨야 한다");
                if aabb.overlaps(dyn_by_slot[other as usize]) {
                    if pairs.len() >= budget {
                        return None;
                    }
                    pairs.push((id, other));
                }
            }
            // 동적-정적
            cand.clear();
            self.static_tree
                .query_candidates_in(aabb, &mut cand, &mut stack);
            for &s in &cand {
                if aabb.overlaps(static_by_slot[s as usize]) {
                    if pairs.len() >= budget {
                        return None;
                    }
                    pairs.push((id, s | STATIC_BIT));
                }
            }
            if blocked {
                pairs[block_start..].sort_unstable();
            }
        }
        if !blocked {
            pairs.sort_unstable();
        }
        debug_assert!(
            pairs.windows(2).all(|w| w[0] <= w[1]),
            "canonical 페어 집합은 전순서로 정렬돼 있어야 한다"
        );
        pairs.dedup();
        Some(pairs)
    }
}

impl Broadphase {
    /// static 트리 후보 질의 (superset — 정밀 판정은 호출자 몫). CCD linear cast가
    /// 이동 sweep AABB로 static 후보를 뽑을 때 쓴다.
    pub fn query_static(&self, q: Aabb, out: &mut Vec<u32>, stack: &mut Vec<u32>) {
        self.static_tree.query_candidates_in(q, out, stack);
    }

    /// **증분 갱신 1/2**: 움직인 동적 아이템의 AABB를 교체하고 dyn 트리 바운드를 리핏한다.
    /// 위상은 그대로 — §4.1(페어 집합은 트리 비의존)이 이것을 합법으로 만든다.
    /// `updates`의 id는 build 때 존재했어야 한다(추가/제거는 구조 변경 → 재빌드가 정답).
    ///
    /// 반환 `false` = 슬롯 캐시 할당 실패(첫 증분 스텝만 할당한다) — 호출자는 구조를
    /// 버리고 재빌드 폴백으로 간다. 실패해도 아이템·캐시는 일관 상태다(부분 갱신 없음).
    #[must_use]
    pub fn refresh_dynamic(&mut self, updates: &[(u32, Aabb)]) -> bool {
        if !self.ensure_slot_caches() {
            return false;
        }
        for &(id, aabb) in updates {
            // build가 아이템을 id 순으로 받는 것이 보장은 아니지만(공개 API는 임의 순서),
            // 증분 경로의 호출자(world)는 항상 슬롯 오름차순으로 만든다 — 정렬이면 이분,
            // 아니면 선형 폴백.
            let i = self
                .dynamic_items
                .binary_search_by_key(&id, |e| e.0)
                .ok()
                .or_else(|| self.dynamic_items.iter().position(|e| e.0 == id))
                .expect("refresh_dynamic: unknown id — 추가/제거는 재빌드로");
            self.dynamic_items[i].1 = aabb;
            self.dyn_by_slot[id as usize] = aabb;
        }
        let by_slot = &self.dyn_by_slot;
        self.dyn_tree.refit(|id| by_slot[id as usize]);
        true
    }

    /// **증분 갱신 2/2**: 움직인 집합 R의 페어만 다시 만들어 직전 canonical 페어와 병합한다.
    ///
    /// 결과는 `canonical_pairs_bounded`를 처음부터 돌린 것과 **정확히 같아야 한다**
    /// (아래 속성 테스트가 잠근다). 등가 논증:
    /// - R에 안 닿는 페어: 양끝 AABB 불변 → 페어 자격 불변 → `prev`에서 그대로 승계.
    /// - R에 닿는 페어: `prev`에서 전부 제거하고, R의 각 원소를 리핏된 트리(전 리프가
    ///   현재값)에 질의해 정밀 판정으로 재생성 — 트리 후보는 superset이므로 누락 없음.
    ///   R×R 페어는 양쪽 질의에서 두 번 나와도 정렬·dedup이 지운다.
    /// - `prev_minus`(R 비접촉)와 `fresh`(전부 R 접촉)는 서로소이고 각자 정렬돼 있어
    ///   병합 결과가 전순서 canonical이다.
    ///
    /// `moved_mask[id]`는 R 소속 판정(호출자가 슬롯 크기로 준비), `moved`는 R의 id 목록.
    /// 반환 = **fresh**(R 접촉 페어만) — 병합된 canonical 전체는 `out`에 채운다(핑퐁 버퍼).
    /// fresh는 "활성 끝점을 가질 수
    /// 있는 페어"의 정확한 상집합이라(활성 ⊆ R일 때) 호출자가 wake/narrow 순회를 이 목록으로
    /// 제한하는 근거가 된다.
    pub fn incremental_pairs_bounded(
        &self,
        moved: &[u32],
        moved_mask: &[bool],
        prev: &[(u32, u32)],
        // `prev_pair_count`: 슬롯별 "직전 목록에서 그 바디가 낀 페어 수"(길이 부족은 0으로).
        // R 전체에서 0이고 `fresh`도 비면 목록이 직전과 완전히 같다 — 아래 fast path.
        prev_pair_count: &[u32],
        budget: usize,
        out: &mut Vec<(u32, u32)>,
    ) -> Option<IncrementalOut> {
        let budget = budget.min(MAX_PAIRS);
        let in_r = |id: u32| moved_mask.get(id as usize).copied().unwrap_or(false);

        // 슬롯 색인 (정밀 판정용 — canonical_pairs와 동일 규칙)
        let slots = |items: &[(u32, Aabb)]| -> Option<Vec<Aabb>> {
            let n = items
                .iter()
                .map(|(id, _)| *id)
                .max()
                .map_or(0, |m| m as usize + 1);
            let mut v: Vec<Aabb> = Vec::new();
            v.try_reserve_exact(n).ok()?;
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
        // 슬롯 캐시가 준비돼 있으면(정상 경로 — 호출 전 `refresh_dynamic` 필수) 그대로
        // 쓴다. 캐시 없이 직접 호출된 경우(공개 API 오용·일부 테스트)만 로컬 구축으로
        // 폴백한다 — 빈 캐시를 그대로 쓰면 전 슬롯이 역전 AABB라 페어가 조용히 0이 된다.
        let (dyn_local, static_local);
        let (dyn_by_slot, static_by_slot): (&[Aabb], &[Aabb]) =
            if !self.dyn_by_slot.is_empty() || self.dynamic_items.is_empty() {
                (&self.dyn_by_slot, &self.static_by_slot)
            } else {
                dyn_local = slots(&self.dynamic_items)?;
                static_local = slots(&self.static_items)?;
                (&dyn_local, &static_local)
            };

        // R의 새 페어 — 리핏된 트리에 질의, 정밀 판정, canonical 키.
        let mut fresh: Vec<(u32, u32)> = Vec::new();
        let mut cand: Vec<u32> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        for &r in moved {
            let q = dyn_by_slot[r as usize];
            cand.clear();
            self.dyn_tree.query_candidates_in(q, &mut cand, &mut stack);
            for &other in &cand {
                if other == r {
                    continue;
                }
                // R×R은 양쪽 질의에서 만나므로 **낮은 id 쪽만** 채택한다(교차검수
                // rev_47788f13 MAJOR): 중복을 쌓고 dedup하면 고유 페어 수는 예산 안인데
                // raw 길이가 먼저 예산에 닿아, 전체 재빌드면 성공할 정상 장면을 증분
                // 경로만 ResourceLimit로 거절한다 — 관측 가능한 경로 분기다.
                // R×비R은 R 쪽만 질의하므로 그대로. 결과적으로 fresh는 구성상 중복이 없다.
                if q.overlaps(dyn_by_slot[other as usize]) && (!in_r(other) || r < other) {
                    if fresh.len() >= budget {
                        return None;
                    }
                    fresh.push((r.min(other), r.max(other)));
                }
            }
            cand.clear();
            self.static_tree
                .query_candidates_in(q, &mut cand, &mut stack);
            for &s in &cand {
                if q.overlaps(static_by_slot[s as usize]) {
                    if fresh.len() >= budget {
                        return None;
                    }
                    fresh.push((r, s | STATIC_BIT));
                }
            }
        }
        fresh.sort_unstable();
        fresh.dedup();

        // **목록이 그대로면 병합을 돌리지 않는다.**
        //
        // 병합 루프는 `prev` 전체를 훑는다 — 정착한 더미 위로 소수가 떨어지는 장면에서는
        // R이 5개인데 prev가 24,810개라 스텝의 58%가 여기서 나갔다(2026-07-31 실측:
        // 낙하체가 5개든 100개든 broad가 같고 페어 수에만 비례했다).
        //
        // 그런데 R이 직전 목록의 어떤 페어에도 끼지 않았고(=제거할 게 없다) `fresh`도 비면
        // (=추가할 게 없다) 결과는 `prev`와 **같은 집합·같은 순서**다. 그 판정을 O(|R|)로
        // 하려고 슬롯별 페어 수를 받는다. 공중에 떠 있는 낙하체가 정확히 이 경우다.
        //
        // 등가 논증: 병합은 `prev`에서 R을 건드리는 항만 빼고 `fresh`를 정렬 병합한다.
        // 뺄 항이 없고 넣을 항도 없으면 항등 사상이다.
        if fresh.is_empty()
            && moved
                .iter()
                .all(|&r| prev_pair_count.get(r as usize).copied().unwrap_or(0) == 0)
        {
            return Some(IncrementalOut {
                fresh,
                unchanged: true,
            });
        }

        // prev에서 R 접촉 페어 제거 (정렬 보존) + fresh와 병합. `out`은 호출자의 영속
        // 스크래치다 — 직전 스텝의 prev 버퍼와 핑퐁하면 워밍업 후 재할당이 없다
        // (예전에는 매 스텝 ~200KB를 새로 할당·해제했다).
        out.clear();
        out.try_reserve((prev.len() + fresh.len()).min(MAX_PAIRS))
            .ok()?;
        let mut fi = 0usize;
        for &(a, b_raw) in prev {
            let b_plain = b_raw & !STATIC_BIT;
            if in_r(a) || (b_raw & STATIC_BIT == 0 && in_r(b_plain)) {
                continue; // R 접촉 — fresh가 재생성 담당
            }
            while fi < fresh.len() && fresh[fi] < (a, b_raw) {
                if out.len() >= budget {
                    return None;
                }
                out.push(fresh[fi]);
                fi += 1;
            }
            if out.len() >= budget {
                return None;
            }
            out.push((a, b_raw));
        }
        while fi < fresh.len() {
            if out.len() >= budget {
                return None;
            }
            out.push(fresh[fi]);
            fi += 1;
        }
        debug_assert!(
            out.windows(2).all(|w| w[0] < w[1]),
            "증분 병합 결과는 중복 없는 전순서여야 한다"
        );
        Some(IncrementalOut {
            fresh,
            unchanged: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 슬롯별 페어 수 — 운영 경로(`world.rs`)와 같은 규약으로 만든다.
    /// 이걸 안 주고 빈 슬라이스를 넘기면 "전부 0"으로 읽혀 fast path가 거짓 양성이 된다.
    fn pair_counts(pairs: &[(u32, u32)]) -> Vec<u32> {
        let n = pairs
            .iter()
            .flat_map(|&(a, b)| [a, b & !STATIC_BIT])
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut c = vec![0u32; n];
        for &(a, b_raw) in pairs {
            c[a as usize] += 1;
            if b_raw & STATIC_BIT == 0 {
                c[b_raw as usize] += 1;
            }
        }
        c
    }

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

    /// 증분 경로의 정의 자체: **여러 라운드의 이동 후에도** refresh+incremental 결과가
    /// 그 시점 아이템으로 처음부터 재빌드한 canonical 페어와 비트 동일해야 한다.
    /// 라운드를 거듭할수록 트리 위상은 낡아간다 — 그래도 같아야 §4.1이 지켜진 것이다.
    #[test]
    fn incremental_pairs_equal_full_rebuild_across_rounds() {
        let mut rng = Rng(0xD1CE);
        let mut dynamic: Vec<(u32, Aabb)> = (0..150).map(|i| (i, rng.aabb(20.0, 1.5))).collect();
        let statics: Vec<(u32, Aabb)> = (0..40).map(|i| (i, rng.aabb(20.0, 3.0))).collect();
        let mut bp = Broadphase::build(dynamic.clone(), statics.clone());
        let mut pairs = bp.canonical_pairs();
        for round in 0..12 {
            // 라운드마다 다른 크기의 이동 집합 (1개 ~ 절반)
            let k = 1 + (round * 7) % 75;
            let mut moved: Vec<u32> = (0..k as u32)
                .map(|j| (j * 13 + round as u32) % 150)
                .collect();
            moved.sort_unstable();
            moved.dedup();
            let mut mask = vec![false; 150];
            let updates: Vec<(u32, Aabb)> = moved
                .iter()
                .map(|&id| {
                    mask[id as usize] = true;
                    let a = rng.aabb(20.0, 1.5);
                    dynamic[id as usize].1 = a;
                    (id, a)
                })
                .collect();
            assert!(bp.refresh_dynamic(&updates));
            let mut merged = Vec::new();
            let counts = pair_counts(&pairs);
            let fresh = bp
                .incremental_pairs_bounded(&moved, &mask, &pairs, &counts, usize::MAX, &mut merged)
                .expect("budget unbounded")
                .fresh;
            // fresh 계약: 병합 결과 중 R 접촉 페어와 정확히 일치해야 한다
            let want_fresh: Vec<(u32, u32)> = merged
                .iter()
                .copied()
                .filter(|&(a, b)| mask[a as usize] || (b & STATIC_BIT == 0 && mask[b as usize]))
                .collect();
            assert_eq!(
                fresh, want_fresh,
                "round {round}: fresh가 R 접촉 부분집합과 불일치"
            );
            pairs = merged;
            let want = Broadphase::build(dynamic.clone(), statics.clone()).canonical_pairs();
            assert_eq!(pairs, want, "round {round}: 증분 페어가 재빌드와 불일치");
        }
    }

    /// 회귀(rev_47788f13 MAJOR): 예산은 canonical 페어 수에 걸려야 한다 — R×R 중복이
    /// raw 길이를 부풀려 정상 장면을 증분 경로만 거절하면 재빌드와 관측이 갈린다.
    #[test]
    fn incremental_budget_counts_canonical_pairs_not_raw_duplicates() {
        // 전원이 서로 겹치는 클러스터 10개 — canonical 페어 = 45.
        let items: Vec<(u32, Aabb)> = (0..10)
            .map(|i| {
                (
                    i,
                    Aabb::from_center(
                        Vec3::new(i as f32 * 0.01, 0.0, 0.0),
                        Vec3::new(1.0, 1.0, 1.0),
                    ),
                )
            })
            .collect();
        let mut bp = Broadphase::build(items.clone(), Vec::new());
        let prev = bp.canonical_pairs();
        assert_eq!(prev.len(), 45);
        // 전원을 미세 이동(전원 R) — canonical 페어 수 불변.
        let updates: Vec<(u32, Aabb)> = (0..10u32)
            .map(|id| {
                (
                    id,
                    Aabb::from_center(
                        Vec3::new(id as f32 * 0.01 + 0.001, 0.0, 0.0),
                        Vec3::new(1.0, 1.0, 1.0),
                    ),
                )
            })
            .collect();
        assert!(bp.refresh_dynamic(&updates));
        let moved: Vec<u32> = (0..10).collect();
        let mask = vec![true; 10];
        // 예산 = 정확히 canonical 페어 수. 중복을 세면 90에 닿아 실패한다.
        let mut merged = Vec::new();
        let counts = pair_counts(&prev);
        let got = bp.incremental_pairs_bounded(&moved, &mask, &prev, &counts, 45, &mut merged);
        assert!(
            got.is_some(),
            "canonical 45페어 == 예산 45인데 증분이 ResourceLimit — raw 중복이 과금됐다"
        );
        assert_eq!(merged.len(), 45);
    }

    /// 리핏 후 질의도 superset 계약을 유지해야 한다 (트리 위상은 낡고 바운드만 현재).
    #[test]
    fn refit_query_remains_superset() {
        let mut rng = Rng(0xF17);
        let mut items: Vec<(u32, Aabb)> = (0..100).map(|i| (i, rng.aabb(20.0, 1.0))).collect();
        let bp_items = items.clone();
        let mut bp = Broadphase::build(bp_items, Vec::new());
        // 전 아이템을 새 위치로 — 위상과 완전히 어긋난 최악 케이스
        let updates: Vec<(u32, Aabb)> = (0..100u32)
            .map(|id| {
                let a = rng.aabb(20.0, 1.0);
                items[id as usize].1 = a;
                (id, a)
            })
            .collect();
        assert!(bp.refresh_dynamic(&updates));
        let moved: Vec<u32> = (0..100).collect();
        let mask = vec![true; 100];
        let mut got = Vec::new();
        bp.incremental_pairs_bounded(&moved, &mask, &[], &[], usize::MAX, &mut got)
            .unwrap();
        let want = brute_force(&items, &[]);
        assert_eq!(got, want, "전량 이동 후 증분 페어가 브루트포스와 불일치");
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
