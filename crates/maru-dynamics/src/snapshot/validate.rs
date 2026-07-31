//! 복원 **구조 검증** — 읽어들인 값들이 서로 모순되지 않는지.
//!
//! 바이트 층(`io`)이 "형식이 맞는가"를, 이 층이 "그 값들이 월드로서 말이 되는가"를 본다.
//! 신뢰하지 않는 스냅샷은 형식이 멀쩡해도 내용이 모순일 수 있다.

use super::io::*;
use crate::body::{BodyHandle, BodyType};
use crate::command::{Command, PendingCommand};
use maru_math::Vec3;

use crate::{Body, Joint};

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_structure(
    bodies: &[Body],
    alive: &[bool],
    oob: &[bool],
    reserved: &[bool],
    generations: &[u32],
    asleep: &[bool],
    island: &[u32],
    joints: &[Joint],
    commands: &[PendingCommand],
    cache: &crate::world::IntMap<
        (u32, u32),
        Vec<(maru_collide::FeatureId, crate::solver::CachedImpulse)>,
    >,
    touching: &crate::world::IntSet<(u32, u32)>,
) -> Result<(), SnapshotError> {
    const STATIC_BIT: u32 = 1 << 31;
    let nb = bodies.len();
    // **OOB는 아일랜드 대표가 될 수 없다**(리뷰 PARTIAL): 라이브는 union에서 제외하는데
    // 리더가 이를 허용하면 조작 스냅샷이 정상 바디의 대표를 OOB로 두어 wake 그룹을
    // 임의로 연결할 수 있다.
    let live_dynamic = |i: usize| i < nb && alive[i] && !oob[i] && bodies[i].is_dynamic();

    for i in 0..nb {
        // 슬롯은 살아있거나, 예약이거나, 비어있다 — 둘을 겸할 수 없다
        if alive[i] && reserved[i] {
            return Err(SnapshotError::InconsistentState);
        }
        // 잠듦은 살아있는 dynamic만의 상태다
        if asleep[i] && !live_dynamic(i) {
            return Err(SnapshotError::InconsistentState);
        }
        // **잠든 바디의 속도는 0이어야 한다** — island sleep이 재우며 0으로 만들고, 모든
        // 변이 API는 wake를 거치므로 잠든 채 속도가 있는 상태는 라이브로 도달할 수 없다
        // (OOB 인접 조인트 임펄스 0 강제와 같은 계열). 이 보장이 스텝 말 위생 패스
        // (denormal flush·속도 클램프·상태 포화)가 잠든 바디를 건너뛰는 것을 정확 등가로
        // 만든다: 복원 직후에도 잠든 바디는 위생 패스가 손댈 게 없는 상태임이 증명된다.
        if asleep[i]
            && (bodies[i].linvel != maru_math::Vec3::ZERO
                || bodies[i].angvel != maru_math::Vec3::ZERO)
        {
            return Err(SnapshotError::InconsistentState);
        }
        // **하이트필드는 static 전용**이다 — 질량·관성 경로가 없고(0을 돌려준다) 솔버가
        // 임펄스를 주면 무한 가속이 된다. 라이브 생성이 이미 거부하므로 복원도 거부한다
        // (입력 경계와 복원 경계의 정책 대칭 — 이 저장소가 네 번 재발로 배운 것).
        // 라이브 생성(`Body::validate`)과 **같은 정책**이어야 한다 — 한쪽만 막으면
        // `restore(snapshot())`가 자기 스냅샷을 거부한다(입력·복원 비대칭 계열).
        if matches!(
            bodies[i].shape,
            maru_collide::Shape::Heightfield { .. } | maru_collide::Shape::TriMesh { .. }
        ) && bodies[i].body_type != BodyType::Static
        {
            return Err(SnapshotError::InconsistentState);
        }
        // OOB는 살아있는 비-static만의 상태이고, sleep과 **겸할 수 없다**
        if oob[i] && (!alive[i] || bodies[i].body_type == BodyType::Static || asleep[i]) {
            return Err(SnapshotError::InconsistentState);
        }
        if oob[i] && island[i] != u32::MAX {
            return Err(SnapshotError::InconsistentState);
        }
        // 아일랜드 대표는 살아있는 dynamic이거나 없음(u32::MAX)
        if island[i] != u32::MAX && !live_dynamic(island[i] as usize) {
            return Err(SnapshotError::BadIndex);
        }
        // 비동적·비생존 슬롯은 아일랜드에 속하지 않는다(설계 §4.3: static은 anchor)
        if island[i] != u32::MAX && !live_dynamic(i) {
            return Err(SnapshotError::InconsistentState);
        }
    }

    // 핸들 해석 — 인덱스·생존·세대가 모두 맞아야 한다
    let resolve = |h: &BodyHandle| -> bool {
        let i = h.index as usize;
        i < nb && alive[i] && generations[i] == h.generation
    };

    for j in joints {
        if !resolve(&j.body_a) || !resolve(&j.body_b) {
            return Err(SnapshotError::BadIndex);
        }
        if j.body_a.index == j.body_b.index {
            return Err(SnapshotError::InconsistentState);
        }
        if !j.anchor_a.is_finite() || !j.anchor_b.is_finite() {
            return Err(SnapshotError::NonFinite);
        }
        // OOB 인접 조인트는 누적 임펄스가 **0이어야 한다** — 라이브 전이가 지우는 값이므로,
        // 0이 아니면 라이브로 도달할 수 없는 상태다(복귀 시 stale kick의 씨앗).
        if (oob[j.body_a.index as usize] || oob[j.body_b.index as usize])
            && (j.linear_impulse != Vec3::ZERO
                || j.angular_impulse[0] != 0.0
                || j.angular_impulse[1] != 0.0)
        {
            return Err(SnapshotError::InconsistentState);
        }
        // 앵커가 **그 바디의 크기**에 비례하는지 — 라이브 `try_add_joint`와 같은 판정
        if !crate::world::anchor_fits(&bodies[j.body_a.index as usize], j.anchor_a)
            || !crate::world::anchor_fits(&bodies[j.body_b.index as usize], j.anchor_b)
        {
            return Err(SnapshotError::InconsistentState);
        }
        // **K 조건수도 라이브와 같은 검사**를 거친다(리뷰 BLOCKER: restore가 우회해서,
        // 라이브가 InvalidValue로 거부하는 K-특이 조인트를 정본 배열에 직접 넣을 수 있었다).
        if !crate::joint::k_is_solvable(bodies, j) {
            return Err(SnapshotError::InconsistentState);
        }
    }

    // 커맨드 대상 검증은 **공개 `enqueue`와 동일한 술어**를 쓴다(리뷰 PARTIAL MAJOR).
    // 세대만 보면, 공개 API로는 만들 수 없는 pending 상태(죽은 비예약 슬롯을 겨냥한
    // SetVelocity·AddJoint)를 조작 스냅샷이 주입할 수 있다 — 정본이 아닌 상태를
    // 정본 큐에 들이는 셈이다.
    // enqueue와 같은 술어: 대상은 살아있거나 예약된 슬롯이고 세대가 일치해야 한다.
    let enqueueable = |h: &BodyHandle| -> bool {
        let i = h.index as usize;
        i < nb && (alive[i] || reserved[i]) && generations[i] == h.generation
    };
    let mut add_targets: Vec<u32> = Vec::new();
    for c in commands {
        let c = match c {
            PendingCommand::Add(h, _) => {
                let i = h.index as usize;
                // AddBody는 **예약된** 슬롯만 채울 수 있다
                if i >= nb || !reserved[i] {
                    return Err(SnapshotError::InconsistentState);
                }
                add_targets.push(h.index);
                continue;
            }
            PendingCommand::Public(c) => c,
        };
        let h = match c {
            Command::RemoveBody(h)
            | Command::SetLinearVelocity(h, _)
            | Command::SetAngularVelocity(h, _)
            | Command::SetPose(h, _)
            | Command::SetMaterial(h, _, _) => h,
            Command::AddJoint(j) => {
                for h in [&j.body_a, &j.body_b] {
                    if !enqueueable(h) {
                        return Err(SnapshotError::BadIndex);
                    }
                }
                if j.body_a.index == j.body_b.index {
                    return Err(SnapshotError::InconsistentState);
                }
                continue;
            }
        };
        if !enqueueable(h) {
            return Err(SnapshotError::BadIndex);
        }
    }
    // 예약 슬롯과 pending AddBody는 **일대일**이다. 짝 없는 예약은 영원히 회수되지 않는
    // 슬롯 누수이고, 중복 AddBody는 둘째가 조용히 거부된다.
    add_targets.sort_unstable();
    let dup = add_targets.windows(2).any(|w| w[0] == w[1]);
    let reserved_count = reserved.iter().filter(|x| **x).count();
    if dup || reserved_count != add_targets.len() {
        return Err(SnapshotError::InconsistentState);
    }

    // 접촉 기준선도 **캐시 키와 같은 규칙**으로 검증한다. 검증을 안 하면 조작된
    // 스냅샷이 임의의 두 핸들로 가짜 Started/Stopped 이벤트를 만들 수 있고,
    // 이벤트로 데미지·센서 상태를 구동하는 호스트가 그대로 속는다
    // (검수 rev_7f7bb07c MAJOR).
    for (a, b_raw) in touching.iter().chain(cache.keys()) {
        let b = (b_raw & !STATIC_BIT) as usize;
        let a = *a as usize;
        if a >= nb || b >= nb || !alive[a] || !alive[b] {
            return Err(SnapshotError::BadIndex);
        }
        let b_is_static = b_raw & STATIC_BIT != 0;
        if b_is_static != (bodies[b].body_type == BodyType::Static) {
            return Err(SnapshotError::InconsistentState);
        }
    }
    // **캐시 ⊆ touching** — 라이브 월드의 불변식이다(스텝 말 retain이 라이브 집합으로
    // 거르고, touching이 곧 그 집합이다). 복원이 이걸 안 지키면 제한 모드의 델타 retain
    // (비-fresh 키 잔류)이 전체 경로(라이브 집합 밖 폐기)와 갈린다 — 라이브 도달불가
    // 상태는 받아주는 대신 거부한다.
    for k in cache.keys() {
        if !touching.contains(k) {
            return Err(SnapshotError::InconsistentState);
        }
    }
    Ok(())
}
