//! resources 관련 테스트.

use super::*;

/// PCM 캐시 엔트리가 **내용에 비해 과대한 용량을 쥐고 있지 않다**.
///
/// 배경(교차검수 rev_8f75295a MAJOR): 캐시를 자리에서 채우면 `Vec` 용량이 스텝 간
/// 살아남는다 — 그게 할당을 없앤 이유지만, 상한이 없으면 한 번 크게 자란 키가 과거
/// 최대치를 영구 잔류시킨다(128 MiB 스텝 회계 **밖**에서). 그래서 채운 뒤
/// `capacity > max(len,4)·4`면 되돌린다.
///
/// **정직한 한계**: 이 테스트는 그 상한 가드를 **격리하지 못한다**. 뮤테이션으로
/// 확인했다 — 가드를 지워도 통과한다. 시도한 모든 씬(복합 바디를 기울여 접점을 줄이는
/// 경로, 한 번에/점진적으로 둘 다)에서 전이 도중 접촉이 끊겨 **캐시 키가 `retain`에
/// 먼저 지워지고**, 그러면 `Vec`이 통째로 해제돼 용량이 어차피 돌아온다.
/// 그 사실은 누수 경로가 좁다는 뜻이기도 하다(같은 키가 접촉을 **한 번도 놓지 않으면서**
/// 접점 수만 줄어야 한다). 가드는 싸고 불변식은 옳으므로 남기되, 여기서 잠그는 것은
/// **불변식**이지 그 줄이 아니다.
/// wasm 레그에서는 건너뛴다 — 복합 8자식 × 150스텝이 단일 wasm 인스턴스에서 순차로 도는
/// 전체 스위트의 메모리 예산을 넘긴다(격리 실행은 통과). 이 불변식은 타깃 무관이다.
#[test]
#[cfg_attr(target_arch = "wasm32", ignore)]
fn pcm_cache_entries_do_not_hoard_capacity() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(30.0, 1.0, 30.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let children: Vec<(maru_collide::Pose, Shape)> = (0..8)
        .map(|i| {
            (
                pose(i as f32 * 1.2 - 4.2, 0.0, 0.0),
                Shape::Box {
                    half: Vec3::new(0.5, 0.25, 0.5),
                    radius: 0.0,
                },
            )
        })
        .collect();
    let comp = w
        .try_add_body(
            Body::try_new_kinematic(Shape::Compound { children }, pose(0.0, 0.25, 0.0))
                .expect("kinematic 복합"),
        )
        .expect("삽입");
    for _ in 0..90 {
        w.step(1.0 / 60.0);
    }
    assert!(
        w.pcm_cache_capacity_bytes() > 0,
        "캐시가 비었다 — 씬이 접촉을 안 만들었다(단언이 아무것도 안 본다)"
    );
    // 접점 수를 줄여 가며 불변식을 계속 확인한다.
    for k in 1..=60 {
        let a = 0.45 * k as f32 / 60.0;
        let p = maru_collide::Pose {
            pos: Vec3::new(0.0, 0.25 + 4.7 * a.sin(), 0.0),
            rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), a),
        };
        w.set_pose(comp, p).expect("자세 설정");
        w.step(1.0 / 60.0);
        if let Some((len, cap)) = w.pcm_cache_worst_capacity_ratio() {
            assert!(
                cap <= len.max(4) * 4,
                "step {k}: 캐시 엔트리가 과대 용량을 쥐고 있다 (len={len}, cap={cap})"
            );
        }
    }
}

/// 참여자 버퍼 예약이 **목표 용량에 실제로 닿는지** (교차검수 rev_518b072b MAJOR).
///
/// `Vec::try_reserve`의 인자는 목표 용량이 아니라 `len`에 더할 여분이다. 목표에서
/// `capacity`를 빼서 넘기면 비워진 버퍼(len=0)가 목표보다 작은 용량으로도 요청을
/// 통과시키고, 그 뒤 push가 **wake 폐쇄 뒤에** 재할당한다("스텝 중 할당 0"·
/// "Err면 상태 불변" 계약이 동시에 깨진다). 되돌리면 이 테스트가 실패한다.
#[test]
fn narrow_buffer_reserve_reaches_target_capacity_after_reuse() {
    let mut b = crate::world::NarrowOut::default();
    b.reserve(150, 150).expect("첫 예약");
    let before = b.capacities();
    assert!(before.0 >= 150);
    // 사용 후 비워진 상태(len=0, capacity≥150)에서 더 큰 목표를 요구하는 상황.
    b.reserve(200, 200).expect("두 번째 예약");
    let after = b.capacities();
    assert!(
        after.0 >= 200 && after.1 >= 200 && after.2 >= 200,
        "예약이 목표에 못 닿았다: {after:?} (목표 200)"
    );
}

/// 참여자 수는 설계 상한(`MAX_THREADS`)을 넘지 않는다 — 넘기면 스핀 설계가 무너지고
/// 참여자별 스크래치가 입력과 무관하게 커진다(교차검수 rev_518b072b MAJOR).
#[test]
fn thread_count_is_clamped_to_design_limit() {
    let mut w = World::new(WorldParams::default());
    w.set_threads(4096);
    assert!(
        w.threads() <= crate::threads::MAX_THREADS,
        "참여자 {}명 — 상한 {}을 넘었다",
        w.threads(),
        crate::threads::MAX_THREADS
    );
    w.set_threads(1);
    assert_eq!(w.threads(), 1, "1은 그대로여야 한다(완전 직렬)");
}

/// 영속 스크래치 예약이 **목표 용량**을 요구하는지 (확인검수 rev_071f1cd9 MAJOR).
///
/// `try_reserve(nbody)`는 "지금 len에 nbody를 **더**"라는 뜻이라, 매 스텝 len이
/// 바디 수인 버퍼에 그대로 넘기면 두 번째 스텝부터 용량이 2배로 부푼다 —
/// 회계는 한 벌만 세므로 128 MiB 상한이 조용히 깨진다. 되돌리면 이 테스트가 실패한다.
#[test]
fn per_body_scratch_does_not_double_each_step() {
    const N: usize = 120;
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    for i in 0..N {
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(i as f32 * 0.5, 3.0, 0.0),
            1.0,
        ));
    }
    let n = w.slot_count();
    for _ in 0..3 {
        w.step(DT);
    }
    for (i, cap) in w.body_scratch_capacities().iter().enumerate() {
        // 고친 경로는 정확히 슬롯 수만큼 잡는다. 버그는 `len + nbody`를 요구해
        // 두 번째 스텝에서 **정확히 2배**가 되므로, 여유를 조금만 두면 갈린다.
        assert!(
            *cap <= n + n / 4 + 16,
            "per-body 스크래치 #{i} 용량 {cap}이 슬롯 수 {n}보다 크게 부풀었다 — 매 스텝 재예약이 쌓인다"
        );
    }
}

/// `SolverState::reserve_for`가 `len < capacity < 목표`에서도 목표에 닿는지
/// (확인검수 rev_071f1cd9 MAJOR — `NarrowOut::reserve`와 같은 계열의 잔존).
#[test]
fn solver_state_reserve_reaches_target_when_capacity_exceeds_len() {
    let mut st = crate::solver::SolverState::default();
    // 큰 용량을 먼저 잡고(=capacity 128) 작은 수로 채운다(=len 100).
    st.reserve_for(128).expect("첫 예약");
    let bodies: Vec<Body> = (0..100)
        .map(|_| Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0))
        .collect();
    let alive = vec![true; 100];
    let oob = vec![false; 100];
    st.fill_from(&bodies, &alive, &oob);
    assert!(st.lin_capacity() >= 128 && st.lin_capacity() > 100);
    // 이제 목표 1000 — `목표 - capacity`로 계산하면 보장이 972에 그친다.
    st.reserve_for(1000).expect("두 번째 예약");
    assert!(
        st.lin_capacity() >= 1000,
        "예약이 목표에 못 닿았다: {} (목표 1000)",
        st.lin_capacity()
    );
}

/// 참여자 버퍼 회계가 **실제 보유 용량**을 세는지 (확인검수 rev_071f1cd9 MAJOR).
/// 영속 버퍼는 줄지 않으므로, 이번 청크 목표가 작아도 이미 쥔 만큼을 세야 한다.
#[test]
fn narrow_buffer_accounting_counts_held_capacity() {
    let mut b = crate::world::NarrowOut::default();
    // 용량은 판정에 필요한 만큼만 — wasm 테스트는 한 인스턴스에서 전 테스트가 돌고
    // 선형 메모리는 줄지 않으므로, 테스트 하나의 발자국이 전체 상한을 밀어 올린다.
    b.reserve(1024, 1024).expect("큰 예약");
    let small_target = crate::world::NarrowOut::bytes_for(8, 8);
    let projected = b.projected_bytes(8, 8);
    assert!(
        projected > small_target * 50,
        "보유 용량(4096)을 안 세고 목표(8)만 셌다: {projected} vs {small_target}"
    );
}

/// **줄어든 월드의 영속 스크래치는 반납된다** (확인검수 rev_21c1a97f MAJOR).
///
/// 회계는 "이번 스텝에 필요한 양"으로 세는데 `Vec` 용량은 저절로 줄지 않는다.
/// 큰 월드를 돌린 뒤 작아지면 그 차이가 회계 밖에 남아 128 MiB 주장이 거짓이 된다.
/// 반대로 "넘쳤으니 거절"만 하면 용량이 안 줄어 그 월드는 영원히 못 돈다 —
/// 그래서 **반납하고 다시 잰다**. 이 테스트는 반납이 실제로 일어나는지 본다.
#[test]
fn shrunken_world_releases_persistent_scratch() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let mut hs = Vec::new();
    for i in 0..400 {
        hs.push(w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(i as f32 * 0.5, 3.0, 0.0),
            1.0,
        )));
    }
    w.step(DT);
    let big = w.body_scratch_capacities();
    assert!(big[0] >= 400, "큰 월드에서 스크래치가 안 자랐다: {big:?}");
    // 대부분을 지운다 — 슬롯은 줄지 않지만(free-list) 용량 회계의 목표는 줄어든다.
    for h in hs.iter().skip(8) {
        w.remove_body(*h).expect("제거");
    }
    // 슬롯 수를 실제로 줄이려면 새 월드가 필요하다 — 여기서는 **반납 경로 자체**를
    // 직접 부른다(스텝 경로는 초과분이 상한을 넘을 때만 부른다).
    w.release_scratch_for_test(8);
    let small = w.body_scratch_capacities();
    // 반납은 **해제**다(축소가 아니라) — 전부 0이어야 한다. 축소로 두면 과거 용량이
    // 목표만큼 남아 회복하지 못하는 경로가 생기고, `shrink_to`는 재할당을 시도할 수
    // 있어 자원이 빠듯한 바로 그 순간에 abort할 수 있다(확인검수 rev_de34bbb5).
    for (i, (b, s)) in big.iter().zip(small.iter()).enumerate() {
        assert!(*s == 0, "스크래치 #{i}가 해제되지 않았다: {b} → {s}");
    }
    // 반납 뒤에도 정상 동작해야 한다(예약이 다시 목표까지 올라간다).
    assert!(w.try_step(DT).is_ok(), "반납 뒤 스텝이 실패했다");
}

/// **접촉 예산을 넘긴 스텝은 물리를 한 톱도 진행하지 않는다** (BLOCKER: 부분 물리 적용).
///
/// 예전에는 `budget_exceeded`를 보면 플래그만 세우고 앞부분 제약으로 솔브·적분을 끝까지
/// 했다. 그러면 `ResourceLimit`("커맨드는 적용, 물리 스텝은 건너뜀") 계약이 깨지고,
/// 호스트가 오류를 보고 재시도할 때마다 불완전한 물리가 한 스텝씩 더 나아간다
/// (확인검수 rev_1a31d83f BLOCKER 1).
///
/// 탈출구도 함께 확인한다: 커맨드는 스텝 0단계에서 이미 적용됐고 `resource_limit`은 매
/// 스텝 초기화되므로, 문제 바디를 지우면 다음 스텝은 정상으로 돌아온다(영구 동결 아님).
#[test]
fn contact_budget_exceeded_step_advances_nothing() {
    let mut w = World::new(WorldParams::default());
    // 256×256 평지(셀 0.1m). 셀 예산은 4096이다.
    let heights = vec![0.0f32; 257 * 257];
    w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    // **AABB는 거대하지만 실제 접촉은 몇 개뿐인 바디**가 예산을 때린다: 반지름 5m 구를
    // 평지에 살짝 닿게 놓으면 AABB는 100×100셀(=10000 > 4096)을 덮는데 접촉은 바닥
    // 몇 셀뿐이다. 넓적한 판으로는 안 된다 — 판은 매니폴드 상한(64)에 먼저 걸려
    // `Saturated`("찾았고 끊었다")가 되고 그건 예산 초과가 아니다.
    // 띄우면(y 밴드가 지형 [0,0] 타일과 안 겹치면) 셀을 세지도 않으므로 닿게 놓는다.
    let slab = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 5.0 },
        pose(12.8, 4.999, 12.8),
        1.0,
    ));
    let before = w.body(slab).pose().pos;
    let before_v = w.body(slab).linvel();
    assert_eq!(w.try_step(1.0 / 60.0), Err(ApiError::ResourceLimit));
    assert_eq!(
        w.body(slab).pose().pos,
        before,
        "예산 초과 스텝이 위치를 바꿨다 — 부분 물리가 적용됐다"
    );
    assert_eq!(
        w.body(slab).linvel(),
        before_v,
        "예산 초과 스텝이 속도를 바꿨다 — 적분이 돌았다"
    );
    assert!(w.contact_events().is_empty());
    // **음성 대조 ①** — 같은 세계에서 작은 바디만 남기면 스텝이 정상 통과하고 실제로
    // 떨어진다(테스트가 '지형이 있으면 항상 실패'를 보는 게 아니다).
    w.remove_body(slab).expect("제거");
    let small = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.2 },
        pose(12.8, 5.0, 12.8),
        1.0,
    ));
    let y0 = w.body(small).pose().pos.y;
    assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
    assert!(
        w.body(small).pose().pos.y < y0,
        "정상 스텝인데 중력이 안 걸렸다 — 대조군이 무의미하다"
    );
}

/// **존재 질의는 하나 찾으면 예산 초과와 무관하게 `Some(true)`다.**
///
/// 부정("접촉 없음")만 순회 완주를 요구한다. 긍정을 `None`으로 돌려주면 지원 범위 안의
/// 고정 장면(넓은 AABB + 접촉 몇 개)이 영구히 답을 못 받는다(확인검수 rev_d5b32e22 MAJOR).
#[test]
fn any_contact_keeps_a_proven_hit_even_when_the_budget_blows() {
    let nx = 257usize;
    let mut heights = vec![0.0f32; nx * nx];
    // 순회 **앞쪽**(예산 4096 안)에 구 표면에 닿는 봉우리를 하나 만든다. 셀 0.1m이므로
    // 격자 좌표 100 ≈ 월드 10.0m이고, 중심 (12.8, 5.0)에서 수평 3.96m 떨어진 지점의
    // 구 표면 높이는 약 1.95m다.
    for gz in 99..=101 {
        for gx in 99..=101 {
            heights[gz * nx + gx] = 1.95;
        }
    }
    let hf = Shape::heightfield(heights, nx as u32, nx as u32, 0.1).expect("격자");
    let p = pose(0.0, 0.0, 0.0);
    let big = Shape::Sphere { radius: 5.0 };
    // 봉우리에 닿는 배치: 접촉을 앞쪽 셀에서 찾고, 그 뒤 수천 셀에서 예산이 터진다.
    assert_eq!(
        maru_collide::any_contact(&big, &pose(12.8, 5.0, 12.8), &hf, &p, 0.02),
        Some(true),
        "증명된 접촉이 예산 초과 때문에 사라졌다"
    );
    // **음성 대조 ①** — 봉우리를 벗어난 자리(같은 크기 AABB, 앞쪽 셀에 접촉 없음)에서는
    // 예산이 터질 때까지 아무것도 못 찾았으므로 `Some(false)`가 아니라 `None`이다.
    // 이쪽이 무너지면 "판정 미완료"가 "접촉 없음"으로 뭉개진다.
    assert_eq!(
        maru_collide::any_contact(&big, &pose(12.8, 5.0, 20.0), &hf, &p, 0.02),
        None,
        "판정 미완료가 '접촉 없음'으로 뭉개졌다"
    );
    // **음성 대조 ②** — 타일 가지치기가 전부 걸러내면 순회는 **완주**한 것이므로
    // `Some(false)`가 맞다(예산과 무관하게 확정적인 '없음').
    assert_eq!(
        maru_collide::any_contact(&big, &pose(12.8, 40.0, 12.8), &hf, &p, 0.02),
        Some(false)
    );
}

/// **`ResourceLimit`의 기준선은 "호출 직전"이 아니라 "커맨드 적용 직후"다.**
///
/// README가 한때 "실패한 스텝은 한 비트도 바꾸지 않는다"라고 단정했는데, 커맨드 flush와
/// 속도 클램프는 실패보다 **먼저** 일어난다(그건 의도된 계약이다 — 빈 월드에 `AddBody`만
/// 큐잉하면 영원히 적용되지 않는 문제를 그렇게 막았다). 호스트가 그 문구를 믿고 오류 후
/// 커맨드를 다시 큐에 넣으면 같은 변경이 두 번 적용된다(확인검수 rev_2e93ebf2 MAJOR).
#[test]
fn resource_limit_applies_commands_but_skips_physics() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 257 * 257];
    w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    let slab = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 5.0 },
        pose(12.8, 4.999, 12.8),
        1.0,
    ));
    let marker = w.enqueue_add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.1 },
        pose(1.0, 1.0, 1.0),
        1.0,
    ));
    let pos_before = w.body(slab).pose().pos;
    assert_eq!(w.try_step(1.0 / 60.0), Err(ApiError::ResourceLimit));
    // 커맨드는 적용됐다 — 재큐잉하면 두 번 들어간다.
    assert!(
        w.is_alive(marker),
        "실패 스텝이 커맨드를 삼켰다 — 호스트가 잃어버린다"
    );
    // 물리는 진행되지 않았다.
    assert_eq!(
        w.body(slab).pose().pos,
        pos_before,
        "실패 스텝이 물리를 진행시켰다"
    );
}

/// BLOCKER: 예약 핸들당 pending 추가는 정확히 하나여야 한다.
/// **타입 수준으로** 보장한다 — 공개 `Command`에 바디 추가 variant가 없으므로
/// 두 번째 추가를 표현할 방법 자체가 없다(정본 경로 = `enqueue_add_body`).
#[test]
fn reserved_slot_has_exactly_one_pending_add() {
    let mut w = World::new(WorldParams::default());
    let h = w.enqueue_add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 3.0, 0.0),
        1.0,
    ));
    assert_eq!(w.pending_commands(), 1);
    // 같은 바디를 또 추가하면 **다른 슬롯**이 예약된다 (핸들이 다르다)
    let h2 = w.enqueue_add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(1.0, 3.0, 0.0),
        1.0,
    ));
    assert_ne!(h.index, h2.index, "같은 슬롯이 두 번 예약됨");
    assert_eq!(w.pending_commands(), 2);
    assert!(w.restore(&w.snapshot()).is_ok());
    w.step(DT);
    assert!(w.is_alive(h) && w.is_alive(h2));
    assert_eq!(w.last_command_report().rejected, 0);
}

/// BLOCKER: 속도 상한은 **문서화된 물리 파라미터**(Jolt mMaxLinearVelocity 계열)다.
/// 이것이 파생 상태를 유계로 만들어 정책이 step에 대해 닫힌다.
#[test]
fn velocity_limit_bounds_derived_state() {
    let params = WorldParams {
        max_linear_velocity: 50.0,
        ..Default::default()
    };
    let mut w = World::new(params);
    let h = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(0.0, 0.0, 0.0),
        1.0,
    ));
    w.set_linear_velocity(h, Vec3::new(1000.0, 0.0, 0.0))
        .unwrap();
    w.step(DT);
    let v = w.body(h).linvel().length();
    assert!(v <= 50.0 + 1e-3, "속도 상한이 적용되지 않음: {v}");
    assert!(
        w.diagnostics().velocity_clamped > 0,
        "클램프가 집계되지 않음"
    );
}

/// 리뷰 권고(커버리지): 속도 상한은 **고반발·구동 kinematic**에서 실제로 닿는다.
/// 상한에 닿아도 상태가 정책 안에 남고 왕복이 성립하는지 확인한다.
#[test]
fn velocity_limit_with_restitution_and_kinematic() {
    let mut w = World::new(WorldParams {
        max_linear_velocity: 3.0, // 일부러 낮춰 상한에 닿게 한다
        solver: crate::solver::SolverParams {
            restitution_threshold: 0.1,
            ..Default::default()
        },
        ..Default::default()
    });
    ground(&mut w);
    // 완전 탄성에 가까운 공을 떨어뜨린다
    let ball = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 4.0, 0.0),
        1.0,
    ));
    w.set_material(ball, 0.2, 0.95).unwrap();
    // 구동 kinematic 판이 아래에서 올라온다
    let paddle = w.add_body(Body::new_kinematic(
        Shape::Box {
            half: Vec3::new(2.0, 0.2, 2.0),
            radius: 0.0,
        },
        pose(0.0, 0.5, 0.0),
    ));
    w.set_linear_velocity(paddle, Vec3::new(0.0, 2.0, 0.0))
        .unwrap();

    let mut clamped_total = 0usize;
    for _ in 0..240 {
        // 상한에 닿는 것은 정상 기능이므로 오류가 아니다
        assert_eq!(w.try_step(DT), Ok(()));
        clamped_total += w.diagnostics().velocity_clamped;
        let v = w.body(ball).linvel().length();
        assert!(v <= 3.0 + 1e-3, "속도 상한을 넘김: {v}");
        assert_eq!(w.diagnostics().state_saturations, 0);
    }
    assert!(
        clamped_total > 0,
        "시나리오가 상한에 닿지 않음 — 판별력 없음"
    );
    assert!(w.restore(&w.snapshot()).is_ok());
}

/// BLOCKER: `ResourceLimit` 플래그가 스텝마다 초기화되지 않으면 **영구 고착**된다 —
/// 원인을 없앤 뒤에도 계속 오류를 돌려주면서 상태만 앞으로 나간다.
#[test]
fn resource_limit_flag_clears_next_step() {
    let mut w = World::new(WorldParams {
        max_pairs_per_body: 1,
        ..Default::default()
    });
    let hs: Vec<_> = (0..200)
        .map(|_| {
            w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.5 },
                pose(0.0, 1.0, 0.0),
                1.0,
            ))
        })
        .collect();
    assert_eq!(w.try_step(DT), Err(ApiError::ResourceLimit));
    // 원인을 제거하면 다음 스텝은 성공해야 한다
    for h in hs.iter().skip(3) {
        w.remove_body(*h).unwrap();
    }
    assert_eq!(w.try_step(DT), Ok(()), "플래그가 고착됨");
    assert_eq!(w.try_step(DT), Ok(()));
}

/// 미해결이었던 **전 단계 fallible 바이트 예산**: 페어 개수만 묶으면 뒤따르는
/// constraint·키·HashSet·SoA가 남는다. 총 바이트로 상한을 정하고 그것으로 페어 상한을
/// 되돌려 계산하므로, 어떤 입력이 와도 한 스텝 스크래치가 상한을 넘지 않는다.
#[test]
fn step_scratch_is_byte_budgeted_and_fallible() {
    // 계수형 예산을 최대로 열어도 **바이트 상한**이 페어를 묶는다
    let mut w = World::new(WorldParams {
        max_pairs_per_body: 4096,
        ..Default::default()
    });
    // **wasm 레그에서는 규모를 줄인다.** 이 테스트는 바이트 상한이 페어를 묶는지를 보는
    // 것이라 바디 수는 "최대 밀집이면 충분"하고, 절대 수가 본질이 아니다. wasm 스위트는
    // 전체가 **하나의 인스턴스**에서 순차로 돌고 선형 메모리는 줄지 않으므로, 알파벳
    // 순서상 뒤쪽인 이 테스트가 앞선 테스트들이 키워 놓은 힙 위에서 돈다 —
    // 이 세션에 스위트가 커지면서 결국 한계에 닿았다(격리 실행은 통과).
    // native에서는 그대로 300으로 둔다(더 센 조건).
    let n_stress = if cfg!(target_arch = "wasm32") {
        120
    } else {
        300
    };
    for _ in 0..n_stress {
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 1.0, 0.0), // 전부 같은 자리 → 최대 밀집
            1.0,
        ));
    }
    let r = w.try_step(DT);
    // 성공하든 ResourceLimit이든 **trap이 아니라 값으로** 돌아와야 한다
    assert!(
        r.is_ok() || r == Err(ApiError::ResourceLimit),
        "예상 밖 오류: {r:?}"
    );
    let d = w.diagnostics();
    // **바이트 상한이 실제로 묶었는지**를 본다. 원래 단언은 `budget * per_pair <=
    // MAX_STEP_SCRATCH_BYTES * 2`였는데 그건 **바이트 상한을 통째로 지워도 통과한다**:
    // 계수형 예산(4096×바디+1024)에 per_pair를 곱해도 두 레그 모두 256 MiB에 한참 못 미친다
    // (native 78 MiB · wasm 31 MiB). 즉 회귀를 못 잡는 녹색 테스트였다
    // (교차검수 rev_74d7e4ff MAJOR — wasm 축소로 드러났지만 native도 같은 구멍이었다).
    // 진짜 수용 기준은 "계수형 상한보다 **더 낮게** 잘렸다"이다.
    let coefficient_cap = 4096usize
        .saturating_mul(n_stress)
        .saturating_add(1024)
        .min(maru_collide::MAX_PAIRS);
    assert!(
        d.pair_budget < coefficient_cap,
        "바이트 상한이 페어 예산을 묶지 못했다: budget={} / 계수형 상한={}",
        d.pair_budget,
        coefficient_cap
    );
    // 상태는 손상되지 않고 스냅샷도 유효하다
    assert!(w.restore(&w.snapshot()).is_ok());
}

/// BLOCKER 회귀: 자원 실패가 **캐노니컬 상태를 훼손하면 안 된다**.
/// `mem::take(&mut self.joints)` 뒤에 예약 실패로 조기 반환해 **모든 조인트가 삭제**되던
/// 경로가 있었다. 선점을 캐노니컬 변이 **전에** 모두 끝내 그 창을 없앴다.
#[test]
fn resource_failure_never_destroys_canonical_state() {
    let mut w = World::new(WorldParams {
        max_pairs_per_body: 1,
        ..Default::default()
    });
    let a = dyn_box(&mut w, 0.0, 3.0);
    let b = dyn_box(&mut w, 1.2, 3.0);
    w.try_add_joint(Joint::point(a, b, Vec3::new(0.6, 0.0, 0.0), Vec3::ZERO))
        .unwrap();
    // 페어 예산을 넘겨 ResourceLimit을 강제한다
    for _ in 0..200 {
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 1.0, 0.0),
            1.0,
        ));
    }
    let before = w.state_hash();
    let r = w.try_step(DT);
    assert_eq!(r, Err(ApiError::ResourceLimit));
    // **조인트가 살아 있고 물리도 전진하지 않았어야 한다**
    assert_eq!(
        w.state_hash(),
        before,
        "자원 실패가 상태를 바꿨다 (조인트 삭제 또는 물리 전진)"
    );
    assert!(w.restore(&w.snapshot()).is_ok());
    // 원인을 없애면 다음 스텝은 정상이고 조인트도 그대로다
    w.set_params(WorldParams {
        max_pairs_per_body: 4096,
        ..Default::default()
    });
    assert_eq!(w.try_step(DT), Ok(()));
}
