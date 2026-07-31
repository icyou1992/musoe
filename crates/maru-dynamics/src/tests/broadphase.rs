//! broadphase 관련 테스트.

use super::*;

/// **복합 바디는 페어 하나가 제약 여러 개다** — 스크래치 회계가 그것을 세는지
/// (교차검수 rev_518b072b BLOCKER).
///
/// 8자식 복합 64개가 전부 겹치면 페어는 2,016개뿐이지만 제약은 최대 64배(≈129k)라
/// 128 MiB 예산을 넘는다. 페어 수로만 세던 회계는 이 스텝을 통과시킨 뒤 wake 폐쇄
/// **뒤에** 재할당했다. 지금은 페어 목록에서 정확한 제약 상한을 구해 미리 거절한다.
///
/// 판정이 **페어 예산이 아니라 제약 회계**에서 났음을 함께 확인한다:
/// 브로드페이즈는 성공해 `diag.pairs`가 차 있고 예산 안이다.
#[test]
fn compound_pairs_are_budgeted_by_constraints_not_pair_count() {
    fn scene(compound: bool) -> (World, Result<(), ApiError>) {
        let mut w = World::new(WorldParams::default());
        for i in 0..64 {
            let shape = if compound {
                Shape::Compound {
                    children: (0..8)
                        .map(|c| {
                            (
                                pose(0.02 * c as f32, 0.0, 0.0),
                                Shape::Sphere { radius: 0.2 },
                            )
                        })
                        .collect(),
                }
            } else {
                Shape::Sphere { radius: 0.2 }
            };
            // 전부 겹치게 촘촘히 둔다 — 페어 수는 같고 제약 수만 64배가 된다.
            w.add_body(Body::new_dynamic(
                shape,
                pose(0.01 * i as f32, 5.0, 0.0),
                1.0,
            ));
        }
        let r = w.try_step(DT);
        (w, r)
    }
    let (w_plain, r_plain) = scene(false);
    assert!(
        r_plain.is_ok(),
        "비복합 대조군이 실패하면 이 테스트는 예산이 아니라 씬을 재는 것이다: {r_plain:?}"
    );
    let pairs = w_plain.diagnostics().pairs;
    assert!(pairs > 1000, "겹침이 부족하다 — 페어 {pairs}개");

    let (w_comp, r_comp) = scene(true);
    assert!(
        matches!(r_comp, Err(ApiError::ResourceLimit)),
        "복합 씬이 제약 회계에 걸리지 않았다: {r_comp:?}"
    );
    let d = w_comp.diagnostics();
    // 복합은 자식이 퍼져 AABB가 커서 페어가 조금 더 많다 — 같은 수를 요구하지 않는다.
    assert!(
        d.pairs >= pairs,
        "브로드페이즈 자체는 통과했어야 한다: {}",
        d.pairs
    );
    assert!(
        d.pairs <= d.pair_budget,
        "페어 예산({})이 먼저 걸렸다 — 제약 회계를 재는 테스트가 아니게 된다",
        d.pair_budget
    );
}

/// **큰 장면이 남긴 스크래치는 다음 스텝 시작에 놓아준다** (확인검수 rev_51f69fbd BLOCKER).
///
/// 총상한 검사와 반납은 브로드페이즈가 새 페어 배열을 만든 뒤에 돈다 — 그 전에 이미
/// 부풀어 있으면 그 할당에서 죽는다(fallible이 아니다). 그래서 스텝 시작에 "현재 필요의
/// 4배" 안으로 눌러 둔다. 되돌리면(위생 패스 제거) 이 테스트가 실패한다.
#[test]
fn stale_scratch_is_released_before_the_next_broadphase() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let mut hs = Vec::new();
    for i in 0..600 {
        hs.push(w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(i as f32 * 0.5, 3.0, 0.0),
            1.0,
        )));
    }
    w.step(DT);
    let big = w.body_scratch_capacities();
    assert!(big[0] >= 600, "큰 장면에서 스크래치가 안 자랐다: {big:?}");
    // 작은 월드로 **복원**한다 — 슬롯 수가 실제로 줄어드는 유일한 경로다.
    let mut small = World::new(WorldParams::default());
    ground(&mut small);
    for i in 0..4 {
        small.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(i as f32 * 0.5, 3.0, 0.0),
            1.0,
        ));
    }
    w.restore(&small.snapshot()).expect("복원");
    w.step(DT);
    let after = w.body_scratch_capacities();
    for (i, (b, a)) in big.iter().zip(after.iter()).enumerate() {
        // #1은 적분 게이트 — 증분 경로 입력이라 위생 패스가 건드리지 않는다(1바이트/원소).
        if i == 1 {
            continue;
        }
        assert!(
            *a <= 4 * 64 + 64,
            "스크래치 #{i}가 큰 장면 용량을 그대로 쥐고 있다: {b} → {a}"
        );
    }
}

/// BLOCKER: pending 속도 커맨드는 **flush 뒤**에 적용되므로, 클램프도 flush 뒤·
/// 브로드페이즈 앞이어야 sweep AABB 폭발을 막는다.
#[test]
fn pending_velocity_is_clamped_before_broadphase() {
    let mut w = World::new(WorldParams {
        max_linear_velocity: 1.0,
        ..Default::default()
    });
    ground(&mut w);
    let hs: Vec<_> = (0..30)
        .map(|i| dyn_box(&mut w, i as f32 * 3.0 - 45.0, 1.0))
        .collect();
    // 합법 입력 상한 속도를 pending으로 넣는다 (스텝 시작 클램프 **뒤**에 적용된다)
    for h in &hs {
        w.enqueue(Command::SetLinearVelocity(
            *h,
            Vec3::new(maru_collide::MAX_SPEED, 0.0, 0.0),
        ))
        .unwrap();
    }
    assert_eq!(w.try_step(crate::world::MAX_DT), Ok(()));
    for h in &hs {
        let v = w.body(*h).linvel().length();
        assert!(v <= 1.0 + 1e-3, "pending 속도가 클램프되지 않음: {v}");
    }
}

/// BLOCKER: 같은 자리에 겹친 N개 바디는 N²/2 페어를 만든다. 입력은 작아도
/// wasm에서는 오류 반환 전에 allocator가 죽으므로 **예산 초과는 오류**여야 한다.
#[test]
fn broadphase_pair_budget_is_an_error_not_a_trap() {
    let mut w = World::new(WorldParams {
        max_pairs_per_body: 1,
        ..Default::default()
    });
    for _ in 0..200 {
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(0.0, 1.0, 0.0), // 전부 같은 자리 → 모든 페어가 겹친다
            1.0,
        ));
    }
    assert_eq!(
        w.try_step(DT),
        Err(ApiError::ResourceLimit),
        "페어 폭발이 오류로 표현되지 않음"
    );
    // 상태는 손상되지 않고 스냅샷도 여전히 유효하다
    assert!(w.restore(&w.snapshot()).is_ok());
}

/// 회귀(캐시 계약): impulse는 속도를 **즉시** 바꾸고 sweep AABB는 속도의 함수이므로,
/// 정착 상태에서 impulse를 주면 다음 스텝은 페어 캐시를 버리고 재빌드해야 한다.
/// 재사용하면 임펄스 이전 속도의 페어로 한 스텝을 돌아 speculative 접촉이 빠진다.
#[test]
fn impulse_on_settled_world_rebuilds_pairs() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let h = dyn_box(&mut w, 0.0, 0.5);
    for _ in 0..300 {
        w.step(DT);
    }
    assert!(w.is_asleep(h), "먼저 잠들어야 의미 있는 테스트다");
    w.step(DT);
    assert!(w.diagnostics().bp_reused, "정착 상태 전제 확인");
    w.apply_impulse(h, Vec3::new(50.0, 0.0, 0.0)).unwrap();
    w.step(DT);
    assert!(
        !w.diagnostics().bp_reused,
        "임펄스 후에도 페어 캐시를 재사용했다 — 임펄스 전 속도의 sweep AABB로 스텝이 돈다"
    );
}

/// **전체 재빌드 직후의 증분 스텝**이 fast path를 거짓 양성으로 타지 않는가
/// (교차검수 rev_3de4e59a BLOCKER를 그대로 옮긴 시나리오).
///
/// 전체 재빌드는 슬롯별 페어 수를 만들지 않는다(만들어도 안 쓰므로 순손해다). 그때
/// 빈 벡터를 다음 스텝이 "전부 0"으로 읽으면, 페어를 가진 바디가 상대에게서 멀어져
/// **없어져야 할 페어**가 그대로 보존된다. 바디 상태는 안 바뀌므로 스냅샷 대조로는
/// 안 잡히고(위 락스텝 테스트가 그 이유로 못 잡았다), 목록만 조용히 부푼다.
///
/// 구성: 정적 다수 + 동적 하나. `moved_ids`는 동적 하나뿐이라 bp가 생긴 **다음 스텝부터
/// 곧바로 증분**이다. 동적 바디를 정적과 겹쳐 두고 위로 쏘아 페어를 잃게 만든다.
/// 전체 재빌드 월드와 매 스텝 페어 수를 대조한다.
#[test]
fn incremental_after_full_rebuild_does_not_keep_stale_pairs() {
    let build = |force_full: bool| {
        let mut w = World::new(no_sleep_params());
        w.set_force_full_broadphase(force_full);
        // 정적 다수 — moved_ids가 바디 수의 절반 아래가 되게 한다(증분 조건).
        for i in 0..20 {
            w.add_body(Body::new_static(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(i as f32 * 4.0, 0.0, 0.0),
            ));
        }
        // 첫 정적과 **겹쳐** 시작 → 첫 스텝에 페어가 생긴다.
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.6, 0.0),
            1.0,
        ));
        // 위로 **아주 빠르게** 쏘아 보낸다 — 취약 창은 "전체 재빌드 바로 다음 스텝"뿐이라,
        // 그 한 스텝 만에 페어를 잃어야 재현된다(천천히 벗어나면 그 사이 느린 병합이
        // 카운트를 만들어 창이 닫힌다).
        w.set_linear_velocity(h, Vec3::new(0.0, 200.0, 0.0))
            .unwrap();
        w
    };
    let (mut a, mut b) = (build(false), build(true));
    let mut saw_incremental = false;
    for step in 0..40 {
        a.step(DT);
        b.step(DT);
        saw_incremental |= a.diagnostics().bp_incremental;
        assert_eq!(
            a.diagnostics().pairs,
            b.diagnostics().pairs,
            "step {step}: 전체 재빌드 직후 증분이 옛 페어를 보존했다 — \
             카운트 '모름'을 '0'으로 읽은 것이다"
        );
    }
    // 이 테스트가 증분 경로를 **실제로** 지났는지 — 안 지났으면 아무것도 잠그지 않는다.
    assert!(
        saw_incremental,
        "증분 경로를 한 번도 타지 않았다 — 시나리오가 무의미하다"
    );
}

/// 증분 브로드페이즈 등가성 — **궤적 전체**로 잠근다: 증분 월드와 전체-재빌드 월드를
/// 같은 입력으로 락스텝 진행하며 매 스텝 스냅샷 바이트가 같아야 한다. 페어 집합이
/// 한 스텝이라도 갈리면 제약·이벤트·wake가 갈려 스냅샷이 갈린다.
/// 시나리오는 증분의 모든 전이를 지나간다: 활성 낙하(R>0) → 정착(전원 수면, fast path)
/// → 임펄스(구조 무효화 + wake) → 바디 추가(구조 무효화) → 재정착.
#[test]
fn incremental_broadphase_trajectory_matches_full_rebuild() {
    let build = |force_full: bool| {
        let mut w = World::new(WorldParams::default());
        w.set_force_full_broadphase(force_full);
        ground(&mut w);
        let mut hs = Vec::new();
        for i in 0..24 {
            let x = (i % 4) as f32 * 1.05 - 2.0;
            let z = ((i / 4) % 3) as f32 * 1.05 - 1.0;
            let y = 0.5 + (i / 12) as f32 * 1.05;
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(x, y, z),
                1.0,
            )));
        }
        // **제한 모드 안에서 접촉 이벤트를 내는 바디** — 이 테스트의 커버리지 조건
        // (`limited_event_steps > 0`)을 우연이 아니라 **구조로** 만족시킨다.
        //
        // 왜 필요한가: 접촉 이벤트는 보통 **기상과 함께** 일어나는데, 기상이 있으면
        // `narrow_limited`가 꺼진다(제한 narrow의 조건이 "이번 스텝에 아무도 안 깼다"이다).
        // 그래서 "이벤트 ∧ 제한"은 서로 배타적이기 쉽고, 예전에는 궤적이 우연히 그 교집합을
        // 밟아 통과했다 — 클리퍼 변경으로 궤적이 조금 움직이자 교집합이 사라져 이 단언이
        // 터졌다(실제로 겪었다: 제한 구간 38~443, 이벤트 444~454로 **한 끗 차이**로 어긋났다).
        // 반발 계수를 준 공을 무리에서 떨어뜨려 두면, 정적 바닥과 접촉을 반복해서 만들고
        // 끊으면서도 **아무도 깨우지 않는다**(정적 바디는 기상 대상이 아니다) → 이벤트와
        // 제한이 같은 스텝에 공존한다.
        hs.push(
            w.add_body(
                Body::new_dynamic(Shape::Sphere { radius: 0.3 }, pose(12.0, 3.0, 12.0), 1.0)
                    .with_restitution(0.85)
                    .expect("반발 계수는 0..=1"),
            ),
        );
        // 낙하 교란체 — 높이 띄워 "잠든 다수 + 활성 소수"(증분 창)를 길게 만든다.
        // R이 슬롯 절반을 넘는 활성 구간은 설계상 재빌드 폴백이라 증분이 안 돈다.
        for i in 0..4 {
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Sphere { radius: 0.4 },
                pose(i as f32 * 1.5 - 2.0, 50.0 + i as f32 * 8.0, 0.3),
                1.0,
            )));
        }
        (w, hs)
    };
    let (mut a, ha) = build(false); // 증분
    let (mut b, hb) = build(true); // 전체 재빌드 (대조군)
    assert_eq!(ha.len(), hb.len());
    let mut incr_steps = 0usize;
    let mut limited_event_steps = 0usize;
    for step in 0..460 {
        // 중간 이벤트: 임펄스(정착 후) · 바디 추가 — 양쪽 동일 적용
        if step == 340 {
            a.apply_impulse(ha[0], Vec3::new(6.0, 3.0, 0.0)).unwrap();
            b.apply_impulse(hb[0], Vec3::new(6.0, 3.0, 0.0)).unwrap();
        }
        if step == 380 {
            let body =
                || Body::new_dynamic(Shape::Sphere { radius: 0.4 }, pose(0.2, 8.0, -0.2), 1.0);
            a.add_body(body());
            b.add_body(body());
        }
        a.step(DT);
        b.step(DT);
        incr_steps += a.diagnostics().bp_incremental as usize;
        if a.diagnostics().narrow_limited && !a.contact_events().is_empty() {
            limited_event_steps += 1;
        }
        assert!(
            !b.diagnostics().bp_incremental,
            "대조군이 증분을 탔다 — 노브 고장, 테스트 무의미"
        );
        // 이벤트는 스냅샷 밖이다 — 제한 모드의 **델타 이벤트**(fresh에서만 차집합)가
        // 전체 차집합과 같은 이벤트를 내는지 명시적으로 대조한다.
        assert_eq!(
            a.contact_events(),
            b.contact_events(),
            "step {step}: 증분 이벤트가 전체 재빌드 이벤트에서 이탈"
        );
        assert_eq!(
            a.snapshot(),
            b.snapshot(),
            "step {step}: 증분 궤적이 전체 재빌드 궤적에서 이탈"
        );
        // **페어 집합도 대조한다 — 스냅샷만으로는 못 잡는다.**
        //
        // 증분 fast path("R이 직전 목록을 안 건드리고 새 페어도 없으면 병합 생략")의
        // 판정이 틀리면 **없어져야 할 페어가 목록에 남는다**. 그런데 그런 페어는 접촉이
        // 아니라 제약을 만들지 않으므로 **바디 상태가 안 바뀐다** — 스냅샷 대조는
        // 초록인 채로 목록만 조용히 부푼다(예산에 닿아야 겨우 드러난다).
        // 실제로 카운트를 일부러 누락시킨 뮤턴트가 277개 테스트를 전부 통과했다.
        // 페어 수는 diagnostics로 관측 가능하니 그것으로 잠근다.
        assert_eq!(
            a.diagnostics().pairs,
            b.diagnostics().pairs,
            "step {step}: 증분 페어 수가 전체 재빌드와 다르다 — fast path 판정이 틀렸다"
        );
    }
    // **여유를 두고 요구한다.** `> 0`이면 궤적이 조금만 움직여도 커버리지가 0으로 떨어져
    // "테스트가 통과했지만 아무것도 검증하지 않은" 상태가 된다(실제로 그렇게 터졌다).
    // 튀는 공이 만드는 실측값은 8이므로, 절반 아래로 내려가면 씬이 의도를 잃은 것이다.
    assert!(
        limited_event_steps >= 4,
        "제한 모드 + 이벤트 스텝이 {limited_event_steps}개뿐이다 — 델타 이벤트 경로의              커버리지가 무너졌다(튀는 공이 무리를 깨우고 있거나 씬 밖으로 나갔는지 보라)"
    );
    assert!(
        incr_steps > 60,
        "증분 경로가 거의 안 돌았다({incr_steps}스텝) — 테스트가 아무것도 증명 안 함"
    );
}
