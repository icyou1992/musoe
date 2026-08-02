//! sleeping 관련 테스트.

use super::*;

/// 잠든 바디에 힘을 주면 깨어나야 한다 — 안 그러면 "왜 안 움직이지"가 된다.
#[test]
fn applying_force_wakes_a_sleeping_body() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        musoe_collide::Shape::Box {
            half: musoe_math::Vec3::new(10.0, 1.0, 10.0),
            radius: 0.0,
        },
        musoe_collide::Pose {
            pos: musoe_math::Vec3::new(0.0, -1.0, 0.0),
            rot: musoe_math::Quat::IDENTITY,
        },
    ));
    let h = w.add_body(Body::new_dynamic(
        musoe_collide::Shape::Sphere { radius: 0.5 },
        musoe_collide::Pose {
            pos: musoe_math::Vec3::new(0.0, 0.5, 0.0),
            rot: musoe_math::Quat::IDENTITY,
        },
        1.0,
    ));
    for _ in 0..240 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    assert!(w.is_asleep(h), "먼저 잠들어야 의미 있는 테스트다");
    w.apply_impulse(h, musoe_math::Vec3::new(0.0, 3.0, 0.0))
        .unwrap();
    assert!(!w.is_asleep(h), "힘을 받으면 깨어나야 한다");
}

/// 잠든 물체는 **보인다**(자는 것은 최적화지 부재가 아니다).
#[test]
fn queries_see_sleeping_bodies() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        musoe_collide::Shape::Box {
            half: musoe_math::Vec3::new(10.0, 1.0, 10.0),
            radius: 0.0,
        },
        musoe_collide::Pose {
            pos: musoe_math::Vec3::new(0.0, -1.0, 0.0),
            rot: musoe_math::Quat::IDENTITY,
        },
    ));
    let h = w.add_body(Body::new_dynamic(
        musoe_collide::Shape::Sphere { radius: 0.5 },
        musoe_collide::Pose {
            pos: musoe_math::Vec3::new(0.0, 0.5, 0.0),
            rot: musoe_math::Quat::IDENTITY,
        },
        1.0,
    ));
    for _ in 0..240 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    assert!(w.is_asleep(h), "먼저 잠들어야 의미 있는 테스트다");
    let hit = w
        .cast_ray(
            musoe_math::Vec3::new(-5.0, 0.5, 0.0),
            musoe_math::Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
    assert_eq!(hit.map(|x| x.body), Some(h), "잠든 물체도 맞아야 한다");
}

/// **작은 스택의 수면 도달** — 접선 잔차가 남으면 스택이 영영 안 잔다
/// (교차검수 rev_ac00206f MAJOR: 상시 게이트에 수면 항목이 없었고, 전원 수면을 요구하는
/// 40단 테스트는 `#[ignore]`라 일반 실행에서 안 돈다). 단일 상자는 멈추는데 스택만
/// 못 자는 회귀는 이 게이트로만 잡힌다 — 그리고 못 자면 정착 장면이 활성 비용을 계속 내
/// **성능 회귀로 직결**된다.
///
/// 기준선: 21바디가 30스텝(=time_to_sleep 0.5초)에 전원 수면, 최대 침투 0.00152 m,
/// COM 옆흐름 dx −0.0012 m. 여기서는 3배 여유를 둔다.
#[test]
fn small_stack_falls_asleep_without_lateral_drift() {
    // 이 테스트는 수면이 본체라 기본 파라미터(수면 ON)를 쓴다.
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let mut hs = Vec::new();
    let mut com0 = Vec3::ZERO;
    for tier in 0..6usize {
        let count = 6 - tier;
        for i in 0..count {
            let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
            let p = Vec3::new(x, 0.5 + tier as f32, 0.0);
            hs.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                Pose {
                    pos: p,
                    rot: Quat::IDENTITY,
                },
                1.0,
            )));
            com0 += p;
        }
    }
    com0 = com0 * (1.0 / hs.len() as f32);
    let mut asleep_at = None;
    let mut max_pen = 0.0f32;
    for i in 0..300 {
        w.step(DT);
        max_pen = max_pen.max(w.diagnostics().max_penetration);
        if asleep_at.is_none() && hs.iter().all(|&h| w.is_asleep(h)) {
            asleep_at = Some(i + 1);
        }
    }
    let at = asleep_at.expect("300스텝(5초) 안에 스택 전원이 잠들지 않았다 — 접선 잔차 의심");
    assert!(at <= 90, "수면 도달이 늦어졌다: {at}스텝 (기준선 30)");
    assert!(
        max_pen < 0.01,
        "침투가 커졌다: {max_pen:.5} m (기준선 0.00152)"
    );
    let com: Vec3 =
        hs.iter().fold(Vec3::ZERO, |a, &h| a + w.body(h).pose().pos) * (1.0 / hs.len() as f32);
    let drift = ((com.x - com0.x).powi(2) + (com.z - com0.z).powi(2)).sqrt();
    assert!(
        drift < 0.02,
        "스택이 옆으로 흘렀다: {drift:.5} m (기준선 0.0012)"
    );
}

/// 슬리핑: 정착한 스택은 잠들고, 새 충돌체 도착 시 level-trigger로 깨어난다 (설계 §4.6)
#[test]
fn stack_sleeps_then_wakes_on_impact() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let base = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 0.5, 0.0),
        1.0,
    ));
    for _ in 0..180 {
        w.step(DT);
    }
    assert!(w.is_asleep(base), "3초 후에도 잠들지 않음");
    // 위에서 공 낙하 → 접촉 페어 생성 → wake
    let ball = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 3.0, 0.0),
        1.0,
    ));
    let mut woke = false;
    for _ in 0..120 {
        w.step(DT);
        if !w.is_asleep(base) {
            woke = true;
        }
    }
    assert!(woke, "충격에도 깨어나지 않음");
    let _ = ball;
}

/// 리뷰 BLOCKER 회귀: 접촉으로만 연결된 스택이 **완전히 잠든 뒤**에도 아일랜드가
/// 유지되어, 한 바디만 깨워도 전원이 함께 깨어나야 한다(설계 §4.6 island 단위 wake).
/// 스냅샷 왕복 후에도 같아야 한다(아일랜드가 캐노니컬 상태에 포함되므로).
#[test]
fn sleeping_stack_wakes_as_one_island() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let mut hs = Vec::new();
    for i in 0..3 {
        hs.push(w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5 + i as f32, 0.0),
            1.0,
        )));
    }
    // 충분히 오래 재워 아일랜드가 잠든 상태로 여러 스텝 유지되게 한다
    for _ in 0..300 {
        w.step(DT);
    }
    assert!(hs.iter().all(|h| w.is_asleep(*h)), "스택이 잠들지 않음");

    // 스냅샷 왕복 후에도 아일랜드가 보존되는지 함께 확인
    let snap = w.snapshot();
    let mut w2 = World::new(WorldParams::default());
    ground(&mut w2);
    for i in 0..3 {
        w2.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5 + i as f32, 0.0),
            1.0,
        ));
    }
    w2.restore(&snap).expect("restore");

    // 맨 아래 바디만 건드린다 → 아일랜드 전원이 깨어나야 함
    w2.set_linear_velocity(hs[0], Vec3::new(0.1, 0.0, 0.0))
        .expect("set velocity");
    for (i, h) in hs.iter().enumerate() {
        assert!(!w2.is_asleep(*h), "restore 후 box {i}가 깨어나지 않음");
    }
}

/// **실패한 스텝은 sleep 상태도 되돌린다** (BLOCKER: 캐노니컬 상태가 앞으로 나감).
///
/// wake 폐쇄는 narrowphase보다 먼저 돈다. 그래서 뒤에서 접촉 예산이 터져 스텝을
/// 포기하면, "물리 스텝은 건너뜀"이라 말하면서 잠들어 있어야 할 바디를 깨워 놓은 채
/// 끝난다 — `asleep`은 스냅샷에 실리는 캐노니컬 상태이므로, 호스트가 문제 바디를
/// 지우고 재시도해도 그 바디는 이미 중력·적분 대상이 되어 실패 전과 다른 미래를 만든다
/// (확인검수 rev_d5b32e22 BLOCKER 1).
#[test]
fn failed_step_rolls_back_wake() {
    let mut w = World::new(WorldParams {
        time_to_sleep: 0.1,
        ..WorldParams::default()
    });
    let heights = vec![0.0f32; 257 * 257];
    w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    let sleeper = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.2 },
        pose(12.8, 0.2, 12.8),
        1.0,
    ));
    // 지면에 앉아 잠들 때까지 돌린다.
    for _ in 0..300 {
        assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
        if w.is_asleep(sleeper) {
            break;
        }
    }
    assert!(w.is_asleep(sleeper), "테스트 전제(잠듦)가 성립하지 않았다");
    let pos_before = w.body(sleeper).pose().pos;
    // 예산을 터뜨리는 큰 바디를 **잠든 바디 위에** 놓는다: wake 폐쇄가 먼저 돌아
    // 잠든 아일랜드를 깨우고, 그 다음 narrowphase가 셀 예산에서 판정을 포기한다.
    let slab = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 5.0 },
        pose(12.8, 4.999, 12.8),
        1.0,
    ));
    // 기준선은 **바디를 추가한 뒤**에 잡는다 — 추가 자체는 캐노니컬 변경이 맞다.
    let hash_before = w.state_hash();
    assert_eq!(w.try_step(1.0 / 60.0), Err(ApiError::ResourceLimit));
    // **가장 강한 단언**: 실패한 스텝은 캐노니컬 상태를 한 비트도 바꾸지 않는다.
    assert_eq!(
        w.state_hash(),
        hash_before,
        "실패 스텝이 캐노니컬 상태를 바꿨다"
    );
    assert!(
        w.is_asleep(sleeper),
        "실패한 스텝이 잠든 바디를 깨운 채 끝났다 — 캐노니컬 상태가 앞으로 나갔다"
    );
    assert_eq!(w.body(sleeper).pose().pos, pos_before);
    // **음성 대조 ①** — 문제 바디를 지우면 다음 스텝은 정상이고, 잠든 바디는 계속 잔다
    // (롤백이 sleep 타이머를 망가뜨려 즉시 깨어나는 일이 없다).
    w.remove_body(slab).expect("제거");
    assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
    assert!(w.is_asleep(sleeper), "복구된 스텝이 잠든 바디를 깨웠다");
}

/// 리뷰 MAJOR 회귀: generational arena가 실제로 동작하는가 —
/// 제거 시 이웃 wake, 핸들 무효화, 슬롯 재사용 시 세대 증가로 옛 핸들 거부.
#[test]
fn body_removal_wakes_neighbors_and_invalidates_handles() {
    let mut w = World::new(WorldParams::default());
    let g = ground(&mut w);
    let base = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 0.5, 0.0),
        1.0,
    ));
    let top = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 1.5, 0.0),
        1.0,
    ));
    for _ in 0..300 {
        w.step(DT);
    }
    assert!(w.is_asleep(top) && w.is_asleep(base), "스택이 잠들지 않음");

    // 받침 제거 → 위 바디가 깨어나 낙하해야 한다 (설계 §4.6 pre-mutation wake)
    w.remove_body(base).expect("remove");
    assert!(!w.is_asleep(top), "받침 제거인데 위 바디가 잠든 채");
    assert!(!w.is_alive(base), "제거된 핸들이 살아있다고 보고됨");
    assert_eq!(w.try_body(base).err(), Some(ApiError::StaleHandle));
    assert_eq!(w.body_count(), 2); // ground + top

    let y0 = w.body(top).pose.pos.y;
    for _ in 0..60 {
        w.step(DT);
    }
    assert!(w.body(top).pose.pos.y < y0 - 0.3, "받침이 없는데 안 떨어짐");

    // 슬롯 재사용: 같은 인덱스라도 세대가 달라 옛 핸들은 거부된다
    let reused = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.2 },
        pose(5.0, 5.0, 0.0),
        1.0,
    ));
    assert_eq!(reused.index, base.index, "최소 인덱스 슬롯 재사용");
    assert_ne!(reused.generation, base.generation, "세대가 증가해야 함");
    assert_eq!(w.try_body(base).err(), Some(ApiError::StaleHandle));
    assert!(w.try_body(reused).is_ok());
    let _ = g;
}

/// MAJOR: `is_asleep`가 raw 인덱스를 읽으면 재사용된 슬롯의 **남의 상태**를 돌려준다.
#[test]
fn is_asleep_rejects_stale_handle() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let h = dyn_box(&mut w, 0.0, 0.5);
    for _ in 0..400 {
        w.step(DT);
    }
    assert!(w.is_asleep(h), "정착한 바디가 잠들지 않음");
    w.remove_body(h).unwrap();
    assert!(!w.is_asleep(h), "stale 핸들이 잠든 상태를 보고함");
}

#[test]
#[should_panic(expected = "time_to_sleep")]
fn negative_time_to_sleep_rejected() {
    World::new(WorldParams {
        time_to_sleep: -1.0,
        ..Default::default()
    });
}

/// BLOCKER: sleep timer는 파생 상태(dt 누적)라 writer/reader 정책이 같아야 한다.
#[test]
fn large_time_to_sleep_survives_roundtrip() {
    let mut w = World::new(WorldParams {
        time_to_sleep: musoe_collide::MAX_INPUT * 0.999,
        ..Default::default()
    });
    ground(&mut w);
    dyn_box(&mut w, 0.0, 0.5);
    for _ in 0..30 {
        w.step(0.2);
    }
    assert!(w.restore(&w.snapshot()).is_ok(), "sleep timer 정책 비대칭");
}

/// BLOCKER: 월드 이탈 바디는 **일반 sleep과 분리**돼야 한다 — 조인트로 연결된
/// 상대가 깨어 있으면 wake 폐쇄가 매 프레임 되살려 이동→고정→sleep→wake를 반복한다.
#[test]
fn out_of_bounds_body_is_not_auto_woken() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let edge = musoe_collide::MAX_INPUT - 5.0;
    let a = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge, 0.0, 0.0),
        1.0,
    ));
    let b = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge - 1.0, 0.0, 0.0),
        1.0,
    ));
    w.add_joint(Joint::point(
        a,
        b,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.0),
    ));
    w.set_linear_velocity(a, Vec3::new(500.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..20 {
        let _ = w.try_step(crate::world::MAX_DT);
    }
    assert!(
        w.is_out_of_bounds(a),
        "경계를 넘은 바디가 OOB로 표시되지 않음"
    );
    // 여러 스텝이 지나도 자동으로 되살아나지 않는다
    for _ in 0..20 {
        let _ = w.try_step(DT);
    }
    assert!(w.is_out_of_bounds(a), "OOB 바디가 자동 wake됨 (진동 경로)");
    assert!(w.restore(&w.snapshot()).is_ok());
    // 호스트가 월드 안으로 되돌리면 해제된다
    w.set_pose(a, pose(0.0, 0.0, 0.0)).unwrap();
    assert!(!w.is_out_of_bounds(a), "set_pose로 OOB가 해제되지 않음");
}

/// MAJOR: `time_to_sleep=0`인 합법 월드에서 OOB가 같은 스텝에 asleep이 되면
/// v4 복원 불변식(`oob ⇒ !asleep`)이 깨져 자기 스냅샷이 복원되지 않는다.
#[test]
fn oob_and_zero_time_to_sleep_keep_the_invariant() {
    let mut w = World::new(WorldParams {
        time_to_sleep: 0.0,
        ..Default::default()
    });
    ground(&mut w);
    let edge = musoe_collide::MAX_INPUT - 5.0;
    let flyer = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge, 50.0, 0.0),
        1.0,
    ));
    w.set_linear_velocity(flyer, Vec3::new(500.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..30 {
        let _ = w.try_step(crate::world::MAX_DT);
        assert!(
            w.restore(&w.snapshot()).is_ok(),
            "oob/asleep 불변식이 깨짐 (time_to_sleep=0)"
        );
    }
    assert!(w.is_out_of_bounds(flyer));
}

/// 수면 임계 런타임 조정: 진입만 게이트하고(관대해지면 더 빨리 잠듦) 잠든 바디는
/// 깨우지 않으며, 파라미터는 캐노니컬이라 스냅샷 왕복에 실린다. 검증은 생성 정본.
#[test]
fn sleep_thresholds_can_be_retuned_at_runtime() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let b = dyn_box(&mut w, 0.0, 0.5);
    for _ in 0..120 {
        w.step(DT);
    }
    assert!(w.is_asleep(b), "정착한 박스가 기본 임계로 잠들어야 한다");
    // 관대한 임계로 조정 — 잠든 바디는 그대로 잠들어 있다(기상 없음).
    w.try_set_sleep_thresholds(0.4, 0.5, 0.25).unwrap();
    w.step(DT);
    assert!(w.is_asleep(b), "임계 조정이 잠든 바디를 깨웠다");
    assert_eq!(w.params().sleep_linear_threshold, 0.4);
    // 스냅샷 왕복 — 파라미터가 캐노니컬로 실린다.
    let snap = w.snapshot();
    let mut w2 = World::new(WorldParams::default());
    w2.restore(&snap).unwrap();
    assert_eq!(w2.params().sleep_linear_threshold, 0.4);
    assert_eq!(snap, w2.snapshot());
    // 검증은 생성과 같은 정본 — NaN·음수는 거부.
    assert!(w.try_set_sleep_thresholds(f32::NAN, 0.5, 0.25).is_err());
    assert!(w.try_set_sleep_thresholds(-1.0, 0.5, 0.25).is_err());
}
