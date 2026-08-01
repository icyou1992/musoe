//! events 관련 테스트.

use super::*;

/// 떨어뜨리면 **닿는 순간** Started, 치우면 Stopped가 나와야 한다.
#[test]
fn contact_events_fire_on_touch_and_separation() {
    let mut w = World::new(WorldParams::default());
    let g = w.add_body(Body::new_static(
        maru_collide::Shape::Box {
            half: maru_math::Vec3::new(10.0, 1.0, 10.0),
            radius: 0.0,
        },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
    ));
    let b = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 3.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    let mut started = 0usize;
    for _ in 0..120 {
        w.try_step(1.0 / 60.0).unwrap();
        for e in w.contact_events() {
            if e.kind == ContactEventKind::Started && e.a == g && e.b == b
                || e.kind == ContactEventKind::Started && e.a == b && e.b == g
            {
                started += 1;
            }
        }
        if started > 0 {
            break;
        }
    }
    assert_eq!(started, 1, "닿을 때 Started가 정확히 한 번");

    // 치우면 Stopped
    w.set_pose(
        b,
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 50.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
    )
    .unwrap();
    w.try_step(1.0 / 60.0).unwrap();
    assert!(
        w.contact_events()
            .iter()
            .any(|e| e.kind == ContactEventKind::Stopped),
        "떨어지면 Stopped가 나와야 한다"
    );
}

/// 이벤트는 **스텝 단위 출력**이다 — 다음 스텝이 비운다.
#[test]
fn contact_events_are_cleared_each_step() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        maru_collide::Shape::Box {
            half: maru_math::Vec3::new(10.0, 1.0, 10.0),
            radius: 0.0,
        },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
    ));
    w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 0.5, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    w.try_step(1.0 / 60.0).unwrap();
    assert!(!w.contact_events().is_empty(), "첫 스텝에 닿는다");
    for _ in 0..5 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    assert!(
        w.contact_events().is_empty(),
        "계속 닿아 있는 동안은 새 이벤트가 없다"
    );
}

/// 충돌 그룹이 다르면 **통과**해야 한다.
#[test]
fn collision_groups_filter_contacts() {
    let make = |a_groups: (u32, u32), b_groups: (u32, u32)| -> f32 {
        let mut w = World::new(WorldParams::default());
        w.add_body(
            Body::new_static(
                maru_collide::Shape::Box {
                    half: maru_math::Vec3::new(10.0, 1.0, 10.0),
                    radius: 0.0,
                },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
            )
            .with_groups(a_groups.0, a_groups.1),
        );
        let b = w.add_body(
            Body::new_dynamic(
                maru_collide::Shape::Sphere { radius: 0.5 },
                maru_collide::Pose {
                    pos: maru_math::Vec3::new(0.0, 2.0, 0.0),
                    rot: maru_math::Quat::IDENTITY,
                },
                1.0,
            )
            .with_groups(b_groups.0, b_groups.1),
        );
        for _ in 0..90 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        w.body(b).pose().pos.y
    };
    let landed = make((1, 1), (1, 1));
    assert!(landed > 0.3, "같은 그룹이면 바닥에 서야 한다: {landed}");
    let fell = make((1, 1), (2, 2));
    assert!(fell < -1.0, "다른 그룹이면 통과해야 한다: {fell}");
    // **양방향**: 한쪽만 상대를 무시해도 통과해야 한다
    let one_way = make((1, 1), (2, 1));
    assert!(one_way < -1.0, "한쪽이 무시하면 양쪽 다 통과다: {one_way}");
}

/// 센서는 **이벤트만** 내고 밀지 않는다.
#[test]
fn sensors_report_but_do_not_push() {
    let mut w = World::new(WorldParams::default());
    let trigger = w.add_body(
        Body::new_static(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(5.0, 0.5, 5.0),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 1.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        )
        .as_sensor(true),
    );
    let b = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 4.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    let mut saw = false;
    for _ in 0..120 {
        w.try_step(1.0 / 60.0).unwrap();
        saw |= w
            .contact_events()
            .iter()
            .any(|e| e.kind == ContactEventKind::Started && (e.a == trigger || e.b == trigger));
    }
    assert!(saw, "센서를 지나가면 이벤트가 나와야 한다");
    assert!(
        w.body(b).pose().pos.y < -2.0,
        "센서는 밀지 않는다 — 그대로 떨어져야 한다: {}",
        w.body(b).pose().pos.y
    );
}

/// 슬롯이 재사용돼도 **닿은 적 없는 물체에 종료 이벤트가 나가면 안 된다**.
#[test]
fn removed_body_does_not_leak_events_to_a_reused_slot() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        maru_collide::Shape::Box {
            half: maru_math::Vec3::new(10.0, 1.0, 10.0),
            radius: 0.0,
        },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, -1.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
    ));
    let a = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 0.5, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    w.try_step(1.0 / 60.0).unwrap();
    assert!(
        !w.contact_events().is_empty(),
        "먼저 닿아야 의미 있는 테스트다"
    );

    w.remove_body(a).unwrap();
    // 같은 슬롯을 재사용하는 새 물체 — 바닥에서 멀리 띄워 닿지 않게 둔다
    let b = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 40.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    w.try_step(1.0 / 60.0).unwrap();
    assert!(
        !w.contact_events().iter().any(|e| e.a == b || e.b == b),
        "새 물체는 아직 아무것도 안 닿았다: {:?}",
        w.contact_events()
    );
}

/// 센서는 **기본적으로 광선에 안 맞는다** — 총알이 트리거에 막히면 안 된다.
#[test]
fn sensors_are_excluded_from_queries_by_default() {
    let mut w = World::new(WorldParams::default());
    let trigger = w.add_body(
        Body::new_static(
            maru_collide::Shape::Sphere { radius: 1.0 },
            maru_collide::Pose {
                pos: maru_math::Vec3::ZERO,
                rot: maru_math::Quat::IDENTITY,
            },
        )
        .as_sensor(true),
    );
    let o = maru_math::Vec3::new(-10.0, 0.0, 0.0);
    let d = maru_math::Vec3::new(1.0, 0.0, 0.0);
    assert!(w.cast_ray(o, d, 100.0).unwrap().is_none(), "기본은 제외");
    assert_eq!(
        w.cast_ray_filtered(o, d, 100.0, QueryFilter::ALL.with_sensors())
            .unwrap()
            .map(|h| h.body),
        Some(trigger),
        "명시하면 맞는다"
    );
}

/// 접촉 이벤트 버퍼가 **스텝 중에 자라지 않는다** — 설계 §6 "스텝 중 할당 0"과
/// "Err면 상태 불변"이 이 경로에서도 성립한다는 관측 가능한 증거.
///
/// 예전에는 `started`/`stopped`를 이벤트 구간에서 `collect()`로 잡고
/// `contact_events`도 그 자리에서 자랐다. 그 지점은 **wake·솔브가 끝난 뒤**라, 할당이
/// 실패하면 이미 수면·자세·기준선이 바뀐 상태에서 오류를 내게 된다(되돌릴 수 없다).
/// 지금은 셋 다 스텝 시작의 통합 fallible 블록에서 선점한다.
///
/// 관측 방법: 이벤트가 **계속 발생하는** 장면에서 용량이 한 번 안정된 뒤 더는 변하지
/// 않아야 한다. 용량이 뛰면 그 스텝의 이벤트 구간에서 재할당이 있었다는 뜻이다.
#[test]
fn contact_event_buffers_do_not_grow_during_a_step() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    // 반발 계수를 준 공 여러 개 — 바닥과 접촉을 끊임없이 만들고 끊는다.
    for i in 0..12 {
        w.add_body(
            Body::new_dynamic(
                Shape::Sphere { radius: 0.3 },
                pose(i as f32 * 1.5 - 8.0, 1.0 + (i % 4) as f32 * 0.7, 0.0),
                1.0,
            )
            .with_restitution(0.9)
            .expect("반발 계수는 0..=1"),
        );
    }
    // 핵심 불변식(`started + stopped <= capacity`)은 `world.rs`의 `debug_assert!`가
    // **이벤트 구간 직전에** 검사한다 — 이 테스트는 그 검사가 실제로 발화하도록
    // 이벤트가 계속 나는 장면을 돌리는 역할이다(디버그 빌드에서 게이트가 돈다).
    // 여기서는 용량이 **단조 증가만** 하는지(선점 블록에서만 자라는지)를 함께 본다.
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let mut cap = w.contact_events_capacity();
    let mut event_steps = 0usize;
    let mut max_events = 0usize;
    for step in 0..300 {
        w.step(1.0 / 60.0);
        let n = w.contact_events().len();
        if n > 0 {
            event_steps += 1;
            max_events = max_events.max(n);
        }
        let now = w.contact_events_capacity();
        // **방출 진입 시점의 용량과 스텝 종료 시점의 용량이 같아야 한다.**
        // 다르면 방출 중에 자란 것이다 = 재할당 = 계약 위반. 스텝 시작의 선점 블록에서
        // 자라는 것은 정상이므로 이 비교여야만 둘이 구분된다.
        // (스텝 후 용량만 보거나 `len <= capacity`를 보는 검사는 어떤 Vec에서도 참이라
        //  아무것도 잠그지 못한다 — 첫 판이 그랬고 교차검수가 MAJOR로 잡았다.)
        assert_eq!(
            now,
            w.contact_events_capacity_at_emit(),
            "step {step}: 이벤트 방출 중에 버퍼가 자랐다 (재할당 = 스텝 중 할당 0 위반)"
        );
        assert!(
            now >= cap,
            "step {step}: 이벤트 버퍼 용량이 줄었다 ({cap} → {now})"
        );
        cap = now;
    }
    // **커버리지**: 이벤트가 실제로 나야 위 단언이 무언가를 검증한다.
    // (이벤트가 0이면 "자라지 않았다"는 당연히 참이고 테스트는 아무것도 잠그지 않는다.)
    assert!(
        event_steps >= 10 && max_events > 0,
        "이벤트가 거의 없어 검증이 성립하지 않는다: {event_steps}스텝, 최대 {max_events}건"
    );
}

// ---------- 접촉력 이벤트 ----------

/// 지면 + 그 위 박스. `threshold`는 지면에만 건다(페어 임계는 min이므로 그걸로 충분하다).
fn force_world(threshold: f32) -> (World, BodyHandle) {
    let params = WorldParams {
        gravity: Vec3::new(0.0, -10.0, 0.0),
        // 수면이 켜져 있으면 정착 후 접촉이 사라져 힘이 0이 된다.
        ..no_sleep_params()
    };
    let mut w = World::new(params);
    w.add_body(
        Body::new_static(
            Shape::Box {
                half: Vec3::new(50.0, 1.0, 50.0),
                radius: 0.0,
            },
            pose(0.0, -1.0, 0.0),
        )
        .with_contact_force_threshold(threshold),
    );
    // 반크기 0.5 박스, 밀도 1 → 부피 1.0, 질량 1.0. 무게 = 10 N.
    let b = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 0.6, 0.0),
        1.0,
    ));
    (w, b)
}

/// 임계를 켠 물체가 없으면 이벤트가 아예 없다 — 기본이 꺼짐이라는 계약.
#[test]
fn contact_force_events_are_off_by_default() {
    let (mut w, _) = force_world(f32::INFINITY);
    for _ in 0..120 {
        w.step(DT);
        assert!(w.contact_force_events().is_empty());
    }
}

/// 임계를 넘긴 접촉이 실린다. 받치는 힘은 대략 무게(m·g)여야 한다.
#[test]
fn contact_force_reports_weight_at_rest() {
    let (mut w, b) = force_world(0.1);
    let mut seen: Option<f32> = None;
    for _ in 0..240 {
        w.step(DT);
        if let Some(e) = w.contact_force_events().first() {
            seen = Some(e.total_force_magnitude);
            let l = e.max_force_direction.length();
            assert!((l - 1.0).abs() < 1e-3, "최대 힘 방향이 단위가 아니다: {l}");
            assert!(e.max_force_magnitude <= e.total_force_magnitude + 1e-3);
            assert!(e.a == b || e.b == b, "이벤트 대상이 다르다");
            // 총 힘 벡터의 크기가 보고된 크기와 일치해야 한다.
            assert!((e.total_force.length() - e.total_force_magnitude).abs() < 1e-3);
        }
    }
    let f = seen.expect("정착한 박스의 접촉력 이벤트가 없다");
    let want = 10.0f32; // 질량 1.0 × 중력 10
    assert!(
        (f - want).abs() < want * 0.25,
        "받치는 힘이 무게와 다르다: {f} (기대 ≈ {want})"
    );
}

/// 임계가 실제로 문지기 역할을 한다 — 높이면 조용해져야 한다.
#[test]
fn contact_force_threshold_gates_the_event() {
    let run = |threshold: f32| {
        let (mut w, _) = force_world(threshold);
        let mut any = false;
        for _ in 0..240 {
            w.step(DT);
            any |= !w.contact_force_events().is_empty();
        }
        any
    };
    assert!(run(0.1), "낮은 임계에서 이벤트가 없다");
    assert!(!run(1.0e6), "높은 임계인데 이벤트가 났다");
}

/// 임계는 바디 상태다 — 스냅샷 왕복에서 살아남아야 한다(기본값 무한대 포함).
#[test]
fn contact_force_threshold_survives_snapshot() {
    let mut w = World::new(WorldParams::default());
    let a = w.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 3.0, 0.0), 1.0)
            .with_contact_force_threshold(42.5),
    );
    let b = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 1.0 },
        pose(5.0, 3.0, 0.0),
        1.0,
    ));
    let snap = w.snapshot();
    let mut w2 = World::new(WorldParams::default());
    w2.restore(&snap).expect("복원");
    assert_eq!(w2.body(a).contact_force_threshold(), 42.5);
    assert_eq!(
        w2.body(b).contact_force_threshold(),
        f32::INFINITY,
        "기본값(꺼짐)이 왕복에서 살아남지 못했다"
    );
}

/// 음수·NaN 임계는 거절한다 — NaN은 비교가 항상 거짓이라 **조용히 꺼진다**.
#[test]
fn invalid_thresholds_are_rejected() {
    for bad in [-1.0f32, f32::NAN, f32::NEG_INFINITY] {
        let b = Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 3.0, 0.0), 1.0)
            .with_contact_force_threshold(bad);
        assert!(b.validate().is_err(), "임계 {bad}를 통과시켰다");
    }
    // 0과 무한대는 정상 값이다.
    for ok in [0.0f32, 1.0, f32::INFINITY] {
        let b = Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 3.0, 0.0), 1.0)
            .with_contact_force_threshold(ok);
        assert!(b.validate().is_ok(), "임계 {ok}를 거절했다");
    }
}

/// **보고되는 힘은 서브스텝 수에 무관해야 한다.**
///
/// 접점 임펄스는 스텝 전체가 아니라 한 서브스텝 분이라, 스텝 dt로 나누면 서브스텝 수만큼
/// 작게 나온다. 서브스텝만 바꿔 같은 장면을 두 번 돌리면 그 실수가 바로 드러난다
/// (처음 구현이 4서브스텝에서 10 N을 2.5 N으로 보고했다).
#[test]
fn contact_force_is_independent_of_substep_count() {
    let measure = |substeps: u32| -> f32 {
        let params = WorldParams {
            gravity: Vec3::new(0.0, -10.0, 0.0),
            substeps,
            ..no_sleep_params()
        };
        let mut w = World::new(params);
        w.add_body(
            Body::new_static(
                Shape::Box {
                    half: Vec3::new(50.0, 1.0, 50.0),
                    radius: 0.0,
                },
                pose(0.0, -1.0, 0.0),
            )
            .with_contact_force_threshold(0.1),
        );
        w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.6, 0.0),
            1.0,
        ));
        let mut last = 0.0;
        for _ in 0..300 {
            w.step(DT);
            if let Some(e) = w.contact_force_events().first() {
                last = e.total_force_magnitude;
            }
        }
        last
    };
    let (a, b) = (measure(2), measure(8));
    assert!(a > 1.0 && b > 1.0, "정착 힘이 안 잡혔다: {a} / {b}");
    assert!(
        (a - b).abs() < a.max(b) * 0.2,
        "서브스텝 수가 보고 값을 바꾼다: 2서브스텝 {a} vs 8서브스텝 {b}"
    );
}

/// **마찰 성분이 총 힘에 들어가야 한다.** 미끄러지는 물체는 법선 힘 말고도 접선 힘을 받는다.
/// 법선만 세면 그 차이가 보고에서 사라진다.
#[test]
fn contact_force_includes_the_friction_component() {
    let measure = |vx: f32| -> Vec3 {
        let params = WorldParams {
            gravity: Vec3::new(0.0, -10.0, 0.0),
            ..no_sleep_params()
        };
        let mut w = World::new(params);
        w.add_body(
            Body::new_static(
                Shape::Box {
                    half: Vec3::new(50.0, 1.0, 50.0),
                    radius: 0.0,
                },
                pose(0.0, -1.0, 0.0),
            )
            .with_contact_force_threshold(0.1),
        );
        let b = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5, 0.0),
            1.0,
        ));
        // 먼저 정착시킨 뒤 옆으로 민다 — 미끄러지는 동안 마찰 임펄스가 돈다.
        for _ in 0..120 {
            w.step(DT);
        }
        w.set_linear_velocity(b, Vec3::new(vx, 0.0, 0.0)).unwrap();
        let mut best = Vec3::ZERO;
        for _ in 0..10 {
            w.step(DT);
            if let Some(e) = w.contact_force_events().first() {
                if e.total_force.x.abs() > best.x.abs() {
                    best = e.total_force;
                }
            }
        }
        best
    };
    let still = measure(0.0);
    let sliding = measure(8.0);
    // 정지 상태에서는 접선 힘이 거의 없고, 미끄러질 때는 뚜렷해야 한다.
    assert!(
        still.x.abs() < 1.0,
        "정지 상태인데 접선 힘이 크다: {still:?}"
    );
    assert!(
        sliding.x.abs() > 1.0,
        "미끄러지는데 접선 힘이 안 잡혔다: {sliding:?}"
    );
}
