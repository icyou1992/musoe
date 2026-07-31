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
