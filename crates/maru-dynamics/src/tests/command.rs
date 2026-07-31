//! command 관련 테스트.

use super::*;

/// 설계 §7 계약: 쓰기는 커맨드 큐로, **enqueue 순서대로** 적용되고,
/// **쿼리는 pending을 보지 않는다**(암묵 플러시 없음). pending은 캐노니컬 상태다.
#[test]
fn command_buffer_contract() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let b = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 5.0, 0.0),
        1.0,
    ));

    // ① 큐에 쌓아도 쿼리는 이전 상태를 본다
    w.enqueue(Command::SetLinearVelocity(b, Vec3::new(3.0, 0.0, 0.0)))
        .unwrap();
    assert_eq!(w.pending_commands(), 1);
    assert_eq!(w.body(b).linvel.x, 0.0, "pending이 쿼리에 보이면 안 된다");

    // ② 명시 flush로만 반영된다
    let rep = w.flush_commands();
    assert_eq!(rep.applied, 1);
    assert_eq!(w.body(b).linvel.x, 3.0);
    assert_eq!(w.pending_commands(), 0);

    // ③ 순서대로 적용된다 (마지막 값이 남는다)
    w.enqueue(Command::SetLinearVelocity(b, Vec3::new(1.0, 0.0, 0.0)))
        .unwrap();
    w.enqueue(Command::SetLinearVelocity(b, Vec3::new(2.0, 0.0, 0.0)))
        .unwrap();
    w.flush_commands();
    assert_eq!(w.body(b).linvel.x, 2.0, "enqueue 순서가 지켜지지 않음");

    // ④ 잘못된 값은 **enqueue에서** 거부된다 — 큐에 들어간 커맨드는 항상 구조적으로
    //    유효하므로 스냅샷 왕복이 자기 자신에게 복원된다(enqueue/restore 계약 일치).
    assert_eq!(
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(f32::NAN, 0.0, 0.0))),
        Err(ApiError::InvalidValue)
    );
    assert_eq!(w.pending_commands(), 0, "무효 커맨드가 큐에 남음");
    w.enqueue(Command::SetMaterial(b, 0.5, 0.5)).unwrap();
    let rep = w.flush_commands();
    assert_eq!(rep.applied, 1);
    assert_eq!(rep.rejected, 0);
    assert!(w.body(b).linvel.is_finite());

    // ④' enqueue 시점엔 유효했지만 **앞선 커맨드가 대상을 지우면** flush에서 거부된다
    //     (스텝을 중단시키지 않고 카운트만 — 설계 §3 스텝 원자성)
    let doomed = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(9.0, 5.0, 0.0),
        1.0,
    ));
    w.enqueue(Command::RemoveBody(doomed)).unwrap();
    w.enqueue(Command::SetLinearVelocity(doomed, Vec3::new(1.0, 0.0, 0.0)))
        .unwrap();
    let rep = w.flush_commands();
    assert_eq!((rep.applied, rep.rejected), (1, 1));

    // ⑤ step 0단계가 pending을 자동 적용한다
    w.enqueue(Command::SetLinearVelocity(b, Vec3::new(0.0, 0.0, 5.0)))
        .unwrap();
    w.step(DT);
    assert_eq!(w.pending_commands(), 0);
    assert!(w.body(b).pose.pos.z > 0.0, "step이 커맨드를 적용하지 않음");

    // ⑥ pending은 캐노니컬 상태: 스냅샷 왕복 후 재생이 일치한다
    let mut w2 = World::new(WorldParams::default());
    ground(&mut w2);
    w2.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 5.0, 0.0),
        1.0,
    ));
    w2.restore(&w.snapshot()).expect("restore");
    w.enqueue(Command::SetLinearVelocity(b, Vec3::new(-2.0, 0.0, 0.0)))
        .unwrap();
    w2.enqueue(Command::SetLinearVelocity(b, Vec3::new(-2.0, 0.0, 0.0)))
        .unwrap();
    let snap_with_pending = w.snapshot();
    let mut w3 = World::new(WorldParams::default());
    ground(&mut w3);
    w3.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 5.0, 0.0),
        1.0,
    ));
    w3.restore(&snap_with_pending).expect("restore");
    assert_eq!(w3.pending_commands(), 1, "pending이 스냅샷에 실리지 않음");
    for _ in 0..10 {
        w.step(DT);
        w3.step(DT);
    }
    assert_eq!(
        w.state_hash(),
        w3.state_hash(),
        "pending 포함 스냅샷 재생이 원 실행과 분기"
    );
}

/// BLOCKER: 커맨드로 만든 바디를 가리킬 방법이 있어야 한다 —
/// `enqueue_add_body`가 **예약 핸들**을 즉시 돌려주고, 같은 배치의 조인트가 그것을 쓴다.
#[test]
fn enqueue_add_body_returns_handle_usable_in_same_batch() {
    let mut w = World::new(WorldParams::default());
    let anchor = w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 5.0, 0.0),
    ));
    let bob = w.enqueue_add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(1.5, 5.0, 0.0),
        1.0,
    ));
    // 아직 flush 전 — 쿼리에는 보이지 않는다 (설계 §7)
    assert!(!w.is_alive(bob), "pending 바디가 쿼리에 보임");
    w.enqueue(Command::AddJoint(Box::new(Joint::point(
        anchor,
        bob,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    ))))
    .unwrap();
    for _ in 0..120 {
        w.step(DT);
    }
    assert_eq!(w.last_command_report().rejected, 0, "배치가 거부됨");
    assert!(w.is_alive(bob), "예약 핸들이 실현되지 않음");
    // 조인트가 실제로 걸렸다면 앵커에서 1.5m 안쪽에 매달려 있다 (자유낙하면 훨씬 아래)
    let p = w.body(bob).pose.pos;
    assert!(
        (p - Vec3::new(0.0, 5.0, 0.0)).length() < 2.0,
        "조인트 미체결 — bob={p:?}"
    );
}
