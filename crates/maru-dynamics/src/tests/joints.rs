//! joints 관련 테스트.

use super::*;

/// **고정 조인트는 자세까지 묶는다** — 중력이 걸려도 상대 회전이 안 생겨야 한다.
///
/// 앵커를 B의 **무게중심에서 떨어뜨린다**(레버암). 앵커가 무게중심에 있으면 중력이
/// 토크를 만들지 않아, 각 제약이 아예 없는 Point 조인트로도 이 테스트가 통과한다 —
/// 실제로 그랬고, 그래서 부호가 뒤집힌 Fixed가 녹색으로 통과했다
/// (검수 `review_existing_joint_tests_pass_without_any_angular_constraint`).
#[test]
fn fixed_joint_holds_relative_orientation() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    w.try_add_joint(
        Joint::try_fixed(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            // B의 무게중심이 아니라 **끝단**을 잡는다 → 중력이 토크를 만든다
            maru_math::Vec3::new(-0.3, 0.0, 0.0),
            maru_math::Quat::IDENTITY,
        )
        .unwrap(),
    )
    .unwrap();
    for _ in 0..300 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let q = w.body(b).pose().rot;
    // 상대 자세가 항등이어야 하므로 |w| ≈ 1
    assert!(
        q.w.abs() > 0.995,
        "고정인데 회전했다: {q:?} (|w|={})",
        q.w.abs()
    );
    let p = w.body(b).pose().pos;
    assert!(
        (p - maru_math::Vec3::new(1.3, 10.0, 0.0)).length() < 0.1,
        "위치도 붙어 있어야 한다: {p:?}"
    );
}

/// 거리 조인트는 **부등식**이다: 구간 안에서는 자유롭고 max를 넘으면 당긴다.
#[test]
fn distance_joint_is_a_rope_not_a_rod() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(0.0, -0.5, 0.0));
    w.try_add_joint(
        Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 0.0, 2.0).unwrap(),
    )
    .unwrap();
    // 처음엔 0.5 떨어져 있으니 자유낙하해야 한다
    w.try_step(1.0 / 60.0).unwrap();
    assert!(
        w.body(b).linvel().y < -0.05,
        "구간 안에서는 자유낙하: {:?}",
        w.body(b).linvel()
    );
    for _ in 0..600 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let d = (w.body(b).pose().pos - maru_math::Vec3::new(0.0, 10.0, 0.0)).length();
    assert!(
        (d - 2.0).abs() < 0.1,
        "max에서 멈춰야 한다 (로프): 거리={d}"
    );
}

/// `min == max`면 막대다 — 양방향으로 붙잡는다.
#[test]
fn distance_joint_with_equal_bounds_is_a_rod() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(0.0, -1.0, 0.0));
    w.try_add_joint(
        Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    for _ in 0..600 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let d = (w.body(b).pose().pos - maru_math::Vec3::new(0.0, 10.0, 0.0)).length();
    assert!((d - 1.0).abs() < 0.08, "막대 길이 유지: {d}");
}

/// hinge 한계는 문이 벽을 뚫고 가지 않게 한다.
#[test]
fn hinge_limit_stops_rotation() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
    w.try_add_joint(
        Joint::try_hinge(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Vec3::ZERO,
            axis,
            axis,
        )
        .unwrap()
        .with_limit(-0.3, 0.3)
        .unwrap(),
    )
    .unwrap();
    // 초기 각속도를 줘서 한계에 **부딪히게** 만든다 — 앵커가 무게중심에 있으면
    // 중력만으로는 토크가 안 생겨 한계가 시험되지 않는다.
    w.set_angular_velocity(b, maru_math::Vec3::new(0.0, 0.0, 6.0))
        .unwrap();
    for _ in 0..600 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    // 한계가 없으면 중력에 끌려 크게 돌아간다. 한계 안에 머물러야 한다.
    let q = w.body(b).pose().rot;
    let ang = 2.0 * q.z.abs().atan2(q.w.abs());
    assert!(ang < 0.6, "한계를 크게 넘었다: {ang} rad");
}

/// 모터는 목표 각속도를 추종하되 `max_impulse`를 넘지 않는다.
#[test]
fn hinge_motor_drives_toward_target_speed() {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
    w.try_add_joint(
        Joint::try_hinge(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Vec3::ZERO,
            axis,
            axis,
        )
        .unwrap()
        .with_motor(3.0, 100.0)
        .unwrap(),
    )
    .unwrap();
    for _ in 0..240 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let wz = w.body(b).angvel().z;
    // **부호까지 본다.** `|wz∓3|` 둘 중 하나만 보면 모터 방향 규약이 시험되지 않는다.
    assert!(
        (wz - 3.0).abs() < 0.5,
        "목표 각속도 +3 rad/s를 부호까지 추종해야 한다: {wz}"
    );
}

/// 새 조인트 종류·한계·모터가 전부 스냅샷을 왕복해야 한다.
#[test]
fn new_joint_kinds_survive_snapshot() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
    w.try_add_joint(
        Joint::try_fixed(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Vec3::ZERO,
            maru_math::Quat::IDENTITY,
        )
        .unwrap(),
    )
    .unwrap();
    w.try_add_joint(
        Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 0.5, 2.0).unwrap(),
    )
    .unwrap();
    w.try_add_joint(
        Joint::try_hinge(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Vec3::ZERO,
            axis,
            axis,
        )
        .unwrap()
        .with_limit(-1.0, 1.0)
        .unwrap()
        .with_motor(2.0, 50.0)
        .unwrap(),
    )
    .unwrap();
    for _ in 0..30 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let snap = w.try_snapshot().unwrap();
    let mut r = World::new(WorldParams::default());
    r.restore(&snap).unwrap();
    assert_eq!(r.try_state_hash().unwrap(), w.try_state_hash().unwrap());
    for _ in 0..30 {
        w.try_step(1.0 / 60.0).unwrap();
        r.try_step(1.0 / 60.0).unwrap();
    }
    assert_eq!(
        r.try_state_hash().unwrap(),
        w.try_state_hash().unwrap(),
        "복원 후 미래가 갈리면 안 된다"
    );
}

/// 잘못된 조인트 파라미터는 거부한다.
#[test]
fn new_joint_parameters_are_validated() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let z = maru_math::Vec3::ZERO;
    // min > max
    assert!(Joint::try_distance(a, b, z, z, 2.0, 1.0).is_err());
    // 음수 거리
    assert!(Joint::try_distance(a, b, z, z, -1.0, 1.0).is_err());
    // NaN
    assert!(Joint::try_distance(a, b, z, z, 0.0, f32::NAN).is_err());
    // 비단위 쿼터니언
    assert!(Joint::try_fixed(
        a,
        b,
        z,
        z,
        maru_math::Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0
        }
    )
    .is_err());
    // 한계·모터는 hinge에만
    let p = Joint::try_point(a, b, z, z).unwrap();
    assert!(p.clone().with_limit(-1.0, 1.0).is_err());
    assert!(p.with_motor(1.0, 1.0).is_err());
}

// ─────────────── Prismatic (슬라이더) — M4 조인트 세트 ───────────────

/// 슬라이더 위 바디는 **축 방향으로만** 움직인다.
///
/// 중력을 축과 수직으로 두면(축 = x, 중력 = −y) 자유 슬라이더는 y로 떨어지지 않아야
/// 한다. 떨어지면 수직 2-DOF 구속이 안 걸린 것이다.
#[test]
fn prismatic_confines_motion_to_its_axis() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let axis = maru_math::Vec3::new(1.0, 0.0, 0.0);
    w.try_add_joint(
        Joint::try_prismatic(
            a,
            b,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            axis,
            maru_math::Quat::IDENTITY,
        )
        .unwrap(),
    )
    .unwrap();
    let y0 = w.body(b).pose().pos.y;
    for _ in 0..300 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let p = w.body(b).pose().pos;
    assert!(
        (p.y - y0).abs() < 0.02,
        "축에 수직인 중력에 끌려 내려갔다: y {y0} → {}",
        p.y
    );
    assert!(p.z.abs() < 0.02, "축 밖 z로 벗어났다: {}", p.z);
}

/// 슬라이더는 상대 자세를 잠근다 — 축 방향 병진만 남긴다.
#[test]
fn prismatic_locks_relative_orientation() {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    w.try_add_joint(
        Joint::try_prismatic(
            a,
            b,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Quat::IDENTITY,
        )
        .unwrap(),
    )
    .unwrap();
    // 축 둘레로 세게 돌려도 각 3-DOF 잠금이 되돌려야 한다.
    w.set_angular_velocity(b, maru_math::Vec3::new(5.0, 3.0, -2.0))
        .unwrap();
    for _ in 0..300 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let q = w.body(b).pose().rot;
    let angle = 2.0 * (q.x * q.x + q.y * q.y + q.z * q.z).sqrt().min(1.0).asin();
    assert!(angle < 0.1, "상대 자세가 풀렸다: {angle} rad");
}

/// 축 방향 한계는 슬라이더를 구간 안에 묶는다.
#[test]
fn prismatic_limit_bounds_travel() {
    let mut w = World::new(WorldParams {
        // 중력을 **축 방향**으로 둔다 — 한계를 실제로 밀어붙이는 하중이 필요하다.
        gravity: maru_math::Vec3::new(-9.81, 0.0, 0.0),
        ..WorldParams::default()
    });
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(0.0, 0.0, 0.0));
    let jh = w
        .try_add_joint(
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Quat::IDENTITY,
            )
            .unwrap()
            .with_limit(-0.5, 0.5)
            .unwrap(),
        )
        .unwrap();
    for _ in 0..600 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let x = w.body(b).pose().pos.x - w.body(a).pose().pos.x;
    assert!(
        x > -0.6,
        "한계가 없으면 계속 미끄러진다 — 하한을 크게 넘었다: {x}"
    );
    assert!(x < 0.6, "상한을 넘었다: {x}");
    // **호스트가 읽는 좌표와 물리가 같은 식이어야 한다** — 표시값이 따로 놀면
    // "문이 얼마나 열렸나"가 한계와 어긋난다.
    let s = w.prismatic_translation(jh).expect("prismatic 좌표");
    assert!(
        (s - x).abs() < 1.0e-4,
        "World가 보고하는 슬라이더 좌표({s})가 실제 변위({x})와 다르다"
    );
    // 다른 종류 조인트는 None이어야 한다.
    let jp = w
        .try_add_joint(
            Joint::try_point(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO).unwrap(),
        )
        .unwrap();
    assert!(w.prismatic_translation(jp).is_none());
    assert!(w.prismatic_translation(9999).is_none());
}

/// 선형 모터는 목표 속도를 **부호까지** 추종한다.
#[test]
fn prismatic_motor_drives_toward_target_speed() {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::ZERO);
    w.try_add_joint(
        Joint::try_prismatic(
            a,
            b,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Quat::IDENTITY,
        )
        .unwrap()
        .with_motor(2.0, 100.0)
        .unwrap(),
    )
    .unwrap();
    for _ in 0..240 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let vx = w.body(b).linvel().x;
    // 목표는 **A 기준 B**의 축 방향 속도다. A가 static이므로 그대로 vx여야 한다.
    assert!(
        (vx - 2.0).abs() < 0.3,
        "목표 속도 +2 m/s를 부호까지 추종해야 한다: {vx}"
    );
}

/// 선형 모터의 최대 힘도 **서브스텝 수에 무관**해야 한다 (hinge 모터와 같은 계약).
#[test]
fn prismatic_motor_effort_is_independent_of_substeps() {
    fn push(substeps: u32) -> f32 {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            substeps,
            ..WorldParams::default()
        });
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::ZERO);
        w.try_add_joint(
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Quat::IDENTITY,
            )
            .unwrap()
            // 목표를 크게 잡아 모터를 항상 포화시킨다 → 전달량 = 상한 × dt
            .with_motor(50.0, 0.2)
            .unwrap(),
        )
        .unwrap();
        w.try_step(1.0 / 60.0).unwrap();
        w.body(b).linvel().x
    }
    let s1 = push(1);
    let s8 = push(8);
    assert!(
        (s1 - s8).abs() < 0.1 * s1.abs().max(1e-6),
        "substeps만 바꿨는데 모터 전달 임펄스가 달라진다: {s1} vs {s8}"
    );
}

/// prismatic도 스냅샷을 왕복하고 복원 후 미래가 갈리지 않아야 한다.
#[test]
fn prismatic_survives_snapshot() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    w.try_add_joint(
        Joint::try_prismatic(
            a,
            b,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::new(0.0, 1.0, 0.0),
            maru_math::Quat::IDENTITY,
        )
        .unwrap()
        .with_limit(-1.0, 1.0)
        .unwrap()
        .with_motor(0.5, 25.0)
        .unwrap(),
    )
    .unwrap();
    for _ in 0..30 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let snap = w.try_snapshot().unwrap();
    let mut r = World::new(WorldParams::default());
    r.restore(&snap).unwrap();
    assert_eq!(r.try_state_hash().unwrap(), w.try_state_hash().unwrap());
    for _ in 0..30 {
        w.try_step(1.0 / 60.0).unwrap();
        r.try_step(1.0 / 60.0).unwrap();
    }
    assert_eq!(
        r.try_state_hash().unwrap(),
        w.try_state_hash().unwrap(),
        "복원 후 미래가 갈리면 안 된다"
    );
}

/// **축이 회전하면 캐시된 수직 반력이 자유축으로 샌다** — warm start가 그걸
/// 그대로 실으면 슬라이더가 스스로 미끄러진다 (교차검수 rev_d2e60241 MAJOR).
///
/// 솔버 상태를 직접 만들어 그 상황을 재현한다: 누적 임펄스를 **현재 자유축과
/// 나란하게** 두고 warm start를 돌린 뒤, 자유축 상대속도가 그대로인지 본다.
#[test]
fn prismatic_warm_start_does_not_leak_into_the_free_axis() {
    use crate::solver::SolverState;
    let axis = maru_math::Vec3::new(1.0, 0.0, 0.0);
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::ZERO);
    let mut j = Joint::try_prismatic(
        a,
        b,
        maru_math::Vec3::ZERO,
        maru_math::Vec3::ZERO,
        axis,
        maru_math::Quat::IDENTITY,
    )
    .unwrap();
    // 이전 프레임에 축이 다른 방향이었을 때 쌓인 수직 반력 — 지금은 자유축과 나란하다.
    j.linear_impulse = axis * 5.0;

    let mut st = SolverState::default();
    st.reserve_for(2).unwrap();
    st.lin = vec![maru_math::Vec3::ZERO; 2];
    st.ang = vec![maru_math::Vec3::ZERO; 2];
    st.pos = vec![maru_math::Vec3::ZERO, maru_math::Vec3::new(1.0, 0.0, 0.0)];
    st.rot = vec![maru_math::Quat::IDENTITY; 2];
    st.inv_m = vec![0.0, 1.0]; // A는 static, B는 dynamic
    st.inv_i = vec![maru_math::Mat3::from_diagonal(maru_math::Vec3::ZERO); 2];
    st.inv_i[1] = maru_math::Mat3::from_diagonal(maru_math::Vec3::new(1.0, 1.0, 1.0));
    st.dynamic = vec![false, true];

    crate::joint::warm_start(&mut st, &j);
    let free = axis.dot(st.lin[1] - st.lin[0]);
    assert!(
        free.abs() < 1.0e-6,
        "warm start가 자유축 속도를 주입했다: {free} m/s"
    );
}

/// 비단위(허용 범위 안) `rel_rot`은 **생성 시점에** 캐노니컬화돼야 한다 —
/// 스냅샷 리더가 정규화하므로, 안 하면 왕복이 항등이 아니다.
#[test]
fn joint_rel_rot_is_canonicalized_so_snapshots_round_trip() {
    // norm² = 0.9549 — validate 허용 범위(0.9~1.1) 안이지만 **단위가 아니다**.
    // 경계값(0.9 정확)은 f32 반올림으로 검증에 걸릴 수 있어 안쪽을 고른다.
    // 항등이 아닌 회전을 써야 정규화가 값을 실제로 바꾼다.
    let loose = maru_math::Quat {
        x: 0.3,
        y: 0.0,
        z: 0.0,
        w: 0.93,
    };
    for prismatic in [false, true] {
        let mut w = World::new(WorldParams::default());
        let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
        let j = if prismatic {
            Joint::try_prismatic(
                a,
                b,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::ZERO,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                loose,
            )
            .unwrap()
        } else {
            Joint::try_fixed(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, loose).unwrap()
        };
        w.try_add_joint(j).unwrap();
        for _ in 0..10 {
            w.try_step(1.0 / 60.0).unwrap();
        }
        let snap = w.try_snapshot().unwrap();
        let mut r = World::new(WorldParams::default());
        r.restore(&snap).unwrap();
        assert_eq!(
            r.try_state_hash().unwrap(),
            w.try_state_hash().unwrap(),
            "prismatic={prismatic}: 비단위 rel_rot이 왕복에서 갈렸다"
        );
        for _ in 0..10 {
            w.try_step(1.0 / 60.0).unwrap();
            r.try_step(1.0 / 60.0).unwrap();
        }
        assert_eq!(
            r.try_state_hash().unwrap(),
            w.try_state_hash().unwrap(),
            "prismatic={prismatic}: 복원 후 미래가 갈렸다"
        );
    }
}

/// prismatic 파라미터 검증 — 축·자세·한계·모터 전부.
#[test]
fn prismatic_parameters_are_validated() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let z = maru_math::Vec3::ZERO;
    let x = maru_math::Vec3::new(1.0, 0.0, 0.0);
    let id = maru_math::Quat::IDENTITY;
    // 0 축 / NaN 축 — **생성 시점에** 걸려야 한다(정규화가 0을 살려내면 안 된다).
    assert!(Joint::try_prismatic(a, b, z, z, z, id).is_err());
    assert!(
        Joint::try_prismatic(a, b, z, z, maru_math::Vec3::new(f32::NAN, 0.0, 0.0), id).is_err()
    );
    // 비단위 rel_rot
    assert!(Joint::try_prismatic(
        a,
        b,
        z,
        z,
        x,
        maru_math::Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0
        }
    )
    .is_err());
    let ok = Joint::try_prismatic(a, b, z, z, x, id).unwrap();
    // lo > hi
    assert!(ok.clone().with_limit(1.0, -1.0).is_err());
    // NaN 한계
    assert!(ok.clone().with_limit(0.0, f32::NAN).is_err());
    // 범위 밖 한계 (앵커 = 셰이프 범위)
    assert!(ok.clone().with_limit(-1.0, 1.0e9).is_err());
    // 음수 최대 힘 / NaN 목표 속도
    assert!(ok.clone().with_motor(1.0, -1.0).is_err());
    assert!(ok.clone().with_motor(f32::NAN, 1.0).is_err());
    // 정상값은 통과
    assert!(ok
        .with_limit(-1.0, 1.0)
        .unwrap()
        .with_motor(1.0, 5.0)
        .is_ok());
}

/// point 조인트: 흔들리는 진자 — 제약 오차(앵커 간 거리)가 계속 작아야 함
#[test]
fn point_joint_pendulum_holds() {
    let mut w = World::new(WorldParams::default());
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.1 },
        pose(0.0, 5.0, 0.0),
    ));
    let bob = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.2, 0.2, 0.2),
            radius: 0.0,
        },
        pose(1.0, 5.0, 0.0), // 수평에서 놓아 흔들림 시작
        1.0,
    ));
    w.add_joint(Joint::point(
        anchor,
        bob,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0), // bob 로컬 (-1,0,0)이 앵커에 붙음
    ));
    let mut max_err: f32 = 0.0;
    for _ in 0..300 {
        w.step(DT);
        let b = w.body(bob);
        let attach = b.pose.pos + b.pose.rot.rotate(Vec3::new(-1.0, 0.0, 0.0));
        max_err = max_err.max((attach - Vec3::new(0.0, 5.0, 0.0)).length());
    }
    let b = w.body(bob);
    assert!(max_err < 0.03, "제약 오차 {max_err}");
    assert!(
        b.pose.pos.y < 5.0,
        "진자가 흔들려 내려와야 함: {:?}",
        b.pose.pos
    );
}

/// 리뷰 지적 검증: hinge 각 제약의 **부호**. cross product는 반평행에서도 0이라
/// |cos|만 보면 부호를 판별할 수 없다 — 작은 초기 오차(20°)에서 시작해
/// **오차가 줄어드는지**로 판별한다(부호가 반대면 커져서 반대축으로 넘어간다).
#[test]
fn hinge_reduces_small_axis_error() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    });
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.05 },
        pose(0.0, 0.0, 0.0),
    ));
    // B를 z축 기준 20° 돌려 놓고, hinge 축은 양쪽 다 +X로 지정
    let tilt = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 20f32.to_radians());
    let b = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.1, 0.1),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::ZERO,
            rot: tilt,
        },
        1.0,
    ));
    w.add_joint(Joint::hinge(
        anchor,
        b,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    ));
    let align = |w: &World| {
        w.body(b)
            .pose
            .rot
            .rotate(Vec3::new(1.0, 0.0, 0.0))
            .dot(Vec3::new(1.0, 0.0, 0.0))
    };
    let start = align(&w); // cos20° ≈ 0.94
    for _ in 0..120 {
        w.step(DT);
    }
    let end = align(&w);
    assert!(
        end > start && end > 0.999,
        "hinge 축 오차가 줄지 않음: {start} → {end} (부호 반대 의심)"
    );
}

/// M2 수용 기준: hinge 체인 20링크 + 접촉 혼합 수렴 (설계 §11 M2).
/// 체인을 **바닥에 눕힌 상태**로 검증한다 — 전 링크가 동시에 접촉 제약과 조인트 제약을
/// 받는 결정적 시나리오. (공중에서 떨어뜨리면 혼돈계라 더미 형태가 플랫폼마다 달라
/// 판정 기준으로 쓸 수 없다 — 낙하 케이스는 아래 무발산 테스트가 담당.)
#[test]
fn hinge_chain_20_links_on_ground() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.05 },
        pose(0.0, 0.1, 0.0),
    ));
    let mut prev = anchor;
    let mut links = Vec::new();
    for i in 0..20 {
        let x = 0.4 + i as f32 * 0.8;
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.4, 0.1, 0.1),
                radius: 0.0,
            },
            pose(x, 0.1, 0.0), // 바닥 위에 정확히 누움
            1.0,
        ));
        let ap = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.4, 0.0, 0.0)
        };
        w.add_joint(Joint::hinge(
            prev,
            h,
            ap,
            Vec3::new(-0.4, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ));
        links.push(h);
        prev = h;
    }
    for _ in 0..480 {
        w.step(DT);
    }
    for (i, h) in links.iter().enumerate() {
        let b = w.body(*h);
        assert!(b.pose.pos.y.is_finite(), "link {i} 발산");
        assert!(b.linvel.length() < 0.05, "link {i} 미정착: {:?}", b.linvel);
        // 접촉이 실제로 유지되는가 (가라앉지도, 튀어오르지도 않음)
        assert!(
            (b.pose.pos.y - 0.1).abs() < 0.02,
            "link {i} 높이 이탈: {:?}",
            b.pose.pos
        );
    }
    // 조인트가 접촉과 동시에 유지되는가 (혼합 수렴의 핵심 판정)
    for i in 1..20 {
        let a = w.body(links[i - 1]);
        let b = w.body(links[i]);
        let pa = a.pose.pos + a.pose.rot.rotate(Vec3::new(0.4, 0.0, 0.0));
        let pb = b.pose.pos + b.pose.rot.rotate(Vec3::new(-0.4, 0.0, 0.0));
        assert!(
            (pa - pb).length() < 0.02,
            "joint {i} 벌어짐 {}",
            (pa - pb).length()
        );
    }
}

/// 낙하 체인(혼돈계): **무발산과 조인트 무결성만** 판정한다.
/// 12초 내 완전 정착은 물리가 보장하는 성질이 아니다(접힌 체인은 계속 미끄러질 수 있음)
/// — 정착 판정은 위 '바닥에 누운 체인'(결정적 시나리오)이 담당한다.
#[test]
fn hinge_chain_dropped_does_not_diverge() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.05 },
        pose(0.0, 3.0, 0.0),
    ));
    let mut prev = anchor;
    let mut links = Vec::new();
    for i in 0..20 {
        let x = 0.4 + i as f32 * 0.8;
        let h = w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.4, 0.1, 0.1),
                radius: 0.0,
            },
            pose(x, 3.0, 0.0),
            1.0,
        ));
        let ap = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.4, 0.0, 0.0)
        };
        w.add_joint(Joint::hinge(
            prev,
            h,
            ap,
            Vec3::new(-0.4, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ));
        links.push(h);
        prev = h;
    }
    for _ in 0..720 {
        w.step(DT);
    }
    for (i, h) in links.iter().enumerate() {
        let b = w.body(*h);
        assert!(b.pose.pos.y.is_finite(), "link {i} 발산");
        // 에너지 폭주만 금지 (자유낙하 3m의 종단속도 ~7.7m/s를 크게 넘지 않아야 함)
        assert!(
            b.linvel.length() < 5.0,
            "link {i} 에너지 폭주: {:?}",
            b.linvel
        );
        assert!(b.pose.pos.y > -1.0, "link {i} 바닥 관통: {:?}", b.pose.pos);
    }
    for i in 1..20 {
        let a = w.body(links[i - 1]);
        let b = w.body(links[i]);
        let pa = a.pose.pos + a.pose.rot.rotate(Vec3::new(0.4, 0.0, 0.0));
        let pb = b.pose.pos + b.pose.rot.rotate(Vec3::new(-0.4, 0.0, 0.0));
        assert!((pa - pb).length() < 0.05, "joint {i} 벌어짐");
    }
}

/// 조인트 각이 **결정적 atan2**를 쓴다 — 플랫폼 libm에 의존하지 않는다.
/// (값 자체는 std와 2e-6 안에서 같다는 것을 maru-math가 잠그고, 여기서는 그 함수가
/// 실제로 물리 경로에 연결돼 있는지를 본다.)
#[test]
fn hinge_angle_uses_the_deterministic_atan2() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    });
    let anchor = w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.2, 0.2, 0.2),
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
    ));
    let arm = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.1, 0.1),
            radius: 0.0,
        },
        pose(0.7, 0.0, 0.0),
        1.0,
    ));
    w.try_add_joint(Joint::hinge(
        anchor,
        arm,
        Vec3::new(0.2, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ))
    .expect("힌지");
    w.set_angular_velocity(arm, Vec3::new(0.0, 0.0, 1.5))
        .expect("각속도");
    for _ in 0..30 {
        w.step(DT);
    }
    // 각도가 유한하고 회전이 실제로 일어났으면 경로가 살아 있다.
    let q = w.body(arm).pose().rot;
    let ang = 2.0 * maru_math::atan2(q.z.abs(), q.w.abs());
    assert!(ang.is_finite() && ang > 0.01, "회전이 없다: {ang}");
}

/// BLOCKER(ABA): 슬롯이 제거·재사용되면 옛 핸들은 새 바디를 가리켜선 안 된다.
/// 조인트 엔드포인트가 raw 인덱스였을 때 성립하던 경로다.
#[test]
fn slot_reuse_kills_old_handle_and_joint_command() {
    let mut w = World::new(WorldParams::default());
    let old = dyn_box(&mut w, 0.0, 3.0);
    let keep = dyn_box(&mut w, 5.0, 3.0);
    w.remove_body(old).unwrap();
    // 최소 인덱스 재사용 → 같은 슬롯, 다른 세대
    let new = w.enqueue_add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(0.0, 3.0, 0.0),
        1.0,
    ));
    assert_eq!(new.index, old.index, "슬롯 재사용 전제가 깨짐");
    assert_ne!(new.generation, old.generation, "세대가 증가하지 않음");
    // 옛 핸들로 만든 조인트는 **enqueue에서** 거부된다 (계약 통일: 큐에 들어간 커맨드는
    // 항상 유효 → 자기 스냅샷이 자기에게 복원된다)
    assert_eq!(
        w.enqueue(Command::AddJoint(Box::new(Joint::point(
            old,
            keep,
            Vec3::ZERO,
            Vec3::ZERO,
        )))),
        Err(ApiError::StaleHandle),
        "stale 핸들 조인트가 큐에 들어감"
    );
    assert_eq!(w.pending_commands(), 1, "예약 AddBody만 남아야 한다");
    w.flush_commands();
    assert!(w.is_alive(new) && !w.is_alive(old));
}

/// MAJOR: kinematic은 **항상 활성**이다 — 구동 중인 kinematic에 매달린 조인트를
/// 비활성으로 스킵하면 체인이 처방 운동을 따라가지 못한다.
#[test]
fn kinematic_drives_jointed_body() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let k = w.add_body(Body::new_kinematic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 5.0, 0.0),
    ));
    let b = dyn_box(&mut w, 1.5, 5.0);
    w.add_joint(Joint::point(
        k,
        b,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    ));
    w.set_linear_velocity(k, Vec3::new(2.0, 0.0, 0.0)).unwrap();
    for _ in 0..120 {
        w.step(DT);
    }
    let kx = w.body(k).pose.pos.x;
    let bx = w.body(b).pose.pos.x;
    assert!(kx > 3.0, "kinematic이 구동되지 않음 x={kx}");
    assert!(
        (bx - (kx + 1.5)).abs() < 0.5,
        "조인트가 kinematic을 따라가지 못함 k={kx} b={bx}"
    );
}

/// BLOCKER: 힌지 각 임펄스는 별도 경로 — 큰 관성에서 `1/k_ang`이 커지므로
/// 각 bias를 **각속도 상한**으로 묶어야 한다(선형 상한 m/s가 아니라).
#[test]
fn misaligned_hinge_with_large_inertia_stays_in_policy() {
    let mut w = World::new(WorldParams {
        substeps: 64,
        solver: crate::solver::SolverParams {
            contact_hertz: 1.0e3,
            max_bias_speed: maru_collide::MAX_SPEED,
            ..Default::default()
        },
        ..Default::default()
    });
    let k = w.add_body(Body::new_kinematic(
        Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
    ));
    // 관성 상한 근처의 무거운 바디 (역관성 하한 1e-10 부근)
    let heavy = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(100.0, 100.0, 100.0),
            radius: 0.0,
        },
        pose(200.0, 0.0, 0.0),
        0.125, // 부피 8e6 × 0.125 = 1e6 kg (질량 상한)
    ));
    // 90° 어긋난 축으로 연결 — 각 제약이 최대 오차에서 출발한다
    w.add_joint(Joint::hinge(
        k,
        heavy,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-100.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ));
    for i in 0..20 {
        assert_eq!(
            w.try_step(crate::world::MIN_DT),
            Ok(()),
            "합법 월드가 step {i}에서 StateOverflow (각 임펄스 경로)"
        );
    }
    assert!(w.restore(&w.snapshot()).is_ok());
}

// ─────────── 리뷰 rev_f724b2dd 회귀 ───────────

/// BLOCKER: 삽입 검증과 솔버가 **같은 K**를 만들어야 한다.
/// 함수로 뽑으며 회전 항의 부호를 뒤집어, 검증이 솔버와 다른 행렬을 보고 있었다.
/// 이제 솔버가 `linear_k`를 직접 호출하므로 구현이 하나다 — 그 사실을
/// **물리적 결과로** 잠근다: 검증을 통과한 조인트는 실제로 제약을 유지해야 한다.
#[test]
fn validated_joint_actually_holds_in_solver() {
    let mut w = World::new(WorldParams::default());
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.1 },
        pose(0.0, 5.0, 0.0),
    ));
    // 이방성 바디 — 검증과 솔버의 K가 갈리면 여기서 드러난다
    let plate = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(2.0, 0.05, 2.0),
            radius: 0.0,
        },
        pose(1.0, 5.0, 0.0),
        100.0,
    ));
    w.try_add_joint(Joint::point(
        anchor,
        plate,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0),
    ))
    .expect("검증을 통과해야 한다");
    for _ in 0..600 {
        w.step(DT);
    }
    // 조인트가 살아 있으면 앵커에서 대략 1m 반경을 유지한다
    let d = (w.body(plate).pose().pos - Vec3::new(0.0, 5.0, 0.0)).length();
    assert!(
        (0.5..2.0).contains(&d),
        "검증을 통과한 조인트가 실제로는 풀렸다 (거리 {d:.2})"
    );
}

/// BLOCKER: 스냅샷 복원도 라이브와 **같은 K 검사**를 거쳐야 한다.
#[test]
fn restore_applies_the_same_joint_k_check() {
    // 라이브가 거부하는 구성을 만들고, 그 거부가 복원 경로에도 존재하는지 확인한다.
    let mut w = World::new(WorldParams::default());
    let rod = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(1000.0, 0.001, 0.001),
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
        125.0,
    ));
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.1 },
        pose(0.0, 1000.0, 0.0),
    ));
    assert_eq!(
        w.try_add_joint(Joint::point(
            rod,
            anchor,
            Vec3::new(0.0, 1000.0, 0.0),
            Vec3::ZERO
        )),
        Err(ApiError::InvalidValue)
    );
    // 정상 조인트만 있는 월드는 왕복이 성립한다
    w.try_add_joint(Joint::point(
        rod,
        anchor,
        Vec3::new(1000.0, 0.0, 0.0),
        Vec3::ZERO,
    ))
    .unwrap();
    w.step(DT);
    assert!(w.restore(&w.snapshot()).is_ok());
}

/// BLOCKER: 자세 변경이 K를 깨뜨리면 **거부**돼야 한다 — 조용히 풀리는 것보다 낫다.
#[test]
fn set_pose_revalidates_connected_joint_k() {
    let mut w = World::new(WorldParams::default());
    let a = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
        1.0,
    ));
    let b = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(2.0, 0.0, 0.0),
        1.0,
    ));
    w.try_add_joint(Joint::point(a, b, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO))
        .unwrap();
    // 정상 자세 변경은 통과한다
    assert!(w.set_pose(b, pose(2.0, 1.0, 0.0)).is_ok());
    // 거부되더라도 자세는 **바뀌지 않은 채** 남아야 한다 (원자성)
    let before = w.body(b).pose().pos;
    let _ = w.set_pose(b, pose(f32::NAN, 0.0, 0.0));
    assert_eq!(w.body(b).pose().pos, before);
}

/// MAJOR: OOB는 아일랜드 union에도 참여하면 안 된다 — 두 정상 바디가 같은 OOB에
/// 매달리면 비활성 조인트를 통해 한 아일랜드로 묶인다.
#[test]
fn oob_joint_does_not_union_islands() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let edge = maru_collide::MAX_INPUT - 5.0;
    let ghost = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge, 0.0, 0.0),
        1.0,
    ));
    let l = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge - 3.0, 0.0, 0.0),
        1.0,
    ));
    let r = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge - 6.0, 0.0, 0.0),
        1.0,
    ));
    w.try_add_joint(Joint::point(ghost, l, Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO))
        .unwrap();
    w.try_add_joint(Joint::point(
        ghost,
        r,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::ZERO,
    ))
    .unwrap();
    w.set_linear_velocity(ghost, Vec3::new(500.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..10 {
        let _ = w.try_step(crate::world::MAX_DT);
    }
    assert!(w.is_out_of_bounds(ghost));
    for _ in 0..60 {
        let _ = w.try_step(DT);
        assert!(
            w.restore(&w.snapshot()).is_ok(),
            "OOB가 아일랜드에 묶여 복원 불변식을 깼다"
        );
    }
}

/// MAJOR: OOB 전이는 인접 조인트의 누적 임펄스를 **지워야** 한다 —
/// 복귀 후 warm start가 옛 자세의 임펄스를 새 자세에 적용하면 큰 kick이 된다.
#[test]
fn oob_transition_clears_adjacent_joint_impulses() {
    let mut w = World::new(WorldParams::default());
    let edge = maru_collide::MAX_INPUT - 5.0;
    // **둘 다 동적**이어야 함께 경계 밖으로 나간다 — 정적 앵커에 매달리면 조인트가 잡는다.
    let a = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(edge - 1.0, 0.0, 0.0),
        1.0,
    ));
    let b = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(edge, 0.0, 0.0),
        1.0,
    ));
    w.try_add_joint(Joint::point(a, b, Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO))
        .unwrap();
    // 중력·구속으로 누적 임펄스를 쌓는다
    for _ in 0..120 {
        w.step(DT);
    }
    // 함께 경계 밖으로
    for h in [a, b] {
        w.set_linear_velocity(h, Vec3::new(500.0, 0.0, 0.0))
            .unwrap();
    }
    for _ in 0..10 {
        let _ = w.try_step(crate::world::MAX_DT);
    }
    assert!(
        w.is_out_of_bounds(a) && w.is_out_of_bounds(b),
        "시나리오 전제 붕괴: OOB로 나가지 않았다"
    );
    // 호스트가 되돌린다 — 다음 스텝에 stale 임펄스 kick이 없어야 한다
    w.set_pose(a, pose(0.0, 1.0, 0.0)).unwrap();
    w.set_pose(b, pose(0.5, 1.0, 0.0)).unwrap();
    w.step(DT);
    let v = w.body(a).linvel().length().max(w.body(b).linvel().length());
    assert!(v < 5.0, "복귀 직후 stale 임펄스 kick: {v}");
}

/// MAJOR: 일반 `set_pose`(텔레포트)도 인접 조인트의 누적 임펄스를 지워야 한다 —
/// 이전 자세의 월드 임펄스를 새 레버암에 적용하면 인공 kick이 된다.
#[test]
fn teleport_clears_adjacent_joint_impulses() {
    let mut w = World::new(WorldParams::default());
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.2 },
        pose(0.0, 5.0, 0.0),
    ));
    let bob = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(1.0, 5.0, 0.0),
        1.0,
    ));
    w.try_add_joint(Joint::point(
        anchor,
        bob,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0),
    ))
    .unwrap();
    for _ in 0..240 {
        w.step(DT); // 하중으로 임펄스를 쌓는다
    }
    // 반대편으로 텔레포트 — 레버암 방향이 뒤집힌다
    w.set_pose(bob, pose(-1.0, 5.0, 0.0)).unwrap();
    w.step(DT);
    let v = w.body(bob).linvel().length();
    assert!(v < 5.0, "텔레포트 직후 stale 임펄스 kick: {v}");
}

/// BLOCKER: 삽입 시 조건수를 검사해도 **정상 적분 중 자유 회전**으로 K가 특이해질 수
/// 있다(point joint는 회전을 구속하지 않는다). 그때 `inverse`가 0 행렬이면 제약이
/// **조용히 풀린다** — 가장 나쁜 실패다. 솔브 시점 정규화로 항상 풀리게 만든다.
#[test]
fn joint_survives_free_rotation_without_silent_release() {
    let mut w = World::new(WorldParams::default());
    let anchor = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.1 },
        pose(0.0, 6.0, 0.0),
    ));
    // 이방성 판 — 자유 회전으로 조건수가 크게 변한다
    let plate = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(3.0, 0.05, 0.4),
            radius: 0.0,
        },
        pose(1.5, 6.0, 0.0),
        50.0,
    ));
    w.try_add_joint(Joint::point(
        anchor,
        plate,
        Vec3::ZERO,
        Vec3::new(-1.5, 0.0, 0.0),
    ))
    .unwrap();
    // 회전을 크게 넣어 자세를 계속 바꾼다
    w.set_angular_velocity(plate, Vec3::new(0.0, 0.0, 8.0))
        .unwrap();
    for i in 0..1200 {
        w.step(DT);
        let d = (w.body(plate).pose().pos - Vec3::new(0.0, 6.0, 0.0)).length();
        assert!(d < 4.0, "스텝 {i}에서 조인트가 풀렸다 (앵커 거리 {d:.2})");
    }
    assert!(w.restore(&w.snapshot()).is_ok());
}

/// **hinge 한계가 실제로 각을 재는가.** `hinge_angle`은 A·B 양쪽 축의
/// `perp_basis`를 비교하는데, 힌지가 두 축을 정렬시키므로 두 기저가 같아져
/// 각이 항상 0이 된다 → 한계가 절대 발동하지 않는다.
#[test]
fn review_hinge_limit_actually_limits_a_pendulum() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hinge_pendulum(&mut w);
    let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
    w.try_add_joint(
        Joint::try_hinge(
            a,
            b,
            maru_math::Vec3::ZERO,
            maru_math::Vec3::new(-1.0, 0.0, 0.0),
            axis,
            axis,
        )
        .unwrap()
        .with_limit(-0.3, 0.3)
        .unwrap(),
    )
    .unwrap();
    // 초기 각속도를 줘서 한계에 **부딪히게** 만든다 — 앵커가 무게중심에 있으면
    // 중력만으로는 토크가 안 생겨 한계가 시험되지 않는다.
    w.set_angular_velocity(b, maru_math::Vec3::new(0.0, 0.0, 6.0))
        .unwrap();
    let mut worst: f32 = 0.0;
    for _ in 0..300 {
        w.try_step(1.0 / 60.0).unwrap();
        let q = w.body(b).pose().rot;
        let ang = 2.0 * q.z.abs().atan2(q.w.abs());
        worst = worst.max(ang);
    }
    assert!(
        worst < 0.45,
        "한계 ±0.3 rad인데 최대 {worst} rad까지 돌았다 (한계가 발동하지 않음)"
    );
}

/// **Fixed의 각 bias 부호.** err = 2·vec(qb·(qa·rel)⁻¹)는 d(err)/dt = ωb − ωa로
/// 자라는데 솔버는 cdot = (ωa − ωb)를 쓴다 — Baumgarte 항의 부호가 뒤집혀
/// 오차를 **되돌리는 대신 키운다**.
#[test]
fn review_fixed_joint_corrects_angular_error() {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let a = w.add_body(Body::new_static(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
    ));
    // B의 초기 자세를 목표(항등)에서 0.1 rad 어긋나게 둔다.
    let b = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Box {
            half: maru_math::Vec3::new(0.3, 0.3, 0.3),
            radius: 0.0,
        },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(1.0, 10.0, 0.0),
            rot: maru_math::Quat::from_axis_angle(maru_math::Vec3::new(1.0, 0.0, 0.0), 0.1),
        },
        1.0,
    ));
    w.try_add_joint(
        Joint::try_fixed(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Vec3::ZERO,
            maru_math::Quat::IDENTITY,
        )
        .unwrap(),
    )
    .unwrap();
    let err0 = 2.0 * w.body(b).pose().rot.x.abs();
    for _ in 0..60 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let err1 = 2.0 * w.body(b).pose().rot.x.abs();
    assert!(
        err1 < err0,
        "고정 조인트가 각 오차를 줄여야 한다: {err0} → {err1}"
    );
}

/// **부등식 게이트 × warm start.** 구간 안으로 들어온 서브스텝에서 누적 임펄스를
/// 0으로 되돌리지만, 같은 서브스텝의 warm start가 이미 옛 임펄스를 속도에 실었다 —
/// 되돌리는 델타(0 − old)를 적용하지 않아 그 운동량이 그대로 남는다.
#[test]
#[allow(clippy::field_reassign_with_default)]
fn review_distance_joint_inactive_leaves_velocity_untouched() {
    use crate::solver::{Softness, SolverState};
    let mut st = SolverState::default();
    // 2바디: 0=static, 1=dynamic. 거리 1.0(구간 [0,2] 안).
    st.lin = vec![maru_math::Vec3::ZERO; 2];
    st.ang = vec![maru_math::Vec3::ZERO; 2];
    st.pos = vec![maru_math::Vec3::ZERO, maru_math::Vec3::new(1.0, 0.0, 0.0)];
    st.rot = vec![maru_math::Quat::IDENTITY; 2];
    st.inv_m = vec![0.0, 1.0];
    st.inv_i = vec![
        maru_math::Mat3::from_diagonal(maru_math::Vec3::ZERO),
        maru_math::Mat3::from_diagonal(maru_math::Vec3::new(1.0, 1.0, 1.0)),
    ];
    st.dynamic = vec![false, true];
    let ha = BodyHandle {
        index: 0,
        generation: 1,
    };
    let hb = BodyHandle {
        index: 1,
        generation: 1,
    };
    let mut j = Joint::try_distance(
        ha,
        hb,
        maru_math::Vec3::ZERO,
        maru_math::Vec3::ZERO,
        0.0,
        2.0,
    )
    .unwrap();
    // 직전 서브스텝에 구간 밖이어서 쌓인 누적 임펄스 (로프가 당기고 있었다).
    j.linear_impulse = maru_math::Vec3::new(-3.0, 0.0, 0.0);
    let soft = Softness::new(60.0, 2.0, 1.0 / 240.0);
    crate::joint::warm_start(&mut st, &j);
    crate::joint::solve(&mut st, &mut j, 1.0 / 240.0, Some((&soft, 4.0, 30.0)));
    assert!(
        st.lin[1].length() < 1e-4,
        "구간 안이면 조인트가 속도를 건드리면 안 된다: {:?}",
        st.lin[1]
    );
}

/// **모터 상한은 스텝당인가 서브스텝당인가.** `max_impulse`가 서브스텝 누적에
/// 걸리고 warm start가 매 서브스텝 전량을 다시 실으므로, 실제 전달 임펄스는
/// substeps에 비례한다 — 같은 월드에서 substeps만 바꾸면 모터 힘이 달라진다.
#[test]
fn review_motor_max_impulse_is_independent_of_substeps() {
    fn spin(substeps: u32) -> f32 {
        let mut w = World::new(WorldParams {
            gravity: maru_math::Vec3::ZERO,
            substeps,
            ..WorldParams::default()
        });
        let a = w.add_body(Body::new_static(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        ));
        let b = w.add_body(Body::new_dynamic(
            maru_collide::Shape::Box {
                half: maru_math::Vec3::new(0.3, 0.3, 0.3),
                radius: 0.0,
            },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(1.0, 10.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        ));
        let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
        w.try_add_joint(
            Joint::try_hinge(
                a,
                b,
                maru_math::Vec3::new(1.0, 0.0, 0.0),
                maru_math::Vec3::ZERO,
                axis,
                axis,
            )
            .unwrap()
            // 목표를 크게 잡아 모터를 항상 포화시킨다 → 전달량 = 상한 × (횟수)
            .with_motor(50.0, 0.2)
            .unwrap(),
        )
        .unwrap();
        w.try_step(1.0 / 60.0).unwrap();
        w.body(b).angvel().z
    }
    let s1 = spin(1);
    let s8 = spin(8);
    assert!(
        (s1 - s8).abs() < 0.1 * s1.abs().max(1e-6),
        "substeps만 바꿨는데 모터 전달 임펄스가 달라진다: {s1} vs {s8}"
    );
}

/// **기존 조인트 테스트 2건은 각 제약을 자극하지 않는다**(`hung_pair`가 앵커를
/// B의 질량중심에 두어 중력 토크가 0이다). 같은 배치를 **Point 조인트**로 바꿔도
/// `fixed_joint_holds_relative_orientation` · `hinge_limit_stops_rotation`의
/// 단언이 그대로 통과한다 — 즉 두 테스트는 각 제약을 검증하지 않는다.
#[test]
fn review_existing_joint_tests_pass_without_any_angular_constraint() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    // Fixed도 Hinge도 아닌 **순수 point** — 각 제약이 전혀 없다.
    w.try_add_joint(
        Joint::try_point(
            a,
            b,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            maru_math::Vec3::ZERO,
        )
        .unwrap(),
    )
    .unwrap();
    for _ in 0..600 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let q = w.body(b).pose().rot;
    // fixed_joint_holds_relative_orientation 의 단언
    assert!(q.w.abs() > 0.995, "{q:?}");
    let p = w.body(b).pose().pos;
    assert!(
        (p - maru_math::Vec3::new(1.0, 10.0, 0.0)).length() < 0.1,
        "{p:?}"
    );
    // hinge_limit_stops_rotation 의 단언
    let ang = 2.0 * q.z.abs().atan2(q.w.abs());
    assert!(ang < 0.6, "{ang}");
}

/// preflight(`serialized_size`)가 새 조인트 종류까지 writer와 맞는가.
/// 기존 `snapshot_preflight_size_matches...` 테스트는 `Point`만 담고 있어
/// Hinge(limit·motor)·Fixed·Distance는 preflight 커버리지가 없었다.
#[test]
fn review_preflight_size_covers_new_joint_kinds() {
    let mut w = World::new(WorldParams::default());
    let (a, b) = hung_pair(&mut w, maru_math::Vec3::new(1.0, 0.0, 0.0));
    let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
    let anchor_a = maru_math::Vec3::new(1.0, 0.0, 0.0);
    w.try_add_joint(
        Joint::try_fixed(
            a,
            b,
            anchor_a,
            maru_math::Vec3::ZERO,
            maru_math::Quat::IDENTITY,
        )
        .unwrap(),
    )
    .unwrap();
    w.try_add_joint(
        Joint::try_distance(a, b, maru_math::Vec3::ZERO, maru_math::Vec3::ZERO, 0.5, 2.0).unwrap(),
    )
    .unwrap();
    w.try_add_joint(
        Joint::try_hinge(a, b, anchor_a, maru_math::Vec3::ZERO, axis, axis)
            .unwrap()
            .with_limit(-1.0, 1.0)
            .unwrap()
            .with_motor(2.0, 50.0)
            .unwrap(),
    )
    .unwrap();
    // limit/motor 없는 hinge도 (Option 길이 고정 확인)
    w.try_add_joint(Joint::try_hinge(a, b, anchor_a, maru_math::Vec3::ZERO, axis, axis).unwrap())
        .unwrap();
    let bytes = w.snapshot();
    assert_eq!(
        crate::snapshot::serialized_size(&w),
        bytes.len(),
        "preflight 크기가 writer와 어긋남 — 한쪽만 고친 것이다"
    );
}

/// **`hinge_angle`은 축 둘레 각을 재지 못한다** (화이트박스).
///
/// `hinge_angle`은 A·B **월드 축**의 `perp_basis`를 비교하는데, `perp_basis`는
/// 입력 축만의 순함수다. 힌지는 두 축을 같게 만들므로 두 기저가 같아지고
/// 각은 항상 0이 된다 — B가 축 둘레로 90° 돌아 있어도 한계가 발동하지 않는다.
/// 기준 벡터는 각 바디의 **로컬** 벡터를 월드로 돌린 것이어야 한다.
#[test]
#[allow(clippy::field_reassign_with_default)]
fn review_hinge_limit_fires_when_b_is_rotated_about_the_axis() {
    use crate::solver::{Softness, SolverState};
    let axis = maru_math::Vec3::new(0.0, 0.0, 1.0);
    let mut st = SolverState::default();
    st.lin = vec![maru_math::Vec3::ZERO; 2];
    st.ang = vec![maru_math::Vec3::ZERO; 2];
    st.pos = vec![maru_math::Vec3::ZERO; 2];
    // B는 힌지 축(z) 둘레로 90° 돌아 있다 — 한계 ±0.1 rad를 크게 벗어난다.
    st.rot = vec![
        maru_math::Quat::IDENTITY,
        maru_math::Quat::from_axis_angle(axis, core::f32::consts::FRAC_PI_2),
    ];
    st.inv_m = vec![0.0, 1.0];
    st.inv_i = vec![
        maru_math::Mat3::from_diagonal(maru_math::Vec3::ZERO),
        maru_math::Mat3::from_diagonal(maru_math::Vec3::new(1.0, 1.0, 1.0)),
    ];
    st.dynamic = vec![false, true];
    let ha = BodyHandle {
        index: 0,
        generation: 1,
    };
    let hb = BodyHandle {
        index: 1,
        generation: 1,
    };
    let mut j = Joint::try_hinge(
        ha,
        hb,
        maru_math::Vec3::ZERO,
        maru_math::Vec3::ZERO,
        axis,
        axis,
    )
    .unwrap()
    .with_limit(-0.1, 0.1)
    .unwrap();
    let soft = Softness::new(60.0, 2.0, 1.0 / 240.0);
    crate::joint::solve(&mut st, &mut j, 1.0 / 240.0, Some((&soft, 4.0, 30.0)));
    assert!(
        j.limit_impulse != 0.0,
        "축 둘레 90°는 한계 ±0.1을 벗어났는데 한계 임펄스가 0이다 (각이 측정되지 않음)"
    );
}
