//! solver 관련 테스트.

use super::*;

/// 붕괴 판정. **상대 기준**(스택 높이의 1%)을 쓴다 — 절대 임계는 단수가 바뀌면 의미가
/// 달라진다(톨러런스에서 배운 것과 같은 함정). soft 접촉의 하중 비례 침하는 정상 거동이고,
/// 판정해야 하는 것은 '형상을 잃었는가'다. 100단 붕괴(4.9% 낙하)는 이 기준에서도 실패한다.
fn assert_pyramid_stable(w: &World, hs: &[BodyHandle], tiers: usize) {
    for (idx, h) in hs.iter().enumerate() {
        let b = w.body(*h);
        assert!(b.pose.pos.y.is_finite(), "box {idx} 발산");
        assert!(b.linvel.length() < 0.2, "box {idx} 미정착: {:?}", b.linvel);
    }
    let top = w.body(*hs.last().unwrap());
    let expect_y = 0.5 + (tiers - 1) as f32;
    let sag = expect_y - top.pose.pos.y;
    let limit = 0.01 * tiers as f32; // 스택 높이의 1%
    assert!(
        sag.abs() < limit,
        "정상 침하 {sag:.3}m > 허용 {limit:.3}m (높이 {expect_y}) — 붕괴"
    );
}

/// 질량중심에서 벗어난 점을 때리면 **회전도** 생긴다.
#[test]
fn off_center_impulse_creates_spin() {
    let (mut w, h) = unit_sphere_world();
    let com = w.body(h).pose().pos;
    // 중심에서 +y로 0.5 떨어진 점을 +x로 때린다 → z축 각속도가 생겨야 한다
    w.apply_impulse_at_point(
        h,
        maru_math::Vec3::new(4.0, 0.0, 0.0),
        com + maru_math::Vec3::new(0.0, 0.5, 0.0),
    )
    .unwrap();
    let m = 1.0 / w.body(h).inv_mass();
    let b = w.body(h);
    assert!(
        (b.linvel().x - 4.0 / m).abs() < 1e-5,
        "선속도는 중심 타격과 같다"
    );
    assert!(
        b.angvel().z < -1e-3,
        "r×J 의 z가 음수여야 한다: {:?}",
        b.angvel()
    );
    assert!(b.angvel().x.abs() < 1e-6 && b.angvel().y.abs() < 1e-6);
}

/// 정착: 박스가 바닥에 얹혀 침투·점프 없이 정지 상태 유지
#[test]
fn box_rests_on_ground_without_sinking() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let bx = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 0.5, 0.0), // 정확히 얹힘
        1.0,
    ));
    for _ in 0..300 {
        w.step(DT);
    }
    let b = w.body(bx);
    assert!((b.pose.pos.y - 0.5).abs() < 0.01, "y={}", b.pose.pos.y);
    assert!(b.linvel.length() < 0.05, "v={:?}", b.linvel);
    assert!(b.angvel.length() < 0.05, "w={:?}", b.angvel);
}

/// 스택 5단 안정 (M2 피라미드 100단의 축소 선행 게이트)
#[test]
fn stack_of_5_boxes_stays_up() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let mut handles = Vec::new();
    for i in 0..5 {
        handles.push(w.add_body(Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 0.5 + i as f32 * 1.0, 0.0),
            1.0,
        )));
    }
    for _ in 0..600 {
        w.step(DT);
    }
    for (i, h) in handles.iter().enumerate() {
        let b = w.body(*h);
        let expect_y = 0.5 + i as f32 * 1.0;
        assert!(
            (b.pose.pos.y - expect_y).abs() < 0.05,
            "box {i}: y={} expect {expect_y}",
            b.pose.pos.y
        );
        assert!(
            b.pose.pos.x.abs() < 0.05 && b.pose.pos.z.abs() < 0.05,
            "box {i} drifted: {:?}",
            b.pose.pos
        );
    }
}

/// restitution: e=0.8 공 낙하 → 반발 apex ≈ e²·h₀ (임팩트 게이트 경유)
#[test]
fn restitution_bounce_apex() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let mut ball = Body::new_dynamic(Shape::Sphere { radius: 0.5 }, pose(0.0, 2.5, 0.0), 1.0);
    ball.restitution = 0.8;
    let h0 = 2.0; // 낙하 높이 (표면 기준)
    let ball = w.add_body(ball);
    let mut apex: f32 = 0.0;
    let mut bounced = false;
    for _ in 0..240 {
        w.step(DT);
        let b = w.body(ball);
        if b.linvel.y > 0.01 {
            bounced = true;
        }
        if bounced {
            apex = apex.max(b.pose.pos.y - 0.5);
        }
    }
    assert!(bounced, "반발 없음");
    let expect = 0.8 * 0.8 * h0;
    assert!(
        (apex - expect).abs() < 0.25 * expect,
        "apex={apex} expect≈{expect}"
    );
}

/// 피라미드 10단 (55박스) — CI 상시 게이트 (100단의 축소판)
#[test]
fn pyramid_10_tiers_stable() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let hs = pyramid(&mut w, 10);
    for _ in 0..600 {
        w.step(DT);
    }
    assert_pyramid_stable(&w, &hs, 10);
}

/// release 전용 — debug는 수 분 걸린다. `cargo test --release -- --ignored`
#[test]
#[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
fn pyramid_40_tiers_long_run_acceptance() {
    const TIERS: usize = 40;
    let mut w = World::new(WorldParams::default());
    // 지면은 `maru-bench --bin pyramid100`과 **동일**해야 한다 — 같은 수용 기준이 서로 다른
    // 씬을 재면 README 봉투 표와 CI 게이트가 어긋난다.
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(120.0, 1.0, 20.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let hs = pyramid(&mut w, TIERS);
    for _ in 0..3000 {
        w.step(DT);
    }
    assert_pyramid_stable(&w, &hs, TIERS);
    let asleep = hs.iter().filter(|h| w.is_asleep(**h)).count();
    assert_eq!(
        asleep,
        hs.len(),
        "{asleep}/{} 만 잠듦 — 잠들지 않은 스택은 장기적으로 발산한다",
        hs.len()
    );
}

/// 봉투 **밖**(65단)은 실제로 붕괴한다 — 기준이 판별력을 갖는지 잠근다.
/// 이 테스트가 실패하면 봉투가 넓어진 것이므로 위 기준을 다시 재야 한다.
#[test]
#[ignore = "release 전용: cargo test --release -p maru-dynamics -- --ignored"]
fn pyramid_65_tiers_is_outside_the_envelope() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(120.0, 1.0, 20.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let hs = pyramid(&mut w, 65);
    for _ in 0..3000 {
        w.step(DT);
    }
    let top = w.body(*hs.last().unwrap()).pose.pos.y;
    assert!(
        top < 0.99 * (0.5 + 64.0),
        "65단이 버텼다 — 봉투가 넓어졌으니 수용 기준을 재측정할 것 (top_y={top:.2})"
    );
}

/// PARTIAL MAJOR: 정상 슬라이딩에서 warm start가 매 프레임 끊기면 안 된다.
/// 허용치가 기하 척도만 쓰면 60Hz·1m/s(프레임당 16.7mm)가 반두께 허용치를 넘는다.
#[test]
fn sliding_contact_keeps_warm_start() {
    let mut w = World::new(WorldParams::default());
    // 얇은 판 위를 구가 미끄러진다 — 구·판 접점은 generic witness(위치 매칭 대상)
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(20.0, 0.01, 20.0),
            radius: 0.0,
        },
        pose(0.0, -0.01, 0.0),
    ));
    let s = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(-5.0, 0.3, 0.0),
        1.0,
    ));
    for _ in 0..60 {
        w.step(DT); // 안착
    }
    w.set_linear_velocity(s, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    // 진단은 **스텝별로 리셋**되므로 매 스텝 누적한다
    let (mut hits, mut total) = (0usize, 0usize);
    for _ in 0..180 {
        w.step(DT);
        let d = w.diagnostics();
        hits += d.warm_hits;
        total += d.warm_total;
    }
    assert!(total > 100, "접촉 표본이 부족: {total}");
    let ratio = hits as f32 / total as f32;
    assert!(
        ratio > 0.8,
        "슬라이딩 중 warm start 계승률 {ratio:.2} — 매 프레임 끊기고 있다"
    );
}

/// 회귀(확인 검수 rev_8672a3eb MAJOR): substeps=64에서 서브스텝당 팁 이동
/// (30/3840×1.5 ≈ 1.17cm)이 skin/4(1.25cm) 아래로 숨는다 — 게이트가 서브스텝
/// 단위면 캐스트가 아예 꺼져 스텝 전체 0.75m 회전 스윕이 무검사가 된다.
/// 게이트는 스텝 전체 봉투여야 한다.
#[test]
fn per_substep_gate_cannot_hide_a_full_step_rotation() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        substeps: 64,
        ..WorldParams::default()
    });
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.05, 3.0, 3.0),
            radius: 0.0,
        },
        pose(1.2, 0.0, 0.0),
    ));
    let b = Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(1.5, 0.05, 0.05),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::ZERO,
            rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2),
        },
        1.0,
    )
    .with_ccd();
    let h = w.add_body(b);
    w.set_angular_velocity(h, Vec3::new(0.0, 30.0, 0.0))
        .unwrap();
    let skin = w.params().speculative_margin;
    for step in 0..120 {
        w.step(DT);
        let p = w.body(h).pose();
        let tip = (p.pos + p.rot.rotate(Vec3::new(1.5, 0.0, 0.0)))
            .x
            .max((p.pos + p.rot.rotate(Vec3::new(-1.5, 0.0, 0.0))).x);
        assert!(
            tip <= 1.15 + skin + 0.02,
            "step {step}: 서브스텝 게이트 아래로 숨은 회전이 벽을 쓸었다 (tip_x={tip})"
        );
    }
}
