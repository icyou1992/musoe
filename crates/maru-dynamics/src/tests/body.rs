//! body 관련 테스트.

use super::*;

/// 힘은 **가속도**로 들어간다: F=ma 이므로 dt 동안 Δv = F·dt/m 이어야 한다.
/// (`new_dynamic`의 셋째 인자는 질량이 아니라 **밀도**다 — 질량은 역질량에서 되찾는다.)
#[test]
fn applied_force_integrates_as_acceleration() {
    let (mut w, h) = unit_sphere_world();
    let m = 1.0 / w.body(h).inv_mass();
    w.apply_force(h, maru_math::Vec3::new(10.0, 0.0, 0.0))
        .unwrap();
    let dt = 1.0 / 60.0;
    w.try_step(dt).unwrap();
    let v = w.body(h).linvel();
    let want = 10.0 / m * dt;
    assert!((v.x - want).abs() < 1e-4, "Δv={} 인데 F·dt/m = {want}", v.x);
    // **한 스텝이 소비하고 지운다** — 다음 스텝에 다시 가속되면 안 된다.
    assert_eq!(w.body(h).force(), maru_math::Vec3::ZERO);
    w.try_step(dt).unwrap();
    assert!(
        (w.body(h).linvel().x - want).abs() < 1e-4,
        "힘이 두 번 먹혔다"
    );
}

/// 여러 번 부르면 더해진다.
#[test]
fn forces_accumulate_until_the_step() {
    let (mut w, h) = unit_sphere_world();
    for _ in 0..4 {
        w.apply_force(h, maru_math::Vec3::new(1.0, 0.0, 0.0))
            .unwrap();
    }
    assert!((w.body(h).force().x - 4.0).abs() < 1e-6);
    let m = 1.0 / w.body(h).inv_mass();
    w.try_step(1.0 / 60.0).unwrap();
    let want = 4.0 / m * (1.0 / 60.0);
    assert!((w.body(h).linvel().x - want).abs() < 1e-4);
}

/// 충격은 즉시 속도를 바꾼다 (스텝을 기다리지 않는다).
#[test]
fn impulse_changes_velocity_immediately() {
    let (mut w, h) = unit_sphere_world();
    let m = 1.0 / w.body(h).inv_mass();
    w.apply_impulse(h, maru_math::Vec3::new(6.0, 0.0, 0.0))
        .unwrap();
    assert!((w.body(h).linvel().x - 6.0 / m).abs() < 1e-5, "Δv = J/m");
}

/// 감쇠는 속도를 지수적으로 줄인다 — `v(t) = v0 / (1 + t·d)` 근사.
#[test]
fn damping_slows_a_body_without_gravity() {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let free = w.add_body(Body::new_dynamic(
        maru_collide::Shape::Sphere { radius: 0.5 },
        maru_collide::Pose {
            pos: maru_math::Vec3::new(0.0, 10.0, 0.0),
            rot: maru_math::Quat::IDENTITY,
        },
        1.0,
    ));
    let damped = w.add_body(
        Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 30.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        )
        .with_damping(2.0, 0.0)
        .unwrap(),
    );
    for h in [free, damped] {
        w.set_linear_velocity(h, maru_math::Vec3::new(10.0, 0.0, 0.0))
            .unwrap();
    }
    for _ in 0..60 {
        w.try_step(1.0 / 60.0).unwrap();
    }
    let vf = w.body(free).linvel().x;
    let vd = w.body(damped).linvel().x;
    assert!(
        (vf - 10.0).abs() < 1e-3,
        "감쇠 없으면 그대로여야 한다: {vf}"
    );
    assert!(vd < 4.0 && vd > 0.0, "감쇠는 줄이되 뒤집지 않는다: {vd}");
}

/// **큰 감쇠에서 부호가 뒤집히면 안 된다** — 암시적 형태를 쓰는 이유.
#[test]
fn huge_damping_does_not_reverse_velocity() {
    let mut w = World::new(WorldParams {
        gravity: maru_math::Vec3::ZERO,
        ..WorldParams::default()
    });
    let h = w.add_body(
        Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 0.5 },
            maru_collide::Pose {
                pos: maru_math::Vec3::ZERO,
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        )
        // h = dt/substeps = 1/240 이므로 h·d = 4 ≫ 1 — 명시적이면 부호가 뒤집힌다
        .with_damping(1000.0, 0.0)
        .unwrap(),
    );
    w.set_linear_velocity(h, maru_math::Vec3::new(10.0, 0.0, 0.0))
        .unwrap();
    w.try_step(1.0 / 60.0).unwrap();
    let v = w.body(h).linvel().x;
    assert!((0.0..1.0).contains(&v), "줄되 뒤집히지 않아야 한다: {v}");
}

/// 해석 검증: 자유낙하 v=gt, y=y0−gt²/2 (Symplectic Euler 오차 허용)
#[test]
fn free_fall_analytic() {
    let mut w = World::new(WorldParams::default());
    let ball = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(0.0, 100.0, 0.0),
        1.0,
    ));
    for _ in 0..60 {
        w.step(DT);
    }
    let b = w.body(ball);
    let t = 1.0;
    assert!((b.linvel.y + 9.81 * t).abs() < 0.01, "v={}", b.linvel.y);
    // Symplectic Euler는 반스텝 오차: |y − 해석| < g·dt·t
    let analytic = 100.0 - 0.5 * 9.81 * t * t;
    assert!(
        (b.pose.pos.y - analytic).abs() < 9.81 * DT * t + 0.01,
        "y={}",
        b.pose.pos.y
    );
}

/// 비표준 환경에서 만든 **Body 자체**가 거부된다 (BLOCKER: 우회 경로).
#[test]
fn body_construction_is_guarded_too() {
    if !maru_math::force_flush_to_zero_for_test() {
        return;
    }
    let r = Body::try_new_dynamic(Shape::Sphere { radius: 0.3 }, pose(0.0, 0.0, 0.0), 1.0);
    let r2 = Body::try_new_static(Shape::Sphere { radius: 0.3 }, pose(0.0, 0.0, 0.0));
    assert!(World::canonicalize_fp_environment());
    assert_eq!(r.err(), Some(ApiError::FpEnvironment));
    assert_eq!(r2.err(), Some(ApiError::FpEnvironment));
    assert!(Body::try_new_dynamic(Shape::Sphere { radius: 0.3 }, pose(0.0, 0.0, 0.0), 1.0).is_ok());
}

/// **멀리 있는 봉우리 하나가 정상 스텝을 비싸게 만들지 않는다** (MAJOR: raw 셀 수).
///
/// 지형은 바디 하나라 전역 AABB가 봉우리까지 포함한다 — 그래서 브로드페이즈는 공중의
/// 물체도 후보로 넣는다. 예전에는 그 후보의 **raw 셀 수**를 세서 후보가 실제로 0개인
/// 질의까지 거절했고, 접촉 순회는 셀 전부를 돌았다. 타일 Y 밴드로 걸러야 한다.
///
/// (셀 **예산** 초과는 이제 자연스러운 장면에서 재현하기 어렵다: 셀을 많이 보려면
/// 바디가 커야 하고, 큰 바디는 접촉을 먼저 64개 채워 포화로 끝난다. 예산은 남겨 둔
/// 최후 방어선이고, 그 신호가 포화와 **구분되는지**는 `ContactWalk` 타입이 잠근다.)
#[test]
fn a_distant_peak_does_not_make_a_hovering_body_expensive() {
    let (nx, nz) = (600u32, 600u32);
    let mut heights = vec![0.0f32; (nx * nz) as usize];
    heights[0] = 3.0; // 전역 AABB만 높이는 봉우리 하나
    let mut w = World::new(WorldParams::default());
    w.try_add_heightfield(&heights, nx, nz, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형");
    // 평지 위 2.5m 공중의 큰 박스 — 봉우리와 멀고 실제 접촉은 없다.
    let h = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(25.0, 1.0, 25.0),
            radius: 0.0,
        },
        pose(30.0, 2.5, 30.0),
        1.0,
    ));
    assert!(
        w.try_step(DT).is_ok(),
        "후보가 0개인 스텝이 자원 오류로 거절됐다 — raw 셀 수를 세고 있다"
    );
    // 질의도 같은 규칙이어야 한다.
    assert!(w
        .sweep_shape(
            &Shape::Sphere { radius: 0.2 },
            pose(30.0, 2.5, 30.0),
            Vec3::new(0.0, 0.2, 0.0),
            QueryFilter::ALL
        )
        .is_ok());
    assert!(w.body(h).pose().pos.y.is_finite());
}

/// **비단위 자세는 찌그러진 형상이 아니라 정규화된 형상으로 판정된다** (BLOCKER: 조용한 왜곡).
///
/// `Quat::rotate`는 비단위 쿼터니언에 회전+축소를 적용한다. 유한성만 검사하면 0.5배한
/// 정상 쿼터니언이 통과해 캐스트가 절반 크기 형상으로 답을 낸다(확인검수 rev_2e93ebf2).
/// 부동소수 드리프트로 정확한 단위는 드물기 때문에 **거절이 아니라 정규화**가 맞고,
/// 정규화 불가(0·NaN)만 거절한다.
#[test]
fn non_unit_pose_rotation_is_normalized_not_trusted() {
    let long = Shape::Box {
        half: Vec3::new(1.0, 0.2, 0.2),
        radius: 0.0,
    };
    let wall = Shape::Box {
        half: Vec3::new(0.2, 1.0, 1.0),
        radius: 0.0,
    };
    let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2);
    let unit = maru_collide::Pose {
        pos: Vec3::ZERO,
        rot: q,
    };
    let scaled = maru_collide::Pose {
        pos: Vec3::ZERO,
        rot: Quat {
            x: q.x * 0.5,
            y: q.y * 0.5,
            z: q.z * 0.5,
            w: q.w * 0.5,
        },
    };
    // 90° 돌린 긴 박스는 z∈[-1,1]을 차지한다 — 벽은 그 **밖에** 둬야 한다(겹쳐서
    // 시작하면 캐스트는 기존 관통이라 `None`이 맞고, 그러면 대조가 성립하지 않는다).
    let pb = pose(0.0, 0.0, 2.4);
    let m = Vec3::new(0.0, 0.0, 1.0);
    let a = maru_collide::try_shape_cast_dispatch(&long, &unit, m, &wall, &pb, 0.02)
        .expect("정상 자세");
    let b = maru_collide::try_shape_cast_dispatch(&long, &scaled, m, &wall, &pb, 0.02)
        .expect("정규화 가능한 자세");
    // **음성 대조** — 이 배치에서 정상 자세는 실제로 맞춘다(둘 다 None이면 무의미하다).
    assert!(
        a.is_some(),
        "정상 자세가 벽을 못 맞췄다 — 배치가 무의미하다"
    );
    assert_eq!(a, b, "비단위 자세가 다른 형상으로 판정됐다");
    // 정규화 불가한 자세는 거절.
    let zero = maru_collide::Pose {
        pos: Vec3::ZERO,
        rot: Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
    };
    assert_eq!(
        maru_collide::try_shape_cast_dispatch(&long, &zero, m, &wall, &pb, 0.02),
        Err(maru_collide::CastReject::InvalidInput)
    );
    // 월드 질의도 같은 규약 — 여기서 정규화하지 않으면 AABB 선판정이 먼저 왜곡된다.
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(wall.clone(), pb));
    let s1 = w
        .sweep_shape(&long, unit, m, QueryFilter::ALL)
        .expect("질의");
    let s2 = w
        .sweep_shape(&long, scaled, m, QueryFilter::ALL)
        .expect("질의");
    assert!(s1.is_some(), "월드 질의에서 정상 자세가 벽을 못 맞췄다");
    assert_eq!(
        s1.map(|h| h.toi),
        s2.map(|h| h.toi),
        "월드 질의가 비단위 자세를 그대로 썼다"
    );
    assert_eq!(
        w.sweep_shape(&long, zero, m, QueryFilter::ALL).err(),
        Some(ApiError::InvalidValue)
    );
}

/// **경계를 넘는 서브스텝도 회전을 빼먹지 않는다** (BLOCKER 지적의 실제 하위 결함).
///
/// 내 첫 OOB 가드는 감지 서브스텝에서 `continue`로 회전 적분까지 건너뛰었다. 그러면 그
/// 서브스텝만 회전이 빠지고 이후 서브스텝(`ccd_frozen && !rot_sweep`)은 전체 회전을
/// 적분하는 **비대칭**이 생긴다 — `ccd_frozen`의 규약은 "이동만 얼린다"이므로 회전은
/// 일관되게 돌아야 한다(확인검수 rev_59dcb940).
#[test]
fn boundary_crossing_substep_still_integrates_rotation() {
    let spin = Vec3::new(0.0, 3.0, 0.0);
    let mk = |x: f32| {
        let mut w = World::new(WorldParams::default());
        let h = w.add_body(
            Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.4, 0.1, 0.1),
                    radius: 0.0,
                },
                pose(x, 0.0, 0.0),
                1.0,
            )
            .with_ccd(),
        );
        w.set_angular_velocity(h, spin).expect("각속도");
        w.set_linear_velocity(h, Vec3::new(w.params().max_linear_velocity, 0.0, 0.0))
            .expect("속도");
        (w, h)
    };
    // ① 경계를 넘는 바디 ② 월드 한복판의 같은 바디 — 회전량이 같아야 한다.
    let (mut w_edge, h_edge) = mk(maru_collide::MAX_INPUT - 4.0);
    let (mut w_mid, h_mid) = mk(0.0);
    assert_eq!(w_edge.try_step(1.0 / 60.0), Ok(()));
    assert_eq!(w_mid.try_step(1.0 / 60.0), Ok(()));
    assert!(
        w_edge.diagnostics().out_of_bounds > 0,
        "테스트 전제(경계 이탈)가 성립하지 않았다"
    );
    let ang = |q: maru_math::Quat| 2.0 * q.y.atan2(q.w);
    let (a_edge, a_mid) = (
        ang(w_edge.body(h_edge).pose().rot),
        ang(w_mid.body(h_mid).pose().rot),
    );
    assert!(
        (a_edge - a_mid).abs() < 1.0e-5,
        "경계를 넘은 바디의 회전이 {a_edge}, 안쪽 바디는 {a_mid} — 서브스텝 하나의 회전이 빠졌다"
    );
    // **음성 대조** — 회전이 실제로 일어났어야 비교가 의미 있다(둘 다 0이면 무의미).
    assert!(
        a_mid.abs() > 1.0e-3,
        "회전이 아예 없었다 — 테스트 배치가 무의미하다"
    );
}

/// 리뷰 MAJOR 회귀: 공개 변이는 불변식을 강제한다 (NaN·범위 밖·static 거부).
#[test]
fn mutation_api_enforces_invariants() {
    let mut w = World::new(WorldParams::default());
    let g = ground(&mut w);
    let b = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 2.0, 0.0),
        1.0,
    ));
    assert_eq!(
        w.set_linear_velocity(b, Vec3::new(f32::NAN, 0.0, 0.0))
            .err(),
        Some(ApiError::InvalidValue)
    );
    assert_eq!(
        w.set_material(b, -1.0, 0.5).err(),
        Some(ApiError::InvalidValue)
    );
    assert_eq!(
        w.set_material(b, 0.5, 2.0).err(),
        Some(ApiError::InvalidValue)
    );
    // static은 불변 (설계 §9)
    assert_eq!(
        w.set_linear_velocity(g, Vec3::ZERO).err(),
        Some(ApiError::ImmutableStatic)
    );
    // 정상 값은 통과하고 상태가 유지된다
    assert!(w.set_linear_velocity(b, Vec3::new(1.0, 0.0, 0.0)).is_ok());
    assert_eq!(w.body(b).linvel.x, 1.0);
    for _ in 0..30 {
        w.step(DT);
    }
    assert!(w.body(b).pose.pos.is_finite());
}

/// MAJOR: 빈 월드에 AddBody만 큐잉하면 `bodies.is_empty()` 조기 반환에 막혀
/// 영원히 적용되지 않았다. 커맨드 플러시는 그 판정보다 **먼저**여야 한다.
#[test]
fn empty_world_applies_pending_add_body() {
    let mut w = World::new(WorldParams::default());
    let h = w.enqueue_add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(0.0, 10.0, 0.0),
        1.0,
    ));
    w.step(DT);
    assert!(w.is_alive(h), "빈 월드의 pending AddBody가 적용되지 않음");
    assert_eq!(w.body_count(), 1);
}

/// 정책 **경계**의 대칭: 라이브 API가 받아들이는 최대치는 스냅샷 리더도 받아야 한다.
/// (리뷰가 지적한 비대칭의 정확한 형태 — 리더만 1e9로 막고 있었다.)
#[test]
fn boundary_legal_values_round_trip() {
    let big = maru_collide::MAX_INPUT * 0.9;
    let fast = maru_collide::MAX_SPEED * 0.9;
    let mut w = World::new(WorldParams::default());
    let h = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(big, 0.0, 0.0),
        1.0,
    ));
    w.set_linear_velocity(h, Vec3::new(fast, 0.0, 0.0)).unwrap();
    let snap = w.snapshot();
    assert!(
        w.restore(&snap).is_ok(),
        "라이브 API가 허용한 최대치를 리더가 거부 — 정책 비대칭"
    );
    assert!((w.body(h).pose().pos.x - big).abs() < big * 1e-6);
}

// ─────────── 리뷰 rev_d76fd99f 회귀 ───────────

/// BLOCKER: 최악 임펄스 논증에서 빠졌던 **조인트 레버암 × 각속도**.
/// 앵커를 셰이프 범위로 묶어 `ω×r` 점 속도를 유계로 만든다.
#[test]
fn large_anchor_with_spinning_kinematic_stays_in_policy() {
    // 앵커는 레버암이므로 전역 상한을 넘을 수 없고,
    // **그 바디의 실제 크기**에도 비례해야 한다(아래 try_add_joint에서 검증).
    assert!(Joint::try_point(
        BodyHandle {
            index: 0,
            generation: 0
        },
        BodyHandle {
            index: 1,
            generation: 0
        },
        Vec3::new(maru_collide::MAX_ANCHOR * 2.0, 0.0, 0.0),
        Vec3::ZERO,
    )
    .is_err());

    // 허용 범위 안의 최대 조합: 큰 앵커 + 상한 각속도 + 무거운 바디
    let mut w = World::new(WorldParams::default());
    let k = w.add_body(Body::new_kinematic(
        Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
    ));
    let heavy = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(2.0, 2.0, 2.0),
            radius: 0.0,
        },
        pose(50.0, 0.0, 0.0),
        1.0e4, // 부피 64m³ × 1e4 = 6.4e5 kg (질량 상한 1e6 안)
    ));
    w.set_angular_velocity(k, Vec3::new(0.0, 100.0, 0.0))
        .unwrap();
    // 앵커는 그 바디 크기에 비례해야 한다 — kinematic 박스(half=1)의 허용 한계 근처
    w.add_joint(Joint::point(k, heavy, Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO));
    for _ in 0..30 {
        // 지원 범위 안의 합법 월드는 StateOverflow를 내면 안 된다
        assert_eq!(
            w.try_step(DT),
            Ok(()),
            "합법 월드가 StateOverflow: sat={}",
            w.diagnostics().state_saturations
        );
    }
    let r = w.restore(&w.snapshot());
    assert!(r.is_ok(), "{r:?}");
}

// ─────────── 리뷰 rev_22b699a7 회귀 ───────────

/// BLOCKER: 새 차원별 범위가 리더에 전파되지 않아 **정상 생성 가능한 경량 바디**가
/// 자기 스냅샷을 복원하지 못했다. 이제 리더가 라이브와 **같은 validator**를 호출한다.
#[test]
fn light_body_with_large_inverses_round_trips() {
    // 반지름 1cm 구, 질량 ≈ 1e-5 kg → inv_mass ≈ 1e5, inv_inertia ≈ 2.5e9
    let mut w = World::new(WorldParams::default());
    let h = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.01 },
        pose(0.0, 1.0, 0.0),
        2.4,
    ));
    let b = w.body(h);
    assert!(b.inv_mass() > 1.0e4, "시나리오 전제 붕괴: {}", b.inv_mass());
    assert!(
        b.inv_inertia_local().x > 1.0e8,
        "시나리오 전제 붕괴: {:?}",
        b.inv_inertia_local()
    );
    let r = w.restore(&w.snapshot());
    assert!(r.is_ok(), "경량 바디 왕복 실패 (차원별 범위 미전파): {r:?}");
}

// ─────────── 리뷰 rev_23aba1ab 회귀 ───────────

/// BLOCKER: 전역 앵커 상한만으로는 부족하다 — 앵커는 **그 바디의 실제 크기**에
/// 비례해야 K 행렬 조건수가 형상 스케일과 무관해진다. 반지름 1mm 구에 1000m 앵커면
/// K의 고유 스케일이 2.4e4 : 6e16이 되어 상대 det가 임계 아래로 떨어지고
/// `Mat3::inverse`가 0 행렬을 돌려준다 — 조인트가 오류 없이 조용히 풀린다.
#[test]
fn anchor_must_scale_with_body() {
    let mut w = World::new(WorldParams::default());
    let tiny = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.001 },
        pose(0.0, 1.0, 0.0),
        1.0e4,
    ));
    let anchor = w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(1.0, 1.0, 1.0),
            radius: 0.0,
        },
        pose(0.0, 3.0, 0.0),
    ));
    // 전역 상한(1000m) 안이지만 1mm 구에는 터무니없는 레버암
    assert_eq!(
        w.try_add_joint(Joint::point(
            tiny,
            anchor,
            Vec3::new(500.0, 0.0, 0.0),
            Vec3::ZERO
        )),
        Err(ApiError::InvalidValue),
        "바디 크기를 무시한 앵커가 통과"
    );
    // 바디 크기에 비례하는 앵커는 정상
    assert!(w
        .try_add_joint(Joint::point(
            tiny,
            anchor,
            Vec3::new(0.002, 0.0, 0.0),
            Vec3::ZERO
        ))
        .is_ok());
}

/// BLOCKER: OOB는 **모든 시뮬레이션 경로**에서 빠져야 한다. 플래그만 분리하고
/// 중력·적분에 그대로 두면 경계에 고정된 바디가 다음 스텝부터 아래로 떨어진다.
/// (직전 테스트는 gravity=0이라 이 결함을 못 봤다.)
#[test]
fn out_of_bounds_body_is_excluded_from_simulation() {
    let mut w = World::new(WorldParams::default()); // **중력 있음**
    let edge = maru_collide::MAX_INPUT - 5.0;
    let flyer = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(edge, 100.0, 0.0),
        1.0,
    ));
    w.set_linear_velocity(flyer, Vec3::new(500.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..10 {
        let _ = w.try_step(crate::world::MAX_DT);
    }
    assert!(w.is_out_of_bounds(flyer));
    let frozen = w.body(flyer).pose().pos;
    // 중력이 있어도 움직이지 않는다
    for _ in 0..120 {
        let _ = w.try_step(DT);
    }
    let now = w.body(flyer).pose().pos;
    assert!(
        (now - frozen).length() < 1e-3,
        "OOB 바디가 중력으로 이동함: {frozen:?} → {now:?}"
    );
    assert_eq!(w.body(flyer).linvel().length(), 0.0, "OOB가 속도를 얻음");
    assert!(!w.is_asleep(flyer), "oob ⇒ !asleep 불변식 위반");
    assert!(w.restore(&w.snapshot()).is_ok());
}

/// MAJOR: SAT 동률이 A/B **대칭**이어야 한다 — 같은 두 박스의 삽입 순서를 뒤집었다고
/// 반력·마찰 방향이 달라지면 물리가 인덱스에 의존하는 것이다.
#[test]
fn sat_tie_break_is_symmetric_under_body_order() {
    let half = Vec3::new(0.5, 0.5, 0.5);
    let mk = |first_at: Vec3, second_at: Vec3| {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let a = w.add_body(Body::new_dynamic(
            Shape::Box { half, radius: 0.0 },
            Pose {
                pos: first_at,
                rot: Quat::IDENTITY,
            },
            1.0,
        ));
        let b = w.add_body(Body::new_dynamic(
            Shape::Box { half, radius: 0.0 },
            Pose {
                pos: second_at,
                rot: Quat::IDENTITY,
            },
            1.0,
        ));
        for _ in 0..30 {
            w.step(DT);
        }
        (w.body(a).pose().pos, w.body(b).pose().pos)
    };
    // 코너 접촉(세 축의 분리가 근소 동률) — 동률 정책이 실제로 개입하는 구성
    let p = Vec3::new(0.0, 0.0, 0.0);
    let q = Vec3::new(0.94, 0.94, 0.94);
    let (a1, b1) = mk(p, q);
    let (b2, a2) = mk(q, p); // 순서를 뒤집어 같은 장면을 만든다
    assert!(
        (a1 - a2).length() < 1e-4 && (b1 - b2).length() < 1e-4,
        "삽입 순서에 따라 결과가 다르다: {a1:?}/{a2:?}, {b1:?}/{b2:?}"
    );
}

/// **월드 축 회전 잠금** (rapier `LockedAxes`의 회전 절반).
///
/// 잠근 축으로는 각속도가 생기지 않고, **안 잠근 축은 그대로 돈다** — 양쪽을 다 봐야
/// "전부 잠갔다"와 "정확히 그 축만 잠갔다"가 구분된다(한쪽만 보면 마스크가 통째로
/// 켜져 있어도 통과한다).
#[test]
fn locked_rotation_axes_do_not_spin() {
    for (lx, ly, lz) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (true, true, true),
    ] {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..no_sleep_params()
        });
        let h = w.add_body(
            Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.3, 0.2),
                    radius: 0.0,
                },
                pose(0.0, 0.0, 0.0),
                1.0,
            )
            .with_locked_rotation(lx, ly, lz),
        );
        assert_eq!(w.body(h).locked_rotation(), (lx, ly, lz));
        // 세 축 모두에 각임펄스를 준다 — 잠긴 축만 반응이 없어야 한다.
        w.apply_angular_impulse(h, Vec3::new(1.0, 1.0, 1.0))
            .unwrap();
        w.step(DT);
        let a = w.body(h).angvel();
        for (lock, v, name) in [(lx, a.x, "x"), (ly, a.y, "y"), (lz, a.z, "z")] {
            if lock {
                assert_eq!(
                    v, 0.0,
                    "잠근 {name}축이 돌았다: {a:?} (마스크 {lx}/{ly}/{lz})"
                );
            } else {
                assert!(
                    v.abs() > 1e-3,
                    "안 잠근 {name}축이 안 돌았다: {a:?} (마스크 {lx}/{ly}/{lz})"
                );
            }
        }
    }
}

/// 회전 잠금은 **스냅샷을 건너 살아남는다** — 복원한 월드가 같은 미래를 만들어야 한다.
#[test]
fn locked_rotation_survives_snapshot() {
    let mut w = World::new(no_sleep_params());
    ground(&mut w);
    let h = w.add_body(
        Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.0,
            },
            pose(0.0, 2.0, 0.0),
            1.0,
        )
        .with_locked_rotation(true, false, true),
    );
    w.apply_angular_impulse(h, Vec3::new(2.0, 2.0, 2.0))
        .unwrap();
    let snap = w.snapshot();
    let mut w2 = World::new(no_sleep_params());
    w2.restore(&snap).expect("복원");
    assert_eq!(w2.body(h).locked_rotation(), (true, false, true));
    for _ in 0..30 {
        w.step(DT);
        w2.step(DT);
    }
    assert_eq!(w.snapshot(), w2.snapshot(), "복원 후 궤적이 갈렸다");
    let a = w.body(h).angvel();
    assert_eq!((a.x, a.z), (0.0, 0.0), "잠근 축이 돌았다: {a:?}");
}

/// **회전한 이방성 바디**의 잠금 유효 관성 — 잠긴 축의 행·열을 0으로 만드는 것(`P·I⁻¹·P`)은
/// 틀린 답이다(교차검수 rev_bb593ab5 BLOCKER).
///
/// 잠금은 이상적 구속이므로 유효 역관성은 `M − M Cᵀ(C M Cᵀ)⁻¹ C M`이고, 이는 허용축 기저
/// `A`에 대해 `A(AᵀIA)⁻¹Aᵀ`와 같다. 관성 주축이 월드 잠금축과 나란할 때만 `P·I⁻¹·P`와 같다.
///
/// 반례: 반크기 (1, 0.1, 0.1) 막대를 **Z로 45° 돌리고** Y·Z를 잠가 월드 X만 허용한다.
/// 정답은 `1/(e_xᵀ I e_x) = 2/(Ix+Iy) = 5.825`인데 `P·I⁻¹·P`는 76.485를 준다(13.13배).
/// **자세를 identity로 두면 이 결함이 안 보인다** — 그래서 회전을 준다.
#[test]
fn locked_rotation_uses_constrained_mobility_not_projection() {
    let (hx, hy, hz) = (1.0f32, 0.1, 0.1);
    // **세 번째 인자는 밀도지 질량이 아니다.** 질량 1을 만들려면 1/부피를 준다
    // (부피 = 8·hx·hy·hz = 0.08). 처음에 1.0을 주고 질량 1로 계산했다가 기대값이
    // 12.5배 어긋났다 — 구현이 아니라 테스트가 틀렸던 것이다.
    let density = 1.0 / (8.0 * hx * hy * hz);
    let (ix, iy) = ((hy * hy + hz * hz) / 3.0, (hx * hx + hz * hz) / 3.0);
    let want = 2.0 / (ix + iy); // = 1/(e_xᵀ I e_x), Z 45° 회전에서 (질량 1)
    let wrong = 0.5 * (1.0 / ix + 1.0 / iy); // P·I⁻¹·P가 주는 값

    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..no_sleep_params()
    });
    let h = w.add_body(
        Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(hx, hy, hz),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 45f32.to_radians()),
            },
            density,
        )
        .with_locked_rotation(false, true, true),
    );
    // 월드 X로 단위 각임펄스 → 각속도 X 성분이 곧 유효 역관성이다.
    w.apply_angular_impulse(h, Vec3::new(1.0, 0.0, 0.0))
        .unwrap();
    let a = w.body(h).angvel();
    assert!(
        (a.x - want).abs() < want * 0.02,
        "구속 유효 역관성이 틀렸다: 실측 {:.4} / 정답 {want:.4} (틀린 답 {wrong:.4})",
        a.x
    );
    assert_eq!((a.y, a.z), (0.0, 0.0), "잠근 축이 돌았다: {a:?}");
}

/// **잠금은 모든 진입점에서 지켜져야 한다** (교차검수 rev_bb593ab5 BLOCKER).
///
/// 유효 역관성을 0으로 만드는 것만으로는 부족하다 — 호스트가 잠긴 축에 각속도를 **직접**
/// 넣으면 솔버는 그걸 없앨 수단이 없고(임펄스가 안 먹는다) 위치 적분이 그대로 돌린다.
/// 검수가 든 세 경로를 모두 막는지 본다: 속도 설정 · 오프센터 임펄스 · 회전 중 잠금 부착.
#[test]
fn locked_rotation_holds_at_every_entry_point() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..no_sleep_params()
    });
    let h = w.add_body(
        Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(0.5, 0.3, 0.2),
                radius: 0.0,
            },
            pose(0.0, 0.0, 0.0),
            1.0,
        )
        .with_locked_rotation(true, false, false),
    );

    // ① 속도 직접 설정
    w.set_angular_velocity(h, Vec3::new(5.0, 5.0, 0.0)).unwrap();
    let a = w.body(h).angvel();
    assert_eq!(
        a.x, 0.0,
        "set_angular_velocity가 잠긴 축을 통과시켰다: {a:?}"
    );
    assert!(a.y > 4.9, "안 잠근 축까지 지워졌다: {a:?}");

    // ② 오프센터 임펄스(스핀을 만든다)
    w.apply_impulse_at_point(h, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.4, 0.0))
        .unwrap();
    let a = w.body(h).angvel();
    assert_eq!(a.x, 0.0, "오프센터 임펄스가 잠긴 축을 돌렸다: {a:?}");

    // ③ 적분이 잠긴 축으로 자세를 돌리지 않는다
    let r0 = w.body(h).pose().rot;
    for _ in 0..30 {
        w.step(DT);
    }
    let r1 = w.body(h).pose().rot;
    // 잠긴 X축 성분의 회전은 없어야 한다 — 쿼터니언 x 성분으로 관측한다.
    assert!(
        (r1.x - r0.x).abs() < 1e-6,
        "잠긴 축으로 자세가 돌았다: {:?} → {:?}",
        r0,
        r1
    );
}

/// 회전 **중인** 바디에 잠금을 붙여 재삽입해도 잠긴 축 각속도가 들어오지 않는다.
/// (검수가 든 우회: `Body`를 clone → `with_locked_rotation` → 재삽입)
#[test]
fn attaching_lock_to_a_spinning_body_projects_its_angvel() {
    let mut w = World::new(no_sleep_params());
    let h = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(0.0, 0.0, 0.0),
        1.0,
    ));
    w.set_angular_velocity(h, Vec3::new(3.0, 3.0, 3.0)).unwrap();
    let spinning = w.body(h).clone();
    let mut w2 = World::new(no_sleep_params());
    let h2 = w2.add_body(spinning.with_locked_rotation(true, false, true));
    let a = w2.body(h2).angvel();
    assert_eq!(
        (a.x, a.z),
        (0.0, 0.0),
        "잠금 부착이 각속도를 안 걸렀다: {a:?}"
    );
    assert!(a.y > 2.9, "안 잠근 축이 지워졌다: {a:?}");
}

/// 예약 비트가 켜진 잠금 마스크는 **거부**한다 — 물리적으로 같은 무잠금 상태가 서로 다른
/// 바이트·상태 해시를 갖는 비정규 상태를 막는다(교차검수 rev_bb593ab5 MINOR).
#[test]
fn locked_rotation_reserved_bits_are_rejected() {
    let mut b = Body::new_dynamic(Shape::Sphere { radius: 0.5 }, pose(0.0, 0.0, 0.0), 1.0);
    b.locked_rot = 0b1000; // 예약 비트
    assert!(b.validate().is_err(), "예약 비트가 통과했다");
    b.locked_rot = 0b111;
    assert!(b.validate().is_ok(), "정상 마스크가 거부됐다");
}
