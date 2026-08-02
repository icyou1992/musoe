//! validation 관련 테스트.

use super::*;

/// static·kinematic은 힘을 **조용히 무시하지 않고** 거부한다.
#[test]
fn forces_are_rejected_on_non_dynamic_bodies() {
    let mut w = World::new(WorldParams::default());
    let s = w.add_body(Body::new_static(
        musoe_collide::Shape::Sphere { radius: 0.5 },
        musoe_collide::Pose {
            pos: musoe_math::Vec3::ZERO,
            rot: musoe_math::Quat::IDENTITY,
        },
    ));
    assert_eq!(
        w.apply_force(s, musoe_math::Vec3::new(1.0, 0.0, 0.0)),
        Err(ApiError::ImmutableStatic)
    );
    assert_eq!(
        w.apply_impulse(s, musoe_math::Vec3::new(1.0, 0.0, 0.0)),
        Err(ApiError::ImmutableStatic)
    );
}

/// 힘 입력은 **입력 범위**로 거른다 (NaN·Inf·상한 초과).
///
/// 누적분에 파생 상한(포화 + `StateOverflow`)을 두긴 하지만, 라이브 API로는 사실상
/// 도달하지 않는다 — 한 번의 입력이 `MAX_INPUT`(1e4)으로 막혀 있어 `MAX_STATE`(1e15)에
/// 닿으려면 1e11번을 불러야 한다. 그 검사의 실제 수요자는 **스냅샷 리더**다(악의적
/// 스냅샷이 거대한 힘을 담을 수 있다). 그래서 여기서는 입력 경계만 확인하고,
/// 복원 경계는 `read_body`가 라이브와 같은 상태 정책으로 읽는 것으로 보장한다.
#[test]
fn force_input_is_validated() {
    let (mut w, h) = unit_sphere_world();
    for bad in [
        musoe_math::Vec3::new(f32::NAN, 0.0, 0.0),
        musoe_math::Vec3::new(f32::INFINITY, 0.0, 0.0),
        musoe_math::Vec3::new(1e30, 0.0, 0.0),
    ] {
        assert_eq!(w.apply_force(h, bad), Err(ApiError::InvalidValue));
        assert_eq!(w.apply_torque(h, bad), Err(ApiError::InvalidValue));
        assert_eq!(w.apply_impulse(h, bad), Err(ApiError::InvalidValue));
    }
    assert_eq!(
        w.body(h).force(),
        musoe_math::Vec3::ZERO,
        "거부는 상태를 남기지 않는다"
    );
}

/// 재질 설정자는 다른 쓰기 경로와 같은 규율을 따른다 — 바꾼 뒤 `validate`를 다시 통과해야 한다.
#[test]
fn material_setters_validate() {
    let mk = || {
        Body::new_dynamic(
            musoe_collide::Shape::Sphere { radius: 0.5 },
            musoe_collide::Pose {
                pos: musoe_math::Vec3::ZERO,
                rot: musoe_math::Quat::IDENTITY,
            },
            1.0,
        )
    };
    assert_eq!(mk().with_friction(0.0).unwrap().friction(), 0.0);
    assert_eq!(mk().with_friction(1.5).unwrap().friction(), 1.5);
    assert_eq!(mk().with_restitution(1.0).unwrap().restitution(), 1.0);
    // 범위 밖은 거부한다 (음수·NaN·반발 1 초과)
    assert!(mk().with_friction(-0.1).is_err());
    assert!(mk().with_friction(f32::NAN).is_err());
    assert!(mk().with_restitution(1.1).is_err());
    assert!(mk().with_restitution(-0.001).is_err());
}

/// **거대한 유한 각은 조용히 항등이 되지 않는다** (MAJOR: 공개 계약 모순).
#[test]
fn huge_finite_axis_angle_is_rejected_not_silently_identity() {
    let axis = Vec3::new(0.0, 0.0, 1.0);
    assert!(Quat::try_from_axis_angle(axis, 1.0).is_some());
    assert!(Quat::try_from_axis_angle(axis, 1.0e9).is_some());
    assert!(
        Quat::try_from_axis_angle(axis, 1.0e20).is_none(),
        "결정적으로 접을 수 없는 각을 받아들였다"
    );
    assert!(Quat::try_from_axis_angle(axis, f32::INFINITY).is_none());
    assert!(Quat::try_from_axis_angle(Vec3::ZERO, 1.0).is_none());
    // 0 축은 데이터에서 자연히 나오는 조건이라 `from_axis_angle`도 항등으로 통과시킨다
    // (아래 패닉 테스트의 **음성 대조**: 패닉이 "무조건"이 아님을 보인다).
    assert_eq!(Quat::from_axis_angle(Vec3::ZERO, 1.0), Quat::IDENTITY);
}

/// **표현 불가 각을 `from_axis_angle`에 넣으면 패닉한다** (조용한 항등 금지).
///
/// `try_` 없는 생성자의 Rust 관례이며, 조용히 항등을 돌려주면 "물체가 왜 회전을
/// 멈췄나"를 추적할 단서가 아무것도 남지 않는다(확인검수 rev_1a31d83f MAJOR 6).
#[test]
#[should_panic(expected = "표현 불가한 각")]
fn from_axis_angle_panics_on_unrepresentable_angle() {
    let _ = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.0e20);
}

#[test]
#[should_panic(expected = "표현 불가한 각")]
fn from_axis_angle_panics_on_nonfinite_angle() {
    let _ = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), f32::NAN);
}

/// **NaN margin이 접촉을 만들어내지 않는다** (MAJOR: 없는 접촉이 생긴다).
///
/// `dist2 > reach*reach` 비교가 NaN 때문에 false가 되면 멀리 떨어진 두 구가 접촉을 만든다.
/// 캐스트에서 막은 "잘못된 입력이 판정으로 위장" 결함이 바로 옆 API에 남아 있었다
/// (확인검수 rev_97887d75 MAJOR).
#[test]
fn contact_api_rejects_bad_arithmetic_input() {
    let a = Shape::Sphere { radius: 0.5 };
    let b = Shape::Sphere { radius: 0.5 };
    let pa = pose(0.0, 0.0, 0.0);
    let far = pose(50.0, 0.0, 0.0);
    for bad in [f32::NAN, -1.0, f32::INFINITY] {
        assert_eq!(
            musoe_collide::for_each_contact(&a, &pa, &b, &far, bad, &mut |_, _, _| {}),
            musoe_collide::ContactWalk::InvalidInput,
            "margin={bad}"
        );
        assert_eq!(musoe_collide::any_contact(&a, &pa, &b, &far, bad), None);
    }
    // **음성 대조** — 정상 margin이면 멀리 있는 쌍은 접촉이 없고, 가까운 쌍은 있다.
    assert_eq!(
        musoe_collide::any_contact(&a, &pa, &b, &far, 0.02),
        Some(false)
    );
    assert_eq!(
        musoe_collide::any_contact(&a, &pa, &b, &pose(0.8, 0.0, 0.0), 0.02),
        Some(true)
    );
}

/// **형상 내부 관계가 깨진 박스는 모든 공개 질의가 같이 거절한다** (BLOCKER: 질의끼리 불일치).
///
/// `Box { half: 1, radius: 2 }`는 `validate`가 거절하지만 질의 경계는 범위만 봐서
/// 통과시켰다. 그 형상에서 AABB·점 질의는 반경 1로, 레이·GJK는 반경 2로 답한다 —
/// 호스트가 공개 AABB로 후보를 쳐낸 뒤 레이를 쓰면 히트가 사라진다
/// (확인검수 rev_b85e4e4a BLOCKER).
#[test]
fn box_with_radius_larger_than_half_is_rejected_everywhere() {
    use musoe_collide::{CastReject, QueryReject, ShapeReject};
    let bad = Shape::Box {
        half: Vec3::new(1.0, 1.0, 1.0),
        radius: 2.0,
    };
    let p = pose(0.0, 0.0, 0.0);
    assert!(bad.validate().is_err(), "생성 경계는 원래 거절했다");
    assert_eq!(bad.try_local_aabb(), Err(ShapeReject::InvalidShape));
    assert_eq!(
        musoe_collide::try_contains_point(&bad, &p, Vec3::new(1.5, 0.0, 0.0)),
        Err(QueryReject::InvalidInput)
    );
    assert_eq!(
        musoe_collide::try_cast_ray(
            &bad,
            &p,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            10.0
        ),
        Err(QueryReject::InvalidInput)
    );
    assert_eq!(
        musoe_collide::try_shape_cast_dispatch(
            &bad,
            &p,
            Vec3::new(1.0, 0.0, 0.0),
            &Shape::Sphere { radius: 0.5 },
            &pose(5.0, 0.0, 0.0),
            0.02
        ),
        Err(CastReject::InvalidInput)
    );
    assert_eq!(
        musoe_collide::any_contact(&bad, &p, &Shape::Sphere { radius: 0.5 }, &p, 0.02),
        None
    );
    // **음성 대조** — 관계가 성립하는 둥근 박스(radius ≤ min half)는 전부 통과한다.
    let ok = Shape::Box {
        half: Vec3::new(1.0, 1.0, 1.0),
        radius: 0.4,
    };
    assert!(ok.validate().is_ok());
    assert!(ok.try_local_aabb().is_ok());
    assert_eq!(
        musoe_collide::try_contains_point(&ok, &p, Vec3::new(0.5, 0.0, 0.0)),
        Ok(true)
    );
}

/// **지원 범위 밖 입력은 "유한하다"는 이유로 통과하지 못한다** (BLOCKER: 거짓 히트/미스).
///
/// 두 가지가 독립적으로 재현됐다(확인검수 rev_b85e4e4a):
/// ① `axis=(f32::MAX,0,0)`은 유한하지만 `length_sq`가 `Inf`라 `try_normalize`가 `None`을
///    내고, 그것을 "축이 없다"로 읽으면 **회전 캐스트가 선형으로 폴백해 제자리 회전
///    히트를 놓친다**.
/// ② 자세를 `±f32::MAX`에 두면 중심 차가 `Inf`가 되고 GJK의 비유한 위생이 그것을 거리 0으로
///    강등해 **무한히 멀리 떨어진 두 형상이 `toi=0` 히트**가 된다.
#[test]
fn out_of_range_arithmetic_input_is_rejected_not_reinterpreted() {
    use musoe_collide::{CastReject, ContactWalk};
    let blade = Shape::Box {
        half: Vec3::new(1.5, 0.05, 0.05),
        radius: 0.0,
    };
    let wall = Shape::Box {
        half: Vec3::new(0.2, 1.0, 1.0),
        radius: 0.0,
    };
    // ① 거대한 축 — 0축(회전 없음)으로 오해하면 안 된다.
    assert_eq!(
        musoe_collide::try_shape_cast_dispatch_rotational(
            &blade,
            &pose(0.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(f32::MAX, 0.0, 0.0),
            core::f32::consts::FRAC_PI_2,
            &wall,
            &pose(0.0, 0.0, 2.0),
            0.02
        ),
        Err(CastReject::InvalidInput)
    );
    // ② 범위 밖 자세 — 무한히 먼 두 형상이 접촉으로 나오면 안 된다.
    let far_a = pose(f32::MAX, 0.0, 0.0);
    let far_b = pose(-f32::MAX, 0.0, 0.0);
    let s = Shape::Sphere { radius: 0.5 };
    assert_eq!(
        musoe_collide::try_shape_cast_dispatch(
            &s,
            &far_a,
            Vec3::new(-1.0, 0.0, 0.0),
            &s,
            &far_b,
            0.02
        ),
        Err(CastReject::InvalidInput)
    );
    assert_eq!(
        musoe_collide::for_each_contact(&s, &far_a, &s, &far_b, 0.02, &mut |_, _, _| {}),
        ContactWalk::InvalidInput
    );
    assert_eq!(
        musoe_collide::any_contact(&s, &far_a, &s, &far_b, 0.02),
        None
    );
    // **음성 대조** — 같은 배치를 범위 안(±5000m)으로 옮기면 정상 판정이다.
    let a = pose(5000.0, 0.0, 0.0);
    let b = pose(-5000.0, 0.0, 0.0);
    assert_eq!(
        musoe_collide::any_contact(&s, &a, &s, &b, 0.02),
        Some(false),
        "범위 안의 먼 쌍은 '접촉 없음'이 확정 답이다"
    );
    // 그리고 정상 회전 캐스트는 실제로 맞춘다(가드가 회전 경로를 죽인 게 아니다).
    let hit = musoe_collide::try_shape_cast_dispatch_rotational(
        &blade,
        &pose(0.0, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        core::f32::consts::FRAC_PI_2,
        &wall,
        &pose(0.0, 0.0, 1.7),
        0.02,
    )
    .expect("정상 입력");
    assert!(
        hit.is_some(),
        "정상 회전 스윕이 벽을 못 맞췄다 — 배치가 무의미하다"
    );
}

/// MAJOR: `World.params`는 비공개 — 교체도 생성자와 같은 검증을 거친다.
#[test]
#[should_panic(expected = "substeps must be in 1..=64")]
fn set_params_revalidates() {
    let mut w = World::new(WorldParams::default());
    w.set_params(WorldParams {
        substeps: 0,
        ..Default::default()
    });
}

// ─────────── 리뷰 rev_1517fd7a 회귀 ───────────

/// BLOCKER: `#[non_exhaustive]`만으로는 필드 변조를 못 막는다 → 필드 비공개화 +
/// **삽입 경계 재검증**. (컴파일 차단 자체는 body.rs의 compile_fail 독테스트가 증명한다.)
#[test]
#[should_panic(expected = "invalid body")]
fn add_body_revalidates_mutated_body() {
    let mut w = World::new(WorldParams::default());
    let mut b = Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 1.0, 0.0), 1.0);
    // 크레이트 안에서만 가능한 변조를 흉내 — 밖에서는 컴파일되지 않는다
    b.shape = Shape::Hull {
        data: musoe_collide::HullData::new(vec![]),
        radius: 0.0,
    };
    w.add_body(b);
}

/// NEW MAJOR: 생성자가 `is_finite`만 봐서 0 쿼터니언이 통과했다.
#[test]
#[should_panic(expected = "normalizable")]
fn zero_quaternion_rejected_at_construction() {
    Body::new_dynamic(
        Shape::Sphere { radius: 1.0 },
        Pose {
            pos: Vec3::ZERO,
            rot: Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        },
        1.0,
    );
}

/// PARTIAL MAJOR: 검사하지 않던 파라미터들. NaN 임계는 비교가 항상 false라
/// 게이트를 조용히 무력화한다.
#[test]
#[should_panic(expected = "restitution_threshold")]
fn nan_restitution_threshold_rejected() {
    World::new(WorldParams {
        solver: crate::solver::SolverParams {
            restitution_threshold: f32::NAN,
            ..Default::default()
        },
        ..Default::default()
    });
}

// ─────────── 리뷰 rev_9a7af148 회귀 ───────────

/// NEW BLOCKER: 유한하지만 과대한 값은 몇 서브스텝 만에 Inf가 된다 —
/// `is_finite()`만으로는 방어가 아니다. 라이브 API도 스냅샷과 **같은 정책**을 쓴다.
#[test]
fn live_api_rejects_huge_finite_values() {
    let mut w = World::new(WorldParams::default());
    let b = dyn_box(&mut w, 0.0, 3.0);
    assert_eq!(
        w.set_linear_velocity(b, Vec3::new(f32::MAX, 0.0, 0.0)),
        Err(ApiError::InvalidValue),
        "f32::MAX 속도가 통과"
    );
    assert_eq!(
        w.enqueue(Command::SetLinearVelocity(b, Vec3::new(f32::MAX, 0.0, 0.0))),
        Err(ApiError::InvalidValue)
    );
    assert_eq!(
        w.set_pose(
            b,
            Pose {
                pos: Vec3::new(1.0e20, 0.0, 0.0),
                rot: Quat::IDENTITY
            }
        ),
        Err(ApiError::InvalidValue)
    );
}

#[test]
#[should_panic(expected = "MAX_INPUT")]
fn huge_pose_rejected_at_construction() {
    Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(1.0e20, 0.0, 0.0), 1.0);
}

/// NEW MAJOR: 생성자가 자기 삽입 정책을 위반하는 Body를 반환하면 안 된다
/// (density=1e-10 → inv_mass≈2.39e9 > 입력 상한).
#[test]
#[should_panic(expected = "size policy")]
fn constructor_rejects_body_violating_insertion_policy() {
    Body::new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 0.0, 0.0), 1.0e-10);
}

/// PARTIAL BLOCKER: 호스트 입력 경로는 **전부** fallible이어야 한다 — wasm trap 금지.
#[test]
fn all_host_input_paths_are_fallible() {
    // 월드 파라미터
    assert!(World::try_new(WorldParams {
        substeps: 0,
        ..Default::default()
    })
    .is_err());
    let mut w = World::try_new(WorldParams::default()).unwrap();
    assert!(w
        .try_set_params(WorldParams {
            time_to_sleep: -1.0,
            ..Default::default()
        })
        .is_err());
    // 바디 (dynamic / static / kinematic)
    let bad = Shape::Hull {
        data: musoe_collide::HullData::new(vec![]),
        radius: 0.0,
    };
    assert!(Body::try_new_dynamic(bad.clone(), pose(0.0, 0.0, 0.0), 1.0).is_err());
    assert!(Body::try_new_static(bad.clone(), pose(0.0, 0.0, 0.0)).is_err());
    assert!(Body::try_new_kinematic(bad, pose(0.0, 0.0, 0.0)).is_err());
    // 조인트 — hinge 축은 Joint 생성 시점에 걸리므로 try_add_joint로는 못 잡는다
    let a = dyn_box(&mut w, 0.0, 3.0);
    let b = dyn_box(&mut w, 2.0, 3.0);
    assert!(Joint::try_hinge(
        a,
        b,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0)
    )
    .is_err());
    assert!(Joint::try_hinge(
        a,
        b,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(f32::NAN, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0)
    )
    .is_err());
    assert!(Joint::try_point(a, b, Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::ZERO).is_err());
    assert!(Joint::try_hinge(
        a,
        b,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0)
    )
    .is_ok());
    // 스텝·스냅샷·해시
    assert!(w.try_step(0.0).is_err());
    assert!(w.try_snapshot().is_ok());
    assert!(w.try_state_hash().is_ok());
}

/// BLOCKER: pending 커맨드 페이로드는 **호스트 입력**이므로 공개 enqueue와 같은 정책이다.
#[test]
fn pending_payload_uses_input_policy() {
    let mut w = World::new(WorldParams::default());
    let h = dyn_box(&mut w, 0.0, 3.0);
    // 공개 경로가 거부하는 값은 스냅샷 왕복으로도 큐에 들어올 수 없다
    assert!(w
        .enqueue(Command::SetLinearVelocity(
            h,
            Vec3::new(musoe_collide::MAX_SPEED * 2.0, 0.0, 0.0)
        ))
        .is_err());
    assert!(w
        .enqueue(Command::SetAngularVelocity(
            h,
            Vec3::new(musoe_collide::MAX_ANGULAR * 2.0, 0.0, 0.0)
        ))
        .is_err());
    // 정상 범위는 통과하고 왕복도 성립
    w.enqueue(Command::SetLinearVelocity(h, Vec3::new(10.0, 0.0, 0.0)))
        .unwrap();
    w.enqueue(Command::SetAngularVelocity(h, Vec3::new(1.0, 0.0, 0.0)))
        .unwrap();
    assert!(w.restore(&w.snapshot()).is_ok());
}

// ─────────── 리뷰 rev_c18c4dc0 회귀 ───────────

/// BLOCKER: 형상·앵커 휴리스틱으로는 **이방성** 조합의 K 특이화를 막을 수 없다.
/// 리뷰가 제시한 구성(길쭉한 박스 + 얇은 축 방향 앵커)은 전역·반경 한계를 모두
/// 통과하지만 K가 diag(4, 1, 1.5e12)가 되어 `Mat3::inverse`가 0 행렬을 돌려준다.
/// 이제 삽입 시 **실제 K**를 만들어 풀리는지 확인한다.
#[test]
fn anisotropic_body_with_long_anchor_is_rejected_by_k_check() {
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
        Err(ApiError::InvalidValue),
        "K가 특이해지는 조인트가 통과 — 조용히 풀린다"
    );
    // 같은 바디라도 **긴 축** 방향 앵커는 조건수가 멀쩡하므로 허용돼야 한다
    assert!(w
        .try_add_joint(Joint::point(
            rod,
            anchor,
            Vec3::new(1000.0, 0.0, 0.0),
            Vec3::ZERO
        ))
        .is_ok());
}

// ─────────── 리뷰 rev_63c4f4fc 회귀 ───────────

/// BLOCKER(L0): 파생 캐시(브로드페이즈 페어)의 무효화가 **한 경로라도** 빠지면
/// 같은 스냅샷의 미래가 캐시 유무로 갈린다. pending Add 실현 경로가 빠져 있었다.
///
/// 재현: static만 있는 월드를 한 스텝 돌려 빈 페어 캐시를 valid로 만든 뒤,
/// 커맨드로 겹치는 dynamic을 추가한다. 원본은 캐시를 재사용해 접촉 없이 적분하고,
/// 복원본은 캐시가 없어 페어를 재빌드해 접촉을 푼다 → 궤적이 갈린다.
#[test]
fn derived_cache_invalidation_is_complete_for_pending_add() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    w.step(DT); // 빈 페어 캐시를 valid로 만든다
                // 지면과 **겹치도록** 커맨드로 추가한다
    let h = w.enqueue_add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.5, 0.5, 0.5),
            radius: 0.0,
        },
        pose(0.0, 0.2, 0.0),
        1.0,
    ));
    w.flush_commands();
    let snap = w.snapshot();

    let mut restored = World::new(WorldParams::default());
    restored.restore(&snap).unwrap();
    // 같은 스냅샷에서 출발했으면 미래도 같아야 한다
    for i in 0..30 {
        w.step(DT);
        restored.step(DT);
        assert_eq!(
            w.state_hash(),
            restored.state_hash(),
            "스텝 {i}에서 궤적이 갈렸다 — 파생 캐시가 L0를 깨고 있다"
        );
    }
    assert!(w.is_alive(h));
}

/// NEW MAJOR: dt 위반은 **wasm trap이 아니라 오류**로 표현돼야 한다.
/// 고정틱 엔진이므로 큰 dt를 쪼개는 것은 호스트(accumulator) 책임이다.
#[test]
fn huge_dt_returns_error_not_trap() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    assert_eq!(w.try_step(8.0), Err(ApiError::InvalidValue));
    assert_eq!(w.try_step(f32::NAN), Err(ApiError::InvalidValue));
    assert_eq!(w.try_step(crate::world::MAX_DT), Ok(()), "상한 자체는 합법");
    // 문서와 구현 통일(리뷰 MAJOR): 0과 음수는 **합법이 아니다**.
    // 일시정지는 step을 부르지 않는 것으로 표현한다 — "성공했는데 아무 일도 없음"이
    // 오류와 구분되지 않으면 호스트가 정지 원인을 진단할 수 없다.
    assert_eq!(w.try_step(0.0), Err(ApiError::InvalidValue));
    assert_eq!(w.try_step(-DT), Err(ApiError::InvalidValue));
    assert_eq!(
        w.try_step(crate::world::MIN_DT * 0.5),
        Err(ApiError::InvalidValue),
        "너무 작은 dt는 1/h를 폭발시킨다"
    );
}

#[test]
fn fallible_constructor_returns_error() {
    assert!(
        Body::try_new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 0.0, 0.0), 1.0e-10).is_err()
    );
    assert!(
        Body::try_new_dynamic(Shape::Sphere { radius: 1.0 }, pose(1.0e20, 0.0, 0.0), 1.0).is_err()
    );
    assert!(Body::try_new_dynamic(Shape::Sphere { radius: 1.0 }, pose(0.0, 0.0, 0.0), 1.0).is_ok());
}

/// MAJOR: 좌표 정밀도와 최소 특징 크기의 정합.
#[test]
fn world_extent_matches_float_precision() {
    // 월드 경계에서의 f32 간격이 최소 특징 크기 이하여야 한다
    let spacing = {
        let x = musoe_collide::MAX_INPUT;
        let next = f32::from_bits(x.to_bits() + 1);
        next - x
    };
    assert!(
        spacing <= musoe_collide::MIN_SHAPE_EXTENT,
        "월드 경계 f32 간격 {spacing} > 최소 특징 크기 {}",
        musoe_collide::MIN_SHAPE_EXTENT
    );
    // 최소 크기 미만 셰이프는 거부된다
    assert!(Body::try_new_dynamic(
        Shape::Sphere {
            radius: musoe_collide::MIN_SHAPE_EXTENT * 0.5
        },
        pose(0.0, 0.0, 0.0),
        1.0
    )
    .is_err());
}
