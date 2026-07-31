//! misc 관련 테스트.

use super::*;

/// **비단위 자식 자세가 기하를 왜곡하지 않는다** (BLOCKER: 1m 규모의 조용한 오차).
///
/// 수용 기준(길이 제곱 1e-3)은 *입력 관용*이고, 기하는 **정규화된 회전으로** 풀어야 한다.
/// 비단위 쿼터니언의 `rotate`는 회전이 아니라 회전+스케일이라, 최대 형상 반경(1000m)에서
/// 0.05%는 약 0.5m다 — speculative margin(0.05m)의 10배, 침투 임계(1e-4m)의 5000배다
/// (확인검수 rev_201e3d20 BLOCKER). AABB·레이·접촉·관성이 모두 이 값을 쓴다.
#[test]
fn non_unit_child_rotation_does_not_distort_geometry() {
    let child = Shape::Box {
        half: Vec3::new(300.0, 5.0, 5.0),
        radius: 0.0,
    };
    let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2);
    // 수용 범위 안에서 최대로 부풀린 자식 회전(길이 제곱 ≈ 1+1e-3).
    let k = (1.0f32 + 9.0e-4).sqrt();
    let mk = |rot: Quat| Shape::Compound {
        children: vec![(
            maru_collide::Pose {
                pos: Vec3::new(400.0, 0.0, 0.0),
                rot,
            },
            child.clone(),
        )],
    };
    let unit = mk(q);
    let scaled = mk(Quat {
        x: q.x * k,
        y: q.y * k,
        z: q.z * k,
        w: q.w * k,
    });
    assert!(scaled.validate().is_ok(), "수용 범위 안인데 거절됐다");
    // ① AABB가 같아야 한다.
    let (a, b) = (
        unit.try_local_aabb().expect("정상"),
        scaled.try_local_aabb().expect("정상"),
    );
    let d = (a.min - b.min).length().max((a.max - b.max).length());
    assert!(
        d < 1.0e-3,
        "비단위 자식이 AABB를 {d} m 만큼 왜곡했다 (침투 임계 1e-4 m)"
    );
    // ② 레이 히트 지점도 같아야 한다.
    let p = pose(0.0, 0.0, 0.0);
    // 자식은 (400,0,0)에 있고 90° 돌아 y로 길어진다 — 그 위에서 아래로 쏜다.
    let o = Vec3::new(400.0, 900.0, 0.0);
    let dir = Vec3::new(0.0, -1.0, 0.0);
    let ha = maru_collide::try_cast_ray(&unit, &p, o, dir, 2000.0).expect("정상");
    let hb = maru_collide::try_cast_ray(&scaled, &p, o, dir, 2000.0).expect("정상");
    // **음성 대조** — 정상 자세가 실제로 맞춰야 비교가 의미 있다.
    let ta = ha
        .expect("정상 자세가 회전한 자식을 못 맞췄다 — 배치가 무의미하다")
        .toi;
    let tb = hb.expect("비단위 자세에서 히트가 사라졌다").toi;
    assert!(
        (ta - tb).abs() < 1.0e-3,
        "레이 toi가 {} vs {}로 갈렸다",
        ta,
        tb
    );
    // ③ 관성·질량도 같아야 한다(동적 복합은 자식 회전으로 텐서를 돌린다).
    let ba = Body::try_new_dynamic(unit, pose(0.0, 0.0, 0.0), 1.0).expect("정상");
    let bb = Body::try_new_dynamic(scaled, pose(0.0, 0.0, 0.0), 1.0).expect("정상");
    assert!((ba.inv_mass() - bb.inv_mass()).abs() <= f32::EPSILON * 4.0);
    let (ia, ib) = (ba.inv_inertia_local(), bb.inv_inertia_local());
    let rel = ((ia - ib).length() / ia.length().max(1.0e-12)).abs();
    assert!(rel < 1.0e-3, "관성 텐서가 상대오차 {rel}로 갈렸다");
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

/// MAJOR: 포화는 **관측 가능**해야 한다. 서로 다른 초과 상태가 같은 값으로 합쳐지므로
/// 상태 해시만으로는 물리 오류가 숨는다.
#[test]
fn state_saturation_is_observable_and_reported() {
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let h = dyn_box(&mut w, 0.0, 3.0);
    w.step(DT);
    assert_eq!(
        w.diagnostics().state_saturations,
        0,
        "정상 장면에서 포화 발생"
    );
    // 지원 범위 안에서는 StateOverflow가 나지 않는다
    assert_eq!(w.try_step(DT), Ok(()));
    assert!(w.body(h).pose().pos.y.is_finite());
}

/// MAJOR: 좌표 정밀도와 최소 특징 크기의 정합.
#[test]
fn world_extent_matches_float_precision() {
    // 월드 경계에서의 f32 간격이 최소 특징 크기 이하여야 한다
    let spacing = {
        let x = maru_collide::MAX_INPUT;
        let next = f32::from_bits(x.to_bits() + 1);
        next - x
    };
    assert!(
        spacing <= maru_collide::MIN_SHAPE_EXTENT,
        "월드 경계 f32 간격 {spacing} > 최소 특징 크기 {}",
        maru_collide::MIN_SHAPE_EXTENT
    );
    // 최소 크기 미만 셰이프는 거부된다
    assert!(Body::try_new_dynamic(
        Shape::Sphere {
            radius: maru_collide::MIN_SHAPE_EXTENT * 0.5
        },
        pose(0.0, 0.0, 0.0),
        1.0
    )
    .is_err());
}

/// MAJOR: 앵커 척도는 **바디 원점 기준**이어야 한다 — AABB 크기만 쓰면 원점에서
/// 떨어진 static hull의 정상 표면 앵커가 거부된다.
#[test]
fn anchor_scale_uses_origin_distance_not_size() {
    let mut w = World::new(WorldParams::default());
    // local x≈100m에 놓인 1m 크기 static hull (원점에서 멀지만 합법)
    let far = w.add_body(Body::new_static(
        Shape::Hull {
            points: vec![
                Vec3::new(99.5, -0.5, -0.5),
                Vec3::new(100.5, -0.5, -0.5),
                Vec3::new(99.5, 0.5, -0.5),
                Vec3::new(99.5, -0.5, 0.5),
                Vec3::new(100.5, 0.5, 0.5),
            ],
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
    ));
    let b = dyn_box(&mut w, 0.0, 3.0);
    // 그 hull의 실제 표면 앵커
    assert!(
        w.try_add_joint(Joint::point(far, b, Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO))
            .is_ok(),
        "원점에서 떨어진 셰이프의 정상 앵커가 거부됨"
    );
}

/// 로드형 진자(작은 추가 1m 밖에 매달림)는 **정상 구성**이다 —
/// 축별 결합 같은 과잉 제약으로 막으면 안 된다.
#[test]
fn rod_style_pendulum_anchor_is_allowed() {
    let mut w = World::new(WorldParams::default());
    let a = w.add_body(Body::new_static(
        Shape::Sphere { radius: 0.1 },
        pose(0.0, 5.0, 0.0),
    ));
    let bob = w.add_body(Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(0.2, 0.2, 0.2),
            radius: 0.0,
        },
        pose(1.0, 5.0, 0.0),
        1.0,
    ));
    assert!(w
        .try_add_joint(Joint::point(a, bob, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0)))
        .is_ok());
}

/// SAT 보조 키는 **부호 불변**이어야 A/B 교환에 대칭이다(signed 사전식은 `n → -n`에서
/// 대소 순서가 보존되지 않아 다른 축을 고른다).
///
/// 일반 구성(`|d·n|`이 유의미)과 **퇴화 구성**(거의 동심 + 수직 법선)을 함께 잠근다.
/// 퇴화에서는 `d`가 방향을 못 정하므로 **바디 자체 데이터**(치수·자세)로 만든
/// 교환 반대칭 키가 방향을 정한다 — manifold.rs의 `oriented` 주석 참조.
#[test]
fn sat_tie_break_symmetric_for_rotated_boxes() {
    let half = Vec3::new(0.5, 0.5, 0.5);
    let rot_a = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3);
    let rot_b = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.4);
    let mk = |pa: Vec3, qa: Quat, pb: Vec3, qb: Quat| {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let a = w.add_body(Body::new_dynamic(
            Shape::Box { half, radius: 0.0 },
            Pose { pos: pa, rot: qa },
            1.0,
        ));
        let b = w.add_body(Body::new_dynamic(
            Shape::Box { half, radius: 0.0 },
            Pose { pos: pb, rot: qb },
            1.0,
        ));
        for _ in 0..40 {
            w.step(DT);
        }
        (w.body(a).pose().pos, w.body(b).pose().pos)
    };
    // 같은 자세를 **부호가 뒤집힌 쿼터니언**으로 표현해도 결과가 같아야 한다
    // (q와 -q는 같은 회전 — 표현이 물리를 바꾸면 안 된다).
    {
        let neg = |q: Quat| Quat {
            x: -q.x,
            y: -q.y,
            z: -q.z,
            w: -q.w,
        };
        let base = mk(Vec3::ZERO, rot_a, Vec3::new(0.02, 0.01, 0.0), rot_b);
        let flipped = mk(
            Vec3::ZERO,
            neg(rot_a),
            Vec3::new(0.02, 0.01, 0.0),
            neg(rot_b),
        );
        assert!(
            (base.0 - flipped.0).length() < 1e-3 && (base.1 - flipped.1).length() < 1e-3,
            "쿼터니언 부호 표현이 물리를 바꾼다: {:?} vs {:?}",
            base,
            flipped
        );
    }
    for (label, pa, pb) in [
        // 일반 구성
        ("일반", Vec3::ZERO, Vec3::new(0.62, 0.31, 0.44)),
        // **퇴화**: 거의 동심 → 정렬 키가 전부 0이고 d가 방향을 못 정한다
        ("퇴화(동심)", Vec3::ZERO, Vec3::new(0.02, 0.01, 0.0)),
        // **퇴화**: 완전 동심
        ("퇴화(완전동심)", Vec3::ZERO, Vec3::ZERO),
    ] {
        let (a1, b1) = mk(pa, rot_a, pb, rot_b);
        let (b2, a2) = mk(pb, rot_b, pa, rot_a);
        assert!(
            (a1 - a2).length() < 1e-3 && (b1 - b2).length() < 1e-3,
            "{label}: 삽입 순서에 따라 결과가 다르다: {a1:?}/{a2:?}, {b1:?}/{b2:?}"
        );
    }
}

/// 내부 에지 판정(§4.2 ghost edge): 공면 공유 에지만 용접 마스크가 선다 —
/// 경계·볼록 능선·비-다양체 에지는 제외.
#[test]
fn terrain_marks_only_coplanar_shared_edges_internal() {
    let mut w = World::new(WorldParams::default());
    // 평평한 사각형 = 2 삼각형, 공유 에지 (1,2)는 공면 → 양쪽 용접.
    let quad = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(-1.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
    ];
    let hs = w
        .try_add_terrain(&quad, &[[0, 1, 2], [1, 3, 2]], 0.02)
        .unwrap();
    let mask = |h: BodyHandle, w: &World| w.body(h).weld_edges;
    // T0의 공유 에지 = (v1,v2) = 에지 1, T1의 공유 에지 = (v1,v2)… T1=[1,3,2]에서
    // (1,2)는 에지 2다. 경계 에지는 비트가 서면 안 된다.
    assert_eq!(mask(hs[0], &w), 0b010, "T0: 공유 에지(1→2)만 내부");
    assert_eq!(mask(hs[1], &w), 0b100, "T1: 공유 에지(2→1)만 내부");

    // 90° 능선: 공유 에지가 공면이 아니다 → 용접 없음.
    let ridge = [
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0), // 수평 면
        Vec3::new(0.0, 1.0, 0.0),  // 수직 면
    ];
    let hs = w
        .try_add_terrain(&ridge, &[[0, 1, 2], [0, 3, 1]], 0.02)
        .unwrap();
    assert_eq!(mask(hs[0], &w), 0, "능선은 용접 금지");
    assert_eq!(mask(hs[1], &w), 0, "능선은 용접 금지");

    // 비-다양체(3면 공유): 보수적으로 용접 없음.
    let fan = [
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.5),
    ];
    let hs = w
        .try_add_terrain(&fan, &[[0, 1, 2], [0, 1, 3], [0, 1, 4]], 0.02)
        .unwrap();
    for h in &hs {
        assert_eq!(mask(*h, &w), 0, "비-다양체 에지는 용접 금지");
    }
}
