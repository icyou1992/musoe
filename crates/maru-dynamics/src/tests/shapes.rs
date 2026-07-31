//! shapes 관련 테스트.

use super::*;

/// 삼각형 메시가 바디를 **떠받친다** — TriMesh 접촉 디스패치의 종단 검증.
///
/// 이 테스트가 없으면 BVH·펼침·용접이 각각 단위 테스트를 통과하면서도 조립하면
/// 물체가 메시를 통과하는 상태가 될 수 있다(조각은 옳은데 배선이 틀린 경우).
///
/// **"제자리에 선다"가 아니라 "표면 위에 있다"를 잰다.** 구가 삼각형 내부 에지 근처에
/// 떨어지면 접선 임펄스를 받아 굴러가는데, 그건 이 구현의 결함이 아니라 **하이트필드도
/// 같은** 기존 거동이다(같은 씬을 하이트필드로 돌려 확인했다: 메시 v≈0.75, 하이트필드
/// v≈0.88로 둘 다 굴러간다). 그래서 굴러가는 동안 **표면 높이를 유지하는지**를 본다 —
/// 그것이 "떠받친다"의 정의이고, 통과하면 즉시 깨진다.
/// **wasm 레그에서는 건너뛴다** — 800삼각형 씬 300스텝이 단일 wasm 인스턴스에서
/// 순차로 도는 전체 스위트의 메모리 예산을 넘긴다(격리 실행은 통과 — 레그의 예산
/// 문제이지 이 테스트의 결함이 아니다). 접촉 디스패치는 타깃 무관이라 native 레그로 족하다.
#[test]
#[cfg_attr(target_arch = "wasm32", ignore)]
fn triangle_mesh_supports_a_body() {
    let mut w = World::new(WorldParams::default());
    // 20×20 격자를 삼각형 800개로 — 평평한 바닥 y=0, x·z ∈ [−10, 10].
    // 구가 굴러도 시험 구간 동안 메시 위에 남을 만큼 넓게 잡는다.
    let (mut verts, mut tris) = (Vec::new(), Vec::new());
    let n = 21usize;
    for iz in 0..n {
        for ix in 0..n {
            verts.push(Vec3::new(ix as f32 - 10.0, 0.0, iz as f32 - 10.0));
        }
    }
    for iz in 0..n - 1 {
        for ix in 0..n - 1 {
            let i = (iz * n + ix) as u32;
            let (r, d) = (i + 1, i + n as u32);
            tris.push([i, r, d]);
            tris.push([r, d + 1, d]);
        }
    }
    assert_eq!(tris.len(), 800);
    let mesh = maru_collide::Shape::trimesh(verts, tris).expect("정상 메시");
    w.add_body(Body::new_static(mesh, pose(0.0, 0.0, 0.0)));
    let ball = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(0.1, 3.0, -0.2),
        1.0,
    ));
    let mut landed = false;
    let mut worst_pen = 0.0f32;
    for step in 0..300 {
        w.step(1.0 / 60.0);
        let p = w.body(ball).pose().pos;
        assert!(p.y.is_finite(), "step {step}: 발산했다");
        // 메시 범위 안에 있는 동안만 판정한다(밖으로 나가면 떨어지는 게 맞다).
        if p.x.abs() < 9.0 && p.z.abs() < 9.0 {
            if p.y < 0.55 {
                landed = true;
            }
            if landed {
                // 반지름 0.5 구의 중심은 표면에서 0.5 — 통과하면 이 값이 급격히 내려간다.
                assert!(
                    p.y > 0.4,
                    "step {step}: 구가 메시를 통과했다 y={} (기대 ≈0.5)",
                    p.y
                );
                worst_pen = worst_pen.max(w.diagnostics().max_penetration);
            }
        }
    }
    assert!(landed, "구가 메시에 닿지도 않았다 — 접촉이 아예 안 났다");
    assert!(
        worst_pen < 0.05,
        "침투가 과하다: {worst_pen} m (접촉은 났지만 밀어내지 못한다)"
    );
}

/// 메시는 **static 전용**이다 — dynamic 생성이 통과하지 않는다.
///
/// 계약은 하이트필드와 같다: `mass_properties`가 0을 돌려주고 `Body::new_dynamic`의
/// 질량 단언이 잡는다(구성 시점의 프로그래밍 오류로 취급 — 호스트 입력 경계인
/// `Shape::validate`와는 다른 층이다). 통과시키면 질량 0인 dynamic 바디가 생겨
/// 솔버의 역질량이 발산한다.
#[test]
#[should_panic(expected = "mass")]
fn trimesh_cannot_be_dynamic() {
    let mesh = maru_collide::Shape::trimesh(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        vec![[0, 1, 2]],
    )
    .expect("정상 메시");
    let _ = Body::new_dynamic(mesh, pose(0.0, 0.0, 0.0), 1.0);
}

/// **메시는 kinematic도 될 수 없다** (교차검수 rev_953115e8 BLOCKER).
///
/// static 전용 검사가 TriMesh를 빠뜨리면 `try_new_kinematic`이 통과하고, 움직이는 메시가
/// 생긴다 — "메시는 CCD/캐스트의 **대상**이지 주체가 아니다"라는 전제가 깨진다.
/// 라이브 생성과 스냅샷 복원 **양쪽**을 본다: 한쪽만 막으면 `restore(snapshot())`가
/// 자기 스냅샷을 거부하는 비대칭이 생긴다(이 저장소가 네 번 재발로 배운 계열).
#[test]
fn trimesh_cannot_be_kinematic() {
    let mesh = || {
        maru_collide::Shape::trimesh(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2]],
        )
        .expect("정상 메시")
    };
    let r = Body::try_new_kinematic(mesh(), pose(0.0, 0.0, 0.0));
    assert!(r.is_err(), "kinematic 메시가 만들어졌다");
    // 월드 삽입 경계도 같은 정책이어야 한다.
    let mut w = World::new(WorldParams::default());
    let before = w.body_count();
    assert!(
        w.try_add_body(Body::new_static(mesh(), pose(0.0, 0.0, 0.0)))
            .is_ok(),
        "static 메시는 받아들여져야 한다(음성 대조)"
    );
    assert_eq!(w.body_count(), before + 1);
}

/// 평평한 하이트필드 위에 떨어뜨린 구는 **표면에 정착한다** — 지형 접촉의 기본.
#[test]
fn sphere_settles_on_a_flat_heightfield() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 9 * 9];
    w.try_add_heightfield(&heights, 9, 9, 1.0, hf_pose())
        .expect("지형 추가");
    let h = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.5 },
        pose(4.0, 3.0, 4.0),
        1.0,
    ));
    for _ in 0..300 {
        w.step(DT);
    }
    let b = w.body(h);
    assert!(
        (b.pose().pos.y - 0.5).abs() < 0.02,
        "표면(0)+반지름(0.5)에 정착해야 한다: y={}",
        b.pose().pos.y
    );
    assert!(b.linvel().length() < 0.05, "정착 속도 {:?}", b.linvel());
}

/// dynamic 하이트필드는 **만들 수조차 없다** — 질량 경로가 없어 생성이 거부한다.
/// (복원 경계도 같은 정책이다: `validate_structure`가 static이 아닌 하이트필드를 거부.)
#[test]
#[should_panic(expected = "dynamic shape must have finite positive mass")]
fn dynamic_heightfield_is_rejected_at_construction() {
    let _ = Body::new_dynamic(
        Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자"),
        hf_pose(),
        1.0,
    );
}

/// 잘못된 격자는 **오류로** 거부된다(패닉이 아니라).
#[test]
fn malformed_heightfields_are_rejected_as_errors() {
    let mut w = World::new(WorldParams::default());
    let cases: Vec<(&str, Vec<f32>, u32, u32, f32)> = vec![
        ("샘플 수 불일치", vec![0.0; 8], 3, 3, 1.0),
        ("너무 작은 격자", vec![0.0; 1], 1, 1, 1.0),
        ("셀 크기 0", vec![0.0; 4], 2, 2, 0.0),
        ("NaN 높이", vec![f32::NAN; 4], 2, 2, 1.0),
        ("무한 높이", vec![f32::INFINITY; 4], 2, 2, 1.0),
    ];
    for (name, heights, nx, nz, cell) in cases {
        let r = w.try_add_heightfield(&heights, nx, nz, cell, hf_pose());
        assert!(r.is_err(), "{name}가 통과했다");
    }
}

/// **타일 가지치기가 긴 질의를 살린다** — 그리고 진짜로 볼 셀이 많으면 오류다.
///
/// 예산 이력: 잘라서 `None`(관통) → 무제한(폭주) → 가짜 히트(영구 동결) → **타일 계층**.
/// 지금은 질의의 Y 밴드와 겹치는 타일만 보므로, 평지 위를 지나는 긴 스윕은 볼 셀이
/// 거의 없어 **성공**한다. 반대로 모든 셀이 밴드와 겹치는 거친 지형에서는 진짜로
/// 볼 일이 많으므로 `ResourceLimit`으로 거절한다(가짜 히트로 위장하지 않는다).
#[test]
fn tile_pruning_keeps_long_queries_cheap_and_still_reports_real_overload() {
    // ① 평지 + 600m 지점의 벽 — 스윕 밴드(y≈1)는 평지 타일(0..0)과 안 겹친다.
    let mut flat = World::new(WorldParams::default());
    let (nx, nz) = (8193u32, 3u32);
    let mut heights = vec![0.0f32; (nx * nz) as usize];
    for iz in 0..nz {
        heights[(iz * nx + 6000) as usize] = 3.0;
    }
    flat.try_add_heightfield(&heights, nx, nz, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형");
    let hit = flat
        .sweep_shape(
            &Shape::Sphere { radius: 0.2 },
            pose(0.0, 1.0, 0.1),
            Vec3::new(800.0, 0.0, 0.0),
            QueryFilter::ALL,
        )
        .expect("타일 가지치기 덕에 예산에 걸리지 않아야 한다");
    assert!(hit.is_some(), "600m 앞의 벽을 놓쳤다");

    // ② 모든 셀이 스윕 밴드와 겹치는 거친 지형 — 진짜로 볼 일이 많다 → 오류.
    let mut rough = World::new(WorldParams::default());
    let mut rh = vec![0.0f32; (nx * nz) as usize];
    for iz in 0..nz {
        for ix in 0..nx {
            // 밴드(y 0.8..1.2)를 가로지르도록 톱니
            rh[(iz * nx + ix) as usize] = if ix % 2 == 0 { 0.0 } else { 2.0 };
        }
    }
    rough
        .try_add_heightfield(&rh, nx, nz, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형");
    let r = rough.sweep_shape(
        &Shape::Sphere { radius: 0.2 },
        pose(0.0, 1.0, 0.1),
        Vec3::new(800.0, 0.0, 0.0),
        QueryFilter::ALL,
    );
    assert_eq!(
        r.err(),
        Some(ApiError::ResourceLimit),
        "볼 셀이 8천 개인 질의가 조용히 통과했다"
    );

    // ③ 레이는 DDA라 격자 크기에 묶여 있어 그대로 먼 벽을 본다.
    let ray = flat
        .cast_ray(Vec3::new(0.05, 1.0, 0.1), Vec3::new(1.0, 0.0, 0.0), 1000.0)
        .expect("질의");
    assert!(ray.is_some(), "레이가 먼 벽을 놓쳤다");
}

/// 하이트필드는 **라이브 생성에서도** static 전용이고, compound 자식이 될 수 없다
/// (BLOCKER: 복원에서만 막으면 `restore(snapshot())`가 자기 스냅샷을 거부한다).
#[test]
fn heightfield_static_only_is_enforced_at_creation_too() {
    let hf = || Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자");
    // **fallible 생성자로** 본다 — 비-fallible 쪽은 계약 위반에 패닉하는 것이 규약이다.
    assert!(
        Body::try_new_kinematic(hf(), pose(0.0, 0.0, 0.0)).is_err(),
        "kinematic 하이트필드가 통과했다 — 스냅샷 왕복이 깨진다"
    );
    // compound 자식으로도 안 된다(접촉 경로가 자식을 GJK에 그대로 넣는다)
    assert!(Body::try_new_static(
        Shape::Compound {
            children: vec![
                (pose(0.0, 0.0, 0.0), Shape::Sphere { radius: 0.5 }),
                (pose(0.0, 0.0, 0.0), hf()),
            ],
        },
        pose(0.0, 0.0, 0.0),
    )
    .is_err());
}

/// **회전 CCD의 봉투가 자식 궤도를 덮는다** (BLOCKER: 먼 자식이 지형을 관통했다).
#[test]
fn rotating_compound_child_far_from_pivot_still_sees_terrain() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        substeps: 8,
        ..WorldParams::default()
    });
    // 넓은 평지
    let heights = vec![0.0f32; 33 * 33];
    w.try_add_heightfield(&heights, 33, 33, 1.0, pose(0.0, 0.0, 0.0))
        .expect("지형");
    // 피벗에서 멀리 떨어진 자식을 가진 복합 바디 — 회전만으로 지형을 쓸고 지나간다.
    let body = Body::new_dynamic(
        Shape::Compound {
            children: vec![
                (pose(0.0, 0.0, 0.0), Shape::Sphere { radius: 0.2 }),
                (pose(8.0, -0.55, 0.0), Shape::Sphere { radius: 0.2 }),
            ],
        },
        pose(16.0, 0.6, 16.0),
        1.0,
    )
    .with_ccd();
    let h = w.add_body(body);
    w.set_angular_velocity(h, Vec3::new(0.0, 12.0, 0.0))
        .expect("각속도");
    for _ in 0..40 {
        w.step(DT);
    }
    // trap 없이 끝나고, 바디가 지형 아래로 꺼지지 않았다.
    let p = w.body(h).pose().pos;
    assert!(p.y.is_finite() && p.y > -0.5, "지형 아래로 빠졌다: {p:?}");
}

// ===== 확인검수 rev_e50670ad 회귀 =====

/// **위조된 타일은 이제 존재할 수 없다** (BLOCKER: L0·복원 대칭 → 구성으로 봉인).
///
/// 예전에는 `Shape::Heightfield { tiles, .. }`가 공개 필드라 평지에 `[(100,100)]`을 넣어
/// 모든 지형을 가지치기로 건너뛰게 만들 수 있었다. 스냅샷은 타일을 저장하지 않으므로
/// 복원 월드는 **올바른** 타일을 갖는다 — 같은 바이트·같은 해시인데 다음 CCD가 달라졌다
/// (rev_e50670ad). 처음에는 `validate`에서 비트 검증으로 막았지만, 그 검사는 O(n)이라
/// 라이브 질의(스텝 안의 CCD가 후보마다 부른다)에 넣을 수 없었고 그래서 라이브 경로에
/// 구멍이 남았다(rev_97887d75 BLOCKER).
///
/// **지금은 검사 대신 구성으로 막는다.** 데이터는 `HeightfieldGrid`가 소유하고 필드는
/// 비공개이므로, 타일을 넣는 코드는 **컴파일되지 않는다** — 어느 경로에서도 재검증이
/// 필요 없고 런타임 비용은 0이다. 이 테스트는 그 대체 계약을 확인한다: 생성자가 유일한
/// 입구이고, 그 입구가 구조 불변식을 세우며, 나온 타일이 실제 높이와 일치한다.
#[test]
fn heightfield_tiles_are_derived_by_the_constructor_and_cannot_be_supplied() {
    // 봉우리가 있는 격자 — 타일 범위가 평평하지 않아야 대조가 의미 있다.
    let (nx, nz) = (9u32, 9u32);
    let mut heights = vec![0.0f32; (nx * nz) as usize];
    heights[(4 * nx + 4) as usize] = 3.0;
    let s = Shape::heightfield(heights.clone(), nx, nz, 1.0).expect("격자");
    assert!(s.validate().is_ok());
    assert!(Body::try_new_static(s.clone(), pose(0.0, 0.0, 0.0)).is_ok());
    // 생성자가 만든 타일이 높이와 일치한다 — 봉우리를 품은 타일의 최대가 3.0이다.
    let hf = s.heightfield_ref().expect("하이트필드");
    let (lo, hi) = hf.tile_range(0, 0);
    assert_eq!((lo, hi), (0.0, 3.0), "타일 범위가 높이와 다르다");
    // 구조 불변식을 어기는 입력은 생성자가 거절한다(개수 불일치·너무 작은 격자·
    // 비유한 높이·0 셀 크기). 예전에는 이런 값이 셰이프 안까지 들어왔다.
    assert!(Shape::heightfield(vec![0.0; 8], 3, 3, 1.0).is_none());
    assert!(Shape::heightfield(vec![0.0; 1], 1, 1, 1.0).is_none());
    assert!(Shape::heightfield(vec![f32::NAN; 4], 2, 2, 1.0).is_none());
    assert!(Shape::heightfield(vec![0.0; 4], 2, 2, 0.0).is_none());
    // **음성 대조** — 정상 입력은 통과한다(생성자가 전부 거절하는 게 아니다).
    assert!(Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).is_some());
    // 크기 **정책도 생성자가 세운다**(rev_7b22e13d BLOCKER 이후): 정책을 `validate`에만
    // 두면 `is_castable`이 하이트필드를 무조건 통과시키므로 공개 질의가 `Inf` AABB를 낸다.
    assert!(
        Shape::heightfield(vec![0.0; 4], 2, 2, 1.0e9).is_none(),
        "셀 크기가 MAX_SHAPE_EXTENT를 넘는데 생성됐다"
    );
    assert!(
        Shape::heightfield(vec![0.0; 4], 2, 2, f32::MAX).is_none(),
        "cell=f32::MAX 격자가 생성됐다 — try_local_aabb가 Inf를 낸다"
    );
    // 격자 **전체 폭**도 상한 안이어야 한다(셀은 작아도 개수가 크면 넘는다).
    assert!(Shape::heightfield(vec![0.0; 4], 2, 2, 999.0).is_some());
    assert!(
        Shape::heightfield(vec![0.0; 9], 3, 3, 900.0).is_none(),
        "전체 폭 1800m가 MAX_SHAPE_EXTENT를 넘는데 생성됐다"
    );
    // 높이 값도 상한 안이어야 한다.
    assert!(Shape::heightfield(vec![1.0e9; 4], 2, 2, 1.0).is_none());
}

/// **비볼록을 볼록 전용 경로에 넣을 방법이 아예 없다** (BLOCKER: trap → miss 위장 → 차단).
///
/// 진화 과정: `unreachable!` 패닉 → `None` 반환(그러면 "접촉 없음"과 구분이 안 돼
/// 호스트가 "지나갈 수 있다"고 오판한다) → **공개 표면에서 제거**.
/// 지금 밖으로 나가는 것은 셰이프 무관 디스패처뿐이고, 그것들은 모든 셰이프를 옳게 다룬다.
/// 이 테스트는 디스패처가 하이트필드를 **정상 처리**하는 것으로 그 계약을 확인한다
/// (볼록 전용 API를 부르는 코드는 이제 컴파일되지 않는다 — 그게 최선의 방어다).
#[test]
fn dispatchers_handle_heightfields_and_convex_only_apis_are_not_public() {
    let heights = vec![0.0f32; 9 * 9];
    let hf = Shape::heightfield(heights, 9, 9, 1.0).expect("격자");
    let sphere = Shape::Sphere { radius: 0.5 };
    let p = pose(0.0, 0.0, 0.0);
    // 디스패처는 어느 쪽에 하이트필드가 있어도 동작한다.
    let mut hits = 0;
    maru_collide::for_each_contact(
        &sphere,
        &pose(4.0, 0.4, 4.0),
        &hf,
        &p,
        0.02,
        &mut |_, _, m| {
            if !m.is_empty() {
                hits += 1;
            }
        },
    );
    assert!(hits > 0, "지형 위 구가 접촉을 못 만들었다");
    assert_eq!(
        maru_collide::any_contact(&sphere, &pose(4.0, 0.4, 4.0), &hf, &p, 0.02),
        Some(true)
    );
    // 스윕 디스패처도 지형을 펼친다.
    assert!(maru_collide::try_shape_cast_dispatch(
        &sphere,
        &pose(4.0, 3.0, 4.0),
        Vec3::new(0.0, -5.0, 0.0),
        &hf,
        &p,
        0.02
    )
    .expect("정상 입력")
    .is_some());
}

/// **불변식이 깨진 형상은 공개 경로에서 값으로 거절된다** (BLOCKER: 패닉/미스 위장).
///
/// `Shape`의 필드가 공개라 호스트는 `Hull { points: vec![] }`를 만들 수 있고, 예전에는
/// 그것이 GJK로 내려가 `support_core`의 `points[0]`에서 프로세스를 죽였다(wasm에서는
/// unwind가 trap이다). 빈 복합은 반대로 `Ok(None)`이 되어 잘못된 형상이 **실제 미스로
/// 위장**됐다 — 확인검수 rev_2e93ebf2 BLOCKER.
#[test]
fn malformed_shapes_are_rejected_by_value_on_public_paths() {
    use maru_collide::CastReject;
    let empty_hull = Shape::Hull {
        points: Vec::new(),
        radius: 0.0,
    };
    let empty_compound = Shape::Compound {
        children: Vec::new(),
    };
    let ok = Shape::Sphere { radius: 0.5 };
    let pa = pose(0.0, 0.0, 0.0);
    let pb = pose(1.0, 0.0, 0.0);
    let m = Vec3::new(1.0, 0.0, 0.0);
    for bad in [&empty_hull, &empty_compound] {
        // 어느 쪽 인자로 들어와도 같은 답이어야 한다(A/B 대칭).
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(bad, &pa, m, &ok, &pb, 0.02),
            Err(CastReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(&ok, &pa, m, bad, &pb, 0.02),
            Err(CastReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_shape_cast_dispatch_rotational(
                bad,
                &pa,
                m,
                Vec3::new(0.0, 1.0, 0.0),
                0.5,
                &ok,
                &pb,
                0.02
            ),
            Err(CastReject::InvalidInput)
        );
        // 접촉 쪽 형제 API도 같은 규칙 — `false`(없음)로 뭉개지 않는다.
        assert_eq!(maru_collide::any_contact(bad, &pa, &ok, &pb, 0.02), None);
        assert_eq!(
            maru_collide::for_each_contact(bad, &pa, &ok, &pb, 0.02, &mut |_, _, _| {}),
            maru_collide::ContactWalk::InvalidShape
        );
    }
    // **음성 대조** — 점 하나만 넣어 정상 hull로 만들면 곧바로 통과하고 실제로 맞춘다.
    let tiny_hull = Shape::Hull {
        points: vec![Vec3::ZERO],
        radius: 0.4,
    };
    let hit = maru_collide::try_shape_cast_dispatch(&tiny_hull, &pa, m, &ok, &pb, 0.02)
        .expect("정상 hull이 거절됐다");
    assert!(hit.is_some(), "정상 hull이 맞은 것을 못 찾았다");
    assert_eq!(
        maru_collide::any_contact(&tiny_hull, &pose(0.6, 0.0, 0.0), &ok, &pb, 0.02),
        Some(true)
    );
}

/// **포화는 완주가 아니다** — 음성 판정은 순회를 끝까지 걸어야 한다 (MAJOR).
///
/// `shape_penetrates`가 `Saturated`를 `Complete`처럼 다루면, 앞 64개 매니폴드가 침투
/// 임계에 못 미치는 얕은 접촉이었을 때 뒤쪽의 깊은 침투를 못 본 채 "안 겹친다"가 된다 —
/// 캐릭터가 지형 안에서 이동을 시작한다(확인검수 rev_97887d75 MAJOR).
///
/// 넓은 판을 평지에 **살짝 얹으면**(침투 없음) 매니폴드가 64개에서 포화한다. 그때 답은
/// `Ok(false)`가 아니라 "판정 미완료"다.
#[test]
fn shape_penetrates_treats_saturation_as_incomplete_not_negative() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 65 * 65];
    w.try_add_heightfield(&heights, 65, 65, 0.5, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    // 12m×12m 판을 지면 **위에** 얹는다(겹침 없음) — 셀마다 접촉이 생겨 64개에서 포화.
    let slab = Shape::Box {
        half: Vec3::new(6.0, 0.2, 6.0),
        radius: 0.0,
    };
    assert_eq!(
        w.shape_penetrates(&slab, pose(8.0, 0.2, 8.0), QueryFilter::ALL),
        Err(ApiError::ResourceLimit),
        "포화를 확정 음성으로 취급했다 — 뒤쪽 깊은 침투를 못 본 채 '안 겹친다'가 된다"
    );
    // **음성 대조 ①** — 확실히 박히면 양성이 먼저 증명되므로 포화와 무관하게 `true`.
    assert_eq!(
        w.shape_penetrates(&slab, pose(8.0, -0.1, 8.0), QueryFilter::ALL),
        Ok(true)
    );
    // **음성 대조 ②** — 작은 바디는 포화하지 않으므로 확정적인 답을 받는다.
    let small = Shape::Box {
        half: Vec3::new(0.2, 0.2, 0.2),
        radius: 0.0,
    };
    assert_eq!(
        w.shape_penetrates(&small, pose(8.0, 2.0, 8.0), QueryFilter::ALL),
        Ok(false)
    );
    assert_eq!(
        w.shape_penetrates(&small, pose(8.0, -0.1, 8.0), QueryFilter::ALL),
        Ok(true)
    );
}

/// **자식 자세 정책이 생성 경계와 질의 경계에서 같다** (BLOCKER: 같은 바디, API마다 다른 답).
///
/// 길이 제곱 1.0404(≈ w 1.02)는 옛 `0.9..=1.1`을 통과해 월드에 들어갔지만 질의 쪽
/// `is_castable`(1e-3)에서는 거절됐다. 그러면 같은 바디가 레이 질의에서는 미스, 점 질의에서는
/// `InvalidValue`, 물리 스텝에서는 `ResourceLimit`이 된다 — 게다가 동적 복합은
/// `bake_compound`가 그 비단위 회전으로 관성 텐서를 계산한다
/// (확인검수 rev_c73f897c BLOCKER — 내가 질의 쪽만 조이고 이 자리를 빠뜨렸다).
#[test]
fn compound_child_rotation_policy_is_the_same_at_both_boundaries() {
    let child = Shape::Box {
        half: Vec3::new(0.3, 0.3, 0.3),
        radius: 0.0,
    };
    let scaled = |k: f32| maru_collide::Pose {
        pos: Vec3::new(0.5, 0.0, 0.0),
        rot: Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: k,
        },
    };
    let make = |k: f32| Shape::Compound {
        children: vec![(scaled(k), child.clone())],
    };
    // w=1.02 → 길이 제곱 1.0404: 옛 기준은 통과, 새 기준은 거절.
    let bad = make(1.02);
    assert!(
        bad.validate().is_err(),
        "생성 경계가 비단위 자식을 통과시켰다 — 질의 경계와 정책이 갈린다"
    );
    assert!(!bad.is_castable_pub_for_test());
    assert!(Body::try_new_static(bad.clone(), pose(0.0, 0.0, 0.0)).is_err());
    assert!(Body::try_new_dynamic(bad, pose(0.0, 0.0, 0.0), 1.0).is_err());
    // **음성 대조** — 드리프트 수준(1e-4)은 양쪽 다 통과한다(정상 입력을 막지 않는다).
    let drift = make(1.0 + 5.0e-5);
    assert!(drift.validate().is_ok(), "정상 드리프트가 거절됐다");
    assert!(drift.is_castable_pub_for_test());
    assert!(Body::try_new_static(drift, pose(0.0, 0.0, 0.0)).is_ok());
    // 자식 수 상한도 두 경계에서 같다 — 콜백 키가 u8이라 래핑하면 자식이 서로 섞인다.
    let many = Shape::Compound {
        children: (0..9)
            .map(|i| {
                (
                    pose(i as f32 * 0.8, 0.0, 0.0),
                    Shape::Sphere { radius: 0.2 },
                )
            })
            .collect(),
    };
    assert!(many.validate().is_err());
    assert!(!many.is_castable_pub_for_test());
}

/// **존재 질의의 답이 삽입 순서에 의존하지 않는다** (MAJOR: 같은 장면, 다른 답).
///
/// 판정 미완료(포화·예산)에서 즉시 오류로 나가면, 포화하는 지형이 앞 슬롯일 때 뒤에 깊게
/// 박힌 바디를 못 본다 — 같은 형상 집합이 삭제·재삽입 순서에 따라 "침투함"과 "판정 불가"로
/// 갈린다(확인검수 rev_c73f897c MAJOR). 미완료는 기억만 하고, 확정 양성이 있으면 그것이 이긴다.
#[test]
fn shape_penetrates_answer_does_not_depend_on_slot_order() {
    let probe = Shape::Box {
        half: Vec3::new(6.0, 0.2, 6.0),
        radius: 0.0,
    };
    let at = pose(8.0, 0.2, 8.0);
    // 지형만 있으면 포화로 "판정 미완료"다(직전 라운드 테스트가 고정한 계약).
    let mut only_hf = World::new(WorldParams::default());
    only_hf
        .try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
        .expect("지형");
    assert_eq!(
        only_hf.shape_penetrates(&probe, at, QueryFilter::ALL),
        Err(ApiError::ResourceLimit)
    );
    // 같은 장면 + 프로브에 **깊게 박힌 구**. 두 삽입 순서 모두 `Ok(true)`여야 한다.
    for hf_first in [true, false] {
        let mut w = World::new(WorldParams::default());
        let add_hf = |w: &mut World| {
            w.try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
                .expect("지형");
        };
        let add_ball = |w: &mut World| {
            w.add_body(Body::new_static(
                Shape::Sphere { radius: 0.5 },
                pose(8.0, 0.2, 8.0),
            ));
        };
        if hf_first {
            add_hf(&mut w);
            add_ball(&mut w);
        } else {
            add_ball(&mut w);
            add_hf(&mut w);
        }
        assert_eq!(
            w.shape_penetrates(&probe, at, QueryFilter::ALL),
            Ok(true),
            "삽입 순서(지형 먼저={hf_first})가 답을 바꿨다"
        );
    }
}

/// PARTIAL MAJOR: hull 재중심화로 엔진 pose ≠ 호스트가 준 원점이 된다 →
/// 렌더 오프셋을 조회할 수 있어야 시각 형상과 물리 형상이 어긋나지 않는다.
#[test]
fn hull_recenter_exposes_render_offset() {
    let origin = pose(1.0, 2.0, 3.0);
    let b = Body::new_dynamic(
        Shape::Hull {
            points: vec![
                Vec3::new(9.5, -0.5, -0.5),
                Vec3::new(10.5, -0.5, -0.5),
                Vec3::new(9.5, 0.5, -0.5),
                Vec3::new(9.5, -0.5, 0.5),
                Vec3::new(10.5, 0.5, 0.5),
            ],
            radius: 0.0,
        },
        origin,
        1.0,
    );
    // 질량중심은 원래 원점에서 (10,0,0) 떨어져 있다
    assert!((b.pose().pos - Vec3::new(11.0, 2.0, 3.0)).length() < 1e-3);
    // 호스트가 원래 정점 버퍼를 그릴 위치 = pose * offset → 원래 원점으로 돌아온다
    let render = b.pose().pos + b.pose().rot.rotate(b.shape_origin_offset());
    assert!(
        (render - origin.pos).length() < 1e-3,
        "렌더 오프셋이 원래 원점을 복원하지 못함: {render:?}"
    );
}

#[test]
#[should_panic(expected = "invalid shape")]
fn oversized_hull_rejected() {
    let points: Vec<Vec3> = (0..(maru_collide::MAX_HULL_POINTS + 1))
        .map(|i| Vec3::new(i as f32 * 1e-3, 0.0, 0.0))
        .collect();
    Body::new_dynamic(
        Shape::Hull {
            points,
            radius: 0.0,
        },
        pose(0.0, 0.0, 0.0),
        1.0,
    );
}

/// MAJOR: 최소 특징 크기는 **모든 셰이프**에 적용된다 (capsule·hull이 빠져 있었다).
#[test]
fn min_feature_size_applies_to_every_shape() {
    let tiny = maru_collide::MIN_SHAPE_EXTENT * 0.1;
    assert!(Body::try_new_static(
        Shape::Capsule {
            half_height: 1.0,
            radius: tiny
        },
        pose(0.0, 0.0, 0.0)
    )
    .is_err());
    // 반지름 0의 단일점 hull = 모든 축에서 퇴화
    assert!(Body::try_new_static(
        Shape::Hull {
            points: vec![Vec3::ZERO],
            radius: 0.0
        },
        pose(0.0, 0.0, 0.0)
    )
    .is_err());
    // 한 축만 얇은 hull도 거부 (월드 경계에서 AABB가 반올림으로 퇴화)
    assert!(Body::try_new_static(
        Shape::Hull {
            points: vec![Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0)],
            radius: 0.0
        },
        pose(0.0, 0.0, 0.0)
    )
    .is_err());
    // 두께가 반지름으로 채워지면 합법
    assert!(Body::try_new_static(
        Shape::Hull {
            points: vec![Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0)],
            radius: maru_collide::MIN_SHAPE_EXTENT
        },
        pose(0.0, 0.0, 0.0)
    )
    .is_ok());
}

/// MAJOR: 축 정렬 span만 보면 **회전된 퇴화 hull**이 통과한다 — 한 평면 위의 점들도
/// x/y/z span은 모두 크다. 회전 불변 두께를 재야 한다.
#[test]
fn rotated_coplanar_hull_is_rejected() {
    // z = x + y 평면 위의 점들 — 3D 두께 0이지만 축별 span은 모두 1
    let coplanar = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 2.0),
    ];
    assert!(
        Body::try_new_static(
            Shape::Hull {
                points: coplanar.clone(),
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_err(),
        "회전된 평면 hull이 통과 (축 정렬 검사만 하고 있다)"
    );
    // 두께가 생기면 합법
    let mut thick = coplanar;
    thick.push(Vec3::new(0.5, 0.5, 0.0));
    assert!(Body::try_new_static(
        Shape::Hull {
            points: thick,
            radius: 0.0
        },
        pose(0.0, 0.0, 0.0)
    )
    .is_ok());
}

/// MAJOR: 최소 폭 스크린은 **한 평면의 높이**가 아니라 여러 면 방향을 봐야 한다.
/// 리뷰가 제시한 sliver(현 계산 1.1mm, 다른 면 기준 0.575mm)를 거부해야 한다.
#[test]
fn sliver_hull_is_rejected_by_multi_face_screen() {
    let pts = vec![
        Vec3::new(-0.684779, 0.439397, 0.415335),
        Vec3::new(-2.926032, 1.807456, 1.821069),
        Vec3::new(-0.745893, 0.486453, 0.446511),
        Vec3::new(-0.655741, 0.426129, 0.392909),
    ];
    assert!(
        Body::try_new_static(
            Shape::Hull {
                points: pts,
                radius: 0.0
            },
            pose(0.0, 0.0, 0.0)
        )
        .is_err(),
        "sliver hull이 통과 (단일 평면 높이만 재고 있다)"
    );
}

/// MAJOR: hull 지원 상한은 **정확히 검증 가능한 범위**여야 한다.
#[test]
fn hull_support_limit_matches_exact_verification() {
    assert_eq!(maru_collide::MAX_HULL_POINTS, 32);
    let too_many: Vec<Vec3> = (0..33)
        .map(|i| {
            let f = i as f32 * 0.1;
            Vec3::new(f.sin(), f.cos(), (f * 0.7).sin())
        })
        .collect();
    assert!(Body::try_new_static(
        Shape::Hull {
            points: too_many,
            radius: 0.0
        },
        pose(0.0, 0.0, 0.0)
    )
    .is_err());
}

/// 복합 질량 베이킹 해석해 대조: 나란한 두 박스 == 한 개의 넓은 박스.
/// COM 재중심화(pose.pos = 질량중심)와 주축 관성까지 단일 박스 해석해와 일치해야 한다.
#[test]
fn compound_mass_baking_matches_analytic_wide_box() {
    let two = Body::new_dynamic(
        Shape::Compound {
            children: vec![
                (
                    Pose {
                        pos: Vec3::new(-0.5, 0.0, 0.0),
                        rot: Quat::IDENTITY,
                    },
                    Shape::Box {
                        half: Vec3::new(0.5, 0.5, 0.5),
                        radius: 0.0,
                    },
                ),
                (
                    Pose {
                        pos: Vec3::new(0.5, 0.0, 0.0),
                        rot: Quat::IDENTITY,
                    },
                    Shape::Box {
                        half: Vec3::new(0.5, 0.5, 0.5),
                        radius: 0.0,
                    },
                ),
            ],
        },
        pose(3.0, 5.0, 0.0),
        1.0,
    );
    let one = Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(1.0, 0.5, 0.5),
            radius: 0.0,
        },
        pose(3.0, 5.0, 0.0),
        1.0,
    );
    assert!(
        (two.inv_mass() - one.inv_mass()).abs() < 1e-5 * one.inv_mass(),
        "질량 불일치: {} vs {}",
        two.inv_mass(),
        one.inv_mass()
    );
    // 주축 관성 비교 — 고유값 순서가 다를 수 있어 정렬 후 대조.
    let sorted = |v: Vec3| {
        let mut a = [v.x, v.y, v.z];
        a.sort_by(f32::total_cmp);
        a
    };
    let (ti, oi) = (
        sorted(two.inv_inertia_local()),
        sorted(one.inv_inertia_local()),
    );
    for k in 0..3 {
        assert!(
            (ti[k] - oi[k]).abs() < 1e-4 * oi[k].abs().max(1e-6),
            "주축 관성 불일치: {ti:?} vs {oi:?}"
        );
    }
    // COM 재중심화: 대칭 배치라 COM == 원 pose 위치.
    assert!((two.pose().pos - Vec3::new(3.0, 5.0, 0.0)).length() < 1e-5);
}

/// 복합×복합 접촉: 두 L자가 서로 얹혀도 발산 없이 정착한다.
#[test]
fn compound_on_compound_stacks() {
    let l_shape = || Shape::Compound {
        children: vec![
            (
                Pose {
                    pos: Vec3::ZERO,
                    rot: Quat::IDENTITY,
                },
                Shape::Box {
                    half: Vec3::new(0.8, 0.25, 0.4),
                    radius: 0.0,
                },
            ),
            (
                Pose {
                    pos: Vec3::new(-0.55, 0.5, 0.0),
                    rot: Quat::IDENTITY,
                },
                Shape::Box {
                    half: Vec3::new(0.25, 0.25, 0.4),
                    radius: 0.0,
                },
            ),
        ],
    };
    let mut w = World::new(WorldParams::default());
    ground(&mut w);
    let a = w.add_body(Body::new_dynamic(l_shape(), pose(0.0, 0.4, 0.0), 1.0));
    let b = w.add_body(Body::new_dynamic(l_shape(), pose(0.15, 1.2, 0.0), 1.0));
    for _ in 0..600 {
        w.step(DT);
    }
    for h in [a, b] {
        let p = w.body(h).pose().pos;
        assert!(
            p.y.is_finite() && p.y > -0.5 && p.y < 3.0,
            "발산/관통: {p:?}"
        );
        assert!(
            w.body(h).linvel().length() < 0.5,
            "600스텝 후에도 큰 잔류 속도: {:?}",
            w.body(h).linvel()
        );
    }
}

/// 회귀(rev_3398738c MAJOR ①②): 내부 에지 판정은 각도 기반·대칭이고, 같은 쪽
/// 중복면은 "2면 공유"라도 용접하지 않는다.
#[test]
fn welding_rejects_skinny_ridges_and_same_side_duplicates() {
    let mut w = World::new(WorldParams::default());
    let mask = |h: BodyHandle, w: &World| w.body(h).weld_edges;

    // 같은 삼각형 두 번(같은 쪽 겹침) — 모든 에지가 정확히 2면 공유지만 반대쪽
    // 검증이 걸러야 한다. 예전 판정(공면만)은 세 에지를 전부 용접했다.
    let v = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let hs = w
        .try_add_terrain(&v, &[[0, 1, 2], [0, 1, 2]], 0.02)
        .unwrap();
    assert_eq!(mask(hs[0], &w), 0, "같은 쪽 중복면이 용접됐다");
    assert_eq!(mask(hs[1], &w), 0, "같은 쪽 중복면이 용접됐다");

    // 스키니 삼각형 + 실각 45° 능선: 예전 판정은 t1의 최대 변(~707m)으로 임계를
    // 부풀려(7e-4) 0.6mm 능선을 공면으로 봤다. 각도 판정은 스케일 불변이라 거른다.
    let v = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.02, 0.0, 0.0),        // 공유 에지 2cm
        Vec3::new(500.0, 0.0, 500.0),     // t1의 먼 정점 (스케일 ~707m)
        Vec3::new(0.01, 6.0e-4, -6.0e-4), // t2 반대 정점 — 평면 위 0.6mm (실각 45°)
    ];
    let hs = w
        .try_add_terrain(&v, &[[0, 1, 2], [1, 0, 3]], 0.001)
        .unwrap();
    assert_eq!(mask(hs[0], &w), 0, "스키니 능선이 용접됐다");
    assert_eq!(mask(hs[1], &w), 0, "스키니 능선이 용접됐다");

    // 대조: 같은 스키니 t1이라도 진짜 공면 이웃(반대쪽·각 0°)은 용접된다.
    let v = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.02, 0.0, 0.0),
        Vec3::new(500.0, 0.0, 500.0),
        Vec3::new(0.01, 0.0, -0.01),
    ];
    let hs = w
        .try_add_terrain(&v, &[[0, 1, 2], [1, 0, 3]], 0.001)
        .unwrap();
    assert_ne!(mask(hs[0], &w), 0, "공면 이웃이 용접되지 않았다");
    assert_ne!(mask(hs[1], &w), 0, "공면 이웃이 용접되지 않았다");
}

/// 회귀(rev_e4075e85): 퇴화 삼각형(동일점·일직선)은 지형에서 전체 거부된다.
#[test]
fn terrain_rejects_degenerate_triangles() {
    let mut w = World::new(WorldParams::default());
    let v = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0), // 0-1-2 일직선
        Vec3::new(0.0, 0.0, 1.0),
    ];
    assert!(w.try_add_terrain(&v, &[[0, 0, 0]], 0.02).is_err(), "동일점");
    assert!(w.try_add_terrain(&v, &[[0, 1, 2]], 0.02).is_err(), "일직선");
    assert!(
        w.try_add_terrain(&v, &[[0, 1, 3]], 0.02).is_ok(),
        "정상 거부 오탐"
    );
}
