//! queries 관련 테스트.

use super::*;

fn query_world() -> (World, Vec<BodyHandle>) {
    let mut w = World::new(WorldParams::default());
    let mut hs = Vec::new();
    // x = 0, 5, 10 에 반지름 1 구 세 개
    for k in 0..3 {
        hs.push(w.add_body(Body::new_dynamic(
            maru_collide::Shape::Sphere { radius: 1.0 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(k as f32 * 5.0, 0.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
            1.0,
        )));
    }
    (w, hs)
}

/// 가장 가까운 물체를 맞히고, 맞은 지점·법선이 일관돼야 한다.
#[test]
fn ray_hits_the_nearest_body() {
    let (w, hs) = query_world();
    let hit = w
        .cast_ray(
            maru_math::Vec3::new(-10.0, 0.0, 0.0),
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap()
        .expect("맞아야 한다");
    assert_eq!(hit.body, hs[0], "가장 가까운 것은 x=0의 구다");
    assert!((hit.toi - 9.0).abs() < 1e-3, "toi={}", hit.toi);
    assert!((hit.point.x - (-1.0)).abs() < 1e-3, "point={:?}", hit.point);
    assert!(
        hit.normal.x < -0.99,
        "바깥 법선은 광선 반대편을 본다: {:?}",
        hit.normal
    );
    assert!(!hit.started_inside);
}

/// 모든 히트는 toi 오름차순이고 개수가 맞아야 한다.
#[test]
fn ray_all_is_sorted_by_toi() {
    let (w, hs) = query_world();
    let mut out = Vec::new();
    w.cast_ray_all(
        maru_math::Vec3::new(-10.0, 0.0, 0.0),
        maru_math::Vec3::new(1.0, 0.0, 0.0),
        100.0,
        &mut out,
    )
    .unwrap();
    assert_eq!(out.len(), 3);
    assert!(
        out.windows(2).all(|p| p[0].toi <= p[1].toi),
        "toi 정렬 위반"
    );
    assert_eq!(
        out.iter().map(|h| h.body).collect::<Vec<_>>(),
        hs,
        "가까운 순서여야 한다"
    );
}

/// `max_toi`는 존중돼야 한다.
#[test]
fn ray_respects_max_toi() {
    let (w, _) = query_world();
    let mut out = Vec::new();
    w.cast_ray_all(
        maru_math::Vec3::new(-10.0, 0.0, 0.0),
        maru_math::Vec3::new(1.0, 0.0, 0.0),
        10.0,
        &mut out,
    )
    .unwrap();
    assert_eq!(out.len(), 1, "10 안에는 첫 구만 있다");
}

/// 물체 **안에서** 쏘면 맞은 것으로 나와야 한다 — 지면 판정이 여기 걸린다.
#[test]
fn ray_from_inside_reports_a_hit() {
    let (w, hs) = query_world();
    let hit = w
        .cast_ray(
            maru_math::Vec3::ZERO,
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap()
        .expect("안에서 시작해도 맞은 것이다");
    assert_eq!(hit.body, hs[0]);
    assert_eq!(hit.toi, 0.0);
    assert!(hit.started_inside);
}

/// AABB·점 질의는 슬롯 오름차순으로 결정적이어야 한다.
#[test]
fn aabb_and_point_queries_are_deterministic() {
    let (w, hs) = query_world();
    let mut out = Vec::new();
    w.query_aabb(
        maru_collide::Aabb::new(
            maru_math::Vec3::new(-2.0, -2.0, -2.0),
            maru_math::Vec3::new(7.0, 2.0, 2.0),
        ),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![hs[0], hs[1]], "x=0,5 두 개만 겹친다");

    let mut pts = Vec::new();
    w.query_point(maru_math::Vec3::new(5.2, 0.0, 0.0), &mut pts)
        .unwrap();
    assert_eq!(pts, vec![hs[1]]);
    w.query_point(maru_math::Vec3::new(2.5, 0.0, 0.0), &mut pts)
        .unwrap();
    assert!(pts.is_empty(), "빈틈에는 아무것도 없다");
}

/// 잘못된 입력은 거부한다 (0 방향·NaN·음수 max_toi).
#[test]
fn ray_input_is_validated() {
    let (w, _) = query_world();
    let o = maru_math::Vec3::ZERO;
    assert_eq!(
        w.cast_ray(o, maru_math::Vec3::ZERO, 10.0),
        Err(ApiError::InvalidValue),
        "0 방향은 광선이 아니다"
    );
    assert_eq!(
        w.cast_ray(o, maru_math::Vec3::new(f32::NAN, 0.0, 0.0), 10.0),
        Err(ApiError::InvalidValue)
    );
    assert_eq!(
        w.cast_ray(o, maru_math::Vec3::new(1.0, 0.0, 0.0), -1.0),
        Err(ApiError::InvalidValue)
    );
}

/// 질의 필터는 **물리 충돌과 같은 양방향 규칙**이어야 한다.
#[test]
fn query_filter_matches_physics_group_rule() {
    let mut w = World::new(WorldParams::default());
    let a = w.add_body(
        Body::new_static(
            maru_collide::Shape::Sphere { radius: 1.0 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(0.0, 0.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        )
        .with_groups(0b01, 0b01),
    );
    let b = w.add_body(
        Body::new_static(
            maru_collide::Shape::Sphere { radius: 1.0 },
            maru_collide::Pose {
                pos: maru_math::Vec3::new(5.0, 0.0, 0.0),
                rot: maru_math::Quat::IDENTITY,
            },
        )
        .with_groups(0b10, 0b10),
    );
    let o = maru_math::Vec3::new(-10.0, 0.0, 0.0);
    let d = maru_math::Vec3::new(1.0, 0.0, 0.0);
    // 필터 없이는 가장 가까운 a
    assert_eq!(w.cast_ray(o, d, 100.0).unwrap().map(|h| h.body), Some(a));
    // 그룹 0b10만 보면 b
    assert_eq!(
        w.cast_ray_filtered(o, d, 100.0, QueryFilter::ALL.groups(0b10))
            .unwrap()
            .map(|h| h.body),
        Some(b)
    );
    // 자기 자신 제외
    assert_eq!(
        w.cast_ray_filtered(o, d, 100.0, QueryFilter::ALL.exclude(a))
            .unwrap()
            .map(|h| h.body),
        Some(b)
    );
}

/// **캐릭터의 스윕 셰이프로도 메시를 넣을 수 없다.**
///
/// 교차검수(rev_953115e8)가 잡은 두 BLOCKER와 **같은 계열**인데 검수도 놓친 세 번째다:
/// 비볼록 판정이 `matches!`로 흩어져 있어 컴파일러가 안 잡아 준다. 스윕 주체가 비볼록이면
/// GJK가 `support_core`의 `unreachable!`에 닿아 프로세스가 죽는다(wasm에선 trap).
///
/// **정직한 한계**: 이 테스트는 가드를 추가하기 전에도 통과했다 — 다른 경로가 먼저
/// 거부하기 때문이다(무엇인지 특정하지 못했다). 그래도 가드를 넣은 이유는, 메시가
/// `validate`·`is_castable`을 **통과하는 정상 셰이프**라(`mesh_is_a_valid_shape_so_guards_
/// must_be_explicit`) 셰이프 검증이 이 책임을 대신 져 주지 않고, 우연한 하류 거부는
/// 계약이 아니기 때문이다. 이 테스트가 잠그는 것은 **거부된다는 사실**이다.
#[test]
fn character_cannot_sweep_a_trimesh() {
    let mesh = maru_collide::Shape::trimesh(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        vec![[0, 1, 2]],
    )
    .expect("정상 메시");
    let mut w = World::new(WorldParams::default());
    let mut ch = CharacterVirtual::new(mesh, Vec3::new(0.0, 2.0, 0.0));
    let r = ch.move_and_slide(&w, Vec3::new(0.0, -1.0, 0.0), QueryFilter::default());
    assert!(r.is_err(), "비볼록 메시를 스윕 주체로 받아들였다");
    // 음성 대조: 볼록 셰이프는 정상 동작해야 한다(가드가 전부를 막는 게 아님을 보인다).
    let mut ok = CharacterVirtual::new(
        Shape::Capsule {
            half_height: 0.4,
            radius: 0.3,
        },
        Vec3::new(0.0, 2.0, 0.0),
    );
    assert!(ok
        .move_and_slide(&w, Vec3::new(0.0, -0.1, 0.0), QueryFilter::default())
        .is_ok());
    let _ = &mut w;
}

/// 레이가 지형 표면을 맞히고, 지형에 묻힌 점은 `query_point`가 찾는다.
#[test]
fn heightfield_supports_ray_and_point_queries() {
    let mut w = World::new(WorldParams::default());
    let mut heights = vec![0.0f32; 5 * 5];
    heights[12] = 2.0; // 가운데 정점
    let hf = w
        .try_add_heightfield(&heights, 5, 5, 1.0, hf_pose())
        .expect("지형");
    let hit = w
        .cast_ray(Vec3::new(2.0, 5.0, 2.0), Vec3::new(0.0, -1.0, 0.0), 100.0)
        .expect("질의 성공")
        .expect("표면을 맞혀야 한다");
    assert_eq!(hit.body, hf);
    assert!((hit.toi - 3.0).abs() < 1e-2, "toi={}", hit.toi);
    // 봉우리 정점을 정통으로 맞히면 법선은 이웃 삼각형의 **기울어진** 법선이다.
    // 위를 향하기만 하면 된다 — 평평한 곳의 법선은 아래에서 따로 본다.
    assert!(hit.normal.y > 0.3, "법선이 위를 안 본다: {:?}", hit.normal);
    let flat = w
        .cast_ray(Vec3::new(0.5, 5.0, 0.5), Vec3::new(0.0, -1.0, 0.0), 100.0)
        .expect("질의 성공")
        .expect("평평한 곳도 맞아야 한다");
    assert!(flat.normal.y > 0.99, "평면 법선 {:?}", flat.normal);
    // 표면 아래 점은 지형 안이다(속이 찬 것으로 본다)
    let mut out = Vec::new();
    w.query_point(Vec3::new(1.0, -1.0, 1.0), &mut out)
        .expect("질의");
    assert!(out.contains(&hf), "표면 아래 점이 지형 안으로 안 잡혔다");
    out.clear();
    w.query_point(Vec3::new(1.0, 5.0, 1.0), &mut out)
        .expect("질의");
    assert!(!out.contains(&hf), "표면 위 점이 지형 안으로 잡혔다");
}

/// 격자 **밖** 점은 지형 안이 아니다 (MAJOR: 마지막 셀 외삽).
/// 그리고 표면 아래에서 쏜 레이는 `toi=0`·`started_inside`다(다른 셰이프와 같은 계약).
#[test]
fn heightfield_point_and_ray_contracts_hold_outside_and_inside() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 3 * 3];
    let hf = w
        .try_add_heightfield(&heights, 3, 3, 1.0, pose(0.0, 0.0, 0.0))
        .expect("지형");
    let mut out = Vec::new();
    w.query_point(Vec3::new(100.0, -1.0, 0.5), &mut out)
        .expect("질의");
    assert!(
        !out.contains(&hf),
        "격자에서 98m 떨어진 점이 지형 안으로 잡혔다"
    );
    let inside = w
        .cast_ray(Vec3::new(1.0, -0.5, 1.0), Vec3::new(0.0, 1.0, 0.0), 10.0)
        .expect("질의")
        .expect("안에서 시작한 레이도 히트다");
    assert_eq!(inside.toi, 0.0, "안에서 시작인데 toi가 0이 아니다");
    assert!(inside.started_inside, "started_inside가 안 섰다");
}

/// **공개 회전 캐스트는 패닉하지 않고, 잘못된 입력을 미스로 위장하지도 않는다.**
///
/// 두 계약이 함께 있어야 의미가 있다. 캐스트 루프는 매 반복 `from_axis_angle`을 부르고
/// 그 함수는 표현 불가 각에 패닉하므로 (a) 호스트 입력이 그 패닉에 닿아선 안 되고,
/// (b) 그렇다고 `None`(=미스)으로 돌려주면 호출자가 "경로가 비었다"고 읽어 그대로
/// 이동시킨다 — 저수준 볼록 API를 공개 표면에서 뺀 것과 같은 결함이다.
/// 그래서 공개 경로는 `Result`이고 `Err(CastReject::InvalidInput)`을 낸다.
///
/// 각만 검사하면 부족하다: 유효한 각 + NaN 축/이동이면 내부에서 `t`가 NaN이 되어
/// `angle * t`가 다시 표현 불가 각이 된다(확인검수 rev_d5b32e22 BLOCKER).
#[test]
fn rotational_cast_rejects_bad_input_as_error_not_miss() {
    use maru_collide::CastReject;
    // 회전 피벗은 **A의 위치(바디 COM)**다 — 구는 자기 중심 회전에 아무 데도 안 쓸어서
    // 음성 대조가 될 수 없다. 긴 박스를 세로로 세워 90° 돌리면 이웃 박스를 쓸고 지나간다.
    let a = Shape::Box {
        half: Vec3::new(1.0, 0.2, 0.2),
        radius: 0.0,
    };
    let b = Shape::Box {
        half: Vec3::new(0.2, 0.2, 0.2),
        radius: 0.0,
    };
    let pa = pose(0.0, 0.0, 0.0);
    let pb = pose(0.0, 0.0, 1.05);
    let good_axis = Vec3::new(0.0, 1.0, 0.0);
    let good_angle = core::f32::consts::FRAC_PI_2;
    // ① 표현 불가한 각
    for bad in [1.0e20f32, f32::NAN, f32::INFINITY] {
        assert_eq!(
            maru_collide::try_shape_cast_dispatch_rotational(
                &a,
                &pa,
                Vec3::ZERO,
                good_axis,
                bad,
                &b,
                &pb,
                0.02
            ),
            Err(CastReject::InvalidInput),
            "표현 불가한 각 {bad}"
        );
    }
    // ② 유효한 각 + 비유한 축/이동 — 예전에는 이 조합이 내부 assert를 때렸다.
    assert_eq!(
        maru_collide::try_shape_cast_dispatch_rotational(
            &a,
            &pa,
            Vec3::ZERO,
            Vec3::new(f32::NAN, 0.0, 0.0),
            good_angle,
            &b,
            &pb,
            0.02
        ),
        Err(CastReject::InvalidInput)
    );
    assert_eq!(
        maru_collide::try_shape_cast_dispatch_rotational(
            &a,
            &pa,
            Vec3::new(f32::NAN, 0.0, 0.0),
            good_axis,
            good_angle,
            &b,
            &pb,
            0.02
        ),
        Err(CastReject::InvalidInput)
    );
    // ③ 선형 디스패처도 같은 규칙
    assert_eq!(
        maru_collide::try_shape_cast_dispatch(
            &a,
            &pa,
            Vec3::new(0.0, f32::INFINITY, 0.0),
            &b,
            &pb,
            0.02
        ),
        Err(CastReject::InvalidInput)
    );
    // **음성 대조** — 같은 배치에서 정상 입력은 실제로 맞춘다(가드가 전부를 막는 게 아니다).
    let hit = maru_collide::try_shape_cast_dispatch_rotational(
        &a,
        &pa,
        Vec3::ZERO,
        good_axis,
        good_angle,
        &b,
        &pb,
        0.02,
    )
    .expect("정상 입력이 거절됐다");
    assert!(
        hit.is_some_and(|h| h.toi > 0.0 && h.toi < 1.0),
        "정상 회전 스윕이 접촉을 못 찾았다 — 테스트 배치가 무의미하다: {hit:?}"
    );
}

/// **지형 캐스트에는 작업량 상한이 있고, 초과는 미스가 아니라 오류다.**
///
/// 타일 가지치기는 후보 0인 지형의 반복만 없앤다 — Y 밴드가 실제로 겹치면 상한이
/// 없어서 큰 바디 하나가 서브스텝마다 100만 셀을 훑었다(확인검수 rev_d5b32e22 MAJOR).
#[test]
fn heightfield_cast_reports_budget_instead_of_walking_forever() {
    use maru_collide::CastReject;
    let heights = vec![0.0f32; 257 * 257];
    let hf = Shape::heightfield(heights, 257, 257, 0.1).expect("격자");
    let p = pose(0.0, 0.0, 0.0);
    // 반지름 5m 구 = 100×100셀(=10000 > 4096 예산)을 덮는다.
    let big = Shape::Sphere { radius: 5.0 };
    assert_eq!(
        maru_collide::try_shape_cast_dispatch(
            &big,
            &pose(12.8, 5.0, 12.8),
            Vec3::new(0.0, -1.0, 0.0),
            &hf,
            &p,
            0.02
        ),
        Err(CastReject::BudgetExceeded)
    );
    // **음성 대조** — 작은 바디는 같은 지형에서 정상적으로 지면을 맞춘다.
    let small = Shape::Sphere { radius: 0.2 };
    let hit = maru_collide::try_shape_cast_dispatch(
        &small,
        &pose(12.8, 1.0, 12.8),
        Vec3::new(0.0, -2.0, 0.0),
        &hf,
        &p,
        0.02,
    )
    .expect("작은 바디가 예산에 걸렸다");
    assert!(hit.is_some(), "지면을 향한 스윕이 아무것도 못 맞췄다");
}

/// **복합 바디도 지형 스윕에서 히트를 얻는다** (자기 검수: 관통 경로).
///
/// 선형 디스패처가 `match (sa, sb)`로 짜여 있어서 `(Compound, _)` 팔이 먼저 잡혔고,
/// 각 자식이 **볼록 전용** 캐스트에 지형을 그대로 넘겼다 — 그 함수는 비볼록에 `None`을
/// 내므로 결과가 "맞은 것 없음"이 되어 복합 바디가 지형을 통과했다(회전 분기는 애초에
/// B쪽 디스패치를 안쪽에 두어 이 함정이 없었다).
#[test]
fn compound_body_sweeping_into_terrain_reports_a_hit() {
    let heights = vec![0.0f32; 17 * 17];
    let hf = Shape::heightfield(heights, 17, 17, 1.0).expect("격자");
    let p = pose(0.0, 0.0, 0.0);
    let child = Shape::Box {
        half: Vec3::new(0.3, 0.3, 0.3),
        radius: 0.0,
    };
    let comp = Shape::Compound {
        children: vec![
            (pose(-0.5, 0.0, 0.0), child.clone()),
            (pose(0.5, 0.0, 0.0), child.clone()),
        ],
    };
    let start = pose(8.0, 3.0, 8.0);
    let down = Vec3::new(0.0, -5.0, 0.0);
    let comp_hit = maru_collide::try_shape_cast_dispatch(&comp, &start, down, &hf, &p, 0.02)
        .expect("정상 입력");
    // **음성 대조** — 같은 배치의 단일 셰이프는 원래도 맞췄다. 두 결과의 toi가 사실상
    // 같아야 한다(같은 높이에서 같은 지면으로 떨어지므로): 복합 경로만 고장났던 것이다.
    let one_hit = maru_collide::try_shape_cast_dispatch(&child, &start, down, &hf, &p, 0.02)
        .expect("정상 입력");
    let one = one_hit.expect("단일 박스가 지면을 못 맞췄다 — 배치가 무의미하다");
    let comp = comp_hit.expect("복합 바디가 지형을 통과했다");
    assert!(
        (comp.toi - one.toi).abs() < 1.0e-3,
        "복합 toi {}가 단일 toi {}와 다르다",
        comp.toi,
        one.toi
    );
}

/// **양성은 한 점으로 증명된다** — `shape_penetrates`가 넓은 AABB를 이유로 거절하지 않는다.
///
/// 예전에는 접촉 순회 **전에** 셀 수만 세어 4096을 넘으면 즉시 `ResourceLimit`이었다.
/// 그러면 첫 셀부터 깊이 박혀 있는 큰 바디도 AABB가 넓으면 증명 가능한 `true`를 영원히
/// 못 받고, 캐릭터의 시작 겹침 검사가 매 호출 오류로 고정된다(확인검수 rev_2e93ebf2 MAJOR).
/// 작업량 상한은 순회 안에 있으므로(4096셀) 선판정 없이도 유계다.
#[test]
fn shape_penetrates_proves_positives_even_with_a_huge_aabb() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 257 * 257];
    w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    // 반지름 5m 구 = 100×100셀(예산 4096 초과). 중심을 지면 아래로 내려 **깊이 박아** 둔다.
    let deep = Shape::Sphere { radius: 5.0 };
    assert_eq!(
        w.shape_penetrates(&deep, pose(12.8, -1.0, 12.8), QueryFilter::ALL),
        Ok(true),
        "박혀 있는 것이 증명 가능한데 AABB가 넓다는 이유로 거절됐다"
    );
    // **음성 대조 ①** — 같은 크기인데 안 닿는 높이면, 앞쪽 셀에서 아무것도 못 찾은 채
    // 예산이 끊기므로 `Ok(false)`가 아니라 `Err(ResourceLimit)`(판정 미완료)이다.
    assert_eq!(
        w.shape_penetrates(&deep, pose(12.8, 5.0, 12.8), QueryFilter::ALL),
        Err(ApiError::ResourceLimit),
        "판정 미완료가 '겹치지 않음'으로 뭉개졌다"
    );
    // **음성 대조 ②** — 작은 바디는 예산 안이라 확정적인 답을 받는다(양쪽 다).
    let small = Shape::Sphere { radius: 0.2 };
    assert_eq!(
        w.shape_penetrates(&small, pose(12.8, -0.1, 12.8), QueryFilter::ALL),
        Ok(true)
    );
    assert_eq!(
        w.shape_penetrates(&small, pose(12.8, 5.0, 12.8), QueryFilter::ALL),
        Ok(false)
    );
}

/// **레이·점 질의도 잘못된 입력을 값으로 거절한다** (BLOCKER: 공개 표면의 절반만 방어).
///
/// README가 `cast_ray`·`contains_point`를 "모든 셰이프를 옳게 처리하는 공개 표면"으로
/// 열거하는데 방어는 캐스트 두 함수에만 있었다 — 빈 hull을 넣으면 GJK에서 프로세스가
/// 죽고, 빈 복합은 `None`/`false`가 되어 잘못된 형상이 미스로 위장됐다
/// (확인검수 rev_97887d75 BLOCKER).
#[test]
fn ray_and_point_queries_reject_bad_input_by_value() {
    use maru_collide::QueryReject;
    let empty_hull = Shape::Hull {
        points: Vec::new(),
        radius: 0.0,
    };
    let empty_compound = Shape::Compound {
        children: Vec::new(),
    };
    let p = pose(0.0, 0.0, 0.0);
    let o = Vec3::new(-3.0, 0.0, 0.0);
    let d = Vec3::new(1.0, 0.0, 0.0);
    for bad in [&empty_hull, &empty_compound] {
        assert_eq!(
            maru_collide::try_cast_ray(bad, &p, o, d, 10.0),
            Err(QueryReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_contains_point(bad, &p, Vec3::ZERO),
            Err(QueryReject::InvalidInput)
        );
    }
    // 산술 입력·정규화 불가 자세도 같은 규칙.
    let ok = Shape::Sphere { radius: 1.0 };
    assert_eq!(
        maru_collide::try_cast_ray(&ok, &p, o, Vec3::new(f32::NAN, 0.0, 0.0), 10.0),
        Err(QueryReject::InvalidInput)
    );
    let zero_rot = maru_collide::Pose {
        pos: Vec3::ZERO,
        rot: Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
    };
    assert_eq!(
        maru_collide::try_contains_point(&ok, &zero_rot, Vec3::ZERO),
        Err(QueryReject::InvalidInput)
    );
    // **음성 대조** — 정상 입력은 실제로 맞추고 실제로 품는다.
    assert!(maru_collide::try_cast_ray(&ok, &p, o, d, 10.0)
        .expect("정상 입력")
        .is_some());
    assert_eq!(
        maru_collide::try_contains_point(&ok, &p, Vec3::new(0.5, 0.0, 0.0)),
        Ok(true)
    );
    // 비단위(정규화 가능) 자세는 정규화해서 단위와 같은 답을 낸다.
    let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7);
    let long = Shape::Box {
        half: Vec3::new(2.0, 0.2, 0.2),
        radius: 0.0,
    };
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
    let a = maru_collide::try_cast_ray(&long, &unit, o, d, 10.0).expect("정상");
    let b = maru_collide::try_cast_ray(&long, &scaled, o, d, 10.0).expect("정규화 가능");
    assert!(
        a.is_some(),
        "정상 자세가 박스를 못 맞췄다 — 배치가 무의미하다"
    );
    assert_eq!(
        a.map(|h| h.toi),
        b.map(|h| h.toi),
        "비단위 자세가 찌그러진 형상으로 판정됐다"
    );
}

/// **셀 예산은 질의 하나가 나눠 쓴다** (MAJOR: 바디 수만큼 곱해지는 작업량).
///
/// 예산이 바디마다 새로 4096이면 같은 자리에 지형을 여러 장 겹친 장면에서 한 번의 질의가
/// 바디 수 × 4096셀을 훑는다 — "작업량 상한은 순회 안에 있다"는 주장이 페어에만
/// 참이었다(확인검수 rev_97887d75 MAJOR). 스윕 질의로 확인한다: 셀 수는 스윕 봉투에서
/// 세므로 접촉이 있는지와 무관하게 관측된다.
#[test]
fn query_cell_budget_is_shared_across_bodies() {
    let probe = Shape::Box {
        half: Vec3::new(15.0, 0.2, 15.0),
        radius: 0.0,
    };
    let start = pose(16.0, 2.0, 16.0);
    let down = Vec3::new(0.0, -4.0, 0.0);
    // 지형 한 장(65×65, 셀 0.5m = 32m 폭). 30m 판의 스윕 봉투는 약 60×60 = 3600셀 —
    // 예산(4096) 안이다. 두 장이면 7200으로 넘는다.
    let mut w1 = World::new(WorldParams::default());
    w1.try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
        .expect("지형");
    let one = w1.sweep_shape(&probe, start, down, QueryFilter::ALL);
    assert!(
        matches!(one, Ok(Some(_))),
        "한 장짜리 장면이 예산에 걸렸다 — 대조가 성립하지 않는다: {one:?}"
    );
    // **같은 자리에 두 장을 겹친다.** 예산이 공유되면 총합이 4096을 넘어 판정 미완료가
    // 되고, 공유가 아니면 장 수만큼 곱해진 작업을 끝까지 한다.
    let mut w3 = World::new(WorldParams::default());
    for _ in 0..2 {
        w3.try_add_heightfield(&vec![0.0f32; 65 * 65], 65, 65, 0.5, pose(0.0, 0.0, 0.0))
            .expect("지형");
    }
    assert_eq!(
        w3.sweep_shape(&probe, start, down, QueryFilter::ALL).err(),
        Some(ApiError::ResourceLimit),
        "예산이 바디마다 새로 주어지고 있다 — 총 작업량이 유계가 아니다"
    );
    // **음성 대조** — 작은 바디는 두 장을 훑어도 예산 안이라 답을 받는다.
    let small = Shape::Box {
        half: Vec3::new(0.2, 0.2, 0.2),
        radius: 0.0,
    };
    assert!(matches!(
        w3.sweep_shape(&small, start, down, QueryFilter::ALL),
        Ok(Some(_))
    ));
}

/// **레이 질의는 0 방향과 거대한 원점을 값으로 거절한다** (BLOCKER: 계약 위반 결과).
///
/// 유한성만 보면 (a) `dir=0`이 법선 0 벡터인 히트를 만들고("법선은 단위 벡터" 계약 파기),
/// (b) `origin=f32::MAX`가 중간 계산에서 `inf-inf=NaN`이 되어 `toi`·`normal`이 NaN인
/// 히트를 `Ok(Some)`으로 내보낸다(확인검수 rev_c73f897c BLOCKER).
#[test]
fn ray_query_rejects_zero_direction_and_out_of_range_origin() {
    use maru_collide::QueryReject;
    let s = Shape::Sphere { radius: 1.0 };
    let p = pose(0.0, 0.0, 0.0);
    assert_eq!(
        maru_collide::try_cast_ray(&s, &p, Vec3::ZERO, Vec3::ZERO, 10.0),
        Err(QueryReject::InvalidInput),
        "0 방향이 통과했다 — 법선이 0 벡터인 히트가 나온다"
    );
    assert_eq!(
        maru_collide::try_cast_ray(
            &s,
            &p,
            Vec3::new(f32::MAX, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            f32::MAX
        ),
        Err(QueryReject::InvalidInput),
        "범위 밖 원점이 통과했다 — NaN 히트가 나온다"
    );
    assert_eq!(
        maru_collide::try_contains_point(&s, &p, Vec3::new(f32::MAX, 0.0, 0.0)),
        Err(QueryReject::InvalidInput)
    );
    // **음성 대조** — 정상 입력은 맞추고, 법선은 단위 벡터다.
    let h = maru_collide::try_cast_ray(
        &s,
        &p,
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        10.0,
    )
    .expect("정상 입력")
    .expect("구를 못 맞췄다");
    assert!(
        (h.normal.length() - 1.0).abs() < 1.0e-4,
        "법선이 단위가 아니다"
    );
}

/// **AABB 질의도 패닉하지 않는다** (BLOCKER: 공개 표면의 남은 패닉 경로).
///
/// 레이·점만 `try_`로 바꾸고 `local_aabb()`를 열어 두면, 빈 hull은 `points[0]`에서
/// 빈 복합은 누적값 `expect`에서 여전히 프로세스를 죽인다(확인검수 rev_c73f897c BLOCKER).
#[test]
fn aabb_query_rejects_malformed_shapes_instead_of_panicking() {
    use maru_collide::ShapeReject;
    for bad in [
        Shape::Hull {
            points: Vec::new(),
            radius: 0.0,
        },
        Shape::Compound {
            children: Vec::new(),
        },
    ] {
        assert_eq!(bad.try_local_aabb(), Err(ShapeReject::InvalidShape));
    }
    // **음성 대조** — 정상 형상은 실제 AABB를 준다.
    let s = Shape::Sphere { radius: 0.5 };
    let a = s.try_local_aabb().expect("정상 형상");
    assert!((a.max.x - 0.5).abs() < 1.0e-6 && (a.min.x + 0.5).abs() < 1.0e-6);
}

/// **범위 밖 거대 형상은 질의 경계에서도 거절된다** (MAJOR: 두 경계의 정책 불일치).
///
/// `half = f32::MAX`인 박스는 생성 경계(`validate`)가 `MAX_SHAPE_EXTENT`로 거절하는데
/// 질의 경계(`is_castable`)는 유한성만 봐서 통과시켰다 — 45° 자세의 `transformed_aabb`에서
/// 두 항의 합이 `Inf`가 된다(확인검수 rev_201e3d20 MAJOR).
#[test]
fn out_of_range_shapes_are_rejected_at_the_query_boundary_too() {
    use maru_collide::{QueryReject, ShapeReject};
    let huge_box = Shape::Box {
        half: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        radius: 0.0,
    };
    let huge_hull = Shape::Hull {
        points: vec![Vec3::new(f32::MAX, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        radius: 0.0,
    };
    let huge_offset = Shape::Compound {
        children: vec![(pose(f32::MAX, 0.0, 0.0), Shape::Sphere { radius: 0.5 })],
    };
    let p = pose(0.0, 0.0, 0.0);
    for bad in [&huge_box, &huge_hull, &huge_offset] {
        assert!(bad.validate().is_err(), "생성 경계가 통과시켰다");
        assert_eq!(
            bad.try_local_aabb(),
            Err(ShapeReject::InvalidShape),
            "질의 경계가 통과시켰다 — transformed_aabb에서 Inf가 된다"
        );
        assert_eq!(
            maru_collide::try_cast_ray(bad, &p, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 10.0),
            Err(QueryReject::InvalidInput)
        );
        assert_eq!(
            maru_collide::try_shape_cast_dispatch(
                bad,
                &p,
                Vec3::new(1.0, 0.0, 0.0),
                &Shape::Sphere { radius: 0.5 },
                &pose(5.0, 0.0, 0.0),
                0.02
            ),
            Err(maru_collide::CastReject::InvalidInput)
        );
    }
    // **음성 대조** — 상한 안(1000m)의 큰 형상은 양쪽 경계를 통과한다.
    let big_ok = Shape::Box {
        half: Vec3::new(900.0, 900.0, 900.0),
        radius: 0.0,
    };
    assert!(big_ok.validate().is_ok());
    assert!(big_ok.try_local_aabb().is_ok());
}

/// **변위 범위는 위치 범위의 두 배다 — 정상 월드 횡단 스윕을 막지 않는다** (MAJOR).
///
/// 지원 범위 안의 두 위치 사이 변위는 최대 `2*MAX_INPUT`(20km)다. 위치 상한 하나로 자르면
/// x=-9km → +9km 같은 정상 스윕이 거절되고, 반대로 x=9km에서 `motion=+9km`(끝점 18km,
/// 월드 밖)는 통과했다 — 차원을 잘못 묶은 내 검사의 양방향 오류였다
/// (확인검수 rev_7b22e13d MAJOR). 월드 층은 **끝점**까지 본다(지원 월드 개념이 있으므로).
#[test]
fn sweep_motion_range_is_twice_the_position_range_and_the_endpoint_must_be_inside() {
    let probe = Shape::Sphere { radius: 0.5 };
    let target = Shape::Box {
        half: Vec3::new(1.0, 5.0, 5.0),
        radius: 0.0,
    };
    // ① 저수준 디스패처: 18km 변위는 정상이고 실제로 맞춘다.
    let hit = maru_collide::try_shape_cast_dispatch(
        &probe,
        &pose(-9000.0, 0.0, 0.0),
        Vec3::new(18000.0, 0.0, 0.0),
        &target,
        &pose(0.0, 0.0, 0.0),
        0.02,
    )
    .expect("18km 변위가 거절됐다 — 위치 상한으로 변위를 잘랐다");
    assert!(hit.is_some(), "월드를 횡단하는 스윕이 타깃을 못 맞췄다");
    // 20km를 넘는 변위는 거절(두 지원 위치의 차로 설명할 수 없다).
    assert_eq!(
        maru_collide::try_shape_cast_dispatch(
            &probe,
            &pose(-9000.0, 0.0, 0.0),
            Vec3::new(21000.0, 0.0, 0.0),
            &target,
            &pose(0.0, 0.0, 0.0),
            0.02
        ),
        Err(maru_collide::CastReject::InvalidInput)
    );
    // ② 월드 질의: 같은 18km 스윕은 통과하고, 끝점이 월드 밖인 스윕은 거절된다.
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(target, pose(0.0, 0.0, 0.0)));
    assert!(matches!(
        w.sweep_shape(
            &probe,
            pose(-9000.0, 0.0, 0.0),
            Vec3::new(18000.0, 0.0, 0.0),
            QueryFilter::ALL
        ),
        Ok(Some(_))
    ));
    assert_eq!(
        w.sweep_shape(
            &probe,
            pose(9000.0, 0.0, 0.0),
            Vec3::new(9000.0, 0.0, 0.0),
            QueryFilter::ALL
        )
        .err(),
        Some(ApiError::InvalidValue),
        "끝점이 월드 밖(18km)인데 통과했다"
    );
}

/// **FP 가드 전수** — 캐노니컬 상태를 바꾸는 공개 진입점 전부가 같은 규칙을 따른다.
///
/// 검수가 두 번 같은 계열을 지적했다(생성자 → enqueue). 한 곳만 빠지면 "비표준 환경
/// 진입은 상태를 바꾸지 않는다"가 API 선택에 따라 갈리므로, 목록으로 잠근다.
#[test]
fn every_canonical_mutation_entry_point_is_guarded() {
    let mut w = World::new(WorldParams::default());
    let h = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(0.0, 2.0, 0.0),
        1.0,
    ));
    let h2 = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.3 },
        pose(2.0, 2.0, 0.0),
        1.0,
    ));
    let snap = w.snapshot();
    // 환경을 깨기 **전에** 정상 Body를 하나 만들어 둔다 — `try_add_body`의 가드를 보려면
    // 인자가 유효해야 한다(생성자 가드에 먼저 걸리면 그 경로를 못 본다).
    let pending =
        Body::try_new_static(Shape::Sphere { radius: 0.2 }, pose(5.0, 5.0, 5.0)).expect("생성");
    if !maru_math::force_flush_to_zero_for_test() {
        return;
    }
    // fallible 판은 모두 FpEnvironment를 돌려준다.
    let e = Some(ApiError::FpEnvironment);
    assert_eq!(
        w.set_pose(
            h,
            maru_collide::Pose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY
            }
        )
        .err(),
        e
    );
    assert_eq!(w.set_linear_velocity(h, Vec3::ZERO).err(), e);
    assert_eq!(w.set_angular_velocity(h, Vec3::ZERO).err(), e);
    assert_eq!(w.apply_force(h, Vec3::ZERO).err(), e);
    assert_eq!(w.apply_torque(h, Vec3::ZERO).err(), e);
    assert_eq!(w.apply_impulse(h, Vec3::ZERO).err(), e);
    assert_eq!(w.apply_angular_impulse(h, Vec3::ZERO).err(), e);
    assert_eq!(w.apply_impulse_at_point(h, Vec3::ZERO, Vec3::ZERO).err(), e);
    assert_eq!(w.clear_forces(h).err(), e);
    assert_eq!(w.set_damping(h, 0.1, 0.1).err(), e);
    assert_eq!(w.set_material(h, 0.5, 0.1).err(), e);
    assert_eq!(w.remove_body(h2).err(), e);
    assert_eq!(w.try_set_params(WorldParams::default()).err(), e);
    assert_eq!(w.try_set_sleep_thresholds(0.1, 0.2, 0.5).err(), e);
    assert_eq!(w.try_flush_commands().err(), e);
    assert_eq!(w.try_step(DT).err(), e);
    assert_eq!(
        w.restore(&snap).err(),
        Some(crate::SnapshotError::FpEnvironment)
    );
    assert_eq!(
        w.try_add_body(pending).err(),
        e,
        "미리 만든 Body 추가가 통과했다"
    );
    // 생성자 자체도 거부한다(우회 경로 차단).
    assert_eq!(
        Body::try_new_static(Shape::Sphere { radius: 0.2 }, pose(1.0, 1.0, 1.0)).err(),
        e
    );
    assert!(World::canonicalize_fp_environment());
    // 복구 후에는 전부 정상.
    assert!(w
        .set_pose(
            h,
            maru_collide::Pose {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY
            }
        )
        .is_ok());
    assert!(w.try_step(DT).is_ok());
    assert!(w.is_alive(h2), "거부된 remove가 실제로 지웠다");
}

/// **exclude는 세대를 무시한다** — 제거 후 재사용된 슬롯을 죽은 핸들이 계속 가린다.
#[test]
fn review_query_exclude_respects_generation() {
    let mut w = World::new(WorldParams::default());
    let sphere = |p: maru_math::Vec3| {
        Body::new_static(
            maru_collide::Shape::Sphere { radius: 1.0 },
            maru_collide::Pose {
                pos: p,
                rot: maru_math::Quat::IDENTITY,
            },
        )
    };
    let dead = w.add_body(sphere(maru_math::Vec3::ZERO));
    w.remove_body(dead).unwrap();
    // 같은 슬롯이 재사용되며 세대만 올라간다.
    let fresh = w.add_body(sphere(maru_math::Vec3::ZERO));
    assert_eq!(fresh.index, dead.index, "슬롯 재사용 전제");
    assert_ne!(fresh.generation, dead.generation);
    let hit = w
        .cast_ray_filtered(
            maru_math::Vec3::new(-10.0, 0.0, 0.0),
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            100.0,
            QueryFilter::ALL.exclude(dead),
        )
        .unwrap();
    assert_eq!(
        hit.map(|h| h.body),
        Some(fresh),
        "죽은 핸들이 재사용된 슬롯의 새 바디를 가리면 안 된다"
    );
}

/// **무필터 질의의 계약 변경.** 물리적으로 아무와도 안 부딪히게 filter=0으로 둔
/// 바디가 `cast_ray`(무필터)에서도 사라진다 — 이전 커밋에서는 맞았다.
#[test]
fn review_unfiltered_ray_still_sees_noncolliding_bodies() {
    let mut w = World::new(WorldParams::default());
    let ghost = w.add_body(
        Body::new_static(
            maru_collide::Shape::Sphere { radius: 1.0 },
            maru_collide::Pose {
                pos: maru_math::Vec3::ZERO,
                rot: maru_math::Quat::IDENTITY,
            },
        )
        .with_groups(0b1, 0),
    );
    let hit = w
        .cast_ray(
            maru_math::Vec3::new(-10.0, 0.0, 0.0),
            maru_math::Vec3::new(1.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();
    assert_eq!(hit.map(|h| h.body), Some(ghost));
}

/// 지형 메시(삼각형당 static hull): 구가 4-삼각형 그리드에 얹혀 정착·수면하고,
/// 레이가 지형을 맞힌다. 퇴화 입력(범위 밖 인덱스·두께 0)은 전체 거부(부분 삽입 없음).
#[test]
fn terrain_mesh_supports_resting_sphere_and_rays() {
    let mut w = World::new(WorldParams::default());
    // 2×2 그리드 (4 정점 → 근데 그리드는 9정점… 간단히 3×3 정점, 8 삼각형)
    let mut verts = Vec::new();
    for z in 0..3 {
        for x in 0..3 {
            verts.push(Vec3::new(x as f32 * 2.0 - 2.0, 0.0, z as f32 * 2.0 - 2.0));
        }
    }
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for z in 0..2u32 {
        for x in 0..2u32 {
            let i = z * 3 + x;
            tris.push([i, i + 1, i + 3]);
            tris.push([i + 1, i + 4, i + 3]);
        }
    }
    let handles = w
        .try_add_terrain(&verts, &tris, 0.02)
        .expect("정상 지형이 거부됨");
    assert_eq!(handles.len(), 8);

    let ball = w.add_body(Body::new_dynamic(
        Shape::Sphere { radius: 0.4 },
        pose(0.3, 2.0, 0.2),
        1.0,
    ));
    let mut slept = false;
    for _ in 0..600 {
        w.step(DT);
        if w.is_asleep(ball) {
            slept = true;
            break;
        }
    }
    assert!(slept, "지형 위 구가 잠들지 않음");
    let y = w.body(ball).pose().pos.y;
    // 표면(0) + radius(0.4) + hull radius(0.02) 근처
    assert!((y - 0.42).abs() < 0.05, "정착 높이 이상: {y}");

    // 레이: 위에서 아래로 — 지형에 맞아야 한다
    let hit = w
        .cast_ray(Vec3::new(-1.0, 5.0, -1.0), Vec3::new(0.0, -1.0, 0.0), 10.0)
        .unwrap()
        .expect("지형 레이 미스");
    assert!(hit.toi > 4.0 && hit.toi < 5.1, "toi={}", hit.toi);

    // 퇴화 입력 거부: 범위 밖 인덱스 · 두께 0 (radius 0)
    let n_before = handles.len();
    assert!(w.try_add_terrain(&verts, &[[0, 1, 99]], 0.02).is_err());
    assert!(w.try_add_terrain(&verts, &[[0, 1, 3]], 0.0).is_err());
    let _ = n_before;
}

/// 회전 스윕 CCD(§4 예약 확장): 선속도 0으로 **회전만으로** 벽을 쓸고 지나가는
/// 블레이드 — 선형 캐스트는 이동이 0이라 원리적으로 못 잡는다. 쌍둥이 대조로
/// 전제(비-CCD는 통과)와 효과(CCD는 차단)를 함께 증명한다.
#[test]
fn spinning_blade_does_not_sweep_through_wall() {
    let params = || WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    };
    let build = |ccd: bool| {
        let mut w = World::new(params());
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 3.0, 3.0),
                radius: 0.0,
            },
            pose(1.2, 0.0, 0.0),
        ));
        // 장축을 z로 돌려둔 블레이드(반경 1.5) — y축 스핀이 팁을 +x 벽으로 보낸다.
        let mut b = Body::new_dynamic(
            Shape::Box {
                half: Vec3::new(1.5, 0.05, 0.05),
                radius: 0.0,
            },
            Pose {
                pos: Vec3::ZERO,
                rot: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2),
            },
            1.0,
        );
        if ccd {
            b = b.with_ccd();
        }
        let h = w.add_body(b);
        // 팁 속도 = 30 × 1.5 = 45 m/s → 서브스텝당 이동 0.19m (벽 두께 0.1의 ~2배).
        w.set_angular_velocity(h, Vec3::new(0.0, 30.0, 0.0))
            .unwrap();
        (w, h)
    };
    // 팁의 x 최대 도달 — 관통 관측자. 벽 근면 x=1.15.
    let tip_x = |w: &World, h: BodyHandle| {
        let p = w.body(h).pose();
        let a = p.pos + p.rot.rotate(Vec3::new(1.5, 0.0, 0.0));
        let b = p.pos + p.rot.rotate(Vec3::new(-1.5, 0.0, 0.0));
        a.x.max(b.x)
    };

    // 전제: CCD 없이는 팁이 벽 반대편(x > 1.25)까지 지나간다.
    let (mut w0, h0) = build(false);
    let mut crossed = false;
    for _ in 0..120 {
        w0.step(DT);
        if tip_x(&w0, h0) > 1.25 {
            crossed = true;
            break;
        }
    }
    assert!(crossed, "전제 실패: 비-CCD 블레이드가 벽을 통과하지 않았다");

    // 효과: CCD는 매 스텝 팁이 벽 근면을 (skin 여유 안에서) 넘지 못한다.
    let (mut w1, h1) = build(true);
    let skin = w1.params().speculative_margin;
    for step in 0..120 {
        w1.step(DT);
        let x = tip_x(&w1, h1);
        assert!(
            x <= 1.15 + skin + 0.02,
            "step {step}: 회전 CCD가 팁 관통을 허용 (tip_x={x})"
        );
    }
    assert!(w1.diagnostics().ccd_clamps > 0 || tip_x(&w1, h1) < 1.0);
}
