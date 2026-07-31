//! ccd 관련 테스트.

use super::*;

/// **CCD가 지형을 뚫지 않는다** — 총알 속도로 쏜 구가 격자에서 멈춘다.
#[test]
fn ccd_bullet_does_not_tunnel_through_heightfield() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 9 * 9];
    w.try_add_heightfield(&heights, 9, 9, 1.0, hf_pose())
        .expect("지형");
    let b = Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(4.0, 6.0, 4.0), 1.0).with_ccd();
    let h = w.add_body(b);
    w.set_linear_velocity(h, Vec3::new(0.0, -400.0, 0.0))
        .expect("속도");
    for _ in 0..20 {
        w.step(DT);
    }
    let y = w.body(h).pose().pos.y;
    assert!(y > -0.5, "지형을 관통했다: y={y}");
}

// ===== 검수 rev_1e81559e 회귀 =====

/// **회전 CCD가 하이트필드를 만나도 trap하지 않는다** (BLOCKER).
/// 회전 스윕은 선속도 0에서도 켜지므로, 지형 옆에서 도는 CCD 바디가 즉시 재현했다.
#[test]
fn rotational_ccd_against_heightfield_does_not_trap() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        substeps: 8,
        ..WorldParams::default()
    });
    let heights = vec![0.0f32; 9 * 9];
    w.try_add_heightfield(&heights, 9, 9, 1.0, pose(0.0, 0.0, 0.0))
        .expect("지형");
    let blade = Body::new_dynamic(
        Shape::Box {
            half: Vec3::new(1.2, 0.05, 0.1),
            radius: 0.0,
        },
        pose(4.0, 0.6, 4.0),
        1.0,
    )
    .with_ccd();
    let h = w.add_body(blade);
    w.set_angular_velocity(h, Vec3::new(0.0, 0.0, 60.0))
        .expect("각속도");
    for _ in 0..30 {
        w.step(DT); // trap이면 여기서 프로세스가 죽는다
    }
    assert!(w.body(h).pose().pos.y.is_finite());
}

/// **내부 CCD 캐스트도 작업량 상한 안에서 돈다** (MAJOR: 스텝 안의 무한 작업량).
///
/// 공개 질의(`sweep_shape`)만 사전 예산을 봤고, 물리 스텝 안의 CCD는 디스패처를 직접
/// 불러 예산을 우회했다 — 1024×1024 지형을 덮는 큰 CCD 바디 하나가 서브스텝마다
/// 100만 셀을 훑어 프레임을 멈출 수 있었다(확인검수 rev_d5b32e22 MAJOR).
///
/// 초과 시 정책은 **보수적 정지**(그 서브스텝 이동·회전 0)이고, 보고는
/// `Diagnostics::ccd_incomplete`다. `ResourceLimit`을 쓸 수 없는 이유는 그 필드 문서에.
#[test]
fn internal_ccd_cast_is_bounded_and_reports_incompleteness() {
    let mut w = World::new(WorldParams::default());
    let heights = vec![0.0f32; 257 * 257];
    w.try_add_heightfield(&heights, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    // 바닥에서 0.3m 뜬 반지름 5m 구. 접촉(내로우페이즈)은 타일 Y밴드 가지치기로 0셀이라
    // 스텝이 예산에 걸리지 않지만, 아래로 빠르게 쏘면 **CCD 스윕 봉투**가 100×100셀을
    // 덮어 예산(4096)을 넘는다 — 예전에는 이 경로에 상한이 아예 없었다.
    let big = w.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 5.0 }, pose(12.8, 5.3, 12.8), 1.0).with_ccd(),
    );
    w.set_linear_velocity(big, Vec3::new(0.0, -400.0, 0.0))
        .expect("속도");
    let y0 = w.body(big).pose().pos.y;
    assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
    assert!(
        w.diagnostics().ccd_incomplete > 0,
        "CCD가 예산을 넘겼는데 보고가 없다 — 상한이 없거나 조용히 강등됐다"
    );
    assert_eq!(
        w.body(big).pose().pos.y,
        y0,
        "판정 미완료인데 이동했다 — 보수적 정지가 아니다"
    );
    // **속도가 쌓이지 않는다.** 위치만 얼리면 중력·솔버가 속도를 계속 키워
    // `max_linear_velocity`(500 m/s)까지 저장되고, 호스트가 지형을 줄여 판정이 가능해지는
    // 순간 그 바디가 최대 속도로 발사된다(확인검수 rev_2e93ebf2 MAJOR — 위치만 본 이
    // 테스트가 놓쳤다). 판정 미완료 서브스텝마다 속도를 0으로 되돌리므로, 한 스텝 안에서
    // 정지 이후 남은 서브스텝의 중력만 남는다(약 0.12 m/s) — 그 이상 자라지 않는다.
    // **저장된 에너지가 없다.** 위치만 얼리면 초기 400 m/s가 그대로 남고 중력이 계속
    // 더해져 `max_linear_velocity`(500)까지 쌓인다 — 호스트가 지형을 줄여 판정이
    // 가능해지는 순간 그 바디가 최대 속도로 발사된다(확인검수 rev_2e93ebf2 MAJOR:
    // 위치만 본 이 테스트가 놓쳤다). 판정 미완료 서브스텝에서 속도까지 0으로 되돌리므로,
    // 이후 2초 동안 속도는 한 자리 수를 넘지 않고 구는 지형 위에 안착한다.
    let mut vmax = 0.0f32;
    for _ in 0..120 {
        assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
        vmax = vmax.max(w.body(big).linvel().length());
    }
    assert!(
        vmax < 20.0,
        "판정 미완료 정지가 에너지를 저장하고 있다: 최대 {vmax} m/s (초기 400, 상한 500)"
    );
    assert!(
        w.body(big).pose().pos.y > 4.0,
        "구가 지형을 뚫고 내려갔다: y={}",
        w.body(big).pose().pos.y
    );
    // **음성 대조** — 같은 지형·같은 속도의 작은 CCD 바디는 예산 안이라 정상적으로
    // 움직이고 지면에서 절단된다(상한이 CCD를 통째로 끈 게 아니다).
    let mut w2 = World::new(WorldParams::default());
    let heights2 = vec![0.0f32; 257 * 257];
    w2.try_add_heightfield(&heights2, 257, 257, 0.1, pose(0.0, 0.0, 0.0))
        .expect("지형 추가");
    let small = w2.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.2 }, pose(12.8, 5.3, 12.8), 1.0).with_ccd(),
    );
    w2.set_linear_velocity(small, Vec3::new(0.0, -400.0, 0.0))
        .expect("속도");
    assert_eq!(w2.try_step(1.0 / 60.0), Ok(()));
    assert_eq!(w2.diagnostics().ccd_incomplete, 0);
    // 대조군은 속도가 살아 있다 — 정지 정책이 CCD 바디 전부를 얼린 게 아니다.
    assert!(w2.body(small).linvel().y < 0.0);
    let y = w2.body(small).pose().pos.y;
    assert!(
        y < 5.3 && y > -0.5,
        "작은 CCD 바디가 안 움직였거나 지형을 관통했다: y={y}"
    );
}

/// **월드 경계를 넘는 CCD 바디는 경계 고정으로 끝난다 — 판정 미완료가 아니다.**
///
/// 이 테스트가 고정하는 것: 빠른 CCD 바디가 스텝 중간에 10km를 넘어도 스텝은 `Ok`이고,
/// 상태는 경계 안으로 고정되며(`out_of_bounds` 보고), `ccd_incomplete`는 0이다.
///
/// **주의 — 이 테스트는 world.rs의 "범위 밖 자세면 캐스트하지 않는다" 가드를 검증하지
/// 않는다.** 그 가드가 도는 조건은 "정적 후보가 스윕 AABB와 겹치는데 캐스트는 빗나간다"이고,
/// 그런 배치는 만들기가 취약하다(벽을 두면 CCD가 먼저 막아 경계를 못 넘고, 안 두면 캐스트가
/// 아예 안 돈다 — 실제로 두 배치를 다 시도해 확인했다). 가드는 원인 오보(범위 이탈이
/// `ccd_incomplete`로 보고되는 것)를 막는 **defense-in-depth**이며, 그 사실을 여기 적어 둔다.
/// 지원 범위 밖 입력을 디스패처가 거절한다는 것 자체는
/// `out_of_range_arithmetic_input_is_rejected_not_reinterpreted`가 단위로 고정한다.
#[test]
fn ccd_body_crossing_the_world_bound_reports_out_of_bounds_not_incomplete() {
    let mut w = World::new(WorldParams::default());
    // 경계(10km) 바로 안쪽에서 바깥으로 달린다. 한 스텝 이동은 **물리 상한**
    // (`max_linear_velocity`, 기본 500 m/s)로 정해지므로 8.3m다 — 그보다 가까이 둔다.
    let max = maru_collide::MAX_INPUT;
    let h = w.add_body(
        Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(max - 4.0, 0.0, 0.0),
            1.0,
        )
        .with_ccd(),
    );
    w.set_linear_velocity(h, Vec3::new(w.params().max_linear_velocity, 0.0, 0.0))
        .expect("속도");
    assert_eq!(w.try_step(1.0 / 60.0), Ok(()));
    assert_eq!(
        w.diagnostics().ccd_incomplete,
        0,
        "범위 이탈이 '판정 미완료'로 보고됐다 — 호스트가 지형 해상도를 의심하게 된다"
    );
    assert!(
        w.diagnostics().out_of_bounds > 0,
        "범위 이탈이 보고되지 않았다 (pos={:?})",
        w.body(h).pose().pos
    );
    // 상태는 경계 안으로 고정된다(OOB 계약).
    assert!(maru_collide::vec_in_range(w.body(h).pose().pos));
    // **음성 대조** — 경계에서 멀리 있는 같은 바디는 아무 진단도 올리지 않는다.
    let mut w2 = World::new(WorldParams::default());
    let h2 = w2.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.5 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
    );
    w2.set_linear_velocity(h2, Vec3::new(100.0, 0.0, 0.0))
        .expect("속도");
    assert_eq!(w2.try_step(1.0 / 60.0), Ok(()));
    assert_eq!(w2.diagnostics().ccd_incomplete, 0);
    assert_eq!(w2.diagnostics().out_of_bounds, 0);
    assert!(w2.body(h2).pose().pos.x > 0.0, "정상 바디가 안 움직였다");
}

/// **정적 후보가 없는 장면에서 CCD 켬/끔은 궤적을 바꾸지 않는다** (BLOCKER: 영구 OOB 오제거).
///
/// 내 OOB 가드가 `d=0` + `ccd_frozen`으로 병진을 얼렸다. 그러면 잠깐 경계를 넘었다가
/// 중력이 안쪽으로 되돌릴 바디가 **복귀하지 못하고 스텝 말에 영구 OOB로 제거**된다.
/// 정적 후보가 0개면 CCD가 할 일이 없으므로 켬/끔이 결과를 바꿀 이유가 없다 — 그 불일치가
/// 이 결함의 반박 불가 증거다(확인검수 rev_cccbba53).
#[test]
fn ccd_does_not_change_the_trajectory_when_there_are_no_static_candidates() {
    // 검수가 제시한 장면: 강한 역방향 중력 + 많은 서브스텝 + 경계 코앞에서 밖으로 향하는 속도.
    // 첫 서브스텝만 경계를 살짝 넘고, 전체 스텝의 순변위는 안쪽이다.
    let params = |()| WorldParams {
        gravity: Vec3::new(-100.0, 0.0, 0.0),
        substeps: 64,
        ..WorldParams::default()
    };
    let build = |ccd: bool| {
        let mut w = World::new(params(()));
        let b = Body::new_dynamic(
            Shape::Sphere { radius: 0.2 },
            pose(maru_collide::MAX_INPUT - 0.01, 0.0, 0.0),
            1.0,
        );
        let h = w.add_body(if ccd { b.with_ccd() } else { b });
        w.set_linear_velocity(h, Vec3::new(10.0, 0.0, 0.0))
            .expect("속도");
        (w, h)
    };
    let (mut w_ccd, h_ccd) = build(true);
    let (mut w_plain, h_plain) = build(false);
    assert_eq!(w_ccd.try_step(0.25), Ok(()));
    assert_eq!(w_plain.try_step(0.25), Ok(()));
    // **음성 대조** — CCD 없는 쪽이 실제로 경계 안으로 돌아와야 비교가 의미 있다.
    assert!(
        !w_plain.is_out_of_bounds(h_plain),
        "대조군이 복귀하지 못했다 — 장면이 무의미하다: x={}",
        w_plain.body(h_plain).pose().pos.x
    );
    assert!(w_plain.body(h_plain).pose().pos.x < maru_collide::MAX_INPUT);
    // CCD를 켠 쪽도 같아야 한다(할 일이 없는 CCD가 궤적을 바꾸면 안 된다).
    assert!(
        !w_ccd.is_out_of_bounds(h_ccd),
        "CCD를 켰더니 영구 OOB로 제거됐다 — 후보가 0개인데 CCD가 궤적을 바꿨다"
    );
    assert_eq!(
        w_ccd.body(h_ccd).pose().pos,
        w_plain.body(h_plain).pose().pos,
        "CCD 켬/끔이 위치를 바꿨다"
    );
    assert_eq!(w_ccd.body(h_ccd).linvel(), w_plain.body(h_plain).linvel());
}

/// **경계 밖의 유효한 정적 기하도 CCD가 잡는다** (BLOCKER: "밖에는 부딪힐 것이 없다"는 오류).
///
/// 정적 바디는 **부모 자세만** `MAX_INPUT` 안이면 되고, 복합 자식 오프셋과 격자는
/// `MAX_SHAPE_EXTENT`(1km)까지 더 뻗을 수 있다. 그래서 "월드 범위 밖에는 충돌할 것이 없다"는
/// 내 전제가 틀렸고, 바디 원점이 범위 밖이라고 캐스트를 생략하면 유효한 장애물을 조용히
/// 관통한다(확인검수 rev_5ca93cb1). 지금은 캐스트를 **상대 좌표**로 돌아 범위 문제 자체가 없다.
#[test]
fn ccd_catches_static_geometry_beyond_the_world_bound() {
    let max = maru_collide::MAX_INPUT;
    let mut w = World::new(WorldParams {
        gravity: Vec3::new(-100.0, 0.0, 0.0),
        substeps: 64,
        ..WorldParams::default()
    });
    // 장애물: 부모 자세는 경계 위(허용), 자식은 로컬 +0.3m → 실제 중심은 경계 **밖**이다.
    // 모든 생성 정책을 통과한다. 오프셋이 이만큼 커야 탄환이 **원점이 이미 범위 밖일 때**
    // 도달한다 — 0.12m로는 첫 서브스텝(원점이 아직 안쪽)에 닿아서 옛 가드로도 잡혔다.
    let obstacle = Shape::Compound {
        children: vec![(pose(0.3, 0.0, 0.0), Shape::Sphere { radius: 0.05 })],
    };
    assert!(
        obstacle.validate().is_ok(),
        "장애물이 정책을 위반한다 — 전제가 다르다"
    );
    w.add_body(Body::new_static(obstacle, pose(max, 0.0, 0.0)));
    // 탄환: 경계 코앞에서 바깥으로. 중력이 되돌리므로 스텝 말에는 다시 안쪽이다
    // (그래서 `out_of_bounds`도 안 뜨고, 관통이 조용히 지나갔던 것이다).
    let h = w.add_body(
        Body::new_dynamic(
            Shape::Sphere { radius: 0.05 },
            pose(max - 0.01, 0.0, 0.0),
            1.0,
        )
        .with_ccd(),
    );
    w.set_linear_velocity(h, Vec3::new(10.0, 0.0, 0.0))
        .expect("속도");
    assert_eq!(w.try_step(0.25), Ok(()));
    assert!(
        !w.ccd_hits().is_empty(),
        "경계 밖 장애물을 CCD가 못 봤다 — 관통이 조용히 지나갔다 (x={}, oob={})",
        w.body(h).pose().pos.x,
        w.diagnostics().out_of_bounds
    );
    // **음성 대조 ①** — 장애물이 없으면 히트도 없다(위 단언이 배치와 무관하게 참이 아니다).
    let mut w0 = World::new(WorldParams {
        gravity: Vec3::new(-100.0, 0.0, 0.0),
        substeps: 64,
        ..WorldParams::default()
    });
    let h0 = w0.add_body(
        Body::new_dynamic(
            Shape::Sphere { radius: 0.05 },
            pose(max - 0.01, 0.0, 0.0),
            1.0,
        )
        .with_ccd(),
    );
    w0.set_linear_velocity(h0, Vec3::new(10.0, 0.0, 0.0))
        .expect("속도");
    assert_eq!(w0.try_step(0.25), Ok(()));
    assert!(w0.ccd_hits().is_empty());
    // **음성 대조 ②** — 같은 장애물을 월드 **안쪽**에 두면 원래도 잡혔다(이 결함은
    // 경계 밖 기하에 한정된다).
    let mut w_in = World::new(WorldParams {
        gravity: Vec3::new(-100.0, 0.0, 0.0),
        substeps: 64,
        ..WorldParams::default()
    });
    w_in.add_body(Body::new_static(
        Shape::Compound {
            children: vec![(pose(0.3, 0.0, 0.0), Shape::Sphere { radius: 0.05 })],
        },
        pose(0.0, 0.0, 0.0),
    ));
    let h_in = w_in.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.05 }, pose(-0.01, 0.0, 0.0), 1.0).with_ccd(),
    );
    w_in.set_linear_velocity(h_in, Vec3::new(10.0, 0.0, 0.0))
        .expect("속도");
    assert_eq!(w_in.try_step(0.25), Ok(()));
    assert!(!w_in.ccd_hits().is_empty());
}

/// **CCD 상대 좌표의 원점 선택이 유효한 후보를 범위 밖으로 밀어내지 않는다** (BLOCKER).
///
/// 원점을 움직이는 바디에 두면 상대 자세 하나가 두 부모 간격 **전체**를 떠안아, 간격이
/// 10km를 넘는 유효한 대형 후보가 디스패처의 `MAX_INPUT` 검사에 걸린다 — 그러면 실제
/// TOI를 계산하지 못하고 바디가 서브스텝 시작점에서 보수적으로 정지한다
/// (확인검수 rev_6f54ca63). 중점을 쓰면 각 상대 자세가 간격의 절반이다.
///
/// 이 성질은 **산술**이므로 씬 대신 직접 고정한다. 씬으로 만들려면 최대 크기 형상 + 최대
/// 속도 + 최대 dt + 대각 배치가 모두 겹쳐야 해서(축별 AABB 겹침 조건 때문에 1차원
/// 배치로는 8.7km가 한계다) 취약하다 — 그 분석도 여기 남긴다.
#[test]
fn ccd_cast_origin_keeps_both_relative_poses_in_range() {
    let max = maru_collide::MAX_INPUT;
    // 후보가 될 수 있는 최악 간격(축별 4464+4464+2500 = 11428m, 대각이면 그 √3배).
    let worst = 11428.0f32 * 3.0f32.sqrt();
    for delta in [
        Vec3::new(worst, 0.0, 0.0),
        Vec3::new(11428.0, 11428.0, 11428.0),
        Vec3::new(-10200.0, 3000.0, -7000.0),
        Vec3::new(max, max, max),
    ] {
        let a = Vec3::new(-max * 0.5, max * 0.25, 0.0);
        let b = a + delta;
        let o = crate::world::ccd_cast_origin_pub_for_test(a, b);
        assert!(
            maru_collide::vec_in_range(a - o) && maru_collide::vec_in_range(b - o),
            "간격 {delta:?}에서 상대 자세가 범위를 벗어났다: {:?} / {:?}",
            a - o,
            b - o
        );
    }
    // **음성 대조** — 원점을 움직이는 바디에 두는 옛 방식은 같은 간격에서 실패한다.
    let a = Vec3::ZERO;
    let b = Vec3::new(11428.0, 0.0, 0.0);
    assert!(
        !maru_collide::vec_in_range(b - a),
        "이 간격이 원래 범위 안이면 대조가 무의미하다"
    );
}

/// CCD의 존재 이유: 고속 소형 바디가 얇은 static을 스텝 사이에 건너뛰는 것.
/// 같은 장면에서 CCD 없이는 관통하고(전제 확인 — 이게 안 뚫리면 테스트가 무의미),
/// CCD를 켜면 벽 앞에서 멈춰야 한다.
#[test]
fn ccd_stops_bullet_that_tunnels_without_it() {
    let wall = |w: &mut World| {
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 5.0, 5.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ))
    };
    let bullet = |ccd: bool| {
        let b = Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0);
        if ccd {
            b.with_ccd()
        } else {
            b
        }
    };
    // 무중력 — 순수 관통 시나리오. 속도 400 m/s = 스텝당 6.7m (벽 두께 0.1의 67배).
    let params = || WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    };

    let mut w0 = World::new(params());
    wall(&mut w0);
    let h0 = w0.add_body(bullet(false));
    w0.set_linear_velocity(h0, Vec3::new(400.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..10 {
        w0.step(DT);
    }
    assert!(
        w0.body(h0).pose().pos.x > 11.0,
        "CCD 없이 관통해야 전제가 성립하는데 x={} — 테스트가 아무것도 증명 못 한다",
        w0.body(h0).pose().pos.x
    );

    let mut w1 = World::new(params());
    wall(&mut w1);
    let h1 = w1.add_body(bullet(true));
    w1.set_linear_velocity(h1, Vec3::new(400.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..10 {
        w1.step(DT);
    }
    let x = w1.body(h1).pose().pos.x;
    assert!(x < 9.95, "CCD가 켜졌는데 벽(근면 x=9.95)을 넘었다: x={x}");
    assert!(x > 8.0, "벽 앞 어딘가에는 도달해야 한다: x={x}");
}

/// **TOI-hit 큐** (설계 §4.5 / M4): 절단 순간이 호스트에게 보고돼야 한다.
///
/// 접촉 이벤트로는 이 정보를 낼 수 없다 — linear cast는 바디를 표면에 세워 두고
/// 반발·정지는 다음 스텝의 접촉이 해석하므로, 절단 자체는 어디에도 안 남았다.
#[test]
fn ccd_reports_the_toi_hit_it_clamped_on() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    });
    let wall = w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        },
        pose(10.0, 0.0, 0.0),
    ));
    let bullet = w.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
    );
    w.set_linear_velocity(bullet, Vec3::new(400.0, 0.0, 0.0))
        .unwrap();
    // 벽까지 6.7m/step → 2스텝째에 맞는다. 맞은 스텝의 히트를 잡는다.
    let mut hits = Vec::new();
    for _ in 0..4 {
        w.step(DT);
        hits.extend_from_slice(w.ccd_hits());
        if !hits.is_empty() {
            break;
        }
    }
    assert!(!hits.is_empty(), "CCD가 절단했는데 히트가 보고되지 않았다");
    let h = hits[0];
    assert_eq!(h.body, bullet);
    assert_eq!(h.other, wall, "부딪힌 static 바디가 잘못 보고됐다");
    assert!(
        (0.0..=1.0).contains(&h.toi),
        "toi는 서브스텝 분율이어야 한다: {}",
        h.toi
    );
    assert!(
        h.substep < WorldParams::default().substeps,
        "서브스텝 인덱스가 범위를 넘었다: {}",
        h.substep
    );
    // 법선은 벽이 총알을 밀어내는 방향(−x)이어야 한다.
    assert!(
        h.normal.x < -0.9,
        "법선이 밀어내는 방향이 아니다: {:?}",
        h.normal
    );
    // 접점은 벽 근면(x≈9.95) 부근이어야 한다.
    assert!(
        (h.point.x - 9.95).abs() < 0.3,
        "접점이 벽 표면에서 멀다: {:?}",
        h.point
    );
    // **스텝 단위 출력**이다 — 매 스텝 비워지므로 누적되지 않는다.
    // (벽에 붙은 뒤에도 잔여 이동이 있으면 매 스텝 새 히트가 날 수 있다 —
    //  그러니 "빈다"가 아니라 "안 쌓인다"가 검증해야 할 성질이다.)
    let cap_per_step = WorldParams::default().substeps as usize;
    for _ in 0..20 {
        w.step(DT);
        assert!(
            w.ccd_hits().len() <= cap_per_step,
            "히트가 스텝을 넘어 누적됐다: {} > 서브스텝 {}",
            w.ccd_hits().len(),
            cap_per_step
        );
    }
}

/// 히트 큐가 **상한에서 포화**해도 결정적이고, 버려진 수를 보고해야 한다.
/// (M4 완료 기준: "TOI 큐 상한 포화 프레임 — 할당 0·오버플로 결정성")
#[test]
fn ccd_hit_queue_saturates_deterministically() {
    let build = || {
        let mut w = World::new(WorldParams {
            gravity: Vec3::ZERO,
            ..WorldParams::default()
        });
        w.add_body(Body::new_static(
            Shape::Box {
                half: Vec3::new(0.05, 200.0, 200.0),
                radius: 0.0,
            },
            pose(10.0, 0.0, 0.0),
        ));
        let n = crate::world::MAX_CCD_HITS + 64;
        for k in 0..n {
            // 격자로 흩어 놓아 서로 겹치지 않게 한다.
            let y = (k % 64) as f32 * 0.5 - 16.0;
            let z = (k / 64) as f32 * 0.5 - 8.0;
            let h = w.add_body(
                Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, y, z), 1.0).with_ccd(),
            );
            w.set_linear_velocity(h, Vec3::new(400.0, 0.0, 0.0))
                .unwrap();
        }
        w
    };
    let run = || {
        let mut w = build();
        let mut seen = Vec::new();
        let mut dropped = 0usize;
        for _ in 0..4 {
            w.step(DT);
            if !w.ccd_hits().is_empty() {
                seen = w.ccd_hits().to_vec();
                dropped = w.diagnostics().ccd_hits_dropped;
                break;
            }
        }
        (seen, dropped)
    };
    let (a, da) = run();
    let (b, db) = run();
    assert!(!a.is_empty(), "포화 시나리오에서 히트가 하나도 없다");
    assert!(
        a.len() <= crate::world::MAX_CCD_HITS,
        "상한을 넘겨 보고했다: {}",
        a.len()
    );
    assert!(
        da > 0,
        "상한을 넘겼는데 버려진 수가 0으로 보고됐다 — 조용한 유실이다"
    );
    // **결정성**: 같은 입력이면 남는 집합도 버려지는 수도 같아야 한다.
    assert_eq!(da, db, "버려진 수가 실행마다 다르다");
    assert_eq!(a.len(), b.len(), "보고된 히트 수가 실행마다 다르다");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.body, y.body, "히트 순서가 실행마다 다르다");
        assert_eq!(x.other, y.other);
        assert_eq!(x.toi.to_bits(), y.toi.to_bits(), "toi가 실행마다 다르다");
    }
}

/// 복원은 **폐기한 미래의 스텝 출력**까지 지워야 한다.
///
/// 히트 큐는 캐노니컬 상태가 아니라 남아 있었는데, 그러면 호스트가 되돌린 타임라인의
/// 히트를 다시 읽어 데미지·이펙트를 두 번 적용한다. `diag`는 복원 시 0이 되므로
/// `ccd_hits_dropped == 0`이라 온전한 목록처럼 보이기까지 한다
/// (교차검수 rev_5e74f64f MAJOR).
#[test]
fn restore_clears_ccd_hits_from_the_discarded_timeline() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    });
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        },
        pose(10.0, 0.0, 0.0),
    ));
    let bullet = w.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
    );
    w.set_linear_velocity(bullet, Vec3::new(400.0, 0.0, 0.0))
        .unwrap();
    // 충돌 **전** 스냅샷을 잡아 둔다.
    let before = w.try_snapshot().unwrap();
    let mut hit = false;
    for _ in 0..4 {
        w.step(DT);
        if !w.ccd_hits().is_empty() {
            hit = true;
            break;
        }
    }
    assert!(hit, "전제: 이 씬에서 CCD 히트가 나야 한다");
    // 되돌린다 — 그 히트는 이제 없던 일이다.
    w.restore(&before).unwrap();
    assert!(
        w.ccd_hits().is_empty(),
        "복원이 폐기한 타임라인의 히트를 남겼다: {}건",
        w.ccd_hits().len()
    );
}

/// CCD 고정 버퍼도 **페어 예산을 역산하기 전에** 총 스크래치에서 빠져야 한다.
///
/// 안 빼면 페어가 바이트 상한까지 찬 월드가 128 MiB를 거의 다 쓴 뒤 히트 버퍼를 더 잡아
/// 총상한을 넘긴다. 관측 가능한 형태로 잠근다: **같은 씬에서 CCD를 켜면 페어 예산이
/// 그만큼 줄어야 한다** (교차검수 rev_5e74f64f MAJOR).
#[test]
fn ccd_hit_buffer_is_charged_against_the_step_scratch_budget() {
    let build = |ccd: bool| {
        let mut w = World::new(WorldParams {
            // 페어 예산을 **바이트 상한이 지배하도록** 개수 상한을 최대로 연다
            // (개수 상한이 이기면 CCD 바이트를 빼도 예산이 안 변해 아무것도 못 잰다).
            max_pairs_per_body: 4096,
            ..WorldParams::default()
        });
        for i in 0..200 {
            let b = Body::new_dynamic(
                Shape::Sphere { radius: 0.1 },
                pose(i as f32 * 3.0, 50.0, 0.0),
                1.0,
            );
            w.add_body(if ccd { b.with_ccd() } else { b });
        }
        w.step(DT);
        w.diagnostics().pair_budget
    };
    let without = build(false);
    let with = build(true);
    assert!(
        with < without,
        "CCD를 켰는데 페어 예산이 그대로다 — 히트 버퍼가 회계에서 빠져 있다 \
         (without={without}, with={with})"
    );
}

/// ccd 플래그는 캐노니컬 상태다 — 스냅샷 왕복에서 살아남고, 왕복한 월드의 미래가 같다.
#[test]
fn ccd_flag_survives_snapshot_roundtrip() {
    let params = WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    };
    let mut w = World::new(params);
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        },
        pose(10.0, 0.0, 0.0),
    ));
    let h = w.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0).with_ccd(),
    );
    w.set_linear_velocity(h, Vec3::new(400.0, 0.0, 0.0))
        .unwrap();
    w.step(DT);
    let snap = w.snapshot();
    let mut w2 = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    });
    w2.restore(&snap).unwrap();
    assert!(w2.body(h).ccd(), "ccd 플래그가 왕복에서 사라졌다");
    for _ in 0..9 {
        w.step(DT);
        w2.step(DT);
    }
    assert_eq!(w.snapshot(), w2.snapshot(), "왕복 후 궤적 분기");
    assert!(
        w.body(h).pose().pos.x < 9.95,
        "왕복 뒤에도 CCD가 벽을 막아야 한다"
    );
}

/// 회귀(rev_3398738c MAJOR ③): frozen(평면 5개째)이 돼도 회전 스윕은 계속
/// 캐스트·절단된다 — 이동만 얼리면 팁이 남은 서브스텝 동안 벽을 쓸고 지나간다.
#[test]
fn frozen_ccd_body_keeps_rotation_clamped() {
    // 서브스텝 8 — 평면은 서브스텝당 최대 1개 쌓이므로 5개째(frozen)에 도달하려면
    // 기본 4로는 불가능하다. toi=0 접촉 히트가 같은 법선을 반복해서 쌓는다.
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        substeps: 8,
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
    let mut froze = false;
    for step in 0..120 {
        w.step(DT);
        froze |= w.diagnostics().ccd_frozen > 0;
        let p = w.body(h).pose();
        let tip = (p.pos + p.rot.rotate(Vec3::new(1.5, 0.0, 0.0)))
            .x
            .max((p.pos + p.rot.rotate(Vec3::new(-1.5, 0.0, 0.0))).x);
        assert!(
            tip <= 1.15 + skin + 0.02,
            "step {step}: frozen 후 회전이 벽을 쓸었다 (tip_x={tip})"
        );
    }
    assert!(
        froze,
        "frozen이 발동하지 않았다 — 테스트가 경로를 증명하지 못함"
    );
}

/// 회귀(rev_e4075e85): 센서+CCD 바디는 벽에 물리적으로 멈추면 안 된다 — 센서 계약.
#[test]
fn sensor_with_ccd_is_not_clamped_by_walls() {
    let mut w = World::new(WorldParams {
        gravity: Vec3::ZERO,
        ..WorldParams::default()
    });
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.05, 5.0, 5.0),
            radius: 0.0,
        },
        pose(10.0, 0.0, 0.0),
    ));
    let s = w.add_body(
        Body::new_dynamic(Shape::Sphere { radius: 0.1 }, pose(0.0, 0.0, 0.0), 1.0)
            .as_sensor(true)
            .with_ccd(),
    );
    w.set_linear_velocity(s, Vec3::new(400.0, 0.0, 0.0))
        .unwrap();
    for _ in 0..10 {
        w.step(DT);
    }
    assert!(
        w.body(s).pose().pos.x > 11.0,
        "센서가 CCD에 막혔다: x={}",
        w.body(s).pose().pos.x
    );
}
