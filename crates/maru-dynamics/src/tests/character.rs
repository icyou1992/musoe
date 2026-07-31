//! character 관련 테스트.

use super::*;

/// 빈 공간에서는 원하는 만큼 그대로 간다 — 그리고 바닥 위면 `on_ground`다.
#[test]
fn character_moves_freely_and_detects_ground() {
    let w = char_world();
    let mut c = capsule_char(0.0, 0.8, 0.0);
    let r = c
        .move_and_slide(&w, Vec3::new(1.0, 0.0, 0.0), QueryFilter::ALL)
        .expect("이동");
    assert!(
        (r.position.x - 1.0).abs() < 1e-3,
        "막힌 것이 없는데 못 갔다: {:?}",
        r.position
    );
    assert!(r.on_ground, "바닥 위인데 지면 판정이 안 됐다");
    assert!(r.ground_normal.y > 0.9, "지면 법선 {:?}", r.ground_normal);
    assert_eq!(r.remaining, Vec3::ZERO);
}

/// 낮은 턱은 **걸어 올라간다**(계단), 높은 벽은 못 올라간다 — 같은 코드가 둘을 가른다.
///
/// 캐릭터를 **상자**로 둔다(캡슐이 아니라): 캡슐은 둥근 바닥이 턱 모서리를 타고 올라가서
/// 계단 로직을 꺼도 통과한다 — 그러면 이 테스트가 계단을 증명하지 않는다(실제로 확인했다).
/// 평평한 바닥은 수직면에서 순수 수평 법선만 얻으므로 오직 계단 경로로만 오를 수 있다.
#[test]
fn character_steps_up_low_ledges_but_not_high_walls() {
    let step_h = 0.25;
    let mut low = char_world();
    low.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(1.0, step_h / 2.0, 5.0),
            radius: 0.0,
        },
        pose(2.0, step_h / 2.0, 0.0),
    ));
    let mut c = box_char(0.0, 0.55, 0.0);
    for _ in 0..6 {
        c.move_and_slide(&low, Vec3::new(0.5, 0.0, 0.0), QueryFilter::ALL)
            .expect("이동");
    }
    assert!(c.position.x > 1.5, "낮은 턱을 못 넘었다: {:?}", c.position);
    assert!(
        c.position.y > 0.55 + step_h * 0.5,
        "턱 위로 올라서지 못했다: y={}",
        c.position.y
    );

    // 같은 장면인데 턱만 높으면 막혀야 한다.
    let mut high = char_world();
    high.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(1.0, 1.0, 5.0),
            radius: 0.0,
        },
        pose(2.0, 1.0, 0.0),
    ));
    let mut c2 = box_char(0.0, 0.55, 0.0);
    for _ in 0..6 {
        c2.move_and_slide(&high, Vec3::new(0.5, 0.0, 0.0), QueryFilter::ALL)
            .expect("이동");
    }
    assert!(c2.position.x < 1.3, "높은 벽을 올라갔다: {:?}", c2.position);
}

/// 같은 입력이면 같은 결과 — 캐릭터도 L0 규약 안에 있다.
#[test]
fn character_movement_is_deterministic() {
    let mut w = char_world();
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.2, 2.0, 5.0),
            radius: 0.0,
        },
        pose(1.5, 2.0, 0.0),
    ));
    let run = || {
        let mut c = capsule_char(0.0, 0.8, 0.0);
        let mut path = Vec::new();
        for i in 0..20 {
            let d = Vec3::new(0.2, 0.0, 0.05 * (i % 3) as f32);
            let r = c.move_and_slide(&w, d, QueryFilter::ALL).expect("이동");
            path.push((
                r.position.x.to_bits(),
                r.position.y.to_bits(),
                r.position.z.to_bits(),
                r.on_ground,
            ));
        }
        path
    };
    assert_eq!(run(), run(), "같은 입력이 다른 경로를 냈다");
}

/// 캐릭터 설정이 **매 호출 검증**된다 (BLOCKER: 음수 skin이면 요청보다 더 간다).
#[test]
fn character_rejects_a_broken_configuration() {
    let mut w = char_world();
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(0.2, 2.0, 5.0),
            radius: 0.0,
        },
        pose(1.5, 2.0, 0.0),
    ));
    let mut c = capsule_char(0.0, 0.8, 0.0);
    c.skin = -10.0;
    assert_eq!(
        c.move_and_slide(&w, Vec3::new(10.0, 0.0, 0.0), QueryFilter::ALL)
            .err(),
        Some(ApiError::InvalidValue),
        "음수 skin이 통과하면 벽 반대편으로 순간이동한다"
    );
    let mut c2 = capsule_char(0.0, 0.8, 0.0);
    c2.up = Vec3::new(0.0, 3.0, 0.0); // 단위가 아니다
    assert!(c2
        .move_and_slide(&w, Vec3::new(1.0, 0.0, 0.0), QueryFilter::ALL)
        .is_err());
    // 비볼록 셰이프는 스윕 주체가 될 수 없다
    let mut c3 = capsule_char(0.0, 0.8, 0.0);
    c3.shape = Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자");
    assert!(c3
        .move_and_slide(&w, Vec3::new(1.0, 0.0, 0.0), QueryFilter::ALL)
        .is_err());
    assert!(w
        .sweep_shape(
            &Shape::heightfield(vec![0.0; 4], 2, 2, 1.0).expect("격자"),
            pose(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            QueryFilter::ALL
        )
        .is_err());
}

// ===== 확인검수 rev_7efe4e35 회귀 =====

/// **시작부터 벽에 묻혀 있으면 이동하지 않는다** (BLOCKER: 스윕이 None을 내 통과했다).
#[test]
fn character_that_starts_overlapped_does_not_pass_through() {
    let mut w = char_world();
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(1.0, 2.0, 5.0),
            radius: 0.0,
        },
        pose(0.0, 2.0, 0.0),
    ));
    // 벽 한가운데에서 시작(스폰·텔레포트·움직이는 지형이 만드는 상황)
    let mut c = capsule_char(0.0, 1.0, 0.0);
    let r = c
        .move_and_slide(&w, Vec3::new(5.0, 0.0, 0.0), QueryFilter::ALL)
        .expect("오류가 아니라 결과로 알린다");
    assert!(r.started_overlapped, "겹침을 감지하지 못했다");
    assert_eq!(r.position, Vec3::new(0.0, 1.0, 0.0), "겹친 채로 이동했다");
    assert_eq!(r.remaining, Vec3::new(5.0, 0.0, 0.0));
    // 정상 위치에서는 그대로 동작한다(대조군)
    let mut ok = capsule_char(3.0, 0.8, 0.0);
    let r2 = ok
        .move_and_slide(&w, Vec3::new(0.5, 0.0, 0.0), QueryFilter::ALL)
        .expect("이동");
    assert!(!r2.started_overlapped);
}
