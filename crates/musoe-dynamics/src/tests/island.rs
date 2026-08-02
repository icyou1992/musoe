//! 아일랜드 구조 계측 — **병렬 솔버 전략의 전제를 재는 테스트**.
//!

//! 아일랜드 구조 계측 — **병렬 솔버 전략의 전제를 재는 테스트**다.
//!
//! "아일랜드는 서로 독립이므로 아일랜드 단위 병렬은 배리어가 없다"는 명제는 참이지만,
//! 그것이 **이득**이 되는지는 장면의 아일랜드 분포에 달렸다. 2026-07-30 실측:
//!
//! | 장면 | 아일랜드 수 | 최대 크기 |
//! |---|---|---|
//! | 피라미드 40단 (820 동적) | **1** | 820 |
//! | 구 더미 3000 | 225 | 14 |
//!
//! 피라미드는 접촉으로 전원이 이어져 **하나의 아일랜드**다 — 우리가 rapier에 지는 바로 그
//! 장면에서 아일랜드 병렬은 병렬성이 0이다. 구 더미는 1.1 간격·반지름 0.5라 열끼리 0.1 틈이
//! 남아 수직 스택 225개(15×15)로 갈린다 — 여기서는 잘 갈린다.
//! 이 테스트는 그 전제가 바뀌면(예: 접촉 생성 규칙 변경) 알려주기 위해 남는다.

use crate::{Body, World, WorldParams};
use musoe_collide::{Pose, Shape};
use musoe_math::{Quat, Vec3};
use std::collections::HashMap;

fn pose(x: f32, y: f32, z: f32) -> Pose {
    Pose {
        pos: Vec3::new(x, y, z),
        rot: Quat::IDENTITY,
    }
}

/// (아일랜드 수, 최대 크기) — 아일랜드 밖(u32::MAX)은 세지 않는다.
fn island_stats(w: &World) -> (usize, usize) {
    let mut hist: HashMap<u32, usize> = HashMap::new();
    for i in 0..w.body_count() {
        let id = w.island_of_index(i);
        if id != u32::MAX {
            *hist.entry(id).or_default() += 1;
        }
    }
    (hist.len(), hist.values().copied().max().unwrap_or(0))
}

/// **wasm 레그에서는 건너뛴다.** 이 테스트는 3001바디 월드를 만들어 120스텝 돌리는데,
/// wasm 테스트는 전체 스위트가 **하나의 인스턴스**에서 순차로 돌아 디버그 빌드의 이 씬이
/// 메모리를 넘겨 스위트를 SIGSEGV로 죽인다(격리 실행은 통과 — 그래서 이 테스트 자체의
/// 결함이 아니라 레그의 예산 문제다). 아일랜드 구조는 타깃 무관한 그래프 성질이라
/// native 레그 하나로 충분하다.
#[test]
#[cfg_attr(target_arch = "wasm32", ignore)]
fn island_structure_of_target_scenes() {
    let d = WorldParams::default();
    // 피라미드 40단 — 접촉으로 전원이 이어져 아일랜드가 하나다.
    let mut w = World::new(WorldParams {
        time_to_sleep: musoe_collide::MAX_INPUT,
        ..d.clone()
    });
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(200.0, 1.0, 40.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    for t in 0..40 {
        let count = 40 - t;
        for i in 0..count {
            let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
            w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(x, 0.5 + t as f32, 0.0),
                1.0,
            ));
        }
    }
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let (count, max) = island_stats(&w);
    assert_eq!(
        count, 1,
        "피라미드는 하나의 아일랜드여야 한다 (전제가 바뀌었다)"
    );
    assert_eq!(max, 820, "동적 바디 전원이 그 아일랜드에 있어야 한다");

    // 구 더미 3000 — 수직 스택 225개로 갈린다(열 간 0.1 틈).
    let mut w = World::new(WorldParams {
        time_to_sleep: musoe_collide::MAX_INPUT,
        ..d
    });
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(30.0, 1.0, 30.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let n = 3000usize;
    let side = (n as f32).cbrt().ceil() as usize;
    let mut c = 0;
    'o: for iy in 0..side * 3 {
        for ix in 0..side {
            for iz in 0..side {
                if c >= n {
                    break 'o;
                }
                w.add_body(Body::new_dynamic(
                    Shape::Sphere { radius: 0.5 },
                    pose(
                        ix as f32 * 1.1 - side as f32 * 0.55,
                        0.55 + iy as f32 * 1.1,
                        iz as f32 * 1.1 - side as f32 * 0.55,
                    ),
                    1.0,
                ));
                c += 1;
            }
        }
    }
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let (count, max) = island_stats(&w);
    assert!(
        count > 100 && max <= 40,
        "구 더미는 잘게 갈려야 한다 (실측 225개·최대 14): count={count} max={max}"
    );
}
