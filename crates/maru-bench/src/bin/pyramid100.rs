//! M2 수용 벤치: 피라미드 100단(5050 박스) 안정성 — release 전용.
//! 실행: cargo run --release -p maru-bench --bin pyramid100

use maru_collide::{Pose, Shape};
use maru_dynamics::{Body, BodyHandle, World, WorldParams};
use maru_math::{Quat, Vec3};
use std::time::Instant;

fn main() {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(120.0, 1.0, 20.0),
            radius: 0.0,
        },
        Pose {
            pos: Vec3::new(0.0, -1.0, 0.0),
            rot: Quat::IDENTITY,
        },
    ));
    let tiers = 100usize;
    let half = 0.5f32;
    for tier in 0..tiers {
        let count = tiers - tier;
        let y = half + tier as f32 * 2.0 * half;
        let x0 = -(count as f32 - 1.0) * half;
        for i in 0..count {
            w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(half, half, half),
                    radius: 0.0,
                },
                Pose {
                    pos: Vec3::new(x0 + i as f32 * 2.0 * half, y, 0.0),
                    rot: Quat::IDENTITY,
                },
                1.0,
            ));
        }
    }
    println!("bodies={} (tiers={tiers})", w.body_count());
    let t = Instant::now();
    let steps = 600;
    for i in 0..steps {
        w.step(1.0 / 60.0);
        if i % 100 == 99 {
            println!(
                "  step {}/{steps}  {:.1}s elapsed",
                i + 1,
                t.elapsed().as_secs_f64()
            );
        }
    }
    let ms_per_step = t.elapsed().as_secs_f64() * 1e3 / steps as f64;

    let mut vmax = 0.0f32;
    let mut top_y = f32::NEG_INFINITY;
    let mut finite = true;
    for i in 1..w.body_count() {
        let b = w.body(BodyHandle {
            index: i as u32,
            generation: 0,
        });
        finite &= b.pose.pos.y.is_finite() && b.linvel.length().is_finite();
        vmax = vmax.max(b.linvel.length());
        top_y = top_y.max(b.pose.pos.y);
    }
    println!(
        "{ms_per_step:.0} ms/step  finite={finite}  vmax={vmax:.4}  top_y={top_y:.2} (expect 99.5)"
    );
    assert!(finite, "발산");
    assert!(vmax < 0.2, "미정착 vmax={vmax}");
    assert!((top_y - 99.5).abs() < 0.3, "붕괴 top_y={top_y}");
    println!("PYRAMID 100 TIERS STABLE — M2 acceptance PASSED");
}
