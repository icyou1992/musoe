//! M2 수용 벤치: 피라미드 100단(5050 박스) 안정성 — release 전용.
//! 실행: cargo run --release -p maru-bench --bin pyramid100

use maru_collide::{Pose, Shape};
use maru_dynamics::{Body, BodyHandle, World, WorldParams};
use maru_math::{Quat, Vec3};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let substeps: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(600);
    let tiers: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    // arg4: sleep 비활성화 플래그 (진단용)
    let no_sleep = args.get(4).map(|s| s == "nosleep").unwrap_or(false);
    let hertz: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let d = WorldParams::default();
    let mut w = World::new(WorldParams {
        substeps,
        time_to_sleep: if no_sleep { 1.0e9 } else { 0.5 },
        solver: maru_dynamics::SolverParams {
            contact_hertz: hertz,
            ..d.solver
        },
        ..d
    });
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
    println!(
        "bodies={} tiers={tiers} substeps={substeps} steps={steps} hertz={hertz}",
        w.body_count()
    );
    let t = Instant::now();
    for i in 0..steps {
        w.step(1.0 / 60.0);
        if i % 100 == 99 || i == 4 {
            println!(
                "  step {}/{steps}  {:.1}s  warm-start {}/{} ({:.0}%)",
                i + 1,
                t.elapsed().as_secs_f64(),
                w.warm_hits,
                w.warm_total,
                100.0 * w.warm_hits as f64 / w.warm_total.max(1) as f64
            );
            println!("      max_penetration={:.4} m", w.max_penetration);
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
    let expect_top = tiers as f32 - 0.5;
    let ok = finite && vmax < 0.2 && (top_y - expect_top).abs() < 0.3;
    println!(
        "{}  (expect top_y={expect_top})",
        if ok {
            "STABLE — PASSED"
        } else {
            "UNSTABLE — FAILED"
        }
    );
}
