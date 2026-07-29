//! 스택 안정 봉투 벤치 — release 전용.
//! 실행: cargo run --release -p maru-bench --bin pyramid100 -- [substeps] [steps] [tiers]
//! M2 수용 기준은 40단(실측 재보정, README 표 참조). 100단은 현 솔버 범위 밖.

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
    // arg6: 수면 선속도 임계 (절벽 = 잔류속도가 이 임계를 건너는 지점인지 가리는 실험용)
    let sleep_lin: f32 = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(WorldParams::default().sleep_linear_threshold);
    let d = WorldParams::default();
    let mut w = World::new(WorldParams {
        substeps,
        time_to_sleep: if no_sleep {
            maru_collide::MAX_INPUT
        } else {
            0.5
        },
        sleep_linear_threshold: sleep_lin,
        sleep_angular_threshold: sleep_lin * 1.667,
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
    // 스레드 수 오버라이드 — 병렬 A/B용. 기본은 엔진의 자동 감지 값이다.
    if let Some(n) = std::env::var("MARU_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        w.set_threads(n);
    }
    println!(
        "bodies={} tiers={tiers} substeps={substeps} steps={steps} hertz={hertz} threads={}",
        w.body_count(),
        w.threads()
    );
    let t = Instant::now();
    let mut acc_phase = [0u64; 5];
    let mut acc_solve = [0u64; 7];
    let mut acc_colors = 0usize;
    let mut acc_work = 0usize;
    for i in 0..steps {
        w.step(1.0 / 60.0);
        {
            let d = w.diagnostics();
            for (a, v) in acc_phase.iter_mut().zip(d.phase_ns.iter()) {
                *a += v;
            }
            for (a, v) in acc_solve.iter_mut().zip(d.solve_ns.iter()) {
                *a += v;
            }
            acc_colors += d.colors;
            acc_work += d.solve_work;
        }
        if i % 100 == 99 || i == 4 {
            println!(
                "  step {}/{steps}  {:.1}s  warm-start {}/{} ({:.0}%)",
                i + 1,
                t.elapsed().as_secs_f64(),
                w.diagnostics().warm_hits,
                w.diagnostics().warm_total,
                100.0 * w.diagnostics().warm_hits as f64 / w.diagnostics().warm_total.max(1) as f64
            );
            println!(
                "      max_penetration={:.4} m",
                w.diagnostics().max_penetration
            );
        }
    }
    let ms_per_step = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    {
        // 전 스텝 평균 귀속 — 마지막 한 스텝만 보면 잡음이 판정을 가른다.
        let n = steps as f64;
        let us = |x: u64| x as f64 / 1e3 / n;
        println!(
            "  avg phase us: aabb {:.0} broad {:.0} narrow {:.0} solve {:.0} post {:.0} | colors {:.1} work {:.0}",
            us(acc_phase[0]), us(acc_phase[1]), us(acc_phase[2]), us(acc_phase[3]), us(acc_phase[4]),
            acc_colors as f64 / n, acc_work as f64 / n
        );
        println!(
            "  avg solve us: intv {:.0} refresh {:.0} warm {:.0} bias {:.0} relax {:.0} intp {:.0} joint {:.0}",
            us(acc_solve[0]), us(acc_solve[1]), us(acc_solve[2]), us(acc_solve[3]),
            us(acc_solve[4]), us(acc_solve[5]), us(acc_solve[6])
        );
    }
    {
        // 마지막 스텝의 단계별 귀속 — 병렬 A/B는 총시간만 보면 어디서 먹혔는지 모른다.
        let d = w.diagnostics();
        let us = |n: u64| n as f64 / 1e3;
        println!(
            "  phase us: aabb {:.0} broad {:.0} narrow {:.0} solve {:.0} post {:.0} | np_threads {}",
            us(d.phase_ns[0]),
            us(d.phase_ns[1]),
            us(d.phase_ns[2]),
            us(d.phase_ns[3]),
            us(d.phase_ns[4]),
            d.narrow_threads
        );
        println!(
            "  solve us: intv {:.0} refresh {:.0} warm {:.0} bias {:.0} relax {:.0} intp {:.0} joint {:.0}",
            us(d.solve_ns[0]),
            us(d.solve_ns[1]),
            us(d.solve_ns[2]),
            us(d.solve_ns[3]),
            us(d.solve_ns[4]),
            us(d.solve_ns[5]),
            us(d.solve_ns[6])
        );
    }

    let mut vmax = 0.0f32;
    let mut top_y = f32::NEG_INFINITY;
    let mut finite = true;
    for i in 1..w.body_count() {
        let b = w.body(BodyHandle {
            index: i as u32,
            generation: 0,
        });
        finite &= b.pose().pos.y.is_finite() && b.linvel().length().is_finite();
        vmax = vmax.max(b.linvel().length());
        top_y = top_y.max(b.pose().pos.y);
    }
    // 단(tier)별 속도 분포 — 지터가 하중 쪽(아래)인지 전파 끝(위)인지 가른다
    {
        let mut per_tier: Vec<(f32, usize)> = vec![(0.0, 0); tiers];
        let mut idx = 0usize;
        for (t, slot) in per_tier.iter_mut().enumerate() {
            let count = tiers - t;
            for _ in 0..count {
                let h = BodyHandle {
                    index: (idx + 1) as u32,
                    generation: 0,
                };
                if w.is_alive(h) {
                    let v = w.body(h).linvel().length();
                    slot.0 = slot.0.max(v);
                    slot.1 += w.is_asleep(h) as usize;
                }
                idx += 1;
            }
        }
        let mut line = String::new();
        for (t, (vmax_t, slept)) in per_tier.iter().enumerate() {
            if t % 3 == 0 || *vmax_t > 0.01 {
                line.push_str(&format!("t{t}:{vmax_t:.3}/{slept} "));
            }
        }
        println!("tier vmax/slept: {line}");
    }
    let d = w.diagnostics();
    let warm_ratio = if d.warm_total > 0 {
        d.warm_hits as f32 / d.warm_total as f32
    } else {
        -1.0
    };
    println!("warm={warm_ratio:.3} ({}/{})", d.warm_hits, d.warm_total);
    println!(
        "{ms_per_step:.0} ms/step  finite={finite}  vmax={vmax:.4}  top_y={top_y:.2}\n\
         pairs={} / budget={} ({:.1}%)  oob={}  vclamp={}  sat={}",
        d.pairs,
        d.pair_budget,
        100.0 * d.pairs as f32 / d.pair_budget.max(1) as f32,
        d.out_of_bounds,
        d.velocity_clamped,
        d.state_saturations
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
