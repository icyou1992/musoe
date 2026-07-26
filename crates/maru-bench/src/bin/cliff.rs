//! 서브스텝 절벽(5=폭발 / 6=안정)의 **첫 발산 프레임까지** 프레임별 궤적을 찍는다.
//!
//! 지금까지 배제한 것: GS 제약 순서 · stale effective mass · h·f(CFL) 비율 ·
//! SAT→GJK 폴백. 남은 후보는 "잔차가 덜 줄면 박스가 미세 회전해 4점 면접촉이
//! 다른 참조면/에지로 바뀌고, warm-start 계보가 끊겨 누적 임펄스가 폐기·재생성되며
//! 에너지가 주입된다"는 것이다. 그 후보는 `warm_hits/warm_total`로 직접 관측된다 —
//! 계보가 끊기면 warm 비율이 먼저 무너져야 한다.
//!
//! 판별: **warm 비율 붕괴가 에너지 상승보다 먼저면** 계보 단절이 기폭제이고,
//! 에너지가 먼저 오르면 계보 단절은 결과다.
//!
//! 실행: cargo run --release -p maru-bench --bin cliff -- [substeps] [steps] [tiers]

use maru_collide::{Pose, Shape};
use maru_dynamics::{Body, BodyHandle, World, WorldParams};
use maru_math::{Quat, Vec3};

fn main() {
    let substeps: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1200);
    let tiers: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let d = WorldParams::default();
    let mut w = World::new(WorldParams {
        substeps,
        // 수면이 들면 관측이 끝난다 — 절벽은 활성 구간의 사건이다.
        sleep_linear_threshold: 0.0,
        sleep_angular_threshold: 0.0,
        ..d
    });
    let half = 0.5f32;
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
    let mut handles = Vec::new();
    for tier in 0..tiers {
        let count = tiers - tier;
        let y = half + tier as f32 * 2.0 * half;
        let x0 = -(count as f32 - 1.0) * half;
        for i in 0..count {
            handles.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(half, half, half),
                    radius: 0.0,
                },
                Pose {
                    pos: Vec3::new(x0 + i as f32 * 2.0 * half, y, 0.0),
                    rot: Quat::IDENTITY,
                },
                1.0,
            )));
        }
    }

    let energy = |w: &World, hs: &[BodyHandle]| -> f64 {
        let mut e = 0.0f64;
        for h in hs {
            let b = w.body(*h);
            let m = 1.0 / b.inv_mass() as f64;
            e += 0.5 * m * b.linvel().length_sq() as f64;
            let ii = b.inv_inertia_local();
            e += 0.5 * (1.0 / ii.x as f64) * b.angvel().length_sq() as f64;
            e += m * 9.81 * b.pose().pos.y as f64;
        }
        e
    };
    let vmax = |w: &World, hs: &[BodyHandle]| -> (f32, f32) {
        let (mut vl, mut va) = (0.0f32, 0.0f32);
        for h in hs {
            let b = w.body(*h);
            vl = vl.max(b.linvel().length());
            va = va.max(b.angvel().length());
        }
        (vl, va)
    };

    println!("substeps={substeps} tiers={tiers} bodies={}", handles.len());
    println!(
        "{:>5} {:>7} {:>7} {:>6} {:>9} {:>8} {:>8} {:>12} {:>9}",
        "step", "warm", "hits", "pairs", "max_pen", "vlin", "vang", "E", "dE"
    );
    let mut prev_e = energy(&w, &handles);
    let e0 = prev_e;
    let mut diverged = 0usize;
    for i in 0..steps {
        let _ = w.try_step(1.0 / 60.0);
        let d = w.diagnostics();
        let (vl, va) = vmax(&w, &handles);
        let e = energy(&w, &handles);
        let warm = if d.warm_total > 0 {
            d.warm_hits as f64 / d.warm_total as f64
        } else {
            -1.0
        };
        // 폭발 시작 = 선속도가 정착 임계를 두 자릿수 넘어선 첫 프레임
        if diverged == 0 && vl > 5.0 {
            diverged = i + 1;
            println!("  ---- 발산 시작 (vlin>5) : step {} ----", i + 1);
        }
        // 초반 정착기 + 발산 부근을 촘촘히, 중간은 성기게
        let dense = i < 40 || (diverged > 0 && i + 1 >= diverged.saturating_sub(30));
        if dense || i % 50 == 49 {
            println!(
                "{:>5} {:>7.3} {:>7} {:>6} {:>9.5} {:>8.3} {:>8.3} {:>12.1} {:>9.1}",
                i + 1,
                warm,
                d.warm_total,
                d.pairs,
                d.max_penetration,
                vl,
                va,
                e,
                e - prev_e
            );
        }
        prev_e = e;
        if diverged > 0 && i + 1 > diverged + 20 {
            break;
        }
    }
    println!("E0={e0:.1}  최종 E={prev_e:.1}  발산 시작 step={diverged}");
}
