//! 스텝당 힙 할당 계측 — 설계 §6 "스텝 중 힙 할당 0" 계약의 게이트.
//!
//! ```bash
//! cargo run --release -p maru-bench --bin allocs                       # 기본 두 장면
//! cargo run --release -p maru-bench --bin allocs -- pyramid 40 80 20   # 장면 하나
//! cargo run --release -p maru-bench --bin allocs -- all 0 80 20 nosleep 3400   # 래칫
//! ```
//!
//! 인자: `[scene] [count] [warmup] [measure] [sleep|nosleep] [max_per_step]`
//! - `scene` = `pyramid` | `spheres` | `all`(기본)
//! - `count` = 피라미드 단수 / 구 개수 (`all`이면 무시하고 40단·3000구)
//! - `max_per_step`을 주면 **래칫**이 된다 — 스텝당 할당(alloc+realloc)이 그 값을 넘으면
//!   비-0으로 종료한다. 계약이 0이 아닌 지금은 "지금보다 나빠지지 않는다"를 잠그는 용도다.
//!
//! **장면 기하는 브라우저·네이티브 벤치와 같다**(`maru-wasm --example abi_bench`,
//! `compare/src/main.rs`) — 할당 수치가 성능 수치와 같은 장면을 가리켜야 원인 귀속이 된다.
//!
//! 기본이 `nosleep`인 이유: 할당은 **깨어 있는 접촉의 수**에 붙으므로 계약을 시험하는 값은
//! 전원 활성의 최악값이다. 잠든 장면은 접촉이 없어 0에 가까워지고, 그건 계약이 지켜져서가
//! 아니라 일을 안 해서다. (속도 A/B는 반대로 수면 ON이 정본이다 — README '수면과의 경주'.)

use maru_bench::alloc_probe::{self, AllocStats};
use maru_collide::{Pose, Shape};
use maru_dynamics::{Body, BodyHandle, World, WorldParams};
use maru_math::{Quat, Vec3};

/// 이 바이너리만 계측 할당자로 돈다 — 시간 벤치에 넣으면 계측이 대상을 바꾼다.
#[global_allocator]
static ALLOC: alloc_probe::CountingAlloc = alloc_probe::CountingAlloc::new();

fn pose(x: f32, y: f32, z: f32) -> Pose {
    Pose {
        pos: Vec3::new(x, y, z),
        rot: Quat::IDENTITY,
    }
}

fn new_world(no_sleep: bool) -> World {
    let d = WorldParams::default();
    World::new(WorldParams {
        time_to_sleep: if no_sleep {
            maru_collide::MAX_INPUT
        } else {
            0.5
        },
        ..d
    })
}

/// 피라미드 — `abi_bench::build_pyramid`와 같은 기하(간격 1.02, 반크기 0.5).
fn build_pyramid(w: &mut World, tiers: usize) -> Vec<BodyHandle> {
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(200.0, 1.0, 40.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let mut h = Vec::new();
    for t in 0..tiers {
        let count = tiers - t;
        let y = 0.5 + t as f32;
        for i in 0..count {
            let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
            h.push(w.add_body(Body::new_dynamic(
                Shape::Box {
                    half: Vec3::new(0.5, 0.5, 0.5),
                    radius: 0.0,
                },
                pose(x, y, 0.0),
                1.0,
            )));
        }
    }
    h
}

/// 구 더미 — `abi_bench::build_spheres`와 같은 기하(간격 1.1, 반지름 0.5).
fn build_spheres(w: &mut World, n: usize) -> Vec<BodyHandle> {
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(30.0, 1.0, 30.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let side = (n as f32).cbrt().ceil() as usize;
    let mut h = Vec::new();
    'outer: for iy in 0..side * 3 {
        for ix in 0..side {
            for iz in 0..side {
                if h.len() >= n {
                    break 'outer;
                }
                let x = ix as f32 * 1.1 - side as f32 * 0.55;
                let z = iz as f32 * 1.1 - side as f32 * 0.55;
                let y = 0.55 + iy as f32 * 1.1;
                h.push(w.add_body(Body::new_dynamic(
                    Shape::Sphere { radius: 0.5 },
                    pose(x, y, z),
                    1.0,
                )));
            }
        }
    }
    h
}

struct Run {
    name: String,
    bodies: usize,
    threads: usize,
    stats: AllocStats,
    steps: usize,
    pairs: usize,
    max_penetration: f32,
    vmax: f32,
    finite: bool,
    asleep: usize,
    step_err: usize,
}

fn measure(
    name: &str,
    mut w: World,
    handles: Vec<BodyHandle>,
    warmup: usize,
    steps: usize,
    trace: Option<(usize, usize)>,
) -> Run {
    const DT: f32 = 1.0 / 60.0;
    let mut step_err = 0usize;
    for _ in 0..warmup {
        if w.try_step(DT).is_err() {
            step_err += 1;
        }
    }
    // ── 측정 창: 여기서는 출력·포맷팅·수집을 하지 않는다(자기 할당 혼입 방지) ──
    alloc_probe::reset();
    if let Some((bucket, max)) = trace {
        alloc_probe::configure_trace(Some(bucket), max);
    }
    for _ in 0..steps {
        if w.try_step(DT).is_err() {
            step_err += 1;
        }
    }
    let stats = alloc_probe::snapshot();
    alloc_probe::configure_trace(None, 0);
    // ── 창 닫음. 아래는 얼마든지 할당해도 된다 ──
    let d = w.diagnostics();
    let mut vmax = 0.0f32;
    let mut finite = true;
    let mut asleep = 0usize;
    for h in &handles {
        if !w.is_alive(*h) {
            continue;
        }
        let b = w.body(*h);
        let v = b.linvel().length();
        finite &= b.pose().pos.y.is_finite() && v.is_finite();
        vmax = vmax.max(v);
        asleep += w.is_asleep(*h) as usize;
    }
    Run {
        name: name.to_string(),
        bodies: w.body_count(),
        threads: w.threads(),
        stats,
        steps,
        pairs: d.pairs,
        max_penetration: d.max_penetration,
        vmax,
        finite,
        asleep,
        step_err,
    }
}

fn report(r: &Run) {
    let n = r.steps as f64;
    let s = &r.stats;
    println!(
        "{:<14} bodies={:<5} pairs={:<6} threads={}  steps={}",
        r.name, r.bodies, r.pairs, r.threads, r.steps
    );
    println!(
        "  per step: gets {:.1} (alloc {:.1} + realloc {:.1})  free {:.1}  bytes_out {:.1} KiB  net {:+.0} B",
        s.gets() as f64 / n,
        s.allocs as f64 / n,
        s.reallocs as f64 / n,
        s.deallocs as f64 / n,
        s.bytes_out as f64 / n / 1024.0,
        s.net_bytes() as f64 / n,
    );
    println!("  size buckets/step: {}", alloc_probe::hist_line(s, n, 6));
    println!(
        "  scene check: {}  vmax={:.4}  max_pen={:.4} m  asleep={}  step_err={}",
        if r.finite { "finite" } else { "NON-FINITE" },
        r.vmax,
        r.max_penetration,
        r.asleep,
        r.step_err,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scene = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let warmup: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(80);
    let steps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let no_sleep = args.get(5).map(|s| s != "sleep").unwrap_or(true);
    let max_per_step: Option<f64> = args.get(6).and_then(|s| s.parse().ok());

    let threads_override: Option<usize> = std::env::var("MARU_THREADS")
        .ok()
        .and_then(|v| v.parse().ok());

    // 추적 모드: 지목된 버킷의 할당 **호출 스택**을 찍는다. 크기 분포는 정황이고 스택은 근거다.
    // 창 안에서 출력하므로 이 모드의 계수값은 오염된다(추적용이지 측정용이 아니다).
    let trace = std::env::var("MARU_ALLOC_TRACE_BUCKET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|b| {
            let max = std::env::var("MARU_ALLOC_TRACE_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3);
            (b, max)
        });
    if let Some((b, max)) = trace {
        println!(
            "  TRACE MODE: bucket {b} (≤{} B) · 최대 {max}건 — 계수값은 오염된다",
            alloc_probe::bucket_upper_bytes(b)
        );
    }

    println!(
        "maru alloc probe — sleep={}  warmup={warmup}  measure={steps}  (설계 §6: 스텝 중 할당 0)",
        if no_sleep { "OFF (worst case)" } else { "ON" }
    );

    let mut jobs: Vec<(String, usize, bool)> = Vec::new();
    match scene {
        "pyramid" => jobs.push((
            format!("pyramid{}", if count == 0 { 40 } else { count }),
            if count == 0 { 40 } else { count },
            true,
        )),
        "spheres" => jobs.push((
            format!("spheres{}", if count == 0 { 3000 } else { count }),
            if count == 0 { 3000 } else { count },
            false,
        )),
        _ => {
            jobs.push(("pyramid40".to_string(), 40, true));
            jobs.push(("spheres3000".to_string(), 3000, false));
        }
    }

    let mut worst = 0.0f64;
    for (name, n, is_pyramid) in jobs {
        let mut w = new_world(no_sleep);
        if let Some(t) = threads_override {
            w.set_threads(t);
        }
        let handles = if is_pyramid {
            build_pyramid(&mut w, n)
        } else {
            build_spheres(&mut w, n)
        };
        let r = measure(&name, w, handles, warmup, steps, trace);
        report(&r);
        worst = worst.max(r.stats.gets() as f64 / r.steps as f64);
    }

    if let Some(max) = max_per_step {
        println!("ratchet: worst {worst:.1} gets/step  limit {max:.1}");
        if worst > max {
            println!("ALLOC RATCHET FAILED — 스텝당 할당이 기록된 상한을 넘었다");
            std::process::exit(1);
        }
        println!("ALLOC RATCHET PASSED");
    }
}
