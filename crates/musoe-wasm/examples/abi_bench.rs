//! 브라우저 벤치(web/bench.html)의 **네이티브 쌍둥이** — 같은 씬, 같은 C ABI, 같은
//! 기본 파라미터. 두 결과의 비율이 곧 "브라우저 세금"이다.
//!
//! ```bash
//! cargo run --release -p musoe-wasm --example abi_bench
//! ```

use musoe_wasm::*;
use std::time::Instant;

const IDENT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

unsafe fn add(w: *mut WasmWorld, body_type: u32, shape: u32, dims: [f32; 4], pos: [f32; 3]) -> u64 {
    musoe_add_body(
        w, body_type, shape, dims[0], dims[1], dims[2], dims[3], pos[0], pos[1], pos[2], IDENT[0],
        IDENT[1], IDENT[2], IDENT[3], 1.0,
    )
}

unsafe fn build_pyramid(w: *mut WasmWorld, tiers: usize) {
    add(
        w,
        BODY_STATIC,
        SHAPE_BOX,
        [200.0, 1.0, 40.0, 0.0],
        [0.0, -1.0, 0.0],
    );
    for t in 0..tiers {
        let count = tiers - t;
        let y = 0.5 + t as f32;
        for i in 0..count {
            let x = (i as f32 - (count as f32 - 1.0) * 0.5) * 1.02;
            add(
                w,
                BODY_DYNAMIC,
                SHAPE_BOX,
                [0.5, 0.5, 0.5, 0.0],
                [x, y, 0.0],
            );
        }
    }
}

unsafe fn build_spheres(w: *mut WasmWorld, n: usize) {
    add(
        w,
        BODY_STATIC,
        SHAPE_BOX,
        [30.0, 1.0, 30.0, 0.0],
        [0.0, -1.0, 0.0],
    );
    let side = (n as f32).cbrt().ceil() as usize;
    let mut count = 0;
    'outer: for iy in 0..side * 3 {
        for ix in 0..side {
            for iz in 0..side {
                if count >= n {
                    break 'outer;
                }
                let x = ix as f32 * 1.1 - side as f32 * 0.55;
                let z = iz as f32 * 1.1 - side as f32 * 0.55;
                let y = 0.55 + iy as f32 * 1.1;
                add(
                    w,
                    BODY_DYNAMIC,
                    SHAPE_SPHERE,
                    [0.5, 0.0, 0.0, 0.0],
                    [x, y, z],
                );
                count += 1;
            }
        }
    }
}

unsafe fn run_scene(name: &str, build: impl FnOnce(*mut WasmWorld), steps: usize) {
    let w = musoe_world_create(0.0, -9.81, 0.0, 0);
    assert!(!w.is_null());
    build(w);
    let dt = 1.0 / 60.0;
    let mut times = vec![0.0f64; steps];
    for (s, slot) in times.iter_mut().enumerate() {
        let t0 = Instant::now();
        let r = musoe_step(w, dt);
        *slot = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(r, OK, "{name}: step {s} -> {r}");
    }
    let n = musoe_slot_count(w) as usize;
    let buf = core::slice::from_raw_parts(musoe_transforms_ptr(w), n * TRANSFORM_STRIDE);
    let (mut lowest, mut top) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 1..n {
        let y = buf[i * TRANSFORM_STRIDE + 1];
        lowest = lowest.min(y);
        top = top.max(y);
    }
    musoe_world_destroy(w);
    let full: f64 = times.iter().sum::<f64>() / steps as f64;
    let settled: f64 = times[steps / 2..].iter().sum::<f64>() / (steps - steps / 2) as f64;
    println!(
        "{name:<12} bodies={:<5} full={full:.3} ms/step  settled={settled:.4} ms/step  \
         lowest={lowest:.3} top={top:.3}",
        n - 1
    );
}

fn main() {
    unsafe {
        run_scene("pyramid20", |w| build_pyramid(w, 20), 240);
        run_scene("pyramid40", |w| build_pyramid(w, 40), 240);
        run_scene("spheres500", |w| build_spheres(w, 500), 240);
        run_scene("spheres3000", |w| build_spheres(w, 3000), 240);
    }
}
