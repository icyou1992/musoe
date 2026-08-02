//! **혼합 장면 브로드페이즈 계측** — 정착한 더미 위에 소수가 떨어지는 구성.
//!
//! 외부 엔진 대조에서 이 장면이 이제 가장 큰 열세다(rapier 대비 2.2~2.5배). 프로파일은
//! `broad`가 지배적인데(240~290us, narrow 16 · solve 84), 이상한 점은 **낙하체가 5개든
//! 100개든 broad가 같다**는 것이다. 즉 비용이 "움직인 물체 수"가 아니라 다른 것에 붙어 있다.
//!
//! 코드를 읽으면 증분 경로(`incremental_pairs_bounded`)는 움직인 물체 R만 트리에 질의하지만,
//! 마지막에 **직전 페어 목록 전체를 훑어 fresh와 병합**한다. 정착 더미의 페어가 24,810개면
//! R이 5개여도 24,810번 복사·판정이 돈다.
//!
//! 이 벤치는 그 가설을 **정착 규모를 바꿔** 검증한다: broad가 페어 수에 비례하고 낙하체 수에
//! 무관하면 병합이 지배한다. 그러면 상한은 "병합을 없앤 값"이고, 그게 유의미할 때만 구조를 바꾼다.
//!
//! 실행: cargo run --release -p musoe-bench --bin mixedbp -- [steps]

use musoe_collide::{Pose, Shape};
use musoe_dynamics::{Body, World, WorldParams};
use musoe_math::{Quat, Vec3};
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

fn pose(x: f32, y: f32, z: f32) -> Pose {
    Pose {
        pos: Vec3::new(x, y, z),
        rot: Quat::IDENTITY,
    }
}

/// 구 `n`개를 격자로 쌓아 정착시킨 뒤, 낙하체 `k`개를 위에 띄운 월드.
/// 정착까지 돌린 상태로 반환하므로 측정 구간은 "정착 + 소수 낙하"만 잰다.
fn mixed(n: usize, k: usize) -> World {
    let mut w = World::new(WorldParams::default());
    w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(60.0, 1.0, 60.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let side = (n as f32).cbrt().ceil() as usize;
    for i in 0..n {
        let (ix, iy, iz) = (i % side, (i / side) % side, i / (side * side));
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.5 },
            pose(
                ix as f32 * 1.01 - side as f32 * 0.5,
                0.5 + iy as f32 * 1.01,
                iz as f32 * 1.01 - side as f32 * 0.5,
            ),
            1.0,
        ));
    }
    // 정착 — 더미가 전부 잠들 때까지.
    for _ in 0..400 {
        w.step(DT);
    }
    // 낙하체는 더미 **위**에 띄운다(닿기 전까지 페어를 만들지 않는다).
    for j in 0..k {
        w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.4 },
            pose(j as f32 * 1.5 - 8.0, 40.0 + (j / 8) as f32 * 1.5, 0.0),
            1.0,
        ));
    }
    w
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    println!("혼합 장면 브로드페이즈 — 정착 더미 n개 + 낙하체 k개, {steps}스텝 평균");
    println!(
        "{:>6} {:>4} {:>8} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "n", "k", "페어", "broad us", "narrow us", "solve us", "ms/step", "증분"
    );
    for n in [400usize, 800, 1600, 3000] {
        for k in [5usize, 100] {
            let mut w = mixed(n, k);
            // 워밍업 — 낙하체가 떨어지는 동안 증분 경로가 안정되게.
            for _ in 0..30 {
                w.step(DT);
            }
            let mut acc = [0u64; 5];
            let mut incr = 0usize;
            let t = Instant::now();
            for _ in 0..steps {
                w.step(DT);
                let d = w.diagnostics();
                for (a, v) in acc.iter_mut().zip(d.phase_ns.iter()) {
                    *a += v;
                }
                incr += d.bp_incremental as usize;
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
            let us = |x: u64| x as f64 / 1e3 / steps as f64;
            let d = w.diagnostics();
            println!(
                "{n:>6} {k:>4} {:>8} {:>9.0} {:>9.0} {:>9.0} {ms:>9.3} {:>4}/{steps}",
                d.pairs,
                us(acc[1]),
                us(acc[2]),
                us(acc[3]),
                incr
            );
        }
    }
    println!(
        "\n판독: broad가 **페어 수에 비례하고 k에 무관**하면 직전 목록 전체를 훑는 병합이 지배한다."
    );
}
