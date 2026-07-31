//! **TriMesh 상한 계측** — "메시를 삼각형 바디 N개로 두는 것"이 얼마나 비싼가.
//!
//! `Shape::TriMesh`(내부 BVH)는 아직 없다. 그런데 하이트필드가 **이미 같은 구조**다:
//! 바디 하나가 내부 격자를 들고 접촉 순간에만 삼각형으로 펼친다. 그래서 같은 삼각형 수를
//! 두 형태로 놓고 재면 TriMesh가 벌어 줄 값의 **상한**이 나온다 — 짓기 전에.
//!
//! - (a) 삼각형 바디 N개: 브로드페이즈 트리에 N개가 들어가고 페어도 N에 비례한다.
//! - (b) 하이트필드 1개: 트리에 1개, 가지치기는 셰이프 내부에서.
//!
//! 두 씬의 **동적 물체는 동일**하다 — 차이는 지형의 표현뿐이다. 지형 삼각형은 평평하게
//! 두어 접촉 기하가 같도록 맞춘다(같은 물리를 재야 비교가 성립한다).
//!
//! 실행: cargo run --release -p maru-bench --bin meshcost -- [격자변] [steps] [동적바디]

use maru_collide::{Pose, Shape};
use maru_dynamics::{Body, BodyHandle, World, WorldParams};
use maru_math::{Quat, Vec3};
use std::time::Instant;

fn pose(x: f32, y: f32, z: f32) -> Pose {
    Pose {
        pos: Vec3::new(x, y, z),
        rot: Quat::IDENTITY,
    }
}

/// 동적 물체는 두 씬에서 **완전히 같게** 놓는다.
fn add_dynamics(w: &mut World, n: usize) -> Vec<BodyHandle> {
    let side = (n as f32).sqrt().ceil() as usize;
    let mut hs = Vec::new();
    for i in 0..n {
        let (ix, iz) = (i % side, i / side);
        hs.push(w.add_body(Body::new_dynamic(
            Shape::Sphere { radius: 0.35 },
            pose(ix as f32 * 1.2 + 1.0, 1.2, iz as f32 * 1.2 + 1.0),
            1.0,
        )));
    }
    hs
}

fn params() -> WorldParams {
    WorldParams {
        // 수면 OFF — 일감을 고정해야 두 씬의 비교가 성립한다.
        time_to_sleep: maru_collide::MAX_INPUT,
        ..WorldParams::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(150);
    let ndyn: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let cell = 1.0f32;
    let tris = side * side * 2;

    // (a) 삼각형 바디 — 두꺼운 hull(두께 0은 생성 경계가 거부한다).
    let mut wa = World::new(params());
    for iz in 0..side {
        for ix in 0..side {
            let (x0, z0) = (ix as f32 * cell, iz as f32 * cell);
            let (x1, z1) = (x0 + cell, z0 + cell);
            for t in 0..2 {
                let p = if t == 0 {
                    [
                        Vec3::new(x0, 0.0, z0),
                        Vec3::new(x1, 0.0, z0),
                        Vec3::new(x0, 0.0, z1),
                    ]
                } else {
                    [
                        Vec3::new(x1, 0.0, z0),
                        Vec3::new(x1, 0.0, z1),
                        Vec3::new(x0, 0.0, z1),
                    ]
                };
                wa.add_body(Body::new_static(
                    Shape::Hull {
                        points: p.to_vec(),
                        radius: 0.02,
                    },
                    pose(0.0, 0.0, 0.0),
                ));
            }
        }
    }
    let ha = add_dynamics(&mut wa, ndyn);

    // (b) 하이트필드 하나 — 같은 삼각형 수, 평평한 높이.
    let mut wb = World::new(params());
    let heights = vec![0.0f32; (side + 1) * (side + 1)];
    let hf = Shape::heightfield(heights, side as u32 + 1, side as u32 + 1, cell)
        .expect("하이트필드 생성");
    wb.add_body(Body::new_static(hf, pose(0.0, 0.0, 0.0)));
    let hb = add_dynamics(&mut wb, ndyn);

    let run = |w: &mut World, label: &str, hs: &[BodyHandle]| {
        for _ in 0..40 {
            w.step(1.0 / 60.0);
        }
        let mut acc = [0u64; 5];
        let t = Instant::now();
        for _ in 0..steps {
            w.step(1.0 / 60.0);
            let d = w.diagnostics();
            for (a, v) in acc.iter_mut().zip(d.phase_ns.iter()) {
                *a += v;
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        let us = |x: u64| x as f64 / 1e3 / steps as f64;
        let d = w.diagnostics();
        let mut lowest = f32::INFINITY;
        let mut finite = true;
        for h in hs {
            let b = w.body(*h);
            lowest = lowest.min(b.pose().pos.y);
            finite &= b.pose().pos.y.is_finite();
        }
        println!(
            "{label:<16} {ms:7.3} ms/step  aabb {:5.0} broad {:5.0} narrow {:5.0} solve {:5.0} post {:5.0} | \
             bodies {:5} pairs {:5}  lowest_y {lowest:.3} finite {finite}",
            us(acc[0]),
            us(acc[1]),
            us(acc[2]),
            us(acc[3]),
            us(acc[4]),
            w.body_count(),
            d.pairs,
        );
        ms
    };

    println!("지형 삼각형 {tris}개 ({side}×{side} 격자), 동적 구 {ndyn}개, 수면 OFF, {steps}스텝");
    // 교차 실행 — 순차로 한 번씩만 재면 기계 드리프트를 차이로 읽는다.
    let (mut a_best, mut b_best) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        a_best = a_best.min(run(&mut wa, "삼각형 바디 N개", &ha));
        b_best = b_best.min(run(&mut wb, "하이트필드 1개", &hb));
        b_best = b_best.min(run(&mut wb, "하이트필드 1개", &hb));
        a_best = a_best.min(run(&mut wa, "삼각형 바디 N개", &ha));
    }
    println!(
        "\nmin-of-6: 바디 {a_best:.3} ms vs 내부구조 {b_best:.3} ms → 내부 구조가 {:.2}배 빠름",
        a_best / b_best
    );
}
