//! 내로우페이즈 하위 단계 분해 — `narrow` 안에서 무엇이 지배하는지 실측한다.
//!
//! 구 더미 3000에서 narrow가 7.4ms인데 산술 추정은 ~1ms다. 7배 차이의 정체를
//! 추측으로 지목하지 않기 위한 계측기다. 실제 시뮬레이션을 돌려 대표 상태를 만든 뒤,
//! 그 상태에서 (a) 브로드페이즈 페어 산출 (b) `contact()`만 (c) 뚱뚱한 `Body` 배열
//! 무작위 접근만 을 따로 잰다.
//!
//! 실행: cargo run --release -p maru-bench --bin narrowsplit -- [n] [warmup]

use maru_collide::{for_each_contact, Aabb, Broadphase, Manifold, Pose, Shape, STATIC_BIT};

/// 볼록 쌍 하나의 매니폴드 — 공개 표면이 디스패처만이라 벤치도 그것을 쓴다(boxsplit와 동일).
fn contact(sa: &Shape, pa: &Pose, sb: &Shape, pb: &Pose, margin: f32) -> Option<Manifold> {
    let mut out = None;
    for_each_contact(sa, pa, sb, pb, margin, &mut |_, _, m| out = Some(m));
    out
}
use maru_dynamics::{Body, World, WorldParams};
use maru_math::{Quat, Vec3};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let warmup: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(240);

    // compare 하네스의 구 더미와 동일 구성 — 수면 OFF(처리량 모드).
    let d = WorldParams::default();
    let mut w = World::new(WorldParams {
        sleep_linear_threshold: 0.0,
        sleep_angular_threshold: 0.0,
        ..d
    });
    let pose = |x: f32, y: f32, z: f32| Pose {
        pos: Vec3::new(x, y, z),
        rot: Quat::IDENTITY,
    };
    let ground = w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(30.0, 1.0, 30.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ));
    let side = (n as f32).cbrt().ceil() as usize;
    let mut handles = vec![ground];
    'outer: for iy in 0..side * 3 {
        for ix in 0..side {
            for iz in 0..side {
                if handles.len() > n {
                    break 'outer;
                }
                handles.push(w.add_body(Body::new_dynamic(
                    Shape::Sphere { radius: 0.5 },
                    pose(
                        ix as f32 * 1.1 - side as f32 * 0.55,
                        0.55 + iy as f32 * 1.1,
                        iz as f32 * 1.1 - side as f32 * 0.55,
                    ),
                    1.0,
                )));
            }
        }
    }
    for _ in 0..warmup {
        let _ = w.try_step(1.0 / 60.0);
    }

    // 대표 상태에서 셰이프·포즈를 뽑는다 (엔진 내부와 같은 슬롯 순서).
    let shapes: Vec<Shape> = handles.iter().map(|h| w.body(*h).shape().clone()).collect();
    let poses: Vec<Pose> = handles.iter().map(|h| w.body(*h).pose()).collect();
    let margin = WorldParams::default().speculative_margin;

    let aabb_of = |i: usize| -> Aabb {
        let p = poses[i];
        let half = match shapes[i] {
            Shape::Sphere { radius } => Vec3::new(radius, radius, radius),
            Shape::Box { half, radius } => {
                // 회전한 박스의 보수적 AABB (정확한 값이 아니어도 계측 목적엔 충분)
                let e = half + Vec3::new(radius, radius, radius);
                let m = e.x.max(e.y).max(e.z);
                Vec3::new(m, m, m)
            }
            _ => Vec3::new(1.0, 1.0, 1.0),
        };
        let g = Vec3::new(margin, margin, margin);
        Aabb::from_center(p.pos, half + g)
    };
    let dynamic_items: Vec<(u32, Aabb)> =
        (1..handles.len()).map(|i| (i as u32, aabb_of(i))).collect();
    let static_items: Vec<(u32, Aabb)> = vec![(0, aabb_of(0))];
    let bp = Broadphase::build(dynamic_items, static_items);
    let pairs = bp.canonical_pairs_bounded(1 << 20).unwrap();

    let reps = 100;
    let best = |f: &mut dyn FnMut() -> u128| -> f64 {
        let mut b = u128::MAX;
        for _ in 0..reps {
            b = b.min(f());
        }
        b as f64 / 1000.0
    };

    // (a) contact()만 — 실제 매니폴드 생성 비용
    let mut hits = 0usize;
    let t_contact = best(&mut || {
        let mut h = 0usize;
        let t = Instant::now();
        for &(a, b_raw) in &pairs {
            let b = (b_raw & !STATIC_BIT) as usize;
            let a = a as usize;
            if let Some(m) = contact(&shapes[a], &poses[a], &shapes[b], &poses[b], margin) {
                h += !m.is_empty() as usize;
            }
        }
        let e = t.elapsed().as_nanos();
        hits = h;
        std::hint::black_box(h);
        e
    });

    // (a2) 셰이프 조합별로 분리 — 어떤 조합이 비용을 먹는지
    let ss: Vec<(u32, u32)> = pairs
        .iter()
        .copied()
        .filter(|(a, b)| {
            let b = (b & !STATIC_BIT) as usize;
            matches!(shapes[*a as usize], Shape::Sphere { .. })
                && matches!(shapes[b], Shape::Sphere { .. })
        })
        .collect();
    let sb: Vec<(u32, u32)> = pairs
        .iter()
        .copied()
        .filter(|p| !ss.contains(p) || false)
        .filter(|(a, b)| {
            let b = (b & !STATIC_BIT) as usize;
            !(matches!(shapes[*a as usize], Shape::Sphere { .. })
                && matches!(shapes[b], Shape::Sphere { .. }))
        })
        .collect();
    let time_set = |set: &Vec<(u32, u32)>| -> f64 {
        let mut b = u128::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let mut h = 0usize;
            for &(a, b_raw) in set {
                let bi = (b_raw & !STATIC_BIT) as usize;
                let ai = a as usize;
                if let Some(m) = contact(&shapes[ai], &poses[ai], &shapes[bi], &poses[bi], margin) {
                    h += !m.is_empty() as usize;
                }
            }
            let e = t.elapsed().as_nanos();
            std::hint::black_box(h);
            b = b.min(e);
        }
        b as f64 / 1000.0
    };
    let t_ss = time_set(&ss);
    let t_sb = time_set(&sb);
    let (n_ss, n_sb) = (ss.len(), sb.len());

    // (b) 뚱뚱한 Body 배열을 페어 순서로 무작위 접근만 (셰이프·포즈 읽기 비용의 하한)
    let t_touch = best(&mut || {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for &(a, b_raw) in &pairs {
            let b = (b_raw & !STATIC_BIT) as usize;
            let ba = w.body(handles[a as usize]);
            let bb = w.body(handles[b]);
            acc += ba.pose().pos.x + bb.pose().pos.x;
        }
        let e = t.elapsed().as_nanos();
        std::hint::black_box(acc);
        e
    });

    // (c) 같은 접근을 **압축 배열**에서 (Body 대신 Pose/Shape 슬라이스)
    let t_touch_compact = best(&mut || {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for &(a, b_raw) in &pairs {
            let b = (b_raw & !STATIC_BIT) as usize;
            acc += poses[a as usize].pos.x + poses[b].pos.x;
        }
        let e = t.elapsed().as_nanos();
        std::hint::black_box(acc);
        e
    });

    println!(
        "바디 {}  페어 {}  접촉 {hits}   (min-of-{reps}, us)",
        handles.len(),
        pairs.len()
    );
    println!("  contact() 전량           {t_contact:8.1}");
    println!(
        "    └ 구-구  {n_ss:6} 페어     {t_ss:8.1}  ({:.3} us/페어)",
        t_ss / n_ss.max(1) as f64
    );
    println!(
        "    └ 그 외  {n_sb:6} 페어     {t_sb:8.1}  ({:.3} us/페어)",
        t_sb / n_sb.max(1) as f64
    );
    println!("  Body 무작위 접근만       {t_touch:8.1}   (128B 구조체)");
    println!("  압축 배열 무작위 접근만  {t_touch_compact:8.1}   (Pose 28B)");
}
