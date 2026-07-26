//! 박스-박스 내로우페이즈 분해 — 피라미드 씬에서 `contact()`가 어느 경로를 타는지.
//!
//! 40단 피라미드에서 narrow가 페어당 1.34us인데 SAT+클리핑 산술은 ~500 flops다.
//! `FEATURE_GENERIC`(GJK/EPA 폴백 표식)은 공개 상수라, 반환된 매니폴드만 보고
//! 어느 경로를 탔는지 셀 수 있다 — 핫패스에 카운터를 심지 않고도 분포가 나온다.
//!
//! 실행: cargo run --release -p maru-bench --bin boxsplit -- [tiers]

use maru_collide::{contact, Aabb, Broadphase, Pose, Shape, FEATURE_GENERIC, STATIC_BIT};
use maru_dynamics::{Body, World, WorldParams};
use maru_math::{Quat, Vec3};
use std::time::Instant;

fn main() {
    let tiers: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let warmup: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(240);

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
    let half = 0.5f32;
    let mut handles = vec![w.add_body(Body::new_static(
        Shape::Box {
            half: Vec3::new(120.0, 1.0, 20.0),
            radius: 0.0,
        },
        pose(0.0, -1.0, 0.0),
    ))];
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
                pose(x0 + i as f32 * 2.0 * half, y, 0.0),
                1.0,
            )));
        }
    }
    for _ in 0..warmup {
        let _ = w.try_step(1.0 / 60.0);
    }

    let shapes: Vec<Shape> = handles.iter().map(|h| w.body(*h).shape().clone()).collect();
    let poses: Vec<Pose> = handles.iter().map(|h| w.body(*h).pose()).collect();
    let margin = WorldParams::default().speculative_margin;

    // 회전 박스의 보수적 AABB (지지점 투영 — 계측 목적엔 정확하지 않아도 되지만
    // 페어 수가 실제와 비슷해야 의미가 있으므로 축 투영으로 제대로 구한다).
    let aabb_of = |i: usize| -> Aabb {
        let p = poses[i];
        let e = match shapes[i] {
            Shape::Box { half, radius } => {
                let h = half + Vec3::new(radius, radius, radius);
                let ax = p.rot.rotate(Vec3::new(1.0, 0.0, 0.0));
                let ay = p.rot.rotate(Vec3::new(0.0, 1.0, 0.0));
                let az = p.rot.rotate(Vec3::new(0.0, 0.0, 1.0));
                Vec3::new(
                    ax.x.abs() * h.x + ay.x.abs() * h.y + az.x.abs() * h.z,
                    ax.y.abs() * h.x + ay.y.abs() * h.y + az.y.abs() * h.z,
                    ax.z.abs() * h.x + ay.z.abs() * h.y + az.z.abs() * h.z,
                )
            }
            Shape::Sphere { radius } => Vec3::new(radius, radius, radius),
            _ => Vec3::new(1.0, 1.0, 1.0),
        };
        Aabb::from_center(p.pos, e + Vec3::new(margin, margin, margin))
    };
    let dyn_items: Vec<(u32, Aabb)> = (1..handles.len()).map(|i| (i as u32, aabb_of(i))).collect();
    let bp = Broadphase::build(dyn_items, vec![(0, aabb_of(0))]);
    let pairs = bp.canonical_pairs_bounded(1 << 20).unwrap();

    // 경로 분류: 매니폴드에 FEATURE_GENERIC 접점이 있으면 GJK/EPA 폴백을 탄 것이다.
    let (mut n_none, mut n_clip, mut n_fallback) = (0usize, 0usize, 0usize);
    let (mut pts_clip, mut pts_fb) = (0usize, 0usize);
    let mut clip_pairs: Vec<(u32, u32)> = Vec::new();
    let mut fb_pairs: Vec<(u32, u32)> = Vec::new();
    for &(a, b_raw) in &pairs {
        let (ai, bi) = (a as usize, (b_raw & !STATIC_BIT) as usize);
        match contact(&shapes[ai], &poses[ai], &shapes[bi], &poses[bi], margin) {
            None => n_none += 1,
            Some(m) => {
                let generic = m.points().iter().any(|p| p.feature.0 == FEATURE_GENERIC);
                if generic {
                    n_fallback += 1;
                    pts_fb += m.points().len();
                    fb_pairs.push((a, b_raw));
                } else {
                    n_clip += 1;
                    pts_clip += m.points().len();
                    clip_pairs.push((a, b_raw));
                }
            }
        }
    }

    let reps = 200;
    let time_set = |set: &[(u32, u32)]| -> f64 {
        let mut best = u128::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let mut h = 0usize;
            for &(a, b_raw) in set {
                let (ai, bi) = (a as usize, (b_raw & !STATIC_BIT) as usize);
                if let Some(m) = contact(&shapes[ai], &poses[ai], &shapes[bi], &poses[bi], margin) {
                    h += m.points().len();
                }
            }
            best = best.min(t.elapsed().as_nanos());
            std::hint::black_box(h);
        }
        best as f64 / 1000.0
    };
    let t_all = time_set(&pairs);
    let t_clip = time_set(&clip_pairs);
    let t_fb = time_set(&fb_pairs);

    let per = |t: f64, n: usize| if n == 0 { 0.0 } else { t / n as f64 };
    println!(
        "{tiers}단  바디 {}  페어 {}   (min-of-{reps}, us)",
        handles.len(),
        pairs.len()
    );
    println!(
        "  contact() 전량            {t_all:8.1}  ({:.3} us/페어)",
        per(t_all, pairs.len())
    );
    println!(
        "    └ SAT+클리핑  {n_clip:6} 페어 {pts_clip:6}점  {t_clip:8.1}  ({:.3} us/페어)",
        per(t_clip, n_clip)
    );
    println!(
        "    └ GJK 폴백    {n_fallback:6} 페어 {pts_fb:6}점  {t_fb:8.1}  ({:.3} us/페어)",
        per(t_fb, n_fallback)
    );
    println!("    └ 접촉 없음   {n_none:6} 페어");
}
