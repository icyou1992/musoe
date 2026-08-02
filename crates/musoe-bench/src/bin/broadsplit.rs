//! 브로드페이즈 하위 단계 분해 — `broad` 안에서 무엇이 지배하는지 실측한다.
//!
//! `broad`가 구 더미 3000에서 스텝의 23%인데, 그 안의 분포를 본 적이 없다.
//! 추측으로 손대지 않으려고 만든 계측기다: 트리 재구축 / 질의 / 정렬을 따로 잰다.
//! 실행: cargo run --release -p musoe-bench --bin broadsplit -- [n]

use musoe_collide::{Aabb, Broadphase, Tree};
use musoe_math::Vec3;
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    // compare 하네스의 구 더미와 같은 배치 (반지름 0.5, 간격 1.1, side³).
    let side = (n as f32).cbrt().ceil() as usize;
    let mut items: Vec<(u32, Aabb)> = Vec::new();
    'outer: for iy in 0..side * 3 {
        for ix in 0..side {
            for iz in 0..side {
                if items.len() >= n {
                    break 'outer;
                }
                let c = Vec3::new(
                    ix as f32 * 0.9 - side as f32 * 0.45,
                    0.55 + iy as f32 * 0.9,
                    iz as f32 * 0.9 - side as f32 * 0.45,
                );
                // 스펙큘레이티브 마진까지 포함한 대략의 프록시 크기
                items.push((
                    items.len() as u32,
                    Aabb::from_center(c, Vec3::new(0.54, 0.54, 0.54)),
                ));
            }
        }
    }
    let statics: Vec<(u32, Aabb)> = vec![(
        0,
        Aabb::from_center(Vec3::new(0.0, -1.0, 0.0), Vec3::new(30.0, 1.0, 30.0)),
    )];

    let reps = 200;
    let best = |f: &mut dyn FnMut() -> u128| -> f64 {
        let mut b = u128::MAX;
        for _ in 0..reps {
            b = b.min(f());
        }
        b as f64 / 1000.0
    };

    let t_build = best(&mut || {
        let t = Instant::now();
        let bp = Broadphase::build(items.clone(), statics.clone());
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&bp);
        e
    });
    // clone 비용을 빼기 위한 대조군
    let t_clone = best(&mut || {
        let t = Instant::now();
        let a = items.clone();
        let b = statics.clone();
        let e = t.elapsed().as_nanos();
        std::hint::black_box((&a, &b));
        e
    });

    let bp = Broadphase::build(items.clone(), statics.clone());
    let t_pairs = best(&mut || {
        let t = Instant::now();
        let p = bp.canonical_pairs_bounded(1 << 20).unwrap();
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&p);
        e
    });

    // 정렬만 따로: 페어 집합을 미리 뽑아 섞은 뒤 정렬 시간을 잰다.
    let mut pairs = bp.canonical_pairs_bounded(1 << 20).unwrap();
    let n_pairs = pairs.len();
    let shuffled: Vec<(u32, u32)> = {
        let mut v = pairs.clone();
        // 결정적 셔플 (xorshift)
        let mut s = 0x1234_5678_9abc_def0u64;
        for i in (1..v.len()).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.swap(i, (s % (i as u64 + 1)) as usize);
        }
        v
    };
    let t_sort = best(&mut || {
        let mut v = shuffled.clone();
        let t = Instant::now();
        v.sort_unstable();
        v.dedup();
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&v);
        e
    });
    let t_sortclone = best(&mut || {
        let t = Instant::now();
        let v = shuffled.clone();
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&v);
        e
    });

    // 트리 재구축만
    let t_tree = best(&mut || {
        let t = Instant::now();
        let tr = Tree::build(&items);
        let e = t.elapsed().as_nanos();
        std::hint::black_box(&tr);
        e
    });

    pairs.clear();
    println!("바디 {n}  페어 {n_pairs}   (min-of-{reps}, us)");
    println!("  Broadphase::build   {t_build:8.1}  (그중 items.clone {t_clone:.1})");
    println!("    └ Tree::build     {t_tree:8.1}");
    println!("  canonical_pairs     {t_pairs:8.1}");
    println!("    └ sort+dedup      {t_sort:8.1}  (clone 대조 {t_sortclone:.1})");
    println!("  합계 ≈ {:.1} us/스텝", t_build - t_clone + t_pairs);
}
