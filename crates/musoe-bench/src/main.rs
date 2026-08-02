//! M0 벤치 하니스 골격 — SIMD 수학 마이크로벤치 (native + wasm 동일 소스).
//!
//! 설계 §6: 벤치는 M0부터 존재. 결정론 스모크로 체크섬을 함께 출력한다 —
//! 같은 타깃·같은 빌드에서 체크섬이 달라지면 L0 위반 신호(설계 §8).
//! 난수는 외부 의존 없이 고정 시드 xorshift(재현성).

use musoe_math::{Quat, Vec3, Vec3x4};
use std::time::Instant;
use wide::f32x4;

const N: usize = 1 << 20; // 1,048,576
const REPS: usize = 8;

struct XorShift(u64);
impl XorShift {
    fn next_f32(&mut self) -> f32 {
        // [-1, 1) 균일 근사 — 벤치 입력용
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
    }
    fn next_vec3(&mut self) -> Vec3 {
        Vec3::new(self.next_f32(), self.next_f32(), self.next_f32())
    }
}

fn bench<F: FnMut() -> f32>(name: &str, ops_per_rep: usize, mut f: F) {
    // 워밍업 1회 + REPS 측정, 최소값 채택(잡음 하한).
    let mut checksum = f();
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        checksum += f();
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let ns_per_op = best * 1e9 / ops_per_rep as f64;
    println!(
        "{name:<28} {ns_per_op:>8.3} ns/op   checksum={:>14.6e}",
        checksum
    );
}

fn main() {
    let target = if cfg!(target_arch = "wasm32") {
        "wasm32"
    } else {
        "native"
    };
    let simd = if cfg!(target_feature = "simd128") || cfg!(target_feature = "sse2") {
        "simd-on"
    } else {
        "simd-OFF" // 조용한 강등 금지(설계 §6) — 눈에 띄게 출력
    };
    println!("musoe-bench M0  target={target}  {simd}  N={N}  reps={REPS}");

    let mut rng = XorShift(0x9E3779B97F4A7C15);
    let a: Vec<Vec3> = (0..N).map(|_| rng.next_vec3()).collect();
    let b: Vec<Vec3> = (0..N).map(|_| rng.next_vec3()).collect();

    // 1) 스칼라 Vec3: dot+cross 체인 (기준선)
    bench("scalar dot+cross", N, || {
        let mut acc = 0.0f32;
        for (av, bv) in a.iter().zip(&b) {
            acc += av.cross(*bv).dot(*av);
        }
        acc
    });

    // 2) SoA Vec3x4: 동일 연산 4-wide (설계 §3 정본 레이아웃)
    let a4: Vec<Vec3x4> = a
        .chunks_exact(4)
        .map(|c| Vec3x4::from_array([c[0], c[1], c[2], c[3]]))
        .collect();
    let b4: Vec<Vec3x4> = b
        .chunks_exact(4)
        .map(|c| Vec3x4::from_array([c[0], c[1], c[2], c[3]]))
        .collect();
    bench("soa4 dot+cross", N, || {
        let mut acc = f32x4::splat(0.0);
        for (av, bv) in a4.iter().zip(&b4) {
            acc += av.cross(*bv).dot(*av);
        }
        acc.reduce_add()
    });

    // 3) 쿼터니언 회전 (조인트/자세 적분 핫패스 대표)
    let q = Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.7);
    bench("quat rotate", N, || {
        let mut acc = 0.0f32;
        for av in &a {
            acc += q.rotate(*av).x;
        }
        acc
    });

    // 4) 자세 적분 (Symplectic Euler 짝, 설계 §4.4)
    bench("quat integrate", N / 16, || {
        let mut qq = Quat::IDENTITY;
        for av in a.iter().take(N / 16) {
            qq = qq.integrate(*av, 1.0 / 240.0);
        }
        qq.w
    });

    println!("done");
}
