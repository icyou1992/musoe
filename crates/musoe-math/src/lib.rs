//! musoe-math — Maru 물리엔진 수학 코어 (M0).
//!
//! 설계 계약(02-architecture.md):
//! - §2: f32 단일 코어, SIMD는 stable `wide`(4레인 = WASM SIMD128 동형), FMA 비의존.
//! - §3: 핫패스는 SoA — 스칼라 `Vec3`/`Quat`은 API 표면, `Vec3x4`/`Aabbx4`가 솔버·브로드페이즈
//!   배치의 정본 레이아웃.
//! - §8: 재결합 금지(기본 프로파일), denormal은 정의 지점 flush(`flush_tiny`).

pub(crate) mod det;
mod fp_env;
mod mat3;
mod scalar;
mod soa;

pub use det::{acos, asin, atan, atan2};
#[cfg(feature = "test-hooks")]
pub use fp_env::force_flush_to_zero_for_test;
pub use fp_env::{canonicalize as fp_canonicalize, verify as fp_verify};
pub use mat3::Mat3;
pub use scalar::{Quat, Vec3};
pub use soa::{Aabbx4, Mat3x4, Quatx4, Vec3x4};

/// 물리적 무의미 임계(설계 §8 denormal 위생 계층). 속도·임펄스 누산 후 정의 지점에서 호출.
pub const TINY_EPS: f32 = 1.0e-8;

#[inline]
pub fn flush_tiny(x: f32) -> f32 {
    if x.abs() < TINY_EPS {
        0.0
    } else {
        x
    }
}
