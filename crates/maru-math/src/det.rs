//! 결정적 초월함수 (설계 §8 L1 — "libm 강제").
//!
//! **왜 std를 안 쓰나.** `f32::atan2`는 플랫폼의 libm으로 내려간다. glibc·musl·Apple·
//! 브라우저 구현이 서로 마지막 ulp에서 다르고, 같은 libc도 버전에 따라 다르다. 물리는
//! 그 차이를 스텝마다 증폭하므로 **록스텝 넷코드가 갈린다** — L1이 막으려는 것이 정확히
//! 그것이다. 여기 있는 함수는 IEEE 기본 연산(+ − × ÷ 비교)만 쓰므로, RNE·flush off라는
//! FP 환경 계약(`fp_env`) 아래서 **어느 플랫폼이든 같은 비트**를 낸다.
//!
//! **L1 전용이 아니라 항상 쓴다.** 설계는 L0와 L1의 차이를 "libm/FMA 경로뿐"이라고 적었지만,
//! 두 빌드가 다른 함수를 부르면 "L0에서는 되는데 L1에서 갈린다"는 계열의 버그가 생긴다.
//! 프로덕션 호출 지점이 두 곳뿐이라 그냥 항상 결정적 경로로 두는 편이 싸고 정직하다.
//!
//! 정확도는 f32 기준 **최대 절대오차 ~1e-6 rad**다(테스트가 잠근다). 조인트 각·주축 분해가
//! 쓰는 곳이라 이 정도면 물리적으로 무의미한 차이다 — 대신 재현성을 얻는다.

/// `atan(|x|)` — Cephes `atanf`와 같은 **2단 범위 축소 + 4항 다항**.
///
/// 인자를 tan(π/8) 아래로 접으면 짧은 다항으로 f32 정밀도(≈1 ulp)가 나온다.
/// 곱셈·덧셈·나눗셈만 쓰므로 IEEE 아래서 비트 재현적이다.
#[inline]
fn atan_pos(x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    const TAN_3PI_8: f32 = 2.414_213_6; // tan(3π/8)
    const TAN_PI_8: f32 = 0.414_213_57; // tan(π/8)
    let (x, y) = if x > TAN_3PI_8 {
        (-1.0 / x, FRAC_PI_2)
    } else if x > TAN_PI_8 {
        ((x - 1.0) / (x + 1.0), FRAC_PI_4)
    } else {
        (x, 0.0)
    };
    let z = x * x;
    y + ((((8.053_744_5e-2 * z - 1.387_768_6e-1) * z + 1.997_771_1e-1) * z - 3.333_294_8e-1)
        * z
        * x)
        + x
}

/// 결정적 `atan`.
#[inline]
pub fn atan(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_sign_negative() {
        -atan_pos(-x)
    } else {
        atan_pos(x)
    }
}

/// 결정적 `atan2` — 사분면 규약은 `f32::atan2`와 같다.
///
/// 부호 있는 0도 std와 같게 다룬다(`atan2(-0.0, 1.0) == -0.0`): 부호 분기가 아니라
/// 나눗셈 결과의 부호가 그대로 흘러가기 때문이다.
pub fn atan2(y: f32, x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, PI};
    if y.is_nan() || x.is_nan() {
        return f32::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 {
            FRAC_PI_2
        } else if y < 0.0 {
            -FRAC_PI_2
        } else {
            // (±0, ±0): std는 x의 부호를 따른다.
            if x.is_sign_negative() {
                if y.is_sign_negative() {
                    -PI
                } else {
                    PI
                }
            } else {
                y // ±0 그대로
            }
        };
    }
    let a = atan(y / x);
    if x > 0.0 {
        a
    } else if y >= 0.0 && !y.is_sign_negative() {
        a + PI
    } else {
        a - PI
    }
}

/// 결정적 `asin` — `|x| ≤ 1` 밖은 클램프한다(물리 입력은 반올림으로 1을 살짝 넘길 수 있다).
pub fn asin(x: f32) -> f32 {
    let x = x.clamp(-1.0, 1.0);
    // asin(x) = atan2(x, sqrt(1−x²)) — sqrt는 IEEE 정확이라 결정적이다.
    atan2(x, (1.0 - x * x).max(0.0).sqrt())
}

/// 결정적 `acos`.
pub fn acos(x: f32) -> f32 {
    let x = x.clamp(-1.0, 1.0);
    atan2((1.0 - x * x).max(0.0).sqrt(), x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// std와의 최대 오차 — **정확도가 아니라 계약**이다(여기 적힌 수치가 곧 문서).
    #[test]
    fn matches_std_within_documented_error() {
        let mut worst = 0.0f32;
        let mut worst_at = (0.0f32, 0.0f32);
        // 사분면·크기 전 영역을 훑는다(0 근처·1 근처·큰 비율 포함).
        let vals = [
            -1e6, -1e3, -100.0, -10.0, -3.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1, -0.01, -1e-6,
            0.0, 1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0, 10.0, 100.0, 1e3, 1e6,
        ];
        for &y in &vals {
            for &x in &vals {
                if x == 0.0 && y == 0.0 {
                    continue;
                }
                let got = atan2(y, x);
                let want = y.atan2(x);
                let e = (got - want).abs();
                if e > worst {
                    worst = e;
                    worst_at = (y, x);
                }
            }
        }
        assert!(
            worst < 2e-6,
            "atan2 최대 오차 {worst} (y,x)={worst_at:?} — 계약(2e-6) 초과"
        );
    }

    /// 사분면·특수값 규약이 std와 같다.
    #[test]
    fn quadrants_and_special_values_match_std() {
        let cases = [
            (1.0f32, 1.0f32),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (-0.0, 1.0),
        ];
        for (y, x) in cases {
            let got = atan2(y, x);
            let want = y.atan2(x);
            assert!(
                (got - want).abs() < 2e-6 && got.is_sign_negative() == want.is_sign_negative(),
                "atan2({y}, {x}) = {got}, std = {want}"
            );
        }
        assert!(atan2(f32::NAN, 1.0).is_nan());
        assert!(atan2(1.0, f32::NAN).is_nan());
    }

    /// asin/acos도 계약 안.
    #[test]
    fn asin_acos_match_std() {
        let mut worst = 0.0f32;
        for i in -100..=100 {
            let x = i as f32 / 100.0;
            worst = worst.max((asin(x) - x.asin()).abs());
            worst = worst.max((acos(x) - x.acos()).abs());
        }
        assert!(worst < 5e-6, "asin/acos 최대 오차 {worst}");
        // 범위 밖은 클램프 — NaN이 아니라 경계값
        assert_eq!(asin(1.5), asin(1.0));
        assert_eq!(acos(-2.0), acos(-1.0));
    }

    /// **결정론의 핵심**: 같은 입력이면 항상 같은 비트. (플랫폼 간 비교는 CI가 한다 —
    /// 여기서는 같은 빌드 안에서의 순수성·비트 재현성을 잠근다.)
    #[test]
    fn is_bit_reproducible() {
        let mut seed = 0x9E37_79B9u32;
        for _ in 0..10_000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let y = f32::from_bits((seed & 0x7FFF_FFFF) | 0x3F00_0000) - 1.0;
            let x = f32::from_bits(((seed >> 3) & 0x7FFF_FFFF) | 0x3F80_0000) - 1.5;
            let a = atan2(y, x);
            let b = atan2(y, x);
            assert_eq!(a.to_bits(), b.to_bits(), "atan2({y},{x})가 두 번 다르다");
        }
    }
}
