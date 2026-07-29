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

/// 결정적 `sin`·`cos` 동시 계산 — Cephes `sinf`/`cosf`와 같은 범위 축소 + 다항.
///
/// **이것도 핫패스에 있었다**(검수 rev_1e81559e BLOCKER): `Quat::from_axis_angle`이
/// 회전 CCD의 매 반복마다 불리고, Jacobi 주축 분해도 결정적 atan2 **직후에** 다시
/// `sin_cos`를 부른다. atan2만 갈아끼우고 "초월함수는 두 곳뿐"이라 한 내 조사가
/// `.sin_cos()`를 놓친 것이다 — 같은 계약을 여기에도 건다.
///
/// 범위 축소는 π/2를 3조각으로 나눠 빼는 Cody-Waite다(한 번에 빼면 큰 각에서 상쇄로
/// 유효숫자가 날아간다). 사분면 4개로 접고 짧은 다항 두 개로 푼다.
pub fn sin_cos(x: f32) -> (f32, f32) {
    if !x.is_finite() {
        return (f32::NAN, f32::NAN);
    }
    // **범위 축소는 f64로 한다.** f32로 접으면 큰 각에서 유효 비트가 남지 않는다 —
    // 1e6 rad에서 f32 한 칸(ulp)이 이미 0.06 rad라 f32 Cody-Waite는 수 %씩 틀리고,
    // 1e9 rad에서는 다항이 폭주해 ±Inf를 낸다(확인검수 rev_7efe4e35 MAJOR:
    // `Quat::from_axis_angle`이 비유한 쿼터니언을 돌려줬다). f64 연산도 IEEE라 결정적이고,
    // 이 경로는 회전당 한 번뿐이라 비용도 무시할 만하다.
    const FOPI_D: f64 = 4.0 / core::f64::consts::PI;
    const PI_4_D: f64 = core::f64::consts::FRAC_PI_4;
    let xd = x as f64;
    let a = xd.abs();
    // **각이 f64 정수 격자를 넘어서면 각도 정보가 남아 있지 않다**(2^52 rad ≈ 7e14 바퀴).
    // 어떤 구현도 의미 있는 답을 못 내므로 항등 회전으로 정한다 — 임의값이지만
    // 결정적이고 유한하며 단위원 위다(공개 계약을 지킨다).
    if a >= 4.5e15 {
        return (0.0, 1.0);
    }
    let mut j = (a * FOPI_D) as i64;
    let mut yf = j as f64;
    // 홀수면 하나 올려 짝수로 — 사분면 판정을 단순화한다(Cephes 규약).
    if j & 1 == 1 {
        j += 1;
        yf += 1.0;
    }
    let mut q = (j & 7) as i32;
    let mut sin_sign = if x.is_sign_negative() { -1.0f32 } else { 1.0 };
    let mut cos_sign = 1.0f32;
    if q > 3 {
        q -= 4;
        sin_sign = -sin_sign;
        cos_sign = -cos_sign;
    }
    if q > 1 {
        cos_sign = -cos_sign;
    }
    // 남은 각은 ±π/4 안이라 f32로 내려도 유효 비트가 충분하다.
    let z = (a - yf * PI_4_D) as f32;
    let zz = z * z;
    let poly_sin = z + z * zz * ((-1.951_529_6e-4 * zz + 8.332_161e-3) * zz - 1.666_665_5e-1);
    let poly_cos =
        1.0 - 0.5 * zz + zz * zz * ((2.443_315_7e-5 * zz - 1.388_731_6e-3) * zz + 4.166_664_6e-2);
    let (s_base, c_base) = if q == 1 || q == 2 {
        (poly_cos, poly_sin)
    } else {
        (poly_sin, poly_cos)
    };
    (sin_sign * s_base, cos_sign * c_base)
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

    /// sin/cos도 계약 안(오차 + 항등식).
    #[test]
    fn sin_cos_matches_std_and_keeps_identity() {
        let mut worst = 0.0f32;
        let mut worst_at = 0.0f32;
        // 작은 각부터 여러 바퀴까지 — 범위 축소가 큰 각에서 무너지는지 본다.
        for i in -20_000..=20_000 {
            let x = i as f32 * 0.001; // ±20 rad ≈ ±3바퀴
            let (s, c) = sin_cos(x);
            let e = (s - x.sin()).abs().max((c - x.cos()).abs());
            if e > worst {
                worst = e;
                worst_at = x;
            }
            // 항등식은 근사와 무관하게 성립해야 한다
            let unit = (s * s + c * c - 1.0).abs();
            assert!(unit < 1e-5, "sin²+cos²−1 = {unit} at x={x}");
        }
        assert!(worst < 5e-6, "sin/cos 최대 오차 {worst} at {worst_at}");
        assert!(sin_cos(f32::NAN).0.is_nan());
        // **큰 각에서도 계약을 지킨다** — f32 축소는 여기서 무너졌다(1e9에서 ±Inf).
        // 이 영역에서 std와의 절대 오차를 요구하는 것은 무의미하지만(f32 한 칸이 이미
        // 수십 rad다), **유한·단위원 위**라는 것은 언제나 참이어야 한다.
        for x in [1e5f32, 1e6, 1e7, 1e9, 1e12, -1e9, f32::MAX] {
            let (s, c) = sin_cos(x);
            assert!(s.is_finite() && c.is_finite(), "sin_cos({x}) = ({s}, {c})");
            assert!(
                (s * s + c * c - 1.0).abs() < 1e-4,
                "sin²+cos²−1이 큰 각에서 깨졌다: x={x} → {}",
                s * s + c * c
            );
        }
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
