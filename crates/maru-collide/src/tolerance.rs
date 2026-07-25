//! 차원별 톨러런스 정책 (리뷰 MAJOR — 절대 임계 계열의 구조적 봉쇄).
//!
//! 이 프로젝트에서 "절대 임계"는 3번 재발했다(EPA 퇴화·Mat3 det·GJK 종료). 원인은 같다:
//! **임계값에 차원이 있는데 상수를 쓰면 씬 스케일이 바뀔 때 깨진다.** 길이(L)·면적(L²)·
//! 부피(L³)·바리센트릭 분모(L⁴)는 각각 다른 차수로 스케일한다 — 1mm 부품 씬에서
//! L⁴ 항은 1e-12보다 작아져 정상 삼각형이 퇴화로 오판된다.
//!
//! 정책: 쌍의 **특성 길이**(두 형상 로컬 AABB 대각의 최대)에서 차원별 epsilon을 유도하고,
//! f32 ULP 바닥으로 하한을 둔다. 절대 상수는 무차원 양(정규화 내적 등)에만 허용한다.

use crate::Shape;

/// 상대 epsilon 계수 (무차원). f32 유효자릿수(~7)에서 안전한 여유.
const REL: f32 = 1.0e-6;
/// f32 정규수 하한 근처의 절대 바닥 — 특성 길이가 0에 수렴해도 0으로 나누지 않게.
const FLOOR: f32 = 1.0e-30;

/// 형상 쌍의 특성 길이 L (로컬 AABB 대각 중 큰 쪽).
pub fn characteristic_length(a: &Shape, b: &Shape) -> f32 {
    let ext = |s: &Shape| {
        let bb = s.local_aabb();
        (bb.max - bb.min).length()
    };
    let l = ext(a).max(ext(b));
    if l.is_finite() && l > 0.0 {
        l
    } else {
        1.0
    }
}

/// 길이 차원 epsilon (L¹).
#[inline]
pub fn length_eps(scale: f32) -> f32 {
    (REL * scale).max(FLOOR)
}

/// 길이² 차원 (제곱 길이 비교·cross 길이²).
#[inline]
pub fn length_sq_eps(scale: f32) -> f32 {
    let e = length_eps(scale);
    (e * e).max(FLOOR)
}

/// 부피 차원 (L³) — 사면체 퇴화 판정.
#[inline]
pub fn volume_eps(scale: f32) -> f32 {
    let e = length_eps(scale);
    (e * e * e).max(FLOOR)
}

/// 바리센트릭 분모 차원 (L⁴) — d00·d11 − d01² 는 길이의 4제곱이다.
#[inline]
pub fn bary_denom_eps(scale: f32) -> f32 {
    let e2 = length_sq_eps(scale);
    (e2 * e2).max(FLOOR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use maru_math::Vec3;

    #[test]
    fn eps_scales_with_dimension() {
        let small = 1.0e-3f32; // 1mm 부품
        let big = 1.0e2f32; // 100m 지형
                            // 차원별로 다른 차수로 스케일해야 한다
        assert!(length_eps(big) / length_eps(small) > 1.0e4);
        assert!(volume_eps(big) / volume_eps(small) > 1.0e14);
        // 0/비유한 입력에도 유한 양수
        assert!(length_eps(0.0) > 0.0 && length_eps(0.0).is_finite());
    }

    #[test]
    fn characteristic_length_uses_larger_shape() {
        let small = Shape::Sphere { radius: 0.01 };
        let large = Shape::Box {
            half: Vec3::new(5.0, 5.0, 5.0),
            radius: 0.0,
        };
        let l = characteristic_length(&small, &large);
        assert!(l > 10.0, "l={l}");
    }
}
