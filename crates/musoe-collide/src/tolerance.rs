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

/// 형상 쌍의 특성 길이 L = **작은 쪽** 로컬 AABB 대각.
/// 접촉 feature의 크기는 작은 형상이 상한이다 — 거대한 지면(대각 243m)을 기준으로 잡으면
/// 그 위 1m 박스의 임계가 240배 느슨해져 침하가 커진다(40단 수용 게이트에서 실측).
pub fn characteristic_length(a: &Shape, b: &Shape) -> f32 {
    let ext = |s: &Shape| {
        let bb = s.local_aabb();
        (bb.max - bb.min).length()
    };
    let l = ext(a).min(ext(b));
    if l.is_finite() && l > 0.0 {
        l
    } else {
        1.0
    }
}

/// 페어의 **최소 특징 크기** — 두 셰이프 중 가장 얇은 방향의 두께(반두께/반지름).
///
/// `characteristic_length`(AABB 대각)는 등방 형상에서만 접촉 국소 스케일을 대변한다.
/// 10×0.01×10 판처럼 종횡비가 큰 형상은 대각이 ~14라 그 5%가 0.7 m가 되고, 판 두께의
/// 70배짜리 "근접" 허용치가 서로 다른 접촉을 같은 것으로 오인하게 한다(리뷰 MAJOR).
/// 접점 매칭처럼 **접촉 국소** 판정에는 이 값을 쓴다.
pub fn min_feature_size(a: &Shape, b: &Shape) -> f32 {
    let thin = |s: &Shape| {
        let bb = s.local_aabb();
        let e = (bb.max - bb.min) * 0.5;
        e.x.min(e.y).min(e.z)
    };
    let l = thin(a).min(thin(b));
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

/// 길이² 차원. **두 용도가 있으니 혼동 금지**:
/// - 제곱 길이(`v.length_sq()`) 비교 — v가 L이면 L²
/// - 면적(`cross(e1,e2).length()`) 비교 — 두 에지의 cross **길이**가 L²
///
/// `cross(...).length_sq()`는 **L⁴**이므로 이 값과 비교하면 안 된다(스케일 k배에서 좌변 k⁴,
/// 임계 k² → 작은 스케일에서 정상 면을 퇴화로 오판. 리뷰 NOT_RESOLVED).
#[inline]
pub fn length_sq_eps(scale: f32) -> f32 {
    let e = length_eps(scale);
    (e * e).max(FLOOR)
}

/// 면적 차원 별칭 — `cross(e1,e2).length()`와 비교할 때 의도를 드러내기 위해.
#[inline]
pub fn area_eps(scale: f32) -> f32 {
    length_sq_eps(scale)
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
    use musoe_math::Vec3;

    /// 차원 계약: 스케일 k배에서 각 epsilon은 해당 차수로 스케일해야 한다.
    #[test]
    fn eps_dimensions_are_consistent() {
        let (s, k) = (1.0f32, 10.0f32);
        let r_len = length_eps(s * k) / length_eps(s);
        let r_area = area_eps(s * k) / area_eps(s);
        let r_vol = volume_eps(s * k) / volume_eps(s);
        let r_bary = bary_denom_eps(s * k) / bary_denom_eps(s);
        assert!((r_len - k).abs() < 1e-3 * k, "L 차수 {r_len}");
        assert!((r_area - k * k).abs() < 1e-3 * k * k, "L² 차수 {r_area}");
        assert!(
            (r_vol - k.powi(3)).abs() < 1e-3 * k.powi(3),
            "L³ 차수 {r_vol}"
        );
        assert!(
            (r_bary - k.powi(4)).abs() < 1e-2 * k.powi(4),
            "L⁴ 차수 {r_bary}"
        );
    }

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

    /// 접촉 임계는 **작은 형상**이 지배한다 (거대 지면이 임계를 느슨하게 만들면 안 됨).
    #[test]
    fn characteristic_length_uses_smaller_shape() {
        let small = Shape::Sphere { radius: 0.01 };
        let large = Shape::Box {
            half: Vec3::new(120.0, 1.0, 20.0),
            radius: 0.0,
        };
        let l = characteristic_length(&small, &large);
        assert!(l < 0.1, "l={l} — 큰 형상이 임계를 지배하면 안 된다");
    }
}
