//! 엔진 전역 크기 정책 — **입력 정책과 상태 정책을 나눈다**(리뷰 BLOCKER).
//!
//! 하나의 상한을 입력과 상태에 같이 쓰면 정책이 `step`에 대해 **닫히지 않는다**:
//! `pos.x = 0.9e9`와 `linvel.x = 0.9e9`는 둘 다 합법이지만 `step(0.25)` 한 번이면
//! 위치가 1.125e9가 되어 자기 상태가 자기 정책을 위반한다(릴리스에서는 조용히 통과하고
//! 이후 snapshot이 복원 불가가 된다). 그래서 두 층으로 나눈다:
//!
//! - **입력 정책 `MAX_INPUT`** — 호스트가 넣는 값(자세·속도·셰이프·질량·파라미터·앵커).
//!   위반은 **호스트의 오류**이므로 거부한다(Err/패닉).
//! - **상태 정책 `MAX_STATE`** — 엔진이 보유·직렬화하는 값(위치·속도·누적 임펄스·PCM 캐시).
//!   솔버가 정당하게 만들어낼 수 있는 값이므로 거부가 아니라 **스텝 말에 클램프**해서
//!   정책을 닫는다. 그 결과 "엔진이 만들 수 있는 상태 = 리더가 받는 상태"가 항상 성립한다.

use maru_math::Vec3;

/// 호스트 입력 상한. 길이 m·속도 m/s·역질량 1/kg 기준으로 1e9는 물리 시뮬레이션의
/// 정상 범위를 훨씬 넘는다(1e9 m ≈ 지구-달 거리 2배, 1e9 m/s ≈ 광속 3배).
pub const MAX_INPUT: f32 = 1.0e9;

/// 엔진 상태 상한. 두 상태값의 곱이 1e36으로 f32 범위(3.4e38) 안에 남는 지점.
///
/// 솔버의 연쇄 곱(`r·I·r`)은 **위치가 아니라 레버암 r**을 쓰고, r은 셰이프 크기로
/// 묶여 있어(입력 정책) 이 상한과 무관하게 안전하다. 상태 상한이 커도 되는 이유다.
pub const MAX_STATE: f32 = 1.0e18;

/// hull 정점 수 상한. support 함수가 O(n) 선형 스캔이라 성능 상한이고,
/// 스냅샷 writer가 만드는 바이트를 묶는 자원 상한이기도 하다.
/// 더 복잡한 메시는 convex decomposition으로 여러 hull로 나눠야 한다(v1 계약).
pub const MAX_HULL_POINTS: usize = 1024;

/// 입력 정책 안인가.
#[inline]
pub fn in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_INPUT
}

/// 입력 정책 안인가 (벡터).
#[inline]
pub fn vec_in_range(v: Vec3) -> bool {
    in_range(v.x) && in_range(v.y) && in_range(v.z)
}

/// 상태 정책 안인가.
#[inline]
pub fn in_state_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_STATE
}

/// 상태 정책 안인가 (벡터).
#[inline]
pub fn vec_in_state_range(v: Vec3) -> bool {
    in_state_range(v.x) && in_state_range(v.y) && in_state_range(v.z)
}

/// 상태를 정책 안으로 클램프한다. **비유한 값도 여기서 정리된다** — NaN은 0으로,
/// ±Inf는 ±MAX_STATE로. 정책을 닫는 유일한 지점이므로 예외를 두지 않는다.
#[inline]
pub fn clamp_state(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(-MAX_STATE, MAX_STATE)
    }
}

/// 상태 벡터 클램프.
#[inline]
pub fn clamp_state_vec(v: Vec3) -> Vec3 {
    Vec3::new(clamp_state(v.x), clamp_state(v.y), clamp_state(v.z))
}
