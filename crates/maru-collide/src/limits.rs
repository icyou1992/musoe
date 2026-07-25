//! 엔진 **지원 범위**와 상태 정책 (설계 §8 위생).
//!
//! # 왜 스칼라 상한 하나로는 안 되는가
//!
//! 단일 상한(예: 모든 값 ≤ 1e9)은 두 방향으로 동시에 틀린다:
//! - **너무 넓다** — 위치 1e18과 역질량 1e-30 조합은 조인트 솔버의 `c_pos × K⁻¹`에서
//!   **중간 연산이 먼저** Inf가 된다. 스텝 말 클램프는 그 Inf를 포화값으로 바꿀 뿐이라
//!   `try_step`이 성공을 반환하면서 상태를 조용히 손상시킨다.
//! - **너무 좁다** — 질량 8e18kg 박스(half=1000m·density=1e9)는 생성 정책을 통과하는데
//!   그 접촉 임펄스가 상한에 잘려 warm start가 매 프레임 부족해진다(조용한 물리 왜곡).
//!
//! 그래서 **차원별 지원 범위**를 정하고, 그 범위 안에서는 중간 연산까지 f32 안에 남도록
//! 상한을 서로 맞춘다. 지원 범위 밖은 **입력 경계에서 거부**한다 — 포화가 아니라 거부다.
//!
//! # 파생 상태 상한은 오류 신호지 정책이 아니다
//!
//! 지원 범위 안의 정상 물리는 `MAX_STATE`에 **닿지 않는다**(아래 여유 계산 참조).
//! 닿았다면 지원 범위 밖 입력이거나 엔진 버그이므로, 상태를 안전하게 포화시키되
//! **조용히 넘어가지 않는다** — `try_step`이 `StateOverflow`를 반환하고 진단에 집계된다.

use maru_math::Vec3;

/// 셰이프 반크기·반지름·hull 정점 상한 (m).
pub const MAX_SHAPE_EXTENT: f32 = 1.0e4;
/// 위치·앵커·일반 파라미터 상한 (m). 월드 범위이기도 하다.
pub const MAX_INPUT: f32 = 1.0e6;
/// 선속도·각속도 상한 (m/s, rad/s).
pub const MAX_SPEED: f32 = 1.0e4;
/// 역질량 상한 (1/kg) = 질량 하한 1e-9 kg. 역관성도 같은 상한을 쓴다.
pub const MAX_INV_MASS: f32 = 1.0e9;
/// 역질량 하한 (1/kg) = **질량 상한 1e9 kg**. 이보다 무거운 물체는 지원하지 않는다 —
/// 그 임펄스가 파생 상한을 위협하고, 물리적으로도 static으로 모델링할 대상이다.
pub const MIN_INV_MASS: f32 = 1.0e-9;
/// hull 정점 수 상한. support가 O(n) 선형 스캔이라 성능 상한이자 직렬화 자원 상한이다.
/// 더 복잡한 메시는 convex decomposition으로 나눈다(v1 계약).
pub const MAX_HULL_POINTS: usize = 1024;

/// 파생 상태 상한 (위치·속도·누적 임펄스·PCM).
///
/// 지원 범위 안의 최악 임펄스 여유: `k ≥ inv_mass ≥ 1e-9` 이므로 유효질량 `1/k ≤ 1e9`,
/// 상대속도 ≤ `2·MAX_SPEED = 2e4` → 임펄스 ≤ `2e13`. `MAX_STATE`는 그 **50배 위**다.
/// 즉 정상 물리는 여기 닿지 않는다 — 닿으면 신호다.
pub const MAX_STATE: f32 = 1.0e15;

/// 스텝 크기 범위. 하한이 필요한 이유: soft 계수의 `1/h`가 폭발해 조인트 bias가
/// 중간 연산에서 f32를 위협한다.
pub const MIN_DT: f32 = 1.0e-4;
pub const MAX_DT: f32 = 0.25;

/// 일반 스칼라 입력 (위치·앵커·파라미터).
#[inline]
pub fn in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_INPUT
}
#[inline]
pub fn vec_in_range(v: Vec3) -> bool {
    in_range(v.x) && in_range(v.y) && in_range(v.z)
}
/// 셰이프 치수.
#[inline]
pub fn extent_in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_SHAPE_EXTENT
}
#[inline]
pub fn vec_extent_in_range(v: Vec3) -> bool {
    extent_in_range(v.x) && extent_in_range(v.y) && extent_in_range(v.z)
}
/// 속도 (선·각 공통).
#[inline]
pub fn speed_in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_SPEED
}
#[inline]
pub fn vec_speed_in_range(v: Vec3) -> bool {
    speed_in_range(v.x) && speed_in_range(v.y) && speed_in_range(v.z)
}
/// 동적 바디의 역질량·역관성 성분.
#[inline]
pub fn inv_mass_in_range(v: f32) -> bool {
    v.is_finite() && (MIN_INV_MASS..=MAX_INV_MASS).contains(&v)
}

/// 파생 상태가 상한 안인가.
#[inline]
pub fn in_state_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_STATE
}
#[inline]
pub fn vec_in_state_range(v: Vec3) -> bool {
    in_state_range(v.x) && in_state_range(v.y) && in_state_range(v.z)
}

/// 상태를 상한 안으로 포화시킨다. **반환값이 바뀌었는지 호출부가 알 수 있어야 한다** —
/// 포화는 정상 처리가 아니라 오류 신호이므로 조용히 삼키지 않는다.
#[inline]
pub fn saturate_state(v: f32) -> (f32, bool) {
    if v.is_nan() {
        (0.0, true)
    } else if v > MAX_STATE {
        (MAX_STATE, true)
    } else if v < -MAX_STATE {
        (-MAX_STATE, true)
    } else {
        (v, false)
    }
}

/// 벡터 포화. 성분 중 하나라도 잘렸으면 true.
#[inline]
pub fn saturate_state_vec(v: Vec3) -> (Vec3, bool) {
    let (x, a) = saturate_state(v.x);
    let (y, b) = saturate_state(v.y);
    let (z, c) = saturate_state(v.z);
    (Vec3::new(x, y, z), a || b || c)
}
