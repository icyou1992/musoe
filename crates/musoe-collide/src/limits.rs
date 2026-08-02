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

use musoe_math::Vec3;

/// 셰이프 반크기·반지름·hull 정점 상한 (m). 레버암 r의 상한이기도 하다.
pub const MAX_SHAPE_EXTENT: f32 = 1.0e3;
/// 셰이프 **최소** 특징 크기 (m). 좌표 정밀도와 정합해야 한다 — 아래 MAX_INPUT 주석 참조.
pub const MIN_SHAPE_EXTENT: f32 = 1.0e-3;
/// 위치·파라미터 상한 (m) = **월드 범위**.
///
/// f32 정밀도와 최소 특징 크기의 정합이 이 값을 정한다(리뷰 MAJOR): 1e4 부근의 f32 간격은
/// 약 9.8e-4 m로 `MIN_SHAPE_EXTENT`(1mm) 이하다. 1e6이면 간격이 0.0625 m가 되어 1cm 떨어진
/// 두 자세가 같은 값으로 반올림되고 작은 AABB가 퇴화한다.
/// 더 넓은 월드는 floating-origin(주기적 rebase)이 필요하며 v1의 계약이 아니다.
pub const MAX_INPUT: f32 = 1.0e4;
/// **조인트 앵커 상한** = 셰이프 범위. 앵커는 바디에 붙은 점이므로 셰이프를 벗어날 이유가
/// 없고, 이것이 레버암 r을 묶어 `ω×r` 점 속도를 유계로 만든다(리뷰 BLOCKER: 앵커를
/// MAX_INPUT까지 허용하면 각속도 100rad/s × 앵커 1e6 = 점 속도 1e8 m/s).
pub const MAX_ANCHOR: f32 = MAX_SHAPE_EXTENT;
/// 호스트가 넣는 선속도 상한 (m/s).
pub const MAX_SPEED: f32 = 1.0e4;
/// 각속도 상한 (rad/s). `MAX_ANGULAR × MAX_SHAPE_EXTENT`가 점 속도 기여의 상한이다.
pub const MAX_ANGULAR: f32 = 1.0e2;
/// 역질량 범위 (1/kg) = **질량 1e-6 ~ 1e6 kg**. 유효질량 `1/k ≤ 1/MIN_INV_MASS`가
/// 임펄스 상한의 첫 인수다.
pub const MIN_INV_MASS: f32 = 1.0e-6;
pub const MAX_INV_MASS: f32 = 1.0e6;
/// 역관성 범위 (1/(kg·m²)). 관성은 차원이 달라 질량과 같은 상한을 쓸 수 없다
/// (반지름 1cm 구의 역관성은 이미 1e9 수준). 임펄스 상한 논증에는 들어가지 않는다 —
/// `k ≥ inv_mass`라 유효질량은 회전 항과 무관하게 묶인다.
/// 하한이 1e-10인 이유는 **힌지 각 임펄스**다(리뷰 BLOCKER): `imp ≈ bias / k_ang`이고
/// `k_ang ≥ MIN_INV_INERTIA`이므로 `1/k_ang ≤ 1e10`. 각 bias를 각속도 상한(1e2)으로
/// 묶으면 `imp ≤ 1e12`로 `MAX_STATE`의 1000분의 1이다.
/// 관성 1e10 kg·m²는 반지름 100m·질량 1e6kg 물체(≈6.7e9)를 여전히 포함한다.
pub const MIN_INV_INERTIA: f32 = 1.0e-10;
pub const MAX_INV_INERTIA: f32 = 1.0e12;
/// hull 정점 수 상한.
///
/// **정확히 검증할 수 있는 범위로 낮췄다**(리뷰 MAJOR). 최소 폭은 삼중조합 전수 열거로만
/// 정확하고, 그 비용이 감당되는 구간이 32점이다. 1024점을 허용하면 고정 support 방향에
/// 걸리지 않는 얇은 면(0.5mm bevel 등)이 스크린을 통과해 `MIN_SHAPE_EXTENT` 계약이
/// 깨진다 — **검증할 수 없는 것을 지원한다고 말하지 않는다.**
/// 더 복잡한 메시는 convex decomposition으로 여러 hull로 나눈다(v1 계약).
pub const MAX_HULL_POINTS: usize = 32;

/// 파생 상태 상한 (위치·속도·누적 임펄스·PCM).
///
/// **최악 임펄스 논증** (리뷰 BLOCKER: 앵커·각속도를 빠뜨렸던 부분을 포함해 재계산):
/// - 유효질량 `1/k ≤ 1/MIN_INV_MASS = 1e6` kg (k는 회전 항이 있어 항상 `≥ inv_mass`)
/// - 접점/앵커 속도 `|v + ω×r| ≤ MAX_SPEED + MAX_ANGULAR·MAX_SHAPE_EXTENT = 1.1e5` m/s
/// - 조인트·접촉 bias 속도는 `max_bias_speed ≤ MAX_SPEED = 1e4`로 클램프된다
///   (bias를 안 묶으면 `c_pos × (1/h)`가 1e12까지 간다)
///
/// 따라서 선형 임펄스 ≤ `1e6 × (2·1.1e5 + 1e4) ≈ 2.3e11`.
///
/// **각 임펄스**는 별도 경로다: `imp ≈ bias_ang / k_ang`, `k_ang ≥ MIN_INV_INERTIA = 1e-10`,
/// 각 bias는 각속도 상한(`MAX_ANGULAR = 1e2`)으로 클램프 → `imp ≤ 1e12`.
/// 두 경로 모두 `MAX_STATE`(1e15) 아래 최소 1000배 여유를 둔다.
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
/// 동적 바디의 역질량.
#[inline]
pub fn inv_mass_in_range(v: f32) -> bool {
    v.is_finite() && (MIN_INV_MASS..=MAX_INV_MASS).contains(&v)
}
/// 동적 바디의 역관성 성분 (차원이 달라 별도 범위).
#[inline]
pub fn inv_inertia_in_range(v: f32) -> bool {
    v.is_finite() && (MIN_INV_INERTIA..=MAX_INV_INERTIA).contains(&v)
}
/// 각속도.
#[inline]
pub fn angular_in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_ANGULAR
}
#[inline]
pub fn vec_angular_in_range(v: Vec3) -> bool {
    angular_in_range(v.x) && angular_in_range(v.y) && angular_in_range(v.z)
}
/// 조인트 앵커 (레버암).
#[inline]
pub fn anchor_in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_ANCHOR
}
#[inline]
pub fn vec_anchor_in_range(v: Vec3) -> bool {
    anchor_in_range(v.x) && anchor_in_range(v.y) && anchor_in_range(v.z)
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
