//! 엔진 전역 크기 한계 — **라이브 API와 역직렬화가 같은 정책을 쓴다**(리뷰 BLOCKER).
//!
//! 스냅샷 리더만 상한을 걸면 비대칭이 생긴다: 위치가 `(1e10,0,0)`인 바디는 정상적으로
//! 생성·추가되고 `snapshot()`도 성공하지만 그 바이트는 리더가 거부해
//! `restore(snapshot())` 계약이 깨진다. 그래서 상한을 **생성·변이 경계에도 동일하게** 건다.

/// 물리량 크기 상한.
///
/// 근거: 모든 |값| ≤ 1e9이면 솔버의 곱(임펄스×역질량, r×임펄스, effective mass)이 1e27
/// 수준에 머물러 f32 범위(3.4e38) 안에 남고, 유한 개수의 합도 오버플로하지 않는다.
/// `f32::MAX`는 **유한하지만** 첫 곱에서 Inf가 되어 상태를 즉시 오염시킨다 —
/// `is_finite()`만으로는 방어가 되지 않는다.
///
/// 단위 해석: 길이 m, 속도 m/s, 역질량 1/kg. 1e9 m는 지구-달 거리의 2배이고
/// 1e9 m/s는 광속의 3배다 — 물리 시뮬레이션의 정상 범위를 훨씬 넘는다.
pub const MAX_MAGNITUDE: f32 = 1.0e9;

/// hull 정점 수 상한. support 함수가 O(n) 선형 스캔이라 성능 상한이기도 하고,
/// 스냅샷 writer가 만드는 바이트를 묶는 자원 상한이기도 하다
/// (수백만 점 hull은 writer에서 이미 수십 MB를 만든 뒤에야 리더가 거부하게 된다).
pub const MAX_HULL_POINTS: usize = 1024;

/// 물리량 스칼라가 정책 안인가 (유한 + 크기).
#[inline]
pub fn in_range(v: f32) -> bool {
    v.is_finite() && v.abs() <= MAX_MAGNITUDE
}

/// 벡터 성분 전부가 정책 안인가.
#[inline]
pub fn vec_in_range(v: maru_math::Vec3) -> bool {
    in_range(v.x) && in_range(v.y) && in_range(v.z)
}
