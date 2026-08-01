//! 접촉 매니폴드 생성 — face clipping + canonical feature ID (설계 §4.2).
//!
//! - GJK/EPA가 준 접촉 법선으로 reference/incident 면을 고르고(동률=낮은 면 인덱스,
//!   ref 소속 동률=A — 결정론), incident 면을 ref 측면 평면들로 클리핑해 최대 4점 생성.
//! - **feature ID는 단일 정본 함수**: 클리핑 계보(원본 정점/에지×평면)에서 결정적으로 부여.
//!   PCM 캐시 매칭·중복 판정 키(설계 §4)가 이 ID를 소비한다.
//! - box-box 면 접촉이 4점을 내는 것이 M1 완료 기준. 면 지배적이 아닌 접촉(에지-에지 등)과
//!   비박스 쌍은 witness 단일점(v1 명시 범위 — hull 면 클리핑은 hull 토폴로지 빌더와 함께).

use crate::epa::epa;
use crate::gjk::{gjk, GjkResult};
use crate::tolerance::{area_eps, characteristic_length, length_sq_eps, min_feature_size};
use crate::{Pose, Shape};
use maru_math::{Mat3, Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct FeatureId(pub u32);

#[derive(Clone, Copy, Debug, Default)]
pub struct ContactPoint {
    /// 월드 좌표 (B 표면 기준).
    pub position: Vec3,
    /// 부호 있는 간격: 음수 = 침투. (겉면 기준 — box는 half가 겉면)
    pub separation: f32,
    pub feature: FeatureId,
}

/// 매니폴드 접점 상한. 클리핑이 `reduce_to_4`로 4점을 넘기지 않고, 해석 경로는 1점이다.
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// 접촉 매니폴드.
///
/// 접점을 **인라인 배열**로 담는다(리뷰 MAJOR: 스텝 중 힙 할당). `Vec`이면 접촉마다
/// 할당이 하나씩 생기는데, 3000구 더미의 5,538 접촉이면 스텝당 수천 회다 —
/// 5,538 접촉 규모에서는 산술보다 allocator·포인터 추적이 지배할 수 있다.
#[derive(Clone, Copy, Debug)]
pub struct Manifold {
    /// A를 +normal로 밀면 분리 (GJK/EPA 관례 승계).
    pub normal: Vec3,
    points: [ContactPoint; MAX_MANIFOLD_POINTS],
    count: u8,
}

impl Default for Manifold {
    fn default() -> Self {
        Self {
            normal: Vec3::ZERO,
            points: [ContactPoint::default(); MAX_MANIFOLD_POINTS],
            count: 0,
        }
    }
}

impl Manifold {
    #[inline]
    pub fn points(&self) -> &[ContactPoint] {
        &self.points[..self.count as usize]
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    #[inline]
    pub fn push(&mut self, p: ContactPoint) {
        if (self.count as usize) < MAX_MANIFOLD_POINTS {
            self.points[self.count as usize] = p;
            self.count += 1;
        }
    }
    /// 테스트·조정용 가변 접근.
    #[inline]
    pub fn points_mut(&mut self) -> &mut [ContactPoint] {
        &mut self.points[..self.count as usize]
    }
    pub fn from_points(normal: Vec3, pts: &[ContactPoint]) -> Self {
        let mut m = Self {
            normal,
            ..Default::default()
        };
        for p in pts {
            m.push(*p);
        }
        m
    }
}

/// 면 지배 판정 임계: |면법선·접촉법선| 이 이 값 이상이면 face contact로 취급.
const FACE_DOT_MIN: f32 = 0.95;

/// 계보 없는 witness 접점 표식. 클리핑 접점의 feature 인코딩(상위 비트=ref_is_a, 면·계보)과
/// 겹치지 않는 예약값이며, 솔버는 이 값을 보면 **위치 근접**으로 캐시를 매칭한다.
pub const FEATURE_GENERIC: u32 = 0xFFFF_0000;
/// 전용(analytic) 경로가 만든 **안정적 단일 접점** ID. generic과 달리 스텝 간 동일 접촉이
/// 같은 ID를 유지하므로 솔버가 위치 근접 게이트 없이 ID로 warm start를 잇는다.
pub const FEATURE_ANALYTIC: u32 = 0xFFFE_0000;

/// 클리핑 정점의 계보 (리뷰 MAJOR 반영 — 구 태그 재사용 인코딩은 다단 클리핑에서 서로 다른
/// 계보가 같은 태그로 충돌했다): incident 원본 정점 비트합 ∪ 관여한 reference 측면 비트합.
#[derive(Clone, Copy)]
struct ClipVertex {
    position: Vec3,
    incident_mask: u8,
    reference_mask: u8,
}

/// 접촉 생성 진입점. `margin` = speculative margin (양의 separation 허용 한계).
/// 복합 자식 feature 네임스페이스 슬롯 — bits 9..15 (7비트).
/// 모든 feature 생산자가 이 비트를 쓰지 않는다: 클립 패킹은 31·28..30·25..27·0..8,
/// 센티널(FEATURE_GENERIC/ANALYTIC)은 16..31만 쓴다. 그래서 `fid | (key<<9)`는
/// key(=자식쌍 식별, 1..=64)별로 **단사**다 — 같은 바디 페어의 서로 다른 자식쌍이
/// 같은 feature를 내도 warm start·PCM 매칭이 절대 섞이지 않는다(설계: 중복 판정 키에
/// source collider 포함 — PCM 키 무변경 달성).
pub const FEATURE_CHILD_SHIFT: u32 = 9;
pub const FEATURE_CHILD_MASK: u32 = 0x7F << FEATURE_CHILD_SHIFT;

impl Manifold {
    /// 자식쌍 키를 전 접점 feature에 접는다. 비복합 경로는 호출하지 않는다(key 0 예약).
    /// A/B 역할을 뒤집는다 — 법선 방향 계약(“A를 +normal로 밀면 분리”)을 지키려면
    /// 법선을 뒤집는 것으로 충분하다(접점 위치·침투는 대칭이다).
    pub fn flip(&mut self) {
        self.normal = -self.normal;
    }

    pub fn namespace_features(&mut self, key: u32) {
        debug_assert!((1..=64).contains(&key), "자식쌍 키는 1..=64");
        for i in 0..self.count as usize {
            debug_assert!(
                self.points[i].feature.0 & FEATURE_CHILD_MASK == 0,
                "feature 생산자가 네임스페이스 비트(9..15)를 침범했다"
            );
            self.points[i].feature.0 |= key << FEATURE_CHILD_SHIFT;
        }
    }
}

pub mod feature;

mod clip;
mod container;
mod dispatch;
mod genclip;
mod sat;
#[cfg(test)]
mod tests;
mod weld;

pub use dispatch::*;
pub use weld::*;
