//! **내부 에지 용접** — 이웃 삼각형과 공유하는 에지에서 나온 법선을 면 법선으로 돌린다.
//!
//! 안 하면 평평한 지형을 미끄러지는 물체가 삼각형 경계마다 걸린다.

use super::*;

/// **지형 내부 에지 용접** (§4.2 ghost edge, Jolt ActiveEdges 동형).
///
/// 삼각형 지형은 삼각형마다 독립 바디라, 이웃과 **공면으로 공유하는 에지**(내부 에지)의
/// 둥근 밴드에서 접촉이 생기면 법선이 면이 아니라 에지에서 나온다 — 평평한 바닥을
/// 구르는 구가 이음새마다 옆으로 차이는 유령 충돌. 등록부(try_add_terrain)가 인접성 +
/// 공면성으로 내부 에지를 판정해 마스크를 주면, 여기서 그 에지 근방 접점의 법선을
/// 면 법선으로 스냅한다.
///
/// 보정 규칙(전부 보수적 — 확신 없으면 무변경):
/// - 법선이 이미 면 법선 원뿔 안(±5°)이면 무변경 (면 접촉은 원래 맞다).
/// - 접점 중 하나라도 내부 에지 세그먼트에서 `radius + tol` 안이어야 스냅한다 —
///   외부 에지·꼭짓점 접촉은 그대로 (경계에서는 에지 법선이 물리적으로 옳다).
/// - 면 법선 부호는 현재 법선 쪽으로 정렬한다(지형은 양면 — A가 어느 쪽이든 성립).
/// - separation은 재계산하지 않는다: 오차는 O(θ·침투)이고 소프트 위치 보정이 흡수한다.
///   (Jolt도 거리 유지 — 방향만 고친다.)
///
/// 볼록 능선·오목 골짜기는 등록부가 마스크를 세우지 않으므로 여기 오지 않는다.
pub fn weld_internal_edges(m: &mut Manifold, tri: [Vec3; 3], radius: f32, internal: u8) {
    if internal == 0 || m.is_empty() {
        return;
    }
    let n_face = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
    let scale = (tri[1] - tri[0])
        .length()
        .max((tri[2] - tri[0]).length())
        .max((tri[2] - tri[1]).length());
    // cross 길이는 L² — 면적 차원 임계로 퇴화 방지(등록부가 이미 거르지만 방어).
    let len = n_face.length();
    if len <= crate::tolerance::area_eps(scale) {
        return;
    }
    let n_face = n_face * (1.0 / len);
    let n_signed = if n_face.dot(m.normal) >= 0.0 {
        n_face
    } else {
        -n_face
    };
    // 5° 원뿔 — Jolt ActiveEdges 기본 임계와 동급. 면 접촉은 여기서 끝난다.
    const COS_FACE: f32 = 0.996;
    if m.normal.dot(n_signed) >= COS_FACE {
        return;
    }
    let band = radius + crate::tolerance::length_eps(scale);
    let near_internal = m.points().iter().any(|p| {
        (0..3u8).any(|i| {
            internal & (1 << i) != 0
                && dist_point_segment(p.position, tri[i as usize], tri[((i + 1) % 3) as usize])
                    <= band
        })
    });
    if near_internal {
        m.normal = n_signed;
    }
}

/// 점-세그먼트 거리 (용접 밴드 판정 전용).
pub(super) fn dist_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let denom = ab.length_sq();
    // 퇴화 세그먼트(등록부가 거르지만 방어): 점 거리로.
    if denom <= 0.0 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / denom).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}
