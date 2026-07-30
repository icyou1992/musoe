//! maru-collide — 셰이프·브로드페이즈·내로우페이즈 (M1).
//!
//! 설계 계약(02-architecture.md):
//! - §4.2: convex radius 1급(shrink-then-reinflate), GJK/EPA + 클리핑 매니폴드.
//! - §4.1: 듀얼 4-ary 트리, **페어 집합은 트리 비의존**(정밀 AABB + canonical 정렬이 정본).

mod aabb;
mod cast;
mod epa;
pub(crate) mod gjk;
mod heightfield;
mod limits;
mod manifold;
mod pairs;
mod ray;
mod shape;
mod tolerance;
mod tree;

pub use aabb::Aabb;
// **공개 표면은 셰이프 무관 디스패처만**이다.
//
// `contact`·`shape_cast`·`shape_cast_rotational`·`gjk`·`epa`·`support_core`는 볼록 전용이라
// 비볼록이 들어오면 의미가 없다. 그걸 `None`으로 돌려주면 "접촉 없음"과 구분되지 않아
// 호스트가 "지나갈 수 있다"고 오판하고(확인검수 rev_1a31d83f BLOCKER), 패닉으로 두면
// 라이브러리가 잘못된 입력에 죽는다. **셋째 길은 애초에 못 부르게 하는 것**이다 —
// 디스패처는 모든 셰이프를 옳게 처리하므로 밖에서 필요한 것은 이것뿐이다.
pub use cast::{
    try_shape_cast_dispatch, try_shape_cast_dispatch_rotational, CastReject, ShapeCastHit,
};
// **`gjk`·`epa`는 크레이트 밖으로 내보내지 않는다.**
//
// 둘은 볼록 전용이고 `support_core`를 직접 부르므로, 비볼록 셰이프가 들어오면
// `unreachable!`로 프로세스가 죽는다 — 공개 API가 잘못된 입력에 trap하는 것은 이 저장소의
// 계약 위반이다(확인검수 rev_e50670ad가 지적한 경로 중 하나였고, `contact`·`shape_cast`는
// `None` 반환으로 닫았지만 이 둘은 반환형에 "입력이 틀렸다"를 담을 자리가 없다).
// 크레이트 안에서는 디스패치가 볼록만 넘긴다는 것이 보장되고, 밖에서 필요한 것은
// `contact`/`shape_cast`/`for_each_contact`뿐이다 — 외부 사용처는 없다(전수 확인).
// (모듈 안에서 직접 경로로 쓴다 — 재수출하지 않는다.)

pub use gjk::{Pose, SupportPoint};
pub use heightfield::{
    build_tiles as build_heightfield_tiles, for_each_overlapping_cell, overlapping_cell_count,
    world_aabb_to_local, HeightfieldGrid, HeightfieldRef, COS_COPLANAR, HEIGHTFIELD_TILE,
    MAX_HEIGHTFIELD_CELLS_PER_QUERY, MAX_HEIGHTFIELD_MANIFOLDS, MAX_HEIGHTFIELD_SAMPLES,
};
pub use limits::{
    anchor_in_range, angular_in_range, extent_in_range, in_range, in_state_range,
    inv_inertia_in_range, inv_mass_in_range, saturate_state, saturate_state_vec, speed_in_range,
    vec_anchor_in_range, vec_angular_in_range, vec_extent_in_range, vec_in_range,
    vec_in_state_range, vec_speed_in_range, MAX_ANCHOR, MAX_ANGULAR, MAX_DT, MAX_HULL_POINTS,
    MAX_INPUT, MAX_SHAPE_EXTENT, MAX_SPEED, MAX_STATE, MIN_DT, MIN_INV_MASS, MIN_SHAPE_EXTENT,
};
pub use manifold::{
    any_contact, for_each_contact, for_each_contact_with_budget, weld_internal_edges, ContactPoint,
    ContactWalk, FeatureId, Manifold, FEATURE_ANALYTIC, FEATURE_CHILD_MASK, FEATURE_CHILD_SHIFT,
    FEATURE_GENERIC, MAX_MANIFOLD_POINTS,
};
pub use maru_math::Aabbx4;
pub use pairs::{Broadphase, MAX_PAIRS, STATIC_BIT};
// **공개 진입은 `try_*`뿐이다** — `cast_ray`·`contains_point`는 검증된 입력을 가정하는
// 크레이트 내부용이다. 잘못된 형상을 `None`/`false`로 돌려주면 "맞은 것 없음"으로 위장되고,
// 빈 hull은 GJK에서 프로세스를 죽인다(확인검수 rev_97887d75 BLOCKER).
pub use ray::{try_cast_ray, try_contains_point, QueryReject, RayCast};
pub use shape::{transformed_aabb, Shape, MAX_COMPOUND_CHILDREN};
pub use tolerance::{
    area_eps, bary_denom_eps, characteristic_length, length_eps, length_sq_eps, min_feature_size,
    volume_eps,
};
pub use tree::Tree;
