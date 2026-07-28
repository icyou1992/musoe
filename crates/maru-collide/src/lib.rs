//! maru-collide — 셰이프·브로드페이즈·내로우페이즈 (M1).
//!
//! 설계 계약(02-architecture.md):
//! - §4.2: convex radius 1급(shrink-then-reinflate), GJK/EPA + 클리핑 매니폴드.
//! - §4.1: 듀얼 4-ary 트리, **페어 집합은 트리 비의존**(정밀 AABB + canonical 정렬이 정본).

mod aabb;
mod cast;
mod epa;
pub(crate) mod gjk;
mod limits;
mod manifold;
mod pairs;
mod ray;
mod shape;
mod tolerance;
mod tree;

pub use aabb::Aabb;
pub use cast::{shape_cast, shape_cast_dispatch, ShapeCastHit};
pub use epa::{epa, Penetration};
pub use gjk::{gjk, GjkResult, Pose, SupportPoint};
pub use limits::{
    anchor_in_range, angular_in_range, extent_in_range, in_range, in_state_range,
    inv_inertia_in_range, inv_mass_in_range, saturate_state, saturate_state_vec, speed_in_range,
    vec_anchor_in_range, vec_angular_in_range, vec_extent_in_range, vec_in_range,
    vec_in_state_range, vec_speed_in_range, MAX_ANCHOR, MAX_ANGULAR, MAX_DT, MAX_HULL_POINTS,
    MAX_INPUT, MAX_SHAPE_EXTENT, MAX_SPEED, MAX_STATE, MIN_DT, MIN_INV_MASS, MIN_SHAPE_EXTENT,
};
pub use manifold::{
    any_contact, contact, for_each_contact, weld_internal_edges, ContactPoint, FeatureId, Manifold,
    FEATURE_ANALYTIC, FEATURE_CHILD_MASK, FEATURE_CHILD_SHIFT, FEATURE_GENERIC,
    MAX_MANIFOLD_POINTS,
};
pub use maru_math::Aabbx4;
pub use pairs::{Broadphase, MAX_PAIRS, STATIC_BIT};
pub use ray::{cast_ray, contains_point, RayCast};
pub use shape::{transformed_aabb, Shape, MAX_COMPOUND_CHILDREN};
pub use tolerance::{
    area_eps, bary_denom_eps, characteristic_length, length_eps, length_sq_eps, min_feature_size,
    volume_eps,
};
pub use tree::Tree;
