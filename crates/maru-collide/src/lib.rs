//! maru-collide — 셰이프·브로드페이즈·내로우페이즈 (M1).
//!
//! 설계 계약(02-architecture.md):
//! - §4.2: convex radius 1급(shrink-then-reinflate), GJK/EPA + 클리핑 매니폴드.
//! - §4.1: 듀얼 4-ary 트리, **페어 집합은 트리 비의존**(정밀 AABB + canonical 정렬이 정본).

mod aabb;
mod epa;
mod gjk;
mod pairs;
mod shape;
mod tree;

pub use aabb::Aabb;
pub use epa::{epa, Penetration};
pub use gjk::{gjk, GjkResult, Pose, SupportPoint};
pub use maru_math::Aabbx4;
pub use pairs::{Broadphase, STATIC_BIT};
pub use shape::Shape;
pub use tree::Tree;
