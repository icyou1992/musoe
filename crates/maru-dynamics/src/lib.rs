//! maru-dynamics — 강체 동역학 (M2).
//!
//! 설계 계약(02-architecture.md): §4.4 TGS-soft, §4.6 슬리핑, §8 결정론·denormal 위생, §9 바디 3종.

mod body;
mod character;
mod command;
mod joint;
mod snapshot;
mod solver;
mod solver_block;
mod threads;
mod world;

pub use body::{Body, BodyHandle, BodyType};
pub use character::{CharacterVirtual, MoveResult};
pub use command::{Command, CommandReport};
pub use joint::{Joint, JointKind, Motor};
pub use snapshot::{SnapshotError, SNAPSHOT_VERSION};
pub use solver::SolverParams;
pub use world::{
    ApiError, ContactEvent, ContactEventKind, Diagnostics, QueryFilter, RayHit, SweepHit, World,
    WorldParams,
};

#[cfg(test)]
mod tests;
