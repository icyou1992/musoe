//! 커맨드 버퍼 — 쓰기의 정본 경로 (설계 §7).
//!
//! 계약:
//! - **쓰기는 배치**다. 호스트(JS)는 커맨드를 큐에 쌓고, 스텝 0단계나 `flush_commands()`에서
//!   **enqueue 순서대로** 적용된다. 순서가 곧 결정론의 근거다(설계 §8 호출 순서 계약).
//! - 각 커맨드는 **단일 원자 절차**로 실행된다(설계 §4.6):
//!   ① 변이 전 그래프에서 wake 대상 수집·적용 → ② 구조 변경 → ③ 조인트·PCM 인접 정리.
//! - **쿼리는 마지막 step 완료 상태를 본다.** pending 커맨드는 보이지 않는다 —
//!   즉시 반영이 필요하면 `flush_commands()`를 명시 호출한다(암묵 플러시 없음).
//! - pending 큐는 **캐노니컬 상태**다: 스냅샷에 포함되어 restore 후 재생이 원 실행과 일치한다.

use crate::body::{Body, BodyHandle};
use crate::joint::Joint;
use maru_collide::Pose;
use maru_math::Vec3;

/// **공개 커맨드 스키마** — 호스트(JS)가 보낼 수 있는 것만 담는다.
///
/// 바디 추가가 여기 없는 것은 의도다(리뷰 MINOR): 바디 생성은 슬롯 예약과 커맨드 등록이
/// **한 원자 연산**이어야 하고(예약 핸들을 즉시 돌려줘야 하므로), 예약 슬롯당 pending
/// 추가는 정확히 하나여야 한다. 공개 열거형에 태그로 노출하면 호스트가 합법적인 명령으로
/// 이해하고 보내지만 항상 거부되는 반쪽 variant가 된다. 정본 경로는 `World::enqueue_add_body`.
#[derive(Clone, Debug)]
pub enum Command {
    RemoveBody(BodyHandle),
    SetLinearVelocity(BodyHandle, Vec3),
    SetAngularVelocity(BodyHandle, Vec3),
    SetPose(BodyHandle, Pose),
    SetMaterial(BodyHandle, f32, f32),
    AddJoint(Box<Joint>),
}

/// **내부 pending 큐의 원소.** 공개 스키마에 없는 바디 추가를 포함한다 —
/// 큐는 캐노니컬 상태라 직렬화 대상이므로 표현이 필요하지만, 호스트에게 열지는 않는다.
#[derive(Clone, Debug)]
pub enum PendingCommand {
    /// 예약된 핸들 + 채울 바디. `enqueue_add_body`만 만든다.
    Add(BodyHandle, Box<Body>),
    Public(Command),
}

/// 커맨드 적용 결과 — 실패한 커맨드는 **드롭되고 카운트된다**(스텝을 중단시키지 않는다:
/// 설계 §3 스텝 원자성). 호스트는 카운터로 오류를 관측한다.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct CommandReport {
    pub applied: usize,
    pub rejected: usize,
}
