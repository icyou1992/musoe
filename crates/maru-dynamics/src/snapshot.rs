//! 캐노니컬 스냅샷 + L0 상태 해시 (설계 §3·§8, M2f).
//!
//! 포함: 바디(자세·속도·질량·셰이프·재질) · 슬롯 생존/예약·세대 · 슬립 상태 · 아일랜드 대표 ·
//! PCM 임펄스 캐시(정렬 직렬화) · 조인트(누적 임펄스 포함) · pending 커맨드 · 직전 커맨드 리포트.
//! **브로드페이즈 트리는 제외** — 순수 파생 가속 구조(설계 §4.1 "페어 집합 트리 비의존" 계약).
//!
//! 계약: `restore(snapshot())` 후 재생은 원 실행과 **같은 타깃에서 비트 동일**(L0).
//!
//! # 신뢰하지 않는 입력 (리뷰 BLOCKER ×2)
//!
//! M3에서 JS가 넘기는 바이트는 짧거나 조작될 수 있다. 두 층으로 방어한다:
//!
//! 0. **입력 총 바이트와 디코딩 후 힙에 각각 상한을 둔다.** 길이 유도 상한만으로는
//!    부족하다 — `RemoveBody` 최소 레코드는 9바이트인데 `Command` 원소는 그보다 수배 크므로,
//!    수십 MB짜리 **유효한** 커맨드 스트림이 wasm 선형 메모리에서 수백 MB를 요구할 수 있다
//!    (리뷰 PARTIAL BLOCKER). count는 남은 바이트뿐 아니라 **누적 디코딩 예산**에도 과금된다.
//! 1. **자원 예산은 입력 길이에서 유도한다.** 고정 상한(`1<<22` 등)은 12바이트짜리 파일이
//!    400만 개를 주장하고 수 GB를 할당하게 둔다. 모든 count는 "남은 바이트 ÷ 레코드
//!    최소 크기"를 넘을 수 없다 — 산술적으로 불가능한 개수는 읽기 전에 거부된다.
//!    hull 점·캐시 엔트리처럼 중첩된 개수도 같은 규칙이라 **누적 총량도 자동으로** 묶인다.
//! 2. **구조·타입 관계를 검증한다.** 유한성·범위만으로는 죽은 슬롯을 가리키는 조인트,
//!    inv_mass=0인 dynamic, 질량을 가진 static, 죽은 아일랜드 대표, 음수 법선 임펄스처럼
//!    엔진 불변식을 깨는 상태가 그대로 들어온다. 전부 거부한다.
//!
//! 검증이 끝나기 전에는 월드를 건드리지 않는다(원자적 복원).

use crate::body::{Body, BodyHandle, BodyType};
use crate::command::{Command, CommandReport, PendingCommand};
use crate::joint::{Joint, JointKind};
use crate::world::World;
use maru_collide::{Pose, Shape};
use maru_math::{Quat, Vec3};

/// 포맷 매직·버전 — 다른 빌드/포맷의 바이트를 조용히 먹지 않기 위해.
const MAGIC: u32 = 0x4d41_5255; // "MARU"
/// v3: 슬롯 예약 플래그 · AddBody 핸들 · last_report · 셰이프 원점 오프셋.
const VERSION: u32 = 3;

/// 입력 자체의 상한. 정상 월드(3k 바디 ≈ 300KB)의 200배 이상 여유가 있으면서,
/// 디코딩 증폭이 선형 메모리를 삼키지 못하게 한다.
const MAX_SNAPSHOT_BYTES: usize = 64 << 20;
/// **디코딩 후** 힙 예산. count × 원소 크기를 누적 과금한다.
const MAX_DECODED_BYTES: usize = 192 << 20;

// 크기 상한은 `maru_collide::limits`가 정본이다. `f32::MAX`처럼 **유한하지만 거대한** 값은
// 유한성 검사를 통과한 뒤 첫 곱에서 Inf가 되므로 `is_finite()`만으로는 방어가 되지 않는다.

/// 레코드 **최소** 직렬화 크기 — count 검증의 근거. 실제 레코드는 이보다 크거나 같다.
mod min_bytes {
    /// 셰이프 최소: 태그(1) + Sphere 반지름(4)
    const SHAPE: usize = 5;
    /// 바디 본문: 타입1 + pos12 + rot16 + lin12 + ang12 + invM4 + invI12 + fric4 + rest4 + offset12
    const BODY_CORE: usize = 1 + 12 + 16 + 12 + 12 + 4 + 12 + 4 + 4 + 12 + SHAPE;
    /// 슬롯 레코드: alive1 + reserved1 + gen4 + 본문 + sleep4 + asleep1 + island4
    pub const SLOT: usize = 1 + 1 + 4 + BODY_CORE + 4 + 1 + 4;
    /// 조인트: 핸들 8+8 + 앵커 24 + kind태그1 + collide1 + lin12 + ang8
    pub const JOINT: usize = 8 + 8 + 24 + 1 + 1 + 12 + 8;
    /// 커맨드: 태그1 + 최소 페이로드(RemoveBody = 핸들8)
    pub const COMMAND: usize = 1 + 8;
    /// 캐시 페어: 키8 + 엔트리 수4
    pub const CACHE_PAIR: usize = 8 + 4;
    /// 캐시 엔트리: fid4 + normal4 + tangent8 + tangent_world12 + position12 + 법선12
    pub const CACHE_ENTRY: usize = 4 + 4 + 8 + 12 + 12 + 12;
    /// hull 점 12 + (뒤따르는 반지름 4는 개수와 무관하므로 제외)
    pub const HULL_POINT: usize = 12;
}

/// 디코딩 후 원소가 힙에서 차지하는 크기 — count 과금용.
mod decoded_size {
    use super::*;
    pub fn slot() -> usize {
        // Body + 슬롯 병렬 배열(alive/reserved/gen/sleep/asleep/island)
        core::mem::size_of::<Body>() + 16
    }
    pub fn joint() -> usize {
        core::mem::size_of::<Joint>()
    }
    pub fn command() -> usize {
        core::mem::size_of::<Command>()
    }
    pub fn cache_pair() -> usize {
        // HashMap 엔트리 + Vec 헤더
        64
    }
    pub fn cache_entry() -> usize {
        core::mem::size_of::<(maru_collide::FeatureId, crate::solver::CachedImpulse)>()
    }
    pub fn hull_point() -> usize {
        core::mem::size_of::<Vec3>()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SnapshotError {
    BadMagic,
    BadVersion(u32),
    Truncated,
    TrailingBytes,
    /// 남은 바이트로는 담을 수 없는 개수, 또는 총 바이트·디코딩 예산 초과.
    CountTooLarge,
    NonFinite,
    BadIndex,
    BadShape,
    /// 유한·범위는 맞지만 엔진 불변식을 깨는 조합.
    InconsistentState,
}

pub struct Writer(pub Vec<u8>);

impl Writer {
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn vec3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn quat(&mut self, q: Quat) {
        self.f32(q.x);
        self.f32(q.y);
        self.f32(q.z);
        self.f32(q.w);
    }
    fn handle(&mut self, h: BodyHandle) {
        self.u32(h.index);
        self.u32(h.generation);
    }
}

pub struct Reader<'a> {
    b: &'a [u8],
    at: usize,
    /// 지금까지 과금된 디코딩 후 힙 바이트.
    decoded: usize,
}

impl Reader<'_> {
    #[inline]
    fn take(&mut self, n: usize) -> Result<&[u8], SnapshotError> {
        let end = self.at.checked_add(n).ok_or(SnapshotError::Truncated)?;
        if end > self.b.len() {
            return Err(SnapshotError::Truncated);
        }
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    #[inline]
    fn remaining(&self) -> usize {
        self.b.len() - self.at
    }
    /// 호스트 입력에서 온 물리량 (앵커·마찰 등) — **입력 정책**.
    fn f32(&mut self) -> Result<f32, SnapshotError> {
        let v = self.raw_f32()?;
        if maru_collide::in_range(v) {
            Ok(v)
        } else {
            Err(SnapshotError::NonFinite)
        }
    }
    /// 엔진이 진화시키는 물리량 (위치·속도·누적 임펄스·PCM 캐시) — **상태 정책**.
    /// 솔버는 큰 질량과 큰 접근 속도의 조합에서 입력 상한을 넘는 임펄스를 정당하게 만든다.
    fn state_f32(&mut self) -> Result<f32, SnapshotError> {
        let v = self.raw_f32()?;
        if maru_collide::in_state_range(v) {
            Ok(v)
        } else {
            Err(SnapshotError::NonFinite)
        }
    }
    fn raw_f32(&mut self) -> Result<f32, SnapshotError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take(1)?[0])
    }
    fn vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    fn state_vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(
            self.state_f32()?,
            self.state_f32()?,
            self.state_f32()?,
        ))
    }
    fn quat(&mut self) -> Result<Quat, SnapshotError> {
        let q = Quat {
            x: self.raw_f32()?,
            y: self.raw_f32()?,
            z: self.raw_f32()?,
            w: self.raw_f32()?,
        };
        if !(q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()) {
            return Err(SnapshotError::NonFinite);
        }
        // 단위 쿼터니언이 아니면 자세가 조용히 붕괴한다
        let l2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if !(0.9..=1.1).contains(&l2) {
            return Err(SnapshotError::NonFinite);
        }
        Ok(q.normalize())
    }
    fn handle(&mut self) -> Result<BodyHandle, SnapshotError> {
        Ok(BodyHandle {
            index: self.u32()?,
            generation: self.u32()?,
        })
    }
    /// **입력 길이에서 유도한 상한**으로 개수를 검증한다(고정 상수 금지 — 리뷰 BLOCKER).
    /// n개를 담으려면 최소 `n * min_each` 바이트가 남아 있어야 한다.
    fn count(&mut self, min_each: usize, decoded_each: usize) -> Result<usize, SnapshotError> {
        let n = self.u32()? as usize;
        let need = n
            .checked_mul(min_each)
            .ok_or(SnapshotError::CountTooLarge)?;
        if need > self.remaining() {
            return Err(SnapshotError::CountTooLarge);
        }
        // 디코딩 후 힙에도 과금한다 — 입력 길이만으로는 증폭을 못 막는다
        let heap = n
            .checked_mul(decoded_each)
            .ok_or(SnapshotError::CountTooLarge)?;
        self.decoded = self
            .decoded
            .checked_add(heap)
            .ok_or(SnapshotError::CountTooLarge)?;
        if self.decoded > MAX_DECODED_BYTES {
            return Err(SnapshotError::CountTooLarge);
        }
        Ok(n)
    }
}

fn write_shape(w: &mut Writer, s: &Shape) {
    match s {
        Shape::Sphere { radius } => {
            w.u8(0);
            w.f32(*radius);
        }
        Shape::Capsule {
            half_height,
            radius,
        } => {
            w.u8(1);
            w.f32(*half_height);
            w.f32(*radius);
        }
        Shape::Box { half, radius } => {
            w.u8(2);
            w.vec3(*half);
            w.f32(*radius);
        }
        Shape::Hull { points, radius } => {
            w.u8(3);
            w.u32(points.len() as u32);
            for p in points {
                w.vec3(*p);
            }
            w.f32(*radius);
        }
    }
}

fn read_shape(r: &mut Reader) -> Result<Shape, SnapshotError> {
    let shape = match r.u8()? {
        0 => Shape::Sphere { radius: r.f32()? },
        1 => Shape::Capsule {
            half_height: r.f32()?,
            radius: r.f32()?,
        },
        2 => Shape::Box {
            half: r.vec3()?,
            radius: r.f32()?,
        },
        3 => {
            // 점 개수도 남은 바이트로 묶인다 → 전체 파일 크기가 곧 총 점 예산이다
            let n = r.count(min_bytes::HULL_POINT, decoded_size::hull_point())?;
            let mut points = Vec::with_capacity(n);
            for _ in 0..n {
                points.push(r.vec3()?);
            }
            Shape::Hull {
                points,
                radius: r.f32()?,
            }
        }
        _ => return Err(SnapshotError::BadShape),
    };
    // 형상 불변식까지 통과해야 월드에 들인다
    shape.validate().map_err(|_| SnapshotError::BadShape)?;
    Ok(shape)
}

fn write_body(w: &mut Writer, b: &Body) {
    w.u8(match b.body_type {
        BodyType::Static => 0,
        BodyType::Dynamic => 1,
        BodyType::Kinematic => 2,
    });
    w.vec3(b.pose.pos);
    w.quat(b.pose.rot);
    w.vec3(b.linvel);
    w.vec3(b.angvel);
    w.f32(b.inv_mass);
    w.vec3(b.inv_inertia_local);
    w.f32(b.friction);
    w.f32(b.restitution);
    w.vec3(b.shape_origin_offset);
    write_shape(w, &b.shape);
}

fn read_body(r: &mut Reader) -> Result<Body, SnapshotError> {
    let body_type = match r.u8()? {
        0 => BodyType::Static,
        1 => BodyType::Dynamic,
        2 => BodyType::Kinematic,
        _ => return Err(SnapshotError::BadShape),
    };
    let pose = Pose {
        pos: r.state_vec3()?, // 위치는 적분으로 진화한다 → 상태 정책(월드 밖 이탈 허용)
        rot: r.quat()?,
    };
    // 속도는 **파생 상태**다: 겹침 해소·충격은 정상 물리로도 입력 상한을 넘긴다.
    // 그래서 리더는 상태 정책으로 받고, 스텝 시작의 속도 상한(물리 파라미터)이
    // 솔버 산술에 들어가기 전에 잡는다.
    let linvel = r.state_vec3()?;
    let angvel = r.state_vec3()?;
    let inv_mass = r.f32()?;
    let inv_inertia_local = r.vec3()?;
    let friction = r.f32()?;
    let restitution = r.f32()?;
    let shape_origin_offset = r.vec3()?;
    let shape = read_shape(r)?;
    if inv_mass < 0.0
        || friction < 0.0
        || !(0.0..=1.0).contains(&restitution)
        || inv_inertia_local.x < 0.0
        || inv_inertia_local.y < 0.0
        || inv_inertia_local.z < 0.0
    {
        return Err(SnapshotError::NonFinite);
    }
    // **타입과 질량의 관계**를 강제한다(리뷰 BLOCKER): inv_mass=0인 dynamic은 솔버에서
    // 0으로 나뉘지는 않지만 중력만 받고 제약은 못 받는 유령이 되고, 질량을 가진 static은
    // 임펄스를 흡수해 벽이 밀린다.
    let zero_mass = inv_mass == 0.0 && inv_inertia_local == Vec3::ZERO;
    match body_type {
        BodyType::Dynamic => {
            // 질량을 **양쪽**으로 묶는다: 너무 작은 inv_mass(=거대 질량)는 조인트 솔버의
            // K⁻¹를 폭발시켜 중간 연산에서 Inf를 만든다(리뷰 BLOCKER 공격 시나리오).
            if !(maru_collide::inv_mass_in_range(inv_mass)
                && maru_collide::inv_mass_in_range(inv_inertia_local.x)
                && maru_collide::inv_mass_in_range(inv_inertia_local.y)
                && maru_collide::inv_mass_in_range(inv_inertia_local.z))
            {
                return Err(SnapshotError::InconsistentState);
            }
        }
        BodyType::Static | BodyType::Kinematic => {
            if !zero_mass {
                return Err(SnapshotError::InconsistentState);
            }
        }
    }
    Ok(Body {
        body_type,
        pose,
        linvel,
        angvel,
        shape,
        inv_mass,
        inv_inertia_local,
        friction,
        restitution,
        shape_origin_offset,
    })
}

fn write_joint(w: &mut Writer, j: &Joint) {
    w.handle(j.body_a);
    w.handle(j.body_b);
    w.vec3(j.anchor_a);
    w.vec3(j.anchor_b);
    match &j.kind {
        JointKind::Point => w.u8(0),
        JointKind::Hinge { axis_a, axis_b } => {
            w.u8(1);
            w.vec3(*axis_a);
            w.vec3(*axis_b);
        }
    }
    w.u8(j.collide_connected as u8);
    w.vec3(j.linear_impulse);
    w.f32(j.angular_impulse[0]);
    w.f32(j.angular_impulse[1]);
}

fn read_joint(r: &mut Reader) -> Result<Joint, SnapshotError> {
    let body_a = r.handle()?;
    let body_b = r.handle()?;
    let anchor_a = r.vec3()?;
    let anchor_b = r.vec3()?;
    let kind = match r.u8()? {
        0 => JointKind::Point,
        1 => {
            let (axis_a, axis_b) = (r.vec3()?, r.vec3()?);
            // 축은 생성 경계에서 정규화된다 — 0이나 비정규 축은 각 제약을 조용히 없앤다
            if !(0.9..=1.1).contains(&axis_a.length_sq())
                || !(0.9..=1.1).contains(&axis_b.length_sq())
            {
                return Err(SnapshotError::InconsistentState);
            }
            JointKind::Hinge { axis_a, axis_b }
        }
        _ => return Err(SnapshotError::BadShape),
    };
    let collide_connected = r.u8()? != 0;
    let linear_impulse = r.state_vec3()?;
    let angular_impulse = [r.state_f32()?, r.state_f32()?];
    Ok(Joint {
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        kind,
        collide_connected,
        linear_impulse,
        angular_impulse,
    })
}

fn write_command(w: &mut Writer, c: &PendingCommand) {
    let c = match c {
        PendingCommand::Add(h, b) => {
            w.u8(0);
            w.handle(*h);
            write_body(w, b);
            return;
        }
        PendingCommand::Public(c) => c,
    };
    match c {
        Command::RemoveBody(h) => {
            w.u8(1);
            w.handle(*h);
        }
        Command::SetLinearVelocity(h, v) => {
            w.u8(2);
            w.handle(*h);
            w.vec3(*v);
        }
        Command::SetAngularVelocity(h, v) => {
            w.u8(3);
            w.handle(*h);
            w.vec3(*v);
        }
        Command::SetPose(h, p) => {
            w.u8(4);
            w.handle(*h);
            w.vec3(p.pos);
            w.quat(p.rot);
        }
        Command::SetMaterial(h, f, rest) => {
            w.u8(5);
            w.handle(*h);
            w.f32(*f);
            w.f32(*rest);
        }
        Command::AddJoint(j) => {
            w.u8(6);
            write_joint(w, j);
        }
    }
}

/// 직렬화 크기를 **쓰기 전에** 계산한다(리뷰 BLOCKER: 다 만든 뒤 길이를 재면 wasm에서
/// 통제된 오류보다 메모리 고갈이 먼저 온다). 아래 `write_*`와 **같은 레이아웃**을 세므로
/// 한쪽을 고치면 다른 쪽도 고쳐야 한다 — 그 일치는 `snapshot_size_preflight_is_exact`가 잠근다.
fn shape_size(s: &Shape) -> usize {
    1 + match s {
        Shape::Sphere { .. } => 4,
        Shape::Capsule { .. } => 8,
        Shape::Box { .. } => 16,
        Shape::Hull { points, .. } => 4 + points.len() * 12 + 4,
    }
}

fn body_size(b: &Body) -> usize {
    1 + 12 + 16 + 12 + 12 + 4 + 12 + 4 + 4 + 12 + shape_size(&b.shape)
}

fn joint_size(j: &Joint) -> usize {
    8 + 8
        + 24
        + match j.kind {
            JointKind::Point => 1,
            JointKind::Hinge { .. } => 1 + 24,
        }
        + 1
        + 12
        + 8
}

fn command_size(c: &PendingCommand) -> usize {
    1 + match c {
        PendingCommand::Add(_, b) => 8 + body_size(b),
        PendingCommand::Public(Command::RemoveBody(_)) => 8,
        PendingCommand::Public(Command::SetLinearVelocity(..))
        | PendingCommand::Public(Command::SetAngularVelocity(..)) => 8 + 12,
        PendingCommand::Public(Command::SetPose(..)) => 8 + 12 + 16,
        PendingCommand::Public(Command::SetMaterial(..)) => 8 + 8,
        PendingCommand::Public(Command::AddJoint(j)) => joint_size(j),
    }
}

/// 스냅샷 바이트 수 (할당 전 preflight).
pub fn serialized_size(world: &World) -> usize {
    let (bodies, _, _, _, _, _, _, joints, cache, commands, _) = world.canonical_parts();
    let mut n = 4 + 4 + 4; // magic + version + slot count
    for (i, b) in bodies.iter().enumerate() {
        let _ = i;
        n += 1 + 1 + 4 + body_size(b) + 4 + 1 + 4;
    }
    n += 4 + joints.iter().map(joint_size).sum::<usize>();
    n += 4 + commands.iter().map(command_size).sum::<usize>();
    n += 4 + 4; // last_report
    n += 4;
    for entry in cache.values() {
        n += 4 + 4 + 4 + entry.len() * (4 + 4 + 8 + 12 + 12 + 12);
    }
    n
}

/// **writer도 총량 정책을 강제한다**(리뷰 BLOCKER: 리더만 막으면 비대칭).
/// 크기는 **할당 전에** 판정하고, 통과하면 정확한 용량으로 한 번에 잡는다.
pub fn write_world(world: &World, w: &mut Writer) -> Result<(), SnapshotError> {
    let size = serialized_size(world);
    if size > MAX_SNAPSHOT_BYTES {
        return Err(SnapshotError::CountTooLarge);
    }
    w.0.reserve_exact(size);
    write_world_bytes(world, w);
    debug_assert_eq!(w.0.len(), size, "preflight 크기 계산이 writer와 어긋남");
    Ok(())
}

fn write_world_bytes(world: &World, w: &mut Writer) {
    w.u32(MAGIC);
    w.u32(VERSION);
    let (
        bodies,
        alive,
        reserved,
        generations,
        sleep_timer,
        asleep,
        island,
        joints,
        cache,
        commands,
        report,
    ) = world.canonical_parts();
    w.u32(bodies.len() as u32);
    for (i, b) in bodies.iter().enumerate() {
        w.u8(alive[i] as u8);
        w.u8(reserved[i] as u8);
        w.u32(generations[i]);
        write_body(w, b);
        w.f32(sleep_timer[i]);
        w.u8(asleep[i] as u8);
        w.u32(island[i]); // 아일랜드 대표 — wake 전파가 restore 후에도 동일해야 L0 유지
    }
    w.u32(joints.len() as u32);
    for j in joints {
        write_joint(w, j);
    }
    // pending 커맨드 — **enqueue 순서 그대로** (순서가 결정론의 근거, 설계 §7)
    w.u32(commands.len() as u32);
    for c in commands {
        write_command(w, c);
    }
    w.u32(report.applied as u32);
    w.u32(report.rejected as u32);

    // PCM 캐시 — canonical 키 정렬 직렬화 (설계 §8 결정론)
    let mut keys: Vec<(u32, u32)> = cache.keys().copied().collect();
    keys.sort_unstable();
    w.u32(keys.len() as u32);
    for k in keys {
        w.u32(k.0);
        w.u32(k.1);
        let entry = &cache[&k];
        w.u32(entry.len() as u32);
        for (fid, imp) in entry {
            w.u32(fid.0);
            w.f32(imp.normal);
            w.f32(imp.tangent[0]);
            w.f32(imp.tangent[1]);
            w.vec3(imp.tangent_world);
            w.vec3(imp.position);
            w.vec3(imp.contact_normal);
        }
    }
}

/// 신뢰하지 않는 바이트에서 월드 복원. **검증이 끝나기 전에는 월드를 건드리지 않는다.**
pub fn read_world(world: &mut World, bytes: &[u8]) -> Result<(), SnapshotError> {
    if bytes.len() > MAX_SNAPSHOT_BYTES {
        return Err(SnapshotError::CountTooLarge);
    }
    let mut r = Reader {
        b: bytes,
        at: 0,
        decoded: 0,
    };
    if r.u32()? != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let v = r.u32()?;
    if v != VERSION {
        return Err(SnapshotError::BadVersion(v));
    }
    let nb = r.count(min_bytes::SLOT, decoded_size::slot())?;
    let mut bodies = Vec::with_capacity(nb);
    let mut sleep_timer = Vec::with_capacity(nb);
    let mut asleep = Vec::with_capacity(nb);
    let mut island = Vec::with_capacity(nb);
    let mut alive = Vec::with_capacity(nb);
    let mut reserved = Vec::with_capacity(nb);
    let mut generations = Vec::with_capacity(nb);
    for _ in 0..nb {
        alive.push(r.u8()? != 0);
        reserved.push(r.u8()? != 0);
        generations.push(r.u32()?);
        bodies.push(read_body(&mut r)?);
        let t = r.f32()?;
        if t < 0.0 {
            return Err(SnapshotError::NonFinite);
        }
        sleep_timer.push(t);
        asleep.push(r.u8()? != 0);
        island.push(r.u32()?);
    }

    let nj = r.count(min_bytes::JOINT, decoded_size::joint())?;
    let mut joints = Vec::with_capacity(nj);
    for _ in 0..nj {
        joints.push(read_joint(&mut r)?);
    }

    // pending 커맨드 (enqueue 순서 보존 — 설계 §7)
    let ncmd = r.count(min_bytes::COMMAND, decoded_size::command())?;
    let mut commands = Vec::with_capacity(ncmd);
    for _ in 0..ncmd {
        let cmd = match r.u8()? {
            0 => {
                let h = r.handle()?;
                PendingCommand::Add(h, Box::new(read_body(&mut r)?))
            }
            1 => PendingCommand::Public(Command::RemoveBody(r.handle()?)),
            2 => {
                let h = r.handle()?;
                PendingCommand::Public(Command::SetLinearVelocity(h, r.state_vec3()?))
            }
            3 => {
                let h = r.handle()?;
                PendingCommand::Public(Command::SetAngularVelocity(h, r.state_vec3()?))
            }
            4 => {
                let h = r.handle()?;
                PendingCommand::Public(Command::SetPose(
                    h,
                    Pose {
                        pos: r.state_vec3()?,
                        rot: r.quat()?,
                    },
                ))
            }
            5 => {
                let h = r.handle()?;
                let (f, rest) = (r.f32()?, r.f32()?);
                if f < 0.0 || !(0.0..=1.0).contains(&rest) {
                    return Err(SnapshotError::InconsistentState);
                }
                PendingCommand::Public(Command::SetMaterial(h, f, rest))
            }
            6 => PendingCommand::Public(Command::AddJoint(Box::new(read_joint(&mut r)?))),
            _ => return Err(SnapshotError::BadShape),
        };
        commands.push(cmd);
    }
    let report = CommandReport {
        applied: r.u32()? as usize,
        rejected: r.u32()? as usize,
    };

    let np = r.count(min_bytes::CACHE_PAIR, decoded_size::cache_pair())?;
    let mut cache = std::collections::HashMap::with_capacity(np);
    for _ in 0..np {
        let key = (r.u32()?, r.u32()?);
        let ne = r.count(min_bytes::CACHE_ENTRY, decoded_size::cache_entry())?;
        let mut entry = Vec::with_capacity(ne);
        for _ in 0..ne {
            let fid = maru_collide::FeatureId(r.u32()?);
            let imp = crate::solver::CachedImpulse {
                normal: r.state_f32()?,
                tangent: [r.state_f32()?, r.state_f32()?],
                tangent_world: r.state_vec3()?,
                position: r.state_vec3()?,
                contact_normal: r.vec3()?, // 단위 벡터 — 입력 정책으로 충분
            };
            // 캐시된 법선은 단위 벡터이거나(정상) 0(초기값)이어야 한다 — 임의 벡터는
            // 매칭 게이트를 무력화한다
            let nl2 = imp.contact_normal.length_sq();
            if !(nl2 == 0.0 || (0.9..=1.1).contains(&nl2)) {
                return Err(SnapshotError::InconsistentState);
            }
            // 법선 임펄스는 **비음수**가 계약이다(단방향 접촉). 음수는 warm start가
            // 바디를 서로 끌어당기게 만든다.
            if imp.normal < 0.0 {
                return Err(SnapshotError::InconsistentState);
            }
            entry.push((fid, imp));
        }
        cache.insert(key, entry);
    }

    if r.at != bytes.len() {
        return Err(SnapshotError::TrailingBytes);
    }

    validate_structure(
        &bodies,
        &alive,
        &reserved,
        &generations,
        &asleep,
        &island,
        &joints,
        &commands,
        &cache,
    )?;

    // 여기까지 통과해야 월드를 교체한다 (원자적 복원)
    world.restore_parts(
        bodies,
        alive,
        reserved,
        generations,
        sleep_timer,
        asleep,
        island,
        joints,
        cache,
        commands,
        report,
    );
    Ok(())
}

/// 슬롯·조인트·커맨드·캐시 사이의 **관계** 불변식 (리뷰 BLOCKER: 값 검증만으로는
/// 죽은 슬롯을 가리키는 조인트, 죽은 아일랜드 대표 같은 상태가 그대로 들어온다).
#[allow(clippy::too_many_arguments)]
fn validate_structure(
    bodies: &[Body],
    alive: &[bool],
    reserved: &[bool],
    generations: &[u32],
    asleep: &[bool],
    island: &[u32],
    joints: &[Joint],
    commands: &[PendingCommand],
    cache: &std::collections::HashMap<
        (u32, u32),
        Vec<(maru_collide::FeatureId, crate::solver::CachedImpulse)>,
    >,
) -> Result<(), SnapshotError> {
    const STATIC_BIT: u32 = 1 << 31;
    let nb = bodies.len();
    let live_dynamic = |i: usize| i < nb && alive[i] && bodies[i].is_dynamic();

    for i in 0..nb {
        // 슬롯은 살아있거나, 예약이거나, 비어있다 — 둘을 겸할 수 없다
        if alive[i] && reserved[i] {
            return Err(SnapshotError::InconsistentState);
        }
        // 잠듦은 살아있는 dynamic만의 상태다
        if asleep[i] && !live_dynamic(i) {
            return Err(SnapshotError::InconsistentState);
        }
        // 아일랜드 대표는 살아있는 dynamic이거나 없음(u32::MAX)
        if island[i] != u32::MAX && !live_dynamic(island[i] as usize) {
            return Err(SnapshotError::BadIndex);
        }
        // 비동적·비생존 슬롯은 아일랜드에 속하지 않는다(설계 §4.3: static은 anchor)
        if island[i] != u32::MAX && !live_dynamic(i) {
            return Err(SnapshotError::InconsistentState);
        }
    }

    // 핸들 해석 — 인덱스·생존·세대가 모두 맞아야 한다
    let resolve = |h: &BodyHandle| -> bool {
        let i = h.index as usize;
        i < nb && alive[i] && generations[i] == h.generation
    };

    for j in joints {
        if !resolve(&j.body_a) || !resolve(&j.body_b) {
            return Err(SnapshotError::BadIndex);
        }
        if j.body_a.index == j.body_b.index {
            return Err(SnapshotError::InconsistentState);
        }
        if !j.anchor_a.is_finite() || !j.anchor_b.is_finite() {
            return Err(SnapshotError::NonFinite);
        }
    }

    // 커맨드 대상 검증은 **공개 `enqueue`와 동일한 술어**를 쓴다(리뷰 PARTIAL MAJOR).
    // 세대만 보면, 공개 API로는 만들 수 없는 pending 상태(죽은 비예약 슬롯을 겨냥한
    // SetVelocity·AddJoint)를 조작 스냅샷이 주입할 수 있다 — 정본이 아닌 상태를
    // 정본 큐에 들이는 셈이다.
    // enqueue와 같은 술어: 대상은 살아있거나 예약된 슬롯이고 세대가 일치해야 한다.
    let enqueueable = |h: &BodyHandle| -> bool {
        let i = h.index as usize;
        i < nb && (alive[i] || reserved[i]) && generations[i] == h.generation
    };
    let mut add_targets: Vec<u32> = Vec::new();
    for c in commands {
        let c = match c {
            PendingCommand::Add(h, _) => {
                let i = h.index as usize;
                // AddBody는 **예약된** 슬롯만 채울 수 있다
                if i >= nb || !reserved[i] {
                    return Err(SnapshotError::InconsistentState);
                }
                add_targets.push(h.index);
                continue;
            }
            PendingCommand::Public(c) => c,
        };
        let h = match c {
            Command::RemoveBody(h)
            | Command::SetLinearVelocity(h, _)
            | Command::SetAngularVelocity(h, _)
            | Command::SetPose(h, _)
            | Command::SetMaterial(h, _, _) => h,
            Command::AddJoint(j) => {
                for h in [&j.body_a, &j.body_b] {
                    if !enqueueable(h) {
                        return Err(SnapshotError::BadIndex);
                    }
                }
                if j.body_a.index == j.body_b.index {
                    return Err(SnapshotError::InconsistentState);
                }
                continue;
            }
        };
        if !enqueueable(h) {
            return Err(SnapshotError::BadIndex);
        }
    }
    // 예약 슬롯과 pending AddBody는 **일대일**이다. 짝 없는 예약은 영원히 회수되지 않는
    // 슬롯 누수이고, 중복 AddBody는 둘째가 조용히 거부된다.
    add_targets.sort_unstable();
    let dup = add_targets.windows(2).any(|w| w[0] == w[1]);
    let reserved_count = reserved.iter().filter(|x| **x).count();
    if dup || reserved_count != add_targets.len() {
        return Err(SnapshotError::InconsistentState);
    }

    // PCM 캐시 키는 살아있는 바디 페어여야 하고, static 비트가 실제 타입과 맞아야 한다
    for (a, b_raw) in cache.keys() {
        let b = (b_raw & !STATIC_BIT) as usize;
        let a = *a as usize;
        if a >= nb || b >= nb || !alive[a] || !alive[b] {
            return Err(SnapshotError::BadIndex);
        }
        let b_is_static = b_raw & STATIC_BIT != 0;
        if b_is_static != (bodies[b].body_type == BodyType::Static) {
            return Err(SnapshotError::InconsistentState);
        }
    }
    Ok(())
}

/// FNV-1a 64 — 캐노니컬 상태 해시 (L0 CI 비교용).
pub fn state_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
