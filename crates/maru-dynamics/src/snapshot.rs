//! 캐노니컬 스냅샷 + L0 상태 해시 (설계 §3·§8, M2f).
//!
//! 포함: 바디(자세·속도·질량·셰이프·재질) · 슬립 상태 · PCM 임펄스 캐시(정렬 직렬화) ·
//! 조인트(누적 임펄스 포함). **브로드페이즈 트리는 제외** — 순수 파생 가속 구조(설계 §4.1
//! "페어 집합 트리 비의존" 계약). 스냅샷은 스텝 경계에서만 호출한다는 계약(설계 §3).
//! 커맨드/이벤트 버퍼는 M3에서 도입되며 그때 스냅샷에 편입된다(설계 §3 — 현재는 부재).
//!
//! 계약: `restore(snapshot())` 후 재생은 원 실행과 **같은 타깃에서 비트 동일**(L0).
//!
//! **역직렬화는 신뢰하지 않는 입력을 다룬다** (리뷰 BLOCKER): M3에서 JS가 넘기는 바이트는
//! 짧거나 조작될 수 있다. 매직·버전·개수 상한·유한성·인덱스 범위·잔여 바이트를 모두 검증하고
//! `Result`로 실패하며, 검증 전에는 월드를 건드리지 않는다(원자적 복원).

use crate::body::{Body, BodyType};
use crate::joint::{Joint, JointKind};
use crate::world::World;
use maru_collide::{Pose, Shape};
use maru_math::{Quat, Vec3};

/// 포맷 매직·버전 — 다른 빌드/포맷의 바이트를 조용히 먹지 않기 위해.
const MAGIC: u32 = 0x4d41_5255; // "MARU"
const VERSION: u32 = 1;
/// 신뢰하지 않는 count의 상한 (OOM 방지).
const MAX_BODIES: u32 = 1 << 22;
const MAX_JOINTS: u32 = 1 << 22;
const MAX_CACHE_ENTRIES: u32 = 1 << 24;
const MAX_HULL_POINTS: u32 = 1 << 20;

#[derive(Debug, PartialEq, Eq)]
pub enum SnapshotError {
    BadMagic,
    BadVersion(u32),
    Truncated,
    TrailingBytes,
    CountTooLarge,
    NonFinite,
    BadIndex,
    BadShape,
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
}

pub struct Reader<'a> {
    b: &'a [u8],
    at: usize,
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
    fn f32(&mut self) -> Result<f32, SnapshotError> {
        let v = f32::from_le_bytes(self.take(4)?.try_into().unwrap());
        if v.is_finite() {
            Ok(v)
        } else {
            Err(SnapshotError::NonFinite)
        }
    }
    /// 유한성을 요구하지 않는 f32 (현재 미사용 — 향후 의도적 non-finite 필드용).
    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take(1)?[0])
    }
    fn vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    fn quat(&mut self) -> Result<Quat, SnapshotError> {
        let q = Quat {
            x: self.f32()?,
            y: self.f32()?,
            z: self.f32()?,
            w: self.f32()?,
        };
        // 단위 쿼터니언이 아니면 자세가 조용히 붕괴한다
        let l2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if !(0.9..=1.1).contains(&l2) {
            return Err(SnapshotError::NonFinite);
        }
        Ok(q.normalize())
    }
    fn count(&mut self, max: u32) -> Result<usize, SnapshotError> {
        let n = self.u32()?;
        if n > max {
            return Err(SnapshotError::CountTooLarge);
        }
        Ok(n as usize)
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
            let n = r.count(MAX_HULL_POINTS)?;
            let mut points = Vec::with_capacity(n.min(4096));
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

pub fn write_world(world: &World, w: &mut Writer) {
    w.u32(MAGIC);
    w.u32(VERSION);
    let (bodies, alive, generations, sleep_timer, asleep, island, joints, cache) =
        world.canonical_parts();
    w.u32(bodies.len() as u32);
    for (i, b) in bodies.iter().enumerate() {
        w.u8(alive[i] as u8);
        w.u32(generations[i]);
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
        write_shape(w, &b.shape);
        w.f32(sleep_timer[i]);
        w.u8(asleep[i] as u8);
        w.u32(island[i]); // 아일랜드 대표 — wake 전파가 restore 후에도 동일해야 L0 유지
    }
    w.u32(joints.len() as u32);
    for j in joints {
        w.u32(j.body_a as u32);
        w.u32(j.body_b as u32);
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
        }
    }
}

/// 신뢰하지 않는 바이트에서 월드 복원. **검증이 끝나기 전에는 월드를 건드리지 않는다.**
pub fn read_world(world: &mut World, bytes: &[u8]) -> Result<(), SnapshotError> {
    let mut r = Reader { b: bytes, at: 0 };
    if r.u32()? != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let v = r.u32()?;
    if v != VERSION {
        return Err(SnapshotError::BadVersion(v));
    }
    let nb = r.count(MAX_BODIES)?;
    let mut bodies = Vec::with_capacity(nb.min(4096));
    let mut sleep_timer = Vec::with_capacity(nb.min(4096));
    let mut asleep = Vec::with_capacity(nb.min(4096));
    let mut island = Vec::with_capacity(nb.min(4096));
    let mut alive = Vec::with_capacity(nb.min(4096));
    let mut generations = Vec::with_capacity(nb.min(4096));
    for _ in 0..nb {
        alive.push(r.u8()? != 0);
        generations.push(r.u32()?);
        let body_type = match r.u8()? {
            0 => BodyType::Static,
            1 => BodyType::Dynamic,
            2 => BodyType::Kinematic,
            _ => return Err(SnapshotError::BadShape),
        };
        let pose = Pose {
            pos: r.vec3()?,
            rot: r.quat()?,
        };
        let linvel = r.vec3()?;
        let angvel = r.vec3()?;
        let inv_mass = r.f32()?;
        let inv_inertia_local = r.vec3()?;
        let friction = r.f32()?;
        let restitution = r.f32()?;
        let shape = read_shape(&mut r)?;
        if inv_mass < 0.0
            || friction < 0.0
            || !(0.0..=1.0).contains(&restitution)
            || inv_inertia_local.x < 0.0
            || inv_inertia_local.y < 0.0
            || inv_inertia_local.z < 0.0
        {
            return Err(SnapshotError::NonFinite);
        }
        bodies.push(Body {
            body_type,
            pose,
            linvel,
            angvel,
            shape,
            inv_mass,
            inv_inertia_local,
            friction,
            restitution,
        });
        let t = r.f32()?;
        if t < 0.0 {
            return Err(SnapshotError::NonFinite);
        }
        sleep_timer.push(t);
        asleep.push(r.u8()? != 0);
        island.push(r.u32()?);
    }
    // 아일랜드 대표는 바디 인덱스이거나 u32::MAX(비동적)이어야 한다
    if island.iter().any(|&i| i != u32::MAX && i as usize >= nb) {
        return Err(SnapshotError::BadIndex);
    }

    let nj = r.count(MAX_JOINTS)?;
    let mut joints = Vec::with_capacity(nj.min(4096));
    for _ in 0..nj {
        let body_a = r.u32()? as usize;
        let body_b = r.u32()? as usize;
        if body_a >= nb || body_b >= nb || body_a == body_b {
            return Err(SnapshotError::BadIndex);
        }
        let anchor_a = r.vec3()?;
        let anchor_b = r.vec3()?;
        let kind = match r.u8()? {
            0 => JointKind::Point,
            1 => JointKind::Hinge {
                axis_a: r.vec3()?,
                axis_b: r.vec3()?,
            },
            _ => return Err(SnapshotError::BadShape),
        };
        let collide_connected = r.u8()? != 0;
        let linear_impulse = r.vec3()?;
        let angular_impulse = [r.f32()?, r.f32()?];
        joints.push(Joint {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            kind,
            collide_connected,
            linear_impulse,
            angular_impulse,
        });
    }

    let nc = r.count(MAX_CACHE_ENTRIES)?;
    let mut cache = std::collections::HashMap::with_capacity(nc.min(4096));
    for _ in 0..nc {
        let key = (r.u32()?, r.u32()?);
        // 키의 바디 인덱스도 범위 안이어야 한다 (상위 비트는 STATIC_BIT 플래그)
        const STATIC_BIT: u32 = 1 << 31;
        if key.0 as usize >= nb || (key.1 & !STATIC_BIT) as usize >= nb {
            return Err(SnapshotError::BadIndex);
        }
        let ne = r.count(MAX_CACHE_ENTRIES)?;
        let mut entry = Vec::with_capacity(ne.min(64));
        for _ in 0..ne {
            let fid = maru_collide::FeatureId(r.u32()?);
            let imp = crate::solver::CachedImpulse {
                normal: r.f32()?,
                tangent: [r.f32()?, r.f32()?],
                tangent_world: r.vec3()?,
            };
            entry.push((fid, imp));
        }
        cache.insert(key, entry);
    }

    if r.at != bytes.len() {
        return Err(SnapshotError::TrailingBytes);
    }
    // 여기까지 통과해야 월드를 교체한다 (원자적 복원)
    world.restore_parts(
        bodies,
        alive,
        generations,
        sleep_timer,
        asleep,
        island,
        joints,
        cache,
    );
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
