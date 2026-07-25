//! 캐노니컬 스냅샷 + L0 상태 해시 (설계 §3·§8, M2f).
//!
//! 포함: 바디(자세·속도·질량·셰이프·재질) · 슬립 상태 · PCM 임펄스 캐시(정렬 직렬화) ·
//! 조인트(누적 임펄스 포함). **브로드페이즈 트리는 제외** — 순수 파생 가속 구조(설계 §4.1
//! "페어 집합 트리 비의존" 계약). 스냅샷은 스텝 경계에서만 호출한다는 계약(설계 §3).
//! 커맨드/이벤트 버퍼는 M3에서 도입되며 그때 스냅샷에 편입된다(설계 §3 — 현재는 부재).
//!
//! 계약: `restore(snapshot())` 후 재생은 원 실행과 **같은 타깃에서 비트 동일**(L0).

use crate::body::{Body, BodyType};
use crate::joint::{Joint, JointKind};
use crate::world::World;
use maru_collide::{Pose, Shape};
use maru_math::{Quat, Vec3};

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

impl<'a> Reader<'a> {
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.at..self.at + 4].try_into().unwrap());
        self.at += 4;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.at..self.at + 4].try_into().unwrap());
        self.at += 4;
        v
    }
    fn u8(&mut self) -> u8 {
        let v = self.b[self.at];
        self.at += 1;
        v
    }
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.f32(), self.f32(), self.f32())
    }
    fn quat(&mut self) -> Quat {
        Quat {
            x: self.f32(),
            y: self.f32(),
            z: self.f32(),
            w: self.f32(),
        }
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

fn read_shape(r: &mut Reader) -> Shape {
    match r.u8() {
        0 => Shape::Sphere { radius: r.f32() },
        1 => Shape::Capsule {
            half_height: r.f32(),
            radius: r.f32(),
        },
        2 => Shape::Box {
            half: r.vec3(),
            radius: r.f32(),
        },
        _ => {
            let n = r.u32() as usize;
            let points = (0..n).map(|_| r.vec3()).collect();
            Shape::Hull {
                points,
                radius: r.f32(),
            }
        }
    }
}

pub fn write_world(world: &World, w: &mut Writer) {
    let (bodies, sleep_timer, asleep, island, joints, cache) = world.canonical_parts();
    w.u32(bodies.len() as u32);
    for (i, b) in bodies.iter().enumerate() {
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
        }
    }
}

pub fn read_world(world: &mut World, bytes: &[u8]) {
    let mut r = Reader { b: bytes, at: 0 };
    let nb = r.u32() as usize;
    let mut bodies = Vec::with_capacity(nb);
    let mut sleep_timer = Vec::with_capacity(nb);
    let mut asleep = Vec::with_capacity(nb);
    let mut island = Vec::with_capacity(nb);
    for _ in 0..nb {
        let body_type = match r.u8() {
            0 => BodyType::Static,
            1 => BodyType::Dynamic,
            _ => BodyType::Kinematic,
        };
        let pose = Pose {
            pos: r.vec3(),
            rot: r.quat(),
        };
        let linvel = r.vec3();
        let angvel = r.vec3();
        let inv_mass = r.f32();
        let inv_inertia_local = r.vec3();
        let friction = r.f32();
        let restitution = r.f32();
        let shape = read_shape(&mut r);
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
        sleep_timer.push(r.f32());
        asleep.push(r.u8() != 0);
        island.push(r.u32());
    }
    let nj = r.u32() as usize;
    let mut joints = Vec::with_capacity(nj);
    for _ in 0..nj {
        let body_a = r.u32() as usize;
        let body_b = r.u32() as usize;
        let anchor_a = r.vec3();
        let anchor_b = r.vec3();
        let kind = match r.u8() {
            0 => JointKind::Point,
            _ => JointKind::Hinge {
                axis_a: r.vec3(),
                axis_b: r.vec3(),
            },
        };
        let collide_connected = r.u8() != 0;
        let linear_impulse = r.vec3();
        let angular_impulse = [r.f32(), r.f32()];
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
    let nc = r.u32() as usize;
    let mut cache = std::collections::HashMap::new();
    for _ in 0..nc {
        let key = (r.u32(), r.u32());
        let ne = r.u32() as usize;
        let entry = (0..ne)
            .map(|_| {
                let fid = maru_collide::FeatureId(r.u32());
                let imp = crate::solver::CachedImpulse {
                    normal: r.f32(),
                    tangent: [r.f32(), r.f32()],
                };
                (fid, imp)
            })
            .collect();
        cache.insert(key, entry);
    }
    world.restore_parts(bodies, sleep_timer, asleep, island, joints, cache);
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
