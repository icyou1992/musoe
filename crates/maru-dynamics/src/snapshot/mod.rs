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

mod codec;
mod io;
mod validate;

use crate::command::CommandReport;
use crate::command::{Command, PendingCommand};
use crate::World;
use codec::*;
use io::*;
pub use io::{Reader, SnapshotError, Writer};
pub use io::{SNAPSHOT_VERSION, VERSION};
use maru_collide::Pose;
use validate::*;

/// 스냅샷 바이트 수 (할당 전 preflight).
pub fn serialized_size(world: &World) -> usize {
    let (_, bodies, _, _, _, _, _, _, _, joints, cache, commands, _) = world.canonical_parts();
    let mut n = 4 + 4 + (12 + 4 + 4 + 16 + 12 + 4 + 4 + 4) + 4; // magic + version + params + count
    n += 4 + world.touching_keys().len() * 8; // 접촉 기준선
    for (i, b) in bodies.iter().enumerate() {
        let _ = i;
        n += 1 + 1 + 1 + 4 + body_size(b) + 4 + 1 + 4;
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
    // `reserve_exact`은 할당 실패를 abort/trap으로 표현한다 — 오류 ABI를 지키려면
    // `try_reserve_exact`이어야 한다(리뷰 MAJOR).
    w.0.try_reserve_exact(size)
        .map_err(|_| SnapshotError::CountTooLarge)?;
    write_world_bytes(world, w);
    debug_assert_eq!(w.0.len(), size, "preflight 크기 계산이 writer와 어긋남");
    Ok(())
}

fn write_world_bytes(world: &World, w: &mut Writer) {
    w.u32(MAGIC);
    w.u32(VERSION);
    let (
        params,
        bodies,
        alive,
        oob,
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
    write_params(w, params);
    w.u32(bodies.len() as u32);
    for (i, b) in bodies.iter().enumerate() {
        w.u8(alive[i] as u8);
        w.u8(oob[i] as u8);
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

    // 접촉 이벤트 기준선 — **캐노니컬 상태**다(복원된 월드가 같은 이벤트를 내야 한다).
    // PCM 캐시와 따로 두는 이유는 수명이 다르기 때문이다(`World::touching` 주석 참조).
    let touching = world.touching_keys();
    w.u32(touching.len() as u32);
    for k in touching {
        w.u32(k.0);
        w.u32(k.1);
    }
}

/// 신뢰하지 않는 바이트에서 월드 복원. **검증이 끝나기 전에는 월드를 건드리지 않는다.**
pub fn read_world(world: &mut World, bytes: &[u8]) -> Result<(), SnapshotError> {
    if bytes.len() > MAX_SNAPSHOT_BYTES {
        return Err(SnapshotError::CountTooLarge);
    }
    let mut r = Reader::new(bytes);
    if r.u32()? != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let v = r.u32()?;
    if v != VERSION {
        return Err(SnapshotError::BadVersion(v));
    }
    let params = read_params(&mut r)?;
    let nb = r.count(min_bytes::SLOT, decoded_size::slot())?;
    // **복원 컨테이너도 할당 실패를 오류로 표현한다**(리뷰 MAJOR): `with_capacity`는
    // 예산 안이라도 wasm 가용 메모리가 모자라면 allocator abort/trap이 된다.
    let mut bodies = try_vec(nb)?;
    let mut sleep_timer = try_vec(nb)?;
    let mut asleep = try_vec(nb)?;
    let mut island = try_vec(nb)?;
    let mut alive = try_vec(nb)?;
    let mut oob = try_vec(nb)?;
    let mut reserved = try_vec(nb)?;
    let mut generations = try_vec(nb)?;
    for _ in 0..nb {
        alive.push(r.u8()? != 0);
        oob.push(r.u8()? != 0);
        reserved.push(r.u8()? != 0);
        generations.push(r.u32()?);
        bodies.push(read_body(&mut r)?);
        // sleep timer는 **파생 상태**다(dt 누적). time_to_sleep가 큰 월드에서 입력 상한을
        // 넘길 수 있는데 writer는 그대로 쓰므로, 리더도 상태 정책으로 읽어야 대칭이 된다.
        let t = r.state_f32()?;
        if t < 0.0 {
            return Err(SnapshotError::NonFinite);
        }
        sleep_timer.push(t);
        asleep.push(r.u8()? != 0);
        island.push(r.u32()?);
    }

    let nj = r.count(min_bytes::JOINT, decoded_size::joint())?;
    let mut joints = try_vec(nj)?;
    for _ in 0..nj {
        joints.push(read_joint(&mut r)?);
    }

    // pending 커맨드 (enqueue 순서 보존 — 설계 §7)
    let ncmd = r.count(min_bytes::COMMAND, decoded_size::command())?;
    let mut commands = try_vec(ncmd)?;
    for _ in 0..ncmd {
        let cmd = match r.u8()? {
            0 => {
                let h = r.handle()?;
                let b = read_body(&mut r)?;
                // 아직 월드에 들어가지 않은 바디 = 호스트 입력. `enqueue_add_body`와 같은
                // 검증을 통과해야 한다(read_body는 파생 상태 범위를 쓴다).
                if b.validate().is_err() {
                    return Err(SnapshotError::InconsistentState);
                }
                PendingCommand::Add(h, Box::new(b))
            }
            1 => PendingCommand::Public(Command::RemoveBody(r.handle()?)),
            // pending 페이로드는 **호스트 입력**이다 — 공개 `enqueue`와 **같은 정책**으로
            // 읽어야 한다(리뷰 BLOCKER: state 범위로 읽으면 조작 스냅샷이 거대 속도 커맨드를
            // 큐에 심고, flush 뒤 브로드페이즈 DoS로 이어진다).
            2 => {
                let h = r.handle()?;
                let v = r.state_vec3()?;
                if !maru_collide::vec_speed_in_range(v) {
                    return Err(SnapshotError::NonFinite);
                }
                PendingCommand::Public(Command::SetLinearVelocity(h, v))
            }
            3 => {
                let h = r.handle()?;
                let v = r.state_vec3()?;
                if !maru_collide::vec_angular_in_range(v) {
                    return Err(SnapshotError::NonFinite);
                }
                PendingCommand::Public(Command::SetAngularVelocity(h, v))
            }
            4 => {
                let h = r.handle()?;
                let pos = r.state_vec3()?;
                if !maru_collide::vec_in_range(pos) {
                    return Err(SnapshotError::NonFinite);
                }
                PendingCommand::Public(Command::SetPose(
                    h,
                    Pose {
                        pos,
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
    let mut cache: crate::world::IntMap<
        (u32, u32),
        Vec<(maru_collide::FeatureId, crate::solver::CachedImpulse)>,
    > = crate::world::IntMap::default();
    cache
        .try_reserve(np)
        .map_err(|_| SnapshotError::CountTooLarge)?;
    for _ in 0..np {
        let key = (r.u32()?, r.u32()?);
        let ne = r.count(min_bytes::CACHE_ENTRY, decoded_size::cache_entry())?;
        // **라이브에서 도달 가능한 크기만 복원한다** (교차검수 rev_ca87695d MAJOR).
        // 한 페어의 캐시는 페어당 매니폴드 상한 × 접점 상한을 넘을 수 없다
        // (하이트필드 64 매니폴드 × 4점 = 256, 복합 8×8 자식쌍도 그 안이다).
        // 상한을 안 걸면 64 MiB 예산 안에서 수십만 항목짜리 페어 캐시가 복원되고,
        // 그 벡터는 라이브가 만들 수 없는 크기로 메모리에 남는다 — 그 페어가 다음 스텝에
        // 처리되면 핫패스의 용량 축소가 거대한 재할당을 강제하고, 안 처리되면 계속 잔류한다.
        // 물리 매칭 쪽 fail-closed(`match_cache_entry`)만으로는 이 계약이 안 닫힌다.
        if ne > crate::solver::CACHE_USED_WORDS * 64 {
            return Err(SnapshotError::CountTooLarge);
        }
        let mut entry = try_vec(ne)?;
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

    // 접촉 이벤트 기준선. 개수는 페어와 같은 규칙으로 남은 바이트에서 유도한다.
    let nt = r.count(8, 8)?;
    let mut touching: crate::world::IntSet<(u32, u32)> = crate::world::IntSet::default();
    touching
        .try_reserve(nt)
        .map_err(|_| SnapshotError::CountTooLarge)?;
    for _ in 0..nt {
        touching.insert((r.u32()?, r.u32()?));
    }

    if !r.finished() {
        return Err(SnapshotError::TrailingBytes);
    }

    validate_structure(
        &bodies,
        &alive,
        &oob,
        &reserved,
        &generations,
        &asleep,
        &island,
        &joints,
        &commands,
        &cache,
        &touching,
    )?;

    // 여기까지 통과해야 월드를 교체한다 (원자적 복원)
    world.restore_parts(
        params,
        bodies,
        alive,
        oob,
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
    world.set_touching(touching);
    Ok(())
}

/// 슬롯·조인트·커맨드·캐시 사이의 **관계** 불변식 (리뷰 BLOCKER: 값 검증만으로는
/// 죽은 슬롯을 가리키는 조인트, 죽은 아일랜드 대표 같은 상태가 그대로 들어온다).
/// FNV-1a 64 — 캐노니컬 상태 해시 (L0 CI 비교용).
pub fn state_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
