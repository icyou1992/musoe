//! 자료형별 **직렬화 포맷** — 셰이프·파라미터·바디·조인트·커맨드.
//!
//! `write_*`와 `read_*`를 **짝지어 붙여 둔다**: 포맷이 바뀔 때 한쪽만 고치면
//! 라운드트립이 깨지므로, 둘이 눈에 같이 들어와야 한다. 크기 헬퍼(`*_size`)도
//! 같은 포맷을 세는 세 번째 표현이라 여기 둔다.

use super::io::*;
use crate::body::BodyType;
use crate::command::{Command, PendingCommand};
use musoe_collide::Pose;

use crate::JointKind;

use crate::{Body, Joint};
use musoe_collide::Shape;

pub(crate) fn write_shape(w: &mut Writer, s: &Shape) {
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
        // v19: 원기둥. 태그는 **뒤에 붙인다** — 기존 태그를 재배치하면 옛 저장본이 다른
        // 셰이프로 읽힌다(조용한 손상).
        Shape::Cylinder {
            half_height,
            radius,
        } => {
            w.u8(8);
            w.f32(*half_height);
            w.f32(*radius);
        }
        // 타일은 높이의 파생물이라 **직렬화하지 않는다** — 복원에서 생성자가 다시 만든다.
        // 파생을 저장하면 "바이트는 같은데 파생이 다른" 상태가 생길 수 있다.
        Shape::Heightfield(g) => {
            w.u8(5);
            w.u32(g.nx());
            w.u32(g.nz());
            w.f32(g.cell());
            // 개수는 nx·nz로 유도되지만 **명시로 쓴다** — 리더가 곱셈 오버플로를 만나기
            // 전에 남은 바이트로 예산을 걸 수 있어야 한다(count 계약).
            w.u32(g.heights().len() as u32);
            for h in g.heights() {
                w.f32(*h);
            }
        }
        // BVH·용접 마스크는 정점·인덱스의 **파생물이라 직렬화하지 않는다** — 복원에서
        // 생성자가 다시 만든다(하이트필드 타일과 같은 정책). 파생을 저장하면 "바이트는 같은데
        // 파생이 다른" 상태가 생기고, 그걸 막으려면 복원에서 비트 검증을 해야 한다.
        Shape::TriMesh(m) => {
            w.u8(6);
            // 개수는 **명시로 쓴다** — 리더가 남은 바이트로 예산을 걸 수 있어야 한다.
            w.u32(m.verts().len() as u32);
            for v in m.verts() {
                w.vec3(*v);
            }
            w.u32(m.tris().len() as u32);
            for t in m.tris() {
                w.u32(t[0]);
                w.u32(t[1]);
                w.u32(t[2]);
            }
        }
        Shape::Triangle { .. } => {
            // 내부 전용 셰이프는 바디가 될 수 없다(validate가 거부) — 여기 오면 계약 위반.
            unreachable!("triangle is an internal shape and is never serialized")
        }
        Shape::Hull { data: hull, radius } => {
            // **점만 저장한다** — 면 구조는 파생물이라 복원에서 다시 만든다
            // (`TriMeshData`의 BVH·`HeightfieldGrid`의 타일과 같은 정책).
            let points = hull.points();
            w.u8(3);
            w.u32(points.len() as u32);
            for p in points {
                w.vec3(*p);
            }
            w.f32(*radius);
        }
        // v11: 복합 — 자식 수 u8 + 자식별 (pose + 재귀 셰이프). 중첩은 validate가 거부.
        Shape::Compound { children } => {
            w.u8(4);
            w.u8(children.len() as u8);
            for (cp, child) in children {
                w.vec3(cp.pos);
                w.quat(cp.rot);
                write_shape(w, child);
            }
        }
    }
}

pub(crate) fn read_shape(r: &mut Reader, allow_compound: bool) -> Result<Shape, SnapshotError> {
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
        8 => Shape::Cylinder {
            half_height: r.f32()?,
            radius: r.f32()?,
        },
        3 => {
            // 점 개수도 남은 바이트로 묶인다 → 전체 파일 크기가 곧 총 점 예산이다
            let n = r.count(min_bytes::HULL_POINT, decoded_size::hull_point())?;
            let mut points = try_vec(n)?;
            for _ in 0..n {
                points.push(r.vec3()?);
            }
            // **면 구조를 만들기 전에 힙과 작업량을 둘 다 과금한다.**
            //
            // 힙: `HullData::new`가 최대 `2n−4`개 면을 infallible하게 잡는데, 그게 장부 밖이면
            // 장부상 한도 안인 스냅샷이 실제로는 allocator abort로 끝난다
            // (검수 rev_8eacf4a5 BLOCKER). 면 하나 = `HullFace` + 정점 최대 n바이트.
            // **크기를 손으로 적지 않는다** — `Vec` 헤더는 wasm32에서 절반이라 상수가 갈린다.
            //
            // 작업량: 빌더가 `C(n,3)·n·2`라 32점 hull 하나가 약 31만 회 점 검사다. 메모리만
            // 묶으면 4만 개가 통과해 한 번의 복원이 120억 회를 돈다(검수 rev_ae6d97b1 BLOCKER).
            let radius = r.f32()?;
            let faces = (2 * n).saturating_sub(4);
            let face_bytes = core::mem::size_of::<musoe_collide::HullFace>() + n;
            r.charge_decoded(faces.saturating_mul(face_bytes))?;
            let triples = n
                .saturating_mul(n.saturating_sub(1))
                .saturating_mul(n.saturating_sub(2))
                / 6;
            r.charge_work(triples.saturating_mul(n).saturating_mul(2))?;
            Shape::Hull {
                // 면 구조를 여기서 다시 만든다 — 빌더가 결정적이라 같은 점이면 같은 면이다.
                data: musoe_collide::HullData::new(points),
                radius,
            }
        }
        5 => {
            let nx = r.u32()?;
            let nz = r.u32()?;
            let cell = r.f32()?;
            // 개수는 **입력 길이에서 유도된 예산**으로 묶는다 — nx·nz를 믿고 곱하면
            // 12바이트 파일이 40억 칸을 주장할 수 있다(스냅샷 계약의 상시 규칙).
            let n = r.count(min_bytes::HEIGHT, decoded_size::height())?;
            let mut heights = try_vec(n)?;
            for _ in 0..n {
                heights.push(r.f32()?);
            }
            // **타일은 생성자가 만든다** — 스냅샷 바이트에 파생물이 없으므로 위조 벡터가
            // 아예 없다(예전에는 `tiles_match`로 비트 검증해야 했다). 구조 불변식을 어기면
            // 생성자가 `None`을 내고, 그건 손상된 스냅샷이므로 형식 오류로 올린다.
            musoe_collide::Shape::heightfield(heights, nx, nz, cell)
                .ok_or(SnapshotError::BadShape)?
        }
        6 => {
            // 정점 → 삼각형 순. 두 개수 모두 **입력 길이에서 유도된 예산**으로 묶는다
            // (nv를 믿고 곱하면 작은 파일이 거대한 할당을 주장할 수 있다 — 스냅샷 상시 규칙).
            let nv = r.count(min_bytes::MESH_VERT, decoded_size::mesh_vert())?;
            let mut verts = try_vec(nv)?;
            for _ in 0..nv {
                verts.push(r.vec3()?);
            }
            let nt = r.count(min_bytes::MESH_TRI, decoded_size::mesh_tri())?;
            let mut tris: Vec<[u32; 3]> = try_vec(nt)?;
            for _ in 0..nt {
                tris.push([r.u32()?, r.u32()?, r.u32()?]);
            }
            // **인덱스 범위·퇴화·크기 정책은 생성자가 본다.** 손상된 스냅샷이 만들어 낸
            // 범위 밖 인덱스는 여기서 형식 오류가 되지, 나중에 인덱싱 패닉이 되지 않는다.
            musoe_collide::Shape::trimesh(verts, tris).ok_or(SnapshotError::BadShape)?
        }
        4 => {
            // **재귀 전에 깊이를 자른다**(교차검수 rev_e4075e85 BLOCKER): 자식을 읽고 나서
            // compound인지 검사하면, 수천 단계 중첩 체인(~30바이트/단계)이 BadShape에
            // 닿기 전에 호출 스택을 고갈시켜 trap이 된다 — 신뢰하지 않는 입력의 실패는
            // 오류지 런타임 종료가 아니어야 한다. allow_compound=false 재귀는 태그 4를
            // 즉시 거부하므로 깊이 1을 구조적으로 강제한다.
            if !allow_compound {
                return Err(SnapshotError::BadShape);
            }
            // 자식 수는 u8 + validate 상한(8)로 이중 유계. 자식 셰이프는 재귀 읽기 —
            // 중첩 compound 바이트는 아래 validate가 거부한다(읽기 자체는 깊이 1에서
            // 태그 4를 다시 만나도 count 예산이 파일 크기로 묶여 폭주하지 않는다).
            let n = r.u8()? as usize;
            if n == 0 || n > musoe_collide::MAX_COMPOUND_CHILDREN {
                return Err(SnapshotError::BadShape);
            }
            let mut children = Vec::new();
            children
                .try_reserve_exact(n)
                .map_err(|_| SnapshotError::CountTooLarge)?;
            for _ in 0..n {
                let pos = r.vec3()?;
                let rot = r.quat()?;
                let child = read_shape(r, false)?;
                children.push((Pose { pos, rot }, child));
            }
            Shape::Compound { children }
        }
        _ => return Err(SnapshotError::BadShape),
    };
    // 형상 불변식까지 통과해야 월드에 들인다
    shape.validate().map_err(|_| SnapshotError::BadShape)?;
    Ok(shape)
}

/// **WorldParams는 캐노니컬 상태다**(리뷰 BLOCKER): 미래 동작을 바꾸는 mutable state인데
/// 스냅샷·해시에서 빠져 있어, gravity=0인 월드의 스냅샷을 기본 파라미터 월드에 복원하면
/// 해시는 같은데 다음 스텝부터 분기했다.
pub(crate) fn write_params(w: &mut Writer, p: &crate::world::WorldParams) {
    w.vec3(p.gravity);
    w.u32(p.substeps);
    w.f32(p.speculative_margin);
    w.f32(p.solver.contact_hertz);
    w.f32(p.solver.contact_damping);
    w.f32(p.solver.max_bias_speed);
    w.f32(p.solver.restitution_threshold);
    w.f32(p.sleep_linear_threshold);
    w.f32(p.sleep_angular_threshold);
    w.f32(p.time_to_sleep);
    w.u32(p.max_pairs_per_body);
    w.f32(p.max_linear_velocity);
    w.f32(p.max_angular_velocity);
}

pub(crate) fn read_params(r: &mut Reader) -> Result<crate::world::WorldParams, SnapshotError> {
    let p = crate::world::WorldParams {
        gravity: r.vec3()?,
        substeps: r.u32()?,
        speculative_margin: r.f32()?,
        solver: crate::solver::SolverParams {
            contact_hertz: r.f32()?,
            contact_damping: r.f32()?,
            max_bias_speed: r.f32()?,
            restitution_threshold: r.f32()?,
        },
        sleep_linear_threshold: r.f32()?,
        sleep_angular_threshold: r.f32()?,
        time_to_sleep: r.f32()?,
        max_pairs_per_body: r.u32()?,
        max_linear_velocity: r.f32()?,
        max_angular_velocity: r.f32()?,
    };
    // 라이브 API와 **같은 검증**
    crate::world::check_params(&p).map_err(|_| SnapshotError::InconsistentState)?;
    Ok(p)
}

pub(crate) fn write_body(w: &mut Writer, b: &Body) {
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
    // 누적 힘·토크는 캐노니컬 상태다 — apply 뒤 step 전 스냅샷이 같은 미래를 만들려면 담겨야 한다.
    w.vec3(b.force);
    w.vec3(b.torque);
    w.u32(b.membership);
    w.u32(b.filter);
    w.u8(b.sensor as u8);
    w.u8(b.ccd as u8); // v11: CCD linear cast opt-in
    w.u8(b.weld_edges); // v12: 지형 내부 에지 용접 마스크
    w.u8(b.locked_rot); // v17: 월드축 회전 잠금 마스크
    w.f32(b.contact_force_threshold); // v18: 접촉력 이벤트 임계
    w.f32(b.linear_damping);
    w.f32(b.angular_damping);
    write_shape(w, &b.shape);
}

/// **리더는 라이브 API와 같은 validator를 호출한다**(리뷰 BLOCKER, 같은 비대칭의 4번째 재발).
/// 필드마다 범위 검사를 따로 적으면 새 차원을 도입할 때마다 한쪽만 고쳐 비대칭이 재생산된다
/// — 실제로 inv_mass·역관성·앵커에서 그렇게 됐다. 값은 유한/상태 범위로만 읽고,
/// **불변식 판정은 `Body::validate`/`Joint::validate` 한 곳**에서 한다.
pub(crate) fn read_body(r: &mut Reader) -> Result<Body, SnapshotError> {
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
    // 질량·관성은 차원이 달라 전용 범위를 쓴다 — 판정은 아래 validate()가 한다.
    let inv_mass = r.state_f32()?;
    let inv_inertia_local = r.state_vec3()?;
    let friction = r.f32()?;
    let restitution = r.f32()?;
    let shape_origin_offset = r.vec3()?;
    // 힘·토크는 여러 번 더해져 자라므로 파생 상태 정책으로 읽는다(라이브 누적과 같은 규칙).
    let force = r.state_vec3()?;
    let torque = r.state_vec3()?;
    let membership = r.u32()?;
    let filter = r.u32()?;
    let sensor = match r.u8()? {
        0 => false,
        1 => true,
        _ => return Err(SnapshotError::InconsistentState),
    };
    let ccd = match r.u8()? {
        0 => false,
        1 => true,
        _ => return Err(SnapshotError::InconsistentState),
    };
    // 범위(＜8)·타입 결합(static 3점 hull)은 아래 validate가 한 곳에서 판정한다.
    let weld_edges = r.u8()?;
    let locked_rot = r.u8()?;
    let contact_force_threshold = r.threshold_f32()?;
    let linear_damping = r.f32()?;
    let angular_damping = r.f32()?;
    let shape = read_shape(r, true)?;
    let b = Body {
        locked_rot,
        contact_force_threshold,
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
        force,
        torque,
        membership,
        filter,
        sensor,
        ccd,
        weld_edges,
        linear_damping,
        angular_damping,
    };
    b.validate().map_err(|_| SnapshotError::InconsistentState)?;
    Ok(b)
}

pub(crate) fn write_joint(w: &mut Writer, j: &Joint) {
    w.handle(j.body_a);
    w.handle(j.body_b);
    w.vec3(j.anchor_a);
    w.vec3(j.anchor_b);
    match &j.kind {
        JointKind::Point => w.u8(0),
        JointKind::Hinge {
            axis_a,
            axis_b,
            limit,
            motor,
        } => {
            w.u8(1);
            w.vec3(*axis_a);
            w.vec3(*axis_b);
            write_opt2(w, limit.as_ref().map(|l| [l[0], l[1]]));
            write_opt2(w, motor.as_ref().map(|m| [m.target_speed, m.max_effort]));
        }
        JointKind::Fixed { rel_rot } => {
            w.u8(2);
            w.quat(*rel_rot);
        }
        JointKind::Distance { min, max } => {
            w.u8(3);
            w.f32(*min);
            w.f32(*max);
        }
        JointKind::Prismatic {
            axis_a,
            rel_rot,
            limit,
            motor,
        } => {
            w.u8(4);
            w.vec3(*axis_a);
            w.quat(*rel_rot);
            write_opt2(w, limit.as_ref().map(|l| [l[0], l[1]]));
            write_opt2(w, motor.as_ref().map(|m| [m.target_speed, m.max_effort]));
        }
    }
    w.u8(j.collide_connected as u8);
    w.vec3(j.linear_impulse);
    for x in j.angular_impulse {
        w.f32(x);
    }
    w.f32(j.limit_impulse);
    w.f32(j.motor_impulse);
}

/// `Option<[f32;2]>` — 태그 1바이트 + 값 8바이트(없으면 0으로 채워 **길이를 고정**한다).
/// 가변 길이면 `joint_size` preflight가 kind 안의 분기까지 알아야 해서 어긋나기 쉽다.
pub(crate) fn write_opt2(w: &mut Writer, v: Option<[f32; 2]>) {
    match v {
        Some([a, b]) => {
            w.u8(1);
            w.f32(a);
            w.f32(b);
        }
        None => {
            w.u8(0);
            w.f32(0.0);
            w.f32(0.0);
        }
    }
}

pub(crate) fn read_opt2(r: &mut Reader) -> Result<Option<[f32; 2]>, SnapshotError> {
    let tag = r.u8()?;
    let a = r.f32()?;
    let b = r.f32()?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some([a, b])),
        _ => Err(SnapshotError::BadShape),
    }
}

pub(crate) fn read_joint(r: &mut Reader) -> Result<Joint, SnapshotError> {
    let body_a = r.handle()?;
    let body_b = r.handle()?;
    let anchor_a = r.state_vec3()?;
    let anchor_b = r.state_vec3()?;
    let kind = match r.u8()? {
        0 => JointKind::Point,
        1 => {
            let axis_a = r.vec3()?;
            let axis_b = r.vec3()?;
            let limit = read_opt2(r)?;
            let motor = read_opt2(r)?.map(|m| crate::joint::Motor {
                target_speed: m[0],
                max_effort: m[1],
            });
            JointKind::Hinge {
                axis_a,
                axis_b,
                limit,
                motor,
            }
        }
        2 => JointKind::Fixed { rel_rot: r.quat()? },
        3 => JointKind::Distance {
            min: r.f32()?,
            max: r.f32()?,
        },
        4 => {
            let axis_a = r.vec3()?;
            let rel_rot = r.quat()?;
            let limit = read_opt2(r)?;
            let motor = read_opt2(r)?.map(|m| crate::joint::Motor {
                target_speed: m[0],
                max_effort: m[1],
            });
            JointKind::Prismatic {
                axis_a,
                rel_rot,
                limit,
                motor,
            }
        }
        _ => return Err(SnapshotError::BadShape),
    };
    let collide_connected = r.u8()? != 0;
    let linear_impulse = r.state_vec3()?;
    let angular_impulse = [r.state_f32()?, r.state_f32()?, r.state_f32()?];
    let limit_impulse = r.state_f32()?;
    let motor_impulse = r.state_f32()?;
    let j = Joint {
        body_a,
        body_b,
        anchor_a,
        anchor_b,
        kind,
        collide_connected,
        linear_impulse,
        angular_impulse,
        limit_impulse,
        motor_impulse,
    };
    // 앵커(레버암)·축 정규성·임펄스 범위 — 전부 라이브 API와 같은 판정
    j.validate().map_err(|_| SnapshotError::InconsistentState)?;
    Ok(j)
}

pub(crate) fn write_command(w: &mut Writer, c: &PendingCommand) {
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
pub(crate) fn shape_size(s: &Shape) -> usize {
    1 + match s {
        Shape::Sphere { .. } => 4,
        Shape::Capsule { .. } => 8,
        Shape::Cylinder { .. } => 8,
        Shape::Box { .. } => 16,
        Shape::Hull { data: hull, .. } => 4 + hull.points().len() * 12 + 4,
        // nx4 + nz4 + cell4 + count4 + 높이 4바이트씩
        Shape::Heightfield(g) => 4 + 4 + 4 + 4 + g.heights().len() * 4,
        // count4 + 정점 12씩 + count4 + 삼각형 12씩
        Shape::TriMesh(m) => 4 + m.verts().len() * 12 + 4 + m.tris().len() * 12,
        Shape::Triangle { .. } => unreachable!("triangle is never serialized"),
        Shape::Compound { children } => {
            1 + children
                .iter()
                .map(|(_, c)| 12 + 16 + shape_size(c))
                .sum::<usize>()
        }
    }
}

pub(crate) fn body_size(b: &Body) -> usize {
    // 타입1 + pos12 + rot16 + lin12 + ang12 + invM4 + invI12 + fric4 + rest4 + offset12
    // + force12 + torque12 + 셰이프
    // + membership4 + filter4 + sensor1 + ccd1(v11) + weld1(v12) + lock1(v17)
    // + contact_force_threshold4(v18) + linear_damping4 + angular_damping4
    1 + 12
        + 16
        + 12
        + 12
        + 4
        + 12
        + 4
        + 4
        + 12
        + 12
        + 12
        + 4
        + 4
        + 1
        + 1
        + 1
        + 1
        + 4
        + 8
        + shape_size(&b.shape)
}

pub(crate) fn joint_size(j: &Joint) -> usize {
    8 + 8
        + 24
        + match j.kind {
            JointKind::Point => 1,
            // 축 24 + limit(태그1+8) + motor(태그1+8)
            JointKind::Hinge { .. } => 1 + 24 + 9 + 9,
            JointKind::Fixed { .. } => 1 + 16,
            JointKind::Distance { .. } => 1 + 8,
            // 축 12 + rel_rot 16 + limit(태그1+8) + motor(태그1+8)
            JointKind::Prismatic { .. } => 1 + 12 + 16 + 9 + 9,
        }
        + 1
        + 12
        + 12 // angular_impulse[3]
        + 8 // limit_impulse + motor_impulse
}

pub(crate) fn command_size(c: &PendingCommand) -> usize {
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
