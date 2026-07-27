//! Jolt 5.0 (joltc-sys) 대조 벤치 — compare/src/main.rs의 pile 씬과 동일 구성.
//!
//! 공정성 명시:
//! - **워커 스레드 0** (JobSystemThreadPool_new3(..., 0) = 호출 스레드에서 실행) —
//!   maru·rapier 대조가 전부 단일 스레드이므로 같은 조건으로 잰다. Jolt의 본령은
//!   멀티스레드 스케일링이라 이 수치는 Jolt의 최선이 아니다 — 그것까지 명시한다.
//! - 수면·솔버 파라미터는 **Jolt 기본값**(하네스 원칙: 각자의 기본 + 헤더에 명시).
//! - 씬: 바닥 박스 30×1×30 @ y=-1, 구 r=0.5 격자 1.1 간격 n개 — pile과 동일.
//!
//! 실행: cargo run --release --manifest-path compare/Cargo.toml --bin jolt -- [n] [steps] [reps]

use joltc_sys::*;
use std::ffi::{c_uint, c_void};
use std::ptr;
use std::time::Instant;

const OL_NON_MOVING: JPC_ObjectLayer = 0;
const OL_MOVING: JPC_ObjectLayer = 1;
const BPL_NON_MOVING: JPC_BroadPhaseLayer = 0;
const BPL_MOVING: JPC_BroadPhaseLayer = 1;

unsafe extern "C" fn bpl_count(_: *const c_void) -> c_uint {
    2
}
unsafe extern "C" fn bpl_map(_: *const c_void, l: JPC_ObjectLayer) -> JPC_BroadPhaseLayer {
    if l == OL_NON_MOVING {
        BPL_NON_MOVING
    } else {
        BPL_MOVING
    }
}
unsafe extern "C" fn ovb(_: *const c_void, l1: JPC_ObjectLayer, l2: JPC_BroadPhaseLayer) -> bool {
    if l1 == OL_NON_MOVING {
        l2 == BPL_MOVING
    } else {
        true
    }
}
unsafe extern "C" fn ovo(_: *const c_void, l1: JPC_ObjectLayer, l2: JPC_ObjectLayer) -> bool {
    if l1 == OL_NON_MOVING {
        l2 == OL_MOVING
    } else {
        true
    }
}

fn vec3(x: f32, y: f32, z: f32) -> JPC_Vec3 {
    JPC_Vec3 { x, y, z, _w: z }
}
fn rvec3(x: Real, y: Real, z: Real) -> JPC_RVec3 {
    JPC_RVec3 { x, y, z, _w: z }
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(240);
    let reps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    println!("Jolt 5.0 (joltc-sys, 워커 0 = 단일 스레드) — pile n={n}, dt=1/60, 기본 설정");

    let mut best_total = f64::MAX;
    let mut best_late = f64::MAX;
    for _ in 0..reps {
        let (total, late, asleep, lowest) = unsafe { run_pile(n, steps) };
        best_total = best_total.min(total);
        best_late = best_late.min(late);
        eprintln!(
            "  rep: {total:.3} ms/step  정착구간 {late:.3} ms/step  수면 {asleep}/{n}  lowest_y={lowest:.2}"
        );
    }
    println!("  jolt   전체 {best_total:>7.3} ms/step   정착구간 {best_late:>7.3} ms/step");
}

unsafe fn run_pile(n: usize, steps: usize) -> (f64, f64, usize, f32) {
    use std::sync::OnceLock;
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        JPC_RegisterDefaultAllocator();
        JPC_FactoryInit();
        JPC_RegisterTypes();
    });

    let temp_allocator = JPC_TempAllocatorImpl_new(64 * 1024 * 1024);
    // 워커 스레드 0 — 단일 스레드 대조 (파일 머리 주석).
    let job_system =
        JPC_JobSystemThreadPool_new3(JPC_MAX_PHYSICS_JOBS as _, JPC_MAX_PHYSICS_BARRIERS as _, 1);

    let bpl = JPC_BroadPhaseLayerInterface_new(
        ptr::null(),
        JPC_BroadPhaseLayerInterfaceFns {
            GetNumBroadPhaseLayers: Some(bpl_count as _),
            GetBroadPhaseLayer: Some(bpl_map as _),
        },
    );
    let ovb_f = JPC_ObjectVsBroadPhaseLayerFilter_new(
        ptr::null_mut(),
        JPC_ObjectVsBroadPhaseLayerFilterFns { ShouldCollide: Some(ovb as _) },
    );
    let ovo_f = JPC_ObjectLayerPairFilter_new(
        ptr::null_mut(),
        JPC_ObjectLayerPairFilterFns { ShouldCollide: Some(ovo as _) },
    );

    let system = JPC_PhysicsSystem_new();
    JPC_PhysicsSystem_Init(system, 8192, 0, 65536, 20480, bpl, ovb_f, ovo_f);
    let bi = JPC_PhysicsSystem_GetBodyInterface(system);

    // 바닥
    let mut floor_shape: *mut JPC_Shape = ptr::null_mut();
    let mut err: *mut JPC_String = ptr::null_mut();
    assert!(JPC_BoxShapeSettings_Create(
        &JPC_BoxShapeSettings { HalfExtent: vec3(30.0, 1.0, 30.0), ..Default::default() },
        &mut floor_shape,
        &mut err
    ));
    let floor = JPC_BodyInterface_CreateBody(
        bi,
        &JPC_BodyCreationSettings {
            Position: rvec3(0.0, -1.0, 0.0),
            MotionType: JPC_MOTION_TYPE_STATIC,
            ObjectLayer: OL_NON_MOVING,
            Shape: floor_shape,
            ..Default::default()
        },
    );
    JPC_BodyInterface_AddBody(bi, JPC_Body_GetID(floor), JPC_ACTIVATION_DONT_ACTIVATE);

    // 구 더미 — pile과 동일 격자
    let mut sphere_shape: *mut JPC_Shape = ptr::null_mut();
    assert!(JPC_SphereShapeSettings_Create(
        &JPC_SphereShapeSettings { Radius: 0.5, ..Default::default() },
        &mut sphere_shape,
        &mut err
    ));
    let side = (n as f32).cbrt().ceil() as usize;
    let mut ids = Vec::new();
    'outer: for iy in 0..side * 3 {
        for ix in 0..side {
            for iz in 0..side {
                if ids.len() >= n {
                    break 'outer;
                }
                let x = ix as f32 * 1.1 - side as f32 * 0.55;
                let z = iz as f32 * 1.1 - side as f32 * 0.55;
                let y = 0.55 + iy as f32 * 1.1;
                let b = JPC_BodyInterface_CreateBody(
                    bi,
                    &JPC_BodyCreationSettings {
                        Position: rvec3(x as Real, y as Real, z as Real),
                        MotionType: JPC_MOTION_TYPE_DYNAMIC,
                        ObjectLayer: OL_MOVING,
                        Shape: sphere_shape,
                        ..Default::default()
                    },
                );
                let id = JPC_Body_GetID(b);
                JPC_BodyInterface_AddBody(bi, id, JPC_ACTIVATION_ACTIVATE);
                ids.push(id);
            }
        }
    }
    JPC_PhysicsSystem_OptimizeBroadPhase(system);

    // 워밍업 60 + 측정 (main.rs pile과 동일 창)
    for _ in 0..60 {
        JPC_PhysicsSystem_Update(system, 1.0 / 60.0, 1, temp_allocator, job_system);
    }
    let t0 = Instant::now();
    let mut t_late = std::time::Duration::ZERO;
    for si in 0..steps {
        let ts = Instant::now();
        JPC_PhysicsSystem_Update(system, 1.0 / 60.0, 1, temp_allocator, job_system);
        if si >= steps / 2 {
            t_late += ts.elapsed();
        }
    }
    let total = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let late = t_late.as_secs_f64() * 1000.0 / (steps - steps / 2) as f64;

    let mut asleep = 0usize;
    let mut lowest = f32::MAX;
    for &id in &ids {
        if !JPC_BodyInterface_IsActive(bi, id) {
            asleep += 1;
        }
        let p = JPC_BodyInterface_GetCenterOfMassPosition(bi, id);
        lowest = lowest.min(p.y as f32);
    }

    JPC_PhysicsSystem_delete(system);
    JPC_BroadPhaseLayerInterface_delete(bpl);
    JPC_ObjectVsBroadPhaseLayerFilter_delete(ovb_f);
    JPC_ObjectLayerPairFilter_delete(ovo_f);
    JPC_JobSystemThreadPool_delete(job_system);
    JPC_TempAllocatorImpl_delete(temp_allocator);
    (total, late, asleep, lowest)
}
