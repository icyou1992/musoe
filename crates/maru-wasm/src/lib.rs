//! Maru 웹 바인딩 — **flat buffer + C ABI**.
//!
//! 설계 §5: JS↔WASM 경계가 제1 성능 변수다(Rapier 2025 실측에서 JS 패키지 2–5× 고속화의
//! 주 출처가 마샬링 계층이었다). 그래서 이 층의 규칙은 셋뿐이다:
//!
//! 1. **핫패스는 버퍼로만 오간다.** 매 프레임 바디마다 함수를 부르지 않는다 —
//!    `maru_transforms_ptr`이 가리키는 선형 메모리를 JS가 `Float32Array`로 직접 읽는다.
//! 2. **호출당 객체를 만들지 않는다.** 반환은 전부 스칼라이거나 미리 잡아둔 버퍼다.
//! 3. **패닉이 경계를 넘지 않는다.** 엔진의 fallible 경로(`try_*`)만 쓰고 실패는
//!    음수 오류 코드로 돌려준다. wasm에서 unwind는 곧 trap이라 호스트가 진단할 수 없다.
//!
//! 핸들은 JS에 `u32` 한 쌍이 아니라 **`u64`로 합쳐** 넘긴다(하위 32비트=슬롯, 상위=세대).
//! JS의 Number는 2^53까지 정확하므로 그대로 다뤄도 안전하다.
//!
//! `crate-type`에 `rlib`을 함께 둔 이유: **게이트가 같은 ABI를 네이티브로 테스트**하기
//! 위해서다. 바인딩만 wasm에서 손으로 확인하는 층이 되면 조용히 썩는다.

use maru_collide::{Pose, Shape};
use maru_dynamics::{ApiError, Body, BodyHandle, ContactEventKind, World, WorldParams};
use maru_math::{Quat, Vec3};

// ---------- 오류 코드 (음수 = 실패) ----------

pub const OK: i32 = 0;
pub const ERR_NULL: i32 = -1;
pub const ERR_INVALID_VALUE: i32 = -2;
pub const ERR_STALE_HANDLE: i32 = -3;
pub const ERR_IMMUTABLE_STATIC: i32 = -4;
pub const ERR_RESOURCE_LIMIT: i32 = -5;
pub const ERR_STATE_OVERFLOW: i32 = -6;

fn code(e: ApiError) -> i32 {
    match e {
        ApiError::InvalidValue => ERR_INVALID_VALUE,
        ApiError::StaleHandle => ERR_STALE_HANDLE,
        ApiError::ImmutableStatic => ERR_IMMUTABLE_STATIC,
        ApiError::ResourceLimit => ERR_RESOURCE_LIMIT,
        ApiError::StateOverflow => ERR_STATE_OVERFLOW,
    }
}

// ---------- 셰이프 태그 ----------

pub const SHAPE_SPHERE: u32 = 0;
pub const SHAPE_BOX: u32 = 1;
pub const SHAPE_CAPSULE: u32 = 2;

pub const BODY_STATIC: u32 = 0;
pub const BODY_DYNAMIC: u32 = 1;

/// 바디 하나가 변환 버퍼에서 차지하는 f32 개수: pos3 + quat4.
pub const TRANSFORM_STRIDE: usize = 7;

/// 월드 + 호스트에 노출하는 스테이징 버퍼.
///
/// 버퍼를 월드가 소유하는 이유: JS가 매 프레임 할당하지 않게 하려면 포인터가
/// 프레임 사이에 살아 있어야 한다. 크기가 바뀌면 포인터가 무효화되므로
/// **`maru_step` 뒤에는 포인터를 다시 물어봐야 한다**(문서 계약).
pub struct WasmWorld {
    world: World,
    transforms: Vec<f32>,
    /// 접촉 이벤트 평탄화: [a_lo, a_hi, b_lo, b_hi, kind] × N
    events: Vec<u32>,
    /// 광선·질의 결과 스테이징
    query: Vec<f32>,
    hits: Vec<u32>,
}

fn pack(h: BodyHandle) -> u64 {
    (h.index as u64) | ((h.generation as u64) << 32)
}

fn unpack(v: u64) -> BodyHandle {
    BodyHandle {
        index: v as u32,
        generation: (v >> 32) as u32,
    }
}

/// `*mut WasmWorld`를 안전하게 되돌린다. null이면 `None`.
///
/// # Safety
/// 호출자가 `maru_world_create`가 준 포인터를, 아직 `maru_world_destroy` 하지 않은
/// 상태로 넘겨야 한다.
unsafe fn world<'a>(p: *mut WasmWorld) -> Option<&'a mut WasmWorld> {
    if p.is_null() {
        None
    } else {
        Some(&mut *p)
    }
}

macro_rules! w_or {
    ($p:expr, $ret:expr) => {
        match unsafe { world($p) } {
            Some(w) => w,
            None => return $ret,
        }
    };
}

// ---------- 생성·소멸 ----------

/// 월드를 만든다. 실패하면 null.
///
/// # Safety
/// 반환 포인터는 `maru_world_destroy`로만 해제한다.
#[no_mangle]
pub extern "C" fn maru_world_create(gx: f32, gy: f32, gz: f32, substeps: u32) -> *mut WasmWorld {
    let d = WorldParams::default();
    let params = WorldParams {
        gravity: Vec3::new(gx, gy, gz),
        substeps: if substeps == 0 { d.substeps } else { substeps },
        ..d
    };
    // **패닉 대신 null.** 잘못된 파라미터로 trap을 내면 호스트가 원인을 못 본다.
    let Ok(world) = World::try_new(params) else {
        return core::ptr::null_mut();
    };
    Box::into_raw(Box::new(WasmWorld {
        world,
        transforms: Vec::new(),
        events: Vec::new(),
        query: Vec::new(),
        hits: Vec::new(),
    }))
}

/// # Safety
/// `p`는 `maru_world_create`가 준 포인터여야 하고, 두 번 해제하면 안 된다.
#[no_mangle]
pub unsafe extern "C" fn maru_world_destroy(p: *mut WasmWorld) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

// ---------- 바디 ----------

/// 바디를 추가하고 핸들(u64)을 돌려준다. 실패하면 `u64::MAX`.
///
/// `shape` 태그별 파라미터: Sphere=(a=반지름), Box=(a,b,c=half, d=convex radius),
/// Capsule=(a=half_height, b=반지름).
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn maru_add_body(
    p: *mut WasmWorld,
    body_type: u32,
    shape: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    density: f32,
) -> u64 {
    let w = w_or!(p, u64::MAX);
    let shape = match shape {
        SHAPE_SPHERE => Shape::Sphere { radius: a },
        SHAPE_BOX => Shape::Box {
            half: Vec3::new(a, b, c),
            radius: d,
        },
        SHAPE_CAPSULE => Shape::Capsule {
            half_height: a,
            radius: b,
        },
        _ => return u64::MAX,
    };
    let pose = Pose {
        pos: Vec3::new(px, py, pz),
        rot: Quat {
            x: qx,
            y: qy,
            z: qz,
            w: qw,
        },
    };
    let body = match body_type {
        BODY_STATIC => Body::try_new_static(shape, pose),
        BODY_DYNAMIC => Body::try_new_dynamic(shape, pose, density),
        _ => return u64::MAX,
    };
    let Ok(body) = body else {
        return u64::MAX;
    };
    pack(w.world.add_body(body))
}

/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_remove_body(p: *mut WasmWorld, h: u64) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.world.remove_body(unpack(h)) {
        Ok(()) => OK,
        Err(e) => code(e),
    }
}

/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn maru_set_pose(
    p: *mut WasmWorld,
    h: u64,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
) -> i32 {
    let w = w_or!(p, ERR_NULL);
    let pose = Pose {
        pos: Vec3::new(px, py, pz),
        rot: Quat {
            x: qx,
            y: qy,
            z: qz,
            w: qw,
        },
    };
    match w.world.set_pose(unpack(h), pose) {
        Ok(()) => OK,
        Err(e) => code(e),
    }
}

/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_apply_impulse(
    p: *mut WasmWorld,
    h: u64,
    x: f32,
    y: f32,
    z: f32,
) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.world.apply_impulse(unpack(h), Vec3::new(x, y, z)) {
        Ok(()) => OK,
        Err(e) => code(e),
    }
}

/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_apply_force(
    p: *mut WasmWorld,
    h: u64,
    x: f32,
    y: f32,
    z: f32,
) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.world.apply_force(unpack(h), Vec3::new(x, y, z)) {
        Ok(()) => OK,
        Err(e) => code(e),
    }
}

// ---------- 스텝 + 변환 버퍼 ----------

/// 한 스텝. 실패하면 음수 오류 코드.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_step(p: *mut WasmWorld, dt: f32) -> i32 {
    let w = w_or!(p, ERR_NULL);
    let r = match w.world.try_step(dt) {
        Ok(()) => OK,
        Err(e) => code(e),
    };
    // 스텝이 실패해도 버퍼는 최신 상태로 맞춰둔다 — 호스트가 마지막 유효 자세를 그린다.
    w.refresh_transforms();
    w.refresh_events();
    r
}

impl WasmWorld {
    fn refresh_transforms(&mut self) {
        let n = self.world.slot_count();
        self.transforms.clear();
        self.transforms.resize(n * TRANSFORM_STRIDE, 0.0);
        for i in 0..n {
            // 살아있지 않은 슬롯은 0으로 남긴다 — 호스트는 `maru_is_alive`로 거른다.
            let Some(b) = self.world.body_at_slot(i) else {
                continue;
            };
            let (pos, rot) = (b.pose().pos, b.pose().rot);
            let o = i * TRANSFORM_STRIDE;
            self.transforms[o] = pos.x;
            self.transforms[o + 1] = pos.y;
            self.transforms[o + 2] = pos.z;
            self.transforms[o + 3] = rot.x;
            self.transforms[o + 4] = rot.y;
            self.transforms[o + 5] = rot.z;
            self.transforms[o + 6] = rot.w;
        }
    }

    fn refresh_events(&mut self) {
        self.events.clear();
        for e in self.world.contact_events() {
            let (a, b) = (pack(e.a), pack(e.b));
            self.events.push(a as u32);
            self.events.push((a >> 32) as u32);
            self.events.push(b as u32);
            self.events.push((b >> 32) as u32);
            self.events.push(match e.kind {
                ContactEventKind::Started => 0,
                ContactEventKind::Stopped => 1,
            });
        }
    }
}

/// 변환 버퍼 포인터 (`Float32Array`로 읽는다). 바디당 7 float: pos3 + quat4(xyzw).
///
/// **`maru_step` 뒤에는 다시 물어봐야 한다** — 바디 수가 바뀌면 버퍼가 재할당돼
/// 포인터가 무효화된다.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_transforms_ptr(p: *mut WasmWorld) -> *const f32 {
    let w = w_or!(p, core::ptr::null());
    w.transforms.as_ptr()
}

/// 변환 버퍼의 슬롯 수 (= 바디 슬롯 수). float 개수는 이것 × `TRANSFORM_STRIDE`.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_slot_count(p: *mut WasmWorld) -> u32 {
    let w = w_or!(p, 0);
    w.world.slot_count() as u32
}

/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_is_alive(p: *mut WasmWorld, h: u64) -> u32 {
    let w = w_or!(p, 0);
    w.world.is_alive(unpack(h)) as u32
}

// ---------- 접촉 이벤트 ----------

/// 이벤트 버퍼 포인터. `[a_lo, a_hi, b_lo, b_hi, kind]` × N (`Uint32Array`).
/// `kind`: 0 = Started, 1 = Stopped.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_events_ptr(p: *mut WasmWorld) -> *const u32 {
    let w = w_or!(p, core::ptr::null());
    w.events.as_ptr()
}

/// 이벤트 개수 (u32 개수가 아니라 **이벤트 개수**).
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_events_len(p: *mut WasmWorld) -> u32 {
    let w = w_or!(p, 0);
    (w.events.len() / 5) as u32
}

// ---------- 질의 ----------

/// 가장 가까운 광선 히트. 맞으면 1, 안 맞으면 0, 실패하면 음수.
/// 결과는 `maru_query_ptr`의 [toi, px,py,pz, nx,ny,nz] 7 float + 핸들은
/// `maru_query_hit_handle`로 가져온다.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn maru_cast_ray(
    p: *mut WasmWorld,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    max_toi: f32,
) -> i32 {
    let w = w_or!(p, ERR_NULL);
    w.query.clear();
    w.hits.clear();
    match w
        .world
        .cast_ray(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz), max_toi)
    {
        Err(e) => code(e),
        Ok(None) => 0,
        Ok(Some(hit)) => {
            w.query.extend_from_slice(&[
                hit.toi,
                hit.point.x,
                hit.point.y,
                hit.point.z,
                hit.normal.x,
                hit.normal.y,
                hit.normal.z,
            ]);
            let h = pack(hit.body);
            w.hits.push(h as u32);
            w.hits.push((h >> 32) as u32);
            1
        }
    }
}

/// 질의 결과 float 버퍼.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_query_ptr(p: *mut WasmWorld) -> *const f32 {
    let w = w_or!(p, core::ptr::null());
    w.query.as_ptr()
}

/// 직전 `maru_cast_ray`가 맞힌 바디 핸들. 못 맞았으면 `u64::MAX`.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_query_hit_handle(p: *mut WasmWorld) -> u64 {
    let w = w_or!(p, u64::MAX);
    if w.hits.len() < 2 {
        return u64::MAX;
    }
    (w.hits[0] as u64) | ((w.hits[1] as u64) << 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **게이트가 C ABI를 네이티브로 돈다** — 바인딩만 손으로 확인하는 층이 되면 썩는다.
    #[test]
    fn abi_roundtrip_drops_a_sphere_on_the_floor() {
        unsafe {
            let w = maru_world_create(0.0, -9.81, 0.0, 0);
            assert!(!w.is_null());
            let ground = maru_add_body(
                w,
                BODY_STATIC,
                SHAPE_BOX,
                10.0,
                1.0,
                10.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            );
            let ball = maru_add_body(
                w,
                BODY_DYNAMIC,
                SHAPE_SPHERE,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            );
            assert_ne!(ground, u64::MAX);
            assert_ne!(ball, u64::MAX);
            assert_eq!(maru_is_alive(w, ball), 1);

            let mut saw_contact = false;
            for _ in 0..180 {
                assert_eq!(maru_step(w, 1.0 / 60.0), OK);
                saw_contact |= maru_events_len(w) > 0;
            }
            assert!(saw_contact, "바닥에 닿으면 이벤트가 나와야 한다");

            // 변환 버퍼를 JS가 읽듯이 읽는다
            let n = maru_slot_count(w) as usize;
            assert_eq!(n, 2);
            let t = maru_transforms_ptr(w);
            assert!(!t.is_null());
            let buf = core::slice::from_raw_parts(t, n * TRANSFORM_STRIDE);
            let ball_y = buf[TRANSFORM_STRIDE + 1];
            assert!(
                (ball_y - 0.5).abs() < 0.05,
                "공이 바닥에 서야 한다: y={ball_y}"
            );

            // 광선으로 맞혀본다
            let hit = maru_cast_ray(w, -5.0, 0.5, 0.0, 1.0, 0.0, 0.0, 100.0);
            assert_eq!(hit, 1, "공을 맞혀야 한다");
            assert_eq!(maru_query_hit_handle(w), ball);
            let q = core::slice::from_raw_parts(maru_query_ptr(w), 7);
            assert!(q[0] > 0.0 && q[0] < 5.0, "toi={}", q[0]);

            maru_world_destroy(w);
        }
    }

    /// null·잘못된 입력이 **패닉 대신 오류 코드**로 나와야 한다 (wasm에서 unwind = trap).
    #[test]
    fn bad_input_returns_codes_not_panics() {
        unsafe {
            assert_eq!(maru_step(core::ptr::null_mut(), 0.016), ERR_NULL);
            assert_eq!(maru_slot_count(core::ptr::null_mut()), 0);
            assert!(maru_transforms_ptr(core::ptr::null_mut()).is_null());

            let w = maru_world_create(0.0, -9.81, 0.0, 0);
            // dt 계약 위반
            assert_eq!(maru_step(w, 0.0), ERR_INVALID_VALUE);
            assert_eq!(maru_step(w, -1.0), ERR_INVALID_VALUE);
            // 알 수 없는 셰이프 태그
            assert_eq!(
                maru_add_body(
                    w,
                    BODY_DYNAMIC,
                    99,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0
                ),
                u64::MAX
            );
            // 죽은 핸들
            assert_eq!(maru_remove_body(w, u64::MAX), ERR_STALE_HANDLE);
            // 0 방향 광선
            assert_eq!(
                maru_cast_ray(w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0),
                ERR_INVALID_VALUE
            );
            maru_world_destroy(w);
        }
    }
}
