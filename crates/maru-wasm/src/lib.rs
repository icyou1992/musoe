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
//! **JS에서 `u64`는 Number가 아니라 `BigInt`다** — WebAssembly JS API는 i64 인자·반환을
//! BigInt로 매핑하고, `Number(handle)`로 저장했다가 되넘기면 TypeError가 난다(검수
//! rev_05c33550 MAJOR). 핸들은 불투명 BigInt로만 보관·전달하고, 실패 센티널 `u64::MAX`는
//! JS에서 `-1n`으로 관측된다(`BigInt.asUintN(64, h) === (1n << 64n) - 1n`과 동치).
//!
//! 스레딩: 이 ABI는 **단일 스레드 전제**다. 같은 월드 포인터에 대한 호출은 직렬이어야
//! 하고(브라우저 메인 스레드에서는 자동으로 성립), 워커 간 공유는 계약 밖이다.
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
/// 스테이징 버퍼(변환·이벤트) 할당 실패. **엔진 상태는 정상**이고 스텝도 이미
/// 반영됐지만, 호스트가 읽는 버퍼가 stale하다 — `maru_sync_transforms`로 재시도한다.
/// 엔진의 `ResourceLimit`(스텝 자체가 거부됨)과 구분해야 해서 코드를 나눈다.
pub const ERR_STAGING_LIMIT: i32 = -7;

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
    /// 콜드 변이(add/remove/set_pose)가 스텝 없이 일어나면 변환 버퍼가 실제 자세와
    /// 어긋난다(검수 rev_05c33550 MAJOR: 첫 스텝 전 렌더가 쓰레기 좌표를 읽음).
    /// 포인터를 내주기 전에 이 플래그로 동기화한다.
    transforms_stale: bool,
    /// 접촉 이벤트 평탄화: [a_lo, a_hi, b_lo, b_hi, kind] × N
    events: Vec<u32>,
    /// 이벤트 스테이징이 월드의 `contact_events`와 어긋남(스테이징 할당 실패 시).
    /// 월드 쪽 원본은 다음 스텝까지 살아 있으므로, 그 전에 동기화하면 복구된다
    /// (확인 검수 rev_b438e81e MAJOR: 복구 경로가 없으면 Started가 영구 유실).
    events_stale: bool,
    /// 월드의 `contact_events`가 **이미 발행 완료된 스텝** 것이라 재스테이징 금지.
    /// `ERR_INVALID_VALUE`(스텝 미실행)가 스테이징을 비운 뒤에도 월드 원본에는 직전
    /// 성공 스텝의 이벤트가 남는다 — 이 상태에서 동기화가 원본을 재복사하면 중복
    /// 재발행 결함이 명시 경로로 부활한다(확인 검수 rev_697a9d54 MAJOR).
    events_suppressed: bool,
    /// 광선·질의 결과 스테이징. 용량은 생성 시 선점(7·2) — 핫패스에서 할당하지 않는다.
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
/// 상태로 넘겨야 한다. 반환된 `&mut` 이 살아 있는 동안 같은 월드에 대한 다른 호출이
/// 없어야 한다(독점 접근) — destroy 후 재사용·이중 destroy·동시 호출은 전부 계약 밖이며
/// C ABI 층에서 감지할 수 없다.
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
    // 질의 스테이징은 크기가 고정(7·2)이므로 여기서 선점한다 — `clear`는 용량을
    // 유지하므로 이후 `extend/push`는 할당하지 않는다(검수 rev_05c33550 MAJOR:
    // 핫패스의 암묵 할당은 오류 코드 대신 trap이 된다).
    let mut query = Vec::new();
    let mut hits = Vec::new();
    if query.try_reserve_exact(7).is_err() || hits.try_reserve_exact(2).is_err() {
        return core::ptr::null_mut();
    }
    Box::into_raw(Box::new(WasmWorld {
        world,
        transforms: Vec::new(),
        transforms_stale: false,
        events: Vec::new(),
        events_stale: false,
        events_suppressed: false,
        query,
        hits,
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
    // **`add_body`가 아니라 `try_add_body`** — `add_body`는 검증 실패에 패닉하고,
    // wasm에서 unwind는 호스트가 진단할 수 없는 trap이 된다.
    match w.world.try_add_body(body) {
        Ok(h) => {
            w.transforms_stale = true;
            pack(h)
        }
        Err(_) => u64::MAX,
    }
}

/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_remove_body(p: *mut WasmWorld, h: u64) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.world.remove_body(unpack(h)) {
        Ok(()) => {
            w.transforms_stale = true;
            OK
        }
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
        Ok(()) => {
            w.transforms_stale = true;
            OK
        }
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
/// 이벤트 버퍼 계약(검수 rev_05c33550 MAJOR): 버퍼는 **이번 호출이 실행한 스텝**의
/// 이벤트다. `ERR_INVALID_VALUE`는 스텝이 아예 실행되지 않은 것이므로 버퍼를 비운다 —
/// 이전 스텝의 Started를 재발행하면 소비자의 데미지·트리거 카운터가 중복 증가한다.
/// `ERR_RESOURCE_LIMIT`/`ERR_STATE_OVERFLOW`는 `step_inner`가 실행된 뒤의 오류라
/// 월드의 이벤트가 곧 이번 스텝 것이고, 그대로 내보낸다.
///
/// `ERR_STAGING_LIMIT`이면 스텝 자체는 성공·반영됐지만 버퍼가 stale하다 —
/// `maru_sync_transforms`·`maru_sync_events`로 재시도한다. 이벤트 원본은 다음 스텝까지
/// 월드에 살아 있으므로 **다음 스텝 전에** 복구해야 한다. 스텝이 오류 코드를 돌려준
/// 경우에도 버퍼를 읽기 전에 같은 동기화를 거치는 것이 안전하다(스텝 오류가 스테이징
/// 오류를 가릴 수 있다).
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_step(p: *mut WasmWorld, dt: f32) -> i32 {
    let w = w_or!(p, ERR_NULL);
    let step = w.world.try_step(dt);
    // 스텝이 실패해도 변환 버퍼는 최신 상태로 맞춰둔다 — 호스트가 마지막 유효 자세를 그린다.
    let staged_t = w.refresh_transforms().is_ok();
    let staged_e = if matches!(step, Err(ApiError::InvalidValue)) {
        // 실행되지 않은 스텝: 빈 버퍼가 곧 정답이다 — stale이 아니고, 월드 원본
        // (직전 성공 스텝 것)은 재스테이징 금지로 봉인한다.
        w.events.clear();
        w.events_stale = false;
        w.events_suppressed = true;
        true
    } else {
        // step_inner가 실행됐다 — 월드의 이벤트는 이번 시도 것이고 발행 대상이다.
        w.events_suppressed = false;
        w.refresh_events().is_ok()
    };
    match step {
        // 스텝 오류가 스테이징 오류에 우선한다 — 호스트가 근본 원인을 먼저 본다.
        Err(e) => code(e),
        Ok(()) if staged_t && staged_e => OK,
        Ok(()) => ERR_STAGING_LIMIT,
    }
}

impl WasmWorld {
    /// 변환 버퍼를 월드와 일치시킨다. 실패하면 버퍼는 stale로 남는다(비워서
    /// 정상처럼 보이게 만들지 않는다 — 길이 0은 "슬롯 0개"라는 유효 상태다).
    fn refresh_transforms(&mut self) -> Result<(), ()> {
        // 실패로 이탈하면 stale로 남는다 — `maru_transforms_ptr`가 다음 취득에서
        // 재동기화를 시도하게 한다. 성공 경로 끝에서만 해제한다.
        self.transforms_stale = true;
        let n = self.world.slot_count();
        let len = n.checked_mul(TRANSFORM_STRIDE).ok_or(())?;
        self.transforms.clear();
        // clear 후 len()==0이므로 additional=len이 곧 목표 용량이다. 용량이 이미
        // 충분하면 무연산 — 정착 프레임에서 재할당하지 않는다.
        self.transforms.try_reserve(len).map_err(|_| ())?;
        self.transforms.resize(len, 0.0);
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
        self.transforms_stale = false;
        Ok(())
    }

    /// 이벤트 스테이징을 월드와 일치시킨다. **clear보다 예약이 먼저다** — 실패하면
    /// 기존 스테이징이 그대로 남고 stale 플래그만 선다. 월드의 `contact_events`는
    /// 다음 스텝까지 불변이므로 재시도(`maru_sync_events`)로 복구할 수 있다
    /// (확인 검수 rev_b438e81e MAJOR: clear를 먼저 하면 실패가 곧 영구 유실이었다).
    fn refresh_events(&mut self) -> Result<(), ()> {
        self.events_stale = true;
        let len = self.world.contact_events().len().checked_mul(5).ok_or(())?;
        // additional은 현재 len() 기준이라 목표 용량과의 차분만 청구한다.
        let need = len.saturating_sub(self.events.len());
        self.events.try_reserve(need).map_err(|_| ())?;
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
        self.events_stale = false;
        Ok(())
    }

    /// 호스트 대면 재스테이징의 단일 정본 — 억제 상태를 소비한다. `maru_step` 내부는
    /// 억제를 직접 갱신하므로 `refresh_events`를 그대로 쓰고, sync·getter는 전부
    /// 이 경로여야 한다(우회하면 rev_697a9d54의 중복 재발행이 부활한다).
    fn restage_events(&mut self) -> Result<(), ()> {
        if self.events_suppressed {
            // 원본은 이미 발행 완료된 스텝 것 — 빈 버퍼가 정답이다.
            self.events.clear();
            self.events_stale = false;
            return Ok(());
        }
        self.refresh_events()
    }
}

/// 변환 버퍼를 월드와 명시적으로 동기화한다. 성공 0, 실패 `ERR_STAGING_LIMIT`.
///
/// 스텝 없이 바디를 추가·제거하거나 pose를 바꾼 직후 버퍼를 읽으려면 이걸 부른다.
/// (`maru_transforms_ptr`가 stale이면 자동으로 동기화하지만, 그 경로는 실패를
/// null로만 알린다 — 오류 코드가 필요하면 이 API를 쓴다.)
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_sync_transforms(p: *mut WasmWorld) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.refresh_transforms() {
        Ok(()) => OK,
        Err(()) => ERR_STAGING_LIMIT,
    }
}

/// 변환 버퍼 포인터 (`Float32Array`로 읽는다). 바디당 7 float: pos3 + quat4(xyzw).
///
/// **`maru_step` 뒤에는 다시 물어봐야 한다** — 바디 수가 바뀌면 버퍼가 재할당돼
/// 포인터가 무효화된다. 성공한 add/remove/set_pose도 **이전에 받아둔 포인터를
/// 무효화한다**: 재할당이 없어도 그 포인터가 보여주는 값은 변이 이전의 스냅샷이다 —
/// 자동 동기화는 이 함수를 다시 부를 때만 실행된다(확인 검수 rev_b438e81e MINOR).
///
/// 콜드 변이(스텝 없는 add/remove/set_pose) 후에는 먼저 동기화한 뒤 내준다 —
/// 동기화가 실패하면 stale 버퍼 대신 null을 돌려준다(검수 rev_05c33550 MAJOR:
/// 첫 스텝 전의 버퍼는 길이 0이라 호스트가 무관한 선형 메모리를 읽는다).
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_transforms_ptr(p: *mut WasmWorld) -> *const f32 {
    let w = w_or!(p, core::ptr::null());
    if w.transforms_stale && w.refresh_transforms().is_err() {
        return core::ptr::null();
    }
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

/// 감쇠 설정 (1/s). 구르는 포탄 같은 바디가 스스로 멈춰 잠들게 한다 —
/// 각 감쇠 없는 구는 마찰로 선속만 줄고 회전은 남아 영원히 구른다.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_set_damping(
    p: *mut WasmWorld,
    h: u64,
    linear: f32,
    angular: f32,
) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.world.set_damping(unpack(h), linear, angular) {
        Ok(()) => OK,
        Err(e) => code(e),
    }
}

/// 수면 임계 조정 (설계 §4.6). 장면 성격에 따라 수면 공격성을 고른다 — 대량 잔해
/// 장면은 높게(빨리 잠듦 = 성능·안정), 톨 스택 정밀 장면은 낮게. 기본값(0.1/0.15/0.5)은
/// 보수적이다 — rapier 기본(0.4)의 4배 엄격. 실패하면 음수 오류 코드.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_set_sleep_thresholds(
    p: *mut WasmWorld,
    linear: f32,
    angular: f32,
    time_to_sleep: f32,
) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w
        .world
        .try_set_sleep_thresholds(linear, angular, time_to_sleep)
    {
        Ok(()) => OK,
        Err(e) => code(e),
    }
}

/// 잠들어 있으면 1 (죽은 핸들·비수면 0). 호스트 시각화·LOD용 — 수면 도시가 어둡게,
/// 충격 전파가 깨어나는 것으로 보인다.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_is_asleep(p: *mut WasmWorld, h: u64) -> u32 {
    let w = w_or!(p, 0);
    w.world.is_asleep(unpack(h)) as u32
}

// ---------- 접촉 이벤트 ----------

/// 이벤트 스테이징을 명시적으로 동기화한다. 성공 0, 실패 `ERR_STAGING_LIMIT`.
///
/// `maru_step`이 `ERR_STAGING_LIMIT`을 돌려줬을 때의 복구 경로다. 월드의 이벤트
/// 원본은 다음 스텝까지 불변이므로 **다음 스텝 전에** 부르면 유실이 없다.
///
/// 직전 호출이 `ERR_INVALID_VALUE`(스텝 미실행)였다면 빈 버퍼를 유지하고 성공을
/// 돌려준다 — 월드에 남은 원본은 이미 발행 완료된 스텝 것이라 재복사하면 중복
/// 재발행이다(확인 검수 rev_697a9d54 MAJOR). 아무 때나 불러도 안전한 이유다.
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_sync_events(p: *mut WasmWorld) -> i32 {
    let w = w_or!(p, ERR_NULL);
    match w.restage_events() {
        Ok(()) => OK,
        Err(()) => ERR_STAGING_LIMIT,
    }
}

/// 이벤트 버퍼 포인터. `[a_lo, a_hi, b_lo, b_hi, kind]` × N (`Uint32Array`).
/// `kind`: 0 = Started, 1 = Stopped.
///
/// 스테이징이 stale이면(직전 스텝의 스테이징 실패) 먼저 동기화한 뒤 내주고,
/// 동기화가 실패하면 null이다 — stale 버퍼를 이번 스텝 것처럼 읽으면 이전 이벤트를
/// 중복 소비한다(확인 검수 rev_b438e81e MAJOR).
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_events_ptr(p: *mut WasmWorld) -> *const u32 {
    let w = w_or!(p, core::ptr::null());
    if w.events_stale && w.restage_events().is_err() {
        return core::ptr::null();
    }
    w.events.as_ptr()
}

/// 이벤트 개수 (u32 개수가 아니라 **이벤트 개수**).
///
/// `maru_events_ptr`와 같은 동기화 규칙을 따른다 — stale이면 동기화하고, 실패하면
/// 0을 돌려준다(개수만 믿고 stale 버퍼를 읽는 조합을 막는다; 이벤트 자체는
/// `maru_sync_events` 성공 후 다시 읽을 수 있다).
///
/// # Safety
/// `p`는 살아있는 월드 포인터여야 한다.
#[no_mangle]
pub unsafe extern "C" fn maru_events_len(p: *mut WasmWorld) -> u32 {
    let w = w_or!(p, 0);
    if w.events_stale && w.restage_events().is_err() {
        return 0;
    }
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
            // 3초 정착이면 잠들어 있어야 하고, 수면 관측 ABI가 그걸 보여야 한다.
            assert_eq!(maru_is_asleep(w, ball), 1, "정착한 공이 잠들지 않았다");
            assert_eq!(maru_is_asleep(w, u64::MAX), 0, "죽은 핸들은 0");

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

    /// 콜드 변이(스텝 없는 add/set_pose) 직후에도 변환 버퍼가 실제 자세를 보여야 한다
    /// (검수 rev_05c33550 MAJOR: 첫 스텝 전의 버퍼는 길이 0이라 호스트가 쓰레기를 읽었다).
    #[test]
    fn transforms_are_fresh_before_the_first_step_and_after_set_pose() {
        unsafe {
            let w = maru_world_create(0.0, -9.81, 0.0, 0);
            let ball = maru_add_body(
                w,
                BODY_DYNAMIC,
                SHAPE_SPHERE,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            );
            assert_ne!(ball, u64::MAX);

            // 스텝 없이 바로 읽는다 — 자동 동기화가 없으면 길이 0 버퍼다.
            let n = maru_slot_count(w) as usize;
            assert_eq!(n, 1);
            let t = maru_transforms_ptr(w);
            assert!(!t.is_null());
            let buf = core::slice::from_raw_parts(t, n * TRANSFORM_STRIDE);
            assert_eq!(&buf[0..3], &[1.0, 2.0, 3.0]);

            // 콜드 set_pose 직후에도 마찬가지다.
            assert_eq!(
                maru_set_pose(w, ball, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0),
                OK
            );
            assert_eq!(maru_sync_transforms(w), OK);
            let buf = core::slice::from_raw_parts(maru_transforms_ptr(w), TRANSFORM_STRIDE);
            assert_eq!(&buf[0..3], &[7.0, 8.0, 9.0]);
            maru_world_destroy(w);
        }
    }

    /// 실패한 스텝은 이전 스텝의 이벤트를 재발행하면 안 된다 (검수 rev_05c33550 MAJOR:
    /// dt=0 실수 한 번에 Started가 두 번 소비돼 트리거 카운터가 중복 증가한다).
    #[test]
    fn a_failed_step_does_not_replay_the_previous_steps_events() {
        unsafe {
            let w = maru_world_create(0.0, -9.81, 0.0, 0);
            maru_add_body(
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
            maru_add_body(
                w,
                BODY_DYNAMIC,
                SHAPE_SPHERE,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            );
            let mut saw = false;
            for _ in 0..180 {
                assert_eq!(maru_step(w, 1.0 / 60.0), OK);
                if maru_events_len(w) > 0 {
                    saw = true;
                    // Started가 버퍼에 있는 바로 이 시점에 잘못된 dt를 보낸다.
                    assert_eq!(maru_step(w, 0.0), ERR_INVALID_VALUE);
                    assert_eq!(
                        maru_events_len(w),
                        0,
                        "실행되지 않은 스텝이 이벤트를 재발행했다"
                    );
                    // 명시 동기화도 같은 답이어야 한다 — 월드 원본(직전 성공 스텝의
                    // Started)을 재복사하면 중복 재발행이 명시 경로로 부활한다
                    // (확인 검수 rev_697a9d54 MAJOR).
                    assert_eq!(maru_sync_events(w), OK);
                    assert_eq!(
                        maru_events_len(w),
                        0,
                        "명시 동기화가 발행 완료된 이벤트를 재복사했다"
                    );
                    break;
                }
            }
            assert!(saw, "공이 바닥에 닿아 Started가 나와야 한다");
            maru_world_destroy(w);
        }
    }

    /// `ERR_STAGING_LIMIT` 복구 계약의 근거: 이벤트 원본은 다음 스텝까지 월드에 살아
    /// 있으므로, 스텝 후 재동기화는 같은 이벤트를 그대로 재구성한다(확인 검수
    /// rev_b438e81e MAJOR — 이 성질이 없으면 `maru_sync_events`는 복구 경로가 아니다).
    #[test]
    fn sync_events_restages_the_same_events_until_the_next_step() {
        unsafe {
            let w = maru_world_create(0.0, -9.81, 0.0, 0);
            maru_add_body(
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
            maru_add_body(
                w,
                BODY_DYNAMIC,
                SHAPE_SPHERE,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            );
            let mut saw = false;
            for _ in 0..180 {
                assert_eq!(maru_step(w, 1.0 / 60.0), OK);
                let n = maru_events_len(w) as usize;
                if n > 0 {
                    saw = true;
                    let before: Vec<u32> =
                        core::slice::from_raw_parts(maru_events_ptr(w), n * 5).to_vec();
                    assert_eq!(maru_sync_events(w), OK);
                    assert_eq!(maru_events_len(w) as usize, n);
                    let after = core::slice::from_raw_parts(maru_events_ptr(w), n * 5);
                    assert_eq!(before, after, "재동기화가 이벤트를 바꿨다");
                    break;
                }
            }
            assert!(saw, "공이 바닥에 닿아 Started가 나와야 한다");
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
