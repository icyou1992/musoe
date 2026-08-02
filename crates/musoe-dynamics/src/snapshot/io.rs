//! 스냅샷 **바이트 층** — 오류 타입, 상한 상수, `Writer`/`Reader` 원시 연산.
//!
//! 포맷의 의미는 `codec`이 안다. 여기는 "바이트를 안전하게 읽고 쓴다"만 책임진다 —
//! 신뢰하지 않는 입력의 경계 검사(길이·개수·잔여 바이트)가 전부 이 층에 모인다.

use crate::body::{Body, BodyHandle};
use crate::command::Command;
use crate::joint::Joint;
use musoe_math::{Quat, Vec3};

/// 포맷 매직·버전 — 다른 빌드/포맷의 바이트를 조용히 먹지 않기 위해.
pub(crate) const MAGIC: u32 = 0x4d41_5255; // "MARU"
/// v11: Body에 CCD linear cast opt-in 플래그(1바이트) 추가.
/// v5: hull 점 상한 1024→32, 조인트 K 조건수 검증 추가.
///
/// **정책 변경은 wire 의미 변경이다.** v4 시절 빌드가 만든 33~1024점 스냅샷은 지금 거부되고,
/// K-특이 조인트를 담은 v4도 거부된다 — 같은 버전 번호가 다른 계약을 뜻하면 안 되므로 올린다.
/// (v4: WorldParams·OOB 플래그·앵커/최소 크기 / v3: 예약 슬롯·AddBody 핸들·last_report)
/// v13: `JointKind::Prismatic`(태그 4) 추가 + `Motor::max_torque` → `max_effort`
///      (선형 조인트에서 단위가 힘이라 이름이 거짓말이 된다 — wire 값 자체는 그대로).
/// v14: 쿼터니언 **복원 정책** 변경 — 무조건 재정규화에서 비트 멱등 캐노니컬화로.
///      바이트 형식은 그대로지만 **같은 바이트가 다른 값으로 복원된다**: 구 빌드가 저장한
///      주기-2 쿼터니언을 구 리더는 한 번 더 정규화하고 새 리더는 비트 그대로 둔다.
///      같은 버전 번호가 두 계약을 뜻하면 안 되므로 올린다(위 정책 그대로).
/// 스냅샷 포맷 버전.
///
/// **정확히 일치해야 복원된다.** 하위 호환 읽기를 두지 않는 이유: 이 형식의 필드는 물리의
/// 캐노니컬 상태이고, 옛 버전을 "적당히" 읽으면 빠진 필드가 기본값으로 채워져 **같은
/// 바이트가 다른 미래**를 만든다(그 사고를 이미 v13→v14에서 겪었다). 버전이 다르면
/// `BadVersion(found)`으로 거부하고, 호스트가 마이그레이션 여부를 스스로 정한다.
pub const VERSION: u32 = 19;

/// 크레이트 밖으로 나가는 이름(재수출).
pub use self::VERSION as SNAPSHOT_VERSION;

/// 입력 자체의 상한. 정상 월드(3k 바디 ≈ 300KB)의 200배 이상 여유가 있으면서,
/// 디코딩 증폭이 선형 메모리를 삼키지 못하게 한다.
pub(crate) const MAX_SNAPSHOT_BYTES: usize = 64 << 20;
/// **디코딩 후** 힙 예산. count × 원소 크기를 누적 과금한다.
pub(crate) const MAX_DECODED_BYTES: usize = 192 << 20;

/// 복원이 **파생물을 만드는 데 쓸 수 있는 작업량** 상한 (단위: 점 검사 1회).
///
/// 메모리만 묶으면 CPU가 열린다: hull 면 빌더는 `C(n,3)·n·2`라 32점 hull 하나가 약 31만 회
/// 점 검사다. 메모리 장부(hull당 4.7 KiB)로는 4만 개가 통과하는데, 그러면 한 번의 복원이
/// 120억 회를 단일 스레드로 돌아 브라우저·서버가 장시간 멈춘다(검수 rev_ae6d97b1 BLOCKER).
///
/// 값은 **시간으로** 잡았다: 점 검사 하나가 대략 1 ns이므로 2^27 ≈ 1.34억 회 ≈ **134 ms**가
/// 최악이다. 1초까지 허용하면(2^30) 복원 한 번이 프레임을 통째로 삼킨다 — 브라우저에서
/// 50 ms 넘는 정지는 이미 체감되는 끊김이라, 한 번의 사용자 동작에 100 ms대가 상한이다.
///
/// 이 상한이 실제로 무엇을 막고 무엇을 통과시키는가(점 검사 = `C(n,3)·n·2`):
/// - 32점 hull: 317,440회 → **423개**까지
/// - 12점 hull: 3,168회 → 42,000개까지
/// - 박스형 8점 hull: 896회 → 149,000개까지
///
/// 즉 상한에 걸리는 것은 "복잡한 볼록 조각 수백 개"뿐이고, 일반적인 레벨 지오메트리는
/// 여유롭게 통과한다.
pub(crate) const MAX_DECODE_WORK: usize = 1 << 27;

// 크기 상한은 `musoe_collide::limits`가 정본이다. `f32::MAX`처럼 **유한하지만 거대한** 값은
// 유한성 검사를 통과한 뒤 첫 곱에서 Inf가 되므로 `is_finite()`만으로는 방어가 되지 않는다.

/// 레코드 **최소** 직렬화 크기 — count 검증의 근거. 실제 레코드는 이보다 크거나 같다.
pub(crate) mod min_bytes {
    /// 셰이프 최소: 태그(1) + Sphere 반지름(4)
    pub(crate) const SHAPE: usize = 5;
    /// 바디 본문: 타입1 + pos12 + rot16 + lin12 + ang12 + invM4 + invI12 + fric4 + rest4
    /// + offset12 + force12 + torque12 + membership4 + filter4 + sensor1 + ccd1(v11)
    /// + weld1(v12) + lock1(v17) + cfthresh4(v18) + damp8
    pub(crate) const BODY_CORE: usize = 1
        + 12
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
        + SHAPE;
    /// 슬롯 레코드: alive1 + oob1 + reserved1 + gen4 + 본문 + sleep4 + asleep1 + island4
    pub const SLOT: usize = 1 + 1 + 1 + 4 + BODY_CORE + 4 + 1 + 4;
    /// 조인트: 핸들 8+8 + 앵커 24 + kind태그1 + collide1 + lin12 + ang12 + limit4 + motor4
    pub const JOINT: usize = 8 + 8 + 24 + 1 + 1 + 12 + 12 + 4 + 4;
    /// 커맨드: 태그1 + 최소 페이로드(RemoveBody = 핸들8)
    pub const COMMAND: usize = 1 + 8;
    /// 캐시 페어: 키8 + 엔트리 수4
    pub const CACHE_PAIR: usize = 8 + 4;
    /// 캐시 엔트리: fid4 + normal4 + tangent8 + tangent_world12 + position12 + 법선12
    pub const CACHE_ENTRY: usize = 4 + 4 + 8 + 12 + 12 + 12;
    /// hull 점 12 + (뒤따르는 반지름 4는 개수와 무관하므로 제외)
    pub const HULL_POINT: usize = 12;
    /// 하이트필드 샘플: f32 하나.
    pub const HEIGHT: usize = 4;
    /// 메시 정점: Vec3 12바이트.
    pub const MESH_VERT: usize = 12;
    /// 메시 삼각형: u32 인덱스 3개.
    pub const MESH_TRI: usize = 12;
}

/// 디코딩 후 원소가 힙에서 차지하는 크기 — count 과금용.
pub(crate) mod decoded_size {
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
        core::mem::size_of::<(musoe_collide::FeatureId, crate::solver::CachedImpulse)>()
    }
    /// 디코딩된 높이 한 칸(f32).
    pub fn height() -> usize {
        core::mem::size_of::<f32>()
    }
    /// 디코딩된 메시 정점/삼각형.
    pub fn mesh_vert() -> usize {
        core::mem::size_of::<musoe_math::Vec3>()
    }
    pub fn mesh_tri() -> usize {
        core::mem::size_of::<[u32; 3]>()
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
    /// **FP 환경이 캐노니컬 상태가 아니다** — 파일은 멀쩡하다.
    ///
    /// 복원도 산술로 상태를 만들므로(정규화·검증) 비표준 환경에서 확정되면 안 된다.
    /// 이걸 `InconsistentState`로 보고하면 호스트가 **멀쩡한 저장본을 손상 파일로 폐기**한다
    /// (확인검수 rev_7efe4e35 MAJOR). 원인을 따로 두어 "환경을 되돌리고 재시도"를 알린다.
    FpEnvironment,
}

pub struct Writer(pub Vec<u8>);

impl Writer {
    pub(crate) fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub(crate) fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub(crate) fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub(crate) fn vec3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    pub(crate) fn quat(&mut self, q: Quat) {
        self.f32(q.x);
        self.f32(q.y);
        self.f32(q.z);
        self.f32(q.w);
    }
    pub(crate) fn handle(&mut self, h: BodyHandle) {
        self.u32(h.index);
        self.u32(h.generation);
    }
}

pub struct Reader<'a> {
    b: &'a [u8],
    at: usize,
    /// 지금까지 과금된 디코딩 후 힙 바이트.
    decoded: usize,
    /// 지금까지 과금된 **파생물 생성 작업량**(점 검사 1회 단위) — 메모리와 별개 장부다.
    work: usize,
}

impl<'a> Reader<'a> {
    /// 바이트 슬라이스로 리더를 연다.
    ///
    /// 필드를 직접 구성하지 못하게 막는다 — `at`·`decoded`는 경계 검사와 디코딩 예산의
    /// 상태라 밖에서 임의로 세팅되면 그 계약이 무너진다(교차검수 rev_819c41cd 지적).
    /// 남은 바이트가 없는가 — "뒤에 쓰레기가 붙어 있지 않다"를 확인하는 자리다.
    pub(crate) fn finished(&self) -> bool {
        self.at == self.b.len()
    }

    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            b: bytes,
            at: 0,
            decoded: 0,
            work: 0,
        }
    }
}

impl Reader<'_> {
    #[inline]
    pub(crate) fn take(&mut self, n: usize) -> Result<&[u8], SnapshotError> {
        let end = self.at.checked_add(n).ok_or(SnapshotError::Truncated)?;
        if end > self.b.len() {
            return Err(SnapshotError::Truncated);
        }
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.b.len() - self.at
    }
    /// 호스트 입력에서 온 물리량 (앵커·마찰 등) — **입력 정책**.
    pub(crate) fn f32(&mut self) -> Result<f32, SnapshotError> {
        let v = self.raw_f32()?;
        if musoe_collide::in_range(v) {
            Ok(v)
        } else {
            Err(SnapshotError::NonFinite)
        }
    }
    /// 엔진이 진화시키는 물리량 (위치·속도·누적 임펄스·PCM 캐시) — **상태 정책**.
    /// 솔버는 큰 질량과 큰 접근 속도의 조합에서 입력 상한을 넘는 임펄스를 정당하게 만든다.
    pub(crate) fn state_f32(&mut self) -> Result<f32, SnapshotError> {
        let v = self.raw_f32()?;
        if musoe_collide::in_state_range(v) {
            Ok(v)
        } else {
            Err(SnapshotError::NonFinite)
        }
    }
    /// **임계값 정책** — 유한 음이 아닌 값 또는 `+∞`("절대 발화 안 함")만 받는다.
    ///
    /// 다른 f32는 비유한을 전부 막는다(물리 상태에 들어가면 월드가 오염된다). 임계는
    /// 물리 상태가 아니라 **비교 상수**라서 `+∞`가 정상 값이다 — "꺼짐"을 그렇게 쓴다.
    /// NaN과 −∞는 여전히 거절한다: NaN은 비교가 항상 거짓이라 **조용히 꺼지고**,
    /// −∞는 모든 접촉을 이벤트로 만들어 버퍼를 넘긴다.
    pub(crate) fn threshold_f32(&mut self) -> Result<f32, SnapshotError> {
        let v = self.raw_f32()?;
        if v >= 0.0 {
            Ok(v) // NaN은 이 비교에서 거짓이라 자동으로 걸린다
        } else {
            Err(SnapshotError::NonFinite)
        }
    }
    pub(crate) fn raw_f32(&mut self) -> Result<f32, SnapshotError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    pub(crate) fn state_vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(
            self.state_f32()?,
            self.state_f32()?,
            self.state_f32()?,
        ))
    }
    pub(crate) fn quat(&mut self) -> Result<Quat, SnapshotError> {
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
        // **`normalize`가 아니라 `try_canonicalize`다.** f32 정규화는 비트 멱등이 아니라
        // (주기 2 진동), 쓰기 쪽이 이미 정규화한 값을 리더가 한 번 더 정규화하면
        // 왕복만으로 상태 해시가 갈린다 — 바디 자세와 조인트 `rel_rot` 양쪽에 걸린다.
        q.try_canonicalize().ok_or(SnapshotError::NonFinite)
    }
    pub(crate) fn handle(&mut self) -> Result<BodyHandle, SnapshotError> {
        Ok(BodyHandle {
            index: self.u32()?,
            generation: self.u32()?,
        })
    }
    /// **입력 길이에서 유도한 상한**으로 개수를 검증한다(고정 상수 금지 — 리뷰 BLOCKER).
    /// n개를 담으려면 최소 `n * min_each` 바이트가 남아 있어야 한다.
    /// **파생물 힙을 명시적으로 과금한다.**
    ///
    /// `count`는 "원소 수 × 원소 크기"만 센다. 복원이 원소에서 **파생물**을 만들면
    /// (hull의 면 구조·메시의 BVH 같은 것) 그 힙은 장부 밖이라, 장부상 한도 안인 스냅샷이
    /// 실제로는 수백 MiB를 잡고 allocator abort로 끝날 수 있다(검수 rev_8eacf4a5 BLOCKER).
    /// 파생물을 만드는 자리는 만들기 **전에** 최악 크기를 여기로 신고한다.
    pub(crate) fn charge_decoded(&mut self, bytes: usize) -> Result<(), SnapshotError> {
        self.decoded = self
            .decoded
            .checked_add(bytes)
            .ok_or(SnapshotError::CountTooLarge)?;
        if self.decoded > MAX_DECODED_BYTES {
            return Err(SnapshotError::CountTooLarge);
        }
        Ok(())
    }

    /// 복원이 파생물을 만드는 **작업량**을 과금한다(단위: 점 검사 1회).
    ///
    /// 메모리 장부와 별개다 — 같은 바이트로도 CPU는 몇 자릿수 차이가 난다.
    pub(crate) fn charge_work(&mut self, units: usize) -> Result<(), SnapshotError> {
        self.work = self
            .work
            .checked_add(units)
            .ok_or(SnapshotError::CountTooLarge)?;
        if self.work > MAX_DECODE_WORK {
            return Err(SnapshotError::CountTooLarge);
        }
        Ok(())
    }

    pub(crate) fn count(
        &mut self,
        min_each: usize,
        decoded_each: usize,
    ) -> Result<usize, SnapshotError> {
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

/// 할당 실패를 오류로 돌려주는 Vec 예약.
pub(crate) fn try_vec<T>(n: usize) -> Result<Vec<T>, SnapshotError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| SnapshotError::CountTooLarge)?;
    Ok(v)
}
