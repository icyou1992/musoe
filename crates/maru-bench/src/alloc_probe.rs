//! 전역 할당자 계측 — 스텝당 힙 할당 횟수·바이트·크기 분포를 센다.
//!
//! 설계 §6은 "스텝 중 힙 할당 0"을 계약으로 둔다(wasm에서 할당 실패는 오류 ABI를 깨므로,
//! 실패할 수 있는 예약은 전부 스텝 시작의 fallible 블록에 모여야 한다). 계약은 **측정
//! 가능해야 계약**이라서 계측기가 저장소에 산다.
//!
//! ## 쓰는 법
//!
//! 계측을 원하는 벤치 바이너리가 전역 할당자를 이 타입으로 바꾼다:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: maru_bench::alloc_probe::CountingAlloc = maru_bench::alloc_probe::CountingAlloc::new();
//! ```
//!
//! 그 다음 측정 창을 `reset()`으로 열고 `snapshot()`으로 닫는다. **측정 창 안에서는
//! 출력·포맷팅을 하지 않는다** — `format!`·`println!`이 자기 할당을 계측값에 섞는다.
//!
//! ## 왜 다른 벤치는 이 할당자를 쓰지 않는가
//!
//! 계수기는 할당마다 원자적 증가 4~5회를 붙인다. 시간을 재는 벤치에 그것을 넣으면
//! **측정 도구가 측정 대상을 바꾼다**. 그래서 할당 계측은 전용 바이너리(`--bin allocs`)에만
//! 설치하고, 시간 벤치(`pyramid100`·`boxsplit` …)는 표준 할당자로 돈다.
//!
//! ## 스레드
//!
//! 계수기는 `Relaxed`로 올린다. 워커가 올린 값이 스냅샷에 보이는 보장은 원자성이 아니라
//! **`World::step`이 돌아오기 전에 배리어에서 만난다**는 사실에서 온다(배리어의 release/acquire가
//! 그 전의 Relaxed 쓰기를 함께 게시한다). 그래서 스냅샷은 항상 `step` **바깥**에서 찍는다.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 크기 분포 버킷 수. 버킷은 요청 크기의 **2의 거듭제곱 천장**으로 정한다 —
/// 버킷 `i`(≥1)는 `(2^(i-2), 2^(i-1)]` 바이트이고 라벨은 그 상한이다(0번은 0바이트 요청).
/// 마지막 버킷은 그 이상 전부를 담는다(2 MiB↑).
pub const BUCKETS: usize = 22;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static DEALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES_OUT: AtomicU64 = AtomicU64::new(0);
static BYTES_IN: AtomicU64 = AtomicU64::new(0);
static HIST: [AtomicU64; BUCKETS] = [const { AtomicU64::new(0) }; BUCKETS];

/// 요청 크기 → 버킷 번호. 0바이트 요청은 0번(실제로 발생한다 — 빈 `Vec`의 `try_reserve(0)` 등).
#[inline]
fn bucket_of(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    // 1..=1 → 1, 2..=3 → 2, 4..=7 → 3 … (= 상한의 log2 + 1)
    let b = usize::BITS as usize - (size - 1).leading_zeros() as usize + 1;
    b.min(BUCKETS - 1)
}

/// 버킷의 상한 크기(표시용). 마지막 버킷은 "그 이상"이다.
pub fn bucket_upper_bytes(b: usize) -> u64 {
    if b == 0 {
        0
    } else {
        1u64 << (b - 1)
    }
}

/// 추적할 버킷 + 1 (0 = 추적 off). 크기 분포가 범인을 **지목**은 하지만 **증명**은 못 하기
/// 때문에 있다 — "256B 버킷이 페어 수와 비슷하다"는 정황이고, 호출 스택은 근거다.
static TRACE_BUCKET: AtomicUsize = AtomicUsize::new(0);
/// 남은 출력 예산 — 스택 하나가 수십 줄이라 상한이 없으면 로그가 무의미해진다.
static TRACE_LEFT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// 추적 자체가 할당한다(백트레이스 심볼화·`eprintln`). 재진입하면 무한 재귀다.
    static IN_TRACE: Cell<bool> = const { Cell::new(false) };
}

/// 추적 모드를 켠다 — **측정 창을 열기 전에** 부른다.
///
/// 환경변수를 할당자 안에서 읽지 않는 이유: `env::var`가 할당하므로 지연 초기화(`OnceLock`)를
/// 쓰면 초기화 중에 자기 자신을 다시 부르게 된다. 설정은 항상 밖에서 밀어 넣는다.
pub fn configure_trace(bucket: Option<usize>, max_prints: usize) {
    TRACE_BUCKET.store(bucket.map_or(0, |b| b + 1), Ordering::Relaxed);
    TRACE_LEFT.store(
        if bucket.is_some() { max_prints } else { 0 },
        Ordering::Relaxed,
    );
}

#[cold]
fn trace_one(size: usize) {
    if IN_TRACE.get() {
        return;
    }
    // 예산을 먼저 깎는다 — 실패하면 아무것도 하지 않는다(경합에서도 상한을 지킨다).
    let mut left = TRACE_LEFT.load(Ordering::Relaxed);
    loop {
        if left == 0 {
            return;
        }
        match TRACE_LEFT.compare_exchange_weak(left, left - 1, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(v) => left = v,
        }
    }
    IN_TRACE.set(true);
    let bt = std::backtrace::Backtrace::force_capture();
    eprintln!("--- alloc trace: {size} B ---\n{bt}");
    IN_TRACE.set(false);
}

#[inline]
fn record_get(size: usize) {
    BYTES_OUT.fetch_add(size as u64, Ordering::Relaxed);
    let b = bucket_of(size);
    HIST[b].fetch_add(1, Ordering::Relaxed);
    if TRACE_BUCKET.load(Ordering::Relaxed) == b + 1 {
        trace_one(size);
    }
}

/// `System`을 그대로 쓰면서 호출 횟수·바이트를 세는 전역 할당자.
///
/// 계수만 하고 할당 정책은 바꾸지 않는다 — 계측기가 대상의 동작을 바꾸면 측정이 아니다.
pub struct CountingAlloc;

impl CountingAlloc {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for CountingAlloc {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: 모든 요청을 `System`에 그대로 위임한다 — 반환 포인터·레이아웃 계약은 `System`의
// 것이고, 계측은 원자 계수기 증가뿐이라 할당 자체를 하지 않는다(계측기가 재진입하면
// 무한 재귀가 된다 — 그래서 여기서는 `format!`·`Vec` 같은 것을 절대 쓰지 않는다).
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        record_get(layout.size());
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        record_get(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES_IN.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // realloc은 **별도로 센다**. alloc으로 합쳐 세면 `Vec::push`의 증폭 성장이
        // 새 할당처럼 보이고, dealloc으로 합쳐도 짝이 안 맞아 보유량 회계가 깨진다.
        REALLOCS.fetch_add(1, Ordering::Relaxed);
        record_get(new_size);
        BYTES_IN.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

/// 한 측정 창의 누적값.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// `alloc`/`alloc_zeroed` 호출 수.
    pub allocs: u64,
    /// `realloc` 호출 수.
    pub reallocs: u64,
    /// `dealloc` 호출 수.
    pub deallocs: u64,
    /// 새로 얻은 바이트 합(alloc + realloc의 새 크기).
    pub bytes_out: u64,
    /// 돌려준 바이트 합(dealloc + realloc의 옛 크기).
    pub bytes_in: u64,
    /// 크기 분포 — 인덱스는 [`bucket_of`]의 버킷.
    pub hist: [u64; BUCKETS],
}

impl AllocStats {
    /// 메모리를 얻는 할당자 호출 수 = `alloc + realloc`. **"스텝당 할당"으로 보고하는 값**이다.
    pub fn gets(&self) -> u64 {
        self.allocs + self.reallocs
    }

    /// 창 동안 순증한 바이트. 스텝이 정상 상태라면 0에 가까워야 한다 —
    /// 계속 양수면 스텝마다 뭔가를 쌓고 있다는 뜻이다(누수든 성장이든).
    pub fn net_bytes(&self) -> i64 {
        self.bytes_out as i64 - self.bytes_in as i64
    }
}

/// 계수기를 0으로 돌린다 — 측정 창을 여는 지점.
pub fn reset() {
    ALLOCS.store(0, Ordering::Relaxed);
    REALLOCS.store(0, Ordering::Relaxed);
    DEALLOCS.store(0, Ordering::Relaxed);
    BYTES_OUT.store(0, Ordering::Relaxed);
    BYTES_IN.store(0, Ordering::Relaxed);
    for h in &HIST {
        h.store(0, Ordering::Relaxed);
    }
}

/// 현재 누적값을 읽는다 — 측정 창을 닫는 지점. `step` **바깥**에서 부른다(모듈 주석의 스레드 절).
pub fn snapshot() -> AllocStats {
    let mut hist = [0u64; BUCKETS];
    for (dst, src) in hist.iter_mut().zip(HIST.iter()) {
        *dst = src.load(Ordering::Relaxed);
    }
    AllocStats {
        allocs: ALLOCS.load(Ordering::Relaxed),
        reallocs: REALLOCS.load(Ordering::Relaxed),
        deallocs: DEALLOCS.load(Ordering::Relaxed),
        bytes_out: BYTES_OUT.load(Ordering::Relaxed),
        bytes_in: BYTES_IN.load(Ordering::Relaxed),
        hist,
    }
}

/// 크기 분포를 사람이 읽는 한 줄로 — 건수 많은 순 상위 `top`개.
/// **측정 창 밖에서만 부른다**(자기 할당이 섞이지 않게).
pub fn hist_line(stats: &AllocStats, steps: f64, top: usize) -> String {
    let mut idx: Vec<usize> = (0..BUCKETS).filter(|&i| stats.hist[i] > 0).collect();
    idx.sort_by_key(|&i| std::cmp::Reverse(stats.hist[i]));
    let mut s = String::new();
    for &i in idx.iter().take(top) {
        let per = stats.hist[i] as f64 / steps;
        let label = if i == BUCKETS - 1 {
            format!(">{}KiB", bucket_upper_bytes(i - 1) / 1024)
        } else if bucket_upper_bytes(i) >= 1024 {
            format!("≤{}KiB", bucket_upper_bytes(i) / 1024)
        } else {
            format!("≤{}B", bucket_upper_bytes(i))
        };
        if !s.is_empty() {
            s.push_str("  ");
        }
        s.push_str(&format!("{label}:{per:.0}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_powers_of_two_upper_bounds() {
        // 버킷은 요청 크기의 **2의 거듭제곱 천장**이다: `upper = 2^ceil(log2(size))`.
        // 그래서 3B는 "≤2B"가 아니라 "≤4B"에 들어간다(첫 작성에서 여기를 틀렸고 wasm 게이트가
        // 잡았다 — 라벨과 버킷이 어긋나면 히스토그램이 범인을 잘못 지목한다).
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(1), 1);
        assert_eq!(bucket_of(2), 2);
        assert_eq!(bucket_of(3), 3);
        assert_eq!(bucket_of(4), 3);
        assert_eq!(bucket_of(32), 6);
        assert_eq!(bucket_of(33), 7);
        assert_eq!(bucket_of(256), 9);
        // 라벨(상한)과 실제 크기의 관계 — 경계값이 자기 상한에 붙는다.
        assert_eq!(bucket_upper_bytes(bucket_of(256)), 256);
        assert_eq!(bucket_upper_bytes(bucket_of(255)), 256);
        assert_eq!(bucket_upper_bytes(bucket_of(257)), 512);
        assert_eq!(bucket_upper_bytes(bucket_of(32)), 32);
        assert_eq!(bucket_upper_bytes(bucket_of(3)), 4);
        // 상한을 넘는 요청은 마지막 버킷에 모인다(포화 — 인덱스 밖으로 나가지 않는다).
        // 시프트 폭은 `usize::BITS`에서 유도한다 — wasm32는 usize가 32비트라
        // 상수 `1 << 40`은 **컴파일 오류**다(wasm 게이트가 잡았다).
        assert_eq!(bucket_of(usize::MAX), BUCKETS - 1);
        assert_eq!(bucket_of(1usize << (usize::BITS - 4)), BUCKETS - 1);
        // 버킷 번호는 포인터 폭과 무관하다 — 두 항이 함께 스케일한다.
        assert_eq!(bucket_upper_bytes(bucket_of(1024)), 1024);
    }

    #[test]
    fn gets_counts_alloc_and_realloc() {
        let s = AllocStats {
            allocs: 3,
            reallocs: 2,
            deallocs: 4,
            bytes_out: 100,
            bytes_in: 40,
            ..Default::default()
        };
        assert_eq!(s.gets(), 5);
        assert_eq!(s.net_bytes(), 60);
    }
}
