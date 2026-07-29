//! 워커 풀 (설계 §5 — "1스레드가 1급, 워커는 승수").
//!
//! 이 모듈은 **물리를 모른다**. 스텝 안에서 "같은 잡을 N개 참여자가 나눠 돌고, 중간에
//! 배리어로 만난다"는 구조만 제공한다. 결정론의 근거는 이 모듈이 아니라 호출자에 있다:
//! 참여자들이 쓰는 메모리가 서로 겹치지 않으면(솔버는 그래프 컬러링, 내로우페이즈는
//! 워커-로컬 출력 버퍼) 완료 순서가 결과를 바꾸지 못한다. 이 모듈은 그 전제 아래
//! **작업 분배와 동기화만** 책임진다.
//!
//! 설계상의 선택 세 가지:
//!
//! 1. **메인 스레드도 참여자 0번이다.** 워커 N-1개를 띄우고 메인이 놀면 코어 하나를
//!    버리는 셈이고, 배리어 참여자 수도 실행 주체 수와 어긋난다.
//! 2. **패스당 dispatch 1회, 색 경계는 잡 내부 스핀 배리어.** 색마다 OS에 일을 맡기면
//!    (서브스텝 4 × 패스 5 × 색 10~20 = 스텝당 수백 회) 동기화가 이득을 먹는다.
//!    dispatch는 mutex+condvar(잠든 워커 깨우기)라 ~us 단위지만, 배리어는 원자 연산
//!    스핀이라 ~100ns 단위다.
//! 3. **스핀 후 park.** 스텝 안의 dispatch 간격(수십 us)은 스핀으로 넘기고, 스텝 사이
//!    간격(수 ms)에는 잠들어 코어를 놓아준다. 게임 엔진이 매 프레임 코어 4개를
//!    100%로 태우면 렌더러·오디오가 굶는다.
//!
//! wasm 타깃에는 스레드가 없다(`wasm32-unknown-unknown`·`wasip1` 모두 `spawn` 불가) —
//! 컴파일 타임에 참여자 1명짜리 직렬 경로로 강등한다. 설계 §5의 "단일스레드 완전 동작"이
//! 폴백이 아니라 1급 경로라는 계약이 여기서 지켜진다.

/// 설계·스크래치 회계가 지원하는 **최대 참여자 수**(메인 포함).
///
/// 설계 §5가 "4–8워커가 최적점"(Jolt 실측: 네이티브도 16코어+에서 정체·역행)이라 상한이
/// 있고, 그보다 중요한 이유는 **가용 코어를 넘기면 스핀 설계가 무너진다**는 것이다
/// (4코어에 5참여자 실측: dispatch 5ms·배리어 4ms — 정상의 수백 배). 호스트가
/// `set_threads`로 더 큰 값을 줘도 여기서 자른다. 자른 사실은 `World::threads()`로
/// 관측된다(설계 §6.3 조용한 폴백 금지 — 값을 숨기지 않고 보고한다).
pub const MAX_THREADS: usize = 8;

/// 잡 안에서 참여자 하나가 보는 컨텍스트.
pub struct Ctx {
    index: usize,
    threads: usize,
    #[cfg(not(target_arch = "wasm32"))]
    inner: *const imp::Inner,
    /// 배리어 sense는 **참여자별 로컬 상태**다 (sense-reversing barrier).
    #[cfg(not(target_arch = "wasm32"))]
    sense: core::cell::Cell<bool>,
}

impl Ctx {
    /// 참여자 번호 (0 = 메인 스레드).
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// 총 참여자 수 (메인 포함). 작업 분배의 분모.
    #[inline]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// 모든 참여자가 도달할 때까지 대기한다.
    ///
    /// **전 참여자가 같은 횟수만큼 불러야 한다** — 한 참여자가 조기 반환하면 나머지가
    /// 영원히 기다린다. 그래서 잡 본문은 "색마다 내 몫을 돌고 배리어"처럼
    /// 참여자와 무관하게 같은 제어 흐름을 타야 한다(자기 몫이 0개여도 배리어는 친다).
    #[inline]
    pub fn barrier(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.threads > 1 {
                // SAFETY: `inner`는 dispatch가 잡을 실행하는 동안만 살아있는 Arc를 가리키고,
                // dispatch는 전 참여자가 끝나기 전에 반환하지 않는다.
                unsafe { &*self.inner }.barrier(&self.sense);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread::JoinHandle;

    /// 스핀 한도 — 이 횟수를 넘기면 condvar에 눕는다(≈ 5-10us 분량).
    ///
    /// **짧게 잡은 것이 측정 결과다.** 길게(2백만 ≈ 수 ms) 잡으면 워커가 직렬 구간
    /// 내내 스핀하며 코어를 물고 있어, 스레드를 켰을 뿐인데 **직렬 단계가 6배 느려졌다**
    /// (피라미드 40단 refresh 320 → 1894us). 코어가 넉넉하지 않은 기계(또는 다른
    /// 프로세스가 도는 기계)에서 스핀은 이득이 아니라 절도다. 지금 기본 구성은
    /// 스텝당 dispatch가 1회(내로우페이즈)뿐이라 깨우는 값도 상각된다 —
    /// 솔버 색 병렬(`World::set_parallel_solver`)을 켜면 dispatch가 스텝당 20회가 되고,
    /// 그때는 이 값을 늘리는 편이 유리할 수 있다(계측 후에).
    const SPIN_LIMIT: u32 = 4_000;

    /// 타입을 지운 잡 참조. dispatch가 살아있는 동안만 유효하다.
    #[derive(Clone, Copy)]
    struct JobRef {
        data: *const (),
        call: unsafe fn(*const (), &super::Ctx),
    }

    /// SAFETY: `JobRef`가 가리키는 클로저는 `Sync` 바운드를 만족하고(dispatch 시그니처),
    /// dispatch가 전 참여자의 완료를 기다린 뒤에야 반환하므로 수명도 유효하다.
    unsafe impl Send for JobRef {}
    unsafe impl Sync for JobRef {}

    pub(super) struct Inner {
        /// dispatch 세대. 증가 = "새 잡이 있다". Release/Acquire로 잡 슬롯 가시성을 싣는다.
        epoch: AtomicU64,
        /// 아직 안 끝난 워커 수(메인 제외). 0이 되면 dispatch가 반환할 수 있다.
        running: AtomicUsize,
        /// 잡 슬롯 — running == 0인 동안에만 메인이 쓴다.
        job: std::cell::UnsafeCell<Option<JobRef>>,
        /// 워커 종료 신호.
        quit: AtomicBool,
        /// 잡 실행 중 패닉이 났다. 배리어 대기가 이걸 보고 빠져나온다(교착 방지).
        poisoned: AtomicBool,
        /// park용.
        park_lock: Mutex<()>,
        park_cv: Condvar,
        /// sense-reversing 배리어.
        bar_count: AtomicUsize,
        bar_sense: AtomicBool,
        threads: usize,
    }

    /// SAFETY: 잡 슬롯(`UnsafeCell`)은 epoch의 Release/Acquire로 순서가 잡힌다 —
    /// 메인은 running==0(직전 dispatch 완료)일 때만 쓰고, 워커는 epoch을 Acquire로
    /// 읽은 뒤에만 읽는다.
    unsafe impl Sync for Inner {}
    unsafe impl Send for Inner {}

    impl Inner {
        pub(super) fn barrier(&self, sense: &std::cell::Cell<bool>) {
            let target = !sense.get();
            sense.set(target);
            if self.bar_count.fetch_add(1, Ordering::AcqRel) == self.threads - 1 {
                // 마지막 도착자가 문을 연다.
                self.bar_count.store(0, Ordering::Relaxed);
                self.bar_sense.store(target, Ordering::Release);
            } else {
                let mut spins = 0u32;
                while self.bar_sense.load(Ordering::Acquire) != target {
                    if self.poisoned.load(Ordering::Acquire) {
                        // 다른 참여자가 패닉했다 — 여기서 계속 기다리면 영구 교착이다.
                        panic!("maru: 워커 패닉으로 배리어가 붕괴했다");
                    }
                    spins += 1;
                    if spins < SPIN_LIMIT {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }

    /// 실제로 워커가 떠 있는 풀. `Pool`은 이것을 **첫 dispatch 때** 만든다.
    struct Spawned {
        inner: Arc<Inner>,
        handles: Vec<JoinHandle<()>>,
    }

    /// 워커 풀. **스레드는 처음 병렬로 돌 때 뜬다(지연 생성).**
    ///
    /// 월드를 만들 때마다 OS 스레드를 띄우면, 작은 월드를 수백 개 만드는 호출자
    /// (테스트 스위트가 그렇다)가 스레드 생성·조인 비용만 내고 병렬은 한 번도 안 쓴다 —
    /// 실측으로 테스트 스위트가 6.7초에서 37초가 됐다. 병렬 문턱을 넘는 월드만 값을 낸다.
    pub struct Pool {
        want: usize,
        imp: std::sync::OnceLock<Spawned>,
    }

    impl Pool {
        /// 참여자 `threads`명(메인 포함)짜리 풀. 스레드를 못 띄우면 **띄운 만큼으로 줄여
        /// 다시 만든다** — `Inner.threads`가 실제 실행 주체 수와 다르면 배리어가
        /// 영원히 안 열리기 때문이다. 강등 사실은 `threads()`로 관측된다
        /// (설계 §6.3 "조용한 폴백 금지").
        pub fn new(threads: usize) -> Self {
            Pool {
                want: threads.clamp(1, super::MAX_THREADS),
                imp: std::sync::OnceLock::new(),
            }
        }

        /// 워커를 띄운다(첫 dispatch에서 1회). 못 띄우면 **띄운 만큼으로 줄여 다시**
        /// 만든다 — `Inner.threads`가 실제 실행 주체 수와 다르면 배리어가 영원히 안 열린다.
        fn spawn_pool(threads: usize) -> Spawned {
            let mut want = threads.max(1);
            loop {
                match Self::try_spawn(want) {
                    Ok(p) => return p,
                    Err(spawned) => {
                        // 다음 시도는 이번에 실제로 띄운 수 이하로 (단조 감소 보장).
                        want = (spawned + 1).min(want - 1).max(1);
                    }
                }
            }
        }

        /// 정확히 `threads`명으로 시도한다. 실패하면 그때까지 띄운 워커 수를 돌려준다.
        fn try_spawn(threads: usize) -> Result<Spawned, usize> {
            let inner = Arc::new(Inner {
                epoch: AtomicU64::new(0),
                running: AtomicUsize::new(0),
                job: std::cell::UnsafeCell::new(None),
                quit: AtomicBool::new(false),
                poisoned: AtomicBool::new(false),
                park_lock: Mutex::new(()),
                park_cv: Condvar::new(),
                bar_count: AtomicUsize::new(0),
                bar_sense: AtomicBool::new(false),
                threads,
            });
            let mut handles = Vec::new();
            for index in 1..threads {
                let w = Arc::clone(&inner);
                match std::thread::Builder::new()
                    .name(format!("maru-worker-{index}"))
                    .spawn(move || worker_loop(w, index))
                {
                    Ok(h) => handles.push(h),
                    Err(_) => {
                        let spawned = handles.len();
                        // 이미 띄운 워커는 정리한다(잡을 준 적이 없어 quit만으로 끝난다).
                        {
                            let _g = inner.park_lock.lock().unwrap_or_else(|e| e.into_inner());
                            inner.quit.store(true, Ordering::Release);
                        }
                        inner.park_cv.notify_all();
                        for h in handles {
                            let _ = h.join();
                        }
                        return Err(spawned);
                    }
                }
            }
            Ok(Spawned { inner, handles })
        }

        /// 참여자 수. 워커가 아직 안 떴으면 **요청한 수**를, 떴으면 **실제 수**를 돌려준다
        /// (스레드를 못 띄운 환경에서는 첫 dispatch 뒤 값이 줄어들 수 있다 —
        /// 설계 §6.3 "조용한 폴백 금지"에 따라 진단은 dispatch **뒤** 값을 보고한다).
        #[inline]
        pub fn threads(&self) -> usize {
            match self.imp.get() {
                Some(s) => s.inner.threads,
                None => self.want,
            }
        }

        /// 잡을 전 참여자가 나눠 돌고, 전원이 끝난 뒤 반환한다.
        ///
        /// 참여자 0번은 **호출한 스레드 자신**이다. 반환 시점에 모든 워커가 클로저 사용을
        /// 끝냈으므로 클로저가 빌린 지역 데이터는 그 뒤에 자유롭게 쓸 수 있다
        /// (`std::thread::scope`와 같은 계약).
        pub fn dispatch<F: Fn(&super::Ctx) + Sync>(&self, f: F) {
            if self.want == 1 {
                // 요청이 1명이면 워커를 **아예 띄우지 않는다** — 설계 §5 "단일스레드 1급".
                // `inner`는 널이지만 `Ctx::barrier`가 `threads > 1`에서만 역참조하므로
                // 이 컨텍스트에서는 절대 읽히지 않는다(참여자 1명 배리어는 no-op이다).
                f(&super::Ctx {
                    index: 0,
                    threads: 1,
                    inner: std::ptr::null(),
                    sense: std::cell::Cell::new(false),
                });
                return;
            }
            let this = self.imp.get_or_init(|| Self::spawn_pool(self.want));
            if this.inner.threads == 1 {
                // 직렬 경로 — 원자 연산도 배리어도 없다. 설계 §5 "단일스레드 1급".
                f(&super::Ctx {
                    index: 0,
                    threads: 1,
                    inner: Arc::as_ptr(&this.inner),
                    sense: std::cell::Cell::new(false),
                });
                return;
            }
            let n_workers = this.inner.threads - 1;
            this.inner.poisoned.store(false, Ordering::Relaxed);
            // 잡 슬롯 쓰기 — 직전 dispatch가 running==0을 확인하고 반환했으므로 배타적이다.
            let job = JobRef {
                data: (&f as *const F).cast::<()>(),
                call: call_impl::<F>,
            };
            // SAFETY: 위 배타성 + 아래 epoch Release가 워커의 Acquire와 짝을 이룬다.
            unsafe { *this.inner.job.get() = Some(job) };
            // **배리어 상태를 dispatch마다 초기화한다.** 참여자 로컬 sense는 매번 false로
            // 시작하므로 첫 배리어의 목표는 항상 true다 — 직전 dispatch가 남긴 sense가
            // true인 채로 들어오면 첫 배리어가 "이미 열린 문"이 되어 통과해 버린다
            // (구간 분리가 조용히 사라진다). 워커가 하나도 안 도는 지금이 유일한 안전 시점.
            this.inner.bar_count.store(0, Ordering::Relaxed);
            this.inner.bar_sense.store(false, Ordering::Relaxed);
            this.inner.running.store(n_workers, Ordering::Relaxed);
            {
                // 잠든 워커의 wakeup 유실을 막으려면 epoch 증가가 park_lock 안이어야 한다.
                let _g = this
                    .inner
                    .park_lock
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                this.inner.epoch.fetch_add(1, Ordering::Release);
            }
            this.inner.park_cv.notify_all();

            // 메인도 참여자다.
            let ctx = super::Ctx {
                index: 0,
                threads: this.inner.threads,
                inner: Arc::as_ptr(&this.inner),
                sense: std::cell::Cell::new(false),
            };
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&ctx)));
            if res.is_err() {
                this.inner.poisoned.store(true, Ordering::Release);
            }

            // 전원 완료 대기 — 여기를 지나야 클로저 참조가 죽어도 안전하다.
            let mut spins = 0u32;
            while this.inner.running.load(Ordering::Acquire) != 0 {
                spins += 1;
                if spins < SPIN_LIMIT {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
            if let Err(p) = res {
                std::panic::resume_unwind(p);
            }
            if this.inner.poisoned.load(Ordering::Acquire) {
                panic!("maru: 워커 스레드가 잡 실행 중 패닉했다");
            }
        }
    }

    unsafe fn call_impl<F: Fn(&super::Ctx) + Sync>(data: *const (), ctx: &super::Ctx) {
        let f = unsafe { &*data.cast::<F>() };
        f(ctx)
    }

    impl Drop for Spawned {
        fn drop(&mut self) {
            {
                let _g = self
                    .inner
                    .park_lock
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                self.inner.quit.store(true, Ordering::Release);
            }
            self.inner.park_cv.notify_all();
            for h in self.handles.drain(..) {
                let _ = h.join();
            }
        }
    }

    fn worker_loop(inner: Arc<Inner>, index: usize) {
        // **FP 환경은 스레드마다 산다** (설계 §8) — 워커도 캐노니컬 상태로 맞춘다.
        // 안 맞추면 어느 워커가 어느 제약을 처리했는지에 따라 해시가 갈린다.
        let _ = maru_math::fp_canonicalize();
        let mut seen = 0u64;
        loop {
            // 새 잡(또는 종료)을 기다린다: 스핀 → park.
            let mut spins = 0u32;
            loop {
                if inner.quit.load(Ordering::Acquire) {
                    return;
                }
                let e = inner.epoch.load(Ordering::Acquire);
                if e != seen {
                    seen = e;
                    break;
                }
                spins += 1;
                if spins < SPIN_LIMIT {
                    std::hint::spin_loop();
                } else {
                    let g = inner.park_lock.lock().unwrap_or_else(|e| e.into_inner());
                    // 락 안에서 재확인 — dispatch도 같은 락 안에서 epoch을 올린다.
                    if inner.epoch.load(Ordering::Acquire) == seen
                        && !inner.quit.load(Ordering::Acquire)
                    {
                        let _unused = inner.park_cv.wait(g);
                    }
                    spins = 0;
                }
            }
            // SAFETY: epoch Acquire가 메인의 Release와 짝 — 슬롯은 이 시점에 확정이다.
            let job = unsafe { *inner.job.get() };
            if let Some(job) = job {
                let ctx = super::Ctx {
                    index,
                    threads: inner.threads,
                    inner: Arc::as_ptr(&inner),
                    sense: std::cell::Cell::new(false),
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    (job.call)(job.data, &ctx)
                }));
                if r.is_err() {
                    // 배리어에 걸린 다른 참여자를 풀어 준다(교착 대신 패닉).
                    inner.poisoned.store(true, Ordering::Release);
                }
            }
            inner.running.fetch_sub(1, Ordering::Release);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    /// wasm 직렬 풀 — 참여자 1명, 동기화 없음.
    pub struct Pool;

    impl Pool {
        pub fn new(_threads: usize) -> Self {
            Pool
        }
        #[inline]
        pub fn threads(&self) -> usize {
            1
        }
        #[inline]
        pub fn dispatch<F: Fn(&super::Ctx) + Sync>(&self, f: F) {
            f(&super::Ctx {
                index: 0,
                threads: 1,
            });
        }
    }
}

pub use imp::Pool;

/// 참여자들이 **서로 다른 인덱스만** 만진다는 전제로 공유되는 가변 슬라이스.
///
/// 이 타입이 병렬 경로의 **유일한 공유 쓰기 표면**이고, 안전성 부담이 전부 여기 모인다:
/// 두 참여자가 같은 인덱스를 만지면 그 즉시 데이터 레이스이자 비결정성이다. 겹치지
/// 않는다는 것은 호출자가 **구조로** 보장해야 하고, 이 저장소에서 그 구조는 두 가지다.
///
/// - **솔버**: 그래프 컬러링. 같은 색의 두 제약은 dynamic 바디를 공유하지 않는다
///   (`solver::color_constraints`). 그리고 솔브가 write하는 바디 집합
///   (`SolverState::dynamic`)은 컬러링이 conflict로 본 집합과 **같은 술어**다
///   (world.rs `conflict` ↔ solver.rs `fill_from`).
/// - **내로우페이즈**: 참여자마다 자기 출력 버퍼 하나(인덱스 = 참여자 번호).
///
/// 겹치는 `&mut`를 절대 만들지 않는다 — 원소(또는 서로 겹치지 않는 구간)마다 따로
/// 빌린다. 전체 슬라이스를 두 참여자가 `&mut`로 잡는 흔한 편법은 쓰지 않았다.
pub struct DisjointSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: core::marker::PhantomData<&'a mut [T]>,
}

/// SAFETY: 원소 접근은 전부 `unsafe fn`이고, 계약(인덱스 비겹침)은 호출자 몫이다.
/// `T: Send`면 원소를 다른 스레드에서 만져도 된다.
unsafe impl<T: Send> Send for DisjointSlice<'_, T> {}
unsafe impl<T: Send> Sync for DisjointSlice<'_, T> {}

impl<'a, T> DisjointSlice<'a, T> {
    pub fn new(s: &'a mut [T]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
            _marker: core::marker::PhantomData,
        }
    }

    /// 원소 수 — 인덱스 계산이 이 값을 넘지 않는지 확인하는 용도.
    #[inline]
    #[allow(dead_code)] // 테스트·디버그 어서션 전용
    pub fn len(&self) -> usize {
        self.len
    }

    /// # Safety
    /// 이 dispatch 안에서 인덱스 `i`를 만지는 참여자가 자기 하나여야 한다.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self, i: usize) -> &'a mut T {
        debug_assert!(i < self.len);
        unsafe { &mut *self.ptr.add(i) }
    }

    /// # Safety
    /// 구간 `[s, e)`가 다른 참여자의 구간과 겹치지 않아야 한다.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn range(&self, s: usize, e: usize) -> &'a mut [T] {
        debug_assert!(s <= e && e <= self.len);
        unsafe { core::slice::from_raw_parts_mut(self.ptr.add(s), e - s) }
    }
}

/// 이 플랫폼에서 쓸 기본 참여자 수 (메인 포함).
///
/// 설계 §5: "4–8워커가 최적점"(Jolt 실측 — 16코어+에서 정체·역행). 코어가 많아도
/// 8을 넘기지 않는다. wasm은 1.
pub fn default_threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, MAX_THREADS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn serial_pool_runs_inline() {
        let pool = Pool::new(1);
        assert_eq!(pool.threads(), 1);
        let hit = AtomicUsize::new(0);
        pool.dispatch(|ctx| {
            assert_eq!(ctx.threads(), 1);
            assert_eq!(ctx.index(), 0);
            ctx.barrier(); // 참여자 1명 — no-op이어야 한다(교착하면 테스트가 멈춘다).
            hit.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hit.load(Ordering::Relaxed), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn every_participant_runs_once_per_dispatch() {
        let pool = Pool::new(4);
        let n = pool.threads();
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        for _ in 0..50 {
            pool.dispatch(|ctx| {
                counts[ctx.index()].fetch_add(1, Ordering::Relaxed);
            });
        }
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 50, "참여자 {i}");
        }
    }

    /// 배리어가 실제로 앞 구간의 쓰기를 뒤 구간에서 보이게 하는지.
    /// (배리어가 없으면 phase 2에서 0을 읽는 참여자가 생긴다)
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn barrier_separates_phases() {
        let pool = Pool::new(4);
        let n = pool.threads();
        if n == 1 {
            return;
        }
        let slots: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let bad = AtomicUsize::new(0);
        for round in 1..200usize {
            pool.dispatch(|ctx| {
                slots[ctx.index()].store(round, Ordering::Relaxed);
                ctx.barrier();
                // 배리어 뒤: 전원이 이번 라운드 값을 썼어야 한다.
                for s in slots.iter() {
                    if s.load(Ordering::Relaxed) != round {
                        bad.fetch_add(1, Ordering::Relaxed);
                    }
                }
                ctx.barrier();
            });
        }
        assert_eq!(bad.load(Ordering::Relaxed), 0, "배리어가 구간을 못 갈랐다");
    }

    /// 잡이 빌린 지역 데이터가 dispatch 반환 시점에 안전한지(스코프 계약).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dispatch_joins_before_returning() {
        let pool = Pool::new(4);
        for _ in 0..100 {
            let local = vec![7u32; 1024];
            let sum = AtomicUsize::new(0);
            pool.dispatch(|ctx| {
                let mut acc = 0usize;
                let mut i = ctx.index();
                while i < local.len() {
                    acc += local[i] as usize;
                    i += ctx.threads();
                }
                sum.fetch_add(acc, Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), 7 * 1024);
        }
    }

    /// 참여자마다 자기 인덱스만 쓰면 결과가 스레드 수와 무관해야 한다.
    #[test]
    fn disjoint_slice_writes_are_thread_count_invariant() {
        let mut want = vec![0u64; 512];
        for (i, w) in want.iter_mut().enumerate() {
            *w = (i as u64) * 3 + 1;
        }
        for t in [1usize, 2, 4] {
            let pool = Pool::new(t);
            let mut data = vec![0u64; 512];
            {
                let view = DisjointSlice::new(&mut data[..]);
                pool.dispatch(|ctx| {
                    let mut i = ctx.index();
                    while i < view.len() {
                        // SAFETY: 인덱스 i ≡ ctx.index() (mod threads) — 참여자마다 배타적이다.
                        unsafe { *view.get(i) = (i as u64) * 3 + 1 };
                        i += ctx.threads();
                    }
                });
            }
            assert_eq!(data, want, "threads={t}");
        }
    }

    /// 계측용(무시 기본): dispatch·배리어 자체 비용.
    /// 실행: cargo test --release -p maru-dynamics bench_pool -- --ignored --nocapture
    ///
    /// **가용 코어보다 참여자가 많으면 스핀 설계는 무너진다** — 이 계측이 그것을
    /// 눈으로 보여준다(4코어 박스에서 참여자 5명: dispatch 5ms, 배리어 4ms).
    /// `default_threads`가 `available_parallelism`으로 상한을 두는 이유다.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "계측용"]
    fn bench_pool_primitives() {
        let n = 2000usize;
        for threads in [1usize, 2, 3, 4] {
            let pool = Pool::new(threads);
            let t = std::time::Instant::now();
            for _ in 0..n {
                pool.dispatch(|_| {});
            }
            let d = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            let t = std::time::Instant::now();
            pool.dispatch(|ctx| {
                for _ in 0..n {
                    ctx.barrier();
                }
            });
            let b = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            println!(
                "threads={} (actual {}): dispatch {:.2} us, barrier {:.2} us",
                threads,
                pool.threads(),
                d,
                b
            );
        }
    }

    /// 워커 패닉이 교착이 아니라 패닉으로 전파되는지 (배리어 밖).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn worker_panic_propagates() {
        let pool = Pool::new(4);
        if pool.threads() == 1 {
            return;
        }
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // 테스트 로그 오염 방지
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.dispatch(|ctx| {
                if ctx.index() == 1 {
                    panic!("의도된 워커 패닉");
                }
            });
        }));
        std::panic::set_hook(prev);
        assert!(r.is_err(), "워커 패닉이 dispatch를 통해 전파돼야 한다");
        // 풀은 이후에도 살아 있어야 한다.
        let hit = AtomicUsize::new(0);
        pool.dispatch(|_| {
            hit.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hit.load(Ordering::Relaxed), pool.threads());
    }
}
