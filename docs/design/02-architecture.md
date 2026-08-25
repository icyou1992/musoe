# Maru(가칭) — 웹 우선 3D 강체 물리엔진 아키텍처 설계서 v0.21 ✅ 설계 최종 확정 (전 검증 층 종결)

> v0.21: 정정 반영 확인 라운드 2회(rev_382862f3 → 잔여 2건 → rev_42e81a94 **approve**) —
> 사실검증 정정의 반영분까지 재검증 완료. 문서 수준 검증 체계 종결.

> **3중 검증 완주**:
> ① 델타 적대 검수 11라운드 approve (rev_217558dd, 2026-07-24)
> ② 전체 홀리스틱 재검수 9라운드 approve (rev_7ffa1293, 2026-07-25)
> ③ 사실검증 라운드 approve (rev_0c5f050f, 2026-07-25) — jarvis 온톨로지 갭 판정(SAB 전제
>   ungrounded → 웹 재검증+GPT 지식 대조로 이중 접지 확인) + 웹 재검증 에이전트(12개 주장 중
>   10 CONFIRMED) + GPT 지식 대조. MINOR 사실 정정 6건 반영(v0.20): Rapier TGS-soft 단정 철회,
>   Jolt 기본 반복수, relaxed-simd Safari 명령별 지원, memory64 서술, WebGPU readback 근거 축소,
>   벤치 수치 신뢰도 주석.
> 총 21라운드 + 3중 트랙, 설계 수준 잔여 BLOCKER/MAJOR 없음. 구현 검증 항목은 로드맵(§11)에서 소화.
> v0.12 변경: wake 대상을 역방향 인접 기반 "모든 동적 아일랜드"로 재정의(비아일랜드 static 삭제
> 커버), 스냅샷에 pending 커맨드·미drain 이벤트 버퍼 포함, TOI 큐 정적 상한+결정적 earliest-K,
> 중복 판정·TOI 키에 source collider 포함(compound), FP 환경을 스레드별 계약으로(워커 생성+step
> 진입 가드), point projection 동률 규칙, M2/M4 완료 기준에 동결 계약 회귀 편입.
> v0.13 (홀리스틱 2차 rev_dac506a7, MAJOR 3 반영): source+target collider 키를 solved-impact·
> 완성 키까지 **전 단계 단일화**(부분 키 금지), 역방향 인접 인덱스=스냅샷 제외 파생 상태(restore
> 직후·커맨드 적용 전 stable key 순 결정적 재구축), canonical FP 상태를 **모드 공통 단일값**
> (RNE+전 flush off, L0/L1 차이는 libm/FMA 경로뿐, ambient 승계 없음)으로 확정.
> v0.14 (홀리스틱 3차 rev_2793309e, MAJOR 1 반영): wake 계약에서 **island ID 의존 제거** —
> 스냅샷에 포함된 영속 접촉/조인트 그래프 순회 + stable body 핸들 오름차순으로 재정의
> (restore 직후 0단계 커맨드 시점엔 아일랜드가 아직 없다는 파이프라인 순서 모순 해소).
> v0.15 (홀리스틱 4차 rev_64007014, MAJOR 1 반영): 변이 커맨드의 **원자 실행 순서** 고정 —
> pre-mutation 그래프에서 wake 수집·정렬 → wake 적용 → 구조 변경·인접 정리, 버퍼 내 다중
> 명령은 순서 보존하며 각자 직전 결과 그래프 기준.
> v0.16 (홀리스틱 5차 rev_00764a62, MAJOR 1 반영): **신규 겹침 wake 규칙** — edge 생성형 변이
> (필터 활성화·콜라이더 확대·kinematic 텔레포트)와 런타임 신규 접촉은 파이프라인 1단계에서
> canonical 페어 집합의 신규 페어가 sleeping 바디를 포함하면 해당 컴포넌트를 솔버 전에 wake
> (커맨드/런타임 공통 단일 정본 규칙).
> v0.17 (홀리스틱 6차 rev_eabb2662, MAJOR 2 반영): 신규 겹침 wake를 **edge-trigger에서
> level-trigger로 교체** — 내로우페이즈 제외는 "양쪽 모두 비활성" 페어뿐, 활성 상대가 있는
> sleeping 페어는 매 스텝 검사하고 해결 접촉 생성 순간 wake. 재수면 후 관통 경로 소멸 +
> 전 스텝 페어 이력 요구(스냅샷 부담) 소멸(입력=현재 상태뿐).
> v0.18 (홀리스틱 7차 rev_815c8d0a, MAJOR 1 반영): "양쪽 비활성 제외"에 **step-local dirty
> 예외** — 이번 스텝 geometry/filter/pose 변이 콜라이더가 포함된 페어는 활성 여부 무관 1회
> 검사(정적 확대·필터 활성화·정지 텔레포트의 관통 잔여 경로 봉합; dirty는 스텝 말 소거라
> 스냅샷 무부담, pending 커맨드 재적용으로 L0 정합).
> v0.19 (홀리스틱 8차 rev_26af824a, MAJOR 1 반영): dirty의 수명을 **커맨드 적용 경계 내 소비**로
> 확정 — `flush_commands()`도 dirty 페어 검사·wake까지 같은 경계에서 완결 후 소거. dirty가
> 스냅샷·쿼리·step 사이 어떤 가시 경계에도 잔존하지 않아 flush→snapshot→restore 분기 소멸.

> 2026-07-24. 근거는 `01-engine-analysis.md`(엔진 비교 분석). 설계 결정마다 [근거] 표기.
> 이름은 플레이스홀더 — crates.io/npm 네임스페이스 확인 후 확정.
> v0.2: GPT 교차검수 1차(rev_4c70067a, BLOCKER 4·MAJOR 10) 반영 — CCD 파이프라인 재배치,
> 결정론 계약 행렬, 조인트-아일랜드 연결, 매니폴드 명세, 복원계수 계약, 메모리 예산,
> 키네마틱 시맨틱, 로드맵 재편.
> v0.3: 재검수(rev_eaac9eea, BLOCKER 1·MAJOR 9·MINOR 1) 반영 — linear cast를 "TOI 정지+다음 스텝
> 해결"로 계약 통일, 결정론 총순서 키·canonical island·부동소수점 규칙, restitution 임팩트 게이트,
> SIMD 배칭 전제 완화, 스냅샷 상태 목록 확정, 제거 시 자동 wake, 스텝 원자성, 벤치 환경 고정.
> v0.4: 3차 검수(rev_a674d2f5, BLOCKER 1·MAJOR 6) 반영 — linear cast 대상을 static/kinematic으로
> 제한(동적-동적은 speculative 전담), TOI 정지 시 속도 보존+프리임팩트 레코드, **정적/키네마틱은
> union 노드가 아니다**(아일랜드 병합 붕괴 수정), 조인트 핸들·row ordinal 포함 총순서 키,
> TOI 큐 canonicalize, 스냅샷에 브로드페이즈 트리 직렬화, 오버플로 preflight+직렬 top-K 슬로우패스,
> denormal 알고리즘적 flush.
> v0.5: 4차 검수(rev_4beaa9f2, BLOCKER 1·MAJOR 5) 반영 — linear cast 대상 **static 한정**(kinematic
> 침입은 speculative 전담), speculative의 한계 봉투 명시, TOI 클램프=워커-로컬 단방향 평면(속도
> 보존·제약 정상 참여), TOI 레코드에서 속도 제거(이중 주입 원천 차단), 주입 전 세대 핸들·기하
> 재검증, ε-동시 hit 보존, 유한 자원별 count/select 2패스 일반화, 스냅샷=스텝 경계+재빌드 폐기
> 규칙, denormal은 FP 환경 계약(MXCSR FTZ/DAZ 해제 강제)으로 격상.
> v0.6: 5차 검수(rev_2de0cec7, MAJOR 3·MINOR 1 — BLOCKER 0 도달) 반영 — 서브스텝별 재캐스트로
> 순차 충돌(바닥→벽) 보존+평면 4개 캡, **페어 집합을 트리 비의존(정밀 AABB+canonical 정렬)으로
> 정의**해 재빌드 publish 타이밍 문제와 스냅샷 트리 직렬화 요구를 동시 소거, FP 환경 계약에
> ARM FPCR(FZ/FZ16)+라운딩 모드(RNE) 추가, 파이프라인 요약의 "정지" 표현을 정본 계약과 일치화.
> v0.7: 6차 검수(rev_afe544dc, MAJOR 4) 반영 — 세대 스왑 전 **catch-up widen publish 불변식**
> (leaf ⊇ 현재 AABB, debug assert), 활성 매니폴드 쌍은 linear cast 제외(임팩트 단일 소비),
> TOI 키를 frame-global TOI+substep ordinal로 정규화, 쿼리 결과 canonicalization(거리·stable 키
> 정렬, 동률 규칙)으로 트리를 완전한 성능 전용 상태로 확정.
> v0.8: 7차 검수(rev_747e8a85, MAJOR 3) 반영 — publish를 스텝 말 직렬 구간의 원자 수행으로 고정
> (widen↔스왑 경쟁 구간 제거), v0.7의 "매니폴드 쌍 캐스트 제외" 철회 → **(collider, feature,
> impact epoch) 단위 단일 소비 토큰**으로 교체(compound 바닥+벽 관통 구멍 봉합), 브로드페이즈
> 세대를 캐노니컬 상태·L0 해시에서 제외(§3↔§8 모순 제거).
> v0.9: 8차 검수(rev_842bdfa8, MAJOR 2) 반영 — 임팩트 중복 판정 키에 **body 핸들+세대 포함**
> (동시 다중 바디 독립 반발), "저장 토큰"을 **스텝 말 canonicalize의 step-local 판정 규칙**으로
> 재정의(스냅샷 경계를 넘는 토큰 상태 자체가 없음 — 스냅샷 대상은 확정된 큐뿐).
> v0.10: 9차 검수(rev_23c30797, MAJOR 2) 반영 — 중복 판정 입력을 separation 상태에서 **솔버의
> 명시적 solved-impact 기록**((body,collider,feature,epoch), restitution 실소비 순간 기록)으로
> 교체, TOI 폐기는 epoch ε_t 일치 시에만 — 스텝 내 재충돌·resting 후 고속 재충돌 보존.
> v0.11: 10차 검수(rev_559ad610, MAJOR 2) 반영 — **canonical feature ID 단일 정본 함수**(welding
> 포함, cast 경로도 동일 함수로 정규화 — 경로 간 ID 일치 CI), solved-impact 기록의 **정적 유도
> 상한**(매니폴드 예산×4×substep, 사전 예약으로 오버플로 불가).

## 0. 목표 / 비목표

**목표**
- 브라우저(WASM)에서 **기준 벤치(§6.4에 반증 가능하게 고정) 통과** — 헤드라인은 "단일스레드 3,000
  동적 바디"지만 계약은 §6.4의 씬·하드웨어·측정 정의다. [근거: Havok WASM(Babylon) ~3,000이
  공개 보고 기준 대표 실측(측정 조건 비공개 — "현존 최고" 단정은 07-25 철회, 01 문서 수치 주석
  참조) / 검수 지적: 씬 정의 없는 바디 수는 반증 불가]
- 스레드 가용 환경(COOP/COEP)에서 4–8워커로 2–4× 추가 스케일.
- 네이티브 빌드(같은 코어)에서 Jolt와 동일 벤치 비교 가능한 수준.
- 3D 강체: 충돌(볼록체·복합·메시·하이트필드) · 접촉/마찰 · 조인트 · CCD · 슬리핑.
- 결정론: 명시된 3단 계약(§8).
- API: JS/TS 1급(Three.js/Babylon 친화), Rust 네이티브 1급, C ABI 노출 가능 구조.

**비목표 (v1)**
- 소프트바디·클로스·유체 (확장 포인트만 예약, §10).
- GPU(WebGPU) 솔버 — 실험 백엔드 슬롯만. [근거: CPU 가시 결과(게임 로직·네트워킹)에 비동기
  readback 동기화 비용·지연이 구조적 — 01 §2 정정 표현 준용, "정확히 1프레임" 단정 아님]
- wasm↔native 비트 단위 결정론. [근거: FMA/라운딩 경로 차이로 달성 불가 — 명시적 비목표로 선언]
- >5km 월드 기본 지원 (f64 위치는 feature 게이트, §4).

## 1. 언어: Rust (C 대비 결정 근거)

| 기준 | Rust | C |
|---|---|---|
| WASM 툴체인 | wasm-bindgen·wasm-pack 1급, 사이즈/성능 성숙 | Emscripten (런타임 셰임 두꺼움, ammo.js 계보) |
| 메모리 안전 | 솔버 밖 코드 전체 안전, unsafe 국소화 가능 | 전체가 수동 |
| SIMD 이식성 | `wide` 크레이트로 x86/NEON/WASM128 단일 소스 | 인트린식 분기 수동 관리 |
| 병렬 안전성 | Send/Sync가 잡 시스템 설계 오류를 컴파일 타임에 차단 | 런타임 디버깅 |
| C ABI | `#[no_mangle] extern "C"`로 노출 가능 → C의 장점 포함 | — |
| 선례 | Rapier(성능 상한 실증), Bevy 생태계 | Box2D v3(단 2D) |

결론: **Rust**. C가 이기는 유일 축(컴파일러 단순성·ABI 안정)은 C ABI 노출로 흡수한다 —
단 `extern "C"` 선언만으로는 부족하고(재검수 지적), maru-ffi가 opaque handle·POD 전용 시그니처·
할당 소유권 규칙·panic 차단(`catch_unwind` 경계)·ABI 버전 상수를 갖출 때 성립하는 주장이다.
이 요건은 maru-ffi 설계 요구사항으로 §2에 귀속.
C++(Jolt 포크/바인딩)를 쓰지 않는 이유: WASM이 1급 타깃인데 Emscripten 경유 C++는 JS 경계
설계 자유도가 낮고(§7이 본 설계의 승부처), 우리가 통제하는 코어가 목적이기 때문.

## 2. 크레이트 구조

```
maru/
├─ maru-math       # Vec3/Quat/Mat, SIMD 추상 (wide 기반, 4레인 f32 = WASM SIMD128 동형)
├─ maru-collide    # 셰이프, GJK/EPA, 접촉 매니폴드, BVH(브로드페이즈 트리), 레이/셰이프 캐스트
├─ maru-dynamics   # 적분기, TGS-soft 솔버, 조인트, 아일랜드, 슬리핑
├─ maru            # World 파사드: 파이프라인 오케스트레이션, 이벤트, 쿼리 API
├─ maru-wasm       # wasm-bindgen 바인딩 + flat buffer 프로토콜 (§7)
├─ maru-ffi        # C ABI (후순위, 네이티브 엔진 임베드용)
└─ maru-bench      # 벤치/샘플 하니스 (Solver2D 방법론, M0부터 존재)
```

- 정밀도: **f32 단일 코어**. `large-world` feature가 위치만 f64로(`Real`/`RVec3` 패턴).
  [근거: Jolt JPH_DOUBLE_PRECISION 5–10% 비용 실측 / Rapier의 f32·f64 포크 크레이트는 유지보수 2배 — 회피]
- SIMD: stable `wide`. nightly `std::simd` 배제. FMA는 `relaxed-simd` 타깃 feature로만 활성,
  수치 계약은 비FMA 기준. [근거: Safari relaxed-simd 지연 이력, WASM SIMD128 비FMA 기본]

## 3. 데이터 모델

- **핸들 = generational index** (`BodyHandle(u32 index, u32 gen)`), 모든 그래프 관계는 flat arena
  인덱스. 참조/포인터 없음. [근거: Rust 생태계 만장일치 — Rapier·Fyrox 동일 결론]
- **바디 저장 = 이중 레이아웃** (검수 반영 — "안정 슬롯"과 "아일랜드 정렬 패킹"은 한 배열로 양립 불가):
  - **월드 배열(캐노니컬)**: 핸들 index로 안정 주소. 위치·회전·속도 등 SoA, vec3는 4-f32 패딩.
    JS 트랜스폼 버퍼(§7)와 스냅샷(§8)은 이 배열에서 나온다.
  - **솔버 뷰(스텝 한정)**: island finalize 시 활성 바디만 아일랜드 순서로 gather한 SIMD128 레인
    정렬 패킹 블록. 솔버 종료 후 월드 배열로 scatter. [근거: Box2D v3 "솔버 상태를 레인 폭에 패킹"]
  - gather/scatter 비용은 공짜가 아니다 — §6.4 벤치에 별도 계측 항목으로 포함하고, 비용이 이득을
    상회하면 "월드 배열 직접 솔브(패킹 생략)"로 폴백하는 결정을 M2에서 데이터로 내린다.
  - **SIMD 이득의 정직한 범위** (재검수 반영): WASM SIMD128에는 일반 gather 명령이 없다.
    보장되는 이득은 ①제약 1개 내부의 벡터 연산(vec3/quat 산술 4레인), ②브로드페이즈 4-ary 순회다.
    **4제약 동시 배칭**(Box2D v3의 AVX2 방식)은 스칼라 gather + 레인 내 바디 중복 회피(같은 색
    안에서도 배치 4개가 같은 바디를 공유하면 안 됨)를 요구하는 별도 설계 — M2에서 스칼라 솔버
    기준선을 먼저 재고 배칭은 벤치 이득이 증명될 때만 도입한다. §6.4 성능 목표는 배칭 없이(①·②만)
    달성 가능해야 한다.
  - 콜드 상태(이름·유저데이터·필터·머티리얼)는 별도 배열.
- **셰이프 = 불변 + ref-count 공유**, 생성 시 질량중심 재중심화. convex radius 1급 파라미터.
  [근거: Jolt]
- **스텝 스크래치 = 고정 용량 arena** (성장 불가 — `bumpalo` 기본 동작은 부족 시 backing 성장이라
  "힙 할당 0" 불변식을 깨므로 사전 reserve된 fixed-capacity 구현 사용, 검수 반영). 스텝 끝 일괄 해제.
  [근거: Jolt TempAllocator; WASM 할당자는 특히 느림]
- **월드 메모리 예산 명시**: 월드 생성 시 최대 바디/페어/매니폴드/이벤트/커맨드 수를 설정(합리적
  기본값 제공). 스텝 중 일반 힙 할당 0은 CI allocation counter로 검증. [근거: §6 규율 3 "조용한
  폴백 금지"의 메모리 버전 — 첫 밀집 프레임의 할당 스파이크/WASM OOM 방지]
- **상한 초과 시 스텝 원자성** (재검수 반영 — 스텝 도중 오류 반환은 부분 갱신 월드를 남긴다):
  상한 초과는 스텝을 중단시키지 않는다. 스텝을 완주해 월드는 항상 유효·일관 상태를 유지하며,
  반환값의 오버플로 플래그+카운터로 호출자가 예산을 키워 대응한다.
  - **결정적 드롭의 실행 계약 — 모든 유한 자원에 동일 패턴** (3·4차 검수 반영): 각 유한 자원마다
    **①싼 상한 추정(count) → ②초과 예고 시 그 스테이지만 직렬 결정적 top-K 선택(select)**의
    2패스 계약을 적용한다. 페어 = 브로드페이즈 후보 카운트; **매니폴드** = 채택된 페어의 셰이프별
    최대 매니폴드 수 합(compound는 자식 수 기반 상한); **이벤트** = 매니폴드 전이 수 상한(대량
    접촉 종료 프레임 포함); **커맨드** = 스텝 중이 아니라 enqueue 시점에 즉시 실패(fast-fail이라
    스텝 원자성과 무관). 워커별 버퍼가 먼저 차는 순서로 유실되는 경로는 존재하지 않는다.
    오버플로 프레임은 예외 상황이므로 느려도 되고, 결정성이 우선이다. 정상 프레임은 이 비용을
    내지 않는다. begin/end 추적 집합 갱신은 드롭 확정 후의 최종 집합으로 canonical 순서에서 1회.
  - **드롭과 이벤트 정합**: 드롭된 쌍은 "이번 스텝에 존재하지 않았음"으로 취급하되, 전 스텝에
    live였던 쌍이 용량 드롭되면 **end 이벤트를 발행**한다 — begin 없는 end, end 없는 소멸을
    금지하는 상태 전이 규칙을 begin/end 추적 집합에 일원화.
- 접촉 캐시: **PCM(persistent contact manifold), 매니폴드당 최대 4접점 축소**.
  [근거: PhysX PCM + Rapier 2025 manifold reduction 실증]
  - **롤백의 정본은 스냅샷이다** (재검수 반영 — "stateless 모드=PCM 끔"만으로는 롤백 후 이벤트/해시
    분기): 캐노니컬 스냅샷(§8)이 바디 상태 외에 **PCM 매니폴드+warm-start 임펄스, 조인트 warm-start,
    슬리핑 타이머, 접촉 begin/end 추적 집합, 핸들 free-list, TOI-hit 큐, 누적 이벤트 시퀀스 번호,
    (v0.6 갱신 — 브로드페이즈 트리는 스냅샷에서 **제외**: §4.1의 "페어 집합은 트리 비의존" 계약
    덕에 트리는 순수 파생 가속 구조가 되었다. restore는 바디 상태로부터 트리를 재구축하며,
    토폴로지가 원 실행과 달라도 페어 집합·해시가 동일함이 계약으로 보장된다 — 3~4차의 트리
    직렬화 요구는 이 계약으로 대체), 그리고 **pending 커맨드 버퍼와 미drain 이벤트 버퍼**
    (홀리스틱 검수 반영 — 커맨드 enqueue 후 step 전 스냅샷, 또는 step 후 drain 전 스냅샷 시 이
    버퍼들이 빠지면 restore 재생이 원 실행과 분기한다)를 전부 포함하고, restore 후 재생은 원
    실행과 L0 비트 동일임을 CI로 보증한다. **스냅샷은 스텝 경계에서만 허용**(스텝 중간 상태
    없음 — 경계에 걸친 사용자 가시 버퍼는 위처럼 스냅샷에 포함되므로 경계 제한과 모순 없음). 별도 "stateless 월드 옵션"은 성능
    실험용 캐시 비활성 스위치로 격하 — 롤백 정확성을 그 옵션에 의존하지 않는다.

## 4. 시뮬레이션 파이프라인 (스텝 구조)

```
step(dt):                            # dt = 고정 틱. 호스트 프레임과 무결합 [근거: Unity FixedUpdate 스터터]
  0 커맨드 버퍼 적용 (생성/삭제/텔레포트/파라미터 — §7 쓰기 배치의 반영 지점)
  1 broadphase 갱신/쿼리 (AABB는 속도×dt만큼 확장 = speculative margin)  ──┐
  2 narrowphase: 접촉 + speculative contact 생성 (GJK/EPA + 클리핑 매니폴드) │ 병렬 구간
    + island union-find 구축 (접촉 edge와 조인트 edge 모두)                 │
  3 island finalize + 결정적 정렬 (O(N) 직렬 2패스 + pair-key 소트)       ──┘
  4 solver: substep × N:       # TGS-soft
       적분(linear-cast CCD 바디는 TOI 클램프 평면 적용 — 속도 보존, TOI-hit 큐 기록) →
       조인트/접촉 임펄스(soft) → 위치 보정
  5 sleeping 판정 (island 단위)
  6 월드 배열 → 트랜스폼 버퍼 기록, 이벤트 버퍼 flush (접촉 시작/종료, 트리거)
```

> **CCD 계약 v0.5** (4차 검수 BLOCKER 반영 — kinematic 대상은 TOI 후에도 계속 이동하므로
> "CCD 바디 동결"로는 관통 차단 불성립. 대상을 static으로 좁히고 나머지는 speculative 봉투로):
> - **speculative contact**(기본, 모든 쌍): 1–2단계에서 확장 AABB(양쪽 속도×dt — kinematic의
>   처방 속도 포함)로 "이번 스텝 도달 예정" 쌍을 접촉으로 미리 생성 → 아일랜드에 정상 포함,
>   솔버가 도착 전에 감속. **동적-동적·동적-kinematic 고속 쌍은 이 경로가 전담한다.**
>   [근거: Box2D v3]
> - **speculative의 정직한 봉투** (4차 검수 반영 — 확장 AABB는 후보 생성일 뿐 연속 내로우페이즈가
>   아니다): 접촉 생성은 현재 자세의 closest feature 법선 기반 — 얇은 셰이프 두 개가 고속으로
>   비스듬히 교차하며 충돌 시점 feature가 바뀌는 경우는 놓칠 수 있다. v1은 이를 **명시 한계**로
>   수용하고 문서에 "스텝당 이동 ≲ 최소 두께" 가이드를 싣는다. conservative advancement 기반
>   pair-level TOI는 §10 확장 포인트로 예약(요구 실증 시).
> - **linear cast**(opt-in, **대상 static 한정** — kinematic 대상 제외가 4차 BLOCKER의 해소):
>   캐스트는 항상 수행한다(7차 검수 — v0.7의 "활성 매니폴드 쌍 전체 제외"는 compound의 바닥
>   매니폴드가 벽 feature 캐스트까지 꺼버리는 과잉 처방이라 철회). **임팩트 중복은 캐스트 생략이
>   아니라 중복 판정 규칙으로 막는다** (8차 검수 반영 — v0.8 "소비 토큰"의 두 결함 수정):
>   ① 판정 키에 **body와 source collider가 포함**된다 — (body 핸들+세대, source collider
>   핸들+세대, target collider 핸들+세대, manifold/feature, impact epoch(frame-global TOI 기준)).
>   두 공이 같은 바닥 feature를 같은 시각에 쳐도, 한 compound 바디의 두 자식 콜라이더가 같은
>   feature를 쳐도 각자 독립 반발한다. ② 이것은 저장되는 토큰이 아니라 **스텝 말 canonicalize에서 1회 계산되는
>   step-local 판정**이다. ③ 판정의 입력은 최종 separation 상태가 아니라 **솔버의 명시적
>   solved-impact 기록**이다(9차 검수 반영 — separation≤0은 resting contact도 해당하고, 스텝 안
>   재충돌(t=0.2 임팩트 후 분리→t=0.8 재임팩트)의 두 번째 임팩트를 오폐기한다): 솔버가
>   restitution을 **실제 소비한 순간마다** (body, source collider, target collider, feature,
>   impact epoch)의 step-local solved-impact 레코드를 남기고, canonicalize는 TOI 레코드를
>   **같은 키에 epoch가 ε_t 이내로 일치하는 solved-impact가 있을 때만** 폐기한다 — 다른 epoch의 별개 임팩트, resting/threshold
>   미만 접촉 뒤의 고속 재충돌은 생존한다. solved-impact 기록도 step-local이라 스텝 경계를 넘지
>   않는다. 스냅샷은 스텝 경계에서만 찍히므로(§3) 스냅샷 대상은 확정된 큐뿐이며, restore 후
>   재생도 같은 규칙을 같은 입력에 적용해 L0 동일.
>   ④ **feature ID는 단일 정본 함수가 부여한다** (10차 검수 반영 — cast와 매니폴드가 다른 ID를
>   만들면 매칭 실패로 restitution 재발화): 콜라이더별 canonical feature ID 함수(메시의 internal-
>   edge welding·remap 포함)는 §4.2 매니폴드 계층에 단 하나 존재하고, **shape cast 경로도 hit
>   feature를 같은 함수로 정규화한 뒤 기록**한다. cast-원시 feature(개별 triangle/edge)를 그대로
>   키에 쓰는 것은 금지. CI에 convex face·edge-edge·compound child·mesh welded edge의 경로 간
>   ID 일치 테스트.
>   ⑤ **solved-impact 기록은 유도 상한을 갖는 유한 자원** (10차 검수 반영): restitution 소비는
>   채택 매니폴드의 접점당 임팩트 epoch당 최대 1회이므로 상한 = 채택 매니폴드 예산 × 4접점 ×
>   substep 수로 **정적으로 유도**된다 — 이미 캡된 자원에서 파생되므로 워커-로컬 버퍼를 그
>   상한으로 사전 예약하면 오버플로가 발생할 수 없다(§3 count/select의 별도 select 패스 불요,
>   debug assert로 상한 검증).
>   4단계에서 자기 아일랜드 워커가 적분 중 셰이프 캐스트로 TOI를 구해 위치를 TOI에서 절단하고,
>   **잔여 서브스텝 동안 그 static 표면에 대한 워커-로컬 단방향 평면 클램프**(침투 방향 위치
>   변화만 사영 제거, 슬라이드 허용)를 적용한다. 단일 바디 + static 표면에만 쓰기가 발생하므로
>   병렬 안전. **속도는 보존**되고 조인트·접촉 솔브에 정상 참여한다(위치 오차는 표면에 얹힌
>   상태와 동일 — 이후 스텝의 위치 보정이 해소). [근거: Jolt LinearCast stop-at-TOI와 동형]
> - **TOI-hit 레코드 = 기하 정보만**: (body 핸들+세대, **source collider 핸들+세대, target
>   collider 핸들+세대** — 홀리스틱 검수 반영: compound 바디의 두 자식 콜라이더가 같은 static
>   feature를 같은 epoch에 칠 수 있으므로 충돌 주체 콜라이더까지 식별. solved-impact 기록과
>   중복 판정 키(아래 ①)도 동일하게 source+target 쌍 포함, 법선, 접점, feature, TOI).
>   **속도는 기록하지 않는다** — 속도가 보존되므로 다음 스텝 접촉 솔브가 실제 접근 속도로
>   restitution을 계산한다(4차 지적 "저장 속도로 이중 주입" 원천 제거).
> - **TOI-hit 큐도 유한 자원 — 정적 유도 상한** (홀리스틱 검수 반영): 서브스텝당 바디당 유효 hit
>   상한 8(클램프 평면 4 + ε-동시 여분)을 계약으로 두고, 초과 hit는 그 서브스텝의 총순서 키 기준
>   **결정적 earliest-K 선택**으로 잘라낸다(오버플로 카운터 노출). 큐 전체 상한 = CCD opt-in 바디
>   수 × substep 수 × 8 — 사전 예약으로 워커-로컬 버퍼가 먼저 차는 유실 경로가 없다.
> - **다음 스텝 주입 전 재검증** (4차 검수 반영): 커맨드 적용 후 세대 핸들 검증 + 해당 쌍
>   내로우페이즈 재실행 — collider 삭제·텔레포트·필터 변경이면 레코드를 카운터와 함께 폐기.
>   빈 공간에서 유령 접촉이 발화하지 않는다.
> - **서브스텝별 재캐스트 — 순차 충돌 보존** (5차 검수 반영 — earliest-hit만으로는 "바닥 t=0.2 →
>   슬라이드 → 벽 t=0.8"의 두 번째 표면을 관통): 클램프 평면은 스텝당 1회가 아니라 **각 서브스텝
>   적분마다 잔여 이동 경로를 static 트리에 재캐스트**해 누적한다(같은 워커의 로컬 작업 — 병렬
>   안전 불변). 활성 클램프 평면은 바디당 최대 4개; 초과하는 hit가 나오면 그 바디의 이번 스텝
>   이동을 그 지점에서 종료한다(구석에 낀 상태 — 결정적이고 안전한 퇴화).
> - **TOI-hit 큐 canonicalize**: 병렬 append는 스크래치 — 스텝 말 직렬 구간에서 총순서 정렬·중복
>   제거. 키의 시각 성분은 재캐스트 로컬 fraction이 아니라 **frame-global TOI = (substep ordinal +
>   local fraction)/N** + substep ordinal(6차 검수 — 로컬 fraction만으로는 다른 서브스텝의 별개
>   임팩트가 같은 키로 합쳐지거나 시간 역순 주입 가능). 완성 키 = (body, **source collider,
>   target collider**, frame-global TOI, substep ordinal, feature) — 홀리스틱 2차 반영: 정렬·
>   dedupe·solved-impact 비교·주입의 **전 단계가 이 단일 키를 사용**한다(부분 키 사용 금지).
>   서브스텝 재캐스트로 수집된 모든 유효 hit(코너의 ε-동시 hit 포함)가 보존되고, 주입은
>   frame-global TOI 순.
> - 솔버가 끝난 위치를 소급 수정하는 단계는 존재하지 않는다.

### 4.1 브로드페이즈 — Jolt 듀얼 트리 이식
- **정적 트리 / 동적 트리 분리.** 정적은 삽입 배치 시에만 리핏, 동적은 매 스텝 증분.
- **4-ary(쿼드) 트리, 노드 바운드는 SoA `[minx4][miny4]...`** → 순회 1스텝 = SIMD128 비교 1회.
  [근거: Jolt 쿼드트리가 SIMD 4레인과 동형 — WASM SIMD128에 정확히 일치하는 유일한 조합]
- 갱신은 widen-only(단조 확장) + 주기적 타이트 재빌드 + 세대 스왑. 단일스레드에선 원자 연산이
  일반 쓰기로 컴파일되므로 비용 없음, 스레드 모드에선 그대로 락프리.
- **트리는 가속 구조일 뿐 — 페어 집합은 트리 비의존으로 정의** (5차 검수 반영, 결정론의 초석):
  트리 순회는 widen된 노드 바운드 때문에 superset 후보를 내놓지만, **최종 페어 집합 = 후보에
  정밀 바디 AABB(자세+speculative margin에서 계산) 겹침 판정을 적용해 canonical 정렬한 결과**로
  정의한다. 용량 count/select(§3)도 이 정밀 판정 후 집합 기준. 따라서 트리 토폴로지·재빌드
  타이밍은 성능에만 영향을 주고 L0 상태 해시에는 영향을 줄 수 없다 — 백그라운드 재빌드의
  publish 시점이 워커 실행시간에 좌우되어도 무해하며, publish 규칙을 따로 고정할 필요가 없다.
- **publish 불변식 — superset 보장은 스왑 순간에도 성립해야 한다** (6차 검수 반영: 과거 자세로
  지은 tight 트리를 그대로 스왑하면 이동한 바디의 현재 AABB를 노드 바운드가 포함하지 못해 실제
  겹침이 후보에서 누락된다): 재빌드 **연산**은 비동기지만 **publish(catch-up widen + 세대 스왑)는
  스텝 말 직렬 구간(§4 6단계)에서만 원자적으로 수행**한다(7차 검수 반영 — widen 읽기와 스왑
  사이에 바디가 다시 움직이는 경쟁 구간 제거: 직렬 구간에는 바디 변이가 존재하지 않는다).
  그 스텝의 직렬 구간까지 재빌드가 미완이면 publish는 다음 스텝 직렬 구간으로 미룬다 — 트리는
  성능 전용(§4.1 페어 집합 계약)이므로 지연은 무해하다. debug 빌드는 스왑 시
  "∀바디: leaf bounds ⊇ 현재 AABB" 불변식을 assert.
- 삽입 API는 **배치가 유일한 fast path**. 단건 삽입은 명시적 slow path로 문서화.
  [근거: Jolt/Guerrilla 스트리밍 스톨 5–10ms 사고]

### 4.2 내로우페이즈
- GJK + EPA, convex radius로 EPA 진입 최소화. 지원 셰이프: sphere/capsule/box/convex hull/
  compound/triangle mesh(정적 전용)/heightfield.
- **매니폴드 생성은 GJK/EPA만으로 안 된다** (검수 반영 — GJK/EPA는 단일 witness/침투 방향만 제공):
  접촉 법선 확정 후 양쪽 supporting face를 찾아 **폴리곤 클리핑으로 face 매니폴드(최대 4점) 생성**,
  각 접점에 feature ID(edge/vertex 조합) 부여. box-on-box 면 접촉이 1점으로 풀리면 스택이 회전
  진동한다. [근거: Jolt·PhysX 모두 face clipping 계열 매니폴드]
  - 재검수 반영 상세 (임의 convex hull 일반화): **edge-edge 접촉** 경로 별도 처리(법선이 면이 아닌
    에지 쌍에서 나올 때 클리핑 아닌 최근접점), **reference/incident face 선택 동률 규칙**(법선 내적
    동률 시 낮은 face index 우선 — 결정론), **4점 축소 규칙 명시**(최심점 고정 + 잔여 중 면적 최대
    조합), feature ID 안정성 테스트는 box뿐 아니라 hull 회전 슬라이드 케이스 포함.
- 접촉은 PCM 캐시에 feature ID 매칭으로 프레임 간 대응시키고 warm-start 임펄스와 함께 저장.
- **트라이앵글 메시 접촉은 전용 문제 영역** (검수 반영): convex-vs-mesh는 BVH 순회 → 삼각형별
  접촉 → **내부 에지 억제**(인접 삼각형 법선으로 active edge 판정/welding — 동일 평면 타일 위를
  미끄러질 때 내부 에지를 법선으로 잡아 걸리는 고전 버그), one-sided/backface 규칙, 퇴화 삼각형
  거부. 전용 마일스톤(M4)과 seam-slide 회귀 테스트를 로드맵에 명시.

### 4.3 아일랜드 — 락프리 union-find (Jolt 방식)
- **연결성 그래프의 edge = 접촉 쌍 + 활성 조인트** (검수 BLOCKER 반영 — 접촉만 union하면 서로
  안 닿은 hinge 체인이 바디마다 별개 아일랜드로 갈라져 조인트 누락/워커 경합). 조인트 edge는
  스텝 시작에 일괄 union, 접촉 edge는 내로우페이즈 중 발견 즉시 union(경로 압축).
- **union은 동적-동적 edge만** (3차 검수 MAJOR 반영 — v0.3의 "두 비동적 쌍만 제외"는 치명적 오류):
  static/kinematic 바디는 **union 노드가 아니다** — 접촉/조인트의 anchor로만 참조된다. 그렇지
  않으면 정적 바닥 하나가 그 위의 서로 무접촉인 동적 바디 전부를 단일 거대 아일랜드로 병합해
  island 병렬성과 개별 슬리핑이 붕괴한다(바닥 위 3,000개 = 아일랜드 1개). 동적-비동적 제약은
  동적 쪽 바디의 아일랜드에 귀속. 센서 접촉(비해결)·필터 비해결 쌍도 union 제외.
  [근거: Box2D·Jolt 공히 static은 island 미참여]
- 직렬 finalize는 O(N) 2패스 + **결정적 정렬**(§8). 재검수 반영으로 규칙을 총순서로 강화:
  - **제약 정렬 키(전순서)** — 3차 검수 반영으로 완성형 명시: endpoint는 (min(bodyA,bodyB),
    max(bodyA,bodyB))로 정규화하고, 키 = (정규화 body쌍, 제약종류, **접촉이면** 정규화 collider쌍
    + manifold/feature ID, **조인트면** stable joint handle + row ordinal). 같은 바디 쌍에 hinge
    2개, 한 조인트의 축/모터/리밋 다중 row가 있어도 동률이 없다.
  - **canonical island ID**: 병렬 union-find의 루트는 워커 순서 의존적 스크래치일 뿐이다 —
    finalize에서 컴포넌트별 **최소 바디 인덱스를 대표로 재계산**하고, 아일랜드 목록·gather 순서·
    이벤트 순서 전부 canonical ID 오름차순으로 고정.
  - **이벤트 버퍼도 flush 전 동일 키로 정렬** — begin/end 이벤트 순서가 워커 스케줄에 좌우되지 않음.
  [근거: Jolt — 병렬 페이즈 간 직렬 병목 80% 절감; Jolt도 결정론을 위해 정렬 패스 유지]
- persistent islands(정착 씬에서 재계산 상각)는 M4 최적화 항목. [근거: Box2D v3·Rapier 2025]

### 4.4 솔버 — TGS-soft
- 프레임을 N 서브스텝으로 분할, 서브스텝마다 적분→속도 임펄스→위치 보정. 접촉/조인트는
  soft constraint(주파수·감쇠 파라미터) — 서브스텝 덕에 실효 Hz가 높아져 강체처럼 보임.
- [근거: Solver2D 실증 + Box2D v3·Avian 채택. Rapier는 유사 임펄스+substep 계열로만 취급
  (07-25 사실검증 — 01 §2 정정 참조). XPBD 기각 사유 = 마찰 강건성·해석 순서 민감성·특허
  리스크(Avian 이주 문서)]
- 마찰: Coulomb, 접촉 법선 임펄스와 동일 루프에서 해석. 기본 반복수는 보수적으로(안정성 우선)
  잡고 벤치로 조정. [근거: Chaos 교훈 — 기본값이 아키텍처만큼 중요]
- **복원계수(restitution) 계약** (검수 반영 — 서브스텝마다 적용하면 충돌 1회에 에너지 다중 주입):
  접촉 생성 시점의 접근 상대속도를 캡처해 두고, **실제 임팩트가 발생한 서브스텝에서 1회만** 목표
  반발 속도로 적용. **임팩트 게이트**(재검수 반영): separation ≤ 0 도달 또는 TOI-hit로 확인된
  경우에만 발화 — speculative contact이 생성만 되고 스텝 말에도 간격이 남으면 restitution은
  발화하지 않는다(벽에 닿기 전 조기 반발 방지). impact threshold(기본 ~1m/s) 미만은 restitution 0
  (정착 안정성). combine 규칙 기본값: restitution = max(a,b), friction = √(a·b). 머티리얼별
  오버라이드 가능. [근거: Box2D v3가 soft-step에서 restitution을 별도 패스로 처리하는 것과 동형]
- **warm-start 임펄스의 서브스텝 스케일링**: 캐시된 임펄스는 서브스텝 dt 기준으로 정규화해 저장
  (N 변경 시 재사용 가능하도록) — 솔버 계약 문서에 수식으로 고정 후 구현.
- 조인트 v1: fixed / point(ball) / hinge / prismatic / distance + 모터·리밋. 6DOF는 v1.1.
  조인트도 soft constraint 정식화(접촉과 동일 수학) — 별도 솔버 없음. 근거는 Avian이 아니라
  (Avian은 조인트에 XPBD 계열을 잠정 유지 — 01 §2 정정) **Box2D v3가 조인트까지 soft-step 단일
  솔버로 처리하는 선례 + 이중 솔버 유지비 회피**다. M2의 혼합 수렴 테스트가 이 선택의 검증 게이트.

### 4.5 CCD (파이프라인 내장 — 별도 단계 아님, §4 서두 참조)
- 기본 = **speculative contact**: 브로드페이즈 AABB를 속도×dt로 확장하고, 내로우페이즈가
  "이번 스텝 도달 예정" 쌍을 거리 있는 접촉으로 생성 → 솔버가 도착 전 감속. TGS-soft와 자연 결합.
- 고속 소형 바디만 opt-in **linear cast** (대상 **static 한정**, 속도 보존 — 계약은 §4 서두 블록이
  유일한 정본. 폐기 이력: v0.2 "같은 서브스텝 해결"=아일랜드 확정 모순, v0.3 "정지·속도 소거"=반발
  유실, v0.4 "kinematic 대상 허용"=이동 대상 관통 불차단). 전역 CCD 없음.
  [근거: Havok per-object opt-in, Jolt LinearCast 저비용 실측, PhysX의 CPU 잔류 CCD 반면교사]
- 한계 명시: speculative는 극고속 회전체의 얇은 벽 통과를 완전히는 못 막는다(법선 방향 근사) —
  그런 바디가 linear cast opt-in 대상이라는 가이드를 API 문서에 명기.

### 4.6 슬리핑
- island 단위 일괄 sleep/wake.
- **제거·변이 시 자동 wake가 기본** (재검수 반영 — v0.2의 Jolt 규칙 복사 철회): 바디 제거,
  콜라이더/필터 변경, kinematic 텔레포트 시 자동 wake. **wake 대상의 정의 — 아일랜드 비의존**
  (홀리스틱 검수 반영: static/kinematic은 아일랜드에 속하지 않고(§4.3), 아일랜드는 스텝 3단계에서
  계산되므로 restore 직후 0단계 커맨드 적용 시점엔 존재하지도 않는다 — 홀리스틱 3차): wake는
  아일랜드가 아니라 **영속 접촉/조인트 그래프(PCM 매니폴드+조인트 정의 — 둘 다 스냅샷 포함)**
  위에서 정의한다. 변이된 바디에서 역방향 인접을 시작점으로 동적 바디 간 접촉/조인트 edge를
  그래프 순회해 도달하는 **모든 sleeping 동적 바디**를 수집하고, **stable body 핸들 오름차순**으로
  깨운다. 아일랜드 구성·canonical island ID는 어디에도 필요 없다 — restore 직후 첫 커맨드가
  받침 삭제여도 스냅샷의 그래프만으로 원 실행과 동일한 wake 집합·순서가 나온다.
  동적 바디 변이면 자기 자신 포함.
  **커맨드별 원자 실행 순서** (홀리스틱 4차 반영 — 구조 변경이 먼저 일어나면 wake의 근거인
  시작점·edge가 사라진다): 제거·콜라이더 변경·필터 변경 등 변이 커맨드 각각은
  ① **변이 전(pre-mutation) 그래프에서 wake 대상 수집·stable 정렬** → ② wake 적용 →
  ③ 구조 변경·인접 정리 — 이 3단을 명령 단위의 단일 원자 절차로 실행한다. 커맨드 버퍼에 변이
  명령이 여러 개면 버퍼 순서를 보존하며 각 명령이 자신의 pre-mutation 그래프(직전 명령들의
  결과)를 기준으로 처리된다.
- **신규 접촉 wake — edge를 만드는 변이·런타임 접근의 정본 규칙** (홀리스틱 5·6차 반영 — v0.16의
  "전 스텝에 없던 신규 페어" edge-trigger는 speculative 후보가 실접촉보다 먼저 생기는 경우
  재수면 후 관통을 허용했고, 전 스텝 페어 집합 이력을 스냅샷에 요구했다 — 둘 다 폐기):
  **level-trigger로 재정의한다.** 내로우페이즈에서 제외되는 페어는 **양쪽 모두 비활성**
  (sleeping/static/정지 kinematic)인 페어뿐이다 — 활성이거나 이동 중인 쪽이 하나라도 있으면
  sleeping 상대를 포함해 매 스텝 정상 검사한다. 그 검사가 sleeping 동적 바디를 포함하는 **해결
  접촉을 생성하는 순간**, 그 바디의 sleeping 동적 컴포넌트(영속 그래프 순회)를 stable body 핸들
  순으로 **솔버 전에** 깨운다. 필터 활성화·콜라이더 확대·kinematic 텔레포트(커맨드 유발)와
  느리게 접근하는 kinematic 플랫폼(런타임 유발) 모두 이 규칙 하나로 커버된다: 접촉이 확정되는
  바로 그 스텝에 발화하므로 "먼저 깨웠다 재수면 후 관통" 경로가 없고, 전 스텝 이력이 필요 없어
  스냅샷 추가 부담도 없다(입력은 현재 상태뿐 — restore 후에도 동일 판정). 비용은 활성체와
  경계를 맞댄 sleeping 페어의 내로우페이즈뿐이며 이는 업계 표준 비용이다.
  [근거: Jolt·Box2D 공히 "한쪽이라도 활성인 페어는 검사, 접촉 발생 시 wake"가 표준]
  **"양쪽 비활성 제외"의 dirty 예외** (홀리스틱 7차 반영 — 비활성 쪽이 방금 변이된 경우까지
  제외하면 정적 콜라이더 확대·static↔sleeping 필터 활성화·속도 0 `set_pose` 텔레포트가 sleeping
  바디를 관통 상태로 남긴다): 0단계 커맨드 적용이 geometry/filter/pose를 변이한 콜라이더에
  **step-local dirty 마크**를 남기고, dirty 콜라이더가 포함된 canonical 페어는 양쪽 활성 여부와
  무관하게 **이번 스텝 1회 내로우페이즈 검사** 대상이 된다(해결 접촉 생성 시 위 규칙대로 wake).
  **dirty는 그것을 만든 커맨드 적용 경계 안에서 소비된다** (홀리스틱 8차 반영 — `flush_commands()`
  가 스텝 밖에서 커맨드를 소비하면 dirty 수명이 붕 뜬다): 커맨드 적용을 수행하는 주체가 스텝
  0단계든 `flush_commands()`든, **같은 경계 안에서 dirty 페어의 1회 내로우페이즈 검사와 wake까지
  완결하고 dirty를 소거**한다. 즉 `flush_commands()` = 커맨드 적용(pre-mutation wake 포함) +
  브로드페이즈 반영 + dirty 페어 검사·wake. dirty는 어떤 사용자 가시 경계(스냅샷·쿼리·step
  사이)에도 잔존하지 않으므로 스냅샷 직렬화 대상이 아니고, "flush 후 snapshot → restore" 경로도
  원 실행과 동일하게 재생된다(양쪽 다 flush 시점에 이미 wake 완료). 이를 위해 비동적 바디도 자신을 참조하는 접촉/제약
  인덱스를 유지한다(아일랜드 미참여와 별개의 adjacency 데이터). **이 역방향 인덱스는 스냅샷에
  포함하지 않는 파생 상태다** (홀리스틱 2차 반영): restore 직후·pending 커맨드 적용 **전에**,
  스냅샷에 포함된 PCM 매니폴드·조인트 정의로부터 stable key 오름차순으로 결정적으로 재구축한다 —
  restore 후 첫 커맨드가 정적 받침 삭제여도 원 실행과 동일한 wake 집합·순서가 나온다.
  대량 스트리밍 언로드용으로만 `remove_batch_no_wake` opt-out을 제공(원거리 청크 언로드가 용례).

## 5. 스레딩 모델 — "1스레드가 1급, 워커는 승수"

- **베이스라인 = 단일스레드 완전 동작.** 모든 병렬 구조는 단일스레드에서 순차 실행으로 강등되며
  원자 연산은 비원자로 특수화(제네릭 executor 트레이트). [근거: COOP/COEP는 서드파티 임베드에서
  흔히 불가 — 웹 위젯 배포가 1급 시나리오]
- 스레드 모드: 기능 감지(SAB 존재) 시 Web Worker 4–8개 + SharedArrayBuffer 잡 큐.
  [근거: Jolt 실측 — 네이티브도 16코어+에서 정체·역행. 4–8워커 최적화는 타협이 아니라 최적점]
- 병렬 단위: ①브로드/내로우페이즈 = 바디쌍 청크, ②솔버 = island 병렬 + **큰 island는 제약
  그래프 컬러링으로 내부 분할**(색상 내 락프리). [근거: Rapier의 island-only 병렬은 조립도가
  커서 큰 단일 더미에서 무력 — Box2D v3 컬러링이 해법]
- 워커 배정은 sticky(캐시 로컬리티) — 브라우저에 affinity 제어가 없으므로 잡→워커 매핑 안정화로 근사.
- **병렬이어도 결과는 결정적** (검수 BLOCKER 반영): 워커 완료 순서가 접촉 배열 순서·부동소수점
  누산 순서를 바꾸면 L0 결정론(§8)이 깨진다. 방어선 — ①접촉/제약은 island finalize에서 pair-key
  정렬(§4.3), ②그래프 컬러링의 색 배정은 정렬된 제약 순서 기반 고정 알고리즘, ③워커 간 누산은
  고정 순서 리덕션. 이 정렬 비용은 L0의 상수 비용이며 벤치에 계측 — "비용 0" 주장은 철회.
  [근거: Jolt가 멀티스레드 결정론을 정렬 패스로 달성하는 것과 동일 접근]

## 6. 성능 규율 (불변식)

1. 스텝 중 힙 할당 0 (bump arena 외). — CI에서 allocation counter로 검증.
2. 핫 루프 자료구조는 전부 SoA + 16바이트 정렬.
3. 티어 강등(SIMD 미지원·스레드 불가·f64 게이트)은 **항상 초기화 리포트에 노출** — 조용한 폴백 금지.
   [근거: PhysX 64버텍스 조용한 CPU 폴백 반면교사]
4. 벤치 하니스(`maru-bench`)는 M0부터 존재: 박스 스택 · 피라미드 · 래그돌 · 컨베이어 · 조인트 체인
   — Chaos가 회귀한 바로 그 워크로드들 + rapier.js/jolt-js/Havok-wasm 비교 러너.
   [근거: Solver2D 방법론 + "인큐번트의 주력 워크로드에서 증명"]

### 6.4 기준 벤치 정의 (성능 목표의 반증 가능 형태 — 검수 반영)

- **씬 "pile-3k"**: 3,000 동적 convex(박스 50%/구 30%/캡슐 20%, 균일 밀도), 정적 그릇 안으로
  낙하 후 정착 — 전 구간 활성(슬리핑 비활성), 정착 구간 평균 접촉 ≥ 3/바디.
- **측정**: 물리 스텝 + JS 트랜스폼 버퍼 발행(§7)까지 포함한 p95 스텝 시간 ≤ **8ms** (60fps 예산
  16.6ms 중 렌더 몫을 남긴 물리 몫). 평균이 아니라 p95 — 스파이크가 게임을 죽인다.
- **기준 환경 고정** (재검수 반영 — "2023년급 미드레인지"는 재현 불가): 벤치 리포지토리의
  `bench-env.lock`에 **구체 장비·조건을 명시하고 그 조건으로만 통과 판정** — 초안: AMD Ryzen 7
  7840U(또는 프로젝트 착수 시점에 고정할 동급 1종), AC 전원 + 고성능 모드, Chrome stable 고정
  버전(락파일에 기록, 분기별 갱신), 워밍업 300스텝 후 1,000스텝 측정, 5회 반복의 p95 중앙값.
  장비 교체는 락파일 갱신 커밋으로만(신구 장비 비교 수치 동봉). iOS Safari(성능 하한 브라우저)는
  별도 트랙으로 수치만 기록.
- 흩어진 3,000개(접촉 0)나 슬리핑 씬으로 통과 주장하는 것을 씬 정의가 차단한다.
- 비교 대상(rapier.js·jolt-js·Havok-wasm)도 동일 씬·동일 측정으로 러너에 포함.

## 7. JS↔WASM 경계 — 본 설계의 승부처

[근거: per-call 경계 왕복 비용의 구조적 실재 — ammo.js 바인더 설계 문서가 per-property 힙 왕복을
회피 대상으로 직접 인정, Havok-Babylon 20× 주장도 ammo.js식 경계 대비. Rapier 2025의 JS 2–5× vs
네이티브 ~2× 격차는 시사적 방증(`[가정]` — 같은 릴리스의 BVH/SIMD 개선과 미분리, 재검수 지적으로
단정 철회). 단정 없이도 결론은 유지된다: flat buffer 설계는 초기 비용이 낮고 후회 없는(no-regret)
선택이며, 반대로 per-call API를 굳히면 되돌리기 어렵다.]

- **핫패스는 함수 호출이 아니라 메모리 뷰.**
  - `world.step()` 후 트랜스폼은 WASM 선형 메모리 위 연속 버퍼(바디 슬롯당 8 f32 = pos3+pad1+quat4)로
    존재. JS는 `Float32Array` 뷰 하나로 전 바디를 읽어 Three.js `InstancedMesh`/개별 오브젝트에 복사.
  - 슬롯 인덱스는 바디 수명 동안 안정(핸들의 index부). 삭제 슬롯은 세대로 무효 표시.
  - 이벤트(접촉 시작/종료, 트리거, sleep 전환)도 flat 이벤트 버퍼 — 스텝당 1회 drain.
- **쓰기도 배치**: 바디 생성/트랜스폼 강제 세팅은 커맨드 버퍼에 쌓아 스텝 전 일괄 적용.
  per-body setter는 존재하되 느린 경로로 문서화. [근거: ammo.js per-property 왕복 반면교사]
- **커맨드 버퍼 ↔ 쿼리 일관성 계약** (검수 반영): 쿼리(ray/overlap)는 **마지막 step이 완료된
  월드 상태**를 본다 — pending 커맨드는 보이지 않는다. 생성/이동 직후 즉시 쿼리가 필요하면
  명시적 `flush_commands()`(브로드페이즈 반영 포함, 배치 삽입 경로 사용)를 호출한다.
  자동 암묵 플러시는 없다 — 일관성 규칙이 조용히 바뀌는 API를 만들지 않는다.
- 보간은 JS 쪽 유틸(prev/curr 버퍼 2개 + alpha)로 제공 — 고정 틱과 렌더 프레임의 분리를 API가 강제.
- 스냅샷: 월드 전체 직렬화 → `Uint8Array` 1회 왕복 (저장/롤백/디버그 전송용). [근거: rapier.js takeSnapshot]
- 패키지: ESM + `-compat`(base64 임베드) 이중 배포는 rapier.js 전례 따름.

## 8. 결정론 — 계약 행렬 (검수 BLOCKER 반영: v0.1의 L1/비목표 상충 해소)

| 계약 | 내용 | 비용 | 용도 |
|---|---|---|---|
| L0 (기본, 항상) | 동일 빌드 + 동일 초기상태 + 동일 호출 순서 + **동일 플랫폼** → 스레드 수와 무관하게 동일 결과 | 정렬 패스 상수 비용 (§5 — "0" 아님) | 리플레이, 디버그 |
| L1-wasm (opt-in) | + libm 강제·FMA/relaxed-simd 비활성 → **모든 wasm 호스트(브라우저·OS·CPU) 간 비트 동일** | 실측 후 공표 (Rapier 전례상 유의미) | 웹 록스텝 넷코드 |
| L1-native (opt-in) | 같은 조건으로 **네이티브 x86 ↔ ARM 간 비트 동일** | 동일 | 네이티브 록스텝, 서버 검증 |
| 비계약 | **wasm ↔ native 교차 비트 일치** | — | 명시적 비목표 (FMA/코드젠 경로 상이) |

- **L1 부동소수점 세부 규칙** (재검수 반영 — libm 강제만으로 부족): NaN canonicalization(비교/분기에
  NaN 유입 차단 — 페이로드 비트 전파는 wasm이 비결정), signed zero 분기 금지, SIMD min/max의 NaN
  시맨틱 명시 구현, 컴파일러 재결합 금지(`-ffast-math` 계열 전면 배제, Rust 기본 준수 + 릴리스
  프로파일 고정), **툴체인 버전 고정**(L1 빌드는 rustc·LLVM 버전을 lockfile로 못 박고 버전 변경 =
  L1 호환성 단절로 취급, 넷코드 프로토콜 버전에 반영).
- **denormal·라운딩 — FP 환경 계약이 1차 방어선** (3~5차 검수 반영: "clamp 지점 전 중간값
  subnormal"은 clamp만으로 못 막는다): **L1 활성 시 초기화에서 FP 환경을 양 플랫폼 모두 명시
  설정·읽기 검증한다** — x86은 MXCSR의 FTZ/DAZ **해제** + round-to-nearest-even, ARM은 FPCR의
  FZ/FZ16 **해제** + round-to-nearest-even을 설정 후 다시 읽어 확인하고, 설정 불가 환경이면
  **L1 초기화가 명시 실패**한다(5차 지적: x86만 다루면 ARM 호스트의 FZ 상태가 계약을 깬다).
  **FP 제어 레지스터는 스레드별 상태다** (홀리스틱 검수 반영 — 초기화 1회로는 부족): 내부 워커는
  생성 시 설정+검증하고, 솔버 작업이 도는 모든 진입점(`step()` 포함)에서 저비용 재검증 가드를
  둔다 — 워커 간 FZ 상태가 갈리면 어느 워커가 어느 제약을 처리했는지에 따라 해시가 갈리기
  때문. 검증 실패 = 명시 오류(조용한 진행 금지). **기준 상태는 모드 공통 단일값이다** (홀리스틱
  2차 반영 — "L0는 무엇과 비교하나" 모호성 제거): L0·L1 구분 없이 엔진의 canonical FP 상태 =
  **RNE + FTZ/DAZ/FZ/FZ16 전부 off**로 고정하고, 월드 생성·워커 생성 시 이 상태로 설정, 모든
  솔버 진입점에서 이 상태와 비교 검증한다. L0와 L1의 차이는 FP 환경이 아니라 libm/FMA 경로뿐.
  ambient 상태 승계는 어떤 모드에도 없다 — 호출자 스레드가 FZ=1이어도 step 진입 가드가 잡아낸다.
  wasm은 IEEE gradual underflow·RNE가 시맨틱으로 고정이라 검증 불요. 이로써 세 클래스 모두
  subnormal·경계 라운딩이 IEEE로 통일된다. 호스트가 FP 환경을 바꿔 놓은 경우(오디오 라이브러리
  등)도 초기화가 잡아낸다. 정의 지점의 **알고리즘적 flush**(|x| < 물리적 무의미 임계 → 0; 속도
  1e-8 m/s 급)와 입력 API의 비정상값(NaN/Inf) 거부는 성능 가드 + 위생 계층으로 유지 — 정확성의
  근거는 FP 환경 계약이다.

- v0.1의 "x86/ARM/wasm 3타깃 비트 동일 CI"는 비목표(wasm↔native)와 상충했으므로 폐기.
  CI는 **클래스 내부 비교**로 재정의: L0 = 매 커밋, 동일 타깃에서 1스레드 vs 4스레드 1,000스텝
  상태 해시 일치. L1-wasm = Chrome/Firefox/Safari(또는 wasmtime x86/ARM) 간 해시 일치.
  L1-native = x86 vs ARM 해시 일치. wasm↔native 비교는 CI에 존재하지 않는다.
- 호출 순서 계약은 문서가 아니라 타입으로: 월드 변이는 커맨드 버퍼 경유가 기본이라 순서가
  버퍼 순서로 고정됨. [근거: Jolt "call-order determinism" + Havok "결정론은 지키는 계약"]
- **직렬화 가능한 캐노니컬 상태와 상태 해시는 M2부터 존재** (검수 반영 — FILE 1 교훈 "롤백 데이터
  모델은 솔버 동결 전에"와 정합하도록 로드맵 §11 수정): PCM 임펄스·슬리핑 카운터·핸들 free-list·
  이벤트 순서까지 포함한 전체 상태가 스냅샷 대상임을 M2 설계에서 확정. **브로드페이즈(트리·세대
  번호)는 캐노니컬 상태·L0 해시에서 제외**(7차 검수 반영 — §3의 "트리=순수 파생 가속 구조" 계약과
  모순 제거: restore 후 세대가 0부터 다시 시작해도 해시가 갈리면 안 된다). 세대 번호는 물리와
  무관한 진단 카운터로만 존재.

## 9. API 표면 (요약)

- Rust: `World` + `BodyBuilder`/`ColliderBuilder`(빌더 패턴) + 핸들. 락킹 개념 없음(단일 소유 월드,
  스레드는 내부 잡에만) — Jolt의 BodyInterface 락 캐스케이드는 "외부 멀티스레드 접근" 요구가 없는
  우리 스코프에선 불필요한 복잡도로 판단, 커맨드 버퍼로 대체. `[가정]` v1은 월드 외부 동시 접근
  비지원 — 요구가 생기면 재설계 분기.
- 필터링: 2계층 — BroadPhaseLayer(소수, 트리 구성 단위) ↔ ObjectLayer(다수, 유저 정의) + 쌍 매트릭스.
  콜백 필터는 최후단. [근거: Jolt 캐스케이드]
- 센서 = 바디 플래그. 캐릭터 컨트롤러 = 솔버 밖 쿼리 기반 `CharacterVirtual` 동형 (v1.1). [근거: Jolt]
- **바디 타입 3종 시맨틱** (검수 반영 — v0.1 누락): `Static` / `Dynamic` / `Kinematic`.
  - Kinematic은 `move_kinematic(target_pose, dt)`로 구동 — 엔진이 목표 자세로부터 스텝 속도를
    유도해 저장하므로, 위에 올라탄 동적 바디가 접촉 상대속도로 플랫폼 속도·마찰을 전달받는다
    (솔버에서 inv-mass 0이지만 속도는 기여). 즉시 `set_pose` 텔레포트는 별도 커맨드로 구분 —
    속도를 만들지 않고 관통 가능성을 호출자가 감수한다는 계약을 이름으로 드러낸다.
  - Kinematic은 동적 트리에 속하고 동적 바디를 깨우며, static-kinematic·kinematic-kinematic 쌍은
    접촉을 생성하지 않는다. CCD는 kinematic에 적용 가능(고속 플랫폼).
- 쿼리: ray/shape cast, overlap, point projection — 브로드페이즈 트리 직접 활용. 일관성 계약은 §7.
- **쿼리 결과 canonicalization** (6차 검수 반영 — 트리를 순수 가속 구조로 유지하려면 쿼리 결과도
  트리 순회 순서에 종속되면 안 된다): all-hits 쿼리는 정밀 교차 판정 후 (거리, stable collider 키,
  feature) 순 정렬로 반환. closest-hit의 거리 동률은 stable collider 키가 낮은 쪽 채택. overlap도
  stable 키 정렬. **point projection도 동일 규칙** (홀리스틱 검수 MINOR 반영): 등거리 동률은
  (거리, stable collider 키, feature)로 확정 — 대칭 형상 사이에서도 결과가 트리 순회와 무관. 호스트가 쿼리 결과로 커맨드를 만들어도 리플레이가 트리 토폴로지·재빌드
  타이밍과 무관하게 재현된다.

## 10. 확장 포인트 (v1에서 예약만)

- 솔버 트레이트 경계: position-based 계열(소프트바디/클로스)이 같은 island/브로드페이즈 인프라를
  공유할 수 있도록 solver 단계만 교체 가능하게.
- WebGPU 백엔드 슬롯: 파티클/대량 디브리 전용 보조 시뮬 아일랜드(게임플레이 비간섭) 한정.
  [근거: readback 지연 — 게임플레이 결합 물리는 CPU 고정]
- 파괴(destruction): Chaos Geometry Collection식 계층 클러스터 + strain 그래프를 조인트 특수형으로
  얹을 수 있는 구조 여지만 확보.

## 11. 로드맵

| 마일스톤 | 내용 | 완료 기준 |
|---|---|---|
| M0 | maru-math + 벤치 하니스 골격 + wasm 빌드 파이프라인 | wasm/native에서 SIMD 수학 벤치 구동 |
| M1 | 브로드페이즈(듀얼 쿼드트리) + convex 셰이프 + GJK/EPA + **클리핑 매니폴드** | 쿼리 정확성 스위트 + **box 면 접촉 4점 매니폴드·feature ID 안정성 테스트** 통과 |
| M2 | TGS-soft 솔버 + PCM + 아일랜드(접촉+조인트 edge) + 슬리핑 + restitution 계약 + **최소 조인트(point·hinge — 접촉+조인트 혼합 수렴을 솔버 동결 전에 검증, 재검수 반영)** + **캐노니컬 상태 직렬화·상태 해시**(단일스레드) | **박스 피라미드 40단 안정**(2026-07-25 실측 재보정 — 아래 주석) + **hinge 체인 20링크+접촉 혼합 수렴 테스트** + **L0 리플레이 해시 CI 녹색** + **정적 받침 삭제 wake 회귀** + **pending 커맨드/미drain 이벤트 경계 스냅샷 restore 재생 일치**(홀리스틱 검수 반영) + 벤치 대비표 1차 |
| M3 | maru-wasm flat buffer 바인딩 + 커맨드/쿼리 일관성 + Three.js 데모 | §6.4 pile-3k 벤치 p95 ≤ 8ms 실측 |
| M4 | 조인트 전체 세트(fixed/prismatic/distance+모터·리밋) + CCD(speculative/linear cast+TOI-hit 큐) + **트라이앵글 메시·heightfield(내부 에지 억제)** + 스레드 모드(그래프 컬러링, 병렬 결정론) | seam-slide 회귀 통과 + 1 vs 4스레드 해시 일치 + **TOI 큐 상한 포화 프레임(할당 0·오버플로 결정성)** + **speculative↔linear-cast restitution 단일 소비** + **compound 다중 콜라이더 동시 임팩트 독립성** + **canonical feature ID 경로 간 일치·ε_t 경계 회귀**(홀리스틱 검수 반영) + rapier.js/jolt-js 비교 벤치 공표 가능 수준 |
| M5 | L1 결정론(wasm/native 클래스별) + 스냅샷 API 마감 + CharacterVirtual | 클래스 내 비트 동일 CI 녹색(§8 행렬 기준) |

> **M2 기준 재보정 (2026-07-25, 구현 실측)**: 최초 기준이던 "피라미드 100단"은 설계 단계에서
> 어떤 레퍼런스와도 대조하지 않고 적은 수치였고, 구현 후 실측에서 현재 솔버 구성(4 서브스텝·
> 접촉 30Hz)의 범위 밖임이 드러났다 — 20/40/50단 안정, 60단 준안정, 100단 붕괴
> (`maru/README.md`에 단수별 침하·ms/step 표). 침하는 soft 접촉의 하중 비례 침투에서 오고
> 단수²에 비례한다. 따라서 M2 게이트는 **40단**으로 재보정하고, 100단급은 솔버 강화(위치 투영·
> 블록 솔버 등) 과제로 M4 이후에 둔다. 이 과정에서 박스-박스 SAT 법선 도입도 시도했으나
> **실측상 순 회귀**(40단 안정→불안정)라 기각했다 — 이론적 정석이어도 측정이 우선.

> 검수 반영 변경점: 매니폴드 생성이 M1으로(스택 안정의 전제), 상태 직렬화·리플레이 해시가
> M5→M2로(솔버 동결 전 — FILE 1 교훈과 정합), 메시/내부 에지가 명시적 M4 항목으로,
> M3 완료 기준이 "3,000바디 60fps"에서 §6.4의 반증 가능 벤치로 교체.

각 마일스톤 ~150줄/3파일 단위로 분해해 진행(공용 정책). 구현 착수는 별도 승인 후.

## 12. 리스크 대장

| 리스크 | 완화 |
|---|---|
| "새 엔진은 수년간 인큐번트에 진다"(Chaos 실증) | 범위를 3D 강체+웹으로 좁힘, M2부터 인큐번트 워크로드 벤치 상시화 |
| TGS-soft 3D 이식 난도 — 확립된 3D 선례 제한적(Box2D v3는 2D, Rapier는 유사 임펄스 계열, Avian 계열 구현 참고 — 01 §2 정정) | Rapier·Avian 소스 리딩 + Solver2D 방법론으로 솔버만 격리 검증, M2 혼합 수렴 벤치가 게이트 |
| WASM 스레드 보급 지연 | 단일스레드가 1급 — 스레드는 승수로만 설계 |
| Rapier가 같은 방향으로 개선 중(2026 로드맵: GPU·로봇틱스) | 차별화 축 = JS 경계 + 웹 기본값 + Jolt식 브로드페이즈. Rapier가 못 깨는 구조적 선택임 |
| 1인 유지보수(bus factor — Rapier도 같은 문제) | 코어 최소주의, 확장은 트레이트 경계 밖으로 |
