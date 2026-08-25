# 기존 물리엔진 비교 분석

> 2026-07-24. 신규 Rust/WASM 물리엔진 설계(`02-architecture.md`)의 근거 문서.
> 조사 방법: 병렬 웹 리서치 5트랙 (PhysX/Unity · Chaos/Havok · Jolt · Rapier/Box2D v3 · WASM 제약).
> 각 주장 옆 출처는 트랙 리포트에 수집된 1차 자료(공식 문서·GDC 발표·개발자 포럼) 기준.

## 1. 한눈 비교

| | PhysX 5 | Chaos (UE5) | Havok | Jolt | Rapier | Box2D v3 |
|---|---|---|---|---|---|---|
| 솔버 | TGS (substep PGS + 위치 바이어스) | 반복 GS 계열, 저반복 안정성 튜닝 | PGS + Baumgarte | Sequential Impulse + soft spring (5.x 기본 10vel+2pos; 4+1은 GDC 벤치 설정) | 임펄스 속도 + 비선형 위치 솔버 (TGS-soft 명시 채택은 GPU 실험 wgrapier — 07-25 정정) | Soft Step (TGS-soft 원조) |
| 브로드페이즈 | SAP/MBP/ABP/PABP + GPU SAP | (비공개 상세) | 교체형 파이프라인 | 듀얼 쿼드트리(정적/동적), SoA+atomic widen-only | 동적 BVH(2025 재작성, SIMD 순회) | 동적 AABB 트리(b2DynamicTree) + persistent islands |
| 내로우페이즈 | PCM (persistent manifold) | — | manifold 기반 | GJK+EPA + convex radius | parry (GJK/EPA, manifold 축소 4점) | 전용 2D 루틴 |
| 아일랜드 | 의존 그래프, GPU 파이프라인에서도 CPU 잔류 | — | — | **락프리 union-find를 내로우페이즈 중에 구축** | IslandManager + persistent islands(2025) | persistent islands + 스레드 분할 |
| 스레딩 | 태스크 그래프(스스로 "완전 병렬 아님" 인정) | 전용 물리 스레드(고정 틱) | (성숙, SPU 시절 DMA 규율) | 잡 그래프, 8스레드 4.9×, 16+에서 정체 | rayon island 단위(조립도 큼) | **제약 그래프 컬러링**(bepu식) + enkiTS |
| 결정론 | 기본 없음. 동일 기종 한정 enhanced 모드 | 고정 틱 기반, 롤백엔 "부족" 보고 | **플랫폼 간 계약으로 보장** | 동일 빌드+**호출 순서 계약** | `enhanced-determinism`(libm 강제, 유료 성능) | (2D, 유사) |
| WASM | 없음 | 없음 | **Babylon 플러그인, ~3,000바디@60fps 단일스레드** | jolt-physics-js (실측 공표 없음) | @dimforge/rapier3d — JS 경계 최적화가 최대 레버였음 | wasm 빌드 가능 |

## 2. 엔진별 핵심 교훈

### PhysX 5 (유니티 기본 물리의 코어)
- **채택**: TGS 서브스텝(PhysX 4→5 마이그레이션 자체가 검증), PCM 접촉 캐시, ABP류 자동 브로드페이즈(사용자 튜닝 불요).
- **회피**: GPU 파이프라인의 이원화 — GPU 모드에서도 CCD·트리거·비D6 조인트·아일랜드 생성은 CPU 잔류, 볼록체 64버텍스 초과 시 **조용한 CPU 폴백**. 기능이 가속 티어를 넘나들며 조용히 강등되는 설계는 금지.
- **회피**: 결정론을 창발적 속성으로 방치 — 동일 기종에서만 성립, Intel/AMD 간 부동소수점 차이로 붕괴.
- 유니티 통합의 교훈: `FixedUpdate` 캐치업 커플링이 스터터 원인 → **시뮬 케이던스와 호스트 프레임 케이던스는 분리**된 서브스텝 API로.
- Unity Physics(DOTS)가 PhysX를 안 쓰고 stateless로 재작성한 이유 = 롤백 넷코드·저장/복원 단순성. **stateful 캐시(성능) vs stateless(롤백)는 둘 다 실수요** — 한쪽만 강제하지 말 것.

### Chaos / Havok (언리얼의 현재와 업계 금본위)
- Chaos는 "아키텍처가 종이 위에서 옳아도" 성숙 엔진 대체 시 수년간 회귀를 겪는다는 실증: 동일 씬 3~30× 느림 보고, 래그돌 5.30ms(PhysX)→8.18ms(Chaos), 마찰 거동 변경으로 전 머티리얼 재튜닝. 상당 부분은 알고리즘이 아니라 **기본값·튜닝 부채** 문제.
  - → 신규 엔진 교훈: **보수적이고 잘 검증된 기본값**이 아키텍처만큼 중요. 성능 주장은 인큐번트의 주력 워크로드(스택·래그돌·차량)에서 벤치로 증명.
- Chaos의 옳은 결정: 고정 틱 전용 물리 스레드를 결정론·네트워크 예측의 토대로 처음부터 배치. 단 롤백 컴포넌트(5.4+)가 솔버 동결 후 늦게 도착해 결정론 갭이 남음 → **리플레이/롤백 데이터 모델은 솔버 동결 전에 1급 API로**.
- Havok: 결정론을 "지키는 계약"으로 명시 보장, PS3 SPU(256KB 로컬스토어) 시절의 데이터 레이아웃 규율이 경쟁력의 근원. **WASM 선형 메모리 모델과 정확히 동형의 교훈** — 캐시/로컬리티 규율은 레이아웃 수준에서 시작.
- Havok CCD: 오브젝트별 opt-in — 전역 CCD 금지.

### Jolt (현대 오픈소스 최강 레퍼런스)
- **채택 (거의 전부)**:
  - 듀얼 브로드페이즈 트리(정적=거대·희소 재빌드 / 동적=매 스텝 재빌드) — 변이 빈도가 다른 것을 한 트리에 넣지 않는다.
  - 쿼드(4-ary) 트리 + 바운드 SoA `float[4]` = SIMD 4레인 순회. **WASM SIMD128(=4×f32)과 정확히 일치**하는 구조.
  - widen-only 원자적 바운드 갱신 + 백그라운드 타이트 재빌드 + 세대 스왑 → 쿼리가 구조 변경에 블록되지 않음.
  - **락프리 union-find 아일랜드를 내로우페이즈 병렬 구간 중에 구축** — 병렬 페이즈 사이의 직렬 병목을 80% 절감. (단일스레드 모드에선 그냥 union-find로 자연 강등. 출처 귀속: GDC 2022 발표 기준 — 공식 Architecture.md 문면은 "충돌 감지 후 finalize"로 기술하며 증분 union 시점을 명시하지 않음, 2026-07-25 재검증)
  - 스텝 스크래치는 arena `TempAllocator`로 프레임 단위 일괄 해제.
  - convex radius(수축 후 재팽창) 1급 파라미터.
  - API: 2계층 레이어 필터(BroadPhaseLayer↔ObjectLayer), 필터를 싼 것→비싼 것 캐스케이드, sensor=바디 플래그, CharacterVirtual=솔버 밖 쿼리 기반 키네마틱, **배치 삽입만 fast path**(단건 삽입이 Guerrilla의 5–10ms 스트리밍 스톨 원인이었음).
  - 결정론 = "동일 빌드 + 동일 호출 순서" 계약으로 명시. f64는 위치에만(`RVec3`), 5–10% 비용, feature 게이트.
- **스케일링 데이터**: 8스레드 4.9×, 16스레드(SMT) 5.7×, 그 이상 정체·역행(메모리 버스·원자 접근 포화), CCX 경계 27% 오버헤드. → **WASM의 4–8워커 상한은 실질적 손해가 아니다.** 잡 그래프를 4–8워커에 최적화하면 됨.
- 실전 성과(HFW): 뮤텍스 대기 500µs→10µs/스레드, 30Hz→2×60Hz를 더 적은 CPU로.

### Rapier / Box2D v3 / Avian (Rust 생태계 + 최신 솔버 연구)
- **솔버 수렴의 근거** (07-25 사실검증 정정 — v0.19까지 "3중 독립 검증"으로 서술했으나 Rapier
  부분 과단정): 확실한 근거 = Erin Catto의 Solver2D 실험 + Box2D v3 "Soft Step" 채택 + Avian의
  접촉 솔버 이주(XPBD → 임펄스+substep+soft, 사유: 마찰 강건성 취약·GS 해석 순서 민감성·선행
  사례 부족·특허 리스크; 조인트는 XPBD 잠정 유지 — 접촉 한정 기각). **Rapier는 유사 계열로만
  취급**: 공식 사용자 문서는 CPU 솔버를 "임펄스 속도 솔버 + 비선형 위치 솔버"로 기술하며
  TGS-soft로 명명하지 않고, Soft-TGS 명시는 GPU 실험 wgrapier다. → 신규 엔진의 **접촉 솔버는
  TGS-soft** — Solver2D 실증과 Box2D v3·Avian 채택이 근거로 충분. 조인트 정식화는 별도 결정.
- Box2D v3: 바디 솔버 상태를 SIMD 레인 폭에 패킹(2D는 32바이트=AVX2 1레지스터). 원리 이식: **3D 솔버 상태를 SIMD128 레인에 맞춰 패킹**. 스레딩은 **제약 그래프 컬러링**으로 색상 내 락프리 병렬 — Rapier의 조립도 큰 rayon island 병렬보다 세밀.
- Rapier 약점(우리가 넘을 지점): 브로드페이즈가 성능 문제의 뿌리였다고 자인(Jolt 대비 ~2× 열세 벤치 존재), f32/f64 포크 크레이트로 유지보수 2배, island 단위 병렬의 조립도.
- Rapier `enhanced-determinism`의 정확한 한계(07-25 정정): 공식 문서상 **`parallel`·`simd-stable`·`simd-nightly`와 동시 활성 불가** — "병렬+SIMD+결정론 동시 달성"의 선례가 아니다. 우리 L1은 자체 정렬·고정 리덕션·FP 계약으로 병렬 결정론을 별도 설계하며(02 §5·§8), Rapier를 그 선례로 인용하지 않는다.
- **Rapier의 시사적 데이터**: 2025년 JS 패키지가 "전년 대비 2–5× 고속화"인데 네이티브는 ~2×에 그침. `[가정]` 이 격차의 상당분이 JS↔WASM 경계 개선에서 나왔다고 추정되나, 같은 릴리스의 동적 BVH·SIMD 개선 효과와 분리된 근거는 없다(GPT 검수 지적으로 단정 철회). 경계를 flat buffer로 설계해야 한다는 결론 자체는 별도 근거로 성립 — per-call 왕복 비용은 ammo.js 바인더 설계 문서가 직접 인정하는 문제고, Havok WASM의 20× 주장도 ammo류 경계 대비다.
- Rust 공학: 그래프 상태는 참조 대신 **generational-index 핸들 + flat arena**(생태계 만장일치), SIMD는 nightly `std::simd` 말고 **stable `wide`**, unsafe는 SIMD gather/scatter·arena에만 국소화.

### WASM 제약 봉투 (2025–2026)
- SIMD128 고정(256비트 없음), FMA는 relaxed-simd에서만 (Chrome 114+/FF 120+; Safari는 18.4에서 relaxed_laneselect만 공식 명시 — **명령별 지원이 상이하므로 버전표가 아니라 런타임 기능 감지가 정본**, 07-25 정정) → **비FMA 정확성 기본, FMA는 기능 감지 후 가속**.
- 스레드 = SharedArrayBuffer + COOP/COEP 격리 필수. 서드파티 임베드 환경에서 흔히 불가 → **단일스레드가 1급 시민, 스레드는 보너스**.
- memory64는 Safari 안정판 포함 **교차 브라우저 배포 기준 미달**(WebKit 구현은 진행 중 — "미정"은 07-25 정정) + 있어도 종종 wasm32보다 느림 → 4GB 선형 메모리 상한을 설계 전제로(결론 불변).
- JS 경계: flat SoA typed-array 뷰(rapier.js), per-property 호출 왕복 회피(ammo.js 반면교사). 렌더 동기화 = 고정 틱 + 보간, 트랜스폼은 연속 버퍼 1회 기록/뷰 읽기.
- **실측 기준선: Havok WASM(Babylon) ~3,000 동적 바디 @60fps 단일스레드** — 공개 보고 기준 대표 실측(커뮤니티 샘플 보고라 측정 조건 비공개, "현존 최고" 단정은 07-25 철회). 이것이 비교 대상이며, 정본 계약은 02 §6.4의 pile-3k 벤치다.
- 수치 신뢰도 주석(07-25 사실검증): 본 문서의 Chaos 3–30×·래그돌 ms·Jolt 스케일링·HFW 개선 수치는 커뮤니티 보고/벤더 발표 기반으로 측정 조건이 제각각(동일 조건 비교 아님) — 방향성 근거로만 사용하며, 우리 성능 주장은 전부 §6.4 기준 벤치로만 한다.
- WebGPU 물리(AVBD 등): **CPU 가시 결과**(게임 로직·네트워킹·CPU 충돌 응답)에는 비동기 readback 동기화 비용·지연이 구조적(07-25 정정 — GPU 안에서 렌더링으로 직접 소비하는 경로는 예외지만, 그 경로는 게임플레이 결합 물리가 아님) → 보조/실험 백엔드로만.

## 3. 종합: 설계를 지배하는 7개 결론

1. **접촉 솔버는 TGS-soft.** 근거 = Solver2D 실증 + Box2D v3·Avian 채택(Rapier는 유사 계열 — §2 정정). XPBD는 접촉에서 기각(Avian이 대신 수업료 지불; 조인트는 별도 결정).
2. **브로드페이즈는 Jolt 듀얼 쿼드트리 이식.** 4-ary SoA가 SIMD128과 정확히 동형 — WASM에서 네이티브만큼의 구조적 이점.
3. **JS 경계는 1급 설계 대상.** flat buffer 뷰 + 배치 API만 핫패스. 근거는 per-call 왕복 비용의 구조적 실재(ammo.js 반면교사, Havok WASM 설계) — Rapier 2025 데이터는 시사적 방증(§2 참조, 단정 아님).
4. **단일스레드 기본, 4–8워커 조건부.** Jolt 스케일링 데이터상 16+코어는 네이티브에서도 무익 — WASM 상한은 실질 페널티가 아님.
5. **결정론은 3단 계약으로 명시.** (같은 빌드+호출 순서) / (크로스플랫폼 opt-in) / (wasm↔native 비트일치는 비목표 선언).
6. **기본값·튜닝이 아키텍처만큼 중요.** Chaos 회귀의 교훈. 벤치 하니스를 M0부터.
7. **조용한 폴백 금지.** 티어 강등(SIMD·스레드·정밀도)은 항상 감지 가능하게.
