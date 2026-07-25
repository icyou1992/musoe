# Maru (가칭) — 웹 우선 3D 강체 물리엔진

설계 정본: `../physics-engine-design/02-architecture.md` (v0.21, GPT 교차검수 23태스크 approve).

## 상태: M2 (강체 동역학 — 솔버·조인트·슬리핑·스냅샷)

| 크레이트 | 내용 |
|---|---|
| `maru-math` | 스칼라 `Vec3`/`Quat`/`Mat3` + `Vec3x4`/`Aabbx4`(SoA 4-wide, 핫패스 정본 — 설계 §3) |
| `maru-collide` | 셰이프(convex radius) · 듀얼 4-ary 트리 브로드페이즈(트리 비의존 canonical 페어) · GJK/EPA · 클리핑 매니폴드(feature ID) |
| `maru-dynamics` | TGS-soft 솔버 · PCM warm-start · 아일랜드/슬리핑 · point·hinge 조인트 · 캐노니컬 스냅샷/L0 리플레이 해시 |
| `maru-bench` | 마이크로벤치 + 피라미드 스택 벤치. native/wasm 동일 소스, 체크섬=결정론 스모크 |

### 측정된 스택 안정 봉투 (2026-07-25, native release, 4 서브스텝 @60Hz)

`cargo run --release -p maru-bench --bin pyramid100 -- <substeps> <steps> <tiers>`

| 피라미드 단수 | 바디 수 | 결과 | 최종 침하 | ms/step |
|---|---|---|---|---|
| 20 | 210 | 안정 | 0.04 m | 1 |
| 40 | 820 | 안정 | 0.14 m | 23 |
| 50 | 1275 | 안정 | 0.22 m | 35 |
| 60 | 1830 | 준안정(침하 지속) | 0.32 m | 51 |
| 100 | 5050 | **붕괴** | — | 141 |

침하는 soft 접촉의 하중 비례 침투(하단이 5000 박스 무게를 받음)에서 오며, 단수²에 비례한다.
100단은 현재 솔버 구성(4 서브스텝·30Hz 접촉)의 범위 밖 — 개선 방향은 M4 이후 과제로 둔다
(측정 근거 없는 목표를 남기지 않기 위해 설계 문서의 M2 기준도 이 데이터로 갱신했다).

## 빌드·실행

```bash
scripts/check.sh                                    # fmt+clippy+test+bench 전체 게이트
cargo test --target wasm32-wasip1                   # wasm 테스트 (Node WASI 러너)
cargo run --release --target wasm32-wasip1 -p maru-bench   # wasm 벤치 (SIMD128)
RUSTFLAGS="" cargo run --release --target wasm32-wasip1 -p maru-bench  # SIMD off 대조
```

- wasm 타깃은 `.cargo/config.toml`에서 `+simd128` 강제. 러너는 `scripts/wasi-run.mjs`(Node 20 WASI).
- 네이티브 빌드는 C 링커 필요: `sudo apt-get install -y build-essential`.

## 규율 (설계 문서에서 상속)

- FMA 비의존(relaxed-simd는 추후 별도 feature) · 재결합 금지(기본 프로파일 유지)
- SIMD 강등은 벤치 배너에 `simd-OFF`로 표시 — 조용한 폴백 금지(설계 §6)
- 벤치는 체크섬을 함께 출력 — 같은 타깃·빌드에서 체크섬 불일치 = L0 결정론 위반 신호(설계 §8)
