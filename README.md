# Maru (가칭) — 웹 우선 3D 강체 물리엔진

설계 정본: `../physics-engine-design/02-architecture.md` (v0.21, GPT 교차검수 23태스크 approve).

## 상태: M0 (math 코어 + 벤치 하니스 + wasm 파이프라인)

| 크레이트 | 내용 |
|---|---|
| `maru-math` | 스칼라 `Vec3`/`Quat`(API 표면) + `Vec3x4`/`Aabbx4`(SoA 4-wide, 핫패스 정본 — 설계 §3) |
| `maru-bench` | 마이크로벤치 하니스. native/wasm 동일 소스, 체크섬=결정론 스모크 |

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
