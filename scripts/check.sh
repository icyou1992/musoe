#!/usr/bin/env bash
# maru 품질 게이트 — M0: fmt + clippy + 테스트 + 벤치(wasm 우선, native는 툴체인 있으면).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

echo "== fmt =="
cargo fmt --check

echo "== clippy (wasm32-wasip1) =="
cargo clippy --target wasm32-wasip1 --all-targets -- -D warnings

echo "== test (wasm32-wasip1, Node WASI 러너) =="
cargo test --target wasm32-wasip1

echo "== 브라우저 wasm 빌드 (wasm32-unknown-unknown) =="
# 바인딩은 **브라우저 타깃으로도 빌드돼야** 한다 — wasip1만 통과하면
# WASI import가 딸린 산출물이 브라우저에서 조용히 죽는다.
cargo build --release -p maru-wasm --target wasm32-unknown-unknown

echo "== bench (wasm32-wasip1, SIMD128 on) =="
cargo run --release --target wasm32-wasip1 -p maru-bench --bin maru-bench

if command -v cc >/dev/null 2>&1; then
  echo "== test (native) =="
  cargo test
  echo "== 유효질량 비트 계약 (SIMD 경로 = 스칼라 경로) =="
  cargo test --release -p maru-dynamics effective_mass_x3
  # 측정 노브(`MARU_SCALAR_EFFMASS`)의 **스칼라 팔도 게이트를 지나야** 한다. 노브는
  # 프로세스당 1회 캐시되므로 기본 팔과 같은 프로세스에서는 절대 실행되지 않는다 —
  # 별도 프로세스가 아니면 "두 팔 모두 결정적"이라는 주장에 증거가 없다
  # (교차검수 rev_4976e200 MAJOR).
  echo "== 결정성 (스칼라 팔, 별도 프로세스) =="
  MARU_SCALAR_EFFMASS=1 cargo test --release -p maru-dynamics -- --ignored thread_determinism
  echo "== bench (native) =="
  cargo run --release -p maru-bench --bin maru-bench
  # 설계 §6 "스텝 중 힙 할당 0" 래칫. 계약은 아직 0이 아니므로(README '알려진 한계')
  # 게이트가 잠그는 것은 **지금보다 나빠지지 않는다**다. 상한 740은 실측 673 + ~10% 여유.
  # 스레드는 1로 못는다 — 참여자별 풀이 늘면 수치가 흔들려 래칫이 잡음을 잡는다.
  echo "== 할당 계약 래칫 (native) =="
  MARU_THREADS=1 cargo run --release -p maru-bench --bin allocs -- pyramid 20 40 20 nosleep 740
else
  echo "!! native 레그 생략: C 링커(cc) 없음 — sudo apt-get install -y build-essential 후 재실행"
fi

echo "ALL GATES PASSED"
