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
