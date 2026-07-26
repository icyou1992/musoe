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
else
  echo "!! native 레그 생략: C 링커(cc) 없음 — sudo apt-get install -y build-essential 후 재실행"
fi

echo "ALL GATES PASSED"
