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

# 계측 팔은 기본 빌드에서 빠져 있으므로(피처 off) 위 clippy가 **그 코드를 보지 않는다** —
# 따로 켜서 훑지 않으면 조용히 썩는다.
echo "== clippy (계측 피처 — 팔이 썩지 않게) =="
cargo clippy --target wasm32-wasip1 -p maru-bench --features pcm-probe,solve-probe -- -D warnings

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
  # 게이트가 잠그는 것은 **지금보다 나빠지지 않는다**다.
  # PCM 캐시를 자리에서 채우게 바꾼 뒤 673 → 84로 떨어졌다 — **래칫도 같이 조인다**.
  # 안 조이면 래칫이 옛 수준까지 8배 퇴행을 통과시킨다(래칫의 값은 현재 수준에 붙어 있을 때 난다).
  # 선점 없이 자라던 컬렉션들을 fallible 블록으로 올릴 때마다 떨어졌다: 673 → 84 → 65 → 59.
  # 상한 68은 실측 59 + ~15% 여유.
  # 스레드는 1로 못는다 — 참여자별 풀이 늘면 수치가 흔들려 래칫이 잡음을 잡는다.
  echo "== 할당 계약 래칫 (native) =="
  MARU_THREADS=1 cargo run --release -p maru-bench --bin allocs -- pyramid 20 40 20 nosleep 68
  # **정착 장면은 래칫이 아니라 0이다** — 여기는 계약이 이미 만족돼 있으므로 상한이 아니라
  # 계약 그대로 잠근다. 위의 nosleep 래칫과 **다른 경로**를 지킨다: 페어 캐시 hit·전원 휴면
  # fast path는 nosleep에서 아예 안 밟힌다. 실제로 브로드페이즈 목록 선점을 캐시 판정보다
  # **앞에** 두었을 때 nosleep 래칫은 59.1로 멀쩡한데 정착만 0 → 2로 퇴행했고, 그 회귀는
  # 교차검수(rev_74d7e4ff BLOCKER)가 잡았다. 게이트가 없어서 놓친 것이므로 여기 붙인다.
  echo "== 할당 계약: 정착 장면은 스텝당 0 (native) =="
  MARU_THREADS=1 cargo run --release -p maru-bench --bin allocs -- pyramid 12 600 40 sleep 0
else
  echo "!! native 레그 생략: C 링커(cc) 없음 — sudo apt-get install -y build-essential 후 재실행"
fi

echo "ALL GATES PASSED"
