# Maru vs 외부 엔진 대조 벤치

엔진 워크스페이스 **밖**의 별도 크레이트다(`exclude = ["compare"]`) — 외부 엔진을
엔진 의존성 그래프와 CI 게이트에 들이지 않기 위해서다.

```bash
cargo run --release --manifest-path compare/Cargo.toml -- [pyramid|pile|all] [steps] [warmup]
```

## 원칙

- **성능만 재지 않는다.** 스택이 무너지는 엔진은 접촉이 사라져 더 빨라진다.
  매 시나리오에서 안정성 판정(최종 높이·최대 속도·유한성·관통)을 함께 내고,
  UNSTABLE이면 그 성능 수치는 무효로 읽는다.
- **각자의 기본 설정으로** 돌린다. 서로의 파라미터를 맞추면 한쪽 튜닝을 다른 쪽에
  강제하는 셈이라 오히려 왜곡된다. 설정은 출력 헤더에 명시한다.
- **기능 차이를 숨기지 않는다.** Maru는 M2(단일 스레드·CCD 없음·셰이프 4종)이고
  Rapier는 CCD·쿼리 파이프라인을 함께 돌린다. "같은 일을 하는" 비교가 아니다.

## Jolt 5.0 대조 (`--bin jolt`)

```bash
# 빌드 요구사항 (한 번):
#   uv tool install cmake                       # cmake 없는 환경
#   uv venv <dir> && uv pip install libclang    # bindgen용 libclang
#   export LIBCLANG_PATH=<dir>/lib/python*/site-packages/clang/native
#   export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
cargo run --release --bin jolt -- [n] [steps] [reps]
```

### 공정성 캐비앳 (2026-07-28 실측에서 확인)
- **워커 0(단일 스레드)** — Jolt의 본령은 멀티스레드 스케일링이라 이 수치는 Jolt의
  최선이 아니다. musoe·rapier 대조가 전부 단일 스레드라 조건을 맞춘 것.
- **정착 수치는 엔진 비교가 아니라 기본 튜닝 비교다**: Jolt 기본 수면 임계(0.03 m/s)는
  큰 구 더미의 잔진동을 못 넘겨 n=1500·3000이 240+60스텝 내 수면 0 — JoltC 0.3.1
  바인딩에 임계 조정 API가 없어 하네스 공통값(0.1)으로 맞출 수 없었다. n=500은 Jolt도
  전원 수면 + ~0.000 ms/step(수면은 어느 엔진이나 공짜다).
- Jolt 미정착 더미는 장기(600스텝+)에서 에너지가 자라 붕괴했다(lowest_y −153) —
  이 구성이 Jolt에 불리한 병적 케이스일 수 있어 **처리량 수치도 참고용**으로만.
- 접촉 제약 버퍼는 temp allocator에서 나온다 — max_contact_constraints × ~0.5KB가
  할당자보다 크면 첫 Update에서 abort(실측: 65536 × 0.5KB > 32MB). 20480 + 64MB로 설정.
