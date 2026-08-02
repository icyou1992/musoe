# Maru 웹 데모

```bash
cargo build --release -p musoe-wasm --target wasm32-unknown-unknown
cp ../target/wasm32-unknown-unknown/release/musoe_wasm.wasm .    # web/ 안에서
python3 -m http.server 8080
# http://localhost:8080/
```

`wasm-bindgen`을 쓰지 않는다. 브라우저가 `WebAssembly.instantiateStreaming`으로 바로 올리고,
JS는 원시 C ABI를 부른다. **핫패스는 함수 호출이 아니라 선형 메모리 뷰다** —
프레임당 `musoe_step` 한 번 + `Float32Array` 한 번 읽기가 전부다(설계 §5).

- `musoe_transforms_ptr`은 **`musoe_step` 뒤에 다시 물어봐야 한다**. 바디 수가 바뀌면
  버퍼가 재할당돼 포인터가 무효화된다. 성공한 add/remove/set_pose도 이전 포인터를
  무효화한다(재할당이 없어도 값이 변이 이전 스냅샷이다) — 다시 취득하면 자동
  동기화된다(오류 코드가 필요하면 `musoe_sync_transforms`). **포인터는 null 검사
  후에 써야 한다** — 동기화 실패는 null이고, null을 base 0으로 읽으면 선형 메모리
  시작부가 좌표로 보인다.
- 핸들은 `u64`(하위 32=슬롯, 상위=세대) — **JS에서는 Number가 아니라 `BigInt`다.**
  WebAssembly JS API가 i64를 BigInt로 매핑하므로 `Number(handle)`로 저장했다 되넘기면
  TypeError가 난다. 불투명 BigInt로만 보관·전달하고, 실패 센티널 `u64::MAX`는 `-1n`으로
  관측된다.
- 실패는 **음수 오류 코드**다. wasm에서 unwind는 곧 trap이라 호스트가 진단할 수 없어서,
  바인딩은 엔진의 fallible 경로(`try_*`)만 쓰고 스테이징 버퍼 성장도 전부 `try_reserve`다
  (실패 = `ERR_STAGING_LIMIT(-7)`, 엔진 상태는 정상·버퍼만 stale). 복구는
  `musoe_sync_transforms`/`musoe_sync_events` — 이벤트 원본은 다음 스텝까지 월드에 살아
  있으므로 **다음 스텝 전에** 동기화하면 유실이 없다.
- 이벤트 버퍼는 **이번 호출이 실행한 스텝**의 이벤트다. 실행되지 않은 스텝
  (`ERR_INVALID_VALUE`)은 버퍼를 비운다 — 이전 Started를 재발행하지 않는다.
  이벤트 getter는 stale이면 자동 동기화하고, 실패하면 ptr=null·len=0이다(이전
  이벤트를 이번 것처럼 중복 소비하는 것보다 관측 가능한 실패가 낫다).

클릭하면 그 방향으로 광선을 쏘고 맞은 공을 위로 튕긴다 — 질의·충격 경로를 눈으로 확인하는 용도다.
