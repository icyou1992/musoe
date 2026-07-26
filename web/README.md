# Maru 웹 데모

```bash
cargo build --release -p maru-wasm --target wasm32-unknown-unknown
cp ../target/wasm32-unknown-unknown/release/maru_wasm.wasm .    # web/ 안에서
python3 -m http.server 8080
# http://localhost:8080/
```

`wasm-bindgen`을 쓰지 않는다. 브라우저가 `WebAssembly.instantiateStreaming`으로 바로 올리고,
JS는 원시 C ABI를 부른다. **핫패스는 함수 호출이 아니라 선형 메모리 뷰다** —
프레임당 `maru_step` 한 번 + `Float32Array` 한 번 읽기가 전부다(설계 §5).

- `maru_transforms_ptr`은 **`maru_step` 뒤에 다시 물어봐야 한다**. 바디 수가 바뀌면
  버퍼가 재할당돼 포인터가 무효화된다.
- 핸들은 `u64`(하위 32=슬롯, 상위=세대). JS Number는 2^53까지 정확해 그대로 다뤄도 된다.
- 실패는 **음수 오류 코드**다. wasm에서 unwind는 곧 trap이라 호스트가 진단할 수 없어서,
  바인딩은 엔진의 fallible 경로(`try_*`)만 쓴다.

클릭하면 그 방향으로 광선을 쏘고 맞은 공을 위로 튕긴다 — 질의·충격 경로를 눈으로 확인하는 용도다.
