// cargo runner: wasm32-wasip1 바이너리를 Node WASI(preview1)로 실행.
// .cargo/config.toml [target.wasm32-wasip1] runner 에서 호출된다.
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
import process from 'node:process';

const wasmPath = process.argv[2];
if (!wasmPath) {
  console.error('usage: wasi-run.mjs <module.wasm> [args...]');
  process.exit(2);
}

const wasi = new WASI({
  version: 'preview1',
  args: process.argv.slice(2),
  env: {},
  preopens: {},
});

const wasm = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(wasm, wasi.getImportObject());
process.exitCode = wasi.start(instance);
