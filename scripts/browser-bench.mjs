#!/usr/bin/env node
// 브라우저 wasm 실측 드라이버 — web/bench.html을 헤드리스 크로미엄에서 돌리고 결과를 회수한다.
//   node scripts/browser-bench.mjs                          # musoe
//   BENCH_PAGE=bench-rapier.html node scripts/browser-bench.mjs   # 같은 씬의 rapier.js 판
// 의존: workspace/verify/node_modules/playwright-core + ~/.cache/ms-playwright 크로미엄.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');
const MIME = { '.html': 'text/html', '.wasm': 'application/wasm', '.js': 'text/javascript' };
const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/bench.html' : req.url.split('?')[0];
    const body = await readFile(join(root, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

// Playwright 는 워크스페이스 공용(tools/pw) 하나만 쓴다. 브라우저는 그 패키지가 아는
// ~/.cache/ms-playwright 리비전. Docker 로 돌릴 때는 공용 node_modules 를 /deps 로 마운트:
//   docker run --rm --network host -v "$PWD":/musoe -v /home/icyou/workspace/tools/pw/node_modules:/deps:ro \
//     mcr.microsoft.com/playwright:v<공용 버전>-noble bash -lc \
//     'PLAYWRIGHT_CORE=/deps/playwright-core/index.mjs CHROMIUM=$(ls /ms-playwright/chromium-*/chrome-linux/chrome|head -1) node /musoe/scripts/browser-bench.mjs'
const { chromium } = await import(
  process.env.PLAYWRIGHT_CORE ?? '/home/icyou/workspace/tools/pw/node_modules/playwright-core/index.mjs'
);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM ?? chromium.executablePath(),
  headless: true,
});
const page = await browser.newPage();
page.on('console', m => console.error('[page]', m.text()));
page.on('pageerror', e => { console.error('[pageerror]', e.message); process.exitCode = 1; });
// 페이지는 환경변수로 고른다 — `BENCH_PAGE=bench-rapier.html`이면 같은 씬을 rapier.js로 돈다.
await page.goto(`http://127.0.0.1:${port}/${process.env.BENCH_PAGE ?? 'bench.html'}`);
await page.waitForFunction(() => document.title === 'DONE', null, { timeout: 300_000 });
const results = await page.evaluate(() => window.__RESULTS__);
console.log(JSON.stringify(results, null, 2));
await browser.close();
server.close();
