#!/usr/bin/env node
// 브라우저 wasm 실측 드라이버 — web/bench.html을 헤드리스 크로미엄에서 돌리고 결과를 회수한다.
//   node scripts/browser-bench.mjs
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

// 호스트에 크로미엄 의존 라이브러리가 없으면 Docker playwright 이미지로:
//   docker run --rm --network host -v "$PWD":/maru -v /home/icyou/workspace/verify/node_modules:/deps \
//     mcr.microsoft.com/playwright:v1.60.0-noble bash -lc \
//     'PLAYWRIGHT_CORE=/deps/playwright-core/index.mjs CHROMIUM=$(ls /ms-playwright/chromium-*/chrome-linux/chrome|head -1) node /maru/scripts/browser-bench.mjs'
const { chromium } = await import(
  process.env.PLAYWRIGHT_CORE ?? '/home/icyou/workspace/verify/node_modules/playwright-core/index.mjs'
);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM ??
    `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
  headless: true,
});
const page = await browser.newPage();
page.on('console', m => console.error('[page]', m.text()));
page.on('pageerror', e => { console.error('[pageerror]', e.message); process.exitCode = 1; });
await page.goto(`http://127.0.0.1:${port}/bench.html`);
await page.waitForFunction(() => document.title === 'DONE', null, { timeout: 300_000 });
const results = await page.evaluate(() => window.__RESULTS__);
console.log(JSON.stringify(results, null, 2));
await browser.close();
server.close();
