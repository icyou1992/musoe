#!/usr/bin/env node
// city.html 데모 실구동 검증 — 헤드리스 크로미엄에서 로드→정착→포격→수치·스크린샷.
//   docker run --rm -v "$PWD":/maru -v .../verify/node_modules:/deps mcr.microsoft.com/playwright:v1.60.0-noble \
//     bash -lc 'PLAYWRIGHT_CORE=/deps/playwright-core/index.mjs CHROMIUM=$(ls /ms-playwright/chromium-*/chrome-linux*/chrome|head -1) node /maru/scripts/city-demo-check.mjs'
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');
const MIME = { '.html': 'text/html', '.wasm': 'application/wasm', '.js': 'text/javascript' };
const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/city.html' : req.url.split('?')[0];
    const body = await readFile(join(root, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise(r => server.listen(0, '0.0.0.0', r));
const port = server.address().port;

const { chromium } = await import(
  process.env.PLAYWRIGHT_CORE ?? '/home/icyou/workspace/verify/node_modules/playwright-core/index.mjs'
);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM ?? undefined,
  headless: true,
  args: ['--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
page.on('pageerror', e => { console.error('[pageerror]', e.message); process.exitCode = 1; });
page.on('console', msg => { if (msg.type() === 'error') console.error('[console]', msg.text()); });
await page.goto(`http://127.0.0.1:${port}/city.html`);
await page.waitForFunction(() => window.__READY__ === true, null, { timeout: 120_000 });

// 정착 대기 — 전량 수면(또는 25초 상한)까지. 네이티브 재현 기준 ~6초.
await page.waitForFunction(() => window.__ASLEEP__ >= 4608, null, { timeout: 25_000 });
const settled = await page.evaluate(() => ({ ...window.__STATS__, asleep: window.__ASLEEP__ }));
console.log('settled:', JSON.stringify(settled));

// 포격 3발 (도시 방향)
for (let i = 0; i < 3; i++) {
  await page.mouse.click(560 + i * 60, 380);
  await page.waitForTimeout(400);
}
await page.waitForTimeout(2000);
const after = await page.evaluate(() => ({ ...window.__STATS__, asleep: window.__ASLEEP__ }));
console.log('after-bombardment:', JSON.stringify(after));

await page.waitForTimeout(15000); // 재정착
const resettled = await page.evaluate(() => ({ ...window.__STATS__, asleep: window.__ASLEEP__ }));
console.log('resettled:', JSON.stringify(resettled));

await page.screenshot({ path: join(root, '..', 'city-demo.png') });
console.log('screenshot: city-demo.png');
await browser.close();
server.close();
