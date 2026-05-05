import { test, expect } from "@playwright/test";

const BASE = "https://goopy68432.github.io/open-books";

const routes = [
  { path: "/", name: "home" },
  { path: "/books/sample-book/", name: "book-cover" },
  { path: "/books/sample-book/intro/hello/", name: "page-hello" },
  { path: "/books/sample-book/basics/syntax/", name: "page-syntax" },
];

for (const r of routes) {
  test(r.name, async ({ page }, testInfo) => {
    const label = testInfo.project.name;
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(`PAGEERR: ${e.message}`));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(`CONSOLE: ${m.text()}`);
    });
    const resp = await page.goto(`${BASE}${r.path}`, { waitUntil: "networkidle" });
    expect(resp?.status()).toBe(200);
    await page.waitForTimeout(500);
    await page.screenshot({
      path: `test-results/shots/${label}-${r.name}.png`,
      fullPage: true,
    });
    const overflow = await page.evaluate(
      () =>
        document.documentElement.scrollWidth - document.documentElement.clientWidth
    );
    const vp = page.viewportSize();
    const fontSize = await page.evaluate(() =>
      parseFloat(getComputedStyle(document.body).fontSize)
    );
    console.log(
      `[${label} ${r.name}] vp=${vp?.width}x${vp?.height} overflow=${overflow}px body-fs=${fontSize}px errors=${errors.length}`
    );
    if (errors.length) console.log(`  ${errors.join("\n  ")}`);
    expect(overflow).toBeLessThanOrEqual(2);
  });
}
