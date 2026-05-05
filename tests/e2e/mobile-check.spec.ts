import { test, expect, devices } from "@playwright/test";

const BASE = "https://goopy68432.github.io/open-books";

test.use({ ...devices["Pixel 7"] });

const routes = [
  { path: "/", name: "home" },
  { path: "/books/sample-book/", name: "book-cover" },
  { path: "/books/sample-book/intro/hello/", name: "page-hello" },
  { path: "/books/sample-book/basics/syntax/", name: "page-syntax" },
  { path: "/search/", name: "search" },
];

for (const r of routes) {
  test(`mobile: ${r.name}`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", e => errors.push(`PAGEERR: ${e.message}`));
    page.on("console", m => { if (m.type() === "error") errors.push(`CONSOLE: ${m.text()}`); });
    const resp = await page.goto(`${BASE}${r.path}`, { waitUntil: "networkidle" });
    expect(resp?.status(), `status for ${r.path}`).toBe(200);
    await page.waitForTimeout(500);
    await page.screenshot({ path: `test-results/mobile-${r.name}.png`, fullPage: true });
    // Layout sanity: no horizontal scroll
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, `${r.path} horizontal overflow`).toBeLessThanOrEqual(2);
    if (errors.length) console.log(`[${r.name}] errors:`, errors);
  });
}

test("mobile: theme toggle works", async ({ page }) => {
  await page.goto(`${BASE}/`);
  const before = await page.locator("html").getAttribute("data-theme");
  await page.getByRole("button", { name: "테마 전환" }).first().click();
  await page.waitForTimeout(200);
  const after = await page.locator("html").getAttribute("data-theme");
  expect(after).not.toBe(before);
});
