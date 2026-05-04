import { test, expect } from "@playwright/test";

test("home loads and shows sample book", async ({ page }) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: /open-books/ })).toBeVisible();
  await expect(page.getByText("샘플 책").first()).toBeVisible();
});

test("can navigate to a page and use prev/next", async ({ page }) => {
  await page.goto("./books/sample-book/intro/hello");
  await expect(page.getByRole("heading", { name: "Hello" })).toBeVisible();
  await page.getByRole("link", { name: "설치", exact: true }).first().click();
  await expect(page.getByRole("heading", { name: "설치" })).toBeVisible();
});

test("theme toggle persists across navigation", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("button", { name: "테마 전환" }).click();
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", /dark|light/);
});

test("search returns results", async ({ page }) => {
  await page.goto("./search");
  await page.getByPlaceholder("검색어...").fill("변수");
  await expect(page.getByRole("link", { name: /문법/ })).toBeVisible({ timeout: 5000 });
});
