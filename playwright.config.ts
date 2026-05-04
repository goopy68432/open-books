import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  webServer: { command: "pnpm preview", url: "http://localhost:4321/open-books/", reuseExistingServer: false, timeout: 120_000 },
  use: { baseURL: "http://localhost:4321/open-books/" },
});
