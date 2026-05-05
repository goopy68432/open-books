import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "devices-check.spec.ts",
  reporter: [["list"]],
  fullyParallel: true,
  workers: 4,
  projects: [
    { name: "pixel7", use: { ...devices["Pixel 7"] } },
    {
      name: "s24plus",
      use: {
        viewport: { width: 412, height: 892 },
        deviceScaleFactor: 3.5,
        isMobile: true,
        hasTouch: true,
        defaultBrowserType: "chromium",
        userAgent:
          "Mozilla/5.0 (Linux; Android 14; SM-S926B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
      },
    },
    {
      name: "tabs9-portrait",
      use: {
        viewport: { width: 800, height: 1280 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
        defaultBrowserType: "chromium",
        userAgent:
          "Mozilla/5.0 (Linux; Android 14; SM-X710) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
    },
    {
      name: "tabs9-landscape",
      use: {
        viewport: { width: 1280, height: 800 },
        deviceScaleFactor: 2,
        isMobile: true,
        hasTouch: true,
        defaultBrowserType: "chromium",
        userAgent:
          "Mozilla/5.0 (Linux; Android 14; SM-X710) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
    },
  ],
});
