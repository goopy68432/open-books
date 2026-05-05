import { spawn } from "node:child_process";

const TIMEOUT_MS = Number(process.env.PAGEFIND_TIMEOUT_MS || 240000);
const SKIP = process.env.PAGEFIND_SKIP === "1";

if (SKIP) {
  console.log("[pagefind] PAGEFIND_SKIP=1 — skipping index build");
  process.exit(0);
}

const child = spawn("npx", ["pagefind", "--site", "dist", "--output-subdir", "pagefind"], {
  stdio: "inherit",
  shell: false,
});

const timer = setTimeout(() => {
  console.warn(`[pagefind] timeout after ${TIMEOUT_MS}ms — killing and continuing without search index`);
  child.kill("SIGKILL");
}, TIMEOUT_MS);

child.on("exit", (code, signal) => {
  clearTimeout(timer);
  if (code === 0) {
    console.log("[pagefind] indexed ok");
    process.exit(0);
  }
  // Best-effort: don't fail the build. Site still ships; search just won't work.
  console.warn(`[pagefind] exited with code=${code} signal=${signal} — continuing without search index`);
  process.exit(0);
});

child.on("error", (e) => {
  clearTimeout(timer);
  console.warn(`[pagefind] spawn error: ${e.message} — continuing without search index`);
  process.exit(0);
});
