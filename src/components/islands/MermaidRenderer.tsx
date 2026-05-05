import { useEffect } from "react";

declare global {
  interface Window { __mermaidLoaded?: Promise<any> }
}

async function loadMermaid() {
  if (!window.__mermaidLoaded) {
    window.__mermaidLoaded = import(
      /* @vite-ignore */ "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"
    ).then((mod: any) => {
      const dark = document.documentElement.dataset.theme === "dark";
      mod.default.initialize({
        startOnLoad: false,
        theme: dark ? "dark" : "default",
        securityLevel: "loose",
        flowchart: { htmlLabels: true, useMaxWidth: true },
        themeVariables: dark
          ? { background: "#1e293b", primaryColor: "#1e3a8a", primaryTextColor: "#f1f5f9", lineColor: "#94a3b8" }
          : {},
      });
      return mod.default;
    });
  }
  return window.__mermaidLoaded;
}

async function renderAll() {
  const blocks = document.querySelectorAll<HTMLElement>(
    ".prose pre > code.language-mermaid, .prose pre > code[class*='language-mermaid']"
  );
  if (blocks.length === 0) return;
  const mermaid = await loadMermaid();
  for (let i = 0; i < blocks.length; i++) {
    const code = blocks[i];
    const pre = code.parentElement!;
    if (pre.dataset.mermaidDone === "1") continue;
    const src = code.textContent || "";
    const id = `mmd-${Date.now()}-${i}`;
    try {
      const { svg } = await mermaid.render(id, src);
      const wrap = document.createElement("div");
      wrap.className = "mermaid-render";
      wrap.innerHTML = svg;
      wrap.style.cssText = "margin:1.2em 0;text-align:center;overflow-x:auto;background:var(--bg-soft);border:1px solid var(--border);border-radius:8px;padding:14px";
      pre.replaceWith(wrap);
    } catch (e) {
      pre.dataset.mermaidDone = "1";
      const err = document.createElement("div");
      err.style.cssText = "color:#dc2626;font-size:.85em;margin:.5em 0";
      err.textContent = `Mermaid 렌더 오류: ${(e as Error).message}`;
      pre.after(err);
    }
  }
}

export default function MermaidRenderer() {
  useEffect(() => {
    renderAll();
    const obs = new MutationObserver(() => { renderAll(); });
    obs.observe(document.body, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, []);
  return null;
}
