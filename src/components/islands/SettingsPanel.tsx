import { useEffect, useRef, useState } from "react";
import { loadState, saveState, applyTheme, type Mode, type FontSize } from "@/lib/theme";

const FS_LABELS: Record<FontSize, string> = {
  1: "아주 작게",
  2: "작게",
  3: "기본",
  4: "크게",
  5: "아주 크게",
};

export default function SettingsPanel() {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<Mode>("auto");
  const [fs, setFs] = useState<FontSize>(3);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const s = loadState();
    setMode(s.theme?.mode || "auto");
    const cur = Number(s.theme?.fontSize ?? 3) as FontSize;
    setFs(([1,2,3,4,5] as FontSize[]).includes(cur) ? cur : 3);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const setBoth = (m: Mode, f: FontSize) => {
    setMode(m); setFs(f); applyTheme(m, f);
    saveState({ ...loadState(), theme: { mode: m, fontSize: f } });
  };

  return (
    <div ref={ref} className="settings">
      <button
        type="button"
        className="settings-trigger"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen(o => !o)}
        title="환경 설정"
      >
        ⚙
      </button>
      {open && (
        <div className="settings-panel" role="dialog" aria-label="환경 설정">
          <div className="row">
            <div className="row-label">테마</div>
            <div className="seg">
              {(["light","auto","dark"] as Mode[]).map(m => (
                <button
                  key={m}
                  type="button"
                  className={"seg-btn" + (mode === m ? " on" : "")}
                  onClick={() => setBoth(m, fs)}
                >
                  {m === "light" ? "라이트" : m === "dark" ? "다크" : "자동"}
                </button>
              ))}
            </div>
          </div>
          <div className="row">
            <div className="row-label">글자 크기 <span className="hint">{FS_LABELS[fs]}</span></div>
            <div className="seg seg-fs">
              {([1,2,3,4,5] as FontSize[]).map(n => (
                <button
                  key={n}
                  type="button"
                  className={"seg-btn fs-" + n + (fs === n ? " on" : "")}
                  onClick={() => setBoth(mode, n)}
                  title={FS_LABELS[n]}
                  aria-label={FS_LABELS[n]}
                >
                  가
                </button>
              ))}
            </div>
          </div>
          <div className="preview">
            본문 미리보기 — 한글과 English가 같은 폭으로 정렬되는 D2 Coding 폰트로 표시됩니다.
          </div>
        </div>
      )}
      <style>{`
        .settings { position: relative; display: inline-flex; }
        .settings-trigger {
          background: transparent;
          border: 1px solid var(--border);
          color: var(--text);
          width: 32px; height: 32px;
          font-size: 14px;
          cursor: pointer;
          border-radius: 0;
          font-family: var(--font-sans);
          line-height: 1;
        }
        .settings-trigger:hover { border-color: var(--text); }
        .settings-panel {
          position: absolute;
          top: calc(100% + 8px);
          right: 0;
          width: 280px;
          background: var(--bg);
          border: 1px solid var(--border-strong);
          padding: 16px;
          z-index: 100;
          box-shadow: 0 4px 12px rgba(0,0,0,.08);
          font-family: var(--font-sans);
        }
        .row { margin-bottom: 14px; }
        .row:last-of-type { margin-bottom: 0; }
        .row-label {
          font-size: .72em;
          letter-spacing: .14em;
          text-transform: uppercase;
          color: var(--text-mute);
          margin-bottom: 8px;
        }
        .hint {
          font-size: .9em;
          letter-spacing: 0;
          text-transform: none;
          color: var(--text-soft);
          font-weight: 400;
          margin-left: 6px;
        }
        .seg {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 0;
          border: 1px solid var(--border);
        }
        .seg-fs { grid-template-columns: repeat(5, 1fr); }
        .seg-btn {
          background: transparent;
          color: var(--text-soft);
          border: 0;
          border-right: 1px solid var(--border);
          padding: 8px 4px;
          cursor: pointer;
          font-family: var(--font-sans);
          font-size: 13px;
          transition: background .1s, color .1s;
        }
        .seg-btn:last-child { border-right: 0; }
        .seg-btn:hover { color: var(--text); }
        .seg-btn.on { background: var(--text); color: var(--bg); }
        .seg-fs .fs-1 { font-size: 11px; }
        .seg-fs .fs-2 { font-size: 13px; }
        .seg-fs .fs-3 { font-size: 15px; }
        .seg-fs .fs-4 { font-size: 17px; }
        .seg-fs .fs-5 { font-size: 19px; }
        .preview {
          margin-top: 14px;
          padding-top: 12px;
          border-top: 1px solid var(--border);
          font-size: var(--fs);
          line-height: var(--lh);
          color: var(--text-soft);
        }
      `}</style>
    </div>
  );
}
