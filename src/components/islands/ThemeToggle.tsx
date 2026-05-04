import { useEffect, useState } from "react";
import { loadState, saveState, applyTheme, type Mode, type FontSize } from "@/lib/theme";

export default function ThemeToggle() {
  const [mode, setMode] = useState<Mode>("auto");
  const [fs, setFs] = useState<FontSize>(2);

  useEffect(() => {
    const s = loadState();
    setMode(s.theme?.mode || "auto");
    setFs((s.theme?.fontSize as FontSize) || 2);
  }, []);

  const update = (m: Mode, f: FontSize) => {
    setMode(m); setFs(f); applyTheme(m, f);
    const s = loadState(); saveState({ ...s, theme: { mode: m, fontSize: f } });
  };
  const cycle = () => { const next: Mode = mode === "light" ? "dark" : mode === "dark" ? "auto" : "light"; update(next, fs); };
  const cycleFs = () => { const next = (fs % 3) + 1 as FontSize; update(mode, next); };

  return (
    <div style={{ display: "flex", gap: ".5rem" }}>
      <button onClick={cycle} aria-label="테마 전환" title={`테마: ${mode}`} style={btn}>
        {mode === "dark" ? "🌙" : mode === "light" ? "☀️" : "🌗"}
      </button>
      <button onClick={cycleFs} aria-label="글자 크기" title={`글자 크기: ${fs}`} style={btn}>
        {fs === 1 ? "A⁻" : fs === 3 ? "A⁺" : "A"}
      </button>
    </div>
  );
}
const btn: React.CSSProperties = { background: "transparent", border: "1px solid var(--border)", borderRadius: 6, padding: ".3em .6em", cursor: "pointer", color: "var(--text)" };
