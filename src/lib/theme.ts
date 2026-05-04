const KEY = "openbook:state";
export type Mode = "light" | "dark" | "auto";
export type FontSize = 1 | 2 | 3;

interface State { version: number; theme?: { mode: Mode; fontSize: FontSize }; [k: string]: any }

export function loadState(): State {
  try { return JSON.parse(localStorage.getItem(KEY) || '{"version":1}'); }
  catch { return { version: 1 }; }
}
export function saveState(s: State) {
  localStorage.setItem(KEY, JSON.stringify({ version: 1, ...s }));
}
export function applyTheme(mode: Mode, fontSize: FontSize) {
  const dark = mode === "dark" || (mode === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.documentElement.dataset.fontsize = String(fontSize);
}
