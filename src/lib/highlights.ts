import { loadState, saveState } from "./theme";
export interface Highlight { id: string; color: "yellow"|"green"|"pink"; text: string; range: { start: number; end: number } }
const k = (b: string, c: string, p: string) => `${b}/${c}/${p}`;
export function getHighlights(b: string, c: string, p: string): Highlight[] {
  return (loadState().highlights || {})[k(b,c,p)] || [];
}
export function addHighlight(b: string, c: string, p: string, h: Highlight) {
  const s = loadState(); s.highlights = s.highlights || {};
  const arr = s.highlights[k(b,c,p)] || []; arr.push(h);
  s.highlights[k(b,c,p)] = arr; saveState(s);
}
export function removeHighlight(b: string, c: string, p: string, id: string) {
  const s = loadState(); if (!s.highlights?.[k(b,c,p)]) return;
  s.highlights[k(b,c,p)] = s.highlights[k(b,c,p)].filter((x: Highlight) => x.id !== id); saveState(s);
}
