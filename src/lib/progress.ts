import { loadState, saveState } from "./theme";

const key = (book: string, ch: string, pg: string) => `${ch}/${pg}`;

export function markRead(book: string, chapter: string, page: string, _total?: number) {
  const s = loadState();
  s.progress = s.progress || {};
  s.progress[book] = s.progress[book] || { read: [], lastVisited: null, lastVisitedAt: 0 };
  const id = key(book, chapter, page);
  if (!s.progress[book].read.includes(id)) s.progress[book].read.push(id);
  s.progress[book].lastVisited = id;
  s.progress[book].lastVisitedAt = Date.now();
  saveState(s);
}
export function unmarkRead(book: string, chapter: string, page: string) {
  const s = loadState();
  if (!s.progress?.[book]) return;
  s.progress[book].read = s.progress[book].read.filter((x: string) => x !== key(book, chapter, page));
  saveState(s);
}
export function isRead(book: string, chapter: string, page: string): boolean {
  const s = loadState();
  return !!s.progress?.[book]?.read.includes(key(book, chapter, page));
}
export function getBookProgress(book: string, total: number): { read: number; total: number; percent: number } {
  const s = loadState();
  const read = s.progress?.[book]?.read.length || 0;
  return { read, total, percent: total ? Math.round(read / total * 100) : 0 };
}
