import { loadState, saveState } from "./theme";
export interface Bookmark { bookSlug: string; chapter: string; page: string; title: string; addedAt: number }
export function listBookmarks(): Bookmark[] { return loadState().bookmarks || []; }
export function isBookmarked(b: string, c: string, p: string) {
  return listBookmarks().some(x => x.bookSlug === b && x.chapter === c && x.page === p);
}
export function toggleBookmark(b: Bookmark) {
  const s = loadState(); s.bookmarks = s.bookmarks || [];
  const i = s.bookmarks.findIndex((x: Bookmark) => x.bookSlug === b.bookSlug && x.chapter === b.chapter && x.page === b.page);
  if (i >= 0) s.bookmarks.splice(i, 1); else s.bookmarks.push({ ...b, addedAt: Date.now() });
  saveState(s);
}
