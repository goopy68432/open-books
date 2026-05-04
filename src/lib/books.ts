import type { CollectionEntry } from "astro:content";

const SLUG_RE = /^(\d{2})-([a-z0-9][a-z0-9-]*)$/;

export function parseSlug(name: string): { order: number; slug: string } {
  const m = name.match(SLUG_RE);
  if (!m) throw new Error(`Invalid NN-slug pattern: "${name}". Expected /^\\d{2}-[a-z0-9-]+$/`);
  return { order: Number(m[1]), slug: m[2] };
}

export interface PageNode { slug: string; title: string; order: number; entry?: CollectionEntry<"pages">; }
export interface ChapterNode { slug: string; title: string; order: number; pages: PageNode[]; }
export interface BookTree { slug: string; chapters: ChapterNode[]; }

export interface FlatPage {
  bookSlug: string;
  chapter: string;
  page: string;
  title: string;
  entry?: CollectionEntry<"pages">;
  prev?: { chapter: string; page: string; title: string };
  next?: { chapter: string; page: string; title: string };
}

export function buildBookTree(bookSlug: string, entries: CollectionEntry<"pages">[]): BookTree {
  const chapterMap = new Map<string, ChapterNode>();
  for (const e of entries) {
    const parts = e.id.split("/");
    if (parts[0] !== bookSlug || parts.length !== 3) continue;
    const ch = parseSlug(parts[1]);
    const pg = parseSlug(parts[2].replace(/\.(md|mdx)$/, ""));
    if (!chapterMap.has(ch.slug)) {
      chapterMap.set(ch.slug, { slug: ch.slug, title: humanize(ch.slug), order: ch.order, pages: [] });
    }
    chapterMap.get(ch.slug)!.pages.push({
      slug: pg.slug, title: e.data.title, order: pg.order, entry: e,
    });
  }
  const chapters = [...chapterMap.values()].sort((a, b) => a.order - b.order);
  for (const c of chapters) c.pages.sort((a, b) => a.order - b.order);
  return { slug: bookSlug, chapters };
}

export function flattenPages(tree: BookTree): FlatPage[] {
  const flat: FlatPage[] = [];
  for (const c of tree.chapters) {
    for (const p of c.pages) {
      flat.push({ bookSlug: tree.slug, chapter: c.slug, page: p.slug, title: p.title, entry: p.entry });
    }
  }
  for (let i = 0; i < flat.length; i++) {
    if (i > 0) flat[i].prev = pick(flat[i - 1]);
    if (i < flat.length - 1) flat[i].next = pick(flat[i + 1]);
  }
  return flat;
}

const pick = (p: FlatPage) => ({ chapter: p.chapter, page: p.page, title: p.title });
const humanize = (s: string) => s.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase());
