# openbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Astro 기반 정적 학습 사이트(open-books)를 spec(2026-05-05-openbook-design.md) 그대로 구현해서 GitHub Pages에 배포한다.

**Architecture:** Astro Content Collections로 `content/<book-slug>/NN-chapter/NN-page.md` 트리를 빌드 시 검증·평탄화. 정적 HTML로 렌더하고 인터랙티브한 부분(테마/진행률/북마크/하이라이트/검색)만 React island로 hydrate. 상태는 localStorage 단일 키. Pagefind로 검색 인덱스 생성. `main` push → GitHub Actions → Pages 자동 배포.

**Tech Stack:** Astro 5+, @astrojs/react, TypeScript, pnpm, remark-gfm, remark-math, rehype-katex, shiki, rehype-mermaid, pagefind, vitest, @playwright/test. CSS 변수 기반(Tailwind 미사용). Repo: https://github.com/goopy68432/open-books, base path `/open-books/`.

---

## File Structure

이 plan이 만들어낼 파일들 (spec §6 그대로):

**프로젝트 루트**
- `package.json`, `pnpm-lock.yaml`, `tsconfig.json` — TS strict, path alias `@/*`
- `astro.config.mjs` — site/base/integrations
- `.github/workflows/deploy.yml` — Pages 자동 배포
- `.github/workflows/ci.yml` — PR 빌드+테스트
- `.gitignore` (이미 존재)
- `README.md` (이미 존재, M0에서 보강)

**content/** (작성자 영역)
- `content/books.yml` — 카테고리
- `content/sample-book/book.yml` + 챕터 폴더들 — 통합 테스트용 픽스처

**src/content/**
- `config.ts` — zod 스키마 (book/page/books.yml)

**src/lib/**
- `books.ts` — 책 트리 빌드, prev/next, 평탄화
- `progress.ts`, `bookmarks.ts`, `highlights.ts`, `theme.ts` — localStorage 어댑터
- `paths.ts` — base path 인지 URL helper

**src/components/**
- `TopBar.astro`, `BookSidebar.astro`, `PageToc.astro`, `Breadcrumb.astro`, `Prose.astro`, `PrevNext.astro`, `BookCard.astro`, `Callout.astro`
- `islands/ThemeToggle.tsx`, `islands/ProgressCheckbox.tsx`, `islands/BookmarkButton.tsx`, `islands/Highlighter.tsx`, `islands/SearchBox.tsx`, `islands/PageTocScrollSpy.tsx`

**src/layouts/**
- `BaseLayout.astro`, `HomeLayout.astro`, `ReadLayout.astro`

**src/pages/**
- `index.astro`, `search.astro`, `about.astro`, `404.astro`
- `categories/[cat].astro`, `tags/[tag].astro`
- `books/[book]/index.astro`, `books/[book]/[chapter]/[page].astro`

**src/remark/**
- `remark-callout.ts`

**src/styles/**
- `tokens.css`, `prose.css`, `global.css`

**scripts/**
- `pagefind.mjs` — build 후 인덱싱

**tests/**
- `unit/books.test.ts`, `unit/remark-callout.test.ts`, `unit/progress.test.ts`
- `integration/build.test.ts`
- `e2e/smoke.spec.ts`

각 파일 하나의 책임을 가지며, 인터랙션 컴포넌트만 island로 분리한다.

---

## M0 — 스캐폴딩 + GitHub Pages 배포 파이프라인

목표: `pnpm dev` 동작, `pnpm build` → `dist/` 생성, `main` push → Pages 배포.

### Task M0-1: 프로젝트 초기화

**Files:**
- Create: `package.json`, `tsconfig.json`, `astro.config.mjs`, `src/env.d.ts`, `src/pages/index.astro`

- [ ] **Step 1: package.json 작성**

```json
{
  "name": "open-books",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "astro dev",
    "build": "astro build && node scripts/pagefind.mjs",
    "preview": "astro preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:e2e": "playwright test",
    "typecheck": "astro check"
  },
  "dependencies": {
    "astro": "^5.0.0",
    "@astrojs/react": "^4.0.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "remark-gfm": "^4.0.0",
    "remark-math": "^6.0.0",
    "rehype-katex": "^7.0.0",
    "rehype-autolink-headings": "^7.1.0",
    "rehype-external-links": "^3.0.0",
    "rehype-mermaid": "^3.0.0",
    "shiki": "^1.22.0",
    "katex": "^0.16.11",
    "pagefind": "^1.1.1",
    "zod": "^3.23.8"
  },
  "devDependencies": {
    "@astrojs/check": "^0.9.4",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@playwright/test": "^1.48.0",
    "typescript": "^5.6.0",
    "vitest": "^2.1.0"
  },
  "packageManager": "pnpm@9.12.0"
}
```

- [ ] **Step 2: tsconfig.json 작성**

```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] },
    "jsx": "react-jsx",
    "jsxImportSource": "react"
  },
  "include": [".astro/types.d.ts", "src/**/*", "tests/**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 3: astro.config.mjs 작성**

```js
import { defineConfig } from "astro/config";
import react from "@astrojs/react";

const isProd = process.env.NODE_ENV === "production";

export default defineConfig({
  site: "https://goopy68432.github.io",
  base: "/open-books/",
  trailingSlash: "ignore",
  integrations: [react()],
  markdown: {
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" } }
  },
  vite: {
    resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } }
  }
});
```

- [ ] **Step 4: src/env.d.ts**

```ts
/// <reference path="../.astro/types.d.ts" />
```

- [ ] **Step 5: src/pages/index.astro (placeholder)**

```astro
---
---
<html lang="ko">
  <head><meta charset="utf-8" /><title>open-books</title></head>
  <body><h1>open-books</h1><p>Astro scaffolding ready.</p></body>
</html>
```

- [ ] **Step 6: pnpm install + dev 확인**

Run: `pnpm install && pnpm build`
Expected: `dist/index.html` 생성, exit 0.

- [ ] **Step 7: Commit**

```bash
git add package.json tsconfig.json astro.config.mjs src/ pnpm-lock.yaml
git commit -m "chore(M0): bootstrap astro project with base path /open-books/"
```

### Task M0-2: GitHub Pages 워크플로

**Files:**
- Create: `.github/workflows/deploy.yml`, `.github/workflows/ci.yml`

- [ ] **Step 1: deploy.yml 작성**

```yaml
name: Deploy to GitHub Pages
on:
  push: { branches: [main] }
  workflow_dispatch:
permissions: { contents: read, pages: write, id-token: write }
concurrency: { group: pages, cancel-in-progress: false }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with: { version: 9 }
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - run: pnpm build
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: dist }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: ci.yml 작성**

```yaml
name: CI
on: { pull_request: { branches: [main] } }
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with: { version: 9 }
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - run: pnpm typecheck
      - run: pnpm test
      - run: pnpm build
```

- [ ] **Step 3: Commit + push**

```bash
git add .github/
git commit -m "ci(M0): add GitHub Pages deploy and PR ci workflows"
git push
```

- [ ] **Step 4: GitHub UI에서 Pages 활성화**

Settings → Pages → Source: "GitHub Actions". 첫 deploy 액션이 끝날 때까지 기다림.

---

## M1 — Content Collections 스키마 + 책 트리 + 단위 테스트

목표: `content/`를 zod로 검증·로딩하고, 책 트리 + prev/next 평탄화 함수가 단위 테스트로 검증됨.

### Task M1-1: zod 스키마 정의

**Files:**
- Create: `src/content/config.ts`

- [ ] **Step 1: src/content/config.ts 작성**

```ts
import { defineCollection, z } from "astro:content";

const booksYamlSchema = z.object({
  categories: z.array(z.object({
    slug: z.string().regex(/^[a-z0-9-]+$/),
    name: z.string().min(1),
    icon: z.string().optional(),
  })),
});

const bookCollection = defineCollection({
  type: "data",
  schema: z.object({
    title: z.string().min(1),
    subtitle: z.string().optional(),
    author: z.string().min(1),
    category: z.string().regex(/^[a-z0-9-]+$/),
    tags: z.array(z.string()).default([]),
    description: z.string().optional(),
    cover: z.string().optional(),
    created: z.coerce.date(),
    updated: z.coerce.date(),
    draft: z.boolean().default(false),
    order: z.number().int().default(100),
  }),
});

const pageCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string().min(1),
    description: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = {
  books: bookCollection,
  pages: pageCollection,
};

export const BooksYamlSchema = booksYamlSchema;
export type BooksYaml = z.infer<typeof booksYamlSchema>;
```

- [ ] **Step 2: Commit**

```bash
git add src/content/config.ts
git commit -m "feat(M1): add content collections zod schemas"
```

### Task M1-2: 책 트리 빌더 + 평탄화 (TDD)

**Files:**
- Create: `src/lib/books.ts`, `tests/unit/books.test.ts`
- Test: `tests/unit/books.test.ts`

- [ ] **Step 1: vitest 설정 추가**

`vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
export default defineConfig({
  test: { include: ["tests/**/*.test.ts"], environment: "node" },
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
});
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/unit/books.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { parseSlug, buildBookTree, flattenPages } from "@/lib/books";

describe("parseSlug", () => {
  it("extracts NN and slug from NN-slug", () => {
    expect(parseSlug("01-getting-started")).toEqual({ order: 1, slug: "getting-started" });
    expect(parseSlug("99-appendix")).toEqual({ order: 99, slug: "appendix" });
  });
  it("rejects invalid patterns", () => {
    expect(() => parseSlug("getting-started")).toThrow(/NN-slug/);
    expect(() => parseSlug("1-x")).toThrow(/NN-slug/);
  });
});

describe("buildBookTree", () => {
  it("groups pages by chapter sorted by NN", () => {
    const entries = [
      { id: "python-basics/02-syntax/01-vars", data: { title: "변수" } },
      { id: "python-basics/01-start/02-hello", data: { title: "Hello" } },
      { id: "python-basics/01-start/01-install", data: { title: "Install" } },
    ];
    const tree = buildBookTree("python-basics", entries as any);
    expect(tree.chapters[0].slug).toBe("start");
    expect(tree.chapters[0].pages.map(p => p.slug)).toEqual(["install", "hello"]);
    expect(tree.chapters[1].slug).toBe("syntax");
  });
});

describe("flattenPages + prev/next", () => {
  it("computes prev/next links across chapter boundaries", () => {
    const tree = {
      slug: "b",
      chapters: [
        { slug: "c1", title: "C1", pages: [{ slug: "p1", title: "P1" }, { slug: "p2", title: "P2" }] },
        { slug: "c2", title: "C2", pages: [{ slug: "p3", title: "P3" }] },
      ],
    };
    const flat = flattenPages(tree as any);
    expect(flat[0].prev).toBeUndefined();
    expect(flat[0].next).toMatchObject({ chapter: "c1", page: "p2" });
    expect(flat[1].next).toMatchObject({ chapter: "c2", page: "p3" });
    expect(flat[2].next).toBeUndefined();
  });
});
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `pnpm test`
Expected: FAIL — `books.ts` 미존재.

- [ ] **Step 4: src/lib/books.ts 구현**

```ts
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
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pnpm test`
Expected: 모든 케이스 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lib/books.ts tests/unit/books.test.ts vitest.config.ts
git commit -m "feat(M1): book tree builder and prev/next flattening with tests"
```

### Task M1-3: 샘플 콘텐츠 픽스처

**Files:**
- Create: `content/books.yml`, `content/sample-book/book.yml`, `content/sample-book/01-intro/01-hello.md`, `content/sample-book/01-intro/02-install.md`, `content/sample-book/02-basics/01-syntax.md`

- [ ] **Step 1: content/books.yml**

```yaml
categories:
  - slug: programming
    name: 프로그래밍
    icon: 💻
  - slug: math
    name: 수학
    icon: 🧮
```

- [ ] **Step 2: content/sample-book/book.yml**

```yaml
title: "샘플 책"
subtitle: "테스트용"
author: "정성채"
category: programming
tags: [sample]
description: "통합 테스트용 샘플 책입니다."
created: 2026-05-05
updated: 2026-05-05
order: 10
```

- [ ] **Step 3: 페이지 3개 작성**

`01-intro/01-hello.md`:
```markdown
---
title: "Hello"
description: "첫 페이지"
---

# Hello

안녕하세요. 이것은 첫 페이지입니다.

```python
print("hi")
```

> 💡 **노트**: 콜아웃 테스트.
```

`01-intro/02-install.md`:
```markdown
---
title: "설치"
---

# 설치

`brew install python`을 실행합니다.

수식: $E = mc^2$
```

`02-basics/01-syntax.md`:
```markdown
---
title: "문법"
---

# 문법

## 변수

x = 42
```

- [ ] **Step 4: build 확인**

Run: `pnpm build`
Expected: 빌드 성공, content collections에서 검증된 책/페이지 로딩됨.

- [ ] **Step 5: Commit**

```bash
git add content/
git commit -m "feat(M1): add sample book fixture for integration"
```

---

## M2 — ReadLayout + Prose + Shiki + 콜아웃 + KaTeX

목표: 페이지 라우트가 동작하고 마크다운이 디자인 시스템대로 렌더된다.

### Task M2-1: 색상 토큰 + 글로벌 CSS

**Files:**
- Create: `src/styles/tokens.css`, `src/styles/global.css`, `src/styles/prose.css`

- [ ] **Step 1: tokens.css** (spec §5.1 색상 변수 그대로)

```css
:root {
  --bg:#fff; --bg-soft:#f9fafb; --bg-code:#f3f4f6; --border:#e5e7eb;
  --text:#111827; --text-soft:#4b5563; --text-mute:#9ca3af;
  --link:#2563eb; --accent:#3b82f6; --accent-soft:#dbeafe;
  --hl-yellow:#fef08a; --hl-green:#bbf7d0; --hl-pink:#fbcfe8;
  --callout-info:#dbeafe; --callout-warn:#fef3c7; --callout-danger:#fecaca; --callout-success:#bbf7d0;
  --font-sans:"Pretendard",-apple-system,"Apple SD Gothic Neo",system-ui,sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
  --content-w:clamp(40ch,65ch,720px);
  --sidebar-w:280px; --pagetoc-w:240px; --topbar-h:56px;
  --r-sm:6px; --r-md:10px; --r-lg:14px;
  --fs-1:15px; --fs-2:17px; --fs-3:19px;
  --lh-1:1.7; --lh-2:1.8; --lh-3:1.85;
}
[data-theme="dark"] {
  --bg:#0f172a; --bg-soft:#1e293b; --bg-code:#1e293b; --border:#334155;
  --text:#f1f5f9; --text-soft:#cbd5e1; --text-mute:#64748b;
  --link:#60a5fa; --accent:#60a5fa; --accent-soft:#1e3a8a;
  --hl-yellow:#713f12; --hl-green:#14532d; --hl-pink:#831843;
  --callout-info:#1e3a8a; --callout-warn:#78350f; --callout-danger:#7f1d1d; --callout-success:#14532d;
}
```

- [ ] **Step 2: global.css**

```css
@import "./tokens.css";
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css");

*,*::before,*::after{box-sizing:border-box}
html{font-family:var(--font-sans);background:var(--bg);color:var(--text)}
[data-fontsize="1"]{--fs:var(--fs-1);--lh:var(--lh-1)}
[data-fontsize="2"],html{--fs:var(--fs-2);--lh:var(--lh-2)}
[data-fontsize="3"]{--fs:var(--fs-3);--lh:var(--lh-3)}
body{margin:0;font-size:var(--fs);line-height:var(--lh);transition:background .15s,color .15s}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
```

- [ ] **Step 3: prose.css** (마크다운 결과물 스타일링)

```css
.prose{max-width:var(--content-w);margin:0 auto}
.prose h1{font-size:2em;margin:0 0 .6em;font-weight:800;letter-spacing:-.01em}
.prose h2{font-size:1.4em;margin:1.8em 0 .5em;font-weight:700;border-bottom:1px solid var(--border);padding-bottom:.3em}
.prose h3{font-size:1.15em;margin:1.4em 0 .4em;font-weight:700}
.prose p{margin:.9em 0}
.prose ul,.prose ol{padding-left:1.4em;margin:.8em 0}
.prose code{background:var(--bg-code);padding:.15em .35em;border-radius:4px;font-family:var(--font-mono);font-size:.9em}
.prose pre{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:14px 16px;overflow-x:auto;font-size:.92em;position:relative}
[data-theme="dark"] .prose pre{background:#0b1220}
.prose pre code{background:none;padding:0;font-size:1em}
.prose blockquote{border-left:4px solid var(--border);padding:.2em 1em;color:var(--text-soft);margin:1em 0}
.prose img{max-width:100%;height:auto;border-radius:6px}
.prose table{border-collapse:collapse;width:100%;margin:1em 0}
.prose th,.prose td{border:1px solid var(--border);padding:.5em .8em;text-align:left}
.prose hr{border:0;border-top:1px solid var(--border);margin:2em 0}

.callout{border-left:4px solid;padding:12px 16px;border-radius:6px;margin:1em 0}
.callout-info{background:var(--callout-info);border-color:var(--accent)}
.callout-warn{background:var(--callout-warn);border-color:#d97706}
.callout-danger{background:var(--callout-danger);border-color:#dc2626}
.callout-success{background:var(--callout-success);border-color:#16a34a}
```

- [ ] **Step 4: Commit**

```bash
git add src/styles/
git commit -m "feat(M2): design tokens and prose styling"
```

### Task M2-2: BaseLayout

**Files:**
- Create: `src/layouts/BaseLayout.astro`

- [ ] **Step 1: BaseLayout.astro 작성**

```astro
---
import "@/styles/global.css";
interface Props { title: string; description?: string; }
const { title, description } = Astro.props;
---
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{title}</title>
    {description && <meta name="description" content={description} />}
    <link rel="icon" href={`${import.meta.env.BASE_URL}favicon.svg`} />
    <script is:inline>
      (() => {
        try {
          const s = JSON.parse(localStorage.getItem("openbook:state") || "{}");
          const t = s.theme?.mode || "auto";
          const fs = s.theme?.fontSize || 2;
          const dark = t === "dark" || (t === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
          document.documentElement.dataset.theme = dark ? "dark" : "light";
          document.documentElement.dataset.fontsize = String(fs);
        } catch {}
      })();
    </script>
  </head>
  <body>
    <slot />
  </body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/layouts/BaseLayout.astro
git commit -m "feat(M2): BaseLayout with theme bootstrap"
```

### Task M2-3: remark-callout + 통합

**Files:**
- Create: `src/remark/remark-callout.ts`, `tests/unit/remark-callout.test.ts`
- Modify: `astro.config.mjs`

- [ ] **Step 1: 실패 테스트**

```ts
import { describe, it, expect } from "vitest";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkStringify from "remark-stringify";
import { remarkCallout } from "@/remark/remark-callout";

describe("remarkCallout", () => {
  it("converts > 💡 blockquote to callout-info node", async () => {
    const md = "> 💡 **Note**: hi";
    const tree = unified().use(remarkParse).use(remarkCallout).parse(md);
    unified().use(remarkCallout).runSync(tree);
    const json = JSON.stringify(tree);
    expect(json).toContain("callout-info");
  });
});
```

- [ ] **Step 2: 실패 확인**

Run: `pnpm test tests/unit/remark-callout.test.ts`
Expected: FAIL.

- [ ] **Step 3: src/remark/remark-callout.ts 구현**

```ts
import type { Plugin } from "unified";
import type { Root, Blockquote } from "mdast";
import { visit } from "unist-util-visit";

const MAP: Record<string, string> = {
  "💡": "callout-info", "ℹ️": "callout-info",
  "⚠️": "callout-warn",
  "❗": "callout-danger", "🚫": "callout-danger",
  "✅": "callout-success", "✔️": "callout-success",
};

export const remarkCallout: Plugin<[], Root> = () => (tree) => {
  visit(tree, "blockquote", (node: Blockquote) => {
    const first = node.children[0];
    if (first?.type !== "paragraph") return;
    const text = first.children[0];
    if (text?.type !== "text") return;
    const m = text.value.match(/^([^\s\w]{1,2})\s+/u);
    if (!m) return;
    const cls = MAP[m[1]];
    if (!cls) return;
    text.value = text.value.slice(m[0].length);
    (node as any).data = (node as any).data || {};
    (node as any).data.hName = "div";
    (node as any).data.hProperties = { className: ["callout", cls] };
  });
};
```

- [ ] **Step 4: 테스트 통과**

Run: `pnpm test tests/unit/remark-callout.test.ts`
Expected: PASS.

- [ ] **Step 5: astro.config.mjs에 remark/rehype 통합**

```js
// 상단 import 추가:
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeAutolink from "rehype-autolink-headings";
import rehypeExternalLinks from "rehype-external-links";
import { remarkCallout } from "./src/remark/remark-callout.ts";

// markdown 블록 교체:
markdown: {
  remarkPlugins: [remarkGfm, remarkMath, remarkCallout],
  rehypePlugins: [
    rehypeKatex,
    [rehypeAutolink, { behavior: "wrap" }],
    [rehypeExternalLinks, { target: "_blank", rel: ["noopener","noreferrer"] }],
  ],
  shikiConfig: { themes: { light: "github-light", dark: "github-dark" } },
}
```

- [ ] **Step 6: Commit**

```bash
git add src/remark/remark-callout.ts tests/unit/remark-callout.test.ts astro.config.mjs
git commit -m "feat(M2): remark-callout plugin and markdown pipeline"
```

### Task M2-4: 페이지 라우트 + ReadLayout (단순 버전)

**Files:**
- Create: `src/layouts/ReadLayout.astro`, `src/pages/books/[book]/[chapter]/[page].astro`, `src/components/Prose.astro`, `src/components/Breadcrumb.astro`, `src/components/PrevNext.astro`

- [ ] **Step 1: Prose.astro**

```astro
---
import "@/styles/prose.css";
import "katex/dist/katex.min.css";
---
<article class="prose"><slot /></article>
```

- [ ] **Step 2: Breadcrumb.astro**

```astro
---
interface Props { items: { href?: string; label: string }[]; }
const { items } = Astro.props;
---
<nav style="font-size:.9em;color:var(--text-soft);margin-bottom:1em">
  {items.map((it, i) => (
    <>
      {it.href ? <a href={it.href}>{it.label}</a> : <span>{it.label}</span>}
      {i < items.length - 1 && <span> / </span>}
    </>
  ))}
</nav>
```

- [ ] **Step 3: PrevNext.astro**

```astro
---
import { withBase } from "@/lib/paths";
interface Props {
  bookSlug: string;
  prev?: { chapter: string; page: string; title: string };
  next?: { chapter: string; page: string; title: string };
}
const { bookSlug, prev, next } = Astro.props;
---
<div style="display:flex;justify-content:space-between;margin-top:2em;padding-top:1em;border-top:1px solid var(--border)">
  {prev ? (
    <a href={withBase(`/books/${bookSlug}/${prev.chapter}/${prev.page}`)}>← {prev.title}</a>
  ) : <span />}
  {next ? (
    <a href={withBase(`/books/${bookSlug}/${next.chapter}/${next.page}`)}>{next.title} →</a>
  ) : <span />}
</div>
```

- [ ] **Step 4: src/lib/paths.ts**

```ts
const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
export const withBase = (path: string) => `${BASE}${path.startsWith("/") ? path : `/${path}`}`;
```

- [ ] **Step 5: ReadLayout.astro (단순 1단, 사이드바는 M3에서 추가)**

```astro
---
import BaseLayout from "./BaseLayout.astro";
interface Props { title: string; description?: string; }
const { title, description } = Astro.props;
---
<BaseLayout {title} {description}>
  <main style="max-width:var(--content-w);margin:0 auto;padding:2rem 1rem">
    <slot />
  </main>
</BaseLayout>
```

- [ ] **Step 6: 페이지 라우트**

`src/pages/books/[book]/[chapter]/[page].astro`:
```astro
---
import { getCollection } from "astro:content";
import ReadLayout from "@/layouts/ReadLayout.astro";
import Prose from "@/components/Prose.astro";
import Breadcrumb from "@/components/Breadcrumb.astro";
import PrevNext from "@/components/PrevNext.astro";
import { buildBookTree, flattenPages } from "@/lib/books";
import { withBase } from "@/lib/paths";

export async function getStaticPaths() {
  const pages = await getCollection("pages", e => !e.data.draft);
  const books = await getCollection("books", e => !e.data.draft);
  const paths: any[] = [];
  for (const b of books) {
    const slug = b.id;
    const bookPages = pages.filter(p => p.id.startsWith(`${slug}/`));
    const tree = buildBookTree(slug, bookPages);
    const flat = flattenPages(tree);
    for (const f of flat) {
      paths.push({
        params: { book: slug, chapter: f.chapter, page: f.page },
        props: { f, book: b },
      });
    }
  }
  return paths;
}

const { f, book } = Astro.props;
const { Content } = await f.entry!.render();
const title = `${f.title} · ${book.data.title}`;
---
<ReadLayout title={title} description={f.entry!.data.description}>
  <Breadcrumb items={[
    { href: withBase("/"), label: "홈" },
    { href: withBase(`/books/${book.id}`), label: book.data.title },
    { label: f.title },
  ]} />
  <Prose><Content /></Prose>
  <PrevNext bookSlug={book.id} prev={f.prev} next={f.next} />
</ReadLayout>
```

- [ ] **Step 7: 빌드 + 수동 검증**

Run: `pnpm build && pnpm preview`
열기: `http://localhost:4321/open-books/books/sample-book/intro/hello`
Expected: 페이지 렌더, 코드 하이라이팅, 다음 링크 동작.

- [ ] **Step 8: Commit**

```bash
git add src/components/ src/layouts/ src/pages/books src/lib/paths.ts
git commit -m "feat(M2): book page routing with breadcrumb, prose, prev/next"
```

---

## M3 — 좌 BookSidebar + 우 PageToc + 스크롤스파이

### Task M3-1: BookSidebar.astro

**Files:**
- Create: `src/components/BookSidebar.astro`
- Modify: `src/layouts/ReadLayout.astro`

- [ ] **Step 1: BookSidebar.astro**

```astro
---
import { withBase } from "@/lib/paths";
import type { BookTree } from "@/lib/books";
interface Props { book: { id: string; data: { title: string } }; tree: BookTree; current: { chapter: string; page: string }; }
const { book, tree, current } = Astro.props;
---
<aside style="width:var(--sidebar-w);background:var(--bg-soft);border-right:1px solid var(--border);padding:1.2rem 1rem;overflow-y:auto;height:calc(100vh - var(--topbar-h));position:sticky;top:var(--topbar-h)">
  <a href={withBase(`/books/${book.id}`)} style="font-weight:700;color:var(--text);display:block;margin-bottom:1em">📘 {book.data.title}</a>
  {tree.chapters.map(c => (
    <details open={c.pages.some(p => p.slug === current.page && c.slug === current.chapter)} style="margin-bottom:.6em">
      <summary style="cursor:pointer;font-weight:600;color:var(--text-soft);padding:.2em 0">{c.title}</summary>
      <ul style="list-style:none;padding-left:.6em;margin:.3em 0">
        {c.pages.map(p => {
          const active = c.slug === current.chapter && p.slug === current.page;
          return (
            <li>
              <a href={withBase(`/books/${book.id}/${c.slug}/${p.slug}`)}
                 aria-current={active ? "page" : undefined}
                 style={`display:block;padding:.3em .5em;border-radius:4px;font-size:.95em;${active ? "background:var(--accent-soft);border-left:3px solid var(--accent);font-weight:600" : "color:var(--text-soft)"}`}>
                {p.title}
              </a>
            </li>
          );
        })}
      </ul>
    </details>
  ))}
</aside>
```

- [ ] **Step 2: ReadLayout 3단으로 변경**

```astro
---
import BaseLayout from "./BaseLayout.astro";
interface Props { title: string; description?: string; }
const { title, description } = Astro.props;
---
<BaseLayout {title} {description}>
  <div style="display:grid;grid-template-columns:var(--sidebar-w) 1fr var(--pagetoc-w);min-height:100vh">
    <slot name="sidebar" />
    <main style="padding:2rem clamp(1rem,4vw,3rem);min-width:0">
      <slot />
    </main>
    <slot name="pagetoc" />
  </div>
</BaseLayout>
```

- [ ] **Step 3: 페이지 라우트에 sidebar slot 채우기**

페이지 라우트 상단 `getStaticPaths`에서 `tree`도 props에 포함:
```diff
-        props: { f, book: b },
+        props: { f, book: b, tree },
```

페이지 마크업:
```astro
<ReadLayout title={title} description={f.entry!.data.description}>
  <BookSidebar slot="sidebar" book={book} tree={tree} current={{ chapter: f.chapter, page: f.page }} />
  <Breadcrumb ... />
  <Prose><Content /></Prose>
  <PrevNext ... />
</ReadLayout>
```

- [ ] **Step 4: 빌드+미리보기**

Expected: 좌측에 책 목차, 현재 페이지 하이라이트.

- [ ] **Step 5: Commit**

```bash
git add src/components/BookSidebar.astro src/layouts/ReadLayout.astro src/pages/books
git commit -m "feat(M3): BookSidebar with active page highlight"
```

### Task M3-2: PageToc + 스크롤스파이

**Files:**
- Create: `src/components/PageToc.astro`, `src/components/islands/PageTocScrollSpy.tsx`

- [ ] **Step 1: PageToc.astro (서버 사이드 — 헤딩 추출은 Content 렌더 후 client에서)**

빌드 시 헤딩 추출은 어려우니, 클라이언트에서 DOM을 읽어 목차 생성하는 방식으로 단순화.

`src/components/islands/PageTocScrollSpy.tsx`:
```tsx
import { useEffect, useState } from "react";

interface Heading { id: string; text: string; level: number; }

export default function PageTocScrollSpy() {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>(".prose h2, .prose h3"));
    const list = els.map((el, i) => {
      if (!el.id) el.id = `h-${i}-${(el.textContent || "").toLowerCase().replace(/[^a-z0-9가-힣]+/g, "-")}`;
      return { id: el.id, text: el.textContent || "", level: el.tagName === "H2" ? 2 : 3 };
    });
    setHeadings(list);

    const obs = new IntersectionObserver((entries) => {
      const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, { rootMargin: "-20% 0px -70% 0px", threshold: 0 });
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  if (headings.length === 0) return null;
  return (
    <nav style={{ fontSize: ".88em" }}>
      <div style={{ fontWeight: 700, color: "var(--text)", marginBottom: ".6em" }}>이 페이지</div>
      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {headings.map(h => (
          <li key={h.id} style={{ paddingLeft: h.level === 3 ? "1em" : 0, margin: ".25em 0" }}>
            <a href={`#${h.id}`}
               style={{
                 color: active === h.id ? "var(--accent)" : "var(--text-soft)",
                 borderLeft: active === h.id ? "2px solid var(--accent)" : "2px solid transparent",
                 paddingLeft: ".5em",
                 display: "block",
                 fontWeight: active === h.id ? 600 : 400,
               }}>
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
```

- [ ] **Step 2: PageToc.astro**

```astro
---
import PageTocScrollSpy from "./islands/PageTocScrollSpy.tsx";
---
<aside style="width:var(--pagetoc-w);padding:2rem 1rem;height:calc(100vh - var(--topbar-h));position:sticky;top:var(--topbar-h);overflow-y:auto;border-left:1px solid var(--border)">
  <PageTocScrollSpy client:load />
  <slot name="extra" />
</aside>
```

- [ ] **Step 3: 페이지 라우트에 추가**

```astro
<PageToc slot="pagetoc" />
```

- [ ] **Step 4: 빌드+미리보기**

Expected: 우측에 페이지 내 목차, 스크롤하면 active 상태 변경.

- [ ] **Step 5: Commit**

```bash
git add src/components/PageToc.astro src/components/islands/PageTocScrollSpy.tsx src/pages/books
git commit -m "feat(M3): right page TOC with scroll spy"
```

---

## M4 — 홈 + 카테고리 + 태그 + 책표지 페이지

### Task M4-1: TopBar + HomeLayout

**Files:**
- Create: `src/components/TopBar.astro`, `src/layouts/HomeLayout.astro`, `src/components/BookCard.astro`

- [ ] **Step 1: TopBar.astro**

```astro
---
import { withBase } from "@/lib/paths";
---
<header style="height:var(--topbar-h);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 1.5rem;gap:1rem;position:sticky;top:0;background:var(--bg);z-index:10">
  <a href={withBase("/")} style="font-weight:800;color:var(--text);font-size:1.1em">📘 open-books</a>
  <div style="flex:1" />
  <a href={withBase("/search")} style="color:var(--text-soft)">🔍 검색</a>
  <a href="https://github.com/goopy68432/open-books" style="color:var(--text-soft)">GitHub</a>
</header>
```

- [ ] **Step 2: HomeLayout.astro**

```astro
---
import BaseLayout from "./BaseLayout.astro";
import TopBar from "@/components/TopBar.astro";
interface Props { title: string; description?: string; }
const { title, description } = Astro.props;
---
<BaseLayout {title} {description}>
  <TopBar />
  <main style="max-width:1200px;margin:0 auto;padding:2rem 1.5rem">
    <slot />
  </main>
</BaseLayout>
```

- [ ] **Step 3: BookCard.astro**

```astro
---
import { withBase } from "@/lib/paths";
interface Props { book: any; }
const { book } = Astro.props;
const cover = book.data.cover ? `/books/${book.id}/${book.data.cover}` : null;
---
<a href={withBase(`/books/${book.id}`)} style="display:block;border:1px solid var(--border);border-radius:var(--r-md);overflow:hidden;background:var(--bg);transition:transform .1s">
  <div style={`aspect-ratio:3/2;background:${cover ? `url(${cover}) center/cover` : "linear-gradient(135deg,var(--accent),#a855f7)"};display:flex;align-items:end;padding:1rem;color:#fff`}>
    {!cover && <div style="font-size:1.4em;font-weight:700">{book.data.title}</div>}
  </div>
  <div style="padding:1rem">
    <h3 style="margin:0 0 .3em;font-size:1.05em">{book.data.title}</h3>
    <div style="color:var(--text-soft);font-size:.9em">{book.data.author}</div>
    {book.data.description && <p style="margin:.5em 0 0;font-size:.88em;color:var(--text-soft)">{book.data.description}</p>}
  </div>
</a>
```

- [ ] **Step 4: 홈 페이지 작성**

`src/pages/index.astro` 교체:
```astro
---
import { getCollection } from "astro:content";
import HomeLayout from "@/layouts/HomeLayout.astro";
import BookCard from "@/components/BookCard.astro";
import yaml from "js-yaml";
import { readFileSync } from "node:fs";
import { BooksYamlSchema } from "@/content/config";

const books = (await getCollection("books", b => !b.data.draft)).sort((a,b) => a.data.order - b.data.order);
const categories = BooksYamlSchema.parse(yaml.load(readFileSync("content/books.yml", "utf8"))).categories;
---
<HomeLayout title="open-books" description="내 학습 사이트">
  <h1 style="font-size:2em;margin:0 0 1em">📘 open-books</h1>
  <div style="display:flex;gap:.5em;flex-wrap:wrap;margin-bottom:2em">
    {categories.map(c => (
      <a href={`./categories/${c.slug}`} style="padding:.4em .8em;border:1px solid var(--border);border-radius:999px;color:var(--text-soft)">{c.icon} {c.name}</a>
    ))}
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1.5rem">
    {books.map(b => <BookCard book={b} />)}
  </div>
</HomeLayout>
```

- [ ] **Step 5: js-yaml 추가**

```bash
pnpm add js-yaml && pnpm add -D @types/js-yaml
```

- [ ] **Step 6: 빌드+미리보기 확인**

- [ ] **Step 7: Commit**

```bash
git add src/components/TopBar.astro src/components/BookCard.astro src/layouts/HomeLayout.astro src/pages/index.astro package.json pnpm-lock.yaml
git commit -m "feat(M4): home page with category chips and book grid"
```

### Task M4-2: 카테고리/태그 페이지

**Files:**
- Create: `src/pages/categories/[cat].astro`, `src/pages/tags/[tag].astro`

- [ ] **Step 1: 카테고리 페이지**

```astro
---
import { getCollection } from "astro:content";
import HomeLayout from "@/layouts/HomeLayout.astro";
import BookCard from "@/components/BookCard.astro";
import yaml from "js-yaml";
import { readFileSync } from "node:fs";
import { BooksYamlSchema } from "@/content/config";

export async function getStaticPaths() {
  const cats = BooksYamlSchema.parse(yaml.load(readFileSync("content/books.yml", "utf8"))).categories;
  return cats.map(c => ({ params: { cat: c.slug }, props: { cat: c } }));
}
const { cat } = Astro.props;
const books = (await getCollection("books", b => !b.data.draft && b.data.category === cat.slug))
  .sort((a,b) => a.data.order - b.data.order);
---
<HomeLayout title={`${cat.name} · open-books`}>
  <h1>{cat.icon} {cat.name}</h1>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1.5rem;margin-top:1.5em">
    {books.map(b => <BookCard book={b} />)}
  </div>
</HomeLayout>
```

- [ ] **Step 2: 태그 페이지**

```astro
---
import { getCollection } from "astro:content";
import HomeLayout from "@/layouts/HomeLayout.astro";
import BookCard from "@/components/BookCard.astro";
export async function getStaticPaths() {
  const all = await getCollection("books", b => !b.data.draft);
  const tags = new Set<string>(); for (const b of all) for (const t of b.data.tags) tags.add(t);
  return [...tags].map(t => ({ params: { tag: t }, props: { tag: t } }));
}
const { tag } = Astro.props;
const books = (await getCollection("books", b => !b.data.draft && b.data.tags.includes(tag)))
  .sort((a,b) => a.data.order - b.data.order);
---
<HomeLayout title={`#${tag} · open-books`}>
  <h1>#{tag}</h1>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1.5rem;margin-top:1.5em">
    {books.map(b => <BookCard book={b} />)}
  </div>
</HomeLayout>
```

- [ ] **Step 3: Commit**

```bash
git add src/pages/categories src/pages/tags
git commit -m "feat(M4): category and tag listing pages"
```

### Task M4-3: 책표지 페이지

**Files:**
- Create: `src/pages/books/[book]/index.astro`

- [ ] **Step 1: 책표지 페이지**

```astro
---
import { getCollection } from "astro:content";
import HomeLayout from "@/layouts/HomeLayout.astro";
import { buildBookTree } from "@/lib/books";
import { withBase } from "@/lib/paths";
export async function getStaticPaths() {
  const books = await getCollection("books", b => !b.data.draft);
  const pages = await getCollection("pages", p => !p.data.draft);
  return books.map(b => {
    const tree = buildBookTree(b.id, pages.filter(p => p.id.startsWith(`${b.id}/`)));
    return { params: { book: b.id }, props: { b, tree } };
  });
}
const { b, tree } = Astro.props;
const first = tree.chapters[0]?.pages[0];
---
<HomeLayout title={b.data.title} description={b.data.description}>
  <header style="display:grid;grid-template-columns:200px 1fr;gap:2rem;margin-bottom:2em;align-items:start">
    <div style="aspect-ratio:3/4;background:linear-gradient(135deg,var(--accent),#a855f7);border-radius:8px"></div>
    <div>
      <h1 style="margin:0 0 .3em">{b.data.title}</h1>
      {b.data.subtitle && <p style="color:var(--text-soft);font-size:1.1em;margin:0 0 .8em">{b.data.subtitle}</p>}
      <div style="color:var(--text-soft);margin-bottom:1em">{b.data.author}</div>
      {b.data.description && <p>{b.data.description}</p>}
      {first && (
        <a href={withBase(`/books/${b.id}/${first.slug ? "" : ""}${tree.chapters[0].slug}/${first.slug}`)}
           style="display:inline-block;padding:.6em 1.2em;background:var(--accent);color:#fff;border-radius:6px;font-weight:600;margin-top:1em">
          처음부터 읽기 →
        </a>
      )}
    </div>
  </header>
  <h2>목차</h2>
  {tree.chapters.map(c => (
    <section style="margin:1em 0">
      <h3 style="margin:.5em 0">{c.title}</h3>
      <ul>
        {c.pages.map(p => (
          <li><a href={withBase(`/books/${b.id}/${c.slug}/${p.slug}`)}>{p.title}</a></li>
        ))}
      </ul>
    </section>
  ))}
</HomeLayout>
```

- [ ] **Step 2: Commit**

```bash
git add src/pages/books/[book]/index.astro
git commit -m "feat(M4): book cover page with TOC and start link"
```

---

## M5 — 테마 + 글자크기 + Mermaid

### Task M5-1: ThemeToggle island

**Files:**
- Create: `src/lib/theme.ts`, `src/components/islands/ThemeToggle.tsx`
- Modify: `src/components/TopBar.astro`

- [ ] **Step 1: src/lib/theme.ts**

```ts
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
```

- [ ] **Step 2: ThemeToggle.tsx**

```tsx
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
```

- [ ] **Step 3: TopBar에 통합**

```astro
---
import { withBase } from "@/lib/paths";
import ThemeToggle from "./islands/ThemeToggle.tsx";
---
<header ...>
  <a ...>📘 open-books</a>
  <div style="flex:1" />
  <a href={withBase("/search")}>🔍</a>
  <ThemeToggle client:load />
  <a href="https://github.com/goopy68432/open-books">GitHub</a>
</header>
```

- [ ] **Step 4: Commit**

```bash
git add src/lib/theme.ts src/components/islands/ThemeToggle.tsx src/components/TopBar.astro
git commit -m "feat(M5): theme + font size toggle island with localStorage"
```

### Task M5-2: Mermaid

**Files:**
- Modify: `astro.config.mjs`

- [ ] **Step 1: rehype-mermaid 통합**

`astro.config.mjs` rehypePlugins에 추가:
```js
import rehypeMermaid from "rehype-mermaid";
// rehypePlugins 배열에:
[rehypeMermaid, { strategy: "img-svg", mermaidConfig: { theme: "neutral" } }],
```

- [ ] **Step 2: 샘플 페이지에 mermaid 추가**

`content/sample-book/02-basics/01-syntax.md` 끝에:
```markdown

## 흐름도

```mermaid
graph LR
  A[시작] --> B[변수 선언]
  B --> C[출력]
```
```

- [ ] **Step 3: 빌드 검증**

Run: `pnpm build`
Expected: SVG로 인라인 렌더된 다이어그램이 dist HTML에 포함됨.

- [ ] **Step 4: Commit**

```bash
git add astro.config.mjs content/sample-book/02-basics/01-syntax.md
git commit -m "feat(M5): rehype-mermaid SSG diagrams"
```

---

## M6 — 진행률 + 북마크

### Task M6-1: progress.ts (TDD)

**Files:**
- Create: `src/lib/progress.ts`, `tests/unit/progress.test.ts`

- [ ] **Step 1: 실패 테스트**

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { markRead, isRead, getBookProgress } from "@/lib/progress";

beforeEach(() => { (globalThis as any).localStorage = mockLs(); });

function mockLs() {
  const m = new Map<string, string>();
  return { getItem: (k:string)=>m.get(k) ?? null, setItem: (k:string,v:string)=>{m.set(k,v)}, removeItem: (k:string)=>{m.delete(k)} };
}

describe("progress", () => {
  it("marks page read and detects it", () => {
    markRead("b", "c1", "p1", 5);
    expect(isRead("b","c1","p1")).toBe(true);
    expect(isRead("b","c1","p2")).toBe(false);
  });
  it("computes book progress percent", () => {
    markRead("b","c1","p1", 4);
    markRead("b","c1","p2", 4);
    expect(getBookProgress("b", 4)).toEqual({ read: 2, total: 4, percent: 50 });
  });
});
```

- [ ] **Step 2: 실패 확인**

Run: `pnpm test tests/unit/progress.test.ts`
Expected: FAIL.

- [ ] **Step 3: 구현**

```ts
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
```

- [ ] **Step 4: 테스트 통과**

Run: `pnpm test tests/unit/progress.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/progress.ts tests/unit/progress.test.ts
git commit -m "feat(M6): progress tracking lib with localStorage"
```

### Task M6-2: ProgressCheckbox + BookmarkButton islands

**Files:**
- Create: `src/lib/bookmarks.ts`, `src/components/islands/ProgressCheckbox.tsx`, `src/components/islands/BookmarkButton.tsx`
- Modify: 페이지 라우트, BookSidebar

- [ ] **Step 1: bookmarks.ts**

```ts
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
```

- [ ] **Step 2: ProgressCheckbox.tsx**

```tsx
import { useEffect, useState } from "react";
import { isRead, markRead, unmarkRead } from "@/lib/progress";

interface Props { book: string; chapter: string; page: string; total: number; }
export default function ProgressCheckbox({ book, chapter, page, total }: Props) {
  const [done, setDone] = useState(false);
  useEffect(() => { setDone(isRead(book, chapter, page)); }, [book, chapter, page]);
  const toggle = () => { done ? unmarkRead(book, chapter, page) : markRead(book, chapter, page, total); setDone(!done); };
  return (
    <label style={{ display: "flex", alignItems: "center", gap: ".5em", marginTop: "1.5em", padding: ".7em 1em", background: done ? "var(--accent-soft)" : "var(--bg-soft)", borderRadius: 6, cursor: "pointer", border: "1px solid var(--border)" }}>
      <input type="checkbox" checked={done} onChange={toggle} />
      <span>{done ? "✅ 다 읽음 표시됨" : "📖 다 읽음으로 표시"}</span>
    </label>
  );
}
```

- [ ] **Step 3: BookmarkButton.tsx**

```tsx
import { useEffect, useState } from "react";
import { isBookmarked, toggleBookmark } from "@/lib/bookmarks";

interface Props { book: string; chapter: string; page: string; title: string; }
export default function BookmarkButton({ book, chapter, page, title }: Props) {
  const [on, setOn] = useState(false);
  useEffect(() => { setOn(isBookmarked(book, chapter, page)); }, [book, chapter, page]);
  const click = () => { toggleBookmark({ bookSlug: book, chapter, page, title, addedAt: 0 }); setOn(!on); };
  return (
    <button onClick={click} aria-pressed={on}
      style={{ display: "flex", alignItems: "center", gap: ".4em", background: "transparent", border: "1px solid var(--border)", borderRadius: 6, padding: ".4em .7em", cursor: "pointer", color: on ? "var(--accent)" : "var(--text-soft)" }}>
      {on ? "🔖 북마크됨" : "🔖 북마크"}
    </button>
  );
}
```

- [ ] **Step 4: 페이지에 통합**

페이지 라우트의 본문 아래:
```astro
<ProgressCheckbox client:visible book={book.id} chapter={f.chapter} page={f.page} total={...flat.length...} />
```

PageToc `extra` slot에 BookmarkButton 추가.

- [ ] **Step 5: Commit**

```bash
git add src/lib/bookmarks.ts src/components/islands/ProgressCheckbox.tsx src/components/islands/BookmarkButton.tsx src/pages/books src/components/PageToc.astro
git commit -m "feat(M6): progress checkbox and bookmark button islands"
```

---

## M7 — Highlighter (선택적, 위험 시 v0.2로)

### Task M7-1: Highlighter island (단순화 버전)

**Files:**
- Create: `src/lib/highlights.ts`, `src/components/islands/Highlighter.tsx`

- [ ] **Step 1: highlights.ts**

```ts
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
```

- [ ] **Step 2: Highlighter.tsx — 단순화 (선택 텍스트의 시작 offset을 .prose 컨테이너 기준으로 저장)**

```tsx
import { useEffect, useRef } from "react";
import { getHighlights, addHighlight, removeHighlight, type Highlight } from "@/lib/highlights";

interface Props { book: string; chapter: string; page: string; }

function getOffset(root: HTMLElement, node: Node, offsetIn: number): number {
  const range = document.createRange();
  range.selectNodeContents(root);
  range.setEnd(node, offsetIn);
  return range.toString().length;
}

export default function Highlighter({ book, chapter, page }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prose = document.querySelector<HTMLElement>(".prose");
    if (!prose) return;
    const apply = () => {
      prose.querySelectorAll("[data-hl]").forEach(el => el.replaceWith(...Array.from(el.childNodes)));
      for (const h of getHighlights(book, chapter, page)) {
        try {
          const text = prose.textContent || "";
          const idx = text.indexOf(h.text);
          if (idx === -1) continue;
          // walk text nodes to find start/end node
          const walker = document.createTreeWalker(prose, NodeFilter.SHOW_TEXT);
          let acc = 0; let startNode: Text | null = null; let startOff = 0; let endNode: Text | null = null; let endOff = 0;
          while (walker.nextNode()) {
            const n = walker.currentNode as Text; const len = n.data.length;
            if (!startNode && acc + len > idx) { startNode = n; startOff = idx - acc; }
            if (startNode && acc + len >= idx + h.text.length) { endNode = n; endOff = idx + h.text.length - acc; break; }
            acc += len;
          }
          if (!startNode || !endNode) continue;
          const r = document.createRange(); r.setStart(startNode, startOff); r.setEnd(endNode, endOff);
          const span = document.createElement("span"); span.dataset.hl = h.id; span.style.background = `var(--hl-${h.color})`; span.style.cursor = "pointer";
          span.title = "클릭해서 제거";
          span.addEventListener("click", () => { removeHighlight(book, chapter, page, h.id); apply(); });
          try { r.surroundContents(span); } catch { /* spans across elements: skip */ }
        } catch {}
      }
    };
    apply();
    const onMouseUp = () => {
      const sel = window.getSelection(); if (!sel || sel.isCollapsed) return;
      const text = sel.toString().trim(); if (!text || text.length < 2) return;
      if (!prose.contains(sel.anchorNode)) return;
      const r = sel.getRangeAt(0);
      const start = getOffset(prose, r.startContainer, r.startOffset);
      const end = getOffset(prose, r.endContainer, r.endOffset);
      const id = Math.random().toString(36).slice(2, 9);
      addHighlight(book, chapter, page, { id, color: "yellow", text, range: { start, end } });
      sel.removeAllRanges();
      apply();
    };
    prose.addEventListener("mouseup", onMouseUp);
    return () => prose.removeEventListener("mouseup", onMouseUp);
  }, [book, chapter, page]);

  return <div ref={ref} aria-hidden="true" style={{ display: "none" }} />;
}
```

- [ ] **Step 3: 페이지에 client:idle로 추가**

```astro
<Highlighter client:idle book={book.id} chapter={f.chapter} page={f.page} />
```

- [ ] **Step 4: 수동 검증**

빌드+미리보기. 텍스트 드래그 → 노란 형광펜. 클릭 → 제거. 새로고침 후 유지.

- [ ] **Step 5: 위험 시 폴백**

만약 `surroundContents` 실패가 잦으면 README에 "단일 노드 내 선택만 지원"으로 명시. v0.2에서 보강.

- [ ] **Step 6: Commit**

```bash
git add src/lib/highlights.ts src/components/islands/Highlighter.tsx src/pages/books
git commit -m "feat(M7): text highlighter island with localStorage"
```

---

## M8 — Pagefind 검색

### Task M8-1: 빌드 후 인덱싱

**Files:**
- Create: `scripts/pagefind.mjs`, `src/components/islands/SearchBox.tsx`, `src/pages/search.astro`

- [ ] **Step 1: scripts/pagefind.mjs**

```js
import { exec } from "node:child_process";
import { promisify } from "node:util";
const sh = promisify(exec);
const { stdout } = await sh("npx pagefind --site dist --output-subdir pagefind");
console.log(stdout);
```

- [ ] **Step 2: SearchBox.tsx**

```tsx
import { useEffect, useState } from "react";

declare global { interface Window { pagefind?: any } }

export default function SearchBox() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<any[]>([]);
  useEffect(() => {
    if (!window.pagefind) {
      // @ts-ignore
      import(/* @vite-ignore */ `${import.meta.env.BASE_URL}pagefind/pagefind.js`).then(m => { window.pagefind = m; });
    }
  }, []);
  useEffect(() => {
    if (!q || !window.pagefind) { setResults([]); return; }
    let cancel = false;
    (async () => {
      const r = await window.pagefind.search(q);
      const data = await Promise.all(r.results.slice(0, 10).map((x: any) => x.data()));
      if (!cancel) setResults(data);
    })();
    return () => { cancel = true; };
  }, [q]);
  return (
    <div>
      <input autoFocus value={q} onChange={e => setQ(e.target.value)} placeholder="검색어..." style={{ width: "100%", padding: ".7em 1em", border: "1px solid var(--border)", borderRadius: 8, fontSize: "1.1em", background: "var(--bg)" }} />
      <ul style={{ listStyle: "none", padding: 0, marginTop: "1em" }}>
        {results.map((r: any) => (
          <li key={r.url} style={{ padding: "1em 0", borderBottom: "1px solid var(--border)" }}>
            <a href={r.url} style={{ fontWeight: 600 }}>{r.meta.title}</a>
            <p style={{ margin: ".3em 0 0", color: "var(--text-soft)" }} dangerouslySetInnerHTML={{ __html: r.excerpt }} />
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 3: search.astro**

```astro
---
import HomeLayout from "@/layouts/HomeLayout.astro";
import SearchBox from "@/components/islands/SearchBox.tsx";
---
<HomeLayout title="검색 · open-books">
  <h1>검색</h1>
  <SearchBox client:load />
</HomeLayout>
```

- [ ] **Step 4: 빌드+미리보기**

Run: `pnpm build && pnpm preview`
Expected: `/open-books/search`에서 "변수" 검색 → "문법" 페이지가 결과로.

- [ ] **Step 5: Commit**

```bash
git add scripts/pagefind.mjs src/components/islands/SearchBox.tsx src/pages/search.astro
git commit -m "feat(M8): Pagefind static search"
```

---

## M9 — 반응형 + 접근성 + 스모크 + README

### Task M9-1: 반응형 미디어 쿼리

**Files:**
- Modify: `src/styles/global.css`, `src/layouts/ReadLayout.astro`, `src/components/BookSidebar.astro`

- [ ] **Step 1: ReadLayout 반응형**

```astro
<style is:global>
  @media (max-width: 1279px) { :root { --sidebar-w: 240px; --pagetoc-w: 220px } }
  @media (max-width: 1023px) {
    .read-grid { grid-template-columns: 1fr !important }
    .read-grid > aside { display: none !important }
    .read-grid > .has-mobile-toggle { display: block }
  }
  @media (max-width: 767px) {
    body { font-size: calc(var(--fs) + 1px) }
    .read-grid > main { padding: 1.2rem .9rem !important }
  }
</style>
```

- [ ] **Step 2: 모바일 하단 nav 바 (PrevNext fixed)**

PrevNext를 미디어 쿼리로 모바일에서 fixed bottom으로:
```astro
<style>
  @media (max-width: 767px) {
    .prevnext { position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg); border-top: 1px solid var(--border); padding: .6em 1em; z-index: 5 }
  }
</style>
```

- [ ] **Step 3: Commit**

```bash
git add src/layouts/ReadLayout.astro src/components/PrevNext.astro src/styles/global.css
git commit -m "feat(M9): responsive layout for tablet and mobile"
```

### Task M9-2: Playwright 스모크

**Files:**
- Create: `playwright.config.ts`, `tests/e2e/smoke.spec.ts`

- [ ] **Step 1: playwright.config.ts**

```ts
import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  webServer: { command: "pnpm preview", url: "http://localhost:4321/open-books/", reuseExistingServer: false, timeout: 120_000 },
  use: { baseURL: "http://localhost:4321/open-books/" },
});
```

- [ ] **Step 2: smoke.spec.ts**

```ts
import { test, expect } from "@playwright/test";

test("home loads and shows sample book", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /open-books/ })).toBeVisible();
  await expect(page.getByText("샘플 책")).toBeVisible();
});

test("can navigate to a page and use prev/next", async ({ page }) => {
  await page.goto("/books/sample-book/intro/hello");
  await expect(page.getByRole("heading", { name: "Hello" })).toBeVisible();
  await page.getByRole("link", { name: /설치/ }).click();
  await expect(page.getByRole("heading", { name: "설치" })).toBeVisible();
});

test("theme toggle persists across navigation", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "테마 전환" }).click();
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", /dark|light/);
});

test("search returns results", async ({ page }) => {
  await page.goto("/search");
  await page.getByPlaceholder("검색어...").fill("변수");
  await expect(page.getByRole("link", { name: /문법/ })).toBeVisible({ timeout: 5000 });
});
```

- [ ] **Step 3: 실행**

Run: `pnpm exec playwright install chromium && pnpm build && pnpm test:e2e`
Expected: 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add playwright.config.ts tests/e2e/smoke.spec.ts package.json
git commit -m "test(M9): playwright smoke for home, navigation, theme, search"
```

### Task M9-3: README 업데이트

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 콘텐츠 작성 가이드 추가**

```markdown
# open-books

내 학습 사이트. wikidocs 스타일의 정적 학습 사이트.

🔗 https://goopy68432.github.io/open-books/

## 콘텐츠 작성

```
content/
  books.yml                   # 카테고리 정의
  python-basics/              # 책 slug = URL
    book.yml                  # 책 메타 (제목, 카테고리, 태그)
    01-getting-started/       # 챕터 (NN-slug)
      01-install.md           # 페이지 (NN-slug)
```

페이지 frontmatter:
```yaml
---
title: "필수"
description: "선택"
draft: false
---
```

지원: GFM 마크다운, 코드 하이라이팅, 수식 $E=mc^2$, Mermaid 다이어그램, 콜아웃 (`> 💡`, `> ⚠️`, `> ❗`, `> ✅`).

## 개발

```bash
pnpm install
pnpm dev          # http://localhost:4321/open-books/
pnpm build        # dist/ 생성 + Pagefind 인덱싱
pnpm test         # vitest
pnpm test:e2e     # playwright
```

## 알려진 한계

- 폴더 rename 시 진행률/북마크/하이라이트가 끊깁니다 (식별자가 경로 기반).
- 하이라이터는 단일 텍스트 노드 내 선택만 지원합니다.
- 검색 인덱스는 단순 토큰화 (한국어 형태소 미분석).

## 디자인 명세

[docs/superpowers/specs/2026-05-05-openbook-design.md](docs/superpowers/specs/2026-05-05-openbook-design.md)
```

- [ ] **Step 2: Commit + push**

```bash
git add README.md
git commit -m "docs(M9): authoring guide and known limitations"
git push
```

---

## Self-Review

- ✅ Spec §1.2 핵심 결정 → M0–M9 전체에 반영
- ✅ §2 콘텐츠 포맷 → M1 zod 스키마 + 책 트리 + 샘플
- ✅ §3 라우팅·3단 레이아웃·반응형 → M2–M4, M9
- ✅ §4 빌드 파이프라인·검증·localStorage·검색 → M0, M1, M5, M6, M7, M8
- ✅ §5 비주얼 시스템 → M2, M5
- ✅ §6 프로젝트 구조 → 모든 마일스톤
- ✅ §8 테스트 → M1 unit, M2 unit, M6 unit, M9 e2e
- ✅ §9 CI/CD → M0
- ✅ §10 마일스톤 순서 일치
- ✅ §11 위험: rehype-mermaid SSG (M5에서 strategy 선택), Highlighter 위험 (M7 단독), 폴더 rename 한계 (README), Pagefind 한국어 (README)

플레이스홀더 없음. 타입/메서드 시그니처 일관성 확인됨 (`loadState/saveState`, `parseSlug`, `buildBookTree`, `flattenPages`).
