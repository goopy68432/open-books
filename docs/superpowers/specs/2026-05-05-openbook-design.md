# openbook — 정적 학습 사이트 디자인 명세

- **상태**: Approved (브레인스토밍 완료)
- **작성일**: 2026-05-05
- **저자**: 정성채 (goopy684@gmail.com)
- **리파지토리**: https://github.com/goopy68432/open-books
- **배포 URL**: https://goopy68432.github.io/open-books/
- **GitHub Pages base path**: `/open-books/` (astro.config.mjs `base`)
- **다음 단계**: writing-plans 스킬로 구현 계획 작성

## 1. 개요

폴더에 마크다운(md) 파일을 정해진 포맷으로 넣으면 PC·태블릿·모바일에서 보기 좋게 학습할 수 있는, GitHub Pages로 배포되는 정적 학습 사이트를 만든다. wikidocs.net을 벤치마킹하되, 모던 docs 사이트(Stripe/Tailwind/Vercel) 수준의 3단 레이아웃과 학습 도구(진행률, 북마크, 하이라이트)를 더한다.

### 1.1 사용자

- **작성자**: 1명 (소유자). git push로 콘텐츠 갱신.
- **독자**: 누구나. 로그인 없음.

### 1.2 핵심 결정 (확정)

| 항목 | 결정 |
|---|---|
| 콘텐츠 모델 | 책(book) → 챕터(chapter) → 페이지(page), 카테고리·태그 |
| 저작 방식 | `content/<book-slug>/` 폴더 = 책. 번호 prefix 폴더/파일명으로 순서. `book.yml`에 메타 |
| 프레임워크 | Astro (Content Collections + Islands) |
| 호스팅 | GitHub Pages (정적 빌드, base path 적용) |
| 인터랙션 저장소 | localStorage (진행률·북마크·하이라이트·테마) |
| 검색 | Pagefind (빌드 시 정적 인덱스) |
| 마크다운 | remark/rehype + Shiki(코드) + KaTeX(수식) + Mermaid(다이어그램) |
| 레이아웃 | 3단 (좌 책 TOC · 본문 · 우 페이지 내 목차/북마크) |

### 1.3 MVP 범위

포함: 진행률 표시, 전문 검색, 사이드바+이전/다음, 다크모드/글자크기, 코드 하이라이팅, KaTeX/Mermaid, 북마크/하이라이트.

비목표: 댓글, 퀴즈, Cmd+K 팔레트, 기기 간 동기화, 다중 작성자 협업, e-book/추천 카운트.

## 2. 콘텐츠 포맷 명세

### 2.1 디렉토리 규칙

```
content/
  books.yml                    카테고리 정의 (전체)
  python-basics/               책 slug (URL: /books/python-basics)
    book.yml                   책 메타 (필수)
    cover.png                  책 표지 (선택, 1200x630 권장)
    01-getting-started/        챕터 (NN- 번호 prefix = 정렬)
      00-intro.md              페이지
      01-install.md
      02-hello-world.md
    02-syntax/
      01-variables.md
    99-appendix/
      01-cheatsheet.md
```

규칙:
- 폴더/파일명은 `NN-slug` 형식 (NN: 두자리 숫자, slug: kebab-case).
- 정렬은 NN 기준, slug는 URL에 사용. NN은 빌드 시 URL에서 제거된다.
- 챕터는 폴더, 페이지는 md. 한 단계 깊이만 허용 (book → chapter → page).
- 이미지 등 자산은 페이지 폴더 안에 둔다.

### 2.2 `books.yml` 스키마 (전체 카테고리)

```yaml
categories:
  - slug: programming
    name: 프로그래밍
    icon: 💻
  - slug: math
    name: 수학
    icon: 🧮
```

### 2.3 `book.yml` 스키마

```yaml
title: "Python 입문"            # 필수
subtitle: "처음 배우는 파이썬"   # 선택
author: "정성채"                 # 필수
category: programming            # 필수, books.yml의 slug
tags: [python, beginner]         # 선택
description: "이 책은..."        # 1-2문장
cover: cover.png                 # 선택. 없으면 자동 그라디언트
created: 2026-05-05              # 필수
updated: 2026-05-05              # 필수
draft: false                     # 선택, 기본 false
order: 10                        # 같은 카테고리 내 정렬 (낮을수록 먼저)
```

### 2.4 페이지 frontmatter

```markdown
---
title: "변수와 자료형"           # 필수
description: "..."               # 선택 (og/검색 미리보기)
draft: false                     # 선택, 기본 false
---

# 변수와 자료형
```

### 2.5 마크다운 자동 처리

- frontmatter `title`과 본문 첫 H1이 동일 텍스트이면 본문 H1을 자동 제거.
- 페이지당 H1은 1개로 강제 (lint 경고).
- `## h2`, `### h3`는 우측 페이지 내 목차에 자동 수집.
- 코드 블록은 언어 명시 시 Shiki 하이라이팅 + 우상단 복사 버튼.
- 수식: `$...$`(인라인), `$$...$$`(블록) → KaTeX.
- ` ```mermaid ` 코드블록 → Mermaid 다이어그램 (SSG 모드).
- 콜아웃: `> 💡`(info), `> ⚠️`(warn), `> ❗`(danger), `> ✅`(success). 첫 줄 마커 검출.
- 외부 링크는 자동 `target=_blank rel=noopener`.
- 깨진 내부 링크는 빌드 경고.

### 2.6 작성자가 외울 4가지

1. 폴더는 `NN-slug`.
2. `book.yml`에 메타.
3. 페이지 frontmatter는 `title`만 필수.
4. GFM + KaTeX + Mermaid + 콜아웃 마커.

## 3. 라우팅과 페이지 구조

### 3.1 URL 구조

| 경로 | 내용 |
|---|---|
| `/` | 홈: 카테고리 탭 + 책 그리드 |
| `/categories/<cat>` | 카테고리별 책 목록 |
| `/tags/<tag>` | 태그별 책 목록 |
| `/books/<book>` | 책 표지: 커버, 메타, 전체 목차, "처음부터 읽기" |
| `/books/<book>/<chapter>/<page>` | 페이지 (3단 레이아웃) |
| `/search` | 검색 결과 (Pagefind) |
| `/about` | 사이트 소개 (선택) |
| `/404` | 친근한 404 |

GitHub Pages base path는 `astro.config.mjs`의 `base`로 처리. 모든 내부 링크는 `Astro.url` 또는 `import.meta.env.BASE_URL` 기반으로 생성한다.

### 3.2 3단 읽기 레이아웃

```
┌─────────────────────────────────────────────────────────┐
│ TopBar  📘 openbook   [본문 검색]  🌗 Aa  [GitHub]      │ 56px
├──────────┬──────────────────────────────────────┬───────┤
│ Book TOC │ 책 / 챕터 / 페이지 (breadcrumb)       │ On    │
│          │                                       │ this  │
│ ▾ 1장    │ # 페이지 제목                          │ page  │
│  • 설치  │ 본문...                               │ H2    │
│ ▸ 2장    │                                       │ H2    │
│ ▸ 3장    │ [← 이전]            [다음 →]          │       │
│ ───────  │                                       │ ───── │
│ 진행률   │ 📊 이 페이지 다 읽음 ☑               │ 🔖 BM │
│ ▓▓░░ 35% │                                       │ 형광펜│
│ 280px    │ min(720px, 본문)                      │ 240px │
└──────────┴──────────────────────────────────────┴───────┘
```

### 3.3 반응형

| 뷰포트 | 동작 |
|---|---|
| ≥ 1280px (PC) | 3단 그대로 |
| 1024–1279 (가로 태블릿) | 좌 240, 우 220, 본문 자동 |
| 768–1023 (세로 태블릿) | 좌 드로어(햄버거), 우 본문 위 접이식, 본문 풀폭 |
| < 768 (모바일) | 모두 드로어, 하단 고정 nav: ←이전/TOC/다음→, 폰트 16px↑ |

### 3.4 핵심 컴포넌트

정적 (.astro): TopBar, BookSidebar, PageToc 셸, Breadcrumb, Prose, PrevNext, BookCard, Callout.
인터랙티브 섬 (.tsx, React): ThemeToggle, ProgressCheckbox, BookmarkButton, Highlighter, SearchBox, PageToc 스크롤스파이.

## 4. 빌드와 데이터 흐름

### 4.1 빌드 파이프라인

1. Astro Content Collections로 `content/`를 로드, zod 스키마로 검증.
2. 책 트리 빌드 → 평탄화된 페이지 배열에서 prev/next 사전 계산.
3. md → HTML: remark-gfm, remark-math + rehype-katex, rehype-shiki, rehype-mermaid(SSG), rehype-autolink-headings, 커스텀 rehype-callout, rehype-external-links.
4. `getStaticPaths()`로 모든 책/챕터/페이지/카테고리/태그 라우트 생성.
5. `astro build` 후 `pagefind --site dist`로 검색 인덱스 생성.
6. `dist/`를 GitHub Pages에 배포.

### 4.2 빌드 검증 규칙

| 상황 | 동작 |
|---|---|
| `book.yml`의 title/author/category/created/updated 누락 | 빌드 실패 (한국어 메시지, 파일 경로 표시) |
| 페이지 frontmatter `title` 누락 | 빌드 실패 |
| `category`가 `books.yml`에 없음 | 빌드 실패 |
| 폴더/파일명이 `NN-slug` 패턴 위반 | 빌드 실패 |
| 깨진 내부 링크 | 경고 |
| 이미지 누락 | 경고 |
| `draft: true` (책 또는 페이지) | 조용히 빌드 제외. 책이 draft면 그 책의 모든 페이지/라우트가 미생성 |

### 4.3 런타임

페이지는 정적 HTML로 즉시 렌더된다. Astro 섬은 hydrate 후 localStorage를 읽어 테마/진행률/북마크/하이라이트 상태를 적용한다. 사용자 인터랙션은 500ms debounce로 localStorage에 기록한다.

### 4.4 localStorage 스키마

키: `openbook:state`. 단일 객체.

```ts
{
  version: 1,
  theme: { mode: "dark" | "light" | "auto", fontSize: 1 | 2 | 3 },
  progress: {
    "<bookSlug>": {
      read: ["<chapter>/<page>", ...],
      lastVisited: "<chapter>/<page>",
      lastVisitedAt: 1777936000
    }
  },
  bookmarks: [
    { bookSlug, chapter, page, title, addedAt }
  ],
  highlights: {
    "<bookSlug>/<chapter>/<page>": [
      { id, color: "yellow"|"green"|"pink", text, range: { start, end } }
    ]
  }
}
```

페이지 식별자는 `<bookSlug>/<chapter>/<page>` 문자열. 폴더 rename 시 진행률/북마크가 끊기는 것은 알려진 한계로 README에 명시한다.

### 4.5 검색

Pagefind. 정적 인덱스가 `dist/pagefind/`에 생성되고 lazy-load 된다. SearchBox 컴포넌트는 추후 FlexSearch로 교체할 수 있도록 어댑터 인터페이스를 둔다.

## 5. 비주얼 시스템

### 5.1 색상 토큰 (CSS 변수)

light/dark 테마. 본문은 `--text`/`--bg`, 사이드바 `--bg-soft`, 악센트는 블루(`--accent` `#3b82f6`/light, `#60a5fa`/dark). 콜아웃 4종(info/warn/danger/success), 하이라이트 3종(yellow/green/pink). 다크 테마는 채도 낮춘 변형 사용. 자세한 값은 `src/styles/tokens.css`에 정의.

### 5.2 타이포그래피

- 본문 sans: Pretendard (CDN), 폴백 system stack.
- 코드: JetBrains Mono.
- 본문 serif 옵션: Noto Serif KR (장기 확장).
- 글자 크기 3단 토글: 15/17/19 px, 행간 1.7/1.8/1.85. 모바일 자동 +1px.
- 본문 컬럼 너비: `clamp(40ch, 65ch, 720px)`.

### 5.3 레이아웃 토큰

좌 사이드바 280/240px, 우 페이지 TOC 240/220px, TopBar 56px 고정. 간격 단위 4/8/12/16/24/32/48/64. 라운드 6/10/14. 그림자는 매우 옅게.

### 5.4 컴포넌트 스타일

- 코드 블록: 옅은 배경, 1px 테두리, 8px 라운드, 우상단 언어 라벨 + 호버 시 복사 버튼. Shiki theme github-light/dark.
- 콜아웃: 좌측 4px 강조 바 + 옅은 배경 + 이모지.
- 진행률 바: 4px 높이, "12 / 34 페이지 (35%)".
- 현재 페이지 (TOC): `--accent-soft` 배경 + 좌측 3px `--accent` 바 + 굵게.
- 하이라이트: 형광펜 톤, 클릭 시 색 변경/삭제 팝오버.

### 5.5 모션

거의 없음. 사이드바 토글 200ms ease-out, 다크모드 150ms (텍스트 색 transition 제외), hover 100ms. 스크롤 스파이는 즉시. `prefers-reduced-motion` 존중.

### 5.6 접근성

- 모든 인터랙션 키보드 가능, 포커스 링 보존.
- 본문 대비 WCAG AA 이상.
- 사이드바 토글 `aria-expanded`, TOC 항목 `aria-current="page"`.
- 코드 블록 스크롤 가능 + `tabindex=0`.

## 6. 프로젝트 구조

```
openbook/
├─ astro.config.mjs              site, base, integrations
├─ package.json
├─ tsconfig.json
├─ pnpm-lock.yaml
├─ .github/workflows/deploy.yml  push to main → build → Pages
├─ .gitignore
├─ README.md                     콘텐츠 작성법 한 페이지
│
├─ content/                      작성자 영역
│  ├─ books.yml
│  └─ <book-slug>/
│     ├─ book.yml
│     ├─ cover.png
│     └─ NN-chapter/NN-page.md
│
├─ src/
│  ├─ content/config.ts          zod 스키마
│  ├─ lib/
│  │  ├─ books.ts                책 트리, prev/next, 평탄화
│  │  ├─ progress.ts             localStorage 어댑터
│  │  ├─ bookmarks.ts
│  │  ├─ highlights.ts
│  │  └─ theme.ts
│  ├─ components/
│  │  ├─ TopBar.astro
│  │  ├─ BookSidebar.astro
│  │  ├─ PageToc.astro
│  │  ├─ Breadcrumb.astro
│  │  ├─ Prose.astro
│  │  ├─ PrevNext.astro
│  │  ├─ BookCard.astro
│  │  ├─ Callout.astro
│  │  └─ islands/
│  │     ├─ ThemeToggle.tsx
│  │     ├─ ProgressCheckbox.tsx
│  │     ├─ BookmarkButton.tsx
│  │     ├─ Highlighter.tsx
│  │     └─ SearchBox.tsx
│  ├─ layouts/
│  │  ├─ BaseLayout.astro
│  │  ├─ HomeLayout.astro
│  │  └─ ReadLayout.astro
│  ├─ pages/
│  │  ├─ index.astro
│  │  ├─ search.astro
│  │  ├─ about.astro
│  │  ├─ 404.astro
│  │  ├─ categories/[cat].astro
│  │  ├─ tags/[tag].astro
│  │  └─ books/[book]/
│  │     ├─ index.astro
│  │     └─ [chapter]/[page].astro
│  ├─ remark/
│  │  └─ remark-callout.ts
│  ├─ styles/
│  │  ├─ tokens.css
│  │  ├─ prose.css
│  │  └─ global.css
│  └─ env.d.ts
│
├─ public/
│  ├─ favicon.svg
│  └─ fonts/
│
├─ scripts/
│  └─ pagefind.mjs
│
├─ tests/
│  ├─ unit/
│  │  ├─ books.test.ts
│  │  └─ remark-callout.test.ts
│  ├─ integration/
│  │  └─ build.test.ts
│  └─ e2e/
│     └─ smoke.spec.ts
│
└─ docs/superpowers/specs/
```

## 7. 외부 의존성

| 용도 | 패키지 |
|---|---|
| 프레임워크 | astro, @astrojs/react |
| md 처리 | @astrojs/mdx(선택), remark-gfm, remark-math, rehype-katex, shiki, rehype-mermaid, rehype-autolink-headings, rehype-external-links |
| 검색 | pagefind |
| 테스트 | vitest, @playwright/test |
| 패키지 매니저 | pnpm |

스타일은 Tailwind 없이 CSS 변수 + 글로벌 CSS만 사용한다.

## 8. 테스트 전략

| 레벨 | 대상 | 도구 |
|---|---|---|
| Unit | 콘텐츠 검증 스키마, prev/next 계산, 콜아웃 변환 | Vitest |
| Integration | 샘플 `content/` 빌드해서 dist HTML 구조 검증 | Vitest + glob |
| E2E (스모크) | 빌드 후 로컬 서빙 → 홈/책/페이지 5개 라우트 200, 다크모드 토글, 검색 | Playwright (1 spec) |
| 시각 회귀 | 핵심 3페이지 모바일+PC 스크린샷 | Playwright (선택, MVP 외) |

## 9. CI/CD

`main` push → GitHub Actions:
1. `pnpm install` (frozen lockfile)
2. `pnpm build` (= `astro build && pagefind --site dist`)
3. `actions/upload-pages-artifact` → `actions/deploy-pages`

PR에서는 빌드+테스트만 실행하고 배포하지 않는다.

## 10. 마일스톤

1. **M0** Astro 스캐폴딩 + 토큰 + GitHub Pages 빌드/배포 파이프라인.
2. **M1** Content Collections 스키마 + 샘플 책 + 책 트리/prev-next 로직 + 단위 테스트.
3. **M2** 3단 ReadLayout + Breadcrumb + Prose + Shiki + 콜아웃 + KaTeX.
4. **M3** 좌 BookSidebar + 우 PageToc + 스크롤스파이.
5. **M4** 홈/카테고리/태그/책표지 페이지.
6. **M5** ThemeToggle + 글자크기 + Mermaid.
7. **M6** ProgressCheckbox + BookmarkButton + 사이드바 진행률 표시.
8. **M7** Highlighter (가장 까다로움, 위험 있으면 v0.2로 미룸).
9. **M8** Pagefind 검색.
10. **M9** 반응형 폴리시 + 접근성 + Playwright 스모크 + README.

각 마일스톤은 독립적으로 동작/테스트 가능한 단위.

## 11. 위험과 결정

| 위험 | 완화 |
|---|---|
| Mermaid SSG는 Chromium 필요 → CI 빌드 무거움 | rehype-mermaid의 inline-svg 전략. 빌드 시간이 너무 길어지면 클라이언트 모드로 폴백. |
| Highlighter는 텍스트 ranges 처리가 까다로움 | 별도 마일스톤 분리. 정 어려우면 v0.2로 미룸. |
| 폴더 rename 시 진행률/북마크 끊김 | 알려진 한계로 README 명시. 향후 `book.yml`의 `aliases` 또는 안정 ID로 보완. |
| 한국어 검색 정확도 | Pagefind는 단순 토큰화. 부족 시 FlexSearch로 교체할 수 있게 SearchBox 인터페이스 격리. |
| GitHub Pages base path 실수 | `site`/`base`를 환경변수로 통일, 모든 내부 링크 `Astro.url` 사용. |
