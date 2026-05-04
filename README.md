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
- Mermaid 다이어그램은 SSG 환경에서 Playwright/Chromium이 필요합니다 (없으면 코드 블록으로 렌더).

## 디자인 명세

[docs/superpowers/specs/2026-05-05-openbook-design.md](docs/superpowers/specs/2026-05-05-openbook-design.md)
