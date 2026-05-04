# openbook용: 기존 마크다운 폴더를 책으로 재구성하는 프롬프트

`<SOURCE_DIR>`(원본 마크다운이 들어있는 폴더)와 `<BOOK_SLUG>`(만들 책의 영문 슬러그)만 채워서 Claude/Codex에 그대로 붙여넣으면 됩니다.

---

## 프롬프트 (복사해서 사용)

```
너는 openbook 정적 학습 사이트의 콘텐츠 큐레이터다. 내 폴더에 있는 마크다운 파일들을
"책 한 권"으로 재구성해서 openbook이 보기 좋게 렌더할 수 있게 만들어라.

## 너의 작업
1. 원본 폴더를 읽고 모든 .md / .markdown 파일을 수집한다.
2. 파일들을 의미 있는 챕터로 그룹핑하고, 각 챕터 안에서 학습 순서로 정렬한다.
3. openbook 포맷에 맞춰 새 폴더 구조를 출력 폴더에 생성한다.
4. 각 페이지에 frontmatter를 추가하고, 콜아웃·코드블록 등 필요한 정리만 한다.
5. 마지막에 book.yml을 작성한다.
6. 작업 끝나면 무엇을 어떻게 분류했는지 요약 보고서를 출력한다.

## 입력
- SOURCE_DIR: <SOURCE_DIR>            # 예: /Users/me/notes/python
- OUTPUT_DIR: content/<BOOK_SLUG>      # openbook 프로젝트의 content/ 안에 만든다
- BOOK_SLUG:  <BOOK_SLUG>              # 예: python-basics (kebab-case 영문)

## openbook 포맷 (반드시 지켜라)

### 디렉토리
content/
  <BOOK_SLUG>/
    book.yml
    01-getting-started/        ← 챕터 폴더. 형식: NN-slug (NN: 두자리 숫자, slug: kebab-case)
      01-intro.md              ← 페이지. 형식: NN-slug.md
      02-install.md
    02-syntax/
      01-variables.md

규칙:
- 폴더/파일명은 정확히 `^\d{2}-[a-z0-9][a-z0-9-]*$` 패턴.
- NN은 정렬용. 빌드 시 URL에서 제거된다 (URL: /books/<BOOK_SLUG>/getting-started/intro).
- 챕터는 폴더, 페이지는 .md. 한 단계 깊이만 (book → chapter → page).
- 빈 챕터 만들지 말 것. 챕터당 최소 1페이지.
- 슬러그는 영문. 한국어 제목은 frontmatter에서 다룬다.
- 큰 부록은 99-appendix 같은 큰 번호로 뒤에 둔다.

### book.yml 스키마
```yaml
title: "<책 제목>"           # 필수, 한국어 OK
subtitle: "<선택 부제>"      # 선택
author: "<저자>"             # 필수
category: programming        # 필수. 가능값: programming | math (없으면 programming 기본)
tags: [<짧은 키워드 3~6개>]  # 선택, 영문 또는 한국어
description: "<1~2문장>"     # 선택, 카드/SEO에 사용
created: <YYYY-MM-DD>        # 필수, 오늘 날짜
updated: <YYYY-MM-DD>        # 필수, 오늘 날짜
order: 100                   # 카테고리 내 정렬용. 일단 100으로
draft: false
```

### 페이지 frontmatter (각 .md 상단)
```yaml
---
title: "<페이지 제목>"        # 필수, 한국어 OK
description: "<선택 한 줄>"   # 선택
draft: false
---
```

frontmatter 다음 줄부터 본문. 본문 첫 줄에 `# title`이 frontmatter title과 같으면 제거 (자동 렌더되므로 중복).

### 본문 정리 규칙
- H1은 페이지당 1개. 추가 H1은 H2로 강등.
- ##, ### 만 사용 (H4 이하 사용 자제).
- 콜아웃은 GFM 인용문 + 첫 줄 이모지 마커:
  - `> 💡 **노트**: ...`     (info)
  - `> ⚠️ **주의**: ...`    (warn)
  - `> ❗ **위험**: ...`    (danger)
  - `> ✅ **확인**: ...`    (success)
- 코드 블록은 언어 명시 (` ```python `, ` ```bash `, ` ```ts `).
- 다이어그램이 필요하면 ` ```mermaid ` 코드블록.
- 인라인 수식 `$...$`, 블록 수식 `$$...$$`.
- 깨진 외부 링크는 그대로 두되, 본문 중복/오타·줄바꿈 깨짐만 살짝 정리.
- 원본의 의도/내용을 임의로 추가하거나 삭제하지 말 것. 재구성과 정리만.

## 챕터 구성 휴리스틱
- 원본 파일들을 다음 시그널로 그룹핑한다:
  1. 파일명/제목의 키워드 (예: install, setup → "시작하기", syntax, types → "기본 문법")
  2. 파일 상단 헤딩 / 첫 문단의 주제어
  3. 원본 폴더 구조가 이미 의미 있다면 최대한 보존
- 각 챕터는 학습 곡선에 맞게 정렬: 개념 소개 → 설치 → 기초 → 응용 → 심화 → 부록.
- 한 챕터에 페이지가 너무 많으면(>10) 챕터를 쪼갠다.
- 한 챕터에 1~2페이지밖에 없으면 인접 챕터에 합친다.
- 페이지 슬러그는 원본 파일명을 영문 kebab-case로 변환. 영문이 아니면 제목을 영문화하거나 짧은 영문 키워드 부여.

## 출력 절차
1. 먼저 SOURCE_DIR 트리와 각 파일의 첫 헤딩/첫 단락을 스캔해 요약한다.
2. 챕터 그룹핑 안을 먼저 텍스트로 출력하고 내가 OK 하면 진행한다.
   (자율 모드면 OK 단계를 생략하고 바로 진행해도 됨)
3. OUTPUT_DIR 아래에 폴더/파일 생성. 기존 OUTPUT_DIR이 있으면 덮어쓰기 전에 먼저 알린다.
4. 작업 종료 후 다음 항목을 보고:
   - 총 페이지 수, 챕터 수
   - 챕터별 목차 (제목 + 페이지 목록)
   - 처리 중 변경한 항목 (강등된 H1, 추가된 콜아웃 등)
   - 영문화한 슬러그 매핑 표 (원본 → 새 슬러그)
   - 의심스러운 항목 / 사람이 검수할 페이지 (예: 제목 모호, 빈 페이지, 중복 같음)

## 절대 하지 말 것
- 원본 파일 수정 (SOURCE_DIR은 read-only)
- 임의로 새 콘텐츠 작성 (요약/번역/보강 X)
- frontmatter 누락
- NN-slug 패턴 위반 (빌드 실패함)
- 한 단계 이상 깊은 폴더 (book → chapter/sub/page X)
- 외부 이미지 다운로드 (참조 URL 그대로 둠)

이제 SOURCE_DIR을 읽기 시작해라.
```

---

## 사용 예시

```
SOURCE_DIR: /Users/me/dropbox/python-notes
BOOK_SLUG:  python-basics
```

위 자리에 채워서 Claude Code에 입력 → Claude가 폴더 스캔 → 챕터 안 제시 → 승인 후 `content/python-basics/` 생성.

생성 후 검증:
```bash
pnpm build  # 스키마 위반 시 한국어 에러로 알려줌
pnpm dev    # 즉시 미리보기
```

## 자율 모드로 돌리고 싶으면

프롬프트 끝에 다음 한 줄 추가:
```
중간 승인 단계 생략. 챕터 안을 결정한 후 바로 파일을 생성하고, 마지막에만 보고해라.
```
