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
