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
