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
