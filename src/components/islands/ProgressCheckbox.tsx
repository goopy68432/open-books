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
