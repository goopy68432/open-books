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
