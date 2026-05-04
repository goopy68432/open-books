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
