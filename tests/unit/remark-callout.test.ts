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
