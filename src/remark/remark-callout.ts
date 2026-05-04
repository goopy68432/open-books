import type { Plugin } from "unified";
import type { Root, Blockquote } from "mdast";
import { visit } from "unist-util-visit";

const MAP: Record<string, string> = {
  "💡": "callout-info", "ℹ️": "callout-info",
  "⚠️": "callout-warn",
  "❗": "callout-danger", "🚫": "callout-danger",
  "✅": "callout-success", "✔️": "callout-success",
};

export const remarkCallout: Plugin<[], Root> = () => (tree) => {
  visit(tree, "blockquote", (node: Blockquote) => {
    const first = node.children[0];
    if (first?.type !== "paragraph") return;
    const text = first.children[0];
    if (text?.type !== "text") return;
    const m = text.value.match(/^([^\s\w]{1,2})\s+/u);
    if (!m) return;
    const cls = MAP[m[1]];
    if (!cls) return;
    text.value = text.value.slice(m[0].length);
    (node as any).data = (node as any).data || {};
    (node as any).data.hName = "div";
    (node as any).data.hProperties = { className: ["callout", cls] };
  });
};
