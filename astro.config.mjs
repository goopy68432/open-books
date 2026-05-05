import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeAutolink from "rehype-autolink-headings";
import rehypeExternalLinks from "rehype-external-links";
import { remarkCallout } from "./src/remark/remark-callout.ts";

export default defineConfig({
  site: "https://goopy68432.github.io",
  base: "/open-books/",
  trailingSlash: "ignore",
  integrations: [react()],
  markdown: {
    remarkPlugins: [remarkGfm, remarkMath, remarkCallout],
    rehypePlugins: [
      rehypeKatex,
      [rehypeAutolink, { behavior: "wrap" }],
      [rehypeExternalLinks, { target: "_blank", rel: ["noopener","noreferrer"] }],
    ],
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" } }
  },
  vite: {
    resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } }
  }
});
