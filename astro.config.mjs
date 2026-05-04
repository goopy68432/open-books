import { defineConfig } from "astro/config";
import react from "@astrojs/react";

const isProd = process.env.NODE_ENV === "production";

export default defineConfig({
  site: "https://goopy68432.github.io",
  base: "/open-books/",
  trailingSlash: "ignore",
  integrations: [react()],
  markdown: {
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" } }
  },
  vite: {
    resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } }
  }
});
