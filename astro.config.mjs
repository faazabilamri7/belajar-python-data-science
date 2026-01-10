// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// https://astro.build/config
export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: "Belajar Data Science",
      head: [
        // KaTeX CSS untuk render formula matematis
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css",
          },
        },
      ],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/faazabilamri",
        },
      ],
      sidebar: [
        {
          label: "🏠 Mulai di Sini",
          items: [{ label: "Tentang Kelas Ini", slug: "tentang" }],
        },
        {
          label: "📚 Pertemuan 1: Data Science Introduction",
          autogenerate: { directory: "pertemuan-1" },
        },
        {
          label: "🐍 Pertemuan 2: Python Fundamentals",
          autogenerate: { directory: "pertemuan-2" },
        },
        {
          label: "🐼 Pertemuan 3: Pandas & NumPy",
          autogenerate: { directory: "pertemuan-3" },
        },
        {
          label: "🔍 Pertemuan 4: EDA & Data Cleaning",
          autogenerate: { directory: "pertemuan-4" },
        },
        {
          label: "📊 Pertemuan 5: Statistik Dasar",
          autogenerate: { directory: "pertemuan-5" },
        },
        {
          label: "📈 Pertemuan 6: Data Visualization",
          autogenerate: { directory: "pertemuan-6" },
        },
        {
          label: "🤖 Pertemuan 7: Machine Learning Intro",
          autogenerate: { directory: "pertemuan-7" },
        },
        {
          label: "🏗️ Pertemuan 8: Building ML Models",
          autogenerate: { directory: "pertemuan-8" },
        },
      ],
    }),
  ],
});
