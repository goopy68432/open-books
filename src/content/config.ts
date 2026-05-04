import { defineCollection, z } from "astro:content";

const booksYamlSchema = z.object({
  categories: z.array(z.object({
    slug: z.string().regex(/^[a-z0-9-]+$/),
    name: z.string().min(1),
    icon: z.string().optional(),
  })),
});

const bookCollection = defineCollection({
  type: "data",
  schema: z.object({
    title: z.string().min(1),
    subtitle: z.string().optional(),
    author: z.string().min(1),
    category: z.string().regex(/^[a-z0-9-]+$/),
    tags: z.array(z.string()).default([]),
    description: z.string().optional(),
    cover: z.string().optional(),
    created: z.coerce.date(),
    updated: z.coerce.date(),
    draft: z.boolean().default(false),
    order: z.number().int().default(100),
  }),
});

const pageCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string().min(1),
    description: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = {
  books: bookCollection,
  pages: pageCollection,
};

export const BooksYamlSchema = booksYamlSchema;
export type BooksYaml = z.infer<typeof booksYamlSchema>;
