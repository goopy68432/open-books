const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
export const withBase = (path: string) => `${BASE}${path.startsWith("/") ? path : `/${path}`}`;
