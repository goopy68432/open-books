import { describe, it, expect, beforeEach } from "vitest";
import { markRead, isRead, getBookProgress } from "@/lib/progress";

beforeEach(() => { (globalThis as any).localStorage = mockLs(); });

function mockLs() {
  const m = new Map<string, string>();
  return { getItem: (k:string)=>m.get(k) ?? null, setItem: (k:string,v:string)=>{m.set(k,v)}, removeItem: (k:string)=>{m.delete(k)} };
}

describe("progress", () => {
  it("marks page read and detects it", () => {
    markRead("b", "c1", "p1", 5);
    expect(isRead("b","c1","p1")).toBe(true);
    expect(isRead("b","c1","p2")).toBe(false);
  });
  it("computes book progress percent", () => {
    markRead("b","c1","p1", 4);
    markRead("b","c1","p2", 4);
    expect(getBookProgress("b", 4)).toEqual({ read: 2, total: 4, percent: 50 });
  });
});
