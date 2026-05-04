import { exec } from "node:child_process";
import { promisify } from "node:util";
const sh = promisify(exec);
const { stdout } = await sh("npx pagefind --site dist --output-subdir pagefind");
console.log(stdout);
