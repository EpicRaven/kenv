// Bundles src/ into ../kenv_ui (index.html + app.js + app.css). No dev server: kenv itself serves the result.
import { build } from "esbuild";
import { mkdirSync, copyFileSync, statSync } from "node:fs";

const out = new URL("../kenv_ui/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
mkdirSync(out, { recursive: true });
await build({
  entryPoints: ["src/main.jsx"], bundle: true, minify: true, legalComments: "none", target: "es2019",
  outfile: out + "app.js", loader: { ".js": "jsx" }, define: { "process.env.NODE_ENV": '"production"' },
});
for (const f of ["index.html", "app.css"]) copyFileSync("src/" + f, out + f);
console.log("kenv_ui built:", (statSync(out + "app.js").size / 1048576).toFixed(2), "MB app.js ->", out);
