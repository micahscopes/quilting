import esbuild from "esbuild";
import { glslify } from "esbuild-plugin-glslify";
import packageConfig from './package.json' assert { type: 'json' };

const opts = {
  entryPoints: [packageConfig.source],
  bundle: true,
  sourcemap: true,
  minify: true,
  target: ["esnext"],
  plugins: [
    glslify({
      extensions: ["vs", "fs", ".glsl", ".frag.shader"],
      compress: true,
      transform: ["glslify-import"],
    }),
  ],
};

// IIFE
esbuild.build({
  ...opts,
  outdir: "dist/iife",
});

// esm
esbuild.build({
  ...opts,
  format: "esm",
  outdir: "dist/esm",
});

// CommonJS
esbuild.build({
  ...opts,
  platform: "node",
  target: ["node10.4"],
  outdir: "dist/cjs",
});
