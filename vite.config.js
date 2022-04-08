import { defineConfig } from "vite";
import path, { dirname } from "path";
import { glslify } from "vite-plugin-glslify";
import glslifyRollup from "rollup-plugin-glslify";

const glslifyOptions = {
  extensions: ["vs", "fs", ".glsl", ".frag.shader"],
  compress: false,
  transform: ["glslify-import"],
};

export default defineConfig({
  optimizeDeps: {
    include: ['ganja.js']
  },

  plugins: [glslify(glslifyOptions), glslifyRollup(glslifyOptions)],
    define: {
      global: {},
    },

  build: {
    commonjsOptions: {
      include: [/ganja.js/, /node_modules/]
    },
    lib: {
      entry: path.resolve(dirname("src/index.ts")),
      name: "quilting",
      fileName: (format) => `quilting.${format}.js`,
    },
  },
});
