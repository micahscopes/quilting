import { defineConfig } from "vite";
import path, { dirname } from "path";
import { glslify } from "vite-plugin-glslify";
import glslifyRollup from "rollup-plugin-glslify";

const glslifyOptions = {
  extensions: ["vs", "fs", ".glsl", ".frag.shader"],
  compress: true,
  transform: ["glslify-import"],
};

export default defineConfig({
  plugins: [glslify(glslifyOptions), glslifyRollup(glslifyOptions)],

  build: {
    lib: {
      entry: path.resolve(dirname("src/index.ts")),
      name: "quilting",
      fileName: (format) => `quilting.${format}.js`,
    },
  },
});
