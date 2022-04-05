import { DrawCall, VertexBuffer } from "picogl";

export const glsl = function (s: TemplateStringsArray, ...values: string[]) {
  let str = "";
  s.forEach((string, i) => {
    str += string + (values[i] || "");
  });

  return str;
};

export const bufferGetter =
  (gl: WebGL2RenderingContext, sourceBuffer: WebGLBuffer) =>
  (destination: ArrayBufferView) => {
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, sourceBuffer);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, destination);
  };

export const setStructUniforms =
  (drawer: DrawCall, baseName: string, values: object) =>
    Object.entries(values).reduce(
      (drawer, [k, v]) => drawer.uniform(`${baseName}.${k}`, v),
      drawer
    );
