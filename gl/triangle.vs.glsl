
precision highp float;
#pragma glslify: import('./patch.glsl')
attribute vec3 normal;
attribute vec3 position;

uniform vec3 p0, p1, p2;
uniform vec4 w0, w1, w2;
uniform mat4 projection, view;
uniform vec3 offset;

varying vec3 n;
varying vec2 uv;

void main() {

  Patch p = bilinearTri(p0, p1, p2, w0, w1, w2, position.x, position.y);

  n = p.normal;
  uv = vec2(position.x, position.y);
  gl_Position = projection * view * vec4(p.vertex + offset, 1.0);
  // gl_Position = vec4(position, 1.0);
}