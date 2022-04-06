#version 300 es
precision highp float;
#pragma glslify: import('./patch.glsl')
layout(location = 0) in vec3 position;
layout(location = 1) in mat3 points;
layout(location = 4) in mat3x4 weights;

uniform mat4 projection, view;
uniform vec3 offset;

out vec3 n;
out vec2 uv;

void main() {

  Patch p = bilinearTri(points[0], points[1], points[2], weights[0], weights[1],
                        weights[2], position.x, position.y);

  n = p.normal;
  uv = vec2(position.x, position.y);
  gl_Position = projection * view * vec4(p.vertex + offset, 1.0);
  // gl_Position = vec4(position, 1.0);
}