#version 300 es
precision highp float;

flat in vec3 v_lods;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_lods, 0.0);
}
