#version 300 es
precision highp float;

flat in vec4 v_lods;
out vec4 frag_color;

void main() {
    frag_color = v_lods;
}
