#version 300 es
#pragma glslify: import('./lib.glsl')

layout(location=0) in mat3 points;
layout(location=3) in mat3x4 weights;

uniform CGA3 transformation;

out mat3 transformedPoints;
out mat3x4 transformedWeights;
out float lod;

void main(){
  transformedPoints = transformation.scalar*points;
  transformedWeights = transformation.scalar*weights;
  lod = transformation.e1;
}