    #version 300 es
    layout(location=0) in vec3 patchBary;
    layout(location=1) in ivec3 I; // permutation indices
    layout(location=2) in mat3x2 corners;
    layout(location=5) in vec3 vLods;
    #pragma glslify: fbm3d = require('glsl-fractal-brownian-noise/3d') 

    uniform mat4 projection, view;
    
    out vec3 vColor; 
    out vec3 cornerColor;

    void main() {
        vec3 subpatchBary;
        int i = gl_VertexID % 3;
        // ivec3 lods = ivec3(vLods[I[0]], vLods[I[1]], vLods[I[2]]);
        vec3 lods = vLods;
        // float lodFactor = float(lods[i])/128.0;

        float lodFactor = ( 
          patchBary[I[0]]*float(lods[0]) + patchBary[I[1]]*float(lods[1]) + patchBary[I[2]]*float(lods[2])
        );
        
        // float lodFactor = vec3(patchBary[I[0]], patchBary[I[1]], patchBary[I[2]]) / 128.0;
        subpatchBary.x = float(i == 0 || i == 1);
        subpatchBary.y = float(i == 1 || i == 2);
        subpatchBary.z = float(i == 2 || i == 0);
        vColor = subpatchBary*(0.5 + 0.5*lodFactor);
        cornerColor = vec3(patchBary[I[0]], patchBary[I[1]], patchBary[I[2]]) * 4.0;
        // cornerColor = patchBary*2.0;

        vec2 p = vec2(
          patchBary[I[0]]*corners[0][0] + patchBary[I[1]]*corners[1][0] + patchBary[I[2]]*corners[2][0],
          patchBary[I[0]]*corners[0][1] + patchBary[I[1]]*corners[1][1] + patchBary[I[2]]*corners[2][1]
        );

        float cloudy = fbm3d(vec3(p.xy*512.0, 1.0), 1);
        // float r = fbm3d(vec3(p.xy*1.0, 1912323.0), 1);
        // float g = fbm3d(vec3(p.xy*1.0, 1281248.0), 1);
        // float b = fbm3d(vec3(p.xy*1.0, 3005209.0), 1);
        // vec3 colorfulClouds = vec3(cloudy + sin(p.x*lodFactor), cloudy + cos(p.y*lodFactor), cloudy + sin(p.y*lodFactor));
        vec3 colorfulClouds = patchBary*8.0*cloudy - vec3(0.2);
        
        float amt = 0.5;
        float h = (1.0 - amt) + amt*(lodFactor * cloudy);
        h = h / 1.0;
        h = cloudy/16.0;
        // vColor = patchBary ? patchBary * 2.0;
        // vColor = patchBary;
        // vColor = vec3(1.0)*(0.5 + 0.5*cloudy); // + vColor*0.8;
        vColor = (1.0-amt) + amt*5.0*(lodFactor+0.25)*colorfulClouds;
        // vColor = vec3(cloudy)*3.0;
        vec3 position = vec3(p.xy * 4.0, h);
        gl_Position = projection * view * vec4(position.xzy, 1.0);
    }