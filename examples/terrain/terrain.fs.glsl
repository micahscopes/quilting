    #version 300 es
    precision highp float;

    in vec3 vColor;
    in vec3 cornerColor;

    out vec4 fragColor;
    void main() {
        // float threshold = 0.95;
        // bool c1 = cornerColor.x > threshold;
        // bool c2 = cornerColor.y > threshold;
        // bool c3 = cornerColor.z > threshold;
        // vec3 color = vec3(
        //   c1 ? 1.0 : !(c2 || c3) ? vColor.x : 0.0, //vColor.r,
        //   c2 ? 1.0 : !(c1 || c3) ? vColor.y : 0.0, //vColor.g,
        //   c3 ? 1.0 : !(c1 || c2) ? vColor.z : 0.0 //vColor.b
        // );
        // fragColor = vec4(color, 1.0);

        fragColor = vec4(vColor, 1.0);
    }