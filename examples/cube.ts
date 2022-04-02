    // import FPS from "fpsmeter";
    // import createREGL from "regl";
    // import createRESL from "resl";
    import {Device} from '@luma.gl/api';
    const matcapUrls = import.meta.glob("./matcaps/*.png");
    import seafoam from "./textures/seafoam.png";
    import createCamera from "inertial-turntable-camera";
    // import interactionEvents from "normalized-interaction-events";
    import Patch from "../src/patch";
    // import { randomPoints, QUAD, TRI } from "../src/patch";
    import { sample } from "lodash-es";
    import cube from "primitive-cube";
    import { instrumentGLContext } from "@luma.gl/gltools";
    console.log(matcapUrls)

    // const meter = new FPSMeter();

    const randomElement = (array) =>
      array[Math.floor(Math.random() * array.length)];

    const product = (...a) =>
      a.reduce((a, b) => a.flatMap((d) => b.map((e) => [d, e].flat())));

    console.log("hello");
    document.addEventListener("DOMContentLoaded", async function () {
      const device = Device.create(); // Prefers WebGL 2 but falls back to WebGL 1

      const canvas = document.createElement("canvas");
      canvas.width = 800;
      canvas.height = 600;
      document.body.appendChild(canvas);

      const gl = instrumentGLContext(canvas.getContext("webgl2"));

      // const regl = createREGL({
      //   extensions: [
      //     "oes_standard_derivatives",
      //     "oes_element_index_uint", // necessary to support high resolution patches
      //   ],
      // });

      const seafoamImage = await new Promise((resolve) => {
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.src = seafoam;
        im.onload = () =>
          resolve(Object.assign(im, { style: "max-width:200px" }));
      });
      const matcapImages = await Promise.all(
        Object.keys(matcapUrls).map(
          (url) =>
            new Promise((resolve) => {
              const im = new Image();
              im.crossOrigin = "anonymous";
              im.src = url;
              im.onload = () =>
                resolve(Object.assign(im, { style: "max-width:200px" }));
            })
        )
      );

      // CGA3D = Algebra(4, 1);
      // const { Mul, Add, Sub, Div } = CGA3D;
      // const ni = CGA3D.Vector(0, 0, 0, 1, 1);
      // const no = CGA3D.Vector(0, 0, 0, -1, 1);

      // // To create a conformal point, you upcast :
      // const upPoint = (x) =>
      //   no.Add(x).Add(x.Mul(x).Mul(x).Mul(ni).Mul(CGA3D.Scalar(0.5)));
      // const upWeight = (w,p) =>

      // const downPoint = (x) => x.Div(CGA3D.Scalar(0).Sub(x.Dot(ni))).slice(1,4)

      // // const upWeight

      // const transformPointWeight = ({ point, weight }) => {
      //   weight = new CGA3D(weight)
      //   point = CGA3D.Vector(...point)
      //   return downPoint(upPoint(point);
      // };

      // console.log(transformPointWeight({ point: [1, 2, 3], weight: [1,0,0,0] }));

      const cubeGeometry = cube();
      console.log(cubeGeometry);
      const cubePolys = cubeGeometry.cells.map((cell) =>
        cell.map((i) => cubeGeometry.positions[i])
      );
      console.log("cube polys", cubePolys);

      const N = 5;
      const patch = Patch(gl);
      // const matcap = regl.texture({ data: randomElement(matcapImages) });
      const matcaps = matcapImages.map((ata) => regl.texture({ data }));
      // const texture = regl.texture({ data: seafoamImage });
      const patchesProps = cubePolys.map(([p0, p1, p2]) => {
        return {
          matcap: sample(matcaps),
          texture,
          p0,
          p1,
          p2,
          w0: [0, 0, 0, 1],
          w1: [0, 0, 0, 1],
          w2: [0, 0, 0, 1],
        };
      });

      const draw = (props) => {
        // for (let p of patches) {
        // props = patchesProps.map((patches) => ({ ...patches, ...props }));

        const batch = [
          ///
          ...props,
        ];
        // console.log(batch.length)
        patch.draw();
        // patch(batch);
        // }
      };

      const decayTime = 100;
      const camera = (window.camera = createCamera({
        aspectRatio: window.innerWidth / window.innerHeight,
        distance: 30,
        center: [0, 0, 0],
        rotationDecayTime: decayTime,
        panDecayTime: decayTime,
        zoomDecayTime: decayTime,
      }));

      // const setCameraUniforms = regl({
      //   uniforms: {
      //     projection: (ctx, camera) => camera.state.projection,
      //     view: (ctx, camera) => camera.state.view,
      //   },
      // });

      const radiansPerHalfScreenWidth = Math.PI * 1;

      //   interactionEvents(regl._gl.canvas)
      //     .on("wheel", function (ev) {
      //       camera.zoom(ev.x, ev.y, Math.exp(-ev.dy) - 1.0);
      //       ev.originalEvent.preventDefault();
      //     })
      //     .on("mousemove", function (ev) {
      //       if (!ev.active || ev.buttons !== 1) return;

      //       if (ev.mods.shift) {
      //         camera.pan(ev.dx, ev.dy);
      //       } else if (ev.mods.meta) {
      //         camera.pivot(ev.dx, ev.dy);
      //       } else {
      //         camera.rotate(
      //           -ev.dx * radiansPerHalfScreenWidth,
      //           -ev.dy * radiansPerHalfScreenWidth
      //         );
      //       }
      //       ev.originalEvent.preventDefault();
      //     })
      //     .on("touchmove", function (ev) {
      //       if (!ev.active) return;
      //       camera.rotate(
      //         -ev.dx * radiansPerHalfScreenWidth,
      //         -ev.dy * radiansPerHalfScreenWidth
      //       );
      //       ev.originalEvent.preventDefault();
      //     })
      //     .on("pinchmove", function (ev) {
      //       if (!ev.active) return;
      //       camera.zoom(ev.x, ev.y, 1 - ev.zoomx);
      //       camera.pan(ev.dx, ev.dy);
      //     })
      //     .on("touchstart", (ev) => ev.originalEvent.preventDefault())
      //     .on("pinchstart", (ev) => ev.originalEvent.preventDefault());

      //   // console.log(camera);
      //   regl.frame(() => {
      //     meter.tickStart();
      //     camera.tick({
      //       near: camera.params.distance * 0.01,
      //       far: camera.params.distance * 2 + 200,
      //     });

      //     setCameraUniforms(camera, () => {
      //       // if (!camera.state.dirty) return;
      //       regl.clear({ color: [0, 0, 0, 1], depth: 1 });
      //       draw({ eye: camera.state.eye });
      //     });
      //     meter.tick();
      //   });

      //   window.addEventListener(
      //     "resize",
      //     function () {
      //       width = window.innerWidth;
      //       height = window.innerHeight;
      //       camera.resize(width / height);
      //     },
      //     false
      //   );
    });