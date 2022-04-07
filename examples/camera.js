import createCamera from "inertial-turntable-camera";
import interactionEvents from "normalized-interaction-events";

const decayTime = 100;
export const makeCamera = () => createCamera({
  aspectRatio: window.innerWidth / window.innerHeight,
  distance: 100,
  center: [0, 0, 0],
  rotationDecayTime: decayTime,
  panDecayTime: decayTime,
  zoomDecayTime: decayTime,
  fovY: Math.PI*5/7,
  far: 100000000000000000,
  near: 0.01
})

const radiansPerHalfScreenWidth = Math.PI * 1;

export const setupCamera = (canvas, camera) => {
  interactionEvents(canvas)
    .on("wheel", function (ev) {
      camera.zoom(ev.x, ev.y, Math.exp(-ev.dy) - 1.0);
      ev.originalEvent.preventDefault();
    })
    .on("mousemove", function (ev) {
      if (!ev.active || ev.buttons !== 1) return;

      if (ev.mods.shift) {
        camera.pan(ev.dx, ev.dy);
      } else if (ev.mods.meta) {
        camera.pivot(ev.dx, ev.dy);
      } else {
        camera.rotate(
          -ev.dx * radiansPerHalfScreenWidth,
          -ev.dy * radiansPerHalfScreenWidth
        );
      }
      ev.originalEvent.preventDefault();
    })
    .on("touchmove", function (ev) {
      if (!ev.active) return;
      camera.rotate(
        -ev.dx * radiansPerHalfScreenWidth,
        -ev.dy * radiansPerHalfScreenWidth
      );
      ev.originalEvent.preventDefault();
    })
    .on("pinchmove", function (ev) {
      if (!ev.active) return;
      camera.zoom(ev.x, ev.y, 1 - ev.zoomx);
      camera.pan(ev.dx, ev.dy);
    })
    .on("touchstart", (ev) => ev.originalEvent.preventDefault())
    .on("pinchstart", (ev) => ev.originalEvent.preventDefault());

  window.addEventListener(
    "resize",
    function () {
      width = window.innerWidth;
      height = window.innerHeight;
      camera.resize(width / height);
    },
    false
  );
};
