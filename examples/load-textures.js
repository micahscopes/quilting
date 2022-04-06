var documentReady = async function () {
  if (document.readyState === "complete") {
    return Promise.resolve();
  }
  return new Promise(function (resolve) {
    window.addEventListener("load", resolve);
  });
}

const matcapUrls = import.meta.glob("./matcaps/*.png");
import seafoam from "./textures/seafoam.png";

export const seafoamImage = await new Promise(async (resolve) => {
  await documentReady();
  const im = new Image();
  im.crossOrigin = "anonymous";
  im.src = seafoam;
  im.onload = () => resolve(Object.assign(im, { style: "max-width:200px" }));
});
export const matcapImages = await Promise.all(
  Object.keys(matcapUrls).map(
    (url) =>
      new Promise(async (resolve) => {
        await documentReady();
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.src = url;
        im.onload = () =>
          resolve(Object.assign(im, { style: "max-width:200px" }));
      })
  )
);
