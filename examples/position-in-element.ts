import { mousemove } from "@most/dom-event";
import { map } from "@most/core";
import { flow } from "lodash-es";

export default map(({ clientX, clientY, target }: MouseEvent) => {
  const el = target! as Element;
  let bound = el?.getBoundingClientRect();

  let x = clientX - bound.left - el?.clientLeft;
  let y = clientY - bound.top - el?.clientTop;
  return { x, y };
});

export const fractionalPositionInElement = map(
  ({ clientX, clientY, target }: MouseEvent) => {
    const el = target! as Element;
    let bound = el?.getBoundingClientRect();

    let x = (clientX - bound.left - el?.clientLeft) / el?.clientWidth;
    let y = (clientY - bound.top - el?.clientTop) / el?.clientHeight;
    return { x, y };
  }
);

export const positionInCanvas = flow(
  fractionalPositionInElement,
  map(({ x, y }) => ({ x: x - 0.5, y: -(y - 0.5) }))
);
