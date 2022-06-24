import { merge, until, constant, join, now, filter, map } from "@most/core";
import { domEvent } from "@most/dom-event";

const visibilitychange = domEvent("visibilitychange", document);
export const tabFocusOn$ = filter(
  () => !document.hidden,
  map(() => !document.hidden, visibilitychange)
);
export const tabFocusOff$ = filter(
  () => document.hidden,
  map(() => document.hidden, visibilitychange)
);

// start/stop a given stream automatically depending on whether or not the tab is in focus
export const whileTabFocus = ($) =>
  join(constant(until(tabFocusOff$, $), merge(tabFocusOn$, now(null))));
