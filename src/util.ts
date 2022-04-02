export const glsl = function (s: TemplateStringsArray, ...values: string[]) {
  let str = "";
  s.forEach((string, i) => {
    str += string + (values[i] || "");
  });

  return str;
};
