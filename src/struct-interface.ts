
export const membersCGA3 = [
  `scalar`,
  `e1`,
  `e2`,
  `e3`,
  `enil`,
  `einf`,
  `e12`,
  `e13`,
  `e1nil`,
  `e1inf`,
  `e23`,
  `e2nil`,
  `e2inf`,
  `e3nil`,
  `e3inf`,
  `enilinf`,
  `e123`,
  `e12nil`,
  `e12inf`,
  `e13nil`,
  `e13inf`,
  `e1nilinf`,
  `e23nil`,
  `e23inf`,
  `e2nilinf`,
  `e3nilinf`,
  `e123nil`,
  `e123inf`,
  `e12nilinf`,
  `e13nilinf`,
  `e23nilinf`,
  `e123nilinf`,
];

export const getStructVarNames = (varName, structMembers) =>
  structMembers.map((member) => `${varName}.${member}`);

export const arrayToCga3StructProps = (array, varName) =>
  fromPairs(zip(cga3structNames(varName), array));