import assert from 'node:assert/strict';

// Independent diagnostic oracle, not a production atlas generator. Enumerate
// square symmetries explicitly rather than translating Fe's cursor algorithm.
export function expectedQuadCodes(maximum = 8) {
  assert.ok(Number.isInteger(maximum) && maximum >= 0 && maximum <= 8);
  const encode = edges => edges.reduce((code, edge) => code * 9 + edge, 0);
  const representatives = new Set();
  for (let a=0; a<=maximum; a++) for (let b=0; b<=maximum; b++)
    for (let c=0; c<=maximum; c++) for (let d=0; d<=maximum; d++) {
      const ring=[a,b,c,d], orbit=[];
      for (const order of [ring,[...ring].reverse()]) {
        for (let shift=0; shift<4; shift++) {
          orbit.push(encode([...order.slice(shift),...order.slice(0,shift)]));
        }
      }
      representatives.add(Math.min(...orbit));
    }
  return [...representatives].sort((a,b)=>a-b);
}

export function verifyQuadJobCoverage(data, maximum = 8) {
  const codes=expectedQuadCodes(maximum);
  // Zero is the cleared-history sentinel, so Fe records key code plus one.
  assert.deepEqual(data.history,codes.map(code=>code+1),
    'every canonical job must appear exactly once in dense ordinal order');
  assert.deepEqual(data.receipt,[codes.length,1,codes.at(-1),0],
    'wrong epoch must preserve the last job; exhaustion must invalidate it');
  return {maximum,canonicalJobs:codes.length,complete:true};
}
