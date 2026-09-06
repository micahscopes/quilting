// Offline diagnostic only. Consumes an initialization snapshot; never produces
// atlas geometry or participates in the Fe rendering/sampling implementation.
// Compare the current immutable index with the hypothesis that permanently
// rejected candidates can be omitted. Counts are traversal work, not timings.
export function censusCandidateIndex(words, count, querySlots) {
  const capacity = (words.length - 1) / 8;
  if (!(words instanceof Uint32Array) || !Number.isSafeInteger(capacity) || capacity < 1
      || !Number.isSafeInteger(count) || count < 1 || count > capacity
      || words[capacity * 8] !== 0)
    throw Error('expected an initialized sampling snapshot');
  for (let s = 0; s < count; s++) {
    const state = words[capacity * 5 + s];
    if (![0,2,3].includes(state) || words[capacity * 6 + s] !== state || words[capacity * 7 + s] !== 0)
      throw Error('snapshot is not the initial immutable generation');
  }
  if (querySlots.length > 256 || querySlots.some(s => !Number.isSafeInteger(s) || s < 0 || s >= count))
    throw Error('expected at most 256 in-range query slots');
  let leaves = 1;
  while (leaves * 16 < capacity) leaves *= 2;
  const build = admissibleOnly => {
    const tree = new Float64Array(2 * leaves * 5);
    for (let n = 1; n < 2 * leaves; n++) { tree[n*5] = Infinity; tree[n*5+1] = Infinity; }
    const include = (n,x,y,X,Y,r) => {
      const b=n*5;
      tree[b]=Math.min(tree[b],x); tree[b+1]=Math.min(tree[b+1],y);
      tree[b+2]=Math.max(tree[b+2],X); tree[b+3]=Math.max(tree[b+3],Y);
      tree[b+4]=Math.max(tree[b+4],r);
    };
    for (let s=0;s<count;s++) {
      if (admissibleOnly && words[capacity*5+s] !== 0) continue;
      const w=s*5;
      include(leaves+Math.floor(s/16),words[w],words[w+1],words[w],words[w+1],words[w+3]);
    }
    for (let n=leaves-1;n>0;n--) for (const child of [n*2,n*2+1])
      include(n,...tree.subarray(child*5,child*5+5));
    return tree;
  };
  const walk = (tree,s) => {
    const x=words[s*5],y=words[s*5+1],radius=words[s*5+3];
    const stack=[1]; let nodes=0,returned=0,admissible=0;
    while (stack.length) {
      const n=stack.pop(),b=n*5; nodes++;
      if (tree[b]===Infinity) continue;
      if (x<=16384 && y<=16384 && tree[b+2]<=16384 && tree[b+3]<=16384) {
        const dx=Math.max(tree[b]-x,0,x-tree[b+2]),dy=Math.max(tree[b+1]-y,0,y-tree[b+3]);
        if (dx*dx+dy*dy >= Math.max(radius,tree[b+4])) continue;
      }
      if (n<leaves) {stack.push(n*2+1,n*2);continue;}
      for (let other=(n-leaves)*16;other<Math.min(count,(n-leaves+1)*16);other++) {
        returned++; if (words[capacity*5+other]===0) admissible++;
      }
    }
    return {nodes,returned,admissible};
  };
  const all=build(false),eligible=build(true),states=[0,0,0,0];
  for (let s=0;s<count;s++) states[words[capacity*5+s]]++;
  return {capacity,count,leaves,states,
    qualification:'Full hierarchy traversal only; excludes rectangular-query selection, early exits, shader cost, and later MIS rounds.',
    queries:querySlots.map(slot=>({slot,all:walk(all,slot),initiallyAdmissible:walk(eligible,slot)}))};
}
