import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { verifyQuadSnapshot } from './verify.mjs';

// Independent finite protocol model. Actual Fe/GPU output is checked against
// the captured mesh corpus separately; this is not execution/performance proof.
const accepted = 1, pending = 0;

test('recorded GPU compaction keeps every canonical preview mesh unchanged', () => {
  const read = name => JSON.parse(readFileSync(new URL(name,import.meta.url),'utf8'));
  const evidence = read('./compaction-browser-evidence.json');
  const canonical = read('./claims-canonical-browser-snapshots.json').cases;
  const baseline = [...canonical,...read('./claims-browser-snapshots.json').cases];
  assert.equal(evidence.passes,22);
  assert.equal(evidence.maximumLod,3);
  assert.equal(evidence.canonicalKeys,55);
  assert.equal(evidence.cases.length,56);
  assert.deepEqual(new Set(evidence.cases.filter(c=>c.seed===42).map(c=>c.key.join(','))),
    new Set(canonical.map(c=>c.key.join(','))));
  for (const observed of evidence.cases) {
    const source=baseline.find(c=>c.params.seed===observed.seed && c.key.join(',')===observed.key.join(','));
    assert(source,'receipt must name a captured baseline');
    verifyQuadSnapshot(source);
    assert.equal(observed.points,source.points.length);
    assert.equal(observed.triangles,source.triangles.length);
    const digest=createHash('sha256').update(JSON.stringify({
      points:source.points,triangles:source.triangles,receipt:source.receipt,
    })).digest('hex');
    assert.equal(observed.sha256,digest);
  }
});

function blocked(states, tile, boundary, capacity, reverse) {
  const blocks = Math.ceil(states.length / tile);
  const local = [], counts = [], bases = [];
  const order = n => Array.from({length:n}, (_,i) => reverse ? n-i-1 : i);
  let unresolved = 0;
  for (const block of order(blocks)) {
    let count = 0;
    for (let i=block*tile;i<Math.min(states.length,(block+1)*tile);++i) {
      local[i] = count;
      count += Number(states[i] === accepted);
      unresolved += Number(states[i] === pending);
    }
    counts[block] = count;
  }
  let count = 0;
  for (let block=0;block<blocks;++block) {
    bases[block] = count;
    count += counts[block];
  }
  const output = [];
  const store = (id, value) => {
    if (id >= capacity) return;
    assert.equal(output[id], undefined, 'scatter writes must be disjoint');
    output[id] = value;
  };
  for (const lane of order(Math.max(states.length,boundary))) {
    if (lane < boundary) store(lane, `b${lane}`);
    if (lane < states.length && states[lane] === accepted) {
      store(boundary+bases[Math.floor(lane/tile)]+local[lane], `c${lane}`);
    }
  }
  return {output,count,unresolved,resident:boundary+count,overflow:boundary+count>capacity};
}

test('blocked compaction preserves order, tail blocks, overflow prefixes and pending receipts', () => {
  for (let length=0;length<=6;++length) {
    for (let code=0;code<4**length;++code) {
      const states=Array.from({length},(_,i)=>(code >>> (2*i)) & 3);
      const selected=states.flatMap((s,i)=>s===accepted?[`c${i}`]:[]);
      for (const tile of [1,2,3,8]) {
        for (const boundary of [0,3,4]) {
          const full=[...Array.from({length:boundary},(_,i)=>`b${i}`),...selected];
          for (const capacity of [0,2,20]) {
            for (const reverse of [false,true]) {
              assert.deepEqual(blocked(states,tile,boundary,capacity,reverse), {
                output:full.slice(0,capacity),count:selected.length,
                unresolved:states.filter(s=>s===pending).length,
                resident:full.length,overflow:full.length>capacity,
              });
            }
          }
        }
      }
    }
  }
});
