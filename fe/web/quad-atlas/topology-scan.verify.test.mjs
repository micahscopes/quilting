import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {referenceTopologyScan,topologyScanFixtures} from './topology-scan.verify.mjs';

test('full GPU preview retains every sampled point, ordered triangle and receipt',()=>{
  const read=name=>JSON.parse(readFileSync(new URL(name,import.meta.url),'utf8'));
  const current=read('./topology-browser-evidence.json');
  const previous=read('./compaction-browser-evidence.json');
  assert.equal(current.passes,24);
  assert.equal(current.maximumLod,3);
  assert.equal(current.canonicalKeys,55);
  assert.deepEqual(current.cases,previous.cases);
});

test('GPU topology scan receipts agree with ordered ownership and overflow oracle',()=>{
  // Saved execution evidence, not a claim that Node executes the GPU shaders.
  const r=JSON.parse(readFileSync(new URL('./topology-scan.browser.json',import.meta.url),'utf8'));
  assert.equal(r.passes,24);
  assert.deepEqual(r.shaders.map(s=>s.name),['topology_rank','topology_scan','topology_offsets']);
  const cases=topologyScanFixtures(r.layout);
  assert.equal(cases.length,46);
  assert.equal(r.observations.length,cases.length);
  for(const [i,c] of cases.entries()){
    const expected=referenceTopologyScan(c.input,r.layout);
    const actual=r.observations[i];
    assert.equal(actual.name,c.name);
    assert.equal(actual.comparedWords,r.layout.words);
    assert.deepEqual(actual.progress,expected.slice(r.layout.progress));
    if(c.name.startsWith('valid-'))assert.deepEqual(actual.progress.slice(6),[0,0]);
    else if(c.name!=='pending-zero')assert(actual.progress[6] || actual.progress[7]);
  }
  const overflow=r.observations.find(c=>c.name==='capacity-overflow');
  assert.deepEqual(overflow.progress.slice(4),[r.layout.faces,1,0,3]);
});

test('block scans preserve every successful exclusive face offset under reordered blocks',()=>{
  for(let n=0;n<=8;++n)for(let code=0;code<3**n;++code){
    let v=code;const widths=Array.from({length:n},()=>{const x=1+v%3;v=Math.floor(v/3);return x;});
    let sum=0;const expected=widths.map(x=>{const before=sum;sum+=x;return before;});
    for(const tile of [1,2,3,4,16]){
      const count=Math.ceil(n/tile),local=[],totals=[],base=[];
      for(let b=count-1;b>=0;--b){let total=0;for(let i=b*tile;i<Math.min(n,(b+1)*tile);++i){local[i]=total;total+=widths[i];}totals[b]=total;}
      let total=0;for(let b=0;b<count;++b){base[b]=total;total+=totals[b];}
      const actual=[];for(let i=n-1;i>=0;--i)actual[i]=local[i]+base[Math.floor(i/tile)];
      assert.deepEqual(actual,expected);assert.equal(total,sum);
    }
  }
});
