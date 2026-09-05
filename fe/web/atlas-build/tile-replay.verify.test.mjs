import test from 'node:test';
import assert from 'node:assert/strict';
import {tileReplayBlocks,replayAtlasTile} from './tile-replay.verify.mjs';
const p = (name, inner) => ({source_entry:name,cycle:{group:0,repeat:1035,...(inner?{inner}: {})}});
const cycle = {group:4,repeat:128};
const fixture = () => [p('begin_atlas'),p('initialize'),p('repair_schedule',cycle),p('repair_apply',cycle),p('repair_advance',cycle),p('repair_finish'),p('reserve_tile')];
test('replay isolates one tile and preserves compiled repair order and count',()=>{
  assert.deepEqual(tileReplayBlocks(fixture()).map(b=>[b.repeat,b.passes.map(p=>p.source_entry)]),[
    [1,['initialize']],[128,['repair_schedule','repair_apply','repair_advance']],[1,['repair_finish']]]);
});
test('replay rejects inconsistent cycle counts',()=>{
  const passes=fixture();passes[3]=p('repair_apply',{...cycle,repeat:64});
  assert.throws(()=>tileReplayBlocks(passes),/inconsistent/);
});
test('replay rejects split cycle rather than silently reordering effects',()=>{
  const passes=fixture();passes.splice(3,0,p('other'));
  assert.throws(()=>tileReplayBlocks(passes),/noncontiguous/);
});
test('replay rejects escaping the outer job cycle',()=>{
  const passes=fixture();passes[5].cycle.group=1;
  assert.throws(()=>tileReplayBlocks(passes),/escapes/);
});
test('invalid job and checkpoint requests fail before allocating GPU resources',async()=>{
  await assert.rejects(replayAtlasTile(null,null,[]),/valid quad job/);
  await assert.rejects(replayAtlasTile(null,null,[0,0,0,8,8,1,1,8,9,9],{rounds:[128,64]}),/checkpoints/);
  await assert.rejects(replayAtlasTile(null,null,[0,0,0,8,8,1,1,8,9,9],{sampleGroups:0}),/sampling dispatch/);
});
