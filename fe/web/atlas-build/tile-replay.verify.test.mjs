import test from 'node:test';
import assert from 'node:assert/strict';
import {tileReplayBlocks,tileReplayPrefix,diagnoseAtlasPrefix,replayAtlasTile,summarizeGpuTimestamps} from './tile-replay.verify.mjs';

test('GPU timestamp census subtracts before number conversion and sums repeated stages',()=>{
  const epoch=2n**60n;
  const result=summarizeGpuTimestamps(['propose','retire','propose'],new BigUint64Array([
    epoch,epoch+2000000n,epoch+9000000n,epoch+12000000n,epoch+20000000n,epoch+21000000n]));
  assert.equal(result.summedPassGpuMs,6);
  assert.deepEqual(result.rows,[
    {entry:'propose',dispatches:2,gpuMs:3,maxDispatchGpuMs:2},
    {entry:'retire',dispatches:1,gpuMs:3,maxDispatchGpuMs:3}]);
});
test('timestamp census rejects missing or reversed intervals and admits zero-duration quantization',()=>{
  assert.throws(()=>summarizeGpuTimestamps(['a'],new BigUint64Array()),/count/);
  assert.throws(()=>summarizeGpuTimestamps(['a'],new BigUint64Array([3n,2n])),/reversed/);
  assert.equal(summarizeGpuTimestamps(['a'],new BigUint64Array([3n,3n])).summedPassGpuMs,0);
});
const p = (name, inner) => ({source_entry:name,cycle:{group:0,repeat:1035,...(inner?{inner}: {})}});
const cycle = {group:4,repeat:128};
const fixture = () => [p('begin_atlas'),p('initialize'),p('repair_schedule',cycle),p('repair_apply',cycle),p('repair_advance',cycle),p('repair_finish'),p('reserve_tile')];
test('diagnostic prefix preserves entire compiled cycles',()=>{
  assert.deepEqual(tileReplayPrefix(fixture(),'initialize'),[{passes:[p('initialize')],repeat:1}]);
  assert.equal(tileReplayPrefix(fixture(),'repair_advance')[1].repeat,128);
  assert.throws(()=>tileReplayPrefix(fixture(),'repair_apply'),/block boundary/);
  assert.throws(()=>tileReplayPrefix(fixture(),'missing'),/block boundary/);
});
test('diagnostic submission bounds fail before touching the GPU',async()=>{
  const job=[0,0,0,0,1,1,1];
  await assert.rejects(diagnoseAtlasPrefix(null,null,[]),/valid atlas job/);
  for (const bound of [0,257,1.5,NaN])
    await assert.rejects(diagnoseAtlasPrefix(null,null,job,{passesPerSubmission:bound}),/submission bound/);
});
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
  await assert.rejects(replayAtlasTile(null,null,[],{shape:'triangle'}),/valid triangle job/);
  await assert.rejects(replayAtlasTile(null,null,[0,0,8,8,1,1,9],{shape:'triangle',rounds:[2,1]}),/checkpoints/);
  await assert.rejects(replayAtlasTile(null,null,[],{shape:'hexagon'}),/unknown atlas shape/);
});
