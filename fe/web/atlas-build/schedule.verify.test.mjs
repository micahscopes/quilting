import test from 'node:test';
import assert from 'node:assert/strict';
import {atlasScheduleCensus} from './schedule.verify.mjs';
const pass = (name,cycle,indirect=false) => ({source_entry:name,layout:{mode:'compute'},cycle,
  ...(indirect?{dispatch_indirect:{resource:'command'}}:{dispatch:[1,1,1]})});
test('count definitions separately from nested scheduled commands',()=>{
  const cycle={group:0,repeat:3,inner:{group:1,repeat:100}};
  const r=atlasScheduleCensus({passes:[pass('begin'),pass('schedule',cycle),pass('work',cycle,true)]});
  assert.equal(r.passDefinitions,3); assert.equal(r.scheduledDispatchCommands,601);
  assert.equal(r.indirectDispatchCommands,300);
});
test('do not guess command work from indirect dispatches',()=>{
  assert.equal(atlasScheduleCensus({passes:[pass('unknown',null,true)]}).scheduledDispatchCommands,1);
});
test('reject schedules whose counts cannot be interpreted exactly',()=>{
  for (const p of [
    {...pass('taper'),taper:{}},
    {...pass('bad-repeat'),repeat:0},
    pass('overflow',{group:0,repeat:Number.MAX_SAFE_INTEGER,inner:{group:1,repeat:2}}),
    pass('recursive',{group:0,repeat:2,inner:{group:0,repeat:2}}),
    {...pass('ambiguous'),dispatch_indirect:{resource:'command'}},
  ]) assert.throws(()=>atlasScheduleCensus({passes:[p]}));
  assert.throws(()=>atlasScheduleCensus({passes:[pass('a',{group:0,repeat:2}),pass('b',{group:0,repeat:3})]}));
});
