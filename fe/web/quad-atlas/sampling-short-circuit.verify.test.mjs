import test from 'node:test';
import assert from 'node:assert/strict';

// Finite decision model, not a sampler or a GPU performance measurement.
// An event records the two conflict facts contributed by one immutable
// neighbor. Coordinates/priority still decide those facts in the Fe provider.
const retirement = (accepted, winner, winningNeighbor) =>
  accepted ? 'rejected' : winner ? 'accepted' : winningNeighbor ? 'rejected' : 'pending';

test('short-circuit conflict queries preserve full-scan decisions', () => {
  for (let length = 0; length <= 6; ++length) {
    for (let word = 0; word < 4 ** length; ++word) {
      const events = Array.from({ length }, (_, i) => (word >>> (2 * i)) & 3);
      const allAccepted = events.some(e => e & 1);
      const allOther = events.some(e => e & 2);
      let accepted = false, preceding = false;
      for (const e of events) {
        if (accepted || preceding) break;
        accepted ||= Boolean(e & 1);
        preceding ||= Boolean(e & 2);
      }
      assert.equal(!accepted && !preceding, !allAccepted && !allOther);
      for (const winner of [false, true]) {
        let accepted = false, winningNeighbor = false;
        for (const e of events) {
          if (accepted || (!winner && winningNeighbor)) break;
          accepted ||= Boolean(e & 1);
          winningNeighbor ||= Boolean(e & 2);
        }
        assert.equal(retirement(accepted, winner, winningNeighbor),
          retirement(allAccepted, winner, allOther));
      }
    }
  }
});

test('a purported winner must still reject a later accepted neighbor', () => {
  // Do not assume that an inconsistent winner mask makes the accepted-first
  // precedence irrelevant. A blanket stop on either conflict would be wrong.
  assert.equal(retirement(true, true, true), 'rejected');
  assert.equal(retirement(false, true, true), 'accepted');
});

test('pending priority can reject irrelevant geometric queries without changing proposal facts',()=>{
  let baselineQueries=0,filteredQueries=0;
  for(const state of ['accepted','pending','rejected','invalid'])
    for(const precedes of [false,true])for(const conflicts of [false,true]) {
      const eligible=state==='accepted'||state==='pending';
      const baseline=eligible&&conflicts
        ? [state==='accepted',state==='pending'&&precedes] : [false,false];
      if(eligible)baselineQueries++;
      const relevant=eligible&&(state==='accepted'||precedes);
      if(relevant)filteredQueries++;
      const filtered=relevant&&conflicts
        ? [state==='accepted',state==='pending'] : [false,false];
      assert.deepEqual(filtered,baseline);
    }
  assert.equal(baselineQueries,8);assert.equal(filteredQueries,6);
  // A finite truth table, not a measured runtime saving or neighbor frequency.
});
