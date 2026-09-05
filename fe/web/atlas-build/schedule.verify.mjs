// Static diagnostic census, not a scheduler or an estimate of GPU execution.
// In particular, zero-work indirect commands still occupy scheduled positions.
export function atlasScheduleCensus(manifest) {
  if (!Array.isArray(manifest.passes)) throw Error('missing pass definitions');
  const positive = n => Number.isSafeInteger(n) && n > 0;
  const cycles = new Map();
  const rows = manifest.passes.map(pass => {
    if (pass.layout?.mode !== 'compute' || pass.taper != null)
      throw Error('census supports untapered compute graphs only');
    let commands = pass.repeat ?? 1;
    if (!positive(commands)) throw Error('invalid pass repetition');
    const path = [], seen = new Set();
    for (let c=pass.cycle; c != null; c=c.inner) {
      if (!Number.isSafeInteger(c.group) || c.group < 0 || !positive(c.repeat) || seen.has(c.group))
        throw Error('invalid nested cycle');
      seen.add(c.group); path.push(c.group);
      const identity = path.join('/');
      if (cycles.has(identity) && cycles.get(identity) !== c.repeat)
        throw Error('inconsistent cycle repetitions');
      cycles.set(identity,c.repeat);
      commands *= c.repeat;
      if (!positive(commands)) throw Error('schedule count exceeds exact integer range');
    }
    const indirect = pass.dispatch_indirect != null;
    if (indirect === (pass.dispatch != null)) throw Error('ambiguous or missing dispatch source');
    return {entry:pass.source_entry, scheduledDispatchCommands:commands, indirect};
  });
  const sum = rows.reduce((n,r)=>n+r.scheduledDispatchCommands,0);
  if (!Number.isSafeInteger(sum)) throw Error('schedule total exceeds exact integer range');
  return {passDefinitions:rows.length, scheduledDispatchCommands:sum,
    indirectDispatchCommands:rows.filter(r=>r.indirect).reduce((n,r)=>n+r.scheduledDispatchCommands,0), rows};
}
