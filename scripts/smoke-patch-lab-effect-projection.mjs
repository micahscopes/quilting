import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const read = path => readFileSync(join(repository, path), 'utf8');
const appStore = read('crates/hyperscope-app/src/lib.rs');
const app = read('crates/hyperscope-app/src/patch_lab.rs');
const patchLabWeb = read('crates/hyperscope-web/src/patch_lab.rs');
const renderWeb = read('crates/hyperscope-web/src/render_controls.rs');

for (const required of [
  'pub struct PatchLabEffects(Vec<PatchLabEffect>);',
  'pub fn from_commit(commit: &AppCommit) -> Self',
  'pub fn from_effects(effects: &[AppEffect]) -> Self',
  'AppEffect::PatchLab(effect) => Some(effect.clone())',
  'pub fn as_slice(&self) -> &[PatchLabEffect]',
  'pub fn into_vec(self) -> Vec<PatchLabEffect>',
]) {
  assert.ok(app.includes(required), `shared Patch Lab effect projection is missing ${required}`);
}

assert.ok(appStore.includes('let effects = PatchLabEffects::from_commit(&commit);'),
  'the typed AppStore Patch Lab port does not own the shared effect projection');
assert.ok(patchLabWeb.includes('store.set_patch_lab_session(intent)?'),
  'Patch Lab controls do not delegate to the typed AppStore port');
assert.ok(renderWeb.includes('PatchLabEffects::from_commit(&commit).into_vec()'),
  'render controls do not delegate to the shared effect projection');

for (const [label, source] of [
  ['Patch Lab controls', patchLabWeb],
  ['render controls', renderWeb],
]) {
  for (const retired of [
    'AppEffect::PatchLab',
    '.filter_map(|effect|',
    'commit.effects.into_iter()',
  ]) {
    assert.equal(source.includes(retired), false,
      `${label} still interprets generic effects through ${retired}`);
  }
}

console.log('Shared Patch Lab effect projection source smoke passed');
