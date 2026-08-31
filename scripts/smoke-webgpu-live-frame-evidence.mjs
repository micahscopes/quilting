#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = dirname(dirname(fileURLToPath(import.meta.url)));
const backend = readFileSync(
  join(repository, 'crates/quilting-wasm/src/webgpu_backend.rs'),
  'utf8',
);
const renderer = readFileSync(
  join(repository, 'crates/quilting-wasm/src/main_renderer.rs'),
  'utf8',
);

for (const required of [
  'frame_evidence_requested: bool',
  'presentation_frame_admitted: bool',
  'last_presentation_style: Option<RenderStyle>',
  'frame_evidence: Option<FrameEvidenceMetadata>',
  'fn frame_destination(',
  'backend.presentation.is_some(),\n            frame_evidence_requested,',
  'backend.presentation_frame_admitted = true',
  'backend.last_presentation_style = Some(style)',
  'backend.frame_evidence = Some(FrameEvidenceMetadata',
  'pub(crate) fn request_frame_evidence()',
  'backend.frame_evidence_requested = true',
  'let evidence = backend.frame_evidence.ok_or_else',
]) {
  assert.ok(backend.includes(required), `live frame evidence is missing ${required}`);
}

for (const required of [
  'crate::webgpu_backend::frame_evidence_ready()',
  'crate::webgpu_backend::request_frame_evidence()',
  'canonicalize_incumbent_clear_alpha(',
  '&mut webgpu.bytes',
  'evidence_clear_alpha_canonicalization_respects_padded_rows',
]) {
  assert.ok(renderer.includes(required), `backend evidence adapter is missing ${required}`);
}

const submitStart = backend.indexOf('pub(crate) fn submit_frame(');
const submitEnd = backend.indexOf('\npub(crate) struct StagedWebGpuFrameEvidence', submitStart);
const submit = backend.slice(submitStart, submitEnd);
assert.ok(submitStart >= 0 && submitEnd > submitStart, 'could not isolate frame submission');
assert.ok(
  submit.indexOf('let presentation_frame = destination == FrameDestination::Presentation;')
    < submit.indexOf('if presentation_frame {'),
  'the destination decision must precede target selection and encoding',
);
assert.ok(
  submit.includes('LiveFrameDisposition::ShadowSubmitted(logical_submission)'),
  'the explicit offscreen frame must continue into the WebGL oracle path',
);

console.log('WebGPU live frame-evidence source smoke passed');
