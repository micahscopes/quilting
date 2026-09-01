// Browser resource lifecycle for the optional local Blender carrier.
//
// This adapter owns live relay and generated WASM peer handles only. Project
// identity and proposal authority become semantic inside Rust; relay URLs and
// bearer credentials remain runtime-only inputs consumed by BrowserLocalPeerRelay.

import { BrowserLocalPeerRelay } from './local_peer_browser.mjs';

export class BrowserLocalPeerSessionError extends Error {}

export class BrowserLocalPeerSession {
  constructor({
    app,
    relayFactory = options => new BrowserLocalPeerRelay(options),
    nowSeconds = () => app.frameElapsedSeconds(),
    onReceipt = null,
    onStatus = null,
  }) {
    if (!app || typeof app.receiveLocalPeerEnvelope !== 'function') {
      throw new BrowserLocalPeerSessionError(
        'local peer session requires the generated Rust/WASM application boundary',
      );
    }
    if (typeof relayFactory !== 'function' || typeof nowSeconds !== 'function') {
      throw new BrowserLocalPeerSessionError('local peer session adapters must be callable');
    }
    this.app = app;
    this.relayFactory = relayFactory;
    this.nowSeconds = nowSeconds;
    this.onReceipt = onReceipt;
    this.onStatus = onStatus;
    this.relay = null;
    this.durablePeer = null;
    this.opening = false;
    this.generation = 0;
  }

  isConnected() {
    return this.relay != null;
  }

  isOpening() {
    return this.opening;
  }

  async connect({
    baseUrl,
    token,
    mode = 'legacy',
    projectId = null,
    proposalRole = 'replica',
  }) {
    if (this.relay || this.opening) {
      throw new BrowserLocalPeerSessionError('local peer session is already active');
    }
    if (!['legacy', 'durable'].includes(mode)) {
      throw new BrowserLocalPeerSessionError('local peer session mode must be legacy or durable');
    }
    if (!['replica', 'admission_authority'].includes(proposalRole)) {
      throw new BrowserLocalPeerSessionError('durable proposal role is invalid');
    }
    if (mode === 'durable'
        && typeof this.app.openDurableAuthoredPeerWithRole !== 'function') {
      throw new BrowserLocalPeerSessionError(
        'this WASM artifact was built without durable authored history',
      );
    }

    const generation = ++this.generation;
    this.opening = true;
    let durablePeer = null;
    try {
      if (mode === 'durable') {
        if (typeof projectId !== 'string' || projectId.trim().length === 0) {
          throw new BrowserLocalPeerSessionError('durable project UUID is required');
        }
        durablePeer = await this.app.openDurableAuthoredPeerWithRole(
          projectId.trim(),
          proposalRole,
        );
        if (generation !== this.generation) {
          freeDurablePeer(durablePeer);
          durablePeer = null;
          throw new BrowserLocalPeerSessionError('local peer opening was cancelled');
        }
      }

      const relay = this.relayFactory({
        baseUrl,
        token,
        app: this.app,
        durablePeer,
        authoredProposalPolicy: mode === 'legacy'
          ? 'legacy'
          : proposalRole === 'admission_authority' ? 'admit' : 'ignore',
        nowSeconds: this.nowSeconds,
        onReceipt: this.onReceipt,
        onStatus: this.onStatus,
      });
      if (!relay || typeof relay.start !== 'function' || typeof relay.stop !== 'function') {
        throw new BrowserLocalPeerSessionError('local peer relay factory returned no lifecycle');
      }
      this.relay = relay;
      this.durablePeer = durablePeer;
      durablePeer = null;
      void relay.start();
      return relay.snapshot?.() ?? null;
    } catch (error) {
      const activeRelay = this.relay;
      const activePeer = this.durablePeer;
      this.relay = null;
      this.durablePeer = null;
      if (activeRelay) {
        try {
          await activeRelay.stop();
        } catch {}
      }
      freeDurablePeer(durablePeer ?? activePeer);
      throw error;
    } finally {
      if (generation === this.generation) this.opening = false;
    }
  }

  async disconnect() {
    this.generation += 1;
    this.opening = false;
    const relay = this.relay;
    const durablePeer = this.durablePeer;
    this.relay = null;
    this.durablePeer = null;
    try {
      if (relay) await relay.stop();
    } finally {
      freeDurablePeer(durablePeer);
    }
  }

  sendAppliedAuthoredEnvelope(envelopeJson) {
    if (!this.relay) throw new BrowserLocalPeerSessionError('local peer is disconnected');
    return this.relay.sendAppliedAuthoredEnvelope(envelopeJson);
  }

  sendPresenceEnvelope(envelopeJson) {
    if (!this.relay) throw new BrowserLocalPeerSessionError('local peer is disconnected');
    return this.relay.sendPresenceEnvelope(envelopeJson);
  }
}

function freeDurablePeer(peer) {
  if (peer == null) return;
  if (typeof peer.free !== 'function') {
    throw new BrowserLocalPeerSessionError('generated durable peer has no free lifecycle');
  }
  peer.free();
}
