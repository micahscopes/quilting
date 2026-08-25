// Thin browser carrier for Hyperscope's generated Rust/WASM peer boundary.
// This module owns HTTP delivery cursors only. It never allocates application
// revisions, parses authored commands, persists history, or repairs gaps.

export const DEFAULT_LOCAL_PEER_RELAY_URL = 'http://127.0.0.1:42117';
export const DEFAULT_LOCAL_PEER_POLL_INTERVAL_MS = 50;
export const DEFAULT_LOCAL_PEER_POLL_LIMIT = 256;
export const MAX_LOCAL_PEER_RESPONSE_BYTES = 2 * 1024 * 1024;

const MAX_U64 = (1n << 64n) - 1n;

export class BrowserLocalPeerRelayError extends Error {}

export class BrowserLocalPeerRelay {
  #token;

  constructor({
    baseUrl = DEFAULT_LOCAL_PEER_RELAY_URL,
    token,
    app,
    fetchImpl = globalThis.fetch?.bind(globalThis),
    nowSeconds = () => performance.now() / 1000,
    pollIntervalMs = DEFAULT_LOCAL_PEER_POLL_INTERVAL_MS,
    pollLimit = DEFAULT_LOCAL_PEER_POLL_LIMIT,
    outboundCapacity = 256,
    onReceipt = null,
    onStatus = null,
  }) {
    this.baseUrl = validateBaseUrl(baseUrl);
    this.#token = validateToken(token);
    if (!app
        || typeof app.receiveLocalPeerEnvelope !== 'function'
        || typeof app.recordLocalAuthoredEnvelope !== 'function') {
      throw new BrowserLocalPeerRelayError(
        'browser relay requires the generated Rust/WASM peer application boundary',
      );
    }
    if (typeof fetchImpl !== 'function') {
      throw new BrowserLocalPeerRelayError('browser relay requires fetch');
    }
    if (typeof nowSeconds !== 'function') {
      throw new BrowserLocalPeerRelayError('browser relay clock must be callable');
    }
    if (!Number.isInteger(pollIntervalMs)
        || pollIntervalMs < 5 || pollIntervalMs > 10_000) {
      throw new BrowserLocalPeerRelayError('poll interval must be in [5, 10000] ms');
    }
    if (!Number.isInteger(pollLimit) || pollLimit < 1 || pollLimit > 1024) {
      throw new BrowserLocalPeerRelayError('poll limit must be in [1, 1024]');
    }
    if (!Number.isInteger(outboundCapacity) || outboundCapacity < 1) {
      throw new BrowserLocalPeerRelayError('outbound capacity must be positive');
    }
    if (onReceipt != null && typeof onReceipt !== 'function') {
      throw new BrowserLocalPeerRelayError('receipt observer must be callable');
    }
    if (onStatus != null && typeof onStatus !== 'function') {
      throw new BrowserLocalPeerRelayError('status observer must be callable');
    }
    this.app = app;
    this.fetchImpl = fetchImpl;
    this.nowSeconds = nowSeconds;
    this.pollIntervalMs = pollIntervalMs;
    this.pollLimit = pollLimit;
    this.outboundCapacity = outboundCapacity;
    this.onReceipt = onReceipt;
    this.onStatus = onStatus;
    this.outbound = [];
    this.pendingOutbound = null;
    this.abortController = null;
    this.loopPromise = null;
    this.stopping = false;
    this.degraded = false;
    this.status = {
      enabled: true,
      state: 'stopped',
      generation: null,
      cursor: '0',
      sentFrames: 0,
      receivedFrames: 0,
      appliedFrames: 0,
      ignoredFrames: 0,
      gaps: 0,
      restarts: 0,
      lastReceipt: null,
      lastError: null,
      lastActivitySeconds: null,
    };
  }

  start() {
    if (this.loopPromise) return this.loopPromise;
    this.stopping = false;
    this.abortController = new AbortController();
    this.setStatus({ state: 'connecting', lastError: null });
    this.loopPromise = this.run().finally(() => {
      this.loopPromise = null;
      this.abortController = null;
      if (this.stopping) this.setStatus({ state: 'stopped', lastError: null });
    });
    return this.loopPromise;
  }

  async stop() {
    this.stopping = true;
    this.abortController?.abort();
    const loop = this.loopPromise;
    if (loop) await loop;
    else this.setStatus({ state: 'stopped', lastError: null });
    const error = new BrowserLocalPeerRelayError('browser relay stopped before delivery');
    if (this.pendingOutbound) this.pendingOutbound.reject(error);
    for (const pending of this.outbound) pending.reject(error);
    this.pendingOutbound = null;
    this.outbound.length = 0;
  }

  snapshot() {
    return Object.freeze({
      ...this.status,
      queuedFrames: this.outbound.length + (this.pendingOutbound ? 1 : 0),
    });
  }

  sendAppliedAuthoredEnvelope(envelopeJson) {
    validateJsonText(envelopeJson, 'local authored envelope');
    this.reserveOutbound();
    // Rust validates and records the already-applied envelope before the
    // carrier can possibly poll its echo on a later turn.
    try {
      this.app.recordLocalAuthoredEnvelope(envelopeJson);
    } catch (error) {
      throw new BrowserLocalPeerRelayError(
        `Rust rejected the local authored envelope: ${errorMessage(error)}`,
      );
    }
    return this.enqueue(`{"lane":"authored","envelope":${envelopeJson}}`);
  }

  sendPresenceEnvelope(envelopeJson) {
    validateJsonText(envelopeJson, 'local presence envelope');
    this.reserveOutbound();
    const frameJson = `{"lane":"presence","envelope":${envelopeJson}}`;
    try {
      this.app.receiveLocalPeerEnvelope(this.nowSeconds(), frameJson);
    } catch (error) {
      throw new BrowserLocalPeerRelayError(
        `Rust rejected the local presence envelope: ${errorMessage(error)}`,
      );
    }
    return this.enqueue(frameJson);
  }

  reserveOutbound() {
    const occupied = this.outbound.length + (this.pendingOutbound ? 1 : 0);
    if (occupied >= this.outboundCapacity) {
      throw new BrowserLocalPeerRelayError('browser relay outbound queue is full');
    }
  }

  enqueue(frameJson) {
    let resolve;
    let reject;
    const completion = new Promise((accepted, failed) => {
      resolve = accepted;
      reject = failed;
    });
    this.outbound.push({ frameJson, resolve, reject });
    return completion;
  }

  async run() {
    let retryMs = this.pollIntervalMs;
    while (!this.stopping) {
      try {
        await this.flushOutbound();
        const hasMore = await this.pollOnce();
        this.setStatus({
          state: this.degraded ? 'degraded' : 'connected',
          lastError: null,
        });
        retryMs = this.pollIntervalMs;
        if (!hasMore) await this.delay(this.pollIntervalMs);
      } catch (error) {
        if (this.stopping && error?.name === 'AbortError') break;
        this.setStatus({ state: 'error', lastError: errorMessage(error) });
        await this.delay(retryMs);
        retryMs = Math.min(Math.max(retryMs * 2, 100), 2000);
      }
    }
  }

  async flushOutbound() {
    for (let count = 0; count < 32; count += 1) {
      if (!this.pendingOutbound) this.pendingOutbound = this.outbound.shift() || null;
      if (!this.pendingOutbound) return;
      const response = await this.requestJson(
        'POST',
        '/v1/frame',
        this.pendingOutbound.frameJson,
      );
      generation(response.generation);
      decimalCursor(response.cursor, 'posted cursor');
      const completed = this.pendingOutbound;
      this.pendingOutbound = null;
      this.status.sentFrames += 1;
      this.markActivity();
      completed.resolve(response);
      this.emitStatus();
    }
  }

  async pollOnce() {
    const requestedAfter = decimalCursor(this.status.cursor, 'local cursor');
    const response = await this.requestJson(
      'GET',
      `/v1/frames?after=${requestedAfter}&limit=${this.pollLimit}`,
      null,
    );
    return this.acceptBatch(response, requestedAfter);
  }

  acceptBatch(batch, requestedAfter) {
    if (!batch || typeof batch !== 'object' || Array.isArray(batch)) {
      throw new BrowserLocalPeerRelayError('relay poll response must be an object');
    }
    const nextGeneration = generation(batch.generation);
    const acknowledged = decimalCursor(batch.requestedAfter, 'requested cursor');
    if (acknowledged !== requestedAfter) {
      throw new BrowserLocalPeerRelayError(
        'relay response acknowledged the wrong requested cursor',
      );
    }
    const latest = decimalCursor(batch.latestCursor, 'latest cursor');
    const resumeAfter = decimalCursor(batch.resumeAfter, 'resume cursor');
    if (resumeAfter > latest) {
      throw new BrowserLocalPeerRelayError('relay resume cursor exceeds latest');
    }
    if (batch.oldestCursor != null) {
      const oldest = decimalCursor(batch.oldestCursor, 'oldest cursor');
      if (oldest === 0n || oldest > latest) {
        throw new BrowserLocalPeerRelayError(
          'relay oldest cursor is outside retained history',
        );
      }
    }
    if (typeof batch.gap !== 'boolean'
        || typeof batch.hasMore !== 'boolean'
        || !Array.isArray(batch.frames)) {
      throw new BrowserLocalPeerRelayError(
        'relay response has invalid gap, pagination, or frames',
      );
    }

    const deliveries = [];
    let previous = batch.gap && resumeAfter > requestedAfter
      ? resumeAfter
      : requestedAfter;
    for (const item of batch.frames) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        throw new BrowserLocalPeerRelayError('relay delivery must be an object');
      }
      const cursor = decimalCursor(item.cursor, 'delivery cursor');
      if (cursor !== previous + 1n || cursor > latest) {
        throw new BrowserLocalPeerRelayError(
          'relay delivery cursors must be contiguous through latest',
        );
      }
      if (typeof item.frameJson !== 'string' || item.frameJson.length === 0) {
        throw new BrowserLocalPeerRelayError(
          'relay delivery frame must be exact JSON text',
        );
      }
      deliveries.push({ cursor, frameJson: item.frameJson });
      previous = cursor;
    }
    if (batch.hasMore
        && (deliveries.length === 0 || deliveries.at(-1).cursor >= latest)) {
      throw new BrowserLocalPeerRelayError(
        'relay pagination cannot make forward progress',
      );
    }
    if (!batch.hasMore
        && deliveries.length > 0
        && deliveries.at(-1).cursor !== latest) {
      throw new BrowserLocalPeerRelayError('final relay page must end at latest');
    }

    if (this.status.generation != null && nextGeneration !== this.status.generation) {
      this.degraded = true;
      this.setStatus({
        generation: nextGeneration,
        cursor: '0',
        restarts: this.status.restarts + 1,
        gaps: this.status.gaps + 1,
      });
      return true;
    }
    if (this.status.generation == null) {
      this.setStatus({ generation: nextGeneration });
    }

    if (batch.gap) {
      this.degraded = true;
      this.setStatus({ gaps: this.status.gaps + 1 });
    }

    for (const delivery of deliveries) {
      let receipt;
      try {
        receipt = this.app.receiveLocalPeerEnvelope(
          this.nowSeconds(),
          delivery.frameJson,
        );
      } catch (error) {
        // Do not advance past a semantic frame Rust rejected. Re-polling the
        // same cursor is noisy but cannot silently discard later authored work.
        throw new BrowserLocalPeerRelayError(
          `Rust rejected relay cursor ${delivery.cursor}: ${errorMessage(error)}`,
        );
      }
      this.setStatus({
        cursor: delivery.cursor.toString(),
        receivedFrames: this.status.receivedFrames + 1,
        appliedFrames: this.status.appliedFrames
          + (receipt?.disposition === 'applied' ? 1 : 0),
        ignoredFrames: this.status.ignoredFrames
          + (receipt?.disposition === 'applied' ? 0 : 1),
        lastReceipt: receipt ?? null,
      });
      this.markActivity();
      try {
        this.onReceipt?.(receipt, delivery.frameJson);
      } catch (error) {
        console.warn('Local peer receipt observer failed:', error);
      }
    }

    let nextCursor = deliveries.length > 0
      ? deliveries.at(-1).cursor
      : requestedAfter;
    if (batch.gap) {
      if (deliveries.length === 0) nextCursor = resumeAfter;
    }
    this.setStatus({ cursor: nextCursor.toString() });
    return batch.hasMore || nextCursor < latest;
  }

  async requestJson(method, path, body) {
    const headers = {
      Authorization: `Bearer ${this.#token}`,
      Accept: 'application/json',
    };
    if (body != null) headers['Content-Type'] = 'application/json';
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body ?? undefined,
      signal: this.abortController?.signal,
    });
    if (!response?.ok) {
      let detail = '';
      try {
        detail = (await response.text()).slice(0, 512);
      } catch {}
      throw new BrowserLocalPeerRelayError(
        `relay HTTP ${response?.status ?? 'failure'}${detail ? `: ${detail}` : ''}`,
      );
    }
    const payload = await response.arrayBuffer();
    if (payload.byteLength > MAX_LOCAL_PEER_RESPONSE_BYTES) {
      throw new BrowserLocalPeerRelayError('relay response exceeds the byte limit');
    }
    try {
      const value = JSON.parse(new TextDecoder().decode(payload));
      if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('not an object');
      }
      return value;
    } catch (error) {
      throw new BrowserLocalPeerRelayError(
        `relay returned invalid JSON: ${errorMessage(error)}`,
      );
    }
  }

  async delay(milliseconds) {
    if (this.stopping) return;
    const signal = this.abortController?.signal;
    await new Promise(resolve => {
      const timer = setTimeout(resolve, milliseconds);
      signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        resolve();
      }, { once: true });
    });
  }

  setStatus(changes) {
    Object.assign(this.status, changes);
    this.emitStatus();
  }

  markActivity() {
    this.status.lastActivitySeconds = this.nowSeconds();
  }

  emitStatus() {
    try {
      this.onStatus?.(this.snapshot());
    } catch (error) {
      console.warn('Local peer status observer failed:', error);
    }
  }
}

function validateBaseUrl(value) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch (error) {
    throw new BrowserLocalPeerRelayError(
      `relay URL is invalid: ${errorMessage(error)}`,
    );
  }
  if (!['http:', 'https:'].includes(parsed.protocol)
      || !parsed.hostname
      || parsed.username || parsed.password
      || parsed.pathname !== '/' || parsed.search || parsed.hash) {
    throw new BrowserLocalPeerRelayError(
      'relay URL must be one HTTP(S) origin without credentials or a path',
    );
  }
  return parsed.origin;
}

function validateToken(value) {
  if (typeof value !== 'string'
      || !/^[A-Za-z0-9_.~-]{1,256}$/.test(value)) {
    throw new BrowserLocalPeerRelayError(
      'relay token must be 1..256 URL-safe ASCII characters',
    );
  }
  return value;
}

function validateJsonText(value, context) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new BrowserLocalPeerRelayError(`${context} must be JSON text`);
  }
}

function decimalCursor(value, context) {
  if (typeof value !== 'string' || !/^(0|[1-9][0-9]*)$/.test(value)) {
    throw new BrowserLocalPeerRelayError(`${context} must be canonical decimal text`);
  }
  const cursor = BigInt(value);
  if (cursor > MAX_U64) {
    throw new BrowserLocalPeerRelayError(`${context} exceeds an unsigned 64-bit integer`);
  }
  return cursor;
}

function generation(value) {
  if (typeof value !== 'string'
      || !/^[A-Za-z0-9_.-]{1,128}$/.test(value)) {
    throw new BrowserLocalPeerRelayError('relay generation is invalid');
  }
  return value;
}

function errorMessage(error) {
  return error?.message || String(error);
}
