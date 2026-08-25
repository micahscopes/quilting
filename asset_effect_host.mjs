const IMPLEMENTATIONS = new Set(['js', 'shadow', 'rust']);
const SCOPES = new Set(['asset', 'primary_scene']);

function requiredString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${label} must be a nonempty string`);
  }
  return value;
}

function commitEffects(commit) {
  if (!commit || !Array.isArray(commit.effects)) {
    throw new TypeError('Rust asset request commit must contain an effects array');
  }
  return commit.effects;
}

function validateEffect(effect) {
  if (!effect || typeof effect !== 'object') {
    throw new TypeError('Rust asset effect must be an object');
  }
  if (effect.type === 'fetch_asset') {
    return Object.freeze({
      type: effect.type,
      requestId: requiredString(effect.request_id, 'fetch request ID'),
      assetId: requiredString(effect.asset_id, 'fetch asset ID'),
      uri: requiredString(effect.uri, 'fetch URI'),
    });
  }
  if (effect.type === 'cancel_asset_load') {
    return Object.freeze({
      type: effect.type,
      requestId: requiredString(effect.request_id, 'cancellation request ID'),
      assetId: requiredString(effect.asset_id, 'cancellation asset ID'),
    });
  }
  throw new TypeError(`unsupported Rust asset effect ${JSON.stringify(effect.type)}`);
}

/**
 * Thin browser host for Rust AppEffects.
 *
 * Rust chooses request/cancellation semantics. This host owns only platform
 * resources: AbortControllers, logical-URI acquisition, and the fence that
 * prevents an obsolete primary parse from being installed after a newer
 * request. It deliberately does not parse models or mutate renderer state.
 */
export class BrowserAssetEffectHost {
  constructor(implementation) {
    if (!IMPLEMENTATIONS.has(implementation)) {
      throw new TypeError(`unsupported asset implementation ${JSON.stringify(implementation)}`);
    }
    this.implementation = implementation;
    this.jobs = new Map();
    this.primary = null;
    this.installTail = Promise.resolve();
  }

  begin({
    requestId,
    assetId,
    uri,
    source,
    scope = 'asset',
    commit = null,
  }) {
    requestId = requiredString(requestId, 'request ID');
    assetId = requiredString(assetId, 'asset ID');
    uri = requiredString(uri, 'asset URI');
    source = requiredString(source, 'asset source');
    if (!SCOPES.has(scope)) {
      throw new TypeError(`unsupported asset scope ${JSON.stringify(scope)}`);
    }

    const observedEffects = this.implementation === 'js'
      ? []
      : commitEffects(commit).map(validateEffect);
    const fetches = observedEffects.filter(effect => effect.type === 'fetch_asset');
    const matchingFetch = fetches.find(effect => effect.requestId === requestId);
    const mismatches = [];
    if (this.implementation !== 'js') {
      if (fetches.length !== 1 || !matchingFetch) {
        mismatches.push('request commit must contain exactly one matching fetch effect');
      } else {
        if (matchingFetch.assetId !== assetId) mismatches.push('fetch asset ID diverged');
        if (matchingFetch.uri !== uri) mismatches.push('fetch URI diverged');
      }
    }
    if (this.implementation === 'rust' && mismatches.length > 0) {
      throw new Error(mismatches.join('; '));
    }

    const controller = new AbortController();
    const token = {
      requestId,
      assetId,
      requestedUri: uri,
      uri: this.implementation === 'rust' && matchingFetch ? matchingFetch.uri : uri,
      source,
      scope,
      signal: controller.signal,
      controller,
      superseded: false,
      disposition: null,
    };

    if (scope === 'primary_scene') {
      const previous = this.primary;
      if (previous && previous !== token) {
        previous.superseded = true;
        if (this.implementation === 'rust' && !previous.signal.aborted) {
          previous.controller.abort('superseded primary scene request');
        }
      }
      this.primary = token;
    }

    for (const effect of observedEffects) {
      if (effect.type !== 'cancel_asset_load') continue;
      const cancelled = this.jobs.get(effect.requestId);
      if (!cancelled || cancelled.assetId !== effect.assetId) {
        mismatches.push(`cancellation did not match active request ${effect.requestId}`);
        continue;
      }
      cancelled.superseded = true;
      if (this.implementation === 'rust' && !cancelled.signal.aborted) {
        cancelled.controller.abort('Rust cancelled asset request');
      }
    }
    this.jobs.set(requestId, token);
    return { token, mismatches };
  }

  recordCompletion(token, disposition) {
    if (!token || this.jobs.get(token.requestId) !== token) return 'unobserved';
    token.disposition = requiredString(disposition, 'completion disposition');
    return token.disposition;
  }

  mayProcess(token) {
    if (this.implementation !== 'rust') return true;
    if (!token || this.jobs.get(token.requestId) !== token) return false;
    if (token.signal.aborted || token.superseded) return false;
    return token.scope !== 'primary_scene' || this.primary === token;
  }

  mayInstall(token) {
    if (!this.mayProcess(token)) return false;
    if (this.implementation !== 'rust') return true;
    if (token.disposition !== 'applied') return false;
    return true;
  }

  isExpectedCancellation(token, error) {
    if (!token || this.implementation !== 'rust') return false;
    return token.signal.aborted || error?.name === 'AbortError';
  }

  runProcess(token, operation) {
    if (typeof operation !== 'function') {
      throw new TypeError('asset processing must be a function');
    }
    if (this.implementation !== 'rust') return operation();
    const turn = this.installTail.then(async () => {
      if (!this.mayProcess(token)) return false;
      return operation();
    });
    this.installTail = turn.catch(() => undefined);
    return turn;
  }

  runInstall(token, operation) {
    if (typeof operation !== 'function') {
      throw new TypeError('asset installation must be a function');
    }
    if (this.implementation !== 'rust') return operation();
    const turn = this.installTail.then(async () => {
      if (!this.mayInstall(token)) return false;
      return operation();
    });
    this.installTail = turn.catch(() => undefined);
    return turn;
  }
}
