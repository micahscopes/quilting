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
  if (effect.type === 'install_primary_scene') {
    return Object.freeze({
      type: effect.type,
      requestId: requiredString(effect.request_id, 'install request ID'),
      assetId: requiredString(effect.asset_id, 'install asset ID'),
    });
  }
  if (effect.type === 'cancel_primary_scene_install') {
    return Object.freeze({
      type: effect.type,
      requestId: requiredString(effect.request_id, 'install cancellation request ID'),
      assetId: requiredString(effect.asset_id, 'install cancellation asset ID'),
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
    const cancellations = observedEffects.filter(effect =>
      effect.type === 'cancel_asset_load'
      || effect.type === 'cancel_primary_scene_install');
    for (const effect of cancellations) {
      const cancelled = this.jobs.get(effect.requestId);
      if (!cancelled || cancelled.assetId !== effect.assetId) {
        mismatches.push(`cancellation did not match active request ${effect.requestId}`);
      }
    }
    const previousPrimary = scope === 'primary_scene' ? this.primary : null;
    if (this.implementation !== 'js' && previousPrimary) {
      let expectedCancellation = null;
      if (previousPrimary.disposition === null) {
        expectedCancellation = 'cancel_asset_load';
      } else if (previousPrimary.disposition === 'applied'
          && previousPrimary.installRequested
          && previousPrimary.installDisposition === null) {
        expectedCancellation = 'cancel_primary_scene_install';
      }
      if (expectedCancellation && !cancellations.some(effect =>
        effect.type === expectedCancellation
        && effect.requestId === previousPrimary.requestId
        && effect.assetId === previousPrimary.assetId)) {
        mismatches.push(`request commit omitted ${expectedCancellation} for the active primary job`);
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
      installRequested: false,
      installDisposition: null,
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

    for (const effect of cancellations) {
      const cancelled = this.jobs.get(effect.requestId);
      if (!cancelled || cancelled.assetId !== effect.assetId) {
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

  beginInstall(token, commit) {
    if (!token || this.jobs.get(token.requestId) !== token) {
      if (this.implementation === 'rust') {
        throw new Error('primary scene install has no matching active asset job');
      }
      return { mismatches: ['primary scene install has no matching active asset job'] };
    }
    const observedEffects = this.implementation === 'js'
      ? []
      : commitEffects(commit).map(validateEffect);
    const installs = observedEffects.filter(effect => effect.type === 'install_primary_scene');
    const matchingInstall = installs.find(effect => effect.requestId === token.requestId);
    const mismatches = [];
    if (this.implementation !== 'js') {
      if (installs.length !== 1 || !matchingInstall) {
        mismatches.push('completion commit must contain exactly one matching install effect');
      } else if (matchingInstall.assetId !== token.assetId) {
        mismatches.push('install asset ID diverged');
      }
    }
    if (this.implementation === 'rust' && mismatches.length > 0) {
      throw new Error(mismatches.join('; '));
    }
    token.installRequested = this.implementation === 'js' || !!matchingInstall;
    return { mismatches };
  }

  recordCompletion(token, disposition) {
    if (!token || this.jobs.get(token.requestId) !== token) return 'unobserved';
    token.disposition = requiredString(disposition, 'completion disposition');
    return token.disposition;
  }

  recordInstallCompletion(token, disposition) {
    if (!token || this.jobs.get(token.requestId) !== token) return 'unobserved';
    token.installDisposition = requiredString(disposition, 'install completion disposition');
    return token.installDisposition;
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
    if (token.scope === 'primary_scene' && !token.installRequested) return false;
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
