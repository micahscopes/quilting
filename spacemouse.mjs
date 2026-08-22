// Dependency-free WebHID support for 3Dconnexion SpaceMouse devices.
//
// The Bluetooth SpaceMouse Wireless (256f:c63a) sends all six signed 16-bit
// axes in report 1. Older USB devices commonly split translation (report 1)
// and rotation (report 2), so the decoder accepts both layouts. Report 3 is a
// button bitmask. Camera policy intentionally stays in hyperscope.html; this
// module owns only permission, connection, decoding, and frame-rate sampling.

export const SPACEMOUSE_VENDOR_ID = 0x256f;
export const SPACEMOUSE_WIRELESS_BT_PRODUCT_ID = 0xc63a;
export const SPACEMOUSE_AXIS_MAX = 350;
export const SPACEMOUSE_LEFT_BUTTON = 1 << 0;
export const SPACEMOUSE_RIGHT_BUTTON = 1 << 1;

// Semantic camera axes produced by mapSpaceMouseFlyAxes. Hyperscope preserves
// its established physical gesture mapping here; the camera integration then
// applies all three rotations in the current local frame.
export const SPACEMOUSE_FLY_RIGHT = 0;
export const SPACEMOUSE_FLY_UP = 1;
export const SPACEMOUSE_FLY_FORWARD = 2;
export const SPACEMOUSE_FLY_PITCH = 3;
export const SPACEMOUSE_FLY_YAW = 4;
export const SPACEMOUSE_FLY_ROLL = 5;

export function createSpaceMouseState() {
  return {
    axes: new Float32Array(6),
    buttons: 0,
  };
}

function normalizedAxis(data, byteOffset) {
  const raw = data.getInt16(byteOffset, true);
  return Math.max(-1, Math.min(1, raw / SPACEMOUSE_AXIS_MAX));
}

/**
 * Decode one WebHID input report into retained state.
 * Returns true when the report belongs to a supported SpaceMouse layout.
 */
export function decodeSpaceMouseReport(reportId, data, state) {
  if (!(data instanceof DataView) || !state?.axes || state.axes.length < 6) {
    return false;
  }

  if (reportId === 1 && data.byteLength >= 12) {
    for (let axis = 0; axis < 6; axis++) {
      state.axes[axis] = normalizedAxis(data, axis * 2);
    }
    return true;
  }
  if (reportId === 1 && data.byteLength >= 6) {
    for (let axis = 0; axis < 3; axis++) {
      state.axes[axis] = normalizedAxis(data, axis * 2);
    }
    return true;
  }
  if (reportId === 2 && data.byteLength >= 6) {
    for (let axis = 0; axis < 3; axis++) {
      state.axes[axis + 3] = normalizedAxis(data, axis * 2);
    }
    return true;
  }
  if (reportId === 3 && data.byteLength >= 1) {
    state.buttons = data.byteLength >= 2
      ? data.getUint16(0, true)
      : data.getUint8(0);
    return true;
  }
  return false;
}

/** Deadzone followed by a signed quadratic response curve. */
export function shapeSpaceMouseAxis(value, deadzone = 0.08) {
  const magnitude = Math.abs(value);
  if (!Number.isFinite(magnitude) || magnitude <= deadzone) return 0;
  const normalized = Math.min(1, (magnitude - deadzone) / (1 - deadzone));
  return Math.sign(value) * normalized * normalized;
}

/**
 * Convert raw HID X/Y/Z/Rx/Ry/Rz into a right/up/forward +
 * pitch/yaw/roll camera-local velocity vector.
 *
 * This intentionally preserves Hyperscope's original gestures and signs:
 * X/Y/Z drive right/up/forward, while Rx/Ry/Rz drive pitch/yaw/roll. The
 * forward component is translation, never orbit-radius zoom.
 */
export function mapSpaceMouseFlyAxes(axes, out = new Float32Array(6)) {
  out[SPACEMOUSE_FLY_RIGHT] = axes[0];
  out[SPACEMOUSE_FLY_UP] = axes[1];
  out[SPACEMOUSE_FLY_FORWARD] = axes[2];
  out[SPACEMOUSE_FLY_PITCH] = -axes[3];
  out[SPACEMOUSE_FLY_YAW] = -axes[4];
  out[SPACEMOUSE_FLY_ROLL] = axes[5];
  return out;
}

/** Select the interaction layer encoded by the two primary puck buttons. */
export function spaceMouseModifierMode(buttons) {
  const left = (buttons & SPACEMOUSE_LEFT_BUTTON) !== 0;
  const right = (buttons & SPACEMOUSE_RIGHT_BUTTON) !== 0;
  // Right wins when both are held: changing camera/transform state and DoF in
  // the same frame is surprising, while releasing it naturally returns to the
  // still-held inversion layer.
  if (right) return 'depth-of-field';
  if (left) return 'inversion';
  return 'camera';
}

export class SpaceMouseController {
  constructor({
    hid = typeof navigator !== 'undefined' ? navigator.hid : null,
    onStatus = () => {},
    staleAfterMs = 180,
    responseHz = 18,
  } = {}) {
    this.hid = hid;
    this.onStatus = onStatus;
    this.staleAfterMs = staleAfterMs;
    this.responseHz = responseHz;
    this.device = null;
    this.state = createSpaceMouseState();
    this.filteredAxes = new Float32Array(6);
    this.lastAxisReportAt = -Infinity;
    this.handleInputReport = this.handleInputReport.bind(this);
    this.handleDisconnect = this.handleDisconnect.bind(this);
    if (this.hid) this.hid.addEventListener('disconnect', this.handleDisconnect);
  }

  get available() {
    return !!this.hid;
  }

  get connected() {
    return !!this.device?.opened;
  }

  get buttons() {
    return this.state.buttons;
  }

  describeDevice(device = this.device) {
    if (!device) return 'SpaceMouse';
    const product = device.productName || 'SpaceMouse';
    const ids = `${device.vendorId.toString(16).padStart(4, '0')}:` +
      device.productId.toString(16).padStart(4, '0');
    return `${product} (${ids})`;
  }

  reportStatus(state, message, error = null) {
    this.onStatus({ state, message, error, device: this.device });
  }

  async restoreGrantedDevice() {
    if (!this.hid) {
      this.reportStatus('unavailable', 'WebHID is unavailable in this browser');
      return false;
    }
    const devices = await this.hid.getDevices();
    const device = devices.find(candidate => candidate.vendorId === SPACEMOUSE_VENDOR_ID);
    if (!device) {
      this.reportStatus('idle', 'Not connected');
      return false;
    }
    return this.open(device);
  }

  async requestDevice() {
    if (!this.hid) {
      this.reportStatus('unavailable', 'WebHID is unavailable in this browser');
      return false;
    }
    // requestDevice must remain directly downstream of the button click so the
    // browser sees the required transient user activation.
    let devices;
    try {
      devices = await this.hid.requestDevice({
        filters: [{ vendorId: SPACEMOUSE_VENDOR_ID }],
      });
    } catch (error) {
      if (error?.name === 'NotFoundError') {
        this.reportStatus('idle', 'No SpaceMouse selected');
      } else {
        this.reportStatus('error', `SpaceMouse chooser failed: ${error?.message || error}`, error);
      }
      return false;
    }
    if (!devices.length) {
      this.reportStatus('idle', 'No SpaceMouse selected');
      return false;
    }
    return this.open(devices[0]);
  }

  async open(device) {
    this.detach();
    this.device = device;
    try {
      if (!device.opened) await device.open();
      device.addEventListener('inputreport', this.handleInputReport);
      this.reportStatus('connected', this.describeDevice(device));
      return true;
    } catch (error) {
      const description = this.describeDevice(device);
      this.device = null;
      this.reportStatus(
        'error',
        `Could not open ${description}; check operating-system HID permissions`,
        error,
      );
      return false;
    }
  }

  detach() {
    if (this.device) {
      this.device.removeEventListener('inputreport', this.handleInputReport);
    }
    this.device = null;
    this.state.axes.fill(0);
    this.state.buttons = 0;
    this.filteredAxes.fill(0);
    this.lastAxisReportAt = -Infinity;
  }

  resetSmoothing() {
    this.filteredAxes.fill(0);
  }

  handleDisconnect(event) {
    if (event.device !== this.device) return;
    const description = this.describeDevice(event.device);
    this.detach();
    this.reportStatus('idle', `${description} disconnected`);
  }

  handleInputReport(event) {
    if (!decodeSpaceMouseReport(event.reportId, event.data, this.state)) return;
    if (event.reportId === 1 || event.reportId === 2) {
      this.lastAxisReportAt = performance.now();
    }
  }

  /**
   * Return one retained six-axis sample for this animation frame.
   * Input reports only update targets; smoothing and stale-input decay happen
   * here, so HID report rate never controls camera integration rate.
   */
  sample(nowMs, deltaSeconds) {
    const stale = nowMs - this.lastAxisReportAt > this.staleAfterMs;
    const alpha = 1 - Math.exp(-Math.max(0, deltaSeconds) * this.responseHz);
    for (let axis = 0; axis < 6; axis++) {
      const target = stale ? 0 : shapeSpaceMouseAxis(this.state.axes[axis]);
      this.filteredAxes[axis] += (target - this.filteredAxes[axis]) * alpha;
      if (Math.abs(this.filteredAxes[axis]) < 1e-5) this.filteredAxes[axis] = 0;
    }
    return this.filteredAxes;
  }
}
