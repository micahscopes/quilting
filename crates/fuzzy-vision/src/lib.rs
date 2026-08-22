//! fuzzy-vision: JFA-based variable per-pixel blur for WebGL2/OpenGL.
//!
//! A standalone post-process blur library built on the Jump Flooding Algorithm.
//! Given any per-pixel weight texture (0=sharp, 1=max blur), fuzzy-vision
//! propagates that weight outward via JFA distance fields, then applies a
//! weighted separable Gaussian blur where each pixel has its own radius.
//!
//! The library is effect-agnostic — callers produce the weight texture,
//! fuzzy-vision handles the blur. Multiple weight sources can be composited
//! into a single weight texture for efficient multi-effect rendering.
//!
//! ## Weight sources (caller-provided)
//! - **Depth of field**: weight = f(depth, focal_distance)
//! - **Conformal blur**: weight = |Jacobian| of Möbius transform
//! - **Transmission roughness**: weight = material roughness per-pixel
//! - **Bloom**: weight = max(0, luminance - threshold)
//! - **Selective focus**: weight = distance from focal point
//!
//! ## Multi-effect efficiency
//! Composite multiple weights into RGBA channels of a single texture,
//! or `max()` them into one channel — one JFA pass serves all effects.
//!
//! ## Lifecycle
//! [`JfaPipeline::new`] compiles every program up front and allocates its render
//! targets at 1x1; call [`JfaPipeline::resize`] before the first
//! [`JfaPipeline::run`] to size them to the viewport. `glow::Context` is not
//! reachable from `Drop`, so GL objects are *not* released automatically —
//! call [`JfaPipeline::destroy`] explicitly when tearing the pipeline down.

use glow::HasContext;

/// Precision mode for intermediate textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// RGBA32F — best quality, requires EXT_color_buffer_float
    Float32,
    /// RGBA16F — good quality, widely supported in WebGL2
    Float16,
}

/// Configuration for the JFA blur pipeline.
#[derive(Debug, Clone)]
pub struct JfaConfig {
    /// Maximum blur radius in pixels at weight=1.0
    pub max_distance: f32,
    /// Blur strength multiplier for the final Gaussian pass
    pub blur_strength: f32,
    /// Downsample factor for JFA computation (e.g. 2 or 4)
    pub downsample: u32,
    /// Number of blur passes (1=standard, 2-3=smoother, higher=silkier)
    pub blur_passes: u32,
    /// Blur mode: 0=DoF(depth), 1=Conformal(stretch), 2=Hybrid(max), 3=Focus field
    pub blur_mode: u32,
    /// Use mip-chain blur instead of multi-pass Gaussian (smoother, no boundary artifacts)
    pub use_mip_blur: bool,
    /// Kawase weight smoothing: 0 = disabled, >0 = number of passes
    pub kawase_passes: u32,
    /// Kawase offset per pass
    pub kawase_offset: f32,
    /// Float precision for intermediate textures
    pub precision: Precision,
    /// Focus point in stretch space: 0.5 = neutral, 0 = max squash, 1 = max expand
    pub focus: f32,
    /// Bandwidth of the Gaussian focus band (smaller = tighter focus)
    pub bandwidth: f32,
    /// Normalize stretch to actual min/max each frame (true) or use absolute sigmoid values (false)
    pub normalize: bool,
    /// CPU-sampled stretch range (from worker). If set, used instead of GPU reduction.
    pub cpu_stretch_min: f32,
    pub cpu_stretch_max: f32,
}

impl Default for JfaConfig {
    fn default() -> Self {
        JfaConfig {
            max_distance: 64.0,
            blur_strength: 1.0,
            downsample: 2, // half-res JFA + bilinear upsample in firmness
            blur_mode: 1,
            blur_passes: 2,
            use_mip_blur: false, // Gaussian pyramid or mip blur (experimental)
            kawase_passes: 0,
            kawase_offset: 0.75,
            precision: Precision::Float16,
            focus: 0.5,
            bandwidth: 0.3,
            normalize: true,
            cpu_stretch_min: 0.5,
            cpu_stretch_max: 0.5,
        }
    }
}

/// Fullscreen vertex shader — generates a triangle covering the viewport from gl_VertexID.
const VS_FULLSCREEN: &str = r#"#version 300 es
out vec2 v_uv;
void main() {
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    v_uv = vec2(x, y) * 0.5 + 0.5;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"#;

/// Generate a radial weight: center = 0 (sharp), edges = 1 (blurred).
/// Good for vignette-style DoF or testing the pipeline.
const FS_WEIGHT_RADIAL: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;

void main() {
    vec2 center = vec2(0.5);
    float dist = distance(v_uv, center) * 2.0; // 0 at center, ~1.4 at corners
    float w = smoothstep(0.3, 1.0, dist);
    o_color = vec4(w, 0.0, 0.0, 1.0);
}
"#;

/// Min/max reduction: downsample 2x2 blocks, tracking min in R and max in G.
/// Chain multiple passes to reduce to 1x1.
const FS_REDUCE_MINMAX: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_source;
uniform vec2 u_source_dims;

void main() {
    // Sample 4 texels from the source (2x2 block)
    vec2 texel = 1.0 / u_source_dims;
    vec2 base = v_uv - texel * 0.5;
    vec4 a = texture(u_source, base);
    vec4 b = texture(u_source, base + vec2(texel.x, 0.0));
    vec4 c = texture(u_source, base + vec2(0.0, texel.y));
    vec4 d = texture(u_source, base + texel);
    // First pass: input has stretch in R only (G may be 0).
    // Subsequent passes: R=min, G=max from prior reduction.
    // Detect first pass: if all G values are 0, use R for both min and max.
    float use_g = step(0.001, a.g + b.g + c.g + d.g);
    float mn = min(min(a.r, b.r), min(c.r, d.r));
    float mx = max(max(mix(a.r, a.g, use_g), mix(b.r, b.g, use_g)),
                   max(mix(c.r, c.g, use_g), mix(d.r, d.g, use_g)));
    o_color = vec4(mn, mx, 0.0, 1.0);
}
"#;

/// Normalize stretch using min/max from reduction pass, pass through for JFA.
/// Focus applied post-JFA in firmness shader.
const FS_WEIGHT_CONFORMAL_NORM: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_stretch;
uniform sampler2D u_minmax;   // 1x1 texture: R=global min, G=global max

void main() {
    float raw = texture(u_stretch, v_uv).r;
    vec4 mm = texture(u_minmax, vec2(0.5));
    float mn = mm.r;
    float mx = mm.g;
    float range = max(mx - mn, 0.001);
    float normalized = (raw - mn) / range;
    // Pass normalized value through — focus applied post-JFA in firmness
    o_color = vec4(normalized, 0.0, 0.0, 1.0);
}
"#;

/// Conformal/DoF weight: select signal for JFA based on mode.
/// Mode 0 = DoF (depth), Mode 1 = Conformal (stretch), Mode 2 = Hybrid (max),
/// Mode 3 = source-space spherical focus field.
const FS_WEIGHT_CONFORMAL: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_stretch; // MRT: R=stretch, G=depth, B=focus field
uniform float u_mode; // 0=dof, 1=conformal, 2=hybrid, 3=focus field

void main() {
    vec4 data = texture(u_stretch, v_uv);
    float stretch = data.r;
    float depth = data.g;
    float w;
    if (u_mode < 0.5) {
        w = depth;           // DoF: use depth directly (dense, every pixel)
    } else if (u_mode < 1.5) {
        w = stretch;         // Conformal: use stretch (sparse, needs JFA)
    } else if (u_mode < 2.5) {
        w = max(stretch, depth); // Hybrid: both signals
    } else {
        w = data.b;         // Selection field: inside sharp, outside blurred
    }
    o_color = vec4(w, 0.0, 0.0, 1.0);
}
"#;

/// JFA initialization: convert a weight texture into seed data.
/// Input: weight texture (R=weight, 0=no blur, 1=max blur)
/// Output: RGBA = (uv.x, uv.y, weight, 0)
const FS_JFA_INIT: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_weight;

void main() {
    float w = texture(u_weight, v_uv).r;
    if (w > 0.001) {
        // Seed pixel: store own UV + weight
        o_color = vec4(v_uv, w, 0.0);
    } else {
        // Empty pixel: no seed yet
        o_color = vec4(v_uv, 0.0, 0.0);
    }
}
"#;

/// JFA propagation step: find nearest seed via jump flooding.
/// Reads from ping texture, writes to pong. Step size halves each pass.
const FS_JFA_STEP: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_source;
uniform int u_step;
uniform vec2 u_dims;

void main() {
    ivec2 dims = ivec2(u_dims);
    ivec2 coords = ivec2(v_uv * u_dims);
    vec2 pixel_uv = (vec2(coords) + 0.5) / u_dims;
    int step_size = max(1, u_step);

    vec4 source = texelFetch(u_source, coords, 0);

    // Best candidate: start with current pixel's seed
    vec4 best = source;
    float best_dist = (source.z > 0.0)
        ? distance(pixel_uv * u_dims, source.xy * u_dims)
        : 1e9;

    // Check 8 neighbors at step_size distance
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            ivec2 nc = coords + ivec2(dx * step_size, dy * step_size);
            if (nc.x < 0 || nc.x >= dims.x || nc.y < 0 || nc.y >= dims.y) continue;

            vec4 neighbor = texelFetch(u_source, nc, 0);
            if (neighbor.z <= 0.0) continue;

            float dist = distance(pixel_uv * u_dims, neighbor.xy * u_dims);

            // Pure Euclidean nearest-seed: no weight bonus distortion.
            // Weight is carried along but doesn't affect which seed "wins".
            // Higher weight only breaks ties at equal distance.
            if (dist < best_dist - 0.5 || (abs(dist - best_dist) <= 0.5 && neighbor.z > best.z)) {
                best = neighbor;
                best_dist = dist;
            }
        }
    }

    o_color = best;
}
"#;

/// Firmness blend: apply distance-based falloff AND focus/bandwidth post-JFA.
/// JFA propagates raw stretch values. Focus band is applied HERE, not before JFA.
/// This makes JFA stable regardless of focus settings.
const FS_JFA_FIRMNESS: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_jfa;
uniform vec2 u_dims;
uniform float u_max_distance;
uniform float u_focus;     // applied post-JFA
uniform float u_bandwidth; // applied post-JFA
uniform sampler2D u_weight;
uniform float u_mode;

void main() {
    // A focus field is already a dense normalized blur mask. Preserve it
    // directly so JFA cannot bleed outside blur back into the sharp sphere.
    if (u_mode > 2.5) {
        float field_weight = texture(u_weight, v_uv).r;
        o_color = vec4(v_uv, field_weight, 0.0);
        return;
    }
    // Bilinear interpolation from JFA
    vec2 texel = 1.0 / u_dims;
    vec2 sp = v_uv * u_dims - 0.5;
    vec2 base = floor(sp) * texel + texel * 0.5;
    vec2 f = fract(sp);
    vec4 j00 = texture(u_jfa, base);
    vec4 j10 = texture(u_jfa, base + vec2(texel.x, 0.0));
    vec4 j01 = texture(u_jfa, base + vec2(0.0, texel.y));
    vec4 j11 = texture(u_jfa, base + texel);
    vec4 jfa = mix(mix(j00, j10, f.x), mix(j01, j11, f.x), f.y);

    if (jfa.z <= 0.0) {
        o_color = vec4(0.0);
        return;
    }

    // jfa.z = raw stretch value (propagated by JFA, NOT focus-applied)
    // Apply focus/bandwidth HERE on the smooth propagated stretch
    float stretch = jfa.z;
    float diff = stretch - u_focus;
    float bw = max(u_bandwidth, 0.01);
    float sharpness = exp(-(diff * diff) / (2.0 * bw * bw));
    float focus_weight = 1.0 - sharpness; // inverted: focused band is sharp

    // Distance falloff from JFA seed
    float dist = distance(v_uv * u_dims, jfa.xy * u_dims);
    float effective_max = u_max_distance * focus_weight;
    float ratio = clamp(dist / max(effective_max, 1.0), 0.0, 1.0);
    float falloff = 1.0 - smoothstep(0.0, 1.0, ratio);
    float weight = focus_weight * falloff;

    o_color = vec4(jfa.xy, weight, ratio);
}
"#;

/// Simple passthrough — draws a texture to the screen (avoids blit issues with multisampled FB).
const FS_PASSTHROUGH: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_tex;
void main() { o_color = texture(u_tex, v_uv); }
"#;

/// Gaussian downsample: 9-tap separable Gaussian at half resolution per mip level.
/// This replaces generateMipmap (box filter) with proper Gaussian quality.
const FS_GAUSS_DOWN: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_tex;
uniform vec2 u_dir; // (1/w, 0) for H pass, (0, 1/h) for V pass

void main() {
    // 5-tap Gaussian: weights [0.06, 0.24, 0.40, 0.24, 0.06]
    vec4 c  = texture(u_tex, v_uv) * 0.40;
    vec4 c1 = texture(u_tex, v_uv - u_dir) * 0.24;
    vec4 c2 = texture(u_tex, v_uv + u_dir) * 0.24;
    vec4 c3 = texture(u_tex, v_uv - 2.0 * u_dir) * 0.06;
    vec4 c4 = texture(u_tex, v_uv + 2.0 * u_dir) * 0.06;
    o_color = c + c1 + c2 + c3 + c4;
}
"#;

/// Mip-chain blur composite: sample from scene mip level based on per-pixel weight.
/// Smooth variable blur with zero boundary artifacts — LINEAR_MIPMAP_LINEAR interpolates.
const FS_MIP_COMPOSITE: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_scene;    // scene with mip chain
uniform sampler2D u_weight;   // firmness mask
uniform float u_max_lod;      // max mip level available
uniform float u_blur_strength;

void main() {
    vec4 w = texture(u_weight, v_uv);
    float blur_weight = clamp(w.z * (1.0 - clamp(w.w, 0.0, 1.0)), 0.0, 1.0);

    // Map weight to mip LOD: weight=0 → LOD 0 (sharp), weight=1 → max LOD (blurry)
    float lod = blur_weight * u_blur_strength * u_max_lod;
    o_color = textureLod(u_scene, v_uv, lod);
}
"#;

/// Kawase blur on the weight mask — smooths JFA boundaries cheaply (5 taps).
const FS_WEIGHT_KAWASE: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_weight;
uniform float u_offset;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_weight, 0));
    float off = u_offset;
    vec4 c  = texture(u_weight, v_uv);
    vec4 tl = texture(u_weight, v_uv + vec2(-off, -off) * texel);
    vec4 tr = texture(u_weight, v_uv + vec2( off, -off) * texel);
    vec4 bl = texture(u_weight, v_uv + vec2(-off,  off) * texel);
    vec4 br = texture(u_weight, v_uv + vec2( off,  off) * texel);
    // Blend weight (z) and distance (w), keep UV (xy) from center
    float avg_w = (c.z + tl.z + tr.z + bl.z + br.z) / 5.0;
    float avg_d = (c.w + tl.w + tr.w + bl.w + br.w) / 5.0;
    o_color = vec4(c.xy, avg_w, avg_d);
}
"#;

/// Directional Gaussian blur — arbitrary angle via u_dir uniform.
/// Used for hex blur: 3 passes at 0°, 60°, 120° for near-circular blur.
const FS_BLUR_DIR: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_scene;
uniform sampler2D u_weight;
uniform float u_blur_radius;
uniform float u_blur_strength;
uniform vec2 u_dir; // normalized direction * texel_size
uniform float u_is_final; // >0.5 = apply mix() crossfade, else full blur

const int MAX_RADIUS = 48;

void main() {
    vec4 w = texture(u_weight, v_uv);
    float blur_weight = clamp(w.z * (1.0 - clamp(w.w, 0.0, 1.0)), 0.0, 1.0);
    vec4 original = texture(u_scene, v_uv);

    if (u_blur_strength <= 0.001) { o_color = original; return; }

    float effective_radius = u_blur_radius * blur_weight * u_blur_strength;
    float sigma = max(effective_radius / 2.0, 0.001);
    int radius = min(max(int(ceil(effective_radius)), 1), MAX_RADIUS);

    vec4 color_sum = vec4(0.0);
    float weight_sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        float d = float(abs(i));
        float gw = exp(-(d * d) / (2.0 * sigma * sigma));
        vec2 uv = clamp(v_uv + float(i) * u_dir, vec2(0.0), vec2(1.0));
        color_sum += texture(u_scene, uv) * gw;
        weight_sum += gw;
    }

    vec4 blurred = color_sum / max(weight_sum, 0.001);
    // Only final pass applies the sharp/blur crossfade.
    // Intermediate passes do full blur — prevents compound halo at boundaries.
    if (u_is_final > 0.5) {
        o_color = mix(original, blurred, blur_weight);
    } else {
        o_color = blurred;
    }
}
"#;

/// Compiled JFA programs and GPU resources.
pub struct JfaPipeline {
    prog_init: glow::Program,
    prog_step: glow::Program,
    prog_firmness: glow::Program,
    prog_blur_dir: glow::Program,
    prog_weight_radial: glow::Program,
    prog_weight_conformal: glow::Program,
    prog_reduce_minmax: glow::Program,
    prog_weight_conformal_norm: glow::Program,
    prog_weight_kawase: glow::Program,
    prog_passthrough: glow::Program,
    prog_mip_composite: glow::Program,
    prog_gauss_down: glow::Program,
    vao: glow::VertexArray,
    // Reduction chain for min/max (ping-pong, shrinks to 1x1)
    reduce_fbo_a: glow::Framebuffer,
    reduce_tex_a: glow::Texture,
    reduce_fbo_b: glow::Framebuffer,
    reduce_tex_b: glow::Texture,
    // Ping-pong FBOs + textures for JFA steps
    ping_fbo: glow::Framebuffer,
    ping_tex: glow::Texture,
    pong_fbo: glow::Framebuffer,
    pong_tex: glow::Texture,
    // Firmness output
    firmness_fbo: glow::Framebuffer,
    firmness_tex: glow::Texture,
    // Blur ping-pong
    blur_fbo: glow::Framebuffer,
    blur_tex: glow::Texture,
    blur_fbo_b: glow::Framebuffer,
    blur_tex_b: glow::Texture,
    // Current allocated size (downsampled for JFA, full for blur)
    jfa_size: (i32, i32),
    full_size: (i32, i32),
    config: JfaConfig,
    internal_format: u32,
    // --- Debug bookkeeping, recorded by `run()` so `debug_blit` shows real data ---
    /// Weight texture handed to the most recent `run()`. `None` before the first run.
    last_weight_tex: Option<glow::Texture>,
    /// Which ping-pong buffer holds the JFA result after the most recent `run()`.
    /// Parity depends on the JFA step count, so it has to be recorded, not guessed.
    jfa_result_in_ping: bool,
}

fn compile_shader(gl: &glow::Context, shader_type: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type)?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(format!("shader compile:\n{log}\n\nSource:\n{source}"));
        }
        Ok(shader)
    }
}

fn link_program(gl: &glow::Context, vs_src: &str, fs_src: &str) -> Result<glow::Program, String> {
    unsafe {
        let vs = compile_shader(gl, glow::VERTEX_SHADER, vs_src)?;
        let fs = compile_shader(gl, glow::FRAGMENT_SHADER, fs_src)?;
        let prog = gl.create_program()?;
        gl.attach_shader(prog, vs);
        gl.attach_shader(prog, fs);
        gl.link_program(prog);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            gl.delete_program(prog);
            return Err(format!("program link: {log}"));
        }
        Ok(prog)
    }
}

fn create_fbo_tex(gl: &glow::Context, w: i32, h: i32, format: u32) -> Result<(glow::Framebuffer, glow::Texture), String> {
    unsafe {
        let tex = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        // RGBA16F needs HALF_FLOAT type, RGBA32F needs FLOAT, RGBA8 needs UNSIGNED_BYTE
        let (ext_format, ext_type) = match format {
            glow::RGBA32F => (glow::RGBA, glow::FLOAT),
            glow::RGBA16F => (glow::RGBA, glow::HALF_FLOAT),
            _ => (glow::RGBA, glow::UNSIGNED_BYTE),
        };
        gl.tex_image_2d(glow::TEXTURE_2D, 0, format as i32, w, h, 0,
            ext_format, ext_type, glow::PixelUnpackData::Slice(None));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

        let fbo = gl.create_framebuffer()?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        Ok((fbo, tex))
    }
}

/// Resolution of the JFA ping-pong buffers for a given viewport.
///
/// JFA runs at `1/downsample` resolution and is bilinearly upsampled in the
/// firmness pass; `downsample` is clamped to at least 1 and the result never
/// degenerates to a zero-sized texture.
fn downsampled_size(width: i32, height: i32, downsample: u32) -> (i32, i32) {
    let ds = downsample.max(1) as i32;
    ((width / ds).max(1), (height / ds).max(1))
}

/// Number of jump-flooding passes needed to cover a `w` x `h` buffer.
///
/// Standard JFA starts at `2^(n-1)` and halves to 1, so `n = ceil(log2(max_dim))`.
/// This is the "O(log n) regardless of blur radius" property the library sells.
fn jfa_step_count(width: i32, height: i32) -> u32 {
    let max_dim = width.max(height).max(1) as f32;
    max_dim.log2().ceil().max(0.0) as u32
}

/// Number of mip levels (including level 0) used by the mip-blur path.
///
/// Capped at 8 — beyond that the levels are a few pixels wide and contribute
/// nothing but bandwidth.
fn mip_level_count(width: i32, height: i32) -> i32 {
    let max_dim = width.max(height).max(1) as f32;
    (max_dim.log2().floor() as i32 + 1).clamp(1, 8)
}

/// Per-sub-pass blur strength for the hex blur.
///
/// The hex blur fires `blur_passes * 3` directional Gaussians; dividing by
/// `sqrt(passes * 3)` keeps total blur roughly constant as `blur_passes` varies,
/// because independent Gaussian variances add.
fn per_pass_blur_strength(blur_strength: f32, blur_passes: u32) -> f32 {
    let n = blur_passes.max(1);
    blur_strength / (n as f32 * 3.0).sqrt()
}

impl JfaPipeline {
    /// Create the JFA pipeline. Call once at init time.
    pub fn new(gl: &glow::Context, config: JfaConfig) -> Result<Self, String> {
        let prog_init = link_program(gl, VS_FULLSCREEN, FS_JFA_INIT)?;
        let prog_step = link_program(gl, VS_FULLSCREEN, FS_JFA_STEP)?;
        let prog_firmness = link_program(gl, VS_FULLSCREEN, FS_JFA_FIRMNESS)?;
        let prog_blur_dir = link_program(gl, VS_FULLSCREEN, FS_BLUR_DIR)?;
        let prog_weight_radial = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_RADIAL)?;
        let prog_weight_conformal = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_CONFORMAL)?;
        let prog_reduce_minmax = link_program(gl, VS_FULLSCREEN, FS_REDUCE_MINMAX)?;
        let prog_weight_conformal_norm = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_CONFORMAL_NORM)?;
        let prog_weight_kawase = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_KAWASE)?;
        let prog_passthrough = link_program(gl, VS_FULLSCREEN, FS_PASSTHROUGH)?;
        let prog_mip_composite = link_program(gl, VS_FULLSCREEN, FS_MIP_COMPOSITE)?;
        let prog_gauss_down = link_program(gl, VS_FULLSCREEN, FS_GAUSS_DOWN)?;
        let vao = unsafe { gl.create_vertex_array()? };
        let (reduce_fbo_a, reduce_tex_a) = create_fbo_tex(gl, 1, 1, glow::RGBA16F)?;
        let (reduce_fbo_b, reduce_tex_b) = create_fbo_tex(gl, 1, 1, glow::RGBA16F)?;

        let fmt = match config.precision {
            Precision::Float32 => glow::RGBA32F,
            Precision::Float16 => glow::RGBA16F,
        };

        // Allocate with placeholder 1x1 — resize() will set real size
        let (ping_fbo, ping_tex) = create_fbo_tex(gl, 1, 1, fmt)?;
        let (pong_fbo, pong_tex) = create_fbo_tex(gl, 1, 1, fmt)?;
        let (firmness_fbo, firmness_tex) = create_fbo_tex(gl, 1, 1, fmt)?;
        let (blur_fbo, blur_tex) = create_fbo_tex(gl, 1, 1, glow::RGBA8)?;
        let (blur_fbo_b, blur_tex_b) = create_fbo_tex(gl, 1, 1, glow::RGBA8)?;

        Ok(JfaPipeline {
            prog_init, prog_step, prog_firmness,
            prog_weight_radial, prog_weight_conformal,
            prog_reduce_minmax, prog_weight_conformal_norm, prog_weight_kawase,
            prog_blur_dir,
            prog_passthrough, prog_mip_composite, prog_gauss_down,
            reduce_fbo_a, reduce_tex_a, reduce_fbo_b, reduce_tex_b,
            vao, ping_fbo, ping_tex, pong_fbo, pong_tex,
            firmness_fbo, firmness_tex,
            blur_fbo, blur_tex, blur_fbo_b, blur_tex_b,
            jfa_size: (0, 0), full_size: (0, 0), config, internal_format: fmt,
            last_weight_tex: None, jfa_result_in_ping: false,
        })
    }

    /// Resize internal textures to match viewport. Call when viewport changes.
    pub fn resize(&mut self, gl: &glow::Context, width: i32, height: i32) {
        if self.full_size == (width, height) { return; }
        self.full_size = (width, height);

        let (jw, jh) = downsampled_size(width, height, self.config.downsample);
        self.jfa_size = (jw, jh);

        let fmt = self.internal_format;
        let (ext_format, ext_type) = match fmt {
            glow::RGBA32F => (glow::RGBA, glow::FLOAT),
            glow::RGBA16F => (glow::RGBA, glow::HALF_FLOAT),
            _ => (glow::RGBA, glow::UNSIGNED_BYTE),
        };
        unsafe {
            // Resize JFA ping/pong (downsampled)
            for tex in [self.ping_tex, self.pong_tex] {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(glow::TEXTURE_2D, 0, fmt as i32, jw, jh, 0,
                    ext_format, ext_type, glow::PixelUnpackData::Slice(None));
            }
            // Resize firmness (full res)
            gl.bind_texture(glow::TEXTURE_2D, Some(self.firmness_tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, fmt as i32, width, height, 0,
                ext_format, ext_type, glow::PixelUnpackData::Slice(None));
            // Resize blur ping-pong (full res, RGBA8)
            for tex in [self.blur_tex, self.blur_tex_b] {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, width, height, 0,
                    glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            }
        }
    }

    /// Run the full JFA blur pipeline.
    ///
    /// `weight_tex`: input texture where R channel = blur weight (0=sharp, 1=max blur)
    /// `scene_tex`: the scene color texture to blur
    /// `output_fbo`: framebuffer to write the blurred result to (None = default FB)
    pub fn run(
        &mut self,
        gl: &glow::Context,
        weight_tex: glow::Texture,
        scene_tex: glow::Texture,
        output_fbo: Option<glow::Framebuffer>,
    ) {
        let (fw, fh) = self.full_size;
        let (jw, jh) = self.jfa_size;
        if fw == 0 || fh == 0 { return; }

        // The weight texture is used as-is — the old pre-blur pass was removed because
        // it produced box halos at geometry edges. JFA+2 with pure Euclidean distance
        // handles boundaries cleanly on its own.
        self.last_weight_tex = Some(weight_tex);

        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(self.vao));

            // The selection field is already dense and normalized. It bypasses
            // seed propagation entirely; other modes retain the JFA path.
            let direct_focus_field = self.config.blur_mode == 3;
            let jfa_result_tex = if direct_focus_field {
                self.jfa_result_in_ping = true;
                self.ping_tex
            } else {
            // --- Stage 1: Init seeds from weight texture (into ping, at JFA resolution) ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.ping_fbo));
            gl.viewport(0, 0, jw, jh);
            gl.use_program(Some(self.prog_init));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(weight_tex));
            if let Some(loc) = gl.get_uniform_location(self.prog_init, "u_weight") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // --- Stage 2: JFA propagation (log2 passes, ping-pong) ---
            let num_steps = jfa_step_count(jw, jh);
            let mut read_from_ping = true;

            for i in 0..num_steps {
                let step = 1u32 << (num_steps - 1 - i);
                let (src_tex, dst_fbo) = if read_from_ping {
                    (self.ping_tex, self.pong_fbo)
                } else {
                    (self.pong_tex, self.ping_fbo)
                };

                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst_fbo));
                gl.use_program(Some(self.prog_step));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_step, "u_source") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_step, "u_step") {
                    gl.uniform_1_i32(Some(&loc), step as i32);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_step, "u_dims") {
                    gl.uniform_2_f32(Some(&loc), jw as f32, jh as f32);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                read_from_ping = !read_from_ping;
            }

            // JFA+2: extra step=1 cleanup passes to fix boundary artifacts.
            // These catch seed assignment errors that the main passes miss.
            for _ in 0..2 {
                let (src_tex, dst_fbo) = if read_from_ping {
                    (self.ping_tex, self.pong_fbo)
                } else {
                    (self.pong_tex, self.ping_fbo)
                };
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst_fbo));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_step, "u_step") {
                    gl.uniform_1_i32(Some(&loc), 1);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                read_from_ping = !read_from_ping;
            }

            // JFA result is in whichever buffer was last written to. Which one that is
            // depends on the step-count parity, so record it for debug_blit().
            self.jfa_result_in_ping = read_from_ping;
            if read_from_ping { self.ping_tex } else { self.pong_tex }
            };

            // --- Stage 3: Firmness blend (full resolution) ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.firmness_fbo));
            gl.viewport(0, 0, fw, fh);
            gl.use_program(Some(self.prog_firmness));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(jfa_result_tex));
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_jfa") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_dims") {
                gl.uniform_2_f32(Some(&loc), jw as f32, jh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_max_distance") {
                gl.uniform_1_f32(Some(&loc), self.config.max_distance / self.config.downsample as f32);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_focus") {
                gl.uniform_1_f32(Some(&loc), self.config.focus);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_bandwidth") {
                gl.uniform_1_f32(Some(&loc), self.config.bandwidth);
            }
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(weight_tex));
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_weight") {
                gl.uniform_1_i32(Some(&loc), 1);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_mode") {
                gl.uniform_1_f32(Some(&loc), self.config.blur_mode as f32);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // --- Stage 3.5: Optional Kawase weight smoothing ---
            if self.config.kawase_passes > 0 {
                for pass in 0..self.config.kawase_passes {
                    let offset = self.config.kawase_offset * (pass as f32 + 1.0);
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blur_fbo_b));
                    gl.viewport(0, 0, fw, fh);
                    gl.use_program(Some(self.prog_weight_kawase));
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(self.firmness_tex));
                    if let Some(loc) = gl.get_uniform_location(self.prog_weight_kawase, "u_weight") {
                        gl.uniform_1_i32(Some(&loc), 0);
                    }
                    if let Some(loc) = gl.get_uniform_location(self.prog_weight_kawase, "u_offset") {
                        gl.uniform_1_f32(Some(&loc), offset);
                    }
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                    // Copy back to firmness
                    gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.blur_fbo_b));
                    gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.firmness_fbo));
                    gl.blit_framebuffer(0, 0, fw, fh, 0, 0, fw, fh, glow::COLOR_BUFFER_BIT, glow::NEAREST);
                }
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }

            // --- Stage 4: Blur application ---
            if self.config.use_mip_blur {
                // Gaussian blur pyramid: build mip chain with proper Gaussian downsample,
                // then composite with textureLod for smooth per-pixel variable blur.

                // Allocate mip levels on scene texture
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(scene_tex));
                let num_mips = mip_level_count(fw, fh);
                // Pre-allocate all mip levels
                for mip in 1..num_mips {
                    let mw = (fw >> mip).max(1);
                    let mh = (fh >> mip).max(1);
                    gl.tex_image_2d(glow::TEXTURE_2D, mip, glow::RGBA8 as i32, mw, mh, 0,
                        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                }
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, num_mips - 1);

                // Build each mip: separable Gaussian from mip (level-1) → blur_tex_b (H) → mip (level) (V)
                let sc_fbo = self.blur_fbo; // reuse as temp FBO for mip writes
                gl.use_program(Some(self.prog_gauss_down));

                for mip in 1..num_mips {
                    let mw = (fw >> mip).max(1);
                    let mh = (fh >> mip).max(1);
                    let pw = (fw >> (mip - 1)).max(1);
                    let ph = (fh >> (mip - 1)).max(1);

                    // H pass: read mip (level-1) from scene → write to blur_tex_b
                    gl.bind_texture(glow::TEXTURE_2D, Some(self.blur_tex_b));
                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, mw, mh, 0,
                        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blur_fbo_b));
                    gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                        glow::TEXTURE_2D, Some(self.blur_tex_b), 0);
                    gl.viewport(0, 0, mw, mh);
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(scene_tex));
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, mip - 1);
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, mip - 1);
                    if let Some(loc) = gl.get_uniform_location(self.prog_gauss_down, "u_tex") {
                        gl.uniform_1_i32(Some(&loc), 0);
                    }
                    if let Some(loc) = gl.get_uniform_location(self.prog_gauss_down, "u_dir") {
                        gl.uniform_2_f32(Some(&loc), 1.0 / pw as f32, 0.0);
                    }
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);

                    // V pass: blur_tex_b → scene_tex mip (level)
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(sc_fbo));
                    gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                        glow::TEXTURE_2D, Some(scene_tex), mip);
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(self.blur_tex_b));
                    if let Some(loc) = gl.get_uniform_location(self.prog_gauss_down, "u_dir") {
                        gl.uniform_2_f32(Some(&loc), 0.0, 1.0 / ph as f32);
                    }
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                }

                // Restore base/max level for textureLod
                gl.bind_texture(glow::TEXTURE_2D, Some(scene_tex));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, 0);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, num_mips - 1);

                // Restore blur_fbo attachment to blur_tex
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(sc_fbo));
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D, Some(self.blur_tex), 0);

                // Final composite: textureLod per pixel weight
                gl.bind_framebuffer(glow::FRAMEBUFFER, output_fbo);
                gl.viewport(0, 0, fw, fh);
                gl.use_program(Some(self.prog_mip_composite));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(scene_tex));
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.firmness_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_mip_composite, "u_scene") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_mip_composite, "u_weight") {
                    gl.uniform_1_i32(Some(&loc), 1);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_mip_composite, "u_max_lod") {
                    gl.uniform_1_f32(Some(&loc), (num_mips - 1) as f32);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_mip_composite, "u_blur_strength") {
                    gl.uniform_1_f32(Some(&loc), self.config.blur_strength);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);

                // Restore scene_tex to non-mipmap mode
                gl.bind_texture(glow::TEXTURE_2D, Some(scene_tex));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            } else {
                // --- Hex blur: 3 directional Gaussian passes at 0°, 60°, 120° ---
                let hex_dirs: [(f32, f32); 3] = [
                    (1.0, 0.0),      // 0°
                    (0.5, 0.866),    // 60°
                    (-0.5, 0.866),   // 120°
                ];
                let num_passes = self.config.blur_passes.max(1);
                let per_pass_strength = per_pass_blur_strength(self.config.blur_strength, num_passes);
                let tx = 1.0 / fw as f32;
                let ty = 1.0 / fh as f32;

                gl.use_program(Some(self.prog_blur_dir));
                // Ping-pong: read from A, write to B, swap
                let mut read_tex = scene_tex;
                let mut write_to_a = true; // true = write to blur_fbo, false = write to blur_fbo_b
                let total_sub = num_passes * 3;
                let mut sub_idx = 0u32;

                for _ in 0..num_passes {
                    for &(dx, dy) in &hex_dirs {
                        sub_idx += 1;
                        let is_last = sub_idx == total_sub;
                        let dst_fbo = if is_last { output_fbo }
                            else if write_to_a { Some(self.blur_fbo) }
                            else { Some(self.blur_fbo_b) };

                        gl.bind_framebuffer(glow::FRAMEBUFFER, dst_fbo);
                        gl.viewport(0, 0, fw, fh);
                        gl.active_texture(glow::TEXTURE0);
                        gl.bind_texture(glow::TEXTURE_2D, Some(read_tex));
                        gl.active_texture(glow::TEXTURE1);
                        gl.bind_texture(glow::TEXTURE_2D, Some(self.firmness_tex));
                        if let Some(loc) = gl.get_uniform_location(self.prog_blur_dir, "u_scene") {
                            gl.uniform_1_i32(Some(&loc), 0);
                        }
                        if let Some(loc) = gl.get_uniform_location(self.prog_blur_dir, "u_weight") {
                            gl.uniform_1_i32(Some(&loc), 1);
                        }
                        if let Some(loc) = gl.get_uniform_location(self.prog_blur_dir, "u_blur_radius") {
                            gl.uniform_1_f32(Some(&loc), self.config.max_distance);
                        }
                        if let Some(loc) = gl.get_uniform_location(self.prog_blur_dir, "u_blur_strength") {
                            gl.uniform_1_f32(Some(&loc), per_pass_strength);
                        }
                        if let Some(loc) = gl.get_uniform_location(self.prog_blur_dir, "u_dir") {
                            gl.uniform_2_f32(Some(&loc), dx * tx, dy * ty);
                        }
                        if let Some(loc) = gl.get_uniform_location(self.prog_blur_dir, "u_is_final") {
                            gl.uniform_1_f32(Some(&loc), if is_last { 1.0 } else { 0.0 });
                        }
                        gl.draw_arrays(glow::TRIANGLES, 0, 3);

                        if !is_last {
                            read_tex = if write_to_a { self.blur_tex } else { self.blur_tex_b };
                            write_to_a = !write_to_a;
                        }
                    }
                }
            } // end if !use_mip_blur

            // Restore state
            gl.enable(glow::DEPTH_TEST);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Generate a weight texture for JFA from the raw MRT data.
    /// First selects the appropriate channel (depth/stretch/hybrid) based on blur_mode,
    /// then optionally normalizes using min/max reduction.
    pub fn generate_conformal_weight(
        &self,
        gl: &glow::Context,
        stretch_tex: glow::Texture,
        output_fbo: glow::Framebuffer,
        width: i32,
        height: i32,
    ) {
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(self.vao));

            if self.config.normalize && self.config.blur_mode != 3 {
                // --- Step 1: Mode selection → firmness_fbo (temporary, overwritten by JFA later) ---
                // This extracts the correct channel (depth/stretch/hybrid) into a single R value.
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.firmness_fbo));
                gl.viewport(0, 0, width, height);
                gl.use_program(Some(self.prog_weight_conformal));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(stretch_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_stretch") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_mode") {
                    gl.uniform_1_f32(Some(&loc), self.config.blur_mode as f32);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);

                // selected_tex now has the mode-selected value in R, G=0
                let selected_tex = self.firmness_tex;

                // CPU range only meaningful for conformal stretch (mode 1)
                let has_cpu_range = self.config.blur_mode == 1
                    && self.config.cpu_stretch_min < self.config.cpu_stretch_max - 0.001;

                if has_cpu_range {
                    // --- CPU-sampled range normalization ---
                    let minmax_tex = self.reduce_tex_a;
                    gl.bind_texture(glow::TEXTURE_2D, Some(minmax_tex));
                    let minmax_data = [self.config.cpu_stretch_min, self.config.cpu_stretch_max, 0.0_f32, 0.0];
                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA16F as i32, 1, 1, 0,
                        glow::RGBA, glow::FLOAT, glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&minmax_data))));

                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_fbo));
                    gl.viewport(0, 0, width, height);
                    gl.use_program(Some(self.prog_weight_conformal_norm));
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(selected_tex));
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(minmax_tex));
                    if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_stretch") {
                        gl.uniform_1_i32(Some(&loc), 0);
                    }
                    if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_minmax") {
                        gl.uniform_1_i32(Some(&loc), 1);
                    }
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                } else {
                    // --- GPU reduce to find min/max of the mode-selected signal ---
                    let mut src_tex = selected_tex;
                    let mut src_w = width;
                    let mut src_h = height;
                    let mut write_to_a = true;

                    gl.use_program(Some(self.prog_reduce_minmax));

                    loop {
                        let dst_w = (src_w / 2).max(1);
                        let dst_h = (src_h / 2).max(1);
                        let (dst_fbo, dst_tex) = if write_to_a {
                            (self.reduce_fbo_a, self.reduce_tex_a)
                        } else {
                            (self.reduce_fbo_b, self.reduce_tex_b)
                        };
                        gl.bind_texture(glow::TEXTURE_2D, Some(dst_tex));
                        gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA16F as i32, dst_w, dst_h, 0,
                            glow::RGBA, glow::HALF_FLOAT, glow::PixelUnpackData::Slice(None));
                        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst_fbo));
                        gl.viewport(0, 0, dst_w, dst_h);
                        gl.active_texture(glow::TEXTURE0);
                        gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
                        if let Some(loc) = gl.get_uniform_location(self.prog_reduce_minmax, "u_source") {
                            gl.uniform_1_i32(Some(&loc), 0);
                        }
                        if let Some(loc) = gl.get_uniform_location(self.prog_reduce_minmax, "u_source_dims") {
                            gl.uniform_2_f32(Some(&loc), src_w as f32, src_h as f32);
                        }
                        gl.draw_arrays(glow::TRIANGLES, 0, 3);
                        src_tex = dst_tex;
                        src_w = dst_w;
                        src_h = dst_h;
                        write_to_a = !write_to_a;
                        if src_w <= 1 && src_h <= 1 { break; }
                    }

                    // Normalize using the reduced min/max
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_fbo));
                    gl.viewport(0, 0, width, height);
                    gl.use_program(Some(self.prog_weight_conformal_norm));
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(selected_tex));
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
                    if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_stretch") {
                        gl.uniform_1_i32(Some(&loc), 0);
                    }
                    if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_minmax") {
                        gl.uniform_1_i32(Some(&loc), 1);
                    }
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                }
            } else {
                // --- Absolute mode: mode selection directly to output ---
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_fbo));
                gl.viewport(0, 0, width, height);
                gl.use_program(Some(self.prog_weight_conformal));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(stretch_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_stretch") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_mode") {
                    gl.uniform_1_f32(Some(&loc), self.config.blur_mode as f32);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }

            gl.bind_vertex_array(None);
        }
    }

    /// Generate a radial weight texture (center sharp, edges blurred).
    /// Writes to the provided FBO+texture at the given resolution.
    pub fn generate_radial_weight(
        &self,
        gl: &glow::Context,
        fbo: glow::Framebuffer,
        width: i32,
        height: i32,
    ) {
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.viewport(0, 0, width, height);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(self.vao));
            gl.use_program(Some(self.prog_weight_radial));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
        }
    }

    /// Convenience: run the full pipeline with a built-in radial weight (vignette DoF).
    /// Requires a weight FBO+texture to be provided (caller manages these).
    pub fn run_radial(
        &mut self,
        gl: &glow::Context,
        weight_fbo: glow::Framebuffer,
        weight_tex: glow::Texture,
        scene_tex: glow::Texture,
        output_fbo: Option<glow::Framebuffer>,
    ) {
        let (fw, fh) = self.full_size;
        if fw == 0 || fh == 0 { return; }
        self.generate_radial_weight(gl, weight_fbo, fw, fh);
        self.run(gl, weight_tex, scene_tex, output_fbo);
    }

    /// Debug: draw an internal pipeline texture to the output framebuffer.
    ///
    /// All stages reflect the most recent [`run`](Self::run); calling this before the
    /// first `run` draws nothing. Stages:
    ///
    /// - `1` — the weight texture that was passed into `run`, exactly as the caller
    ///   supplied it. The library does not pre-smooth it, so this is the raw JFA input.
    /// - `2` — the raw JFA output at JFA resolution, upscaled to fill the viewport.
    ///   RG = the nearest seed's UV, B = that seed's weight.
    /// - `3` — the firmness mask (post-JFA falloff + focus band), the texture the blur
    ///   passes actually read. Same content as [`weight_mask_tex`](Self::weight_mask_tex).
    ///
    /// Any other value draws nothing.
    pub fn debug_blit(
        &self,
        gl: &glow::Context,
        debug_stage: u32,
        output_fbo: Option<glow::Framebuffer>,
    ) {
        let (fw, fh) = self.full_size;
        if fw == 0 || fh == 0 { return; }
        // Every stage shows a product of the last run; before that they are all
        // uninitialized textures, so draw nothing rather than garbage.
        let Some(weight_tex) = self.last_weight_tex else { return };
        let tex = match debug_stage {
            1 => weight_tex,
            // Ping-pong parity depends on the JFA step count, so use the buffer
            // `run` actually finished in rather than assuming ping.
            2 => if self.jfa_result_in_ping { self.ping_tex } else { self.pong_tex },
            3 => self.firmness_tex,
            _ => return,
        };
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, output_fbo);
            gl.viewport(0, 0, fw, fh);
            gl.disable(glow::DEPTH_TEST);
            gl.bind_vertex_array(Some(self.vao));
            gl.use_program(Some(self.prog_passthrough));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if let Some(loc) = gl.get_uniform_location(self.prog_passthrough, "u_tex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.enable(glow::DEPTH_TEST);
            gl.bind_vertex_array(None);
        }
    }

    /// Access the weight mask texture (firmness output) for external use.
    /// Useful for debugging or for feeding into other effects.
    pub fn weight_mask_tex(&self) -> glow::Texture {
        self.firmness_tex
    }

    /// Update configuration. Textures are NOT resized — call resize() if downsample changed.
    pub fn set_config(&mut self, config: JfaConfig) {
        self.config = config;
    }

    pub fn config(&self) -> &JfaConfig {
        &self.config
    }
}

impl JfaPipeline {
    /// Explicitly release every GL object this pipeline created.
    ///
    /// `glow::Context` is not reachable from `Drop`, so cleanup cannot be automatic —
    /// call this before dropping the pipeline or the resources leak for the lifetime
    /// of the GL context. The lists below must stay in sync with the fields on
    /// [`JfaPipeline`]; every program, texture, framebuffer and VAO created in
    /// [`new`](Self::new) appears exactly once.
    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            for prog in [
                self.prog_init,
                self.prog_step,
                self.prog_firmness,
                self.prog_blur_dir,
                self.prog_weight_radial,
                self.prog_weight_conformal,
                self.prog_reduce_minmax,
                self.prog_weight_conformal_norm,
                self.prog_weight_kawase,
                self.prog_passthrough,
                self.prog_mip_composite,
                self.prog_gauss_down,
            ] {
                gl.delete_program(prog);
            }
            gl.delete_vertex_array(self.vao);
            for tex in [
                self.reduce_tex_a, self.reduce_tex_b,
                self.ping_tex, self.pong_tex,
                self.firmness_tex,
                self.blur_tex, self.blur_tex_b,
            ] {
                gl.delete_texture(tex);
            }
            for fbo in [
                self.reduce_fbo_a, self.reduce_fbo_b,
                self.ping_fbo, self.pong_fbo,
                self.firmness_fbo,
                self.blur_fbo, self.blur_fbo_b,
            ] {
                gl.delete_framebuffer(fbo);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_never_produces_empty_buffers() {
        assert_eq!(downsampled_size(1920, 1080, 2), (960, 540));
        assert_eq!(downsampled_size(1920, 1080, 4), (480, 270));
        // downsample=0 is treated as 1 rather than dividing by zero
        assert_eq!(downsampled_size(800, 600, 0), (800, 600));
        // A viewport smaller than the downsample factor still yields a 1x1 buffer
        assert_eq!(downsampled_size(3, 1, 8), (1, 1));
    }

    #[test]
    fn jfa_step_count_covers_the_buffer() {
        // The first jump is 2^(n-1), which must reach at least half way across the
        // buffer for the flood to be complete — equivalently 2^n >= max_dim.
        for &(w, h) in &[(1, 1), (2, 2), (17, 5), (960, 540), (1920, 1080), (4096, 2160)] {
            let n = jfa_step_count(w, h);
            let reach = 1u64 << n;
            assert!(
                reach >= w.max(h) as u64,
                "{w}x{h}: {n} steps only reach {reach}px",
            );
        }
    }

    #[test]
    fn jfa_step_count_is_logarithmic_not_linear() {
        // The whole pitch of the scatter formulation: cost grows with log of the
        // buffer size, not with the blur radius.
        assert_eq!(jfa_step_count(1, 1), 0);
        assert_eq!(jfa_step_count(2, 2), 1);
        assert_eq!(jfa_step_count(256, 256), 8);
        assert_eq!(jfa_step_count(257, 1), 9);
        // Doubling the buffer adds exactly one pass.
        assert_eq!(jfa_step_count(1024, 1024) + 1, jfa_step_count(2048, 2048));
    }

    #[test]
    fn mip_levels_stay_addressable() {
        // Every level the mip-blur path allocates must be at least one texel.
        for &(w, h) in &[(1, 1), (64, 64), (1920, 1080), (8192, 8192)] {
            let n = mip_level_count(w, h);
            assert!((1..=8).contains(&n), "{w}x{h} produced {n} levels");
            assert!(w >> (n - 1) >= 1 && h.max(w) >> (n - 1) >= 1);
        }
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(1920, 1080), 8); // capped
        assert_eq!(mip_level_count(64, 32), 7);
    }

    #[test]
    fn per_pass_strength_keeps_total_blur_stable() {
        // Independent Gaussian variances add, so N sub-passes at strength s/sqrt(N)
        // sum back to the requested strength.
        for passes in 1..=6u32 {
            let s = per_pass_blur_strength(1.0, passes);
            let total = (s * s * (passes * 3) as f32).sqrt();
            assert!((total - 1.0).abs() < 1e-5, "{passes} passes -> {total}");
        }
        // blur_passes=0 is clamped to a single pass instead of dividing by zero.
        assert_eq!(per_pass_blur_strength(1.0, 0), per_pass_blur_strength(1.0, 1));
    }

    #[test]
    fn default_config_is_internally_consistent() {
        let c = JfaConfig::default();
        assert!(c.downsample >= 1, "downsample must not be zero");
        assert!(c.blur_passes >= 1, "blur_passes must not be zero");
        assert!(c.bandwidth > 0.0, "zero bandwidth collapses the focus band");
        assert!(c.max_distance > 0.0);
        // JFA runs at the downsampled resolution, so max_distance is divided by
        // `downsample` when uploaded — that must stay a meaningful radius.
        assert!(c.max_distance / c.downsample as f32 >= 1.0);
    }
}
