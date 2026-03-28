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
            downsample: 1, // full-res JFA eliminates 2x2 block artifacts
            blur_passes: 2,
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

/// Normalize stretch using min/max from reduction pass, then apply Gaussian band.
const FS_WEIGHT_CONFORMAL_NORM: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_stretch;
uniform sampler2D u_minmax;   // 1x1 texture: R=global min, G=global max
uniform float u_focus;
uniform float u_bandwidth;

void main() {
    float raw = texture(u_stretch, v_uv).r;
    // Read global min/max from 1x1 reduction result
    vec4 mm = texture(u_minmax, vec2(0.5));
    float mn = mm.r;
    float mx = mm.g;
    // Normalize to [0,1] using actual range
    float range = max(mx - mn, 0.001);
    float normalized = (raw - mn) / range;
    // Inverted Gaussian: focused band is SHARP, everything else blurred
    float diff = normalized - u_focus;
    float bw = max(u_bandwidth, 0.01);
    float sharpness = exp(-(diff * diff) / (2.0 * bw * bw));
    float w = 1.0 - sharpness;
    o_color = vec4(w, 0.0, 0.0, 1.0);
}
"#;

/// Focused conformal weight: read raw stretch texture, apply Gaussian band selection.
/// Focus and bandwidth are uniforms controlling which stretch band gets blurred.
const FS_WEIGHT_CONFORMAL: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_stretch;
uniform float u_focus;     // 0.5 = neutral, 0 = squash, 1 = expand
uniform float u_bandwidth; // width of Gaussian band (smaller = tighter)

void main() {
    float raw_stretch = texture(u_stretch, v_uv).r; // [0,1] via sigmoid
    // Inverted Gaussian: focused band is SHARP, everything else blurred
    float diff = raw_stretch - u_focus;
    float bw = max(u_bandwidth, 0.01);
    float sharpness = exp(-(diff * diff) / (2.0 * bw * bw));
    float w = 1.0 - sharpness;
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
uniform float u_max_distance;

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
            // Weight bonus: higher-weight seeds are more attractive
            float bonus = neighbor.z * max(float(step_size), u_max_distance * 0.05);
            float adj_dist = dist - bonus;
            float adj_best = best_dist - best.z * max(float(step_size), u_max_distance * 0.05);

            if (adj_dist < adj_best - 0.5 || (abs(adj_dist - adj_best) <= 0.5 && neighbor.z > best.z)) {
                best = neighbor;
                best_dist = dist;
            }
        }
    }

    o_color = best;
}
"#;

/// Firmness blend: apply distance-based falloff to JFA result.
/// Converts raw JFA output to a smooth weight mask.
const FS_JFA_FIRMNESS: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_jfa;
uniform sampler2D u_analytical;  // original full-res weight texture
uniform vec2 u_dims;
uniform float u_max_distance;

void main() {
    // Read analytical weight at full resolution (pixel-perfect boundary)
    float analytical_w = texture(u_analytical, v_uv).r;

    // Bilinear interpolation from downsampled JFA
    vec2 texel = 1.0 / u_dims;
    vec2 sp = v_uv * u_dims - 0.5;
    vec2 base = floor(sp) * texel + texel * 0.5;
    vec2 f = fract(sp);
    vec4 j00 = texture(u_jfa, base);
    vec4 j10 = texture(u_jfa, base + vec2(texel.x, 0.0));
    vec4 j01 = texture(u_jfa, base + vec2(0.0, texel.y));
    vec4 j11 = texture(u_jfa, base + texel);
    vec4 jfa = mix(mix(j00, j10, f.x), mix(j01, j11, f.x), f.y);

    // Hybrid boundary: analytical weight is authoritative where it exists.
    // JFA only extends the blur BEYOND the analytical region (falloff zone).
    // This prevents JFA from ever bleeding into the sharp region.
    float weight;
    float ratio = 0.0;
    if (analytical_w > 0.001) {
        // Inside the analytical weight region — use it directly (pixel-perfect)
        weight = analytical_w;
    } else if (jfa.z > 0.0) {
        // Outside analytical region — use JFA distance falloff
        float dist = distance(v_uv * u_dims, jfa.xy * u_dims);
        float effective_max = u_max_distance * jfa.z;
        ratio = clamp(dist / max(effective_max, 1.0), 0.0, 1.0);
        float falloff = 1.0 - smoothstep(0.0, 1.0, ratio);
        weight = jfa.z * falloff;
    } else {
        weight = 0.0;
    }

    if (weight <= 0.0) {
        o_color = vec4(0.0);
        return;
    }

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

/// Pre-blur the analytical weight — smooths per-face scallops at the boundary.
/// Separable 5-tap Gaussian, radius ~2px. Applied to weight texture before JFA init.
const FS_WEIGHT_PREBLUR_H: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_weight;

void main() {
    vec2 texel = vec2(1.0 / float(textureSize(u_weight, 0).x), 0.0);
    float w0 = texture(u_weight, v_uv - 2.0 * texel).r * 0.06;
    float w1 = texture(u_weight, v_uv - texel).r * 0.24;
    float w2 = texture(u_weight, v_uv).r * 0.40;
    float w3 = texture(u_weight, v_uv + texel).r * 0.24;
    float w4 = texture(u_weight, v_uv + 2.0 * texel).r * 0.06;
    o_color = vec4(w0 + w1 + w2 + w3 + w4, 0.0, 0.0, 1.0);
}
"#;

const FS_WEIGHT_PREBLUR_V: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_weight;

void main() {
    vec2 texel = vec2(0.0, 1.0 / float(textureSize(u_weight, 0).y));
    float w0 = texture(u_weight, v_uv - 2.0 * texel).r * 0.06;
    float w1 = texture(u_weight, v_uv - texel).r * 0.24;
    float w2 = texture(u_weight, v_uv).r * 0.40;
    float w3 = texture(u_weight, v_uv + texel).r * 0.24;
    float w4 = texture(u_weight, v_uv + 2.0 * texel).r * 0.06;
    o_color = vec4(w0 + w1 + w2 + w3 + w4, 0.0, 0.0, 1.0);
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

/// Weighted Gaussian blur — horizontal pass.
/// Blur radius varies per-pixel based on the weight mask.
const FS_BLUR_H: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_scene;
uniform sampler2D u_weight;
uniform float u_blur_radius;
uniform float u_blur_strength;

const int MAX_RADIUS = 48;

void main() {
    vec4 w = texture(u_weight, v_uv);
    float blur_weight = clamp(w.z * (1.0 - clamp(w.w, 0.0, 1.0)), 0.0, 1.0);
    vec4 original = texture(u_scene, v_uv);

    if (u_blur_strength <= 0.001) { o_color = original; return; }

    float effective_radius = u_blur_radius * blur_weight * u_blur_strength;
    float sigma = max(effective_radius / 2.0, 0.001);
    int radius = min(max(int(ceil(effective_radius)), 1), MAX_RADIUS);
    float texel = 1.0 / float(textureSize(u_scene, 0).x);

    vec4 color_sum = vec4(0.0);
    float weight_sum = 0.0;
    for (int x = -radius; x <= radius; x++) {
        float d = float(abs(x));
        float gw = exp(-(d * d) / (2.0 * sigma * sigma));
        vec2 uv = clamp(vec2(v_uv.x + float(x) * texel, v_uv.y), vec2(0.0), vec2(1.0));
        color_sum += texture(u_scene, uv) * gw;
        weight_sum += gw;
    }

    vec4 blurred = color_sum / max(weight_sum, 0.001);
    o_color = mix(original, blurred, blur_weight);
}
"#;

/// Weighted Gaussian blur — vertical pass.
const FS_BLUR_V: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_scene;
uniform sampler2D u_weight;
uniform float u_blur_radius;
uniform float u_blur_strength;

const int MAX_RADIUS = 48;

void main() {
    vec4 w = texture(u_weight, v_uv);
    float blur_weight = clamp(w.z * (1.0 - clamp(w.w, 0.0, 1.0)), 0.0, 1.0);
    vec4 original = texture(u_scene, v_uv);

    if (u_blur_strength <= 0.001) { o_color = original; return; }

    float effective_radius = u_blur_radius * blur_weight * u_blur_strength;
    float sigma = max(effective_radius / 2.0, 0.001);
    int radius = min(max(int(ceil(effective_radius)), 1), MAX_RADIUS);
    float texel = 1.0 / float(textureSize(u_scene, 0).y);

    vec4 color_sum = vec4(0.0);
    float weight_sum = 0.0;
    for (int y = -radius; y <= radius; y++) {
        float d = float(abs(y));
        float gw = exp(-(d * d) / (2.0 * sigma * sigma));
        vec2 uv = clamp(vec2(v_uv.x, v_uv.y + float(y) * texel), vec2(0.0), vec2(1.0));
        color_sum += texture(u_scene, uv) * gw;
        weight_sum += gw;
    }

    vec4 blurred = color_sum / max(weight_sum, 0.001);
    o_color = mix(original, blurred, blur_weight);
}
"#;

/// Compiled JFA programs and GPU resources.
pub struct JfaPipeline {
    prog_init: glow::Program,
    prog_step: glow::Program,
    prog_firmness: glow::Program,
    prog_blur_h: glow::Program,
    prog_blur_v: glow::Program,
    prog_weight_radial: glow::Program,
    prog_weight_conformal: glow::Program,
    prog_reduce_minmax: glow::Program,
    prog_weight_conformal_norm: glow::Program,
    prog_weight_kawase: glow::Program,
    prog_preblur_h: glow::Program,
    prog_preblur_v: glow::Program,
    prog_passthrough: glow::Program,
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
    // Smoothed weight (preserved for debug — not overwritten by scene blur)
    smoothed_fbo: glow::Framebuffer,
    smoothed_tex: glow::Texture,
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
}

fn compile_shader(gl: &glow::Context, shader_type: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type).map_err(|e| format!("{e}"))?;
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
        let prog = gl.create_program().map_err(|e| format!("{e}"))?;
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
        let tex = gl.create_texture().map_err(|e| format!("{e}"))?;
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

        let fbo = gl.create_framebuffer().map_err(|e| format!("{e}"))?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        Ok((fbo, tex))
    }
}

impl JfaPipeline {
    /// Create the JFA pipeline. Call once at init time.
    pub fn new(gl: &glow::Context, config: JfaConfig) -> Result<Self, String> {
        let prog_init = link_program(gl, VS_FULLSCREEN, FS_JFA_INIT)?;
        let prog_step = link_program(gl, VS_FULLSCREEN, FS_JFA_STEP)?;
        let prog_firmness = link_program(gl, VS_FULLSCREEN, FS_JFA_FIRMNESS)?;
        let prog_blur_h = link_program(gl, VS_FULLSCREEN, FS_BLUR_H)?;
        let prog_blur_v = link_program(gl, VS_FULLSCREEN, FS_BLUR_V)?;
        let prog_weight_radial = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_RADIAL)?;
        let prog_weight_conformal = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_CONFORMAL)?;
        let prog_reduce_minmax = link_program(gl, VS_FULLSCREEN, FS_REDUCE_MINMAX)?;
        let prog_weight_conformal_norm = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_CONFORMAL_NORM)?;
        let prog_weight_kawase = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_KAWASE)?;
        let prog_preblur_h = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_PREBLUR_H)?;
        let prog_preblur_v = link_program(gl, VS_FULLSCREEN, FS_WEIGHT_PREBLUR_V)?;
        let prog_passthrough = link_program(gl, VS_FULLSCREEN, FS_PASSTHROUGH)?;
        let vao = unsafe { gl.create_vertex_array().map_err(|e| format!("{e}"))? };
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
        let (smoothed_fbo, smoothed_tex) = create_fbo_tex(gl, 1, 1, glow::RGBA8)?;
        let (blur_fbo, blur_tex) = create_fbo_tex(gl, 1, 1, glow::RGBA8)?;
        let (blur_fbo_b, blur_tex_b) = create_fbo_tex(gl, 1, 1, glow::RGBA8)?;

        Ok(JfaPipeline {
            prog_init, prog_step, prog_firmness, prog_blur_h, prog_blur_v,
            prog_weight_radial, prog_weight_conformal,
            prog_reduce_minmax, prog_weight_conformal_norm, prog_weight_kawase,
            prog_preblur_h, prog_preblur_v, prog_passthrough,
            reduce_fbo_a, reduce_tex_a, reduce_fbo_b, reduce_tex_b,
            vao, ping_fbo, ping_tex, pong_fbo, pong_tex,
            firmness_fbo, firmness_tex, smoothed_fbo, smoothed_tex,
            blur_fbo, blur_tex, blur_fbo_b, blur_tex_b,
            jfa_size: (0, 0), full_size: (0, 0), config, internal_format: fmt,
        })
    }

    /// Resize internal textures to match viewport. Call when viewport changes.
    pub fn resize(&mut self, gl: &glow::Context, width: i32, height: i32) {
        if self.full_size == (width, height) { return; }
        self.full_size = (width, height);

        let ds = self.config.downsample.max(1) as i32;
        let jw = (width / ds).max(1);
        let jh = (height / ds).max(1);
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
            // Resize smoothed weight + blur ping-pong (full res, RGBA8)
            gl.bind_texture(glow::TEXTURE_2D, Some(self.smoothed_tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, width, height, 0,
                glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
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
        &self,
        gl: &glow::Context,
        weight_tex: glow::Texture,
        scene_tex: glow::Texture,
        output_fbo: Option<glow::Framebuffer>,
    ) {
        // Store weight_tex for the firmness shader's analytical hybrid
        let analytical_weight_tex = weight_tex;
        let (fw, fh) = self.full_size;
        let (jw, jh) = self.jfa_size;
        if fw == 0 || fh == 0 { return; }

        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(self.vao));

            // --- Stage 0.5: Pre-blur analytical weight to smooth per-face scallops ---
            // Two passes of separable 5-tap Gaussian (~4px effective radius).
            // Pass 1: weight_tex → blur_tex_b (H) → smoothed_tex (V)
            // Pass 2: smoothed_tex → blur_tex_b (H) → smoothed_tex (V)
            let mut preblur_src = weight_tex;
            for _ in 0..2 {
                // H pass
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blur_fbo_b));
                gl.viewport(0, 0, fw, fh);
                gl.use_program(Some(self.prog_preblur_h));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(preblur_src));
                if let Some(loc) = gl.get_uniform_location(self.prog_preblur_h, "u_weight") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                // V pass
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.smoothed_fbo));
                gl.use_program(Some(self.prog_preblur_v));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.blur_tex_b));
                if let Some(loc) = gl.get_uniform_location(self.prog_preblur_v, "u_weight") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                preblur_src = self.smoothed_tex;
            }
            let smoothed_weight = self.smoothed_tex;

            // --- Stage 1: Init seeds from weight texture (into ping, at JFA resolution) ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.ping_fbo));
            gl.viewport(0, 0, jw, jh);
            gl.use_program(Some(self.prog_init));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(smoothed_weight));
            if let Some(loc) = gl.get_uniform_location(self.prog_init, "u_weight") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // --- Stage 2: JFA propagation (log2 passes, ping-pong) ---
            let max_dim = jw.max(jh) as u32;
            let num_steps = (max_dim as f32).log2().ceil() as u32;
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
                if let Some(loc) = gl.get_uniform_location(self.prog_step, "u_max_distance") {
                    gl.uniform_1_f32(Some(&loc), self.config.max_distance / self.config.downsample as f32);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                read_from_ping = !read_from_ping;
            }

            // JFA result is in whichever buffer was last written to
            let jfa_result_tex = if read_from_ping { self.ping_tex } else { self.pong_tex };

            // --- Stage 3: Firmness blend (full resolution) ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.firmness_fbo));
            gl.viewport(0, 0, fw, fh);
            gl.use_program(Some(self.prog_firmness));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(jfa_result_tex));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(smoothed_weight));
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_jfa") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_analytical") {
                gl.uniform_1_i32(Some(&loc), 1);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_dims") {
                gl.uniform_2_f32(Some(&loc), jw as f32, jh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_max_distance") {
                gl.uniform_1_f32(Some(&loc), self.config.max_distance / self.config.downsample as f32);
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

            // --- Stage 4: Multi-pass weighted Gaussian blur (separable H+V) ---
            // Each pass reads from the previous output, doubling effective kernel width.
            // Pass 1: scene → blur_tex (H) → blur_tex_b (V)
            // Pass 2: blur_tex_b → blur_tex (H) → blur_tex_b (V)
            // ...
            // Final V writes to output_fbo.
            let num_passes = self.config.blur_passes.max(1);
            let per_pass_strength = self.config.blur_strength / (num_passes as f32).sqrt();

            for pass in 0..num_passes {
                let src = if pass == 0 { scene_tex } else { self.blur_tex_b };
                let is_last = pass == num_passes - 1;

                // H pass: src → blur_tex
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blur_fbo));
                gl.viewport(0, 0, fw, fh);
                gl.use_program(Some(self.prog_blur_h));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(src));
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.firmness_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_h, "u_scene") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_h, "u_weight") {
                    gl.uniform_1_i32(Some(&loc), 1);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_h, "u_blur_radius") {
                    gl.uniform_1_f32(Some(&loc), self.config.max_distance);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_h, "u_blur_strength") {
                    gl.uniform_1_f32(Some(&loc), per_pass_strength);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);

                // V pass: blur_tex → blur_tex_b (or output on last pass)
                let v_dst = if is_last { output_fbo } else { Some(self.blur_fbo_b) };
                gl.bind_framebuffer(glow::FRAMEBUFFER, v_dst);
                gl.viewport(0, 0, fw, fh);
                gl.use_program(Some(self.prog_blur_v));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.blur_tex));
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.firmness_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_v, "u_scene") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_v, "u_weight") {
                    gl.uniform_1_i32(Some(&loc), 1);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_v, "u_blur_radius") {
                    gl.uniform_1_f32(Some(&loc), self.config.max_distance);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_blur_v, "u_blur_strength") {
                    gl.uniform_1_f32(Some(&loc), per_pass_strength);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }

            // Restore state
            gl.enable(glow::DEPTH_TEST);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Generate a focused conformal weight from a raw stretch texture.
    /// If config.normalize is true, reduces to find global min/max first.
    /// Otherwise uses absolute sigmoid values directly.
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

            let has_cpu_range = self.config.cpu_stretch_min < self.config.cpu_stretch_max - 0.001;

            if self.config.normalize && has_cpu_range {
                // --- Normalized mode with CPU-sampled range (stable, no GPU reduction) ---
                // Write min/max into a 1x1 texture for the normalization shader
                let minmax_tex = self.reduce_tex_a;
                gl.bind_texture(glow::TEXTURE_2D, Some(minmax_tex));
                let minmax_data = [self.config.cpu_stretch_min, self.config.cpu_stretch_max, 0.0_f32, 0.0];
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA16F as i32, 1, 1, 0,
                    glow::RGBA, glow::FLOAT, glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&minmax_data))));

                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_fbo));
                gl.viewport(0, 0, width, height);
                gl.use_program(Some(self.prog_weight_conformal_norm));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(stretch_tex));
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(minmax_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_stretch") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_minmax") {
                    gl.uniform_1_i32(Some(&loc), 1);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_focus") {
                    gl.uniform_1_f32(Some(&loc), self.config.focus);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_bandwidth") {
                    gl.uniform_1_f32(Some(&loc), self.config.bandwidth);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            } else if self.config.normalize {
                // --- Normalized mode: GPU reduce to find min/max ---
                let mut src_tex = stretch_tex;
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

                // Normalized weight with Gaussian band
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_fbo));
                gl.viewport(0, 0, width, height);
                gl.use_program(Some(self.prog_weight_conformal_norm));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(stretch_tex));
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(src_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_stretch") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_minmax") {
                    gl.uniform_1_i32(Some(&loc), 1);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_focus") {
                    gl.uniform_1_f32(Some(&loc), self.config.focus);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal_norm, "u_bandwidth") {
                    gl.uniform_1_f32(Some(&loc), self.config.bandwidth);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            } else {
                // --- Absolute mode: use raw sigmoid stretch values directly ---
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(output_fbo));
                gl.viewport(0, 0, width, height);
                gl.use_program(Some(self.prog_weight_conformal));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(stretch_tex));
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_stretch") {
                    gl.uniform_1_i32(Some(&loc), 0);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_focus") {
                    gl.uniform_1_f32(Some(&loc), self.config.focus);
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_weight_conformal, "u_bandwidth") {
                    gl.uniform_1_f32(Some(&loc), self.config.bandwidth);
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
        &self,
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

    /// Debug: draw an internal texture to the output framebuffer for visualization.
    /// 1=smoothed weight (pre-JFA), 2=JFA result, 3=firmness mask
    pub fn debug_blit(
        &self,
        gl: &glow::Context,
        debug_stage: u32,
        output_fbo: Option<glow::Framebuffer>,
    ) {
        let (fw, fh) = self.full_size;
        if fw == 0 || fh == 0 { return; }
        let tex = match debug_stage {
            1 => self.smoothed_tex,  // smoothed weight (preserved)
            2 => self.ping_tex,      // JFA result
            3 => self.firmness_tex,  // firmness
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

impl Drop for JfaPipeline {
    fn drop(&mut self) {
        // Note: glow::Context not available in Drop. Resources leak on drop.
        // Call destroy() explicitly before dropping if cleanup is needed.
    }
}

impl JfaPipeline {
    /// Explicitly destroy GPU resources.
    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.prog_init);
            gl.delete_program(self.prog_step);
            gl.delete_program(self.prog_firmness);
            gl.delete_program(self.prog_blur_h);
            gl.delete_program(self.prog_blur_v);
            gl.delete_program(self.prog_weight_radial);
            gl.delete_program(self.prog_weight_conformal);
            gl.delete_program(self.prog_reduce_minmax);
            gl.delete_program(self.prog_weight_conformal_norm);
            gl.delete_vertex_array(self.vao);
            for tex in [self.ping_tex, self.pong_tex, self.firmness_tex, self.blur_tex, self.blur_tex_b,
                        self.reduce_tex_a, self.reduce_tex_b] {
                gl.delete_texture(tex);
            }
            for fbo in [self.ping_fbo, self.pong_fbo, self.firmness_fbo, self.blur_fbo, self.blur_fbo_b,
                        self.reduce_fbo_a, self.reduce_fbo_b] {
                gl.delete_framebuffer(fbo);
            }
        }
    }
}
