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
    /// Float precision for intermediate textures
    pub precision: Precision,
}

impl Default for JfaConfig {
    fn default() -> Self {
        JfaConfig {
            max_distance: 64.0,
            blur_strength: 1.0,
            downsample: 2,
            precision: Precision::Float16,
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
uniform vec2 u_dims;
uniform float u_max_distance;

void main() {
    ivec2 coords = ivec2(v_uv * u_dims);
    vec4 jfa = texelFetch(u_jfa, coords, 0);

    if (jfa.z <= 0.0) {
        o_color = vec4(0.0);
        return;
    }

    float dist = distance(v_uv * u_dims, jfa.xy * u_dims);
    float effective_max = u_max_distance * jfa.z;
    float ratio = clamp(dist / max(effective_max, 1.0), 0.0, 1.0);
    float falloff = 1.0 - smoothstep(0.0, 1.0, ratio);
    float weight = jfa.z * falloff;

    // Output: RG=seed_uv, B=attenuated weight, A=distance ratio
    o_color = vec4(jfa.xy, weight, ratio);
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
    float blur_weight = w.z * (1.0 - clamp(w.w, 0.0, 1.0));
    vec4 original = texture(u_scene, v_uv);

    if (blur_weight <= 0.001 || u_blur_strength <= 0.001) {
        o_color = original;
        return;
    }

    float effective_radius = u_blur_radius * blur_weight * u_blur_strength;
    float sigma = max(effective_radius / 2.0, 0.001);
    int radius = min(int(ceil(effective_radius)), MAX_RADIUS);
    float texel = 1.0 / float(textureSize(u_scene, 0).x);

    vec4 color_sum = vec4(0.0);
    float weight_sum = 0.0;

    for (int x = -radius; x <= radius; x++) {
        float d = float(abs(x));
        float gw = exp(-(d * d) / (2.0 * sigma * sigma));
        vec2 uv = vec2(v_uv.x + float(x) * texel, v_uv.y);
        uv = clamp(uv, vec2(0.0), vec2(1.0));
        color_sum += texture(u_scene, uv) * gw;
        weight_sum += gw;
    }

    o_color = color_sum / max(weight_sum, 0.001);
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
    float blur_weight = w.z * (1.0 - clamp(w.w, 0.0, 1.0));
    vec4 original = texture(u_scene, v_uv);

    if (blur_weight <= 0.001 || u_blur_strength <= 0.001) {
        o_color = original;
        return;
    }

    float effective_radius = u_blur_radius * blur_weight * u_blur_strength;
    float sigma = max(effective_radius / 2.0, 0.001);
    int radius = min(int(ceil(effective_radius)), MAX_RADIUS);
    float texel = 1.0 / float(textureSize(u_scene, 0).y);

    vec4 color_sum = vec4(0.0);
    float weight_sum = 0.0;

    for (int y = -radius; y <= radius; y++) {
        float d = float(abs(y));
        float gw = exp(-(d * d) / (2.0 * sigma * sigma));
        vec2 uv = vec2(v_uv.x, v_uv.y + float(y) * texel);
        uv = clamp(uv, vec2(0.0), vec2(1.0));
        color_sum += texture(u_scene, uv) * gw;
        weight_sum += gw;
    }

    o_color = color_sum / max(weight_sum, 0.001);
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
    vao: glow::VertexArray,
    // Ping-pong FBOs + textures for JFA steps
    ping_fbo: glow::Framebuffer,
    ping_tex: glow::Texture,
    pong_fbo: glow::Framebuffer,
    pong_tex: glow::Texture,
    // Firmness output
    firmness_fbo: glow::Framebuffer,
    firmness_tex: glow::Texture,
    // Blur intermediate
    blur_fbo: glow::Framebuffer,
    blur_tex: glow::Texture,
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
        let vao = unsafe { gl.create_vertex_array().map_err(|e| format!("{e}"))? };

        let fmt = match config.precision {
            Precision::Float32 => glow::RGBA32F,
            Precision::Float16 => glow::RGBA16F,
        };

        // Allocate with placeholder 1x1 — resize() will set real size
        let (ping_fbo, ping_tex) = create_fbo_tex(gl, 1, 1, fmt)?;
        let (pong_fbo, pong_tex) = create_fbo_tex(gl, 1, 1, fmt)?;
        let (firmness_fbo, firmness_tex) = create_fbo_tex(gl, 1, 1, fmt)?;
        let (blur_fbo, blur_tex) = create_fbo_tex(gl, 1, 1, glow::RGBA8)?;

        Ok(JfaPipeline {
            prog_init, prog_step, prog_firmness, prog_blur_h, prog_blur_v, prog_weight_radial,
            vao, ping_fbo, ping_tex, pong_fbo, pong_tex,
            firmness_fbo, firmness_tex, blur_fbo, blur_tex,
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
            // Resize blur intermediate (full res, RGBA8)
            gl.bind_texture(glow::TEXTURE_2D, Some(self.blur_tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, width, height, 0,
                glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
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
        let (fw, fh) = self.full_size;
        let (jw, jh) = self.jfa_size;
        if fw == 0 || fh == 0 { return; }

        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(self.vao));

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
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_jfa") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_dims") {
                // Use JFA dims since that's what the JFA texture is at
                gl.uniform_2_f32(Some(&loc), jw as f32, jh as f32);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_firmness, "u_max_distance") {
                gl.uniform_1_f32(Some(&loc), self.config.max_distance / self.config.downsample as f32);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // --- Stage 4: Weighted Gaussian blur (separable H+V) ---
            // H pass: scene → blur_tex
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.blur_fbo));
            gl.viewport(0, 0, fw, fh);
            gl.use_program(Some(self.prog_blur_h));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(scene_tex));
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
                gl.uniform_1_f32(Some(&loc), self.config.blur_strength);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // V pass: blur_tex → output
            gl.bind_framebuffer(glow::FRAMEBUFFER, output_fbo);
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
                gl.uniform_1_f32(Some(&loc), self.config.blur_strength);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // Restore state
            gl.enable(glow::DEPTH_TEST);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
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
            gl.delete_vertex_array(self.vao);
            for tex in [self.ping_tex, self.pong_tex, self.firmness_tex, self.blur_tex] {
                gl.delete_texture(tex);
            }
            for fbo in [self.ping_fbo, self.pong_fbo, self.firmness_fbo, self.blur_fbo] {
                gl.delete_framebuffer(fbo);
            }
        }
    }
}
