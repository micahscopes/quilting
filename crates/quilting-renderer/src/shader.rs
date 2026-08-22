//! Shader compilation: WGSL -> naga -> GLSL ES 300 -> glow GL programs.
//!
//! Uses quilting-shaders to compile WGSL modules with naga-oil imports,
//! then creates OpenGL shader programs via glow.

use glow::HasContext;

#[cfg(target_arch = "wasm32")]
fn log_info(msg: &str) {
    web_sys::console::info_1(&msg.into());
}
#[cfg(not(target_arch = "wasm32"))]
fn log_info(msg: &str) {
    eprintln!("{}", msg);
}

/// All compiled shader programs for the quilting rendering pipeline.
pub struct Programs {
    pub matcap: glow::Program,
    pub wire: glow::Program,
    pub normals: glow::Program,
    pub pbr: glow::Program,
    pub stretch: glow::Program,
    pub pick: glow::Program,
}

impl Programs {
    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.matcap);
            gl.delete_program(self.wire);
            gl.delete_program(self.normals);
            gl.delete_program(self.pbr);
            gl.delete_program(self.stretch);
            gl.delete_program(self.pick);
        }
    }
}

/// Uniform block binding points.
/// Naga emits `layout(std140) uniform BlockName { ... }` blocks.
/// We bind these to UBO binding points.
pub const VERTEX_UNIFORMS_BINDING: u32 = 0;
pub const WIRE_UNIFORMS_BINDING: u32 = 1;
pub const PBR_UNIFORMS_BINDING: u32 = 2;
pub const JOINT_MATRICES_BINDING: u32 = 4;

/// Compiled GLSL source for a vertex/fragment pair.
pub struct CompiledGlsl {
    pub vertex: String,
    pub fragment: String,
}

/// Fragment shader modes that quilting-shaders supports.
pub(crate) const FRAGMENT_MODES: &[&str] =
    &["matcap", "wire", "normals", "pbr", "stretch", "pick"];

/// Compile all WGSL shaders to GLSL via quilting-shaders (naga).
/// Uses the "native" emission path (no Y-flip / Z-remap).
/// Returns raw GLSL strings for each program.
pub fn compile_all_glsl() -> Result<Vec<(&'static str, CompiledGlsl)>, String> {
    let vertex_glsl = quilting_shaders::compile_vertex_glsl_native()
        .map_err(|e| format!("vertex GLSL: {e}"))?;

    let mut programs = Vec::new();

    for &mode in FRAGMENT_MODES {
        let frag_glsl = quilting_shaders::compile_fragment_glsl_native(mode)
            .map_err(|e| format!("{mode} fragment GLSL: {e}"))?;

        programs.push((mode, CompiledGlsl {
            vertex: vertex_glsl.clone(),
            fragment: frag_glsl,
        }));
    }

    Ok(programs)
}

/// Compile a single GL shader (vertex or fragment) from GLSL source.
fn compile_gl_shader(
    gl: &glow::Context,
    shader_type: u32,
    source: &str,
) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type)
            .map_err(|e| format!("create_shader: {e}"))?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);

        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(format!("shader compile error:\n{log}\n\nSource:\n{source}"));
        }

        Ok(shader)
    }
}

/// Link a vertex + fragment shader into a GL program.
fn link_program(
    gl: &glow::Context,
    vert: glow::Shader,
    frag: glow::Shader,
    transform_feedback_varyings: Option<&[&str]>,
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program()
            .map_err(|e| {
                gl.delete_shader(vert);
                gl.delete_shader(frag);
                format!("create_program: {e}")
            })?;
        gl.attach_shader(program, vert);
        gl.attach_shader(program, frag);
        if let Some(varyings) = transform_feedback_varyings {
            gl.transform_feedback_varyings(program, varyings, glow::INTERLEAVED_ATTRIBS);
        }
        gl.link_program(program);

        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            gl.delete_shader(vert);
            gl.delete_shader(frag);
            return Err(format!("program link error: {log}"));
        }

        // Shaders can be detached after linking
        gl.detach_shader(program, vert);
        gl.detach_shader(program, frag);
        gl.delete_shader(vert);
        gl.delete_shader(frag);

        Ok(program)
    }
}

/// Create a GL program from GLSL vertex + fragment source.
pub fn create_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, String> {
    let vert = compile_gl_shader(gl, glow::VERTEX_SHADER, vertex_src)?;
    let frag = compile_gl_shader(gl, glow::FRAGMENT_SHADER, fragment_src)
        .map_err(|e| {
            unsafe { gl.delete_shader(vert); }
            e
        })?;
    link_program(gl, vert, frag, None)
}

/// Create a transform-feedback GL program from GLSL source. Varyings are
/// interleaved in the supplied order, which therefore defines the destination
/// buffer's record layout.
pub fn create_transform_feedback_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
    varyings: &[&str],
) -> Result<glow::Program, String> {
    let vert = compile_gl_shader(gl, glow::VERTEX_SHADER, vertex_src)?;
    let frag = compile_gl_shader(gl, glow::FRAGMENT_SHADER, fragment_src)
        .map_err(|e| {
            unsafe { gl.delete_shader(vert); }
            e
        })?;
    link_program(gl, vert, frag, Some(varyings))
}

/// Bind uniform block indices to known binding points for a program.
/// Naga names uniform blocks like:
///   - `Uniforms_block_0Vertex` for @group(0) @binding(0) in vertex stage
///   - `WireUniforms_block_0Fragment` for @group(0) @binding(1) in fragment stage
///   - `PbrUniforms_block_1Fragment` for @group(0) @binding(1) in PBR fragment stage
pub fn bind_uniform_blocks(gl: &glow::Context, program: glow::Program) {
    unsafe {
        // Vertex uniform block: Uniforms at group(0) binding(0)
        if let Some(idx) = gl.get_uniform_block_index(program, "Uniforms_block_0Vertex") {
            gl.uniform_block_binding(program, idx, VERTEX_UNIFORMS_BINDING);
        }

        // Wire fragment uniform block: WireUniforms at group(0) binding(1)
        if let Some(idx) = gl.get_uniform_block_index(program, "WireUniforms_block_0Fragment") {
            gl.uniform_block_binding(program, idx, WIRE_UNIFORMS_BINDING);
        }

        // Enumerate ALL uniform blocks and bind appropriately.
        {
            let num_blocks = gl.get_program_parameter_i32(program, glow::ACTIVE_UNIFORM_BLOCKS);
            let mut all_names = Vec::new();
            for i in 0..(num_blocks as u32) {
                let name = gl.get_active_uniform_block_name(program, i);
                all_names.push(name.clone());
                // Auto-bind by name pattern
                if name.contains("Pbr") || name.contains("pbr") {
                    gl.uniform_block_binding(program, i, PBR_UNIFORMS_BINDING);
                }
            }
            if num_blocks > 2 {
                log_info(&format!("All {} UBO blocks: {:?}", num_blocks, all_names));
            }
        }

        // Joint matrices: JointMatrices at group(0) binding(1) in vertex stage
        if let Some(idx) = gl.get_uniform_block_index(program, "JointMatrices_block_1Vertex") {
            gl.uniform_block_binding(program, idx, JOINT_MATRICES_BINDING);
        }

        // Matcap fragment uniform block: MatcapUniforms at group(0) binding(1)
        // Try multiple possible naga-generated block names
        for name in &[
            "MatcapUniforms_block_0Fragment",
            "matcap_u_block_0Fragment",
            "MatcapUniforms_block_1Fragment",
        ] {
            if let Some(idx) = gl.get_uniform_block_index(program, name) {
                gl.uniform_block_binding(program, idx, WIRE_UNIFORMS_BINDING);
                break;
            }
        }

        // Texture samplers: bind to matching units
        gl.use_program(Some(program));
        // Vertex shader textures
        if let Some(loc) = gl.get_uniform_location(program, "_group_0_binding_2_vs") {
            gl.uniform_1_i32(Some(&loc), SKINNING_TEX_UNIT as i32);
        }
        if let Some(loc) = gl.get_uniform_location(program, "_group_0_binding_3_vs") {
            gl.uniform_1_i32(Some(&loc), MORPH_TEX_UNIT as i32);
        }
        // Fragment shader textures — bind to texture units matching prototype layout
        // Matcap: binding 2/3 → unit 0
        // PBR: bindings 2-17 → units 0-7 (base_color, mr, normal, emissive, occlusion, env, irrad, sheen)
        let fs_sampler_bindings: &[(u32, i32)] = &[
            (2, 0),   // base_color_tex / matcap_tex
            (3, 0),   // base_color_sampler / matcap_sampler
            (4, 1),   // metallic_roughness_tex
            (5, 1),   // metallic_roughness_sampler
            (6, 2),   // normal_tex
            (7, 2),   // normal_sampler
            (8, 3),   // emissive_tex
            (9, 3),   // emissive_sampler
            (10, 4),  // occlusion_tex
            (11, 4),  // occlusion_sampler
            (12, 5),  // env_prefiltered
            (13, 5),  // env_prefiltered_sampler
            (14, 6),  // env_irradiance
            (15, 6),  // env_irradiance_sampler
            (16, 7),  // sheen_e_lut
            (17, 7),  // sheen_e_sampler
            (18, 8),  // scene_color_tex (transmission refraction)
            (19, 8),  // scene_color_sampler
            (20, 9),  // scene_color_blurred
            (21, 9),  // scene_color_blurred_sampler
            (22, 10), // transmission_tex
            (23, 10), // transmission_tex_sampler
        ];
        for &(binding, unit) in fs_sampler_bindings {
            let name = format!("_group_0_binding_{}_fs", binding);
            if let Some(loc) = gl.get_uniform_location(program, &name) {
                gl.uniform_1_i32(Some(&loc), unit);
            }
        }
        gl.use_program(None);

        // Log which blocks were found (for debugging binding issues)
        let mut found = Vec::new();
        for name in &[
            "Uniforms_block_0Vertex", "JointMatrices_block_1Vertex",
            "WireUniforms_block_0Fragment", "PbrUniforms_block_1Fragment",
            "MatcapUniforms_block_0Fragment", "matcap_u_block_0Fragment",
            "MatcapUniforms_block_1Fragment",
        ] {
            if gl.get_uniform_block_index(program, name).is_some() {
                found.push(*name);
            }
        }
        log_info(&format!("UBO blocks bound for program: {:?}", found));
    }
}

/// Texture units matching the prototype's GL texture binding.
pub const SKINNING_TEX_UNIT: u32 = 15;
pub const MORPH_TEX_UNIT: u32 = 14;

/// Compile all WGSL shaders to GLSL, create GL programs, and bind uniform blocks.
pub fn compile_programs(gl: &glow::Context) -> Result<Programs, String> {
    let glsl_sources = compile_all_glsl()?;

    let mut matcap = None;
    let mut wire = None;
    let mut normals = None;
    let mut pbr = None;
    let mut stretch = None;
    let mut pick = None;

    for (name, compiled) in &glsl_sources {
        log_info(&format!("Compiling program '{}': VS {} chars, FS {} chars",
            name, compiled.vertex.len(), compiled.fragment.len()));

        let program = create_program(gl, &compiled.vertex, &compiled.fragment)
            .map_err(|e| format!("program '{name}': {e}"))?;

        bind_uniform_blocks(gl, program);

        match *name {
            "matcap" => matcap = Some(program),
            "wire" => wire = Some(program),
            "normals" => normals = Some(program),
            "pbr" => pbr = Some(program),
            "stretch" => stretch = Some(program),
            "pick" => pick = Some(program),
            _ => {}
        }
    }
    log_info(&format!("All {} shader programs compiled and linked", glsl_sources.len()));

    Ok(Programs {
        matcap: matcap.ok_or("matcap program not compiled")?,
        wire: wire.ok_or("wire program not compiled")?,
        normals: normals.ok_or("normals program not compiled")?,
        pbr: pbr.ok_or("pbr program not compiled")?,
        stretch: stretch.ok_or("stretch program not compiled")?,
        pick: pick.ok_or("pick program not compiled")?,
    })
}
