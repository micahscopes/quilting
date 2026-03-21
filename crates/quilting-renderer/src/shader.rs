//! Shader compilation: WGSL -> naga -> GLSL ES 300 -> glow GL programs.
//!
//! Uses quilting-shaders to compile WGSL modules with naga-oil imports,
//! then creates OpenGL shader programs via glow.

use glow::HasContext;

/// All compiled shader programs for the quilting rendering pipeline.
pub struct Programs {
    pub matcap: glow::Program,
    pub wire: glow::Program,
    pub normals: glow::Program,
}

impl Programs {
    /// Delete all GL programs.
    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.matcap);
            gl.delete_program(self.wire);
            gl.delete_program(self.normals);
        }
    }
}

/// Uniform block binding points.
/// Naga emits `layout(std140) uniform BlockName { ... }` blocks.
/// We bind these to UBO binding points.
pub const VERTEX_UNIFORMS_BINDING: u32 = 0;
pub const WIRE_UNIFORMS_BINDING: u32 = 1;

/// Compiled GLSL source for a vertex/fragment pair.
pub struct CompiledGlsl {
    pub vertex: String,
    pub fragment: String,
}

/// Fragment shader modes that quilting-shaders supports.
const FRAGMENT_MODES: &[&str] = &["matcap", "wire", "normals"];

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
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program()
            .map_err(|e| format!("create_program: {e}"))?;
        gl.attach_shader(program, vert);
        gl.attach_shader(program, frag);
        gl.link_program(program);

        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
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
    link_program(gl, vert, frag)
}

/// Bind uniform block indices to known binding points for a program.
/// Naga names uniform blocks like:
///   - `Uniforms_block_0Vertex` for @group(0) @binding(0) in vertex stage
///   - `WireUniforms_block_0Fragment` for @group(0) @binding(1) in fragment stage
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
    }
}

/// Compile all WGSL shaders to GLSL, create GL programs, and bind uniform blocks.
pub fn compile_programs(gl: &glow::Context) -> Result<Programs, String> {
    let glsl_sources = compile_all_glsl()?;

    let mut matcap = None;
    let mut wire = None;
    let mut normals = None;

    for (name, compiled) in &glsl_sources {
        let program = create_program(gl, &compiled.vertex, &compiled.fragment)
            .map_err(|e| format!("program '{name}': {e}"))?;

        bind_uniform_blocks(gl, program);

        match *name {
            "matcap" => matcap = Some(program),
            "wire" => wire = Some(program),
            "normals" => normals = Some(program),
            _ => {}
        }
    }

    Ok(Programs {
        matcap: matcap.ok_or("matcap program not compiled")?,
        wire: wire.ok_or("wire program not compiled")?,
        normals: normals.ok_or("normals program not compiled")?,
    })
}
