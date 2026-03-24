//! GPU texture management: upload glTF images, resolve material texture references.

use glow::HasContext;

/// Manages GPU textures for a loaded model.
pub struct TextureCache {
    textures: Vec<Option<glow::Texture>>,
    placeholder: glow::Texture,
}

impl TextureCache {
    /// Create a new cache with a 1x1 white placeholder texture.
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        let placeholder = create_placeholder(gl)?;
        Ok(TextureCache {
            textures: Vec::new(),
            placeholder,
        })
    }

    /// Upload RGBA8 images from glTF and store them indexed by image index.
    pub fn upload_images(
        &mut self,
        gl: &glow::Context,
        images: &[(u32, u32, &[u8])], // (width, height, rgba_pixels)
    ) {
        // Delete old textures
        for tex in self.textures.drain(..) {
            if let Some(t) = tex {
                unsafe { gl.delete_texture(t); }
            }
        }

        self.textures = images.iter().map(|&(width, height, pixels)| {
            upload_rgba8(gl, width, height, pixels).ok()
        }).collect();
    }

    /// Resolve an image index to a GPU texture handle, with placeholder fallback.
    pub fn get(&self, index: Option<usize>) -> glow::Texture {
        index
            .and_then(|i| self.textures.get(i))
            .and_then(|t| *t)
            .unwrap_or(self.placeholder)
    }

    /// The 1x1 white placeholder.
    pub fn placeholder(&self) -> glow::Texture {
        self.placeholder
    }

    pub fn destroy(&mut self, gl: &glow::Context) {
        for tex in self.textures.drain(..) {
            if let Some(t) = tex {
                unsafe { gl.delete_texture(t); }
            }
        }
        unsafe { gl.delete_texture(self.placeholder); }
    }
}

fn create_placeholder(gl: &glow::Context) -> Result<glow::Texture, String> {
    unsafe {
        let tex = gl.create_texture().map_err(|e| format!("placeholder tex: {e}"))?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        let white = [255u8, 255, 255, 255];
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            1,
            1,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(&white)),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        Ok(tex)
    }
}

fn upload_rgba8(
    gl: &glow::Context,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<glow::Texture, String> {
    unsafe {
        let tex = gl.create_texture().map_err(|e| format!("tex upload: {e}"))?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(pixels)),
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::REPEAT as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::REPEAT as i32,
        );
        Ok(tex)
    }
}

/// Bind a material's textures to the expected texture units for PBR rendering.
///
/// Units: 0=base_color, 1=metallic_roughness, 2=normal, 3=emissive, 4=occlusion
///        5=env_prefiltered, 6=env_irradiance, 7=sheen_lut
pub fn bind_material_textures(
    gl: &glow::Context,
    mat_textures: &super::buffer::MaterialTextures,
    env: &super::buffer::EnvironmentMaps,
    placeholder: glow::Texture,
) {
    let bind = |unit: u32, tex: Option<glow::Texture>| {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex.unwrap_or(placeholder)));
        }
    };
    let bind_cube = |unit: u32, tex: Option<glow::Texture>| {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            if let Some(t) = tex {
                gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(t));
            }
        }
    };

    bind(0, mat_textures.base_color);
    bind(1, mat_textures.metallic_roughness);
    bind(2, mat_textures.normal);
    bind(3, mat_textures.emissive);
    bind(4, mat_textures.occlusion);
    bind_cube(5, env.prefiltered);
    bind_cube(6, env.irradiance);
    bind(7, env.sheen_lut);
}
