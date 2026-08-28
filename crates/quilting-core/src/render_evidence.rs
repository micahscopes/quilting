//! Backend-neutral diagnostic image evidence.
//!
//! WebGL2 reads framebuffer rows bottom-first while WebGPU staging copies use
//! texture-native rows, and surface formats may expose RGBA or BGRA bytes.
//! This module removes those representation differences before comparison. It
//! deliberately owns no GPU API and is intended for explicit parity gates,
//! not per-frame telemetry.

use serde::{Serialize, Serializer};

pub const RENDER_IMAGE_MISMATCH_EXAMPLE_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderImageOrigin {
    TopLeft,
    BottomLeft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderImageChannelOrder {
    Rgba,
    Bgra,
}

#[derive(Clone, Copy, Debug)]
pub struct Rgba8ImageView<'a> {
    size: [u32; 2],
    bytes_per_row: usize,
    origin: RenderImageOrigin,
    channel_order: RenderImageChannelOrder,
    bytes: &'a [u8],
}

impl<'a> Rgba8ImageView<'a> {
    pub fn new(
        size: [u32; 2],
        bytes_per_row: usize,
        origin: RenderImageOrigin,
        channel_order: RenderImageChannelOrder,
        bytes: &'a [u8],
    ) -> Result<Self, RenderImageEvidenceError> {
        if size[0] == 0 || size[1] == 0 {
            return Err(RenderImageEvidenceError::ZeroSize);
        }
        let width =
            usize::try_from(size[0]).map_err(|_| RenderImageEvidenceError::DimensionsOverflow)?;
        let height =
            usize::try_from(size[1]).map_err(|_| RenderImageEvidenceError::DimensionsOverflow)?;
        let minimum_row = width
            .checked_mul(4)
            .ok_or(RenderImageEvidenceError::DimensionsOverflow)?;
        if bytes_per_row < minimum_row {
            return Err(RenderImageEvidenceError::RowTooShort {
                actual: bytes_per_row,
                minimum: minimum_row,
            });
        }
        let expected = bytes_per_row
            .checked_mul(height)
            .ok_or(RenderImageEvidenceError::DimensionsOverflow)?;
        if bytes.len() != expected {
            return Err(RenderImageEvidenceError::ByteLength {
                actual: bytes.len(),
                expected,
            });
        }
        Ok(Self {
            size,
            bytes_per_row,
            origin,
            channel_order,
            bytes,
        })
    }

    pub fn size(self) -> [u32; 2] {
        self.size
    }

    fn canonical_pixel(self, x: usize, y: usize) -> [u8; 4] {
        let height = self.size[1] as usize;
        let source_y = match self.origin {
            RenderImageOrigin::TopLeft => y,
            RenderImageOrigin::BottomLeft => height - 1 - y,
        };
        let offset = source_y * self.bytes_per_row + x * 4;
        let pixel: [u8; 4] = self.bytes[offset..offset + 4]
            .try_into()
            .expect("validated image rows contain complete RGBA8 pixels");
        match self.channel_order {
            RenderImageChannelOrder::Rgba => pixel,
            RenderImageChannelOrder::Bgra => [pixel[2], pixel[1], pixel[0], pixel[3]],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderImageEvidenceError {
    ZeroSize,
    DimensionsOverflow,
    RowTooShort {
        actual: usize,
        minimum: usize,
    },
    ByteLength {
        actual: usize,
        expected: usize,
    },
    ShapeMismatch {
        expected: [u32; 2],
        actual: [u32; 2],
    },
}

impl std::fmt::Display for RenderImageEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSize => formatter.write_str("render image dimensions must be nonzero"),
            Self::DimensionsOverflow => {
                formatter.write_str("render image dimensions exceed address space")
            }
            Self::RowTooShort { actual, minimum } => write!(
                formatter,
                "render image row has {actual} bytes; at least {minimum} are required"
            ),
            Self::ByteLength { actual, expected } => write!(
                formatter,
                "render image has {actual} bytes; expected exactly {expected}"
            ),
            Self::ShapeMismatch { expected, actual } => write!(
                formatter,
                "render image shape mismatch: expected {}x{}, got {}x{}",
                expected[0], expected[1], actual[0], actual[1]
            ),
        }
    }
}

impl std::error::Error for RenderImageEvidenceError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderImageSignature {
    pub size: [u32; 2],
    pub covered_pixels: u64,
    pub channel_sums: [u64; 4],
    pub channel_square_sums: [u64; 4],
    #[serde(serialize_with = "serialize_u64_hex")]
    pub rgba8_hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderImageMismatchExample {
    pub x: u32,
    pub y: u32,
    pub expected: [u8; 4],
    pub actual: [u8; 4],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderImageComparison {
    pub expected: RenderImageSignature,
    pub actual: RenderImageSignature,
    pub compared_pixels: u64,
    pub mismatched_pixels: u64,
    pub mismatched_pixel_millionths: u32,
    pub coverage_mismatches: u64,
    pub coverage_mismatch_millionths: u32,
    pub absolute_channel_error: [u64; 4],
    pub maximum_channel_delta: [u8; 4],
    pub mean_absolute_error_millionths: u32,
    pub examples: Vec<RenderImageMismatchExample>,
}

impl RenderImageComparison {
    pub fn is_exact(&self) -> bool {
        self.mismatched_pixels == 0
    }

    pub fn within(&self, tolerance: RenderImageTolerance) -> bool {
        self.maximum_channel_delta
            .iter()
            .zip(tolerance.maximum_channel_delta)
            .all(|(&actual, allowed)| actual <= allowed)
            && self.mean_absolute_error_millionths <= tolerance.mean_absolute_error_millionths
            && self.mismatched_pixel_millionths <= tolerance.mismatched_pixel_millionths
            && self.coverage_mismatch_millionths <= tolerance.coverage_mismatch_millionths
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderImageTolerance {
    pub maximum_channel_delta: [u8; 4],
    /// Mean absolute error as millionths of the full 0..255 RGBA range.
    pub mean_absolute_error_millionths: u32,
    pub mismatched_pixel_millionths: u32,
    pub coverage_mismatch_millionths: u32,
}

impl RenderImageTolerance {
    pub const EXACT: Self = Self {
        maximum_channel_delta: [0; 4],
        mean_absolute_error_millionths: 0,
        mismatched_pixel_millionths: 0,
        coverage_mismatch_millionths: 0,
    };
}

pub fn render_image_signature(
    image: Rgba8ImageView<'_>,
    coverage_alpha_threshold: u8,
) -> RenderImageSignature {
    let [width, height] = image.size;
    let mut covered_pixels = 0u64;
    let mut channel_sums = [0u64; 4];
    let mut channel_square_sums = [0u64; 4];
    let mut hash = 0xcbf29ce484222325u64;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = image.canonical_pixel(x, y);
            covered_pixels += u64::from(pixel[3] > coverage_alpha_threshold);
            for (channel, value) in pixel.into_iter().enumerate() {
                let value = u64::from(value);
                channel_sums[channel] = channel_sums[channel].saturating_add(value);
                channel_square_sums[channel] =
                    channel_square_sums[channel].saturating_add(value * value);
                hash ^= value;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    RenderImageSignature {
        size: image.size,
        covered_pixels,
        channel_sums,
        channel_square_sums,
        rgba8_hash: hash,
    }
}

pub fn compare_render_images(
    expected: Rgba8ImageView<'_>,
    actual: Rgba8ImageView<'_>,
    coverage_alpha_threshold: u8,
) -> Result<RenderImageComparison, RenderImageEvidenceError> {
    if expected.size != actual.size {
        return Err(RenderImageEvidenceError::ShapeMismatch {
            expected: expected.size,
            actual: actual.size,
        });
    }
    let [width, height] = expected.size;
    let compared_pixels = u64::from(width) * u64::from(height);
    let mut mismatched_pixels = 0u64;
    let mut coverage_mismatches = 0u64;
    let mut absolute_channel_error = [0u64; 4];
    let mut maximum_channel_delta = [0u8; 4];
    let mut examples = Vec::with_capacity(RENDER_IMAGE_MISMATCH_EXAMPLE_LIMIT);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let expected_pixel = expected.canonical_pixel(x, y);
            let actual_pixel = actual.canonical_pixel(x, y);
            if (expected_pixel[3] > coverage_alpha_threshold)
                != (actual_pixel[3] > coverage_alpha_threshold)
            {
                coverage_mismatches = coverage_mismatches.saturating_add(1);
            }
            if expected_pixel != actual_pixel {
                mismatched_pixels = mismatched_pixels.saturating_add(1);
                if examples.len() < RENDER_IMAGE_MISMATCH_EXAMPLE_LIMIT {
                    examples.push(RenderImageMismatchExample {
                        x: x as u32,
                        y: y as u32,
                        expected: expected_pixel,
                        actual: actual_pixel,
                    });
                }
            }
            for channel in 0..4 {
                let delta = expected_pixel[channel].abs_diff(actual_pixel[channel]);
                absolute_channel_error[channel] =
                    absolute_channel_error[channel].saturating_add(u64::from(delta));
                maximum_channel_delta[channel] = maximum_channel_delta[channel].max(delta);
            }
        }
    }
    let absolute_error = absolute_channel_error
        .iter()
        .fold(0u128, |sum, &value| sum + u128::from(value));
    let error_denominator = u128::from(compared_pixels) * 4 * 255;
    Ok(RenderImageComparison {
        expected: render_image_signature(expected, coverage_alpha_threshold),
        actual: render_image_signature(actual, coverage_alpha_threshold),
        compared_pixels,
        mismatched_pixels,
        mismatched_pixel_millionths: millionths(mismatched_pixels, compared_pixels),
        coverage_mismatches,
        coverage_mismatch_millionths: millionths(coverage_mismatches, compared_pixels),
        absolute_channel_error,
        maximum_channel_delta,
        mean_absolute_error_millionths: ratio_millionths(absolute_error, error_denominator),
        examples,
    })
}

fn millionths(numerator: u64, denominator: u64) -> u32 {
    ratio_millionths(u128::from(numerator), u128::from(denominator))
}

fn ratio_millionths(numerator: u128, denominator: u128) -> u32 {
    if denominator == 0 {
        return 0;
    }
    u32::try_from((numerator * 1_000_000) / denominator).unwrap_or(u32::MAX)
}

fn serialize_u64_hex<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&format!("{value:016x}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_channel_order_and_row_padding_normalize_exactly() {
        let top_left_rgba = [
            1, 2, 3, 255, 4, 5, 6, 0, // top row
            7, 8, 9, 128, 10, 11, 12, 255, // bottom row
        ];
        let bottom_left_bgra_padded = [
            9, 8, 7, 128, 12, 11, 10, 255, 99, 99, 99, 99, // bottom row
            3, 2, 1, 255, 6, 5, 4, 0, 88, 88, 88, 88, // top row
        ];
        let expected = Rgba8ImageView::new(
            [2, 2],
            8,
            RenderImageOrigin::TopLeft,
            RenderImageChannelOrder::Rgba,
            &top_left_rgba,
        )
        .unwrap();
        let actual = Rgba8ImageView::new(
            [2, 2],
            12,
            RenderImageOrigin::BottomLeft,
            RenderImageChannelOrder::Bgra,
            &bottom_left_bgra_padded,
        )
        .unwrap();

        let comparison = compare_render_images(expected, actual, 0).unwrap();

        assert!(comparison.is_exact());
        assert!(comparison.within(RenderImageTolerance::EXACT));
        assert_eq!(comparison.expected, comparison.actual);
        assert_eq!(comparison.expected.covered_pixels, 3);
    }

    #[test]
    fn comparison_reports_normalized_error_and_bounded_examples() {
        let expected = vec![0u8; 10 * 4];
        let mut actual = expected.clone();
        for (pixel, channels) in actual.chunks_exact_mut(4).enumerate() {
            channels[0] = pixel as u8 + 1;
            channels[3] = u8::from(pixel % 2 == 0) * 255;
        }
        let expected = Rgba8ImageView::new(
            [10, 1],
            40,
            RenderImageOrigin::TopLeft,
            RenderImageChannelOrder::Rgba,
            &expected,
        )
        .unwrap();
        let actual = Rgba8ImageView::new(
            [10, 1],
            40,
            RenderImageOrigin::TopLeft,
            RenderImageChannelOrder::Rgba,
            &actual,
        )
        .unwrap();

        let comparison = compare_render_images(expected, actual, 0).unwrap();

        assert_eq!(comparison.mismatched_pixels, 10);
        assert_eq!(comparison.mismatched_pixel_millionths, 1_000_000);
        assert_eq!(comparison.coverage_mismatches, 5);
        assert_eq!(comparison.coverage_mismatch_millionths, 500_000);
        assert_eq!(comparison.maximum_channel_delta, [10, 0, 0, 255]);
        assert_eq!(
            comparison.examples.len(),
            RENDER_IMAGE_MISMATCH_EXAMPLE_LIMIT
        );
        assert!(!comparison.within(RenderImageTolerance::EXACT));
    }

    #[test]
    fn malformed_images_fail_before_comparison() {
        assert_eq!(
            Rgba8ImageView::new(
                [2, 1],
                7,
                RenderImageOrigin::TopLeft,
                RenderImageChannelOrder::Rgba,
                &[0; 7],
            )
            .unwrap_err(),
            RenderImageEvidenceError::RowTooShort {
                actual: 7,
                minimum: 8,
            },
        );
        assert_eq!(
            Rgba8ImageView::new(
                [1, 2],
                4,
                RenderImageOrigin::TopLeft,
                RenderImageChannelOrder::Rgba,
                &[0; 4],
            )
            .unwrap_err(),
            RenderImageEvidenceError::ByteLength {
                actual: 4,
                expected: 8,
            },
        );
    }
}
