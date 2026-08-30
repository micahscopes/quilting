//! Backend-neutral diagnostic render evidence.
//!
//! WebGL2 reads framebuffer rows bottom-first while WebGPU staging copies use
//! texture-native rows, and surface formats may expose RGBA or BGRA bytes.
//! This module removes those representation differences before comparison. It
//! It also compares one source-addressed pick without conflating transient
//! renderer handles with application identity. The module deliberately owns no
//! GPU API and is intended for explicit parity gates, not per-frame telemetry.

use serde::{Deserialize, Serialize, Serializer};

pub const RENDER_IMAGE_MISMATCH_EXAMPLE_LIMIT: usize = 8;

/// Backend-neutral source-addressed result from one depth-tested patch query.
/// `packed_node` is deliberately transient; semantic identity is joined later
/// through Hyperscape's epoch-fenced interaction target table.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPickHit {
    pub packed_node: u32,
    pub source_face: u32,
    pub source_barycentric: [f32; 3],
    pub source_position: [f32; 3],
    pub output_distance: f32,
}

impl RenderPickHit {
    pub fn new(
        packed_node: u32,
        source_face: u32,
        source_barycentric: [f32; 3],
        source_position: [f32; 3],
        output_distance: f32,
    ) -> Result<Self, RenderPickEvidenceError> {
        let hit = Self {
            packed_node,
            source_face,
            source_barycentric,
            source_position,
            output_distance,
        };
        hit.validate()?;
        Ok(hit)
    }

    pub fn validate(self) -> Result<(), RenderPickEvidenceError> {
        if self
            .source_barycentric
            .into_iter()
            .chain(self.source_position)
            .chain([self.output_distance])
            .any(|value| !value.is_finite())
        {
            return Err(RenderPickEvidenceError::NonFinite);
        }
        if self.output_distance < 0.0
            || self
                .source_barycentric
                .into_iter()
                .any(|coordinate| coordinate < -1.0e-4)
        {
            return Err(RenderPickEvidenceError::OutsideSurface);
        }
        let sum = self.source_barycentric.into_iter().sum::<f32>();
        if (sum - 1.0).abs() > 1.0e-3 {
            return Err(RenderPickEvidenceError::UnnormalizedBarycentric { sum });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RenderPickEvidenceError {
    NonFinite,
    OutsideSurface,
    UnnormalizedBarycentric { sum: f32 },
    InvalidReport(&'static str),
    InconsistentComparison,
}

impl std::fmt::Display for RenderPickEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("render pick contains a non-finite value"),
            Self::OutsideSurface => {
                formatter.write_str("render pick lies outside the rendered surface")
            }
            Self::UnnormalizedBarycentric { sum } => write!(
                formatter,
                "render pick barycentric sum is {sum}; expected one",
            ),
            Self::InvalidReport(reason) => formatter.write_str(reason),
            Self::InconsistentComparison => formatter.write_str(
                "render pick comparison does not match its expected and actual samples",
            ),
        }
    }
}

impl std::error::Error for RenderPickEvidenceError {}

/// Exact topology plus measured numeric drift between an incumbent and a
/// candidate renderer query. Numeric fields exist only when both renderers
/// report a hit; a coverage mismatch cannot masquerade as zero error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPickComparison {
    pub expected: Option<RenderPickHit>,
    pub actual: Option<RenderPickHit>,
    pub coverage_matches: bool,
    pub identity_matches: bool,
    pub maximum_barycentric_error: Option<f32>,
    pub maximum_source_position_error: Option<f32>,
    pub output_distance_error: Option<f32>,
}

impl RenderPickComparison {
    pub fn between(
        expected: Option<RenderPickHit>,
        actual: Option<RenderPickHit>,
    ) -> Result<Self, RenderPickEvidenceError> {
        if let Some(hit) = expected {
            hit.validate()?;
        }
        if let Some(hit) = actual {
            hit.validate()?;
        }
        let coverage_matches = expected.is_some() == actual.is_some();
        let identity_matches = match (expected, actual) {
            (Some(expected), Some(actual)) => {
                expected.packed_node == actual.packed_node
                    && expected.source_face == actual.source_face
            }
            (None, None) => true,
            _ => false,
        };
        let errors = expected.zip(actual).map(|(expected, actual)| {
            (
                maximum_absolute_error(expected.source_barycentric, actual.source_barycentric),
                maximum_absolute_error(expected.source_position, actual.source_position),
                (expected.output_distance - actual.output_distance).abs(),
            )
        });
        Ok(Self {
            expected,
            actual,
            coverage_matches,
            identity_matches,
            maximum_barycentric_error: errors.map(|errors| errors.0),
            maximum_source_position_error: errors.map(|errors| errors.1),
            output_distance_error: errors.map(|errors| errors.2),
        })
    }

    pub fn topology_matches(self) -> bool {
        self.coverage_matches && self.identity_matches
    }

    pub fn within(self, tolerance: RenderPickTolerance) -> bool {
        if !self.topology_matches() {
            return false;
        }
        match (
            self.maximum_barycentric_error,
            self.maximum_source_position_error,
            self.output_distance_error,
        ) {
            (None, None, None) => true,
            (Some(barycentric), Some(source_position), Some(output_distance)) => {
                barycentric <= tolerance.maximum_barycentric_error
                    && source_position <= tolerance.maximum_source_position_error
                    && output_distance <= tolerance.maximum_output_distance_error
            }
            _ => false,
        }
    }
}

/// One incumbent/candidate pick comparison tied to exact retained renderer and
/// interaction-residency epochs. The packet is diagnostic evidence, never an
/// interaction command.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPickEvidenceReport {
    pub webgl_render_call: u64,
    pub webgpu_frame_revision: u64,
    pub viewport: [u32; 2],
    pub pixel: [u32; 2],
    pub target_epoch: u32,
    pub comparison: RenderPickComparison,
    pub staging_ms: f64,
    pub readback_ms: f64,
    pub total_ms: f64,
}

impl RenderPickEvidenceReport {
    pub fn validate(self) -> Result<(), RenderPickEvidenceError> {
        if self.webgl_render_call == 0 || self.webgpu_frame_revision == 0 {
            return Err(RenderPickEvidenceError::InvalidReport(
                "render pick evidence requires nonzero frame identities",
            ));
        }
        if self.viewport.into_iter().any(|extent| extent == 0)
            || self.pixel[0] >= self.viewport[0]
            || self.pixel[1] >= self.viewport[1]
        {
            return Err(RenderPickEvidenceError::InvalidReport(
                "render pick evidence pixel lies outside its nonempty viewport",
            ));
        }
        if [self.staging_ms, self.readback_ms, self.total_ms]
            .into_iter()
            .any(|duration| !duration.is_finite() || duration < 0.0)
        {
            return Err(RenderPickEvidenceError::InvalidReport(
                "render pick evidence timings must be finite and nonnegative",
            ));
        }
        if self.total_ms < self.staging_ms || self.total_ms < self.readback_ms {
            return Err(RenderPickEvidenceError::InvalidReport(
                "render pick evidence total time cannot be shorter than a measured phase",
            ));
        }
        let canonical = RenderPickComparison::between(
            self.comparison.expected,
            self.comparison.actual,
        )?;
        if canonical != self.comparison {
            return Err(RenderPickEvidenceError::InconsistentComparison);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPickTolerance {
    pub maximum_barycentric_error: f32,
    pub maximum_source_position_error: f32,
    pub maximum_output_distance_error: f32,
}

impl RenderPickTolerance {
    pub const EXACT: Self = Self {
        maximum_barycentric_error: 0.0,
        maximum_source_position_error: 0.0,
        maximum_output_distance_error: 0.0,
    };
}

fn maximum_absolute_error<const N: usize>(expected: [f32; N], actual: [f32; N]) -> f32 {
    expected
        .into_iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f32::max)
}

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

    fn pick(face: u32, node: u32, barycentric: [f32; 3]) -> RenderPickHit {
        RenderPickHit::new(node, face, barycentric, [1.0, -2.0, 0.5], 4.0).unwrap()
    }

    #[test]
    fn pick_comparison_keeps_misses_distinct_from_zero_error_hits() {
        let misses = RenderPickComparison::between(None, None).unwrap();
        assert!(misses.topology_matches());
        assert!(misses.within(RenderPickTolerance::EXACT));
        assert_eq!(misses.maximum_barycentric_error, None);

        let expected = pick(3, 7, [0.2, 0.3, 0.5]);
        let coverage_mismatch = RenderPickComparison::between(Some(expected), None).unwrap();
        assert!(!coverage_mismatch.coverage_matches);
        assert!(!coverage_mismatch.identity_matches);
        assert!(!coverage_mismatch.within(RenderPickTolerance {
            maximum_barycentric_error: 1.0,
            maximum_source_position_error: 1.0,
            maximum_output_distance_error: 1.0,
        }));
        assert_eq!(coverage_mismatch.maximum_barycentric_error, None);
    }

    #[test]
    fn pick_comparison_reports_topology_and_numeric_drift_independently() {
        let expected = pick(3, 7, [0.2, 0.3, 0.5]);
        let actual =
            RenderPickHit::new(7, 3, [0.201, 0.297, 0.502], [1.004, -2.0, 0.499], 4.006).unwrap();
        let comparison = RenderPickComparison::between(Some(expected), Some(actual)).unwrap();
        assert!(comparison.topology_matches());
        assert!((comparison.maximum_barycentric_error.unwrap() - 0.003).abs() < 1.0e-6);
        assert!((comparison.maximum_source_position_error.unwrap() - 0.004).abs() < 1.0e-6);
        assert!((comparison.output_distance_error.unwrap() - 0.006).abs() < 1.0e-6);
        assert!(comparison.within(RenderPickTolerance {
            maximum_barycentric_error: 0.004,
            maximum_source_position_error: 0.005,
            maximum_output_distance_error: 0.007,
        }));
        assert!(!comparison.within(RenderPickTolerance::EXACT));

        let wrong_face =
            RenderPickComparison::between(Some(expected), Some(pick(4, 7, [0.2, 0.3, 0.5])))
                .unwrap();
        assert!(wrong_face.coverage_matches);
        assert!(!wrong_face.identity_matches);
        assert!(!wrong_face.topology_matches());
    }

    #[test]
    fn malformed_pick_evidence_fails_before_comparison() {
        assert_eq!(
            RenderPickHit::new(7, 3, [0.2, 0.3, 0.4], [0.0; 3], 1.0).unwrap_err(),
            RenderPickEvidenceError::UnnormalizedBarycentric { sum: 0.9 },
        );
        assert_eq!(
            RenderPickHit::new(7, 3, [0.2, 0.3, 0.5], [f32::NAN, 0.0, 0.0], 1.0).unwrap_err(),
            RenderPickEvidenceError::NonFinite,
        );
    }

    fn pick_report(comparison: RenderPickComparison) -> RenderPickEvidenceReport {
        RenderPickEvidenceReport {
            webgl_render_call: 17,
            webgpu_frame_revision: 9,
            viewport: [1600, 900],
            pixel: [812, 417],
            target_epoch: 29,
            comparison,
            staging_ms: 0.2,
            readback_ms: 0.8,
            total_ms: 1.1,
        }
    }

    #[test]
    fn pick_report_round_trips_with_canonical_camel_case_fields() {
        let hit = pick(3, 7, [0.2, 0.3, 0.5]);
        let report = pick_report(
            RenderPickComparison::between(Some(hit), Some(hit)).unwrap(),
        );
        report.validate().unwrap();
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["webglRenderCall"], 17);
        assert_eq!(json["webgpuFrameRevision"], 9);
        assert_eq!(json["targetEpoch"], 29);
        assert_eq!(json["comparison"]["coverageMatches"], true);
        assert_eq!(
            serde_json::from_value::<RenderPickEvidenceReport>(json).unwrap(),
            report,
        );
    }

    #[test]
    fn pick_report_rejects_impossible_geometry_timing_and_comparison_claims() {
        let hit = pick(3, 7, [0.2, 0.3, 0.5]);
        let comparison = RenderPickComparison::between(Some(hit), Some(hit)).unwrap();

        let mut outside = pick_report(comparison);
        outside.pixel[0] = outside.viewport[0];
        assert!(matches!(
            outside.validate(),
            Err(RenderPickEvidenceError::InvalidReport(_))
        ));

        let mut impossible_time = pick_report(comparison);
        impossible_time.readback_ms = impossible_time.total_ms + 1.0;
        assert!(matches!(
            impossible_time.validate(),
            Err(RenderPickEvidenceError::InvalidReport(_))
        ));

        let mut contradictory = pick_report(comparison);
        contradictory.comparison.identity_matches = false;
        assert_eq!(
            contradictory.validate(),
            Err(RenderPickEvidenceError::InconsistentComparison),
        );
    }

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
