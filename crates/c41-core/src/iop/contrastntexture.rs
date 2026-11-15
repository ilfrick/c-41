//! Pure Contrast & Texture parameter preparation and full-frame resource planning.
//!
//! Parameter and ROI formulas are ported from darktable commit
//! `97912518a5a9a9786aeee3c38116fbc999412ff8`,
//! `src/iop/contrastntexture.c`. This module does not process pixels and the
//! memory plan is not yet consumed by a pipeline stage.

use crate::resource_budget::ResourceBudget;

/// Darktable's maximum number of C&T detail levels.
pub const MAX_ITERATIONS: usize = 20;

/// User-facing C&T parameters from darktable's version-2 parameter struct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastTextureParams {
    /// Coarse-band gain, range -1..=5, default 0.
    pub gain_coarse: f32,
    /// Medium-band gain, range -1..=5, default 0.
    pub gain_medium: f32,
    /// Fine-band gain, range -1..=5, default 0.
    pub gain_fine: f32,
    /// Logarithmic detail scale, range 1..=15, default 5.
    pub detail_level: f32,
    /// Halo-control setting, range -10..=10, default 0.
    pub halo_control: f32,
    /// Luminance noise bias, range 0..=1, default 0.001.
    pub noise_bias: f32,
    /// Shadow gain, range -5..=5, default 0.
    pub gain_shadows: f32,
    /// Highlight gain, range -5..=5, default 0.
    pub gain_highlights: f32,
}

impl Default for ContrastTextureParams {
    fn default() -> Self {
        Self {
            gain_coarse: 0.0,
            gain_medium: 0.0,
            gain_fine: 0.0,
            detail_level: 5.0,
            halo_control: 0.0,
            noise_bias: 0.001,
            gain_shadows: 0.0,
            gain_highlights: 0.0,
        }
    }
}

/// Parameter-preparation failures; invalid persisted parameters are not clamped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepareError {
    InvalidParameter,
    InvalidDimensions,
    InvalidScale,
    ArithmeticOverflow,
}

impl ContrastTextureParams {
    /// Validate all fields against the pinned darktable UI ranges and finiteness.
    pub fn validate(self) -> Result<Self, PrepareError> {
        let fields = [
            (self.gain_coarse, -1.0, 5.0),
            (self.gain_medium, -1.0, 5.0),
            (self.gain_fine, -1.0, 5.0),
            (self.detail_level, 1.0, 15.0),
            (self.halo_control, -10.0, 10.0),
            (self.noise_bias, 0.0, 1.0),
            (self.gain_shadows, -5.0, 5.0),
            (self.gain_highlights, -5.0, 5.0),
        ];
        if fields
            .iter()
            .any(|(value, min, max)| !value.is_finite() || *value < *min || *value > *max)
        {
            return Err(PrepareError::InvalidParameter);
        }
        Ok(self)
    }

    /// Migrate the pinned darktable version-1 parameter layout to version 2.
    /// `filter_iterations` is retained for the legacy signature but ignored by
    /// darktable's migration, matching the source implementation.
    pub fn migrate_v1(
        gain_local_contrast: f32,
        detail_level: f32,
        edge_protection: f32,
        _filter_iterations: i32,
        noise_bias: f32,
    ) -> Result<Self, PrepareError> {
        let legacy_gain = gain_local_contrast - 1.0;
        Self {
            gain_coarse: legacy_gain,
            gain_medium: legacy_gain,
            gain_fine: legacy_gain,
            detail_level,
            halo_control: edge_protection,
            noise_bias,
            gain_shadows: 0.0,
            gain_highlights: 0.0,
        }
        .validate()
    }
}

/// Derived values shared by the later per-pixel C&T implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedContrastTexture {
    /// `log2(max_piece_size * piece_scale) - detail_level - 1`.
    pub max_detail_level: f32,
    /// Highest processed level, selected by darktable's radius-<2 rule.
    pub max_used_level: usize,
    /// Full-frame blur radius for each detail level, in processing pixels.
    pub radius_details: [f32; MAX_ITERATIONS],
    /// Per-level scale factors from darktable's `commit_params`.
    pub scale_details: [f32; MAX_ITERATIONS],
    /// Per-level noise-bias values.
    pub noise_bias: [f32; MAX_ITERATIONS],
    /// Guided-filter feathering after halo-control conversion.
    pub feathering: f32,
    /// Log-domain shadow slope.
    pub slope_shadows: f32,
    /// Log-domain highlight slope.
    pub slope_highlights: f32,
    /// Fixed width of the midtone transition.
    pub midtones_width: f32,
    /// Quadratic coefficients for the midtone transition.
    pub midtones_polynomial: [f32; 3],
}

fn finite(value: f32) -> Result<f32, PrepareError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PrepareError::InvalidScale)
    }
}

/// Prepare C&T values from both darktable image scales and the active ROI size.
///
/// `max_piece_dimension` is `max(piece.iwidth, piece.iheight)` before scaling.
/// `piece_scale` is `piece.iscale`, used by darktable's `max_detail_level`;
/// `roi_scale` is `roi_in.scale`, used by `modify_roi_in` to derive radii.
/// `width` and `height` are the actual current ROI frame dimensions and are
/// separate because ROI size does not determine the full-image detail scale.
pub fn prepare(
    params: ContrastTextureParams,
    max_piece_dimension: f32,
    piece_scale: f32,
    roi_scale: f32,
    width: usize,
    height: usize,
) -> Result<PreparedContrastTexture, PrepareError> {
    let params = params.validate()?;
    if width == 0 || height == 0 {
        return Err(PrepareError::InvalidDimensions);
    }
    width
        .checked_mul(height)
        .ok_or(PrepareError::ArithmeticOverflow)?;
    if !max_piece_dimension.is_finite()
        || max_piece_dimension <= 0.0
        || !piece_scale.is_finite()
        || piece_scale <= 0.0
        || !roi_scale.is_finite()
        || roi_scale <= 0.0
    {
        return Err(PrepareError::InvalidScale);
    }

    let max_image_size = finite(max_piece_dimension * piece_scale)?;
    let base_diameter = finite(max_piece_dimension * roi_scale)?;
    if max_image_size <= 0.0 || base_diameter <= 0.0 {
        return Err(PrepareError::InvalidScale);
    }
    let max_detail_level = finite(max_image_size.log2() - params.detail_level - 1.0)?;
    let slope_shadows = finite(2.0_f32.powf(-params.gain_shadows))?;
    let slope_highlights = finite(2.0_f32.powf(params.gain_highlights))?;
    let midtones_width = 0.3;
    let mut scale_details = [0.0; MAX_ITERATIONS];
    let mut noise_bias = [0.0; MAX_ITERATIONS];
    let mut radius_details = [0.0; MAX_ITERATIONS];

    for level in 0..MAX_ITERATIONS {
        let level_f = level as f32;
        scale_details[level] = finite(2.0_f32.powf(-params.detail_level - level_f))?;
        noise_bias[level] = finite(
            params.noise_bias
                * 2.0_f32.powf((level_f - max_detail_level).min(0.0)),
        )?;
        radius_details[level] = finite(
            0.5 * scale_details[level].min(0.5) * base_diameter,
        )?;
    }

    let mut max_used_level = MAX_ITERATIONS - 1;
    for level in (0..MAX_ITERATIONS).rev() {
        if radius_details[level] < 2.0 {
            max_used_level = level;
        }
    }

    let midtones_polynomial = [
        -slope_shadows * midtones_width,
        slope_shadows,
        (slope_highlights - slope_shadows) / (4.0 * midtones_width),
    ];
    if midtones_polynomial.iter().any(|value| !value.is_finite()) {
        return Err(PrepareError::InvalidScale);
    }

    Ok(PreparedContrastTexture {
        max_detail_level,
        max_used_level,
        radius_details,
        scale_details,
        noise_bias,
        feathering: finite(0.2 * 2.0_f32.powf(-params.halo_control))?,
        slope_shadows,
        slope_highlights,
        midtones_width,
        midtones_polynomial,
    })
}

/// Errors from a checked, full-frame memory plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPlanError {
    InvalidDimensions,
    InvalidSigma,
    TooManyLevels,
    FloatToSize,
    ArithmeticOverflow,
    BudgetExceeded { required: u64, available: u64 },
    BufferExceeded { required: u64, limit: u64 },
}

/// A peak estimate for all C&T-owned and helper-owned buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContrastTextureMemoryPlan {
    /// Number of full-resolution pixels.
    pub pixels: usize,
    /// Maximum downsampled pixels over all used C&T levels.
    pub max_downsampled_pixels: usize,
    /// Number of used levels represented by `per_level_downsampled_pixels`.
    pub level_count: usize,
    /// Downsampled pixel count per used level; remaining entries are zero.
    pub per_level_downsampled_pixels: [usize; MAX_ITERATIONS],
    /// Estimated peak bytes, including `external_live_bytes`.
    pub estimated_bytes: u64,
    /// Largest individual buffer, including the caller's external maximum.
    pub largest_buffer_bytes: u64,
}

/// Compute a conservative C&T full-frame peak plan before processing.
///
/// This accounts one RGBA frame (`4N` floats), three full-resolution C&T
/// luminance buffers (`3N`), the EIGF helper's full-resolution averages (`2N`),
/// and its simultaneously-live downsampled buffers (`15D`). The 15D term is
/// `D + 2D + 4D + 4D + 4D`: downsampled guide, packed moments, blurred RGBA,
/// packed RGBA input for Gaussian, and `Gaussian.buf`. The averages are `2N`,
/// not `2D`. Thus the internal peak is `9N + 15D` floats. The helper uses f32
/// division before integer truncation for downsample dimensions, matching its
/// allocation behavior.
///
/// `external_live_bytes` accounts for other simultaneously-live pipeline data;
/// `external_largest_buffer_bytes` is the largest individual external buffer.
/// Allocator metadata/capacity overhead and Linux overcommit are not included,
/// so this estimate is a resource-policy check, not a general OOM guarantee.
pub fn memory_plan(
    width: usize,
    height: usize,
    used_sigmas: &[f32],
    external_live_bytes: u64,
    external_largest_buffer_bytes: u64,
    budget: ResourceBudget,
) -> Result<ContrastTextureMemoryPlan, MemoryPlanError> {
    if width == 0 || height == 0 || used_sigmas.is_empty() {
        return Err(MemoryPlanError::InvalidDimensions);
    }
    if used_sigmas.len() > MAX_ITERATIONS {
        return Err(MemoryPlanError::TooManyLevels);
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let mut per_level_downsampled_pixels = [0; MAX_ITERATIONS];
    let mut max_downsampled_pixels = 0;

    for (level, &sigma) in used_sigmas.iter().enumerate() {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(MemoryPlanError::InvalidSigma);
        }
        let scaling = sigma.min(4.0).max(1.0);
        let downsampled_width = (width as f32 / scaling) as usize;
        let downsampled_height = (height as f32 / scaling) as usize;
        if downsampled_width == 0 || downsampled_height == 0 {
            return Err(MemoryPlanError::InvalidDimensions);
        }
        let downsampled_pixels = downsampled_width
            .checked_mul(downsampled_height)
            .ok_or(MemoryPlanError::ArithmeticOverflow)?;
        per_level_downsampled_pixels[level] = downsampled_pixels;
        max_downsampled_pixels = max_downsampled_pixels.max(downsampled_pixels);
    }

    let full_rgba_floats = pixels
        .checked_mul(4)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let luminance_floats = pixels
        .checked_mul(3)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let averages_floats = pixels
        .checked_mul(2)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let helper_downsampled_floats = max_downsampled_pixels
        .checked_mul(15)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let internal_floats = full_rgba_floats
        .checked_add(luminance_floats)
        .and_then(|n| n.checked_add(averages_floats))
        .and_then(|n| n.checked_add(helper_downsampled_floats))
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let internal_bytes = u64::try_from(internal_floats)
        .map_err(|_| MemoryPlanError::ArithmeticOverflow)?
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let estimated_bytes = internal_bytes
        .checked_add(external_live_bytes)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;

    let helper_largest_floats = max_downsampled_pixels
        .checked_mul(4)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let largest_internal_floats = full_rgba_floats.max(helper_largest_floats);
    let largest_internal_bytes = u64::try_from(largest_internal_floats)
        .map_err(|_| MemoryPlanError::ArithmeticOverflow)?
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or(MemoryPlanError::ArithmeticOverflow)?;
    let largest_buffer_bytes = largest_internal_bytes.max(external_largest_buffer_bytes);

    if largest_buffer_bytes > budget.cpu_single_buffer_bytes {
        return Err(MemoryPlanError::BufferExceeded {
            required: largest_buffer_bytes,
            limit: budget.cpu_single_buffer_bytes,
        });
    }
    if estimated_bytes > budget.cpu_available_bytes {
        return Err(MemoryPlanError::BudgetExceeded {
            required: estimated_bytes,
            available: budget.cpu_available_bytes,
        });
    }

    Ok(ContrastTextureMemoryPlan {
        pixels,
        max_downsampled_pixels,
        level_count: used_sigmas.len(),
        per_level_downsampled_pixels,
        estimated_bytes,
        largest_buffer_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn large_budget() -> ResourceBudget {
        ResourceBudget {
            cpu_available_bytes: u64::MAX,
            cpu_single_buffer_bytes: u64::MAX,
            mipmap_bytes: 0,
            opencl_available_bytes: 0,
        }
    }

    #[test]
    fn defaults_and_v1_migration_match_darktable() {
        assert_eq!(
            ContrastTextureParams::default(),
            ContrastTextureParams {
                gain_coarse: 0.0,
                gain_medium: 0.0,
                gain_fine: 0.0,
                detail_level: 5.0,
                halo_control: 0.0,
                noise_bias: 0.001,
                gain_shadows: 0.0,
                gain_highlights: 0.0,
            }
        );
        let migrated = ContrastTextureParams::migrate_v1(2.0, 5.0, 1.0, 99, 0.001).unwrap();
        assert_eq!(migrated.gain_coarse, 1.0);
        assert_eq!(migrated.gain_medium, 1.0);
        assert_eq!(migrated.gain_fine, 1.0);
        assert_eq!(migrated.halo_control, 1.0);
        assert_eq!(migrated.gain_shadows, 0.0);
        assert_eq!(migrated.gain_highlights, 0.0);
    }

    #[test]
    fn preparation_matches_commit_params_and_roi_formulas() {
        let params = ContrastTextureParams {
            halo_control: 1.0,
            gain_shadows: 1.0,
            gain_highlights: -1.0,
            ..ContrastTextureParams::default()
        };
        let prepared = prepare(params, 4096.0, 1.0, 0.5, 100, 80).unwrap();
        assert_eq!(prepared.max_detail_level, 6.0);
        assert_eq!(prepared.radius_details[0], 32.0);
        assert_eq!(prepared.radius_details[1], 16.0);
        assert!((prepared.noise_bias[0] - 0.001 / 64.0).abs() < 1e-9);
        assert!((prepared.feathering - 0.1).abs() < 1e-6);
        assert_eq!(prepared.slope_shadows, 0.5);
        assert_eq!(prepared.slope_highlights, 0.5);
        assert_eq!(prepared.midtones_polynomial, [-0.15, 0.5, 0.0]);
        assert_eq!(prepared.max_used_level, 5);
    }

    #[test]
    fn preparation_rejects_invalid_inputs_and_overflow() {
        assert_eq!(
            prepare(ContrastTextureParams::default(), f32::NAN, 1.0, 1.0, 2, 2),
            Err(PrepareError::InvalidScale)
        );
        assert_eq!(
            prepare(ContrastTextureParams::default(), 1.0, 1.0, 1.0, 0, 2),
            Err(PrepareError::InvalidDimensions)
        );
        assert_eq!(
            prepare(
                ContrastTextureParams::default(),
                1.0,
                1.0,
                1.0,
                usize::MAX,
                2,
            ),
            Err(PrepareError::ArithmeticOverflow)
        );
        assert!(ContrastTextureParams {
            noise_bias: f32::NAN,
            ..ContrastTextureParams::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn memory_plan_uses_actual_helper_peak_and_scales() {
        let plan = memory_plan(100, 80, &[0.5, 4.0, 8.0], 0, 0, large_budget()).unwrap();
        assert_eq!(plan.pixels, 8000);
        assert_eq!(plan.level_count, 3);
        assert_eq!(plan.per_level_downsampled_pixels[..3], [8000, 500, 500]);
        assert_eq!(plan.max_downsampled_pixels, 8000);
        // 4N input + 3N luminance + 2N full-resolution EIGF averages + 15D.
        assert_eq!(plan.estimated_bytes, (9 * 8000 + 15 * 8000) * 4);
        assert_eq!(plan.largest_buffer_bytes, 4 * 8000 * 4);
    }

    #[test]
    fn memory_plan_checks_total_single_buffer_and_external_usage() {
        let plan = memory_plan(100, 80, &[4.0], 1000, 200_000, large_budget()).unwrap();
        assert_eq!(plan.estimated_bytes, (9 * 8000 + 15 * 500) * 4 + 1000);
        assert_eq!(plan.largest_buffer_bytes, 200_000);

        let total_too_small = ResourceBudget {
            cpu_available_bytes: plan.estimated_bytes - 1,
            ..large_budget()
        };
        assert!(matches!(
            memory_plan(100, 80, &[4.0], 1000, 200_000, total_too_small),
            Err(MemoryPlanError::BudgetExceeded { .. })
        ));

        let single_buffer_too_small = ResourceBudget {
            cpu_single_buffer_bytes: plan.largest_buffer_bytes - 1,
            ..large_budget()
        };
        assert!(matches!(
            memory_plan(100, 80, &[4.0], 1000, 200_000, single_buffer_too_small),
            Err(MemoryPlanError::BufferExceeded { .. })
        ));
    }

    #[test]
    fn memory_plan_rejects_invalid_scale_dimensions_levels_and_overflow() {
        assert_eq!(
            memory_plan(2, 2, &[4.0], 0, 0, large_budget()),
            Err(MemoryPlanError::InvalidDimensions)
        );
        assert_eq!(
            memory_plan(10, 10, &[f32::NAN], 0, 0, large_budget()),
            Err(MemoryPlanError::InvalidSigma)
        );
        assert_eq!(
            memory_plan(
                10,
                10,
                &[1.0; MAX_ITERATIONS + 1],
                0,
                0,
                large_budget(),
            ),
            Err(MemoryPlanError::TooManyLevels)
        );
        assert_eq!(
            memory_plan(usize::MAX, 2, &[1.0], 0, 0, large_budget()),
            Err(MemoryPlanError::ArithmeticOverflow)
        );
    }
}
