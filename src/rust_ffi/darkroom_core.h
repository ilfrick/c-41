#pragma once
/*
 * C declarations for functions exported by the c41-core Rust crate
 * (crates/c41-core/src/iop/exposure.rs).
 *
 * This header is hand-maintained for now. When more IOPs migrate to Rust,
 * regenerate it with:
 *   cbindgen --config crates/c41-core/cbindgen.toml \
 *             --output src/rust_ffi/darkroom_core.h
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Color-contrast IOP -- affine transform on Lab a/b channels.
 *
 * Replaces the two OMP loops in src/iop/colorcontrast.c::process().
 * unbound != 0: no clamping; unbound == 0: a/b clamped to [-128, 128].
 */
void darkroom_colorcontrast_process(const float *in_buf,
                                    float *out_buf,
                                    size_t npixels,
                                    float a_steepness,
                                    float a_offset,
                                    float b_steepness,
                                    float b_offset,
                                    int unbound);

/*
 * Vibrance IOP -- saturation-weighted chroma boost.
 *
 * Replaces the OMP loop in src/iop/vibrance.c::process().
 * amount must be pre-scaled by 0.01 (done in C commit_params).
 */
void darkroom_vibrance_process(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               float amount);

/*
 * Levels IOP -- black/white-point + gamma correction via LUT.
 *
 * Replaces the OMP loop in src/iop/levels.c::process().
 * lut points to dt_iop_levels_data_t.lut (65536 floats).
 * level_range = d->levels[2] - d->levels[0], pre-computed by caller.
 */
void darkroom_levels_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             float level_black,
                             float level_range,
                             float inv_gamma,
                             const float *lut);

/*
 * Color-correction IOP -- luminance-dependent Lab a/b scaling with saturation.
 *
 * Replaces the OMP loop in src/iop/colorcorrection.c::process().
 * out.a = saturation * (in.a + in.L * a_scale + a_base)
 * out.b = saturation * (in.b + in.L * b_scale + b_base)
 */
void darkroom_colorcorrection_process(const float *in_buf,
                                      float *out_buf,
                                      size_t npixels,
                                      float a_scale,
                                      float a_base,
                                      float b_scale,
                                      float b_base,
                                      float saturation);

/*
 * Relight IOP -- gaussian-weighted L-channel boost in Lab colorspace.
 *
 * Replaces the OMP loop in src/iop/relight.c::process().
 * GAUSS(a=1, b, c, x) = expf(-(x-b)^2 / c^2)  [no 2x in denominator]
 */
void darkroom_relight_process(const float *in_buf,
                              float *out_buf,
                              size_t npixels,
                              float ev,
                              float center,
                              float width);

/*
 * Colorize IOP -- replace a/b with fixed Lab color, blend L from input.
 *
 * Replaces the OMP loop in src/iop/colorize.c::process().
 * Alpha is always written as 0 (matching C copy_pixel({0,a,b,0})).
 */
void darkroom_colorize_process(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               float color_l,
                               float color_a,
                               float color_b,
                               float mix);

/*
 * Velvia IOP -- film-emulation saturation boost in RGB colorspace.
 *
 * Replaces the OMP loop in src/iop/velvia.c::process().
 * strength must be pre-scaled by 0.01 (data->strength / 100.0f).
 * Handles strength <= 0 internally (copies input).
 */
void darkroom_velvia_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             float strength,
                             float bias);

/*
 * Colisa IOP -- contrast/brightness (LUT) + saturation in Lab colorspace.
 *
 * Replaces the OMP loop in src/iop/colisa.c::process().
 * ctable/ltable point to dt_iop_colisa_data_t.ctable/ltable (65536 floats each).
 * cunbounded_coeffs/lunbounded_coeffs each have 3 floats.
 */
void darkroom_colisa_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             const float *ctable,
                             const float *cunbounded_coeffs,
                             const float *ltable,
                             const float *lunbounded_coeffs,
                             float saturation);
void darkroom_colisa_build_contrast_lut(float *ctable, float contrast);
void darkroom_colisa_build_brightness_lut(float *ltable, float gamma);

/*
 * Split-toning IOP -- shadow/highlight colorization via HSL.
 *
 * Replaces the OMP loop in src/iop/splittoning.c::process().
 * compress must be pre-scaled: (data->compress / 110.0f) / 2.0f
 */
void darkroom_splittoning_process(const float *in_buf,
                                  float *out_buf,
                                  size_t npixels,
                                  float shadow_hue,
                                  float shadow_saturation,
                                  float highlight_hue,
                                  float highlight_saturation,
                                  float balance,
                                  float compress);

/*
 * Negadoctor IOP -- film negative scan inversion.
 *
 * Replaces the OMP loop in src/iop/negadoctor.c::process().
 * dmin, wb_high, offset each point to 4 floats (dt_aligned_pixel_t).
 * black, gamma, soft_clip, soft_clip_comp, exposure are scalar floats.
 */
void darkroom_negadoctor_process(const float *in_buf,
                                 float *out_buf,
                                 size_t npixels,
                                 const float *dmin,
                                 const float *wb_high,
                                 const float *offset,
                                 float black,
                                 float gamma,
                                 float soft_clip,
                                 float soft_clip_comp,
                                 float exposure);

/*
 * Channel-mixer IOP -- linear RGB and HSL channel remapping.
 *
 * Replaces the process_rgb/gray/hsl_v1/hsl_v2 loops in channelmixer.c::process().
 * hsl_matrix and rgb_matrix each point to 9 floats (3x3, row-major).
 * operation_mode: 0=RGB, 1=GRAY, 2=HSL_V1, 3=HSL_V2.
 */
void darkroom_channelmixer_process(const float *in_buf,
                                   float *out_buf,
                                   size_t npixels,
                                   const float *hsl_matrix,
                                   const float *rgb_matrix,
                                   int operation_mode);

/*
 * Lowlight IOP -- scotopic vision simulation in Lab colorspace.
 *
 * Replaces the OMP loop in src/iop/lowlight.c::process().
 * lut points to d->lut (DT_IOP_LOWLIGHT_LUT_RES = 65536 floats).
 */
void darkroom_lowlight_process(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               float blueness,
                               const float *lut);

/*
 * Tone-curve IOP -- 3-channel Lab LUT with four autoscale modes.
 *
 * Replaces the OMP loop in src/iop/tonecurve.c::process().
 * table_l/a/b each point to d->table[ch_L/a/b] (65536 floats each).
 * unbounded_coeffs_l: 3 floats (d->unbounded_coeffs_L).
 * unbounded_coeffs_ab: 12 floats (d->unbounded_coeffs_ab).
 * autoscale_ab: 0=MANUAL, 1=AUTOMATIC, 2=AUTOMATIC_XYZ, 3=AUTOMATIC_RGB.
 */
void darkroom_tonecurve_process(const float *in_buf,
                                float *out_buf,
                                size_t npixels,
                                const float *table_l,
                                const float *table_a,
                                const float *table_b,
                                const float *unbounded_coeffs_l,
                                const float *unbounded_coeffs_ab,
                                int autoscale_ab,
                                int unbound_ab,
                                int preserve_colors);

/*
 * Primaries IOP -- linear RGB color matrix adjustment.
 *
 * Replaces the OMP loop in src/iop/primaries.c::process().
 * matrix points to dt_colormatrix_t (float[4][4] = 16 floats, row-major).
 * dt_apply_transposed_color_matrix: out[r] = sum matrix[c][r] * in[c] for c=0..2
 */
void darkroom_primaries_process(const float *in_buf,
                                float *out_buf,
                                size_t npixels,
                                const float *matrix);

/*
 * Profile-gamma IOP -- logarithmic or gamma LUT tone mapping.
 *
 * Replaces the OMP loops in src/iop/profile_gamma.c::process().
 * mode: 0=LOG (all ch*npixels elements), 1=GAMMA (channels 0..2 only).
 * grey = data->grey_point / 100.0f (LOG mode only).
 * table: 65536 floats (GAMMA mode); unbounded_coeffs: 3 floats (GAMMA mode).
 */
void darkroom_profile_gamma_process(const float *in_buf,
                                    float *out_buf,
                                    size_t npixels,
                                    int mode,
                                    float grey,
                                    float dynamic_range,
                                    float shadows_range,
                                    const float *table,
                                    const float *unbounded_coeffs);
void darkroom_profile_gamma_build_lut(float *table, float gamma, float linear);

/*
 * Graduated ND filter IOP -- exponential density gradient.
 *
 * Replaces the OMP loops in src/iop/graduatednd.c::process().
 * Geometry scalars must be pre-computed by C caller (see source for formulas).
 * density > 0: divides; density < 0: multiplies (negated length).
 * color / color1 each point to 4 floats (dt_aligned_pixel_t).
 */
void darkroom_graduatednd_process(const float *in_buf,
                                  float *out_buf,
                                  int width,
                                  int height,
                                  float density,
                                  float length_base,
                                  float length_inc,
                                  float cosv_hh_inv,
                                  float filter_hardness,
                                  int iy,
                                  const float *color,
                                  const float *color1);

/*
 * Grain IOP -- photographic film grain via simplex noise on L channel.
 *
 * Replaces the OMP loop in src/iop/grain.c::process() (non-filter path only).
 * grain_lut: 128x128 floats from data->grain_lut; if NULL, built from midtones_bias.
 * strength = data->strength / 100.0f
 * zoom = (1.0 + 8*data->scale/100) / 800.0
 * wd = fminf(piece->buf_in.width, piece->buf_in.height)
 * hash = _hash_string(filename) % max(roi->width*0.3, 1)
 */
void darkroom_grain_process(const float *in_buf,
                            float *out_buf,
                            int roi_x,
                            int roi_y,
                            int width,
                            int height,
                            float strength,
                            double zoom,
                            double wd,
                            double scale,
                            int hash,
                            int filter,
                            double filtermul,
                            const float *grain_lut);

/*
 * RGB-curve IOP -- per-channel or linked LUT tone mapping.
 *
 * Replaces the OMP loop in src/iop/rgbcurve.c::process().
 * autoscale: 0 = AUTOMATIC_RGB (linked, R curve applied to all channels)
 *            1 = MANUAL_RGB (independent per-channel curves)
 * preserve_colors: 0 = NONE, non-zero = luma-norm mode (see color.rs rgb_norm).
 * table_r/g/b: 65536 floats each; unbounded_r/g/b: 3 floats each.
 * xm_r/g/b = 1.0f / unbounded_coeffs[ch][0], pre-computed by caller.
 */
void darkroom_rgbcurve_process(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               const float *table_r,
                               const float *table_g,
                               const float *table_b,
                               const float *unbounded_r,
                               const float *unbounded_g,
                               const float *unbounded_b,
                               float xm_r,
                               float xm_g,
                               float xm_b,
                               int autoscale,
                               int preserve_colors);

/*
 * Color-zones IOP -- luminance/chroma/hue equalizer in LCH space.
 *
 * Replaces process_v1/process_v3 in src/iop/colorzones.c.
 * mode: 0 = smooth/v3 (DT_IOP_COLORZONES_MODE_SMOOTH), non-zero = flat/v1.
 * channel: 0 = L, 1 = C, 2 = h (drives LUT selection index).
 * lut_l/a/b: each DT_IOP_COLORZONES_LUT_RES (65536) floats -- d->lut[0..2].
 */
void darkroom_colorzones_process(const float *in_buf,
                                 float *out_buf,
                                 size_t npixels,
                                 int mode,
                                 int channel,
                                 const float *lut_l,
                                 const float *lut_a,
                                 const float *lut_b);
void darkroom_colorzones_display(const float *in_buf,
                                 float *out_buf,
                                 size_t npixels,
                                 int channel,
                                 const float *lut);

/*
 * Vignette IOP -- radial brightness/saturation falloff with optional dithering.
 *
 * Replaces the OMP loop in src/iop/vignette.c::process().
 * All geometry scalars must be pre-computed by the C caller.
 * dither_amt: 0.0 = off, 1/256 = 8-bit, 1/65536 = 16-bit.
 * unbound: 0 = clamp output to [0,1], non-zero = no clamp.
 */
void darkroom_vignette_process(const float *in_buf,
                               float *out_buf,
                               int width,
                               int height,
                               float xscale,
                               float yscale,
                               float roi_center_x,
                               float roi_center_y,
                               float dscale,
                               float fscale,
                               float exp1,
                               float exp2,
                               float dither_amt,
                               float brightness,
                               float saturation,
                               int unbound);

/*
 * Sigmoid IOP -- RGB-ratio path: luma-based tone curve + hyperbolic gamut compression.
 *
 * Replaces process_loglogistic_rgb_ratio in src/iop/sigmoid.c.
 * white_target / black_target = module_data->white_target / black_target.
 * paper_exp / film_fog / contrast_power / skew_power = from module_data.
 */
void darkroom_sigmoid_rgb_ratio_process(const float *in_buf,
                                        float *out_buf,
                                        size_t npixels,
                                        float white_target,
                                        float black_target,
                                        float paper_exp,
                                        float film_fog,
                                        float contrast_power,
                                        float skew_power);

/*
 * Sigmoid IOP -- per-channel path: per-channel tone curve + hue preservation.
 *
 * Replaces process_loglogistic_per_channel in src/iop/sigmoid.c.
 * pipe_to_base / base_to_rendering / rendering_to_pipe: each 16 floats
 * (dt_colormatrix_t), pre-computed by C caller via _calculate_adjusted_primaries.
 */
void darkroom_sigmoid_per_channel_process(const float *in_buf,
                                          float *out_buf,
                                          size_t npixels,
                                          float white_target,
                                          float paper_exp,
                                          float film_fog,
                                          float contrast_power,
                                          float skew_power,
                                          float hue_preservation,
                                          const float *pipe_to_base,
                                          const float *base_to_rendering,
                                          const float *rendering_to_pipe);

/*
 * RGB-levels IOP -- per-channel or luma-linked black/white-point + gamma correction.
 *
 * Replaces the two DT_OMP_FOR loops in src/iop/rgblevels.c::process().
 * mode: 0 = independent channels (INDEPENDENT or preserve_colors==NONE)
 *       1 = linked via rgb_norm luma
 * preserve_colors: dt_rgb_norm mode for linked path.
 * min_levels / max_levels / inv_gamma: 3 floats each (R, G, B).
 * lut_r/g/b: 65536 floats each (d->lut[0..2]).
 */
void darkroom_rgblevels_process(const float *in_buf,
                                float *out_buf,
                                size_t npixels,
                                int mode,
                                int preserve_colors,
                                const float *min_levels,
                                const float *max_levels,
                                const float *inv_gamma,
                                const float *lut_r,
                                const float *lut_g,
                                const float *lut_b);

/*
 * Basic Adjustments IOP pixel loop.
 *
 * Replaces the DT_OMP_FOR loop in src/iop/basicadj.c::process().
 * lut_gamma and lut_contrast are 65536-entry float arrays.
 * plain_contrast and preserve_colors are mutually exclusive (C enforces this).
 */
void darkroom_basicadj_process(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               float black_point,
                               float scale,
                               int process_hlcompr,
                               float hlcomp,
                               float hlrange,
                               float lum_r,
                               float lum_g,
                               float lum_b,
                               int process_gamma,
                               float gamma,
                               const float *lut_gamma,
                               int plain_contrast,
                               int preserve_colors,
                               float contrast,
                               float middle_grey,
                               float inv_middle_grey,
                               const float *lut_contrast,
                               int process_saturation_vibrance,
                               float saturation,
                               float vibrance);

/*
 * Zonesystem IOP pixel loop.
 *
 * Replaces the DT_OMP_FOR loop in src/iop/zonesystem.c::process().
 * zonemap_offset and zonemap_scale are arrays of `size` floats.
 */
void darkroom_zonesystem_process(const float *in_buf,
                                 float *out_buf,
                                 size_t npixels,
                                 float rzscale,
                                 const float *zonemap_offset,
                                 const float *zonemap_scale,
                                 size_t size);

/*
 * Zonesystem GUI zone-map preview helpers.
 *
 * Replace the strided-copy and CLAMPS fill DT_OMP_FOR loops in
 * src/iop/zonesystem.c::process_common_cleanup(). extract_channel pulls the
 * luma (channel 0) out of an npixels*ch RGBA buffer; build_zonemap quantises a
 * blurred luma buffer (0..100) into guchar zone indices clamped to [0, size-2].
 */
void darkroom_zonesystem_extract_channel(const float *in_buf,
                                         float *out_buf,
                                         size_t npixels,
                                         size_t ch);

void darkroom_zonesystem_build_zonemap(const float *blurred,
                                       unsigned char *zonemap,
                                       size_t npixels,
                                       size_t size);

/*
 * Overlay IOP pixel loop.
 *
 * Replaces the DT_OMP_FOR(collapse(2)) loop in src/iop/overlay.c::process().
 * image is a Cairo ARGB32 buffer (byte order [B, G, R, A]) with `stride` bytes per row.
 * opacity is pre-divided by 100 (range 0..1).
 */
void darkroom_overlay_process(const float *in_buf,
                              float *out_buf,
                              size_t width,
                              size_t height,
                              const unsigned char *image,
                              size_t stride,
                              float opacity);

/*
 * Exposure IOP pixel loop.
 *
 * Replaces the inner loop in src/iop/exposure.c::process():
 *   for(size_t k = 0; k < ch * npixels; k++)
 *       out[k] = (in[k] - black) * scale;
 *
 * in_buf and out_buf must be non-overlapping arrays of length npixels*channels.
 */
void darkroom_exposure_process(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               size_t channels,
                               float black,
                               float scale);

/*
 * Low-pass IOP pixel loop (contrast + brightness LUT, saturation on a/b).
 *
 * Replaces the DT_OMP_FOR loop in src/iop/lowpass.c::process() (after the blur).
 * out_buf must already contain the gaussian/bilateral blurred Lab image.
 *
 * ctable/ltable: float[0x10000] LUTs for contrast/brightness (L in [0..100] -> new L)
 * cunbounded/lunbounded: float[3] extrapolation coeffs (dt_iop_eval_exp) for L >= 100
 * saturation: d->saturation (multiplier on a/b channels)
 * lab_min_ab/lab_max_ab: +/-128 normally, +/-FLT_MAX when unbound=1
 * Alpha is copied from in_buf (original pre-blur pixel).
 */
void darkroom_lowpass_process(const float *in_buf,
                              float *out_buf,
                              size_t npixels,
                              const float *ctable,
                              const float *cunbounded,
                              const float *ltable,
                              const float *lunbounded,
                              float saturation,
                              float lab_min_ab,
                              float lab_max_ab);
void darkroom_lowpass_build_contrast_lut(float *ctable, float contrast);
void darkroom_lowpass_build_brightness_lut(float *ltable, float gamma);

/*
 * Color Balance IOP pixel loop (LEGACY / LGG / SOP modes).
 *
 * Replaces the DT_OMP_FOR block in src/iop/colorbalance.c::process().
 *
 * mode: 0=LEGACY, 1=LIFT_GAMMA_GAIN, 2=SLOPE_OFFSET_POWER
 * param1[4]: lift (LEGACY/LGG) or lift_sop (SOP)
 * param2[4]: gamma_inv_legacy / gamma_inv_lgg (LEGACY/LGG) or gamma_sop (SOP)
 * gain[4]:   pre-computed gain vector
 * grey = d->grey / 100.0f
 * saturation = d->saturation; saturation_out = d->saturation_out
 * contrast_power[4]: { 1/d->contrast, ... } -- all four elements equal
 * (grey/saturation/saturation_out/contrast_power are ignored in LEGACY mode)
 */
void darkroom_colorbalance_process(const float *in_buf,
                                   float *out_buf,
                                   size_t npixels,
                                   int mode,
                                   const float *param1,
                                   const float *param2,
                                   const float *gain,
                                   float grey,
                                   float saturation,
                                   float saturation_out,
                                   const float *contrast_power);

/*
 * Soften IOP initial pixel loop.
 *
 * Replaces the DT_OMP_FOR loop in src/iop/soften.c::process() (before dt_box_mean).
 * Converts each pixel RGB->HSL, scales saturation and lightness, writes back RGB.
 *
 * brightness = 1.0f / exp2f(-d->brightness)
 * saturation = d->saturation / 100.0f
 * Output alpha is always 0 (matches hsl2rgb() C behaviour).
 */
void darkroom_soften_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             float brightness,
                             float saturation);

/*
 * Sharpen IOP — separable Gaussian blur + unsharp mask.
 *
 * Replaces the DT_OMP_FOR row loop in src/iop/sharpen.c::process().
 * `mat` is the Gaussian kernel built by init_gaussian_kernel(); the first
 * 2*rad + 1 taps are used. Luma (channel 0) is sharpened; chroma (1, 2) pass
 * through; alpha (3) is left untouched in the interior and copied on borders.
 * Caller guarantees rad >= 1 and width/height >= 2*rad + 1.
 */
void darkroom_sharpen_process(const float *in_buf,
                              float *out_buf,
                              const float *mat,
                              size_t width,
                              size_t height,
                              int rad,
                              float threshold,
                              float amount);

/*
 * Shadows/Highlights IOP pixel loop.
 *
 * Replaces the DT_OMP_FOR loop in src/iop/shadhi.c::process().
 * IMPORTANT: the caller must first run the gaussian/bilateral blur so that
 * out_buf already contains the blurred Lab image when this is called.
 *
 * All scalar params are pre-computed in process() from dt_iop_shadhi_data_t:
 *   shadows    = 2 * clamp(data->shadows / 100, -1, 1)
 *   highlights = 2 * clamp(data->highlights / 100, -1, 1)
 *   whitepoint = max(1 - data->whitepoint / 100, 0.01)
 *   compress   = clamp(data->compress / 100, 0, 0.99)
 *   shadows_ccorrect / highlights_ccorrect: as computed in process()
 *   low_approximation = data->low_approximation
 *   flags      = data->flags  (UNBOUND_* bitmask)
 *   unbound_mask = (algo==BILATERAL && UNBOUND_BILATERAL) || (algo==GAUSSIAN && UNBOUND_GAUSSIAN)
 */
void darkroom_shadhi_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             float shadows,
                             float highlights,
                             float whitepoint,
                             float compress,
                             float shadows_ccorrect,
                             float highlights_ccorrect,
                             float low_approximation,
                             unsigned int flags,
                             int unbound_mask);

/*
 * Highpass IOP -- invert+pack and blend, split around dt_box_mean blur.
 *
 * Pass 1: darkroom_highpass_invert
 *   Writes out[k] = 100 - clamp(in[4*k], 0, 100) into a packed 1-channel buffer.
 *   The caller then blurs out_buf with dt_box_mean (1 channel, BOX_ITERATIONS).
 *
 * Pass 2: darkroom_highpass_blend
 *   Reads packed blurred out[k] and original in[4*k], writes desaturated 4-ch pixel.
 *   Traverses in REVERSE (k = npixels-1 .. 0) so reads of the packed region are safe.
 *   Replaces both _blend() OMP calls and the final sequential loop in C.
 *   contrast_scale = ((data->contrast / 100) * 7.5) * 0.5  (pre-computed by caller).
 */
void darkroom_highpass_invert(const float *in_buf,
                              float *out_buf,
                              size_t npixels);

void darkroom_highpass_blend(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             float contrast_scale);

/*
 * Monochrome IOP -- two-pass Lab desaturation with bilateral-filtered blend.
 *
 * Pass 1 (before bilateral blur):
 *   darkroom_monochrome_colorfilter -- L_out = 100 * exp(-clamp(dist^2/sigma2, 0,1))
 *   where dist^2 = (a_in - a)^2 + (b_in - b)^2; sets a_out=b_out=0.
 *   sigma2 = 2 * (d->size * 128)^2
 *
 * Pass 2 (after bilateral blur of out):
 *   darkroom_monochrome_blend -- blends bilateral result with original L.
 *   highlights = d->highlights (0..1).
 */
void darkroom_monochrome_colorfilter(const float *in_buf,
                                     float *out_buf,
                                     size_t npixels,
                                     float a,
                                     float b,
                                     float sigma2);

void darkroom_monochrome_blend(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               float highlights);

/*
 * Global Tonemap IOP -- Reinhard / filmic (Hable) / Drago per-pixel operators.
 *
 * Each function replaces the DT_OMP_FOR loop inside process_reinhard/filmic/drago().
 * ch = piece->colors (stride, normally 4).  Only L (ch*k+0) is tone-mapped;
 * a (ch*k+1) and b (ch*k+2) are copied unchanged.  Alpha is not touched.
 *
 * Drago pre-conditions (computed in C before calling):
 *   ldc = data->drago.max_light * 0.01f / log10f(lwmax + 1)
 *   bl  = logf(max(eps, data->drago.bias)) / logf(0.5f)
 *   eps = 0.0001f  (constant)
 */
/* Globaltonemap IOP -- max luminance scan: max(initial, in[k*ch]*0.01).
 * Matches the DT_OMP_FOR(reduction(max:lwmax)) at globaltonemap.c:221. */
float darkroom_globaltonemap_luma_max(const float *in_buf,
                                       size_t npixels, size_t ch,
                                       float initial);

void darkroom_globaltonemap_reinhard(const float *in_buf,
                                     float *out_buf,
                                     size_t npixels,
                                     size_t ch);

void darkroom_globaltonemap_filmic(const float *in_buf,
                                   float *out_buf,
                                   size_t npixels,
                                   size_t ch);

void darkroom_globaltonemap_drago(const float *in_buf,
                                  float *out_buf,
                                  size_t npixels,
                                  size_t ch,
                                  float ldc,
                                  float bl,
                                  float lwmax,
                                  float eps);

/*
 * Bloom IOP -- threshold gather + screen-blend, split around dt_box_mean blur.
 *
 * Pass 1: darkroom_bloom_gather fills a packed 1-channel buffer (npixels floats)
 *   with scaled L values above threshold; zeros elsewhere.
 *   scale = 1.0f / exp2f(-1.0f * (fmin(100,strength+1) / 100.0f))
 * The caller then blurs that buffer with dt_box_mean().
 * Pass 2: darkroom_bloom_blend screen-blends the blurred L back into the 4-ch output.
 */
void darkroom_bloom_gather(const float *in_buf,
                           float *blur_buf,
                           size_t npixels,
                           float threshold,
                           float scale);

void darkroom_bloom_blend(const float *in_buf,
                          float *out_buf,
                          const float *blur_buf,
                          size_t npixels);

/*
 * Invert IOP -- non-mosaiced (4-channel RGBA) path only.
 *
 * Replaces the non-raw DT_OMP_FOR loop in src/iop/invert.c::process().
 * color points to 4 floats: { d->color[0], d->color[1], d->color[2], 1.0f }.
 * X-Trans and Bayer mosaic paths remain in C.
 * out[k*4+c] = color[c] - in[k*4+c] for c=0..3
 */
void darkroom_invert_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             const float *color);

/*
 * Dither IOP -- posterize path only.
 *
 * Replaces the DT_OMP_FOR loop in _process_posterize() in src/iop/dither.c.
 * f = levels - 1  (pre-computed by caller).
 * rf = 1.0f / f   (pre-computed by caller).
 * _quantize(x) = rf * ceilf(x*f - 0.5) -- rounds up only when frac > 0.5.
 * All 4 channels including alpha are quantized identically.
 */
void darkroom_dither_posterize(const float *in_buf,
                               float *out_buf,
                               size_t npixels,
                               float f,
                               float rf);

/*
 * AgX IOP -- full per-pixel tone mapping pipeline.
 *
 * Replaces the DT_OMP_FOR loop in src/iop/agx.c::process().
 * pipe_to_base / base_to_rendering / rendering_to_pipe / rendering_to_xyz:
 *   each 16 floats (dt_colormatrix_t = float[4][4] row-major, transposed).
 * base_working_same_profile: non-zero skips the pipe_to_base matrix.
 * params: pointer to tone_mapping_params_t (same ABI as AgxToneMappingParams).
 */
void darkroom_agx_process(const float *in_buf,
                          float *out_buf,
                          size_t npixels,
                          const float *pipe_to_base,
                          const float *base_to_rendering,
                          const float *rendering_to_pipe,
                          const float *rendering_to_xyz,
                          int base_working_same_profile,
                          const void *params);

/* Non-mosaiced white-balance multiply.
 * Replaces the DT_OMP_FOR else-branch in temperature.c::process().
 * coeffs[4] = d->coeffs -- one scalar multiplier per RGBA channel.
 */
void darkroom_temperature_process_rgb(const float *in_buf,
                                      float *out_buf,
                                      size_t npixels,
                                      const float *coeffs);

/* Alpha-composite a Cairo BGRA watermark over a float RGBA image.
 * Replaces the DT_OMP_FOR loop in watermark.c::process().
 * watermark: Cairo-rendered 8-bit BGRA (4 bytes per pixel).
 * o[rgb] = (1-alpha)*in[rgb] + opacity*(wm[rgb]/255); o[3] = in[3].
 */
void darkroom_watermark_blend(const float *in_buf,
                              float *out_buf,
                              size_t npixels,
                              const unsigned char *watermark,
                              float opacity);

/* 3D-LUT interpolation -- trilinear, tetrahedral, and pyramid variants.
 * Replace DT_OMP_FOR loops in _correct_pixel_* in lut3d.c.
 * clut: 3 x level^3 floats (RGB per grid point, no alpha padding).
 * Output alpha is always 0.
 */
void darkroom_lut3d_trilinear(const float *in_buf, float *out_buf,
                              size_t npixels,
                              const float *clut, uint16_t level);
void darkroom_lut3d_tetrahedral(const float *in_buf, float *out_buf,
                                size_t npixels,
                                const float *clut, uint16_t level);
void darkroom_lut3d_pyramid(const float *in_buf, float *out_buf,
                            size_t npixels,
                            const float *clut, uint16_t level);

/* Wavelet residue add: out[k] += add[k] for k=0..n-1.
 * Replaces the DT_OMP_FOR_SIMD residue-add loop at the end of atrous.c process().
 */
void darkroom_add_buffers(float *out_buf, const float *add_buf, size_t n);

/* Camera-RGB -> Lab via 4x4 colour matrix (cam->XYZ) + D50 XYZ->Lab.
 * Replaces the per-pixel loop in _cmatrix_fastpath_simple() in colorin.c.
 * corr:    4 white-balance correction coefficients.
 * cmatrix: 16 floats, dt_colormatrix_t row-major (float[4][4]).
 * Output alpha is always 0.
 */
void darkroom_colorin_cmatrix_fastpath_simple(const float *in_buf,
                                              float *out_buf,
                                              size_t npixels,
                                              const float *corr,
                                              const float *cmatrix);

/*
 * colorin IOP -- camera-RGB -> Lab via the input colour matrix, tone-curve
 * ("shaper") path. Replaces the per-pixel loop in _process_cmatrix_bm().
 *
 * Applies the input profile's tone curves (lut/unbounded_coeffs; per-channel,
 * skipped for linear profiles where lut[c][0] < 0), the blue mapping, then
 * cmatrix->XYZ->Lab, or when clipping != 0 nmatrix->clamp[0,1]->lmatrix->XYZ->Lab.
 * cmatrix/nmatrix/lmatrix: 16 floats each (dt_colormatrix_t, untransposed).
 * lut: 3 x 0x10000 floats; unbounded_coeffs: 3 x 3 floats. Output alpha is 0.
 */
void darkroom_colorin_cmatrix_bm(const float *in_buf,
                                 float *out_buf,
                                 size_t npixels,
                                 const float *cmatrix,
                                 const float *nmatrix,
                                 const float *lmatrix,
                                 const float *lut,
                                 const float *unbounded_coeffs,
                                 int clipping);

/*
 * ColorBalanceRGB IOP -- the four OMP loops of colorbalancergb.c (m4-137).
 *
 * darkroom_colorbalancergb_process replaces the DT_OMP_FOR pixel loop in
 * process() (:662). input_matrix_trans/output_matrix_trans are the premultiplied
 * RGB->LMS(2006 D65) / XYZ(D65)->pipeline-RGB matrices, 16 floats each, passed
 * exactly as stored (dt_colormatrix_t; the Rust side applies them transposed).
 * global/shadows/highlights/midtones/chroma/saturation_v/brilliance: dt_aligned_pixel_t.
 * scalars are the commit_params-derived dt_iop_colorbalancergb_data_t fields
 * (hue_angle already in radians; contrast already 1+p). gamut_lut: LUT_ELEM
 * floats. saturation_formula: 0 = JzAzBz, anything else = dt UCS. mask_display
 * gates the checkerboard branch; mask_type is 0..3 (MASK_SHADOWS..MASK_NONE);
 * checker_1 is the DPI-scaled checker cell size. Output alpha lane follows the C
 * (0 normally, 1.0 under mask display).
 */
void darkroom_colorbalancergb_process(const float *in_buf,
                                      float *out_buf,
                                      size_t npixels,
                                      size_t out_width,
                                      const float *input_matrix_trans,
                                      const float *output_matrix_trans,
                                      const float *global,
                                      const float *shadows,
                                      const float *highlights,
                                      const float *midtones,
                                      const float *chroma,
                                      const float *saturation_v,
                                      const float *brilliance,
                                      float chroma_global,
                                      float vibrance,
                                      float contrast,
                                      float saturation_global,
                                      float brilliance_global,
                                      float midtones_y,
                                      float hue_angle,
                                      float shadows_weight,
                                      float highlights_weight,
                                      float midtones_weight,
                                      float mask_grey_fulcrum,
                                      float white_fulcrum,
                                      float grey_fulcrum,
                                      int saturation_formula,
                                      const float *gamut_lut,
                                      int mask_display,
                                      int mask_type,
                                      size_t checker_1,
                                      const float *checker_color_1,
                                      const float *checker_color_2);

/*
 * Replaces the JzAzBz-branch gamut-LUT build in commit_params (:1197): samples
 * the STEPS^3 RGB cube through input_matrix (premultiplied RGB->XYZ D65, passed
 * exactly as stored) keeping max saturation per hue bin, then a 5-tap cyclic box
 * average. Serial == any OpenMP thread count (reduction(max:) order-independent).
 * gamut_lut receives LUT_ELEM floats.
 */
void darkroom_colorbalancergb_build_gamut_lut_jzazbz(const float *input_matrix,
                                                     float *gamut_lut);

/*
 * Fills data (packed ARGB32 bytes, graph_height*line_height*4 -- cairo's stride
 * for ARGB32 is width*4) with the vertical-alpha-fading checkerboard gradient
 * used by the GUI graph draw (:1511). No-op when checker_1 is 0.
 */
void darkroom_colorbalancergb_checkerboard_fill(unsigned char *data,
                                                size_t graph_height,
                                                size_t line_height,
                                                size_t checker_1);

/*
 * Fills the three opacity-mask curve LUTs (LUT_ELEM floats each) shown under the
 * zone sliders (:1555). Derives midtones_weight and the powered mask fulcrum
 * from shadows_weight/highlights_weight (already 2+2p) and the raw params
 * mask_grey_fulcrum, exactly as the draw callback did.
 */
void darkroom_colorbalancergb_opacity_luts(float *lut_shadows,
                                           float *lut_midtones,
                                           float *lut_highlights,
                                           float shadows_weight,
                                           float highlights_weight,
                                           float mask_grey_fulcrum_param);

/*
 * ColorReconstruction IOP -- the bespoke 4-field bilateral grid
 * (dt_iop_colorreconstruct_Lab_t = { float L, a, b, weight; }, 16 bytes, x-fastest
 * index xi + size_x*(yi + size_y*zi)).  The three exports replace this IOP's
 * former OpenMP loops (splat / blur_line / slice) and run the SAME serial scalar
 * code as the pure-Rust module.  All three refuse NULL buffers and degenerate
 * dims (real grids are always >= 5 cells per axis) instead of crashing.
 *
 * darkroom_colorreconstruct_splat: scatters every sub-threshold pixel of `in`
 *   (packed Lab, width*height*4 floats -- width/height are the INIT roi dims
 *   stored in b->width/b->height) into grid_buf (size_x*size_y*size_z cells)
 *   by nearest-integer cell with per-pixel weight from precedence:
 *   0=NONE (weight 1), 1=CHROMA (sqrt(a^2+b^2)), 2=HUE (gaussian around `hue`
 *   [radians] with variance hue_sigma_sq; unknown values behave like NONE).
 *   sigma_s/sigma_r are the grid header fields; width/height/x/y/scale are
 *   passed for header symmetry with the C struct -- only width/height are
 *   read on this path (splat never rescales or offsets).
 *
 * darkroom_colorreconstruct_blur_line: one separable [1 4 6 4 1]/16 pass along
 *   offset3 over size3 cells for each of size1 x size2 lines.  buf must hold at
 *   least the highest touched cell -- index
 *   offset1*(size1-1) + offset2*(size2-1) + offset3*(size3-1) -- plus one.
 *   Call it three times to reproduce dt_iop_colorreconstruct_bilateral_blur
 *   (x, then y, then z axes).
 *
 * darkroom_colorreconstruct_slice: trilinear read-back; rewrites only a/b where
 *   blend > 0, passing L and alpha through.  in/out hold roi_width*roi_height*4
 *   packed-Lab floats each (out may alias in).  The roi_ fields and iscale are
 *   the SLICE-time roi and piece->iscale; grid header fields as above.
 */
void darkroom_colorreconstruct_splat(float *grid_buf,
                                     size_t size_x, size_t size_y, size_t size_z,
                                     size_t width, size_t height,
                                     int x, int y,
                                     float scale, float sigma_s, float sigma_r,
                                     const float *in,
                                     float threshold,
                                     int precedence,
                                     float hue, float hue_sigma_sq);

void darkroom_colorreconstruct_blur_line(float *buf,
                                         size_t offset1, size_t offset2, size_t offset3,
                                         size_t size1, size_t size2, size_t size3);

void darkroom_colorreconstruct_slice(const float *grid_buf,
                                     size_t size_x, size_t size_y, size_t size_z,
                                     int x, int y,
                                     float scale, float sigma_s, float sigma_r,
                                     const float *in, float *out,
                                     float threshold,
                                     int roi_x, int roi_y, int roi_width, int roi_height,
                                     float roi_scale,
                                     float iscale);

/*
 * ChannelMixerRGB IOP -- per-pixel chromatic adaptation + mix + luma/chroma.
 *
 * Replaces the DT_OMP_FOR pixel loop inside _loop_switch() in channelmixerrgb.c.
 * The C caller pre-computes RGB_to_LMS and MIX_to_XYZ from kind, then transposes
 * all four matrices before calling here.  All matrix pointers are flat float[4][4]
 * (16 floats, row-stride 4, pre-transposed).
 * illuminant/saturation/lightness/grey: each 4 floats (dt_aligned_pixel_t).
 * minval: 0.0 when clip==true, -FLT_MAX otherwise.
 * p: Bradford power = powf(illuminant[2]/BRADFORD_D50[2], 0.0834).
 * gamut: chromaticity compression exponent (0 = off).
 * kind: 0=LINEAR_BRADFORD, 1=CAT16, 2=FULL_BRADFORD, 3=XYZ, 4=RGB/bypass.
 * version: 0=V1, 1=V2, 2=V3.
 */
void darkroom_channelmixerrgb_loop_switch(const float *in_buf,
                                          float *out_buf,
                                          size_t npixels,
                                          const float *rgb_to_xyz_trans,
                                          const float *rgb_to_lms_trans,
                                          const float *mix_to_xyz_trans,
                                          const float *xyz_to_rgb_trans,
                                          float minval,
                                          const float *illuminant,
                                          const float *saturation,
                                          const float *lightness,
                                          const float *grey,
                                          float p,
                                          float gamut,
                                          int clip,
                                          int apply_grey,
                                          int kind,
                                          int version);
void darkroom_channelmixerrgb_rgb_to_xyY(const float *in_buf,
                                          float *temp,
                                          size_t width,
                                          size_t height,
                                          size_t ch,
                                          const float *rgb_to_xyz,
                                          float d50_x,
                                          float d50_y);

/*
 * colorout tone-curve application -- in-place per-channel LUT + exp extrapolation.
 *
 * Replaces both DT_OMP_FOR loops in process_fastpath_apply_tonecurves() in colorout.c.
 * lut:              3 x LUT_SAMPLES (65536) floats, row-major (channel c at c*65536).
 * unbounded_coeffs: 3 x 3 floats, row-major (channel c at c*3).
 *   eval_exp(c, v) = coeff[1] * pow(v * coeff[0], coeff[2])  -- matches dt_iop_eval_exp.
 * lut_active:       3 ints; non-zero -> apply LUT+exp for that channel.
 */
void darkroom_colorout_apply_tonecurves(float *buf,
                                        size_t npixels,
                                        const float *lut,
                                        const float *unbounded_coeffs,
                                        const int *lut_active);

/* colorout Lab->XYZ->RGB using pre-transposed 3x4 colormatrix.
 * Replaces DT_OMP_FOR in _transform_cmatrix_linear() in colorout.c.
 * cmatrix: 12 floats, row-major (3 rows x 4), output of transpose_3xSSE().
 * Output alpha is always 0.
 */
void darkroom_colorout_cmatrix_linear(const float *in_buf,
                                      float *out_buf,
                                      size_t npixels,
                                      const float *cmatrix);
void darkroom_colorout_cmatrix_tonecurve(const float *in_buf,
                                         float *out_buf,
                                         size_t npixels,
                                         const float *cmatrix,
                                         const float *lut,
                                         const float *unbounded_coeffs);

/*
 * Filmic IOP pixel loop (Lab-space filmic tone-mapping).
 *
 * Replaces the DT_OMP_FOR loop in src/iop/filmic.c::process().
 * All parameters are pre-computed from dt_iop_filmic_data_t by the caller.
 * table and grad_2 are float[0x10000] LUTs from data->table / data->grad_2.
 * output_power is data->output_power (scalar, applied per channel).
 * desaturate = (data->global_saturation != 100.0f).
 * saturation  = data->global_saturation / 100.0f.
 * eps         = powf(2.0f, -16).
 * Output alpha is always 0 (matching Lab copy_pixel_nontemporal behaviour).
 */
void darkroom_filmic_process(const float *in_buf,
                             float *out_buf,
                             size_t npixels,
                             float grey_source,
                             float black_source,
                             float inv_dynamic_range,
                             float output_power,
                             float saturation,
                             float eps,
                             int desaturate,
                             int preserve_color,
                             const float *table,
                             const float *grad_2);
void darkroom_filmic_average_luts(float *table, const float *table_temp, size_t len);
void darkroom_filmic_build_grad2_lut(float *grad2, float center, float sigma);

/*
 * Basecurve IOP -- legacy (no preserve-colors) per-channel tone curve via integer-truncation LUT.
 *
 * Matches apply_legacy_curve() in src/iop/basecurve.c.
 * table:            65536 floats -- single shared LUT for all RGB channels.
 * unbounded_coeffs: 3 floats -- [c0, c1, c2] for eval_exp extrapolation (f >= 1.0).
 * mul:              pre-scalar applied to every channel value before lookup.
 */
void darkroom_basecurve_apply_legacy_curve(const float *in_buf,
                                           float *out_buf,
                                           size_t npixels,
                                           float mul,
                                           const float *table,
                                           const float *unbounded_coeffs);

/*
 * Basecurve IOP -- preserve-colors tone curve (preserve_colors != NONE path).
 *
 * Matches apply_curve() in src/iop/basecurve.c. Only the LUMINANCE norm consults
 * the working ICC profile; its fields are passed flat. When has_work_profile is
 * 0 the work-profile pointers may be NULL (LUMINANCE falls back to camera
 * primaries); the luts may be NULL when nonlinearlut is 0.
 *   matrix_in:     16 floats ([4][4], only the Y row is read)
 *   lut0/lut1/lut2: lutsize floats each (read only when nonlinearlut != 0)
 *   unbounded_in:  9 floats ([3][3])
 */
void darkroom_basecurve_apply_curve(const float *in_buf,
                                    float *out_buf,
                                    size_t npixels,
                                    float mul,
                                    int preserve_colors,
                                    const float *table,
                                    const float *unbounded_coeffs,
                                    int has_work_profile,
                                    const float *matrix_in,
                                    const float *lut0,
                                    const float *lut1,
                                    const float *lut2,
                                    const float *unbounded_in,
                                    int lutsize,
                                    int nonlinearlut);

/*
 * Basecurve IOP -- exposure-fusion feature map written into alpha channel in-place.
 *
 * Matches compute_features() in src/iop/basecurve.c.
 * Writes sat * well_exposedness into buf[k*4+3] for every pixel k.
 */
void darkroom_basecurve_compute_features(float *buf,
                                         size_t npixels);
void darkroom_basecurve_gauss_blur(const float *input, float *output,
                                    size_t wd, size_t ht);
void darkroom_basecurve_gauss_expand(const float *coarse, float *fine,
                                      size_t wd, size_t ht);
void darkroom_basecurve_weight_update(float *col0, const float *out_buf,
                                       size_t npixels);
void darkroom_basecurve_pyramid_blend(float *comb_k, const float *col_k,
                                       const float *out_buf, size_t npixels,
                                       int is_base);
void darkroom_basecurve_normalize_alpha(float *comb_k, size_t npixels);
void darkroom_basecurve_add_layers(float *comb_k, const float *out_buf,
                                    size_t npixels);
void darkroom_basecurve_copy_output(const float *comb0, const float *in_buf,
                                     float *out_buf, size_t npixels);

/* Generic sRGB/Lab batch converters (no work_profile required).
 * Matches dt_Rec709_to_XYZ_D50 + dt_XYZ_to_Lab (forward)
 * and dt_Lab_to_XYZ + dt_XYZ_to_linearRGB (inverse). */
void darkroom_color_rgb_to_lab(const float *in_buf, float *out_buf, size_t npixels);
void darkroom_color_lab_to_rgb(float *buf, size_t npixels);

/* Retouch IOP helpers -- 5 portable DT_OMP_FOR loops. */
void darkroom_retouch_copy_rows(const float *in_buf, float *out_buf,
                                 int y_to, int xoffs, int yoffs,
                                 int in_width, int out_width, int ch,
                                 size_t rowsize);
void darkroom_retouch_build_mask(const float *mask, float *mask_tmp,
                                  int roi_mask_x, int roi_mask_y,
                                  int roi_mask_w, int roi_mask_h,
                                  int roi_ms_x, int roi_ms_y,
                                  int roi_ms_w, int roi_ms_h,
                                  int x_to, int y_to, float scale);
void darkroom_retouch_copy_masked(const float *src, float *dest,
                                   int dest_roi_x, int dest_roi_y, int dest_roi_w,
                                   size_t dest_npixels,
                                   const float *mask,
                                   int mask_roi_x, int mask_roi_y,
                                   int mask_w, int mask_h, float opacity);
void darkroom_retouch_copy_mask_to_alpha(float *img,
                                          int roi_img_x, int roi_img_y, int roi_img_w,
                                          size_t img_npixels, int ch,
                                          const float *mask,
                                          int mask_roi_x, int mask_roi_y,
                                          int mask_w, int mask_h, float opacity);
void darkroom_retouch_fill(float *dest,
                            int roi_in_x, int roi_in_y, int roi_in_w,
                            size_t dest_npixels,
                            const float *mask,
                            int mask_roi_x, int mask_roi_y,
                            int mask_w, int mask_h, float opacity,
                            const float *fill_color);

/*
 * Hazeremoval IOP -- per-pixel dark channel.
 * Writes min(R,G,B) of each RGBA input pixel into a gray scalar output.
 * Matches the inner loop of _dark_channel() in src/iop/hazeremoval.c.
 */
void darkroom_hazeremoval_dark_channel(const float *in_buf,
                                       float *out_buf,
                                       size_t npixels);

/*
 * Hazeremoval IOP -- per-pixel transition map.
 * out[i] = 1 - min(min(R*a0_inv[0], G*a0_inv[1]), B*a0_inv[2]) * strength
 * Matches the inner loop of _transition_map() in src/iop/hazeremoval.c.
 * a0_inv is a 3-float array of reciprocal ambient-light values.
 */
void darkroom_hazeremoval_transition_map(const float *in_buf,
                                         float *out_buf,
                                         size_t npixels,
                                         const float *a0_inv,
                                         float strength);

/*
 * Hazeremoval IOP -- final dehaze.
 *   t = max(trans_map[i], t_min)
 *   out[4i + c] = (in[4i + c] - a0[c]) / t + a0[c]   for c in 0..4
 * Matches the final loop in `process()` (hazeremoval.c).
 * a0 is a 4-float ambient-light array (RGB + alpha pad).
 */
void darkroom_hazeremoval_dehaze(const float *in_buf,
                                 float *out_buf,
                                 const float *trans_map,
                                 size_t npixels,
                                 const float *a0,
                                 float t_min);
void darkroom_hazeremoval_ambient_light(const float *dark_channel,
                                        const float *in_rgba,
                                        size_t size,
                                        float crit_haze_level,
                                        float crit_brightness,
                                        float *a0_out,
                                        size_t *count_out);

/*
 * Censorize IOP -- pixelate (mosaic) effect.
 * Divides the RGBA image into 2*pixel_radius sized blocks; for each block,
 * averages five sample points and fills every pixel of the block with that
 * average colour. No-op if pixel_radius == 0.
 * Matches the inner pixelate loop in src/iop/censorize.c (process()).
 */
void darkroom_censorize_pixelate(const float *in_buf,
                                 float *out_buf,
                                 size_t width,
                                 size_t height,
                                 size_t pixel_radius);

/* Censorize IOP -- add per-pixel Gaussian noise.
 * norm = out[pix+1]; eps = gaussian_noise(norm, noise*norm, flip) / norm
 * out[c] = max(out[c]*eps, 0) for c in 0..3.
 * Matches make_noise() in src/iop/censorize.c:107. */
void darkroom_censorize_make_noise(float *output, float noise,
                                    size_t width, size_t height);

/* invert.c:253 -- invert X-Trans mosaic: out = clamp(film[FCxtrans]-in, 0,1) */
void darkroom_invert_xtrans(const float *in_buf, float *out_buf,
                             size_t width, size_t height,
                             const unsigned char *xtrans,
                             int roi_x, int roi_y,
                             const float *film_rgb);

/* invert.c:304 -- invert Bayer mosaic: out = clamp(film[FC]-in, 0,1) */
void darkroom_invert_bayer(const float *in_buf, float *out_buf,
                            size_t width, size_t height,
                            unsigned int filters,
                            int roi_x, int roi_y,
                            const float *film_rgb);

/* temperature.c:552 -- white-balance X-Trans: out = in * coeffs[FCNxtrans] */
void darkroom_temperature_xtrans(const float *in_buf, float *out_buf,
                                  size_t width, size_t height,
                                  const unsigned char *xtrans,
                                  const float *coeffs);

/* temperature.c:590 -- white-balance Bayer: out = in * coeffs[FC] */
void darkroom_temperature_bayer(const float *in_buf, float *out_buf,
                                 size_t width, size_t height,
                                 unsigned int filters,
                                 const float *coeffs);

/*
 * Overexposed IOP -- per-channel "any RGB" clipping preview.
 * For each pixel k:
 *   if any of R,G,B in img_tmp >= upper      -> out[k] = upper_color
 *   else if R,G,B all <= lower               -> out[k] = lower_color
 *   else                                     -> out[k] = in[k]
 * upper_color and lower_color are 4-float RGBA arrays.
 * Matches the DT_CLIPPING_PREVIEW_ANYRGB branch in src/iop/overexposed.c.
 */
void darkroom_overexposed_anyrgb(const float *in_buf,
                                 float *out_buf,
                                 const float *img_tmp,
                                 size_t npixels,
                                 float upper,
                                 float lower,
                                 const float *upper_color,
                                 const float *lower_color);

/*
 * Overexposed IOP -- work-profile-luminance clipping preview.
 * Same upper/lower decision tree as ANYRGB, but the test is run on the
 * matrix-derived Y value (with optional TRC linearisation when the
 * working profile is non-linear). Mirrors dt_ioppr_get_rgb_matrix_luminance
 * exactly. `matrix_in` is the full 4x4 colour-matrix-to-XYZ array (16
 * floats, only row 1 is read). `lut0/1/2` are the three per-channel TRC
 * LUTs (each `lutsize` floats). `unbounded_coeffs` is 3*3 = 9 floats.
 * Matches the DT_CLIPPING_PREVIEW_LUMINANCE branch in src/iop/overexposed.c.
 */
void darkroom_overexposed_luminance(const float *in_buf,
                                    float *out_buf,
                                    const float *img_tmp,
                                    size_t npixels,
                                    float upper,
                                    float lower,
                                    const float *upper_color,
                                    const float *lower_color,
                                    const float *matrix_in,
                                    const float *lut0,
                                    const float *lut1,
                                    const float *lut2,
                                    size_t lutsize,
                                    const float *unbounded_coeffs,
                                    int nonlinear_lut);

/*
 * Overexposed IOP -- gamut clipping preview (luminance + per-channel
 * saturation test). Same signature as the LUMINANCE variant.
 * Matches the DT_CLIPPING_PREVIEW_GAMUT branch in src/iop/overexposed.c.
 */
void darkroom_overexposed_gamut(const float *in_buf,
                                float *out_buf,
                                const float *img_tmp,
                                size_t npixels,
                                float upper,
                                float lower,
                                const float *upper_color,
                                const float *lower_color,
                                const float *matrix_in,
                                const float *lut0,
                                const float *lut1,
                                const float *lut2,
                                size_t lutsize,
                                const float *unbounded_coeffs,
                                int nonlinear_lut);

/*
 * Overexposed IOP -- saturation-only preview. Same signature as the
 * LUMINANCE variant. Tests the saturation+RGB clipping only when
 * luminance is inside (lower, upper); otherwise the input is passed
 * through.
 * Matches the DT_CLIPPING_PREVIEW_SATURATION branch in src/iop/overexposed.c.
 */
void darkroom_overexposed_saturation(const float *in_buf,
                                     float *out_buf,
                                     const float *img_tmp,
                                     size_t npixels,
                                     float upper,
                                     float lower,
                                     const float *upper_color,
                                     const float *lower_color,
                                     const float *matrix_in,
                                     const float *lut0,
                                     const float *lut1,
                                     const float *lut2,
                                     size_t lutsize,
                                     const float *unbounded_coeffs,
                                     int nonlinear_lut);

/*
 * Hotpixels IOP -- Bayer-sensor hot-pixel correction.
 * For each interior pixel above threshold, examines the four same-colour
 * Bayer neighbours (offsets +/-2, +/-2*width). If at least `min_neighbours`
 * of them satisfy `pixel*multiplier > neighbour`, replaces the pixel
 * with the maximum of those neighbours. When `mark_fixed` is true,
 * stamps the original value at column offsets +/-2..+/-10 (step 2) for the
 * UI debug overlay. Returns the count of pixels replaced.
 * Matches _process_bayer() in src/iop/hotpixels.c.
 */
int darkroom_hotpixels_bayer(const float *in_buf,
                             float *out_buf,
                             size_t width,
                             size_t height,
                             float threshold,
                             float multiplier,
                             int min_neighbours,
                             int mark_fixed);

/*
 * Hotpixels IOP -- multi-plane monochrome hot-pixel correction.
 * Same shape as the Bayer variant but neighbour offsets are +/-planes and
 * +/-planes*width (so we examine adjacent pixels of the same channel
 * rather than skipping a Bayer cell). When fixed, every plane of the
 * pixel is replaced with the same maximum neighbour value. Returns the
 * count of pixels replaced.
 * Matches _process_monochrome() in src/iop/hotpixels.c.
 */
int darkroom_hotpixels_monochrome(const float *in_buf,
                                  float *out_buf,
                                  size_t width,
                                  size_t height,
                                  size_t planes,
                                  float threshold,
                                  float multiplier,
                                  int min_neighbours,
                                  int mark_fixed);

/*
 * Hotpixels IOP -- X-Trans variant.
 * For each (row, col) in 2..h-2 x 2..w-2, examines the 4 pre-computed same-
 * colour neighbours in the 6x6 X-Trans CFA. `xtrans` is a flat 36-byte 6x6
 * pattern. The mark_fixed overlay stamps same-row pixels at column offsets
 * +/-2..+/-10 where the CFA colour matches the centre. Returns the count
 * of pixels replaced.
 * Matches _process_xtrans() in src/iop/hotpixels.c.
 */
int darkroom_hotpixels_xtrans(const float *in_buf,
                              float *out_buf,
                              size_t width,
                              size_t height,
                              const unsigned char *xtrans,
                              float threshold,
                              float multiplier,
                              int min_neighbours,
                              int mark_fixed);

/*
 * Defringe IOP -- per-pixel edge-chroma map + optional global average sum.
 *   edge = (in.a - out.a)^2 + (in.b - out.b)^2
 *   out.alpha = edge
 *   sum += edge   (only when use_global_average != 0)
 * `out_buf` arrives pre-filled with the gaussian-blurred copy of `in_buf`
 * (the C side calls dt_gaussian_blur_4c before invoking us). Returns the
 * chroma sum so the caller can divide by pixel count for the average.
 * Matches the DT_OMP_FOR_SIMD loop in src/iop/defringe.c (process()).
 */
float darkroom_defringe_edge_chroma_pass(const float *in_buf,
                                         float *out_buf,
                                         size_t npixels,
                                         int use_global_average);

/*
 * Colorchecker IOP -- thin-plate-spline colour correction.
 * Per pixel:
 *   res[c] = patches[N][c]                            (intercept)
 *          + polynomial_<c> dot input_Lab             (affine fall-off)
 *          + sum_p patches[p][c] * kernel(input, sources[p])  (RBF sum)
 * where kernel(x,y) = r^2 * fastlog(max(1e-8, r^2)).
 * `sources` is num_patches * 4 floats; `patches` is (num_patches + 1) * 4
 * floats (last row is the intercept). `polynomial_<c>` are 3 floats each.
 * The alpha channel of out is zeroed (matches the C aligned-pixel init).
 * Matches the process() loop in src/iop/colorchecker.c.
 */
void darkroom_colorchecker_process(const float *in_buf,
                                   float *out_buf,
                                   size_t npixels,
                                   size_t num_patches,
                                   const float *sources,
                                   const float *patches,
                                   const float *polynomial_L,
                                   const float *polynomial_a,
                                   const float *polynomial_b);

/*
 * Rasterfile IOP -- single-plane visualisation overlay.
 *   out[k] = 0.2 * clamp(sqrt(out[k]), 0, 0.5) + (mask[k] if mask else 0.0)
 * `out_buf` is read-modified. `mask` may be NULL.
 * Matches the `ch == 1` branch of process() in src/iop/rasterfile.c.
 */
void darkroom_rasterfile_visual_single(float *out_buf,
                                       const float *mask,
                                       size_t npixels);

/*
 * Rasterfile IOP -- RGBA visualisation overlay (grey-collapse).
 * For each pixel:
 *   val = 0.2 * clamp(sqrt(0.33*(R+G+B)), 0, 0.5) + mask[k]
 *   R, G, B := val      (alpha untouched)
 * Matches the `ch != 1` branch of process() in src/iop/rasterfile.c.
 */
void darkroom_rasterfile_visual_rgba(float *out_buf,
                                     const float *mask,
                                     size_t npixels);
void darkroom_rasterfile_mask_from_u8(const unsigned char *buf, float *mask,
                                      size_t npixels, unsigned int mode);
void darkroom_rasterfile_mask_from_u16be(const unsigned char *buf, float *mask,
                                         size_t npixels, unsigned int mode);
void darkroom_rasterfile_mask_from_pfm(const float *image, float *mask,
                                       size_t npixels, unsigned int mode);

/*
 * Diffuse IOP -- per-pixel mask builder.
 *   mask[k] = (in[4k] > threshold || in[4k+1] > threshold || in[4k+2] > threshold)
 * Matches build_mask() in src/iop/diffuse.c. Used by the inpaint /
 * reconstruction pre-pass.
 */
void darkroom_diffuse_build_mask(const float *in_buf,
                                 unsigned char *mask,
                                 size_t npixels,
                                 float threshold);

/*
 * Diffuse IOP -- inpaint mask initialisation with deterministic noise.
 * Masked pixels: inpainted[k+c] = abs(gaussian_noise(orig[k+c], orig[k+c]))
 *   using per-pixel xoshiro128+ state seeded from pixel position.
 * Unmasked pixels: inpainted[k..k+4] = original[k..k+4].
 * Matches inpaint_mask() in src/iop/diffuse.c:1302.
 */
void darkroom_diffuse_inpaint_mask(float *inpainted_buf,
                                   const float *original_buf,
                                   const unsigned char *mask_buf,
                                   size_t width,
                                   size_t height);

/*
 * Diffuse IOP -- anisotropic heat-transfer diffusion over one wavelet HF/LF
 * layer pair. Replaces the DT_OMP_FOR pixel loop in heat_PDE_diffusion().
 *
 * high_freq/low_freq/output: RGBA, width*height*4 floats (distinct buffers).
 * mask: width*height bytes (used iff has_mask != 0). anisotropy/abcd: 4 floats;
 * isotropy_type: 4 ints (0 isotrope, 1 isophote, 2 gradient). Writes
 * clamp(HF*strength + sum_k derivatives_k*ABCD_k / variance + LF, >=0), or HF+LF
 * where the mask is 0.
 */
void darkroom_diffuse_heat_pde(const float *high_freq,
                               const float *low_freq,
                               const unsigned char *mask,
                               int has_mask,
                               float *output,
                               size_t width,
                               size_t height,
                               const float *anisotropy,
                               const int *isotropy_type,
                               float regularization,
                               float variance_threshold,
                               float current_radius_square,
                               int mult,
                               const float *abcd,
                               float strength);

/*
 * Colortransfer IOP -- L-histogram-matching pass.
 *
 * Per pixel:
 *   src_bin    = clamp(HISTN * in_L / 100, 0, HISTN - 1)
 *   target_bin = cdf_lut[src_bin]                          (already normalised)
 *   out_L      = clamp(inverse_cdf[target_bin], 0, 100)
 *
 * Only touches the L channel; the ab clustering pass that follows in C
 * is responsible for the rest. `cdf_lut` is produced by capture_histogram()
 * (values in [0, HISTN-1]); `inverse_cdf` is produced by invert_histogram()
 * (values in [0, 100)). Both LUTs are `histn` entries long.
 *
 * Matches the first DT_OMP_FOR loop of the APPLY branch in
 * src/iop/colortransfer.c (line 327).
 */
void darkroom_colortransfer_apply_l_histogram(const float *in_buf,
                                              float *out_buf,
                                              size_t width,
                                              size_t height,
                                              size_t ch,
                                              const int *cdf_lut,
                                              const float *inverse_cdf,
                                              size_t histn);

/*
 * Colortransfer IOP -- a/b cluster-transfer pass (fuzzy weighting). Replaces the
 * second DT_OMP_FOR loop of the APPLY branch. Leaves L untouched (set by the
 * L-histogram pass); writes a/b and copies alpha. mean/var are this image's n
 * input clusters (n*2 floats each); data_mean/data_var the target clusters;
 * mapio maps each input cluster to a target (n ints). n <= MAXN (5).
 */
void darkroom_colortransfer_apply_ab(const float *in_buf,
                                     float *out_buf,
                                     size_t width,
                                     size_t height,
                                     size_t ch,
                                     int n,
                                     const float *mean,
                                     const float *var,
                                     const float *data_mean,
                                     const float *data_var,
                                     const int *mapio);

/*
 * Cacorrectrgb IOP -- per-pixel manifold normalisation.
 * For each pixel k (with confidence weight stored in the alpha channel):
 *   weighth = max(higher[k*4+3], 1e-2)
 *   weightl = max(lower[k*4+3],  1e-2)
 *   higher[k*4+guide] /= weighth ; lower[k*4+guide] /= weightl
 *   for the two non-guide channels c:
 *     higher[k*4+c] = exp2(higher[k*4+c] / weighth) * higher[k*4+guide]
 *     lower[k*4+c]  = exp2(lower[k*4+c]  / weightl) * lower[k*4+guide]
 *   if weighth < 0.05: smooth blend higher -> blurred_in by (1 - w)
 *   if weightl < 0.05: smooth blend lower  -> blurred_in by (1 - w)
 * `guide` is the guide channel index (0=R, 1=G, 2=B); values >= 3 are a
 * wiring bug and the function returns without touching the buffers.
 * Matches normalize_manifolds() in src/iop/cacorrectrgb.c.
 */
void darkroom_cacorrectrgb_normalize_manifolds(
    const float *blurred_in,
    float *blurred_manifold_lower,
    float *blurred_manifold_higher,
    size_t width,
    size_t height,
    unsigned int guide);

/* Build initial per-pixel manifolds (get_manifolds first pass). */
void darkroom_cacorrectrgb_build_manifolds(
    const float *in_buf,
    const float *blurred_in,
    float *manifold_lower,
    float *manifold_higher,
    size_t width,
    size_t height,
    unsigned int guide);

/* Refinement pass: update manifolds using first-pass estimates. */
void darkroom_cacorrectrgb_refine_manifolds(
    const float *in_buf,
    const float *blurred_in,
    const float *blurred_manifold_lower,
    const float *blurred_manifold_higher,
    float *manifold_lower,
    float *manifold_higher,
    size_t width,
    size_t height,
    unsigned int guide);

/* Pack two 4-ch manifolds into one 6-ch buffer (alpha dropped). */
void darkroom_cacorrectrgb_pack_manifolds(
    const float *blurred_manifold_lower,
    const float *blurred_manifold_higher,
    float *manifolds_out,
    size_t npixels);

/* Apply manifold-based CA correction. mode: 0=standard,1=darken,2=brighten. */
void darkroom_cacorrectrgb_apply_correction(
    const float *in_buf,
    const float *manifolds,
    size_t width,
    size_t height,
    unsigned int guide,
    unsigned int mode,
    float *out_buf);

/* Pack in/out channel pairs for the reduce_artifacts blur step. */
void darkroom_cacorrectrgb_pack_inout(
    const float *in_buf,
    const float *out_buf,
    float *inout_buf,
    size_t npixels,
    unsigned int guide);

/* Weighted blend of correction toward input when averages diverge. */
void darkroom_cacorrectrgb_blend_artifacts(
    const float *in_buf,
    const float *blurred_inout,
    float *out_buf,
    size_t npixels,
    unsigned int guide,
    float safety);

/*
 * Rawdenoise IOP -- Bayer collect: gather one Bayer channel into a
 * half-size monochrome buffer applying the sqrt variance-stabilising
 * transform. `c` selects the channel (0=R, 1=G1, 2=G2, 3=B).
 * `halfwidth` must equal (width - ((c&2)>>1) + 1) / 2 (the C formula).
 * Matches the first DT_OMP_FOR in wavelet_denoise() (rawdenoise.c:221).
 */
void darkroom_rawdenoise_bayer_collect(
    const float *in_buf, float *fimg_buf,
    size_t width, size_t height, size_t halfwidth, unsigned int c);

/*
 * Rawdenoise IOP -- Bayer scatter: distribute denoised Bayer channel back,
 * squaring to invert the sqrt transform.
 * Same halfwidth constraint as bayer_collect.
 * Matches the second DT_OMP_FOR in wavelet_denoise() (rawdenoise.c:237).
 */
void darkroom_rawdenoise_bayer_scatter(
    const float *fimg_buf, float *out_buf,
    size_t width, size_t height, size_t halfwidth, unsigned int c);

/*
 * Rawdenoise IOP -- X-Trans collect: nearest-neighbour scatter of one CFA
 * channel (c: 0=R,1=G,2=B) into a full-size buffer with vstransform.
 * `xtrans` is a flat 36-byte 6x6 CFA pattern.
 * The caller must pre-fill row 0 and row height-1 with 0.5 before calling.
 * Matches the DT_OMP_FOR(num_threads) in wavelet_denoise_xtrans() (:339).
 */
void darkroom_rawdenoise_xtrans_collect(
    const float *in_buf, float *fimg_buf,
    size_t width, size_t height,
    const unsigned char *xtrans, unsigned int c);

/*
 * Rawdenoise IOP -- X-Trans scatter: write denoised CFA channel back,
 * squaring to invert vstransform.
 * Matches the DT_OMP_FOR in wavelet_denoise_xtrans() (:454).
 */
void darkroom_rawdenoise_xtrans_scatter(
    const float *fimg_buf, float *out_buf,
    size_t width, size_t height,
    const unsigned char *xtrans, unsigned int c);

/*
 * Colormapping IOP -- find the min/max of the a and b Lab channels.
 * Returns via four out-pointers. Sentinels (FLT_MAX / -FLT_MAX) are written
 * when npixels == 0. Matches the reduction loop in kmeans() (colormapping.c:298).
 */
void darkroom_colormapping_ab_range(const float *col, size_t npixels,
                                    float *out_a_min, float *out_a_max,
                                    float *out_b_min, float *out_b_max);

/*
 * Colormapping IOP -- compute the blended L-delta for every pixel.
 * out[k*4] = clamp(0.5 * ((L*(1-eq) + source_ihist[target_hist[bin]]*eq) - L) + 50, 0, 100)
 * `target_hist` and `source_ihist` are both of length `histn`.
 * Matches the DT_OMP_FOR loop in process() (colormapping.c:492).
 */
void darkroom_colormapping_l_delta(const float *in_buf, float *out_buf,
                                   size_t npixels,
                                   const int *target_hist,
                                   const float *source_ihist,
                                   size_t histn,
                                   float equalization);

/* Colorequal IOP -- initialise per-pixel UV covariance (U*U, U*V, V*V). */
void darkroom_colorequal_init_covariance(const float *uv_buf, float *cov_buf,
                                         size_t pixels);
/* Colorequal IOP -- finalise covariance by subtracting avg(x)*avg(y). */
void darkroom_colorequal_finish_covariance(const float *uv_buf, float *cov_buf,
                                           size_t pixels);
/* Colorequal IOP -- compute guided-filter regression coefficients (a, b). */
void darkroom_colorequal_prepare_prefilter(const float *uv_buf,
                                           const float *cov_buf,
                                           float *a_buf,
                                           float *b_buf,
                                           size_t pixels,
                                           float eps);
/*
 * Colorequal IOP -- apply guided-filter regression with sigmoid blending.
 * w = get_satweight(sat[k] - sat_shift) -- linear interpolation in the
 * precomputed logistic table (length 2*satsize+1); caller passes the
 * live C static array produced by _init_satweights(contrast).
 */
void darkroom_colorequal_apply_prefilter(float *uv_buf,
                                         const float *saturation,
                                         const float *a_buf,
                                         const float *b_buf,
                                         size_t npixels,
                                         float sat_shift,
                                         const float *satweights,
                                         size_t satsize);

/* colorequal.c:698 -- build guidexcorrections correlation cross-products */
void darkroom_colorequal_init_correlations(
    const float *uv_buf, const float *corrections_buf,
    const float *b_corrections, float *corr_buf, size_t pixels);

/* colorequal.c:727 -- subtract averages from correlations */
void darkroom_colorequal_finish_correlations(
    const float *uv_buf, const float *corrections_buf,
    const float *b_corrections, float *corr_buf, size_t pixels);

/* colorequal.c:755 -- compute guided-filter regression params from correlations */
void darkroom_colorequal_compute_guided_params(
    const float *uv_buf, const float *covariance_buf, const float *correlations,
    const float *corrections_buf, const float *b_corrections,
    float *a_buf, float *b_buf, size_t pixels, float eps);

/* colorequal.c:823 -- apply guided filter with sigmoid weighting to corrections */
void darkroom_colorequal_apply_guided_filter(
    const float *uv_buf, const float *saturation, const float *gradients,
    const float *a_buf, const float *b_buf,
    float *corrections, float *b_corrections,
    size_t npixels, float sat_shift, float bright_shift,
    const float *satweights, size_t satsize);

/* colorequal.c:944 -- STEP 1: RGB -> dt UCS UV + saturation + L*.
 * input_matrix is flat 16-float (4x4) non-transposed:
 *   XYZ_D50_to_D65_CAT16 x work_profile->matrix_in */
void darkroom_colorequal_rgb_to_ucs_uv(
    const float *in_buf, float *uv_buf, float *saturation, float *lscharr,
    size_t npixels, size_t ch, const float *input_matrix);

/* colorequal.c:974 -- STEP 3: Lab-UV -> JCH -> HSB + hue/sat/brightness corrections.
 * lut_hue/saturation/brightness are each LUT_ELEM=512 floats. */
void darkroom_colorequal_compute_hsb_corrections(
    const float *uv_buf, float *lscharr, const float *saturation,
    float *pix_out_buf, float *corrections, float *b_corrections,
    size_t npixels, size_t width, size_t height,
    float white, float gradient_amp, int use_filter,
    const float *lut_hue, const float *lut_saturation, const float *lut_brightness);

/* colorequal.c:1035 -- STEP 5: apply corrections + convert HSB -> RGB.
 * gamut_lut is LUT_ELEM=512 floats; output_matrix is flat 16-float (4x4). */
void darkroom_colorequal_apply_corrections(
    float *pix_out_buf, const float *corrections, const float *b_corrections,
    size_t npixels, float white,
    const float *gamut_lut, const float *output_matrix);

/* colorequal.c:930 -- mask-display visualization overlay.
 * mode: 0=BRIGHTNESS, 1=SATURATION, 2=BRIGHTNESS_GRAD, 3=SATURATION_GRAD, 4=HUE */
void darkroom_colorequal_mask_display(
    float *pix_out_buf, const float *corrections, const float *b_corrections,
    const float *saturation, const float *lscharr,
    size_t npixels, int mode, float white,
    float sat_shift, float bright_shift,
    const float *satweights, size_t satsize);

/*
 * Toneequal IOP -- LUT-based correction apply.
 * For each pixel: correction = lut[round((clamp(log2(lum), min_ev, max_ev) - min_ev) * lut_res)]
 * All 4 channels multiplied by correction. lut_len = pixel_chan * lut_resolution + 1.
 * Matches apply_toneequalizer() DT_OMP_FOR (toneequal.c:789).
 */
void darkroom_toneequal_apply_lut(const float *in_buf,
                                   const float *luminance,
                                   float *out_buf,
                                   size_t npixels,
                                   const float *lut,
                                   size_t lut_len,
                                   float min_ev,
                                   float max_ev,
                                   float lut_resolution);

/*
 * Toneequal IOP -- build the correction LUT from Gaussian RBF.
 * lut[j] = clamp(sum_i exp(-(j/res+min_ev - centers[i])^2 / (2sigma^2)) * factors[i], 0.25, 4)
 * Matches build_correction_lut() DT_OMP_FOR (toneequal.c:1231).
 */
void darkroom_toneequal_build_lut(float *lut,
                                   const float *factors,
                                   const float *centers,
                                   size_t pixel_chan,
                                   size_t lut_resolution,
                                   float sigma,
                                   float min_ev);

/*
 * Toneequal IOP -- luminance mask display overlay.
 * intensity = sqrt(clamp((lum - 1/256) / (1 - 1/256), 0, 1))
 * All 4 out channels written to intensity; alpha overwritten from in.
 * in_height is the full input-buffer height (needed for safe bounds).
 * Matches the mask-display DT_OMP_FOR(collapse(2)) (toneequal.c:967).
 */
void darkroom_toneequal_mask_display(const float *in_buf,
                                     const float *luminance,
                                     float *out_buf,
                                     size_t out_width,
                                     size_t out_height,
                                     size_t in_width,
                                     size_t in_height,
                                     size_t offset_x,
                                     size_t offset_y);
void darkroom_toneequal_build_log_histogram(const float *luminance,
                                            size_t num_elem,
                                            int *hist,
                                            size_t temp_samples);
void darkroom_toneequal_compute_channels_factors(const float *factors,
                                                  float *out,
                                                  float sigma);
void darkroom_toneequal_build_gui_lut(float *lut,
                                      const float *factors,
                                      float sigma,
                                      float offset,
                                      float scaling);

/*
 * Useless IOP -- checkerboard dimming example module.
 * For each pixel: if ((wi/checker_scale + wj/checker_scale) & 1):
 *   out[c] = in[c] * (1 - factor); mask[k] = 1.0 (if mask non-NULL)
 * else: out[c] = in[c]   (passthrough)
 * where wi = trunc((roi_in_x + i) * scale), wj = trunc((roi_in_y + j) * scale).
 * Matches process() in src/iop/useless.c:393.
 */
void darkroom_useless_process(const float *in_buf,
                               float *out_buf,
                               float *mask_buf,
                               size_t width,
                               size_t height,
                               size_t ch,
                               int roi_in_x,
                               int roi_in_y,
                               float scale,
                               int checker_scale,
                               float factor);

/* Gamma IOP -- copy float RGBA to uint8 BGR (no sRGB gamma, just clamp/round).
 * buffsize = width*height*4 (f32 element count = u8 byte count for 4-ch RGBA).
 * Matches _copy_output() in src/iop/gamma.c:269. */
void darkroom_gamma_copy_output(const float *in_buf, unsigned char *out_buf,
                                 size_t buffsize);

/* Gamma IOP -- monochrome channel display with sRGB gamma + yellow mask overlay.
 * Uses in[j+1] (second channel) as grey value. */
void darkroom_gamma_display_monochrome(const float *in_buf, unsigned char *out_buf,
                                        size_t buffsize, float alpha);

/* Gamma IOP -- false-colour single-channel display.
 * mode 0=R, 1=G, 2=B, 3=saturation.
 * Applies sRGB gamma + yellow MASK_COLOR blend. */
void darkroom_gamma_display_false_color_simple(const float *in_buf,
                                               unsigned char *out_buf,
                                               size_t buffsize,
                                               float alpha,
                                               unsigned int mode);

/* Gamma IOP -- luminance mask overlay with configurable grey mix.
 * mix = dt_conf_get_float("darkroom/ui/develop_mask_mix").
 * interpolatef(mix, in[j+3], luma) = mix*(alpha - luma) + luma. */
void darkroom_gamma_mask_display(const float *in_buf, unsigned char *out_buf,
                                  size_t buffsize, float alpha, float mix);
void darkroom_gamma_display_a_channel(const float *in_buf, unsigned char *out_buf,
                                       size_t buffsize, float alpha);
void darkroom_gamma_display_b_channel(const float *in_buf, unsigned char *out_buf,
                                       size_t buffsize, float alpha);
void darkroom_gamma_display_lch_h(const float *in_buf, unsigned char *out_buf,
                                   size_t buffsize, float alpha);
void darkroom_gamma_display_hsl_h(const float *in_buf, unsigned char *out_buf,
                                   size_t buffsize, float alpha);
void darkroom_gamma_display_jz_hz(const float *in_buf, unsigned char *out_buf,
                                   size_t buffsize, float alpha);

/* Blurs IOP -- restore alpha channel after Gaussian blur overwrites it.
 * out[k*4+3] = in[k*4+3] for k in 0..npixels.
 * Matches src/iop/blurs.c:601. */
void darkroom_blurs_alpha_restore(const float *in_buf, float *out_buf,
                                   size_t npixels);
void darkroom_blurs_bspline_2d(const float *in_buf, float *out_buf,
                                size_t width, size_t height);

/* Denoiseprofile IOP -- 6 Anscombe/power-law VST loops. */
void darkroom_denoise_precondition(const float *in_buf, float *out_buf,
                                    size_t npixels,
                                    const float *a, const float *b);
void darkroom_denoise_backtransform(float *buf, size_t npixels,
                                     const float *a, const float *b);
void darkroom_denoise_precondition_v2(const float *in_buf, float *out_buf,
                                       size_t npixels,
                                       float a, const float *p, float b,
                                       const float *wb);
void darkroom_denoise_backtransform_v2(float *buf, size_t npixels,
                                        float a, const float *p, float b,
                                        float bias, const float *wb);
void darkroom_denoise_precondition_yuv(const float *in_buf, float *out_buf,
                                        size_t npixels,
                                        float a, const float *p, float b,
                                        const float *to_yuv);
/* Liquify IOP -- 6 DT_OMP_FOR loops (float complex stored as [re,im] f32 pairs). */
/* Clipping IOP -- 4 DT_OMP_FOR loops (keystone + affine transforms + pixel warps). */
void darkroom_clipping_distort_transform(float *points, size_t n,
    int k_apply, const float *k_space,
    float ma, float mb, float md, float me, float mg, float mh,
    float kxa, float kya, float tx, float ty,
    const float *inv_m, float k_h, float k_v, int flip,
    float enlarge_x, float enlarge_y, float cix, float ciy, float factor);
void darkroom_clipping_distort_backtransform(float *points, size_t n,
    int k_apply, const float *k_space,
    float ma, float mb, float md, float me, float mg, float mh,
    float kxa, float kya, float tx, float ty,
    const float *m, float k_h, float k_v, int flip,
    float enlarge_x, float enlarge_y, float cix, float ciy, float factor);
void darkroom_clipping_distort_mask(const float *in_buf, float *out_buf,
    float roi_out_x, float roi_out_y, float roi_out_scale,
    int roi_out_w, int roi_out_h,
    float roi_in_x, float roi_in_y, float roi_in_scale,
    int roi_in_w, int roi_in_h,
    int k_apply, const float *k_space,
    float ma, float mb, float md, float me, float mg, float mh,
    float kxa, float kya, float tx, float ty,
    const float *m, float k_h, float k_v, int flip,
    float enlarge_x, float enlarge_y, float cix, float ciy,
    unsigned int interp_type);
void darkroom_clipping_process(const float *in_buf, float *out_buf,
    float roi_out_x, float roi_out_y, float roi_out_scale,
    int roi_out_w, int roi_out_h,
    float roi_in_x, float roi_in_y, float roi_in_scale,
    int roi_in_w, int roi_in_h,
    int k_apply, const float *k_space,
    float ma, float mb, float md, float me, float mg, float mh,
    float kxa, float kya, float tx, float ty,
    const float *m, float k_h, float k_v, int flip,
    float enlarge_x, float enlarge_y, float cix, float ciy,
    int ch, unsigned int interp_type);

void darkroom_liquify_apply_stamp(float *center, size_t global_width,
                                   size_t iradius,
                                   const float *lookup_table, size_t table_size,
                                   size_t oversample, int warp_type,
                                   float strength_re, float strength_im,
                                   float abs_strength);
void darkroom_liquify_apply_map(const float *in_buf, float *out_buf,
                                 int roi_in_x, int roi_in_y,
                                 int roi_in_w, int roi_in_h,
                                 int roi_out_x, int roi_out_y,
                                 int roi_out_w, int roi_out_h,
                                 int extent_x, int extent_y,
                                 int extent_w, int extent_h,
                                 const float *map, int ch,
                                 unsigned int interp_type);
void darkroom_liquify_invert_map(const float *map, float *imap,
                                  int width, int height);
void darkroom_liquify_fill_gaps(float *imap, int width, int height);
void darkroom_liquify_bounding_box(const float *points, size_t n, float scale,
                                    float *xmin, float *xmax,
                                    float *ymin, float *ymax);
void darkroom_liquify_apply_distortion(float *points, size_t n, float scale,
                                        const float *map,
                                        int extent_x, int extent_y,
                                        int extent_w, int map_size);

void darkroom_denoise_backtransform_yuv(float *buf, size_t npixels,
                                         float a, const float *p, float b,
                                         float bias, const float *wb,
                                         const float *to_rgb);
void darkroom_blurs_init_kernel(float *buf, size_t n);
void darkroom_blurs_gauss_kernel(float *buf, size_t width, size_t height);
void darkroom_blurs_lens_kernel(float *buf, size_t width, size_t height,
                                 float n, float m, float k, float rotation);
void darkroom_blurs_motion_kernel(float *buf, size_t width,
                                   float a, float offset,
                                   const float *rot_m,
                                   float radius, float eps);
void darkroom_blurs_gui_rgba(const float *kernel, unsigned char *rgba, size_t n);
float darkroom_blurs_compute_norm(const float *buf, size_t n);
void darkroom_blurs_normalize(float *buf, size_t n, float norm);
void darkroom_blurs_pad_image(const float *in, float *padded,
                               size_t in_height, size_t in_width, size_t padded_width);
void darkroom_blurs_pad_right_edge(const float *in, float *padded,
                                    size_t in_height, size_t in_width, size_t padded_width);
void darkroom_blurs_pad_bottom_edge(const float *in, float *padded,
                                     size_t in_height, size_t in_width,
                                     size_t padded_width, size_t padded_height);
void darkroom_blurs_pad_kernel(const float *kernel, float *padded_kernel,
                                size_t kernel_width, size_t padded_width,
                                size_t offset_i, size_t offset_j);

/* Blurs IOP -- sparse spatial convolution for lens/motion blur paths.
 * Uses precomputed (offsets, values) for interior pixels and full `kernel`
 * with clamping for edge pixels. `offsets` is in f32-element units from the
 * centre pixel pointer (maps to ptrdiff_t in C). `kernel` is (2*radius+1)^2
 * floats. Output alpha is always taken from the input centre pixel.
 * Matches the DT_OMP_FOR(collapse(2)) at src/iop/blurs.c:652. */
void darkroom_blurs_sparse_convolve(const float *in_buf, float *out_buf,
                                     size_t out_width, size_t out_height,
                                     size_t in_width,  size_t in_height,
                                     int radius, int ox, int oy,
                                     const ptrdiff_t *offsets,
                                     const float     *values,
                                     size_t n_nonzero,
                                     const float *kernel);

/* Filmicrgb IOP -- build highlight reconstruction mask.
 * weight = clamp(1/(1+2^(-pix_max*normalize+feathering)), 0, 1)
 * Returns count of pixels where argument < 4 (non-negligible transition).
 * Matches reconstruct_highlights_build_mask() in filmicrgb.c:1050. */
int darkroom_filmicrgb_build_reconstruction_mask(const float *in_buf,
                                                  float *mask_buf,
                                                  size_t npixels,
                                                  float normalize,
                                                  float feathering);

/* Filmicrgb IOP -- broadcast scalar mask to all 4 output channels.
 * Matches display_mask() in filmicrgb.c:2012. */
void darkroom_filmicrgb_display_mask(const float *mask_buf, float *out_buf,
                                      size_t npixels);

/* Filmicrgb IOP -- restore pixels from ratio/norm decomposition.
 * ratios[k*4+c] = clamp(ratios[k*4+c], 0, 1) * norms[k]
 * Matches restore_ratios() in filmicrgb.c:2051. */
void darkroom_filmicrgb_restore_ratios(float *ratios_buf,
                                        const float *norms_buf,
                                        size_t npixels);

/*
 * Filmicrgb IOP -- inpaint_noise(): add statistical noise to highlights to
 * seed the wavelet reconstruction. Uses per-pixel deterministic xoshiro128+.
 * noise_distribution: 0=uniform, 1=gaussian, 2=poissonian.
 * Matches inpaint_noise() in src/iop/filmicrgb.c:1062.
 */
void darkroom_filmicrgb_inpaint_noise(const float *in_buf,
                                       const float *mask_buf,
                                       float *inpainted,
                                       float noise_level,
                                       float threshold,
                                       unsigned int noise_distribution,
                                       size_t width,
                                       size_t height);

/*
 * Filmic RGB IOP -- chroma-free "split" tone mapping (colour-science v1 and
 * v2/v3). Replace the DT_OMP_FOR loops in filmic_split_v1 / filmic_split_v2_v3.
 *
 * Only the LUMINANCE step consults the working ICC profile; its fields are
 * passed flat (NULL when has_work_profile == 0 -> camera-primary fallback; luts
 * NULL when nonlinearlut == 0). matrix_in is 16 floats ([4][4], Y row used);
 * unbounded_in is 9 floats ([3][3]); lut0/1/2 are lutsize floats each.
 * m1..m5 are the spline factor vectors (4 floats; indices 0/1/2 used);
 * type0/type1 are the toe/shoulder curve types (0=poly4,1=poly3,2=rational).
 */
void darkroom_filmicrgb_split_v1(const float *in_buf, float *out_buf, size_t npixels,
                                 int has_work_profile, const float *matrix_in,
                                 const float *lut0, const float *lut1, const float *lut2,
                                 const float *unbounded_in, int lutsize, int nonlinearlut,
                                 float grey_source, float black_source, float dynamic_range,
                                 float sigma_toe, float sigma_shoulder, float saturation,
                                 float output_power,
                                 const float *m1, const float *m2, const float *m3,
                                 const float *m4, const float *m5,
                                 float latitude_min, float latitude_max, int type0, int type1);

void darkroom_filmicrgb_split_v2_v3(const float *in_buf, float *out_buf, size_t npixels,
                                    int has_work_profile, const float *matrix_in,
                                    const float *lut0, const float *lut1, const float *lut2,
                                    const float *unbounded_in, int lutsize, int nonlinearlut,
                                    float grey_source, float black_source, float dynamic_range,
                                    float sigma_toe, float sigma_shoulder, float saturation,
                                    float output_power,
                                    const float *m1, const float *m2, const float *m3,
                                    const float *m4, const float *m5,
                                    float latitude_min, float latitude_max, int type0, int type1);

/*
 * Filmic RGB IOP -- chroma-preserving (ratio-preserving) tone mapping, v1 and
 * v2/v3. Replace the DT_OMP_FOR loops in filmic_chroma_v1 / filmic_chroma_v2_v3.
 * `variant` is the RGB-norm method (dt_iop_filmicrgb_methods_type_t); for v2_v3,
 * `colorscience_version == 2` (DT_FILMIC_COLORSCIENCE_V3) enables re-normalisation.
 * Work-profile / spline parameters follow the same flat convention as the split
 * functions above.
 */
void darkroom_filmicrgb_chroma_v1(const float *in_buf, float *out_buf, size_t npixels, int variant,
                                  int has_work_profile, const float *matrix_in,
                                  const float *lut0, const float *lut1, const float *lut2,
                                  const float *unbounded_in, int lutsize, int nonlinearlut,
                                  float grey, float black, float dynamic_range,
                                  float sigma_toe, float sigma_shoulder, float saturation,
                                  float output_power,
                                  const float *m1, const float *m2, const float *m3,
                                  const float *m4, const float *m5,
                                  float latitude_min, float latitude_max, int type0, int type1);

void darkroom_filmicrgb_chroma_v2_v3(const float *in_buf, float *out_buf, size_t npixels, int variant,
                                     int colorscience_version,
                                     int has_work_profile, const float *matrix_in,
                                     const float *lut0, const float *lut1, const float *lut2,
                                     const float *unbounded_in, int lutsize, int nonlinearlut,
                                     float grey, float black, float dynamic_range,
                                     float sigma_toe, float sigma_shoulder, float saturation,
                                     float output_power,
                                     const float *m1, const float *m2, const float *m3,
                                     const float *m4, const float *m5,
                                     float latitude_min, float latitude_max, int type0, int type1);

/*
 * Filmic RGB IOP -- colour-science v4/v5 gamut-mapped tone mapping. Replace the
 * DT_OMP_FOR loops in filmic_chroma_v4 / filmic_split_v4 / filmic_v5.
 *
 * Work-profile / spline parameters follow the same flat convention as the split
 * functions above. The six colour matrices (each 16 floats, [4][4]) are prepared
 * on the C side by filmic_v4_prepare_matrices and passed flat:
 *   input_matrix_trans         pipeline RGB -> CIE 2006 LMS (transposed)
 *   output_matrix              CIE 2006 LMS -> pipeline RGB
 *   output_matrix_trans        CIE 2006 LMS -> pipeline RGB (transposed)
 *   export_input_matrix_trans  output RGB -> CIE 2006 LMS (transposed)
 *   export_output_matrix       CIE 2006 LMS -> output RGB
 *   export_output_matrix_trans CIE 2006 LMS -> output RGB (transposed)
 * use_output_profile selects the export-profile gamut path. norm_min/norm_max are
 * exp_tonemapping_v2(0/1). display_black/display_white are the display bounds.
 */
void darkroom_filmicrgb_chroma_v4(const float *in_buf, float *out_buf, size_t npixels, int variant,
                                  int has_work_profile, const float *matrix_in,
                                  const float *lut0, const float *lut1, const float *lut2,
                                  const float *unbounded_in, int lutsize, int nonlinearlut,
                                  float grey, float black, float dynamic_range,
                                  float output_power, float saturation,
                                  const float *m1, const float *m2, const float *m3,
                                  const float *m4, const float *m5,
                                  float latitude_min, float latitude_max, int type0, int type1,
                                  const float *input_matrix_trans, const float *output_matrix,
                                  const float *output_matrix_trans, const float *export_input_matrix_trans,
                                  const float *export_output_matrix, const float *export_output_matrix_trans,
                                  int use_output_profile, float norm_min, float norm_max,
                                  float display_black, float display_white);

void darkroom_filmicrgb_split_v4(const float *in_buf, float *out_buf, size_t npixels,
                                 float grey, float black, float dynamic_range,
                                 float output_power, float saturation,
                                 const float *m1, const float *m2, const float *m3,
                                 const float *m4, const float *m5,
                                 float latitude_min, float latitude_max, int type0, int type1,
                                 const float *input_matrix_trans, const float *output_matrix,
                                 const float *output_matrix_trans, const float *export_input_matrix_trans,
                                 const float *export_output_matrix, const float *export_output_matrix_trans,
                                 int use_output_profile, float display_black, float display_white);

void darkroom_filmicrgb_v5(const float *in_buf, float *out_buf, size_t npixels,
                           int has_work_profile, const float *matrix_in,
                           const float *lut0, const float *lut1, const float *lut2,
                           const float *unbounded_in, int lutsize, int nonlinearlut,
                           float grey, float black, float dynamic_range,
                           float output_power, float saturation,
                           const float *m1, const float *m2, const float *m3,
                           const float *m4, const float *m5,
                           float latitude_min, float latitude_max, int type0, int type1,
                           const float *input_matrix_trans, const float *output_matrix,
                           const float *output_matrix_trans, const float *export_input_matrix_trans,
                           const float *export_output_matrix, const float *export_output_matrix_trans,
                           int use_output_profile, float norm_min, float norm_max,
                           float display_black, float display_white);

/*
 * Filmic RGB IOP -- highlights-reconstruction helpers.
 * init_reconstruct: multiplied-alpha blend of non/partially-clipped pixels
 *   (reconstructed = max(in * (1 - mask), 0)); mask is npixels floats.
 * compute_ratios: per-pixel norm + per-channel ratios; `variant` is the RGB-norm
 *   method; work-profile fields follow the same flat convention as the split
 *   functions (only the LUMINANCE variant consults the profile). norms is
 *   npixels floats; ratios is npixels*4 floats.
 */
void darkroom_filmicrgb_init_reconstruct(const float *in_buf, const float *mask_buf,
                                         float *reconstructed, size_t npixels);

void darkroom_filmicrgb_compute_ratios(const float *in_buf, float *norms_buf, float *ratios_buf,
                                       size_t npixels, int variant,
                                       int has_work_profile, const float *matrix_in,
                                       const float *lut0, const float *lut1, const float *lut2,
                                       const float *unbounded_in, int lutsize, int nonlinearlut);

/*
 * Filmic RGB IOP -- à-trous wavelet highlight reconstruction.
 * wavelet_hf: high-frequency scale, HF = detail - LF over npixels*4 floats.
 * wavelets_reconstruct_{rgb,ratios}: accumulate the reconstructed clipped
 *   highlights into `reconstructed` (+=). hf/lf/texture/reconstructed are
 *   npixels*4 floats; mask is npixels floats. s is the current scale, scales
 *   the total; the residual term is only added at the last scale (s==scales-1).
 */
void darkroom_filmicrgb_wavelet_hf(const float *detail, const float *lf, float *hf, size_t npixels);

void darkroom_filmicrgb_wavelets_reconstruct_rgb(const float *hf, const float *lf, const float *texture,
                                                 const float *mask, float *reconstructed, size_t npixels,
                                                 float gamma, float gamma_comp, float beta, float beta_comp,
                                                 float delta, size_t s, size_t scales);

void darkroom_filmicrgb_wavelets_reconstruct_ratios(const float *hf, const float *lf, const float *texture,
                                                    const float *mask, float *reconstructed, size_t npixels,
                                                    float gamma, float gamma_comp, float beta, float beta_comp,
                                                    float delta, size_t s, size_t scales);

/*
 * Colorharmonizer IOP -- fused single-pass (smoothing <= 0).
 * matrix_in/out_transposed are flat 16-float (4x4) working-space matrices.
 * nodes/node_saturation are num_nodes floats; both may be NULL if num_nodes=0.
 */
void darkroom_colorharmonizer_fused(
    const float *in_buf, float *out_buf, size_t npixels, size_t ch,
    const float *matrix_in_transposed, const float *matrix_out_transposed,
    float l_white,
    const float *nodes, int num_nodes,
    float pull_width, float pull_strength, float cutoff,
    const float *node_saturation);

/* Colorharmonizer -- pass 1 of the smoothing path: RGB -> JCH cache + corrections. */
void darkroom_colorharmonizer_cache_pass(
    const float *in_buf, float *jch_cache, float *corrections,
    size_t npixels, size_t ch,
    const float *matrix_in_transposed, float l_white,
    const float *nodes, int num_nodes, float pull_width,
    const float *node_saturation);

/* Colorharmonizer -- pass 2 of the smoothing path: apply corrections. */
void darkroom_colorharmonizer_apply_pass(
    const float *in_buf, float *out_buf,
    const float *jch_cache, const float *corrections,
    size_t npixels, size_t ch,
    const float *matrix_out_transposed,
    float l_white, float cutoff, float pull_strength);

/* Cacorrect IOP -- copy non-green Bayer channel to half-res buffer.
 * oldraw[row*h_width + col/2] = in[row*full_width + col]
 * Matches DT_OMP_FOR at src/iop/cacorrect.c:327. */
void darkroom_cacorrect_save_oldraw(const float *in_buf, float *oldraw_buf,
                                     size_t full_width, size_t height,
                                     size_t h_width, unsigned int filters);

/* Cacorrect IOP -- compute per-pixel R/B correction factors.
 * nongreen[(row/2)*h_width + col/2] = clamp(oldraw/in, 0.5, 2.0)
 * Matches DT_OMP_FOR at src/iop/cacorrect.c:1125. */
void darkroom_cacorrect_compute_factors(const float *in_buf,
                                         const float *oldraw_buf,
                                         float *red_buf, float *blue_buf,
                                         size_t full_width, size_t height,
                                         size_t h_width, unsigned int filters);

/* Cacorrect IOP -- apply blurred correction factors to the output buffer.
 * out[row*w + col] *= nongreen[row/2*h_width + col/2]  for interior pixels.
 * Matches DT_OMP_FOR at src/iop/cacorrect.c:1172. */
void darkroom_cacorrect_apply_factors(float *out_buf,
                                       const float *red_buf, const float *blue_buf,
                                       size_t full_width, size_t height,
                                       size_t h_width, unsigned int filters);

/* Cacorrect IOP -- write corrected buffer to roi_out with scale factor.
 * output[ox] = corrected[irow*in_width + icol] * scaler  (bounds-guarded).
 * Matches DT_OMP_FOR(collapse(2)) at src/iop/cacorrect.c:1190. */
void darkroom_cacorrect_writeout(const float *corrected, float *output,
                                  size_t out_width, size_t out_height,
                                  size_t in_width,  size_t in_height,
                                  int roi_out_x, int roi_out_y,
                                  float scaler);

/* Geometry helpers -- shared across crop, flip, borders, enlargecanvas.
 * All operate on a flat [x0,y0,x1,y1,...] coordinate buffer. */

/* Add (dx,dy) to every coordinate pair. Matches distort_transform in
 * crop.c, borders.c, enlargecanvas.c. */
void darkroom_geom_shift_coords(float *pts, size_t points_count,
                                 float dx, float dy);

/* Subtract (dx,dy) from every coordinate pair (backtransform). */
void darkroom_geom_unshift_coords(float *pts, size_t points_count,
                                   float dx, float dy);

/* Apply dt_image_orientation_t flip/transpose. Matches distort_transform
 * in flip.c. orientation bits: FLIP_Y=1, FLIP_X=2, SWAP_XY=4. */
void darkroom_geom_flip_coords(float *pts, size_t points_count,
                                unsigned int orientation,
                                float img_width, float img_height);

/* Inverse of flip_coords (backtransform). Matches distort_backtransform
 * in flip.c. */
void darkroom_geom_unflip_coords(float *pts, size_t points_count,
                                  unsigned int orientation,
                                  float img_width, float img_height);

/* Row-by-row memcpy blit with border offset. Matches distort_mask
 * DT_OMP_FOR in borders.c:420 and enlargecanvas.c:324. */
void darkroom_geom_blit_rows(const float *in_buf, float *out_buf,
                              size_t in_width, size_t in_height,
                              size_t out_width,
                              size_t border_x, size_t border_y);

/* Apply 2x2 rotation matrix + translation to every coordinate pair.
 * pi = (x - rx*scale, y - ry*scale); o = M * pi.
 * m is [m00,m01,m10,m11] (row-major).
 * Matches distort_transform in src/iop/rotatepixels.c:138. */
/* 2D separable pixel interpolation -- Rust port of dt_interpolation_compute_pixel4c().
 * interp_type: 0=bilinear, 1=bicubic, 2=lanczos2, 3=lanczos3.
 * linestride: floats per image row (= width * 4 for RGBA). */
void darkroom_interpolate_pixel4c(const float *in_buf, float *out,
                                   float x, float y,
                                   int img_width, int img_height,
                                   int linestride, unsigned int interp_type);

/* Full rotatepixels process() loop: back-transforms each output coordinate,
 * samples the input with the given interpolator, zeroes out-of-bounds pixels.
 * m: 4-float 2x2 rotation matrix (d->m).
 * interp_type: same encoding as darkroom_interpolate_pixel4c. */
/* Homographic (perspective correction) resampling.
 * ihomograph: 9-float row-major 3x3 inverse homography matrix.
 * cx/cy: clipping offsets (roi_out->scale * fullwidth * data->cl, etc.).
 * Matches ashift.c distort_mask (1-ch, CLIP) and process (4-ch) loops. */
void darkroom_ashift_transform_coords(float *pts, size_t n,
                                       const float *homograph, float cx, float cy);
void darkroom_ashift_backtransform_coords(float *pts, size_t n,
                                           const float *ihomograph, float cx, float cy);
void darkroom_ashift_rgb_to_gray(const float *in_buf, double *out_buf, size_t npixels);
void darkroom_ashift_sobel_1d(const double *in_buf, double *out_buf,
                               int width, int height, int direction);
void darkroom_ashift_sobel_border(double *buf, int width, int height);
void darkroom_ashift_gradient_magnitude(const double *gx, const double *gy,
                                         double *out, size_t n);
void darkroom_ashift_gamma_correct(const float *in_buf, float *out_buf, size_t npixels);

void darkroom_ashift_distort_mask(const float *in_buf, float *out_buf,
                                   int out_width, int out_height,
                                   float roi_out_x, float roi_out_y, float roi_out_scale,
                                   int in_width, int in_height,
                                   float roi_in_x, float roi_in_y, float roi_in_scale,
                                   float cx, float cy,
                                   const float *ihomograph, unsigned int interp_type);
void darkroom_ashift_process(const float *in_buf, float *out_buf,
                              int out_width, int out_height,
                              float roi_out_x, float roi_out_y, float roi_out_scale,
                              int in_width, int in_height,
                              float roi_in_x, float roi_in_y, float roi_in_scale,
                              float cx, float cy,
                              const float *ihomograph, unsigned int interp_type);

void darkroom_scalepixels_process(const float *in_buf, float *out_buf,
                                   int out_width, int out_height,
                                   int in_width, int in_height,
                                   float x_scale, float y_scale,
                                   unsigned int interp_type);
void darkroom_scalepixels_distort_mask(const float *in_buf, float *out_buf,
                                        int out_width, int out_height,
                                        int in_width, int in_height,
                                        float x_scale, float y_scale,
                                        unsigned int interp_type);

void darkroom_rotatepixels_process(const float *in_buf, float *out_buf,
                                    int out_width, int out_height,
                                    float roi_out_x, float roi_out_y,
                                    int in_width, int in_height,
                                    float roi_in_x, float roi_in_y,
                                    float scale,
                                    const float *m, float rx, float ry,
                                    unsigned int interp_type);

void darkroom_geom_rotate_coords(float *pts, size_t points_count,
                                  const float *m,
                                  float rx, float ry, float scale);

/* Inverse rotation: o = M^T * x + (rx*scale, ry*scale).
 * Matches distort_backtransform in src/iop/rotatepixels.c:162. */
void darkroom_geom_unrotate_coords(float *pts, size_t points_count,
                                    const float *m,
                                    float rx, float ry, float scale);

/*
 * CLAHE (Contrast-Limited Adaptive Histogram Equalisation).
 * Two-pass algorithm: builds a per-pixel luminance map = (max(RGB)+min(RGB))/2,
 * then for each row maintains a sliding (2*rad+1)^2 histogram of luminance
 * around the centre pixel, clips it at `slope*n/BINS` with redistribution
 * to convergence, looks up the equalised CDF value, and applies it as the
 * new HSL.L component (round-tripping through HSL to preserve hue+saturation).
 * Matches process() in src/iop/clahe.c. `width`/`height` are the image
 * dimensions and the in/out buffers are tightly packed RGBA float arrays.
 */
void darkroom_clahe_process(const float *in_buf,
                            float *out_buf,
                            size_t width,
                            size_t height,
                            int rad,
                            float slope);

/*
 * Rawprepare IOP -- uint16 Bayer/X-Trans mosaic linearisation.
 *   out[j*w + i] = (in[(j+csy)*in_w + (i+csx)] - sub[id]) / div[id]
 * where `id = ((j+y0)&1)<<1 | ((i+x0)&1)`. `sub`/`div` are 4-float arrays.
 * Matches the TYPE_UINT16 branch of process() in src/iop/rawprepare.c.
 */
void darkroom_rawprepare_mosaic_u16(const unsigned short *in_buf,
                                    float *out_buf,
                                    size_t out_width,
                                    size_t out_height,
                                    size_t in_width,
                                    int csx,
                                    int csy,
                                    int x0,
                                    int y0,
                                    const float *sub,
                                    const float *div_);

/*
 * Rawprepare IOP -- float Bayer/X-Trans mosaic linearisation.
 * Same as the uint16 variant but reads f32. Matches the TYPE_FLOAT branch.
 */
void darkroom_rawprepare_mosaic_f32(const float *in_buf,
                                    float *out_buf,
                                    size_t out_width,
                                    size_t out_height,
                                    size_t in_width,
                                    int csx,
                                    int csy,
                                    int x0,
                                    int y0,
                                    const float *sub,
                                    const float *div_);

/*
 * Rawprepare IOP -- pre-downsampled RGBA buffer: per-channel black/scale.
 *   out[k*ch + c] = (in[k_in*ch + c] - sub[c]) / div[c]
 * Matches the no-mosaic else-branch of process() in src/iop/rawprepare.c.
 */
void darkroom_rawprepare_rgba(const float *in_buf,
                              float *out_buf,
                              size_t out_width,
                              size_t out_height,
                              size_t in_width,
                              int csx,
                              int csy,
                              const float *sub,
                              const float *div_,
                              size_t ch);
void darkroom_rawprepare_distort_transform(float *points, size_t points_count,
                                           float dx, float dy);
void darkroom_rawprepare_distort_backtransform(float *points, size_t points_count,
                                               float dx, float dy);
void darkroom_rawprepare_apply_gainmaps(float *out,
                                        int out_width, int out_height,
                                        int roi_x, int roi_y,
                                        int csx, int csy,
                                        int top, int left,
                                        float im_to_rel_x, float im_to_rel_y,
                                        float rel_to_map_x, float rel_to_map_y,
                                        float map_origin_h, float map_origin_v,
                                        unsigned int map_w, unsigned int map_h,
                                        const float *const *maps);

/*
 * Highlights IOP -- sRAW (RGB) clipping-mask builder.
 *   refs[c] = max(0.5, 0.95 * clips[c])
 *   tmp[k]  = max_over_c((in[4k+c] - refs[c]) / refs[c]),  floored at 0.
 * `clips` is 4 floats; only the first 3 (RGB) are read.
 * Matches the `filters == 0` branch of _provide_raster_mask() in
 * src/iop/highlights.c.
 */
void darkroom_highlights_mask_sraw(const float *in_buf,
                                   float *tmp_buf,
                                   size_t width,
                                   size_t height,
                                   const float *clips);

/*
 * Highlights IOP -- Bayer / X-Trans mosaic clipping-mask builder.
 * For each pixel:
 *   c = fcol(row + irow_offset, col + icol_offset, filters, xtrans)
 *   tmp[k] = max(0, (in[k] - refs[c]) / refs[c])
 * `xtrans` is a flat 36-byte buffer (6x6 pattern); read only when filters==9.
 * Matches the `filters != 0` branch of _provide_raster_mask() in
 * src/iop/highlights.c.
 */
void darkroom_highlights_mask_mosaic(const float *in_buf,
                                     float *tmp_buf,
                                     size_t width,
                                     size_t height,
                                     unsigned int filters,
                                     const unsigned char *xtrans,
                                     const float *clips,
                                     int irow_offset,
                                     int icol_offset);

/*
 * Highlights IOP -- CLIP mode, sRAW path.
 * out[k] = fminf(clip, in[k]) for every float in the buffer.
 * NaN propagation matches the C fminf semantics exactly.
 * Matches the `ch == 4` branch of process_clip() in src/iop/highlights.c.
 */
void darkroom_highlights_clip_sraw(const float *in_buf,
                                   float *out_buf,
                                   size_t nfloats,
                                   float clip);

/*
 * Highlights IOP -- visualise mode, sRAW path.
 * For every pixel k and c in 0..3:
 *   out[k+c] = (in[k+c] < clips[c]) ? 0.2 * in[k+c] : 1.0
 *   out[k+3] = 0.0
 * Matches the `filters == 0` branch of process_visualize() in
 * src/iop/highlights.c.
 */
void darkroom_highlights_visualize_sraw(const float *in_buf,
                                        float *out_buf,
                                        size_t npixels,
                                        const float *clips);

/*
 * Highlights IOP -- visualise mode, mosaic path.
 * For every output (row, col):
 *   irow = row + irow_offset
 *   icol = col + icol_offset
 *   if in-bounds: c = fcol(irow, icol, filters, xtrans);
 *                 out = in < clips[c] ? 0.2*in : 1.0
 *   else:        out = 0.0
 * Matches the `filters != 0` branch of process_visualize() in
 * src/iop/highlights.c.
 */
void darkroom_highlights_visualize_mosaic(const float *in_buf,
                                          float *out_buf,
                                          size_t out_width,
                                          size_t out_height,
                                          size_t in_width,
                                          size_t in_height,
                                          unsigned int filters,
                                          const unsigned char *xtrans,
                                          const float *clips,
                                          int irow_offset,
                                          int icol_offset);

/*
 * Highlights IOP -- LCH reconstruction (src/iop/hlreconstruct/lch.c).
 * Bayer: in/out are single-channel width*height planes (both indexed with
 *   roi_out->width, matching the C). filters is the Bayer mask.
 * X-Trans: out is width_out*height_out; in rows are strided by width_in
 *   (roi_in->width >= width_out); xtrans is the 6x6 CFA byte table.
 */
void darkroom_highlights_lch_bayer(const float *in_buf, float *out_buf,
                                   size_t width, size_t height,
                                   unsigned int filters, float clip);

void darkroom_highlights_lch_xtrans(const float *in_buf, float *out_buf,
                                    size_t width_out, size_t height_out,
                                    size_t width_in,
                                    const unsigned char *xtrans, float clip);

/*
 * Highlights IOP -- opposed reconstruction (src/iop/hlreconstruct/opposed.c).
 * The mask buffer is the 6*msize char buffer (channels 0..2 = clipped
 * superpixels, 3..5 = dilated); msize = round4(mwidth)*round4(mheight).
 * mask_* return non-zero when any superpixel clipped. sums/cnts are
 * caller-zeroed 4-float accumulators; clips 3 floats; chrominance 3 floats;
 * correction 4 floats; xtrans 36 bytes. The sraw input/output buffers are
 * RGBA (4 floats/pixel); the raw ones single-channel. In output_raw, tmpout
 * may be NULL (reconstruction recomputed on the fly).
 */
int darkroom_highlights_opposed_mask_sraw(const float *in_buf, unsigned char *mask_buf,
                                          size_t width, size_t height,
                                          size_t mwidth, size_t mheight, size_t msize,
                                          const float *clips);

void darkroom_highlights_opposed_dilate_sraw(unsigned char *mask_buf,
                                             size_t mwidth, size_t mheight, size_t msize);

void darkroom_highlights_opposed_chroma_sraw(const float *in_buf, const unsigned char *mask_buf,
                                             size_t width, size_t height,
                                             size_t mwidth, size_t mheight, size_t msize,
                                             const float *clips, float *sums, float *cnts);

void darkroom_highlights_opposed_output_sraw(const float *in_buf, float *out_buf, size_t npixels,
                                             const float *clips, const float *chrominance);

int darkroom_highlights_opposed_mask_raw(const float *in_buf, unsigned char *mask_buf,
                                         size_t width, size_t mwidth, size_t mheight, size_t msize,
                                         unsigned int filters, const unsigned char *xtrans,
                                         const float *clips);

void darkroom_highlights_opposed_dilate_raw(unsigned char *mask_buf,
                                            size_t mwidth, size_t mheight, size_t msize);

void darkroom_highlights_opposed_chroma_raw(const float *in_buf, const unsigned char *mask_buf,
                                            size_t width, size_t height,
                                            size_t mwidth, size_t mheight, size_t msize,
                                            unsigned int filters, const unsigned char *xtrans,
                                            const float *clips, const float *correction,
                                            float *sums, float *cnts);

void darkroom_highlights_opposed_tmpout_raw(const float *in_buf, float *tmpout,
                                            size_t width, size_t height,
                                            unsigned int filters, const unsigned char *xtrans,
                                            const float *clips, const float *chrominance,
                                            const float *correction);

void darkroom_highlights_opposed_output_raw(const float *in_buf, const float *tmpout, float *out_buf,
                                            size_t out_width, size_t out_height,
                                            int out_x, int out_y,
                                            size_t in_width, size_t in_height,
                                            unsigned int filters, const unsigned char *xtrans,
                                            const float *clips, const float *chrominance,
                                            const float *correction);

/*
 * Segmentation morphology (src/iop/hlreconstruct/segmentation.c).
 * Dilate/erode the border-inset interior of a width*height uint32 bitmap
 * with the progressive-radius ring tests (dilate radius 1..8, erode 1..5).
 * Caller guarantees border >= radius.
 */
void darkroom_segmentation_dilate(const uint32_t *img, uint32_t *out,
                                  size_t width, size_t height, int border, int radius);

void darkroom_segmentation_erode(const uint32_t *img, uint32_t *out,
                                 size_t width, size_t height, int border, int radius);

/*
 * Highlights IOP -- laplacian reconstruction (src/iop/hlreconstruct/laplacian.c).
 * interpolate_and_mask: bilinear CFA demosaic into an RGBA plane (channel 3 =
 *   Euclidean norm) + RGBA clipping mask (channel 3 = any-clipped flag).
 * remosaic_and_replace: back to the CFA, alpha-blended by the mask.
 * guide_laplacians / heat_pde_diffusion: one wavelet scale of the guided
 *   linear-fit / anisotropic-PDE reconstruction. `scale` is the
 *   wavelets_scale_t bitmask; rows are processed dwt-interleaved.
 * RGBA buffers hold width*height*4 floats; raw planes width*height.
 */
void darkroom_highlights_interpolate_and_mask(const float *input, float *interpolated,
                                              float *clipping_mask,
                                              const float *clips, const float *wb,
                                              unsigned int filters, size_t width, size_t height);

void darkroom_highlights_remosaic_and_replace(const float *input, const float *interpolated,
                                              const float *clipping_mask, float *output,
                                              const float *wb,
                                              unsigned int filters, size_t width, size_t height);

void darkroom_highlights_guide_laplacians(const float *high_freq, const float *low_freq,
                                          const float *clipping_mask, float *output,
                                          size_t width, size_t height,
                                          int mult, float noise_level, int salt,
                                          unsigned int scale, float radius_sq);

void darkroom_highlights_heat_pde_diffusion(const float *high_freq, const float *low_freq,
                                            const float *clipping_mask, float *output,
                                            size_t width, size_t height,
                                            int mult, unsigned int scale,
                                            float first_order_factor);

/*
 * Highlights IOP -- segmentation-based reconstruction
 * (src/iop/hlreconstruct/segbased.c). The dt_iop_segmentation_t struct and
 * the flood-fill stay in C; struct fields are passed individually.
 * All planes (luminance/distance/gradient/tmp/colour/refavg/seg data) are
 * pwidth*pheight elements unless noted; xtrans is the 36-byte CFA table;
 * clips/correction/cube_coeffs are 4-float vectors; bounding boxes are
 * caller-clamped half-open [xmin,xmax) x [ymin,ymax) ranges.
 */
void darkroom_segbased_initial_gradients(const float *luminance, const float *distance,
                                         float *gradient, size_t pwidth, size_t pheight);

float darkroom_segbased_maxdistance(const float *distance, const uint32_t *seg_data,
                                    size_t seg_width, size_t seg_height,
                                    int xmin, int xmax, int ymin, int ymax, uint32_t id);

void darkroom_segbased_distance_ring(float *gradient, const float *distance,
                                     const uint32_t *seg_data,
                                     size_t seg_width, size_t seg_height,
                                     int xmin, int xmax, int ymin, int ymax,
                                     float attenuate, float dist, uint32_t id);

/* tmp is a dense (ymax-ymin)*(xmax-xmin) box for dt_box_mean */
void darkroom_segbased_box_in(const float *gradient, float *tmp, size_t seg_width,
                              int xmin, int xmax, int ymin, int ymax);

void darkroom_segbased_box_out(float *gradient, const float *tmp, const uint32_t *seg_data,
                               size_t seg_width, int xmin, int xmax, int ymin, int ymax,
                               uint32_t id);

void darkroom_segbased_apply_strength(float *gradient, const uint32_t *seg_data,
                                      size_t seg_width, int xmin, int xmax, int ymin, int ymax,
                                      uint32_t id, float strength);

void darkroom_masks_extend_border(float *mask, size_t width, size_t height, int border);

/* planes/refavgs: 3 colour-plane pointers; seg_datas: 4 (r,g,b,all).
 * Returns anyclipped; *has_allclipped set non-zero when a superpixel clips
 * in all three planes. tmpout is the roi_in->width*height raw plane. */
int32_t darkroom_segbased_populate_planes(const float *tmpout, size_t width, size_t height,
                                          unsigned int filters, const unsigned char *xtrans,
                                          const float *correction, const float *cube_coeffs,
                                          int xshifter,
                                          float *const *planes, float *const *refavgs,
                                          uint32_t *const *seg_datas,
                                          size_t pwidth, size_t pheight,
                                          int32_t *has_allclipped);

/* val1/val2 hold seg_nrs[c] floats per colour */
void darkroom_segbased_candidates_apply(const float *input, float *tmpout,
                                        size_t width, size_t height,
                                        unsigned int filters, const unsigned char *xtrans,
                                        const float *clips, const float *correction,
                                        float *const *planes,
                                        const uint32_t *const *seg_datas,
                                        const float *const *seg_val1s,
                                        const float *const *seg_val2s,
                                        const int32_t *seg_nrs,
                                        size_t pwidth, size_t pheight, int seg_border);

void darkroom_segbased_prepare_lumdist(const float *plane0, const float *plane1,
                                       const float *plane2, const float *icoeffs,
                                       float *tmp, float *distance,
                                       const uint32_t *segall_data,
                                       size_t pwidth, size_t pheight, int border);

void darkroom_segbased_apply_recovery(const float *input, float *tmpout,
                                      size_t width, size_t height,
                                      unsigned int filters, const unsigned char *xtrans,
                                      const float *clips,
                                      const float *distance, const float *gradient,
                                      size_t pwidth, size_t pheight,
                                      float strength, float dshift);

/*
 * Highlights IOP -- colour inpainting (src/iop/hlreconstruct/inpaint.c,
 * a1ex / magic lantern). Each call runs two directional passes for every
 * row (passes 0:+x, 1:-x) or column (passes 2:+y, 3:-y); pass 3 averages
 * the four accumulated estimates. The Bayer variant indexes input and
 * output with the same width (as the C did with roi_out->width); the
 * X-Trans variant keeps the C's roi_in-width row bases. clips has 4
 * floats; xtrans is the 36-byte CFA table.
 */
void darkroom_highlights_inpaint_xtrans_rows(const float *in_buf, float *out_buf,
                                             size_t in_width, size_t in_height,
                                             size_t out_width, size_t out_height,
                                             const float *clips, const unsigned char *xtrans);

void darkroom_highlights_inpaint_xtrans_cols(const float *in_buf, float *out_buf,
                                             size_t in_width, size_t in_height,
                                             size_t out_width, size_t out_height,
                                             const float *clips, const unsigned char *xtrans);

void darkroom_highlights_inpaint_bayer_rows(const float *in_buf, float *out_buf,
                                            size_t width, size_t height,
                                            const float *clips, unsigned int filters);

void darkroom_highlights_inpaint_bayer_cols(const float *in_buf, float *out_buf,
                                            size_t width, size_t height,
                                            const float *clips, unsigned int filters);

void darkroom_segbased_final_output(float *output, const float *tmpout,
                                    const float *luminance, const float *gradient,
                                    size_t out_width, size_t out_height, int out_x, int out_y,
                                    size_t in_width, size_t in_height,
                                    unsigned int filters, const unsigned char *xtrans,
                                    const uint32_t *const *seg_datas,
                                    const float *const *seg_val1s, const int32_t *seg_nrs,
                                    const uint32_t *segall_data, int32_t segall_nr,
                                    size_t pwidth, size_t pheight, int seg_border,
                                    int do_masking, int vmode, float strength);

/*
 * Raw-overexposure indicator (src/iop/rawoverexposed.c). fill_coords
 * writes one row's pre-distortion pixel coordinates (2*width floats);
 * mark_row paints output pixels whose back-transformed raw photosite
 * value reaches thresholds[cfa colour] -- mode 0 marks with the CFA
 * colour, 1 with solid_color (4 floats), 2 zeroes the clipped channel.
 * The dt_dev_distort_backtransform_plus call between them stays in C.
 */
void darkroom_rawoverexposed_fill_coords(float *buf, int row, size_t width,
                                         int x, int y, float scale);

void darkroom_rawoverexposed_mark_row(float *out_row, size_t width, size_t ch,
                                      const float *coords,
                                      const uint16_t *raw_buf,
                                      size_t raw_width, size_t raw_height,
                                      unsigned int filters, const unsigned char *xtrans,
                                      const uint32_t *thresholds,
                                      int mode, const float *solid_color);

/*
 * Demosaic basics (src/iop/demosaicing/basics.c): green pre-median
 * filter, RGB colour smoothing (median of R-G / B-G differences,
 * in-place on a 4-ch buffer), and the local-average / full-average
 * Bayer green equilibrations. Single-channel buffers are width*height
 * floats; color_smoothing's buffer is width*height*4.
 */
void darkroom_demosaic_pre_median(float *out, const float *in_buf,
                                  size_t width, size_t height,
                                  uint32_t filters, int num_passes, float threshold);

void darkroom_demosaic_color_smoothing(float *out, size_t width, size_t height,
                                       int num_passes);

void darkroom_demosaic_green_eq_lavg(float *out, const float *in_buf,
                                     size_t width, size_t height,
                                     uint32_t filters, float thr);

void darkroom_demosaic_green_eq_favg(float *out, const float *in_buf,
                                     size_t width, size_t height,
                                     uint32_t filters);

/* Passthrough "demosaics" (src/iop/demosaicing/passthrough.c): monochrome
 * replicates the raw channel into RGB (alpha 0); color places each
 * photosite in its CFA channel. out is width*height*4 floats. */
void darkroom_demosaic_passthrough_monochrome(float *out, const float *in_buf,
                                              size_t width, size_t height);

void darkroom_demosaic_passthrough_color(float *out, const float *in_buf,
                                         size_t width, size_t height,
                                         uint32_t filters, const unsigned char *xtrans);

/* Dual demosaic (src/iop/demosaicing/dual.c): copy the detail mask into
 * the alpha channel, or lerp the high-frequency RGBA image towards the
 * VNG one by the mask (alpha zeroed). msize = width*height pixels. */
void darkroom_demosaic_dual_mask_to_alpha(float *high_data, const float *mask,
                                          size_t msize);

void darkroom_demosaic_dual_blend(float *high_data, const float *vng_image,
                                  const float *mask, size_t msize);

/* 3x3 box-average fallback demosaic (src/iop/demosaicing/rcd.c
 * demosaic_box3). out is width*height*4 floats. */
void darkroom_demosaic_box3(float *out, const float *in_buf,
                            size_t width, size_t height,
                            uint32_t filters, const unsigned char *xtrans);

/* PPG demosaic sweeps (src/iop/demosaicing/ppg.c). green: direction-
 * selected green interpolation from the (possibly median-filtered)
 * `input` plane; `in_orig` is the unfiltered raw the C cursor switches
 * to after the margin+3 ring skip. redblue: in-place R/B interpolation
 * on the RGBA buffer. margin >= 0; PPG passes a huge sentinel (ring
 * skip disabled), RCD/LMMSE pass their border widths. */
void darkroom_demosaic_ppg_green(float *out, const float *input,
                                 const float *in_orig,
                                 size_t width, size_t height,
                                 uint32_t filters, int margin);

void darkroom_demosaic_ppg_redblue(float *out, size_t width, size_t height,
                                   uint32_t filters, int margin);

/* VNG linear interpolation sweeps (src/iop/demosaicing/vng.c,
 * _vng_lininterpolate). border: 1-pixel-frame colour averaging into the
 * RGBA buffer. lookup: threshold-gradient interior interpolation driven by
 * the C-built table (flat int[16][16][32]); `border` gates the ring skip
 * (VNG passes 1000000 for Bayer, pad_tile for X-Trans). */
void darkroom_demosaic_vng_border(float *out, const float *in,
                                  size_t width, size_t height,
                                  uint32_t filters, const unsigned char *xtrans);

void darkroom_demosaic_vng_lookup(float *out, const float *in,
                                  size_t width, size_t height,
                                  uint32_t filters, int border,
                                  const int *lookup);

/* VNG output finishing pass (vng.c:265). mix_greens averages the two
 * separated green channels into channel 1 and zeroes channel 3; then all
 * four channels of every pixel are clipped to >= 0. npixels == width*height. */
void darkroom_demosaic_vng_finish(float *out, size_t npixels, int mix_greens);

/* VNG gradient interpolation, one image row (vng.c:201). Walks the C-built
 * per-pixel `code_row` stream (selected by col%pcol), accumulates 8
 * directional gradients, thresholds, averages qualifying neighbours, and
 * writes the refined RGBA pixel into `brow2` (the ring-buffer row). Reads
 * `out` read-only; `code_row` is code[row%prow] (pcol stream pointers). */
void darkroom_demosaic_vng_gradient_row(const float *out, float *brow2,
                                        size_t width, size_t height, int row,
                                        uint32_t filters4, const unsigned char *xtrans,
                                        int colors, const int *const *code_row, int pcol);

/* Capture-sharpen radius-selected Gaussian convolutions (capture.c). Where
 * blend[i] > 0: mul scales out[i] *= conv(in); div sets
 * out[i] = luminance[i] / max(conv(in), 0.001). `kernels` is the
 * 256*32-float gauss_coeffs buffer; `table[i]` selects the per-pixel kernel;
 * idx_small = _sigma_to_index(CAPTURE_SMALL) picks the 2- vs 4-pixel radius. */
void darkroom_capture_blur_mul(const float *in, float *out, const float *blend,
                               const float *kernels, const unsigned char *table,
                               int w1, int height, unsigned char idx_small);

void darkroom_capture_blur_div(const float *in, float *out, const float *luminance,
                               const float *blend, const float *kernels,
                               const unsigned char *table,
                               int w1, int height, unsigned char idx_small);

/* Capture-sharpen blend-mask prep/modify (capture.c). prepare_blend: write
 * BT.709 luminance into Yold and zero the (caller-prefilled) mask over
 * clipped/dark interior pixels' 21-px diamond + the 2-px border. cfa is
 * single-channel for Bayer/X-Trans, RGBA for mono; whites[color] are clip
 * points. modify_blend: scale blend by a sigmoid of the local coefficient of
 * variation of Yold and copy Yold into luminance. */
void darkroom_capture_prepare_blend(const float *cfa, const float *rgb,
                                    uint32_t filters, const unsigned char *xtrans,
                                    float *mask, float *Yold, const float *whites,
                                    int w1, int height);

void darkroom_capture_modify_blend(float *blend, const float *Yold, float *luminance,
                                   float dthresh, int width, int height);

/* Capture-sharpen output stage (capture.c, _capture_sharpen). blend_combine:
 * per-pixel sigmoid blend of the unblurred (tmp2) and blurred (blendmask)
 * masks, CLIP-ed back into blendmask. show_variance_mask / show_sigma_mask:
 * debug views writing the mask / normalised kernel index into out's alpha.
 * apply_sharpen: where blendmask>0, scale all RGBA channels by
 * interpolatef(CLIP(blendmask), tmp1, luminance) / max(luminance, 0.001). */
void darkroom_capture_blend_combine(float *blendmask, const float *tmp2, size_t pixels);
void darkroom_capture_show_variance_mask(float *out, const float *blendmask, size_t pixels);
void darkroom_capture_show_sigma_mask(float *out, const unsigned char *gauss_idx, size_t pixels);
void darkroom_capture_apply_sharpen(float *out, const float *tmp1, const float *luminance,
                                    const float *blendmask, size_t pixels);

/* Capture-sharpen per-pixel Gaussian kernel-index map (_cs_precalc_gauss_idx,
 * capture.c:125): fill `table` (width*height bytes) with a radial sigma index
 * that grows with distance from the optical centre and tapers at image edges.
 * rwidth/rheight/mdim/dx/dy describe the centre & scale; cboost is derived. */
void darkroom_capture_precalc_gauss_idx(unsigned char *table, int width, int height,
                                        int dx, int dy, int rwidth, int rheight,
                                        float mdim, float isigma, float boost, float centre);

/* Capture-sharpen auto-radius green/green-ratio scans (capture.c). Return the
 * raw maxRatio (largest non-clipped green/green ratio); the C wraps it as
 * sqrtf(1/logf(maxRatio)). bayer covers Bayer (fc0/fc1 = FC(0,0)/FC(1,0)) and
 * mono (fc0=fc1=0); xtrans scans from the caller's precomputed start offset. */
float darkroom_capture_radius_bayer(const float *in, int width, int height,
                                    uint32_t fc0, uint32_t fc1);
float darkroom_capture_radius_xtrans(const float *in, int width, int height,
                                     int startx, int starty);

/* Capture auto-radius centre-region extraction (_calc_auto_radius, capture.c):
 * copy the centre-60% CFA into single-channel `input`, white-balancing by
 * coeff[colour]. wbon X-Trans uses coeff[FCNxtrans], Bayer coeff[FC]; !wbon is
 * the mono path reading channel 0 of a 4-channel `in`. coeff is 4 floats. */
void darkroom_capture_extract_centre_wb(float *input, const float *in,
                                        int owidth, int oheight, int iwidth, int iheight,
                                        int dx, int dy, uint32_t filters,
                                        const unsigned char *xtrans, const float *coeff, int wbon);

/* Frank Markesteijn's X-Trans demosaic (xtrans.c, xtrans_markesteijn_interpolate).
 * Whole-function port: tiled 3-pass interpolation writing the RGBA `out` from
 * the single-channel CFA `in`. passes is 1 or 3. The trailing
 * _vng_lininterpolate edge pass stays in C. */
void darkroom_xtrans_markesteijn(float *out, const float *in, int width, int height,
                                 const unsigned char *xtrans, int passes, uint32_t filters);

/* Frank Markesteijn's frequency-domain-chroma (FDC) X-Trans demosaic
 * (xtrans.c, xtrans_fdc_interpolate). Single-pass tiled interpolation with
 * frequency-domain chroma. hybrid0/hybrid1 are the C hybrid_fdc[2] (computed
 * C-side from ISO vs the fdc_xover_iso config). Writes the RGBA `out`. */
void darkroom_xtrans_fdc(float *out, const float *in, int width, int height,
                         const unsigned char *xtrans, float hybrid0, float hybrid1);

/*
 * ICC engine (c41-core::icc) -- pure-Rust replacement for the LCMS
 * cmsCreateTransform/cmsDoTransform pixel paths in colorin/colorout.
 * One handle = one (src, dst, intent) transform assembled from raw ICC profile
 * bytes; built once in commit_params, applied per band row, freed in cleanup --
 * mirroring the cmsHTRANSFORM lifetime it sits beside. Colour lanes carry raw
 * Lab (L in [0,100], a/b in [-128,127]) or raw D50-referenced XYZ floats --
 * the same domain as LCMS's TYPE_LabA_FLT / TYPE_XYZA_FLT / TYPE_RGBA_FLT
 * float formats. Handles may be used concurrently from multiple band threads;
 * apply supports in==out (each pixel's triplet is read before its write).
 * darkroom_icc_transform_new returns NULL when either profile fails to parse,
 * the pair is not 3-channel RGB/XYZ/Lab, or intent > 3 -- callers fall back to
 * cmsCreateTransform exactly as before for anything the engine refuses.
 */
void *darkroom_icc_transform_new(const unsigned char *src, size_t src_len,
                                 const unsigned char *dst, size_t dst_len,
                                 unsigned int intent);
void darkroom_icc_transform_free(void *t);
void darkroom_icc_transform_apply_rgba(const void *t, const float *in_buf,
                                       float *out_buf, size_t npixels);

/*
 * Drawn-mask shape rendering (c41-core::masks) -- pure kernels extracted from
 * the OMP loops of src/develop/masks/<shape>.c <shape>_get_mask / _get_mask_roi.
 * The pixelpipe distort callbacks stay C-side; only the coordinate/value
 * arithmetic crossed the boundary. All buffers are plain float arrays in the
 * shapes' own layouts; every export validates dimensions and refuses (no-op)
 * instead of panicking across the FFI.
 */

/* circle.c _circle_get_mask: fill `points` (2*w*h floats) with the pipe-area
 * coordinate grid (pos_x + j, pos_y + i), then evaluate the feathered-circle
 * value at each back-transformed point into `buffer` (n floats). */
void darkroom_masks_circle_coord_grid(float *points, size_t w, size_t h,
                                      float pos_x, float pos_y);
void darkroom_masks_circle_fill(float *buffer, const float *points, size_t n,
                                float center_x, float center_y,
                                float total2, float border2);

/* circle.c _circle_get_mask_roi: write the outer-circle outline (`circpts`
 * points, a multiple of 8 -- see dt_masks_roundup(MIN(360, DT_2PI_F*total2),
 * 8)) around (center_x, center_y) with radius `total`; then populate the
 * bbw*bbh bbox grid in module coordinates ((grid*i + px) computed in integer
 * arithmetic before the float conversion); then evaluate mask values into the
 * even lanes of `points` in place. `iscale` is 1/roi->scale. */
void darkroom_masks_circle_outline(float *circ, size_t circpts,
                                   float center_x, float center_y, float total);
void darkroom_masks_circle_grid(float *points, size_t bbw, size_t bbh,
                                int bbxm, int bbym, int px, int py,
                                float iscale, int grid);
void darkroom_masks_circle_values(float *points, size_t npoints,
                                  float center_x, float center_y,
                                  float total2, float border2);

/* circle.c _circle_get_mask_roi final interpolation: splat the bbw*bbh sampled
 * values (even lanes of `points`) over rows [start_j,end_j) x cols
 * [start_i,end_i) of the w-wide ROI `buffer` by bilinear weighting within each
 * grid*grid cell. start_i/start_j are bbxm*grid / bbym*grid as in the C loop;
 * end_i/end_j must satisfy MIN(w, bbXM*grid) / MIN(h, bbYM*grid) so the
 * neighbour lookups stay inside the bbox exactly as they do in C. */
void darkroom_masks_circle_interp(float *buffer, size_t w, size_t height,
                                  const float *points, size_t bbw, size_t bbh,
                                  int start_i, int end_i,
                                  int start_j, int end_j, int grid);

/* ellipse.c _ellipse_get_mask: coord-grid fill identical to circle's,
 * then _fill_mask (with out_scale=0) evaluates the projected-ellipse
 * feathered value at each back-transformed point into `bufptr` (n floats).
 * `a,b` are the inner semi-axes, `ta,tb` the outer (with border), `alpha`
 * the rotation in radians. */
void darkroom_masks_ellipse_coord_grid(float *points, size_t w, size_t h,
                                       float pos_x, float pos_y);
void darkroom_masks_ellipse_fill(float *bufptr, const float *points, size_t n,
                                 float center_x, float center_y,
                                 float a, float b, float ta, float tb,
                                 float alpha);

/* ellipse.c _ellipse_get_mask_roi: parametric outline (no 8-fold symmetry —
 * the pixelpipe can shear the ellipse) into `ell` (ellpts x,y pairs, using
 * the pre-computed cosa/sina and outer radii ta,tb); grid-points fill (same
 * integer-arithmetic indexing as circle); in-place _fill_mask (out_scale=1)
 * writing mask values to even lanes; then bilinear splat into `buffer`. */
void darkroom_masks_ellipse_outline(float *ell, size_t ellpts,
                                    float center_x, float center_y,
                                    float ta, float tb, float cosa, float sina);
void darkroom_masks_ellipse_grid(float *points, size_t bbw, size_t bbh,
                                 int bbxm, int bbym, int px, int py,
                                 float iscale, int grid);
void darkroom_masks_ellipse_values(float *points, size_t npoints,
                                   float center_x, float center_y,
                                   float a, float b, float ta, float tb,
                                   float alpha);
void darkroom_masks_ellipse_interp(float *buffer, size_t w, size_t height,
                                   const float *points, size_t bbw, size_t bbh,
                                   int start_i, int end_i,
                                   int start_j, int end_j, int grid);

/* gradient.c _gradient_get_mask / _gradient_get_mask_roi:
 * grid-points fill (same fill_grid_points with bbxm=0,bbym=0; iscale=1.0 for
 * whole-pipe, roi->scale inverse for ROI); LUT build (sigmoidal via erff or
 * linear); in-place mask-value evaluation at back-transformed points (rotation
 * + quadratic distance + LUT lookup with ±4·compression clamping); then
 * bilinear splat into the output buffer. */
void darkroom_masks_gradient_grid(float *points, size_t bbw, size_t bbh,
                                  int bbxm, int bbym, int px, int py,
                                  float iscale, int grid);
void darkroom_masks_gradient_lut(float *lut, size_t lutsize, int lutmax,
                                 float hwscale, float normf, float compression,
                                 int state);
void darkroom_masks_gradient_values(float *points, size_t count,
                                    const float *lut, int lutmax,
                                    float cosv, float sinv,
                                    float xoffset, float yoffset,
                                    float hwscale, float ihwscale,
                                    float curvature, float compression);
void darkroom_masks_gradient_interp(float *buffer, size_t w, size_t height,
                                    const float *points, size_t bbw, size_t bbh,
                                    int start_i, int end_i,
                                    int start_j, int end_j, int grid);

/* brush.c _brush_get_mask: per-segment falloff for each brush stroke segment
 * from points[start_idx..end_idx]. p0/p1 are float→int truncated (truncation
 * toward zero) from the points/border arrays; payload[i*2]=hardness,
 * payload[i*2+1]=density. posx/posy offset the buffer origin in image coords;
 * bw is the buffer stride. Only the falloff fill is ported — _brush_bounding_box
 * and _brush_get_pts_border stay in C. */
void darkroom_masks_brush_falloff(float *buffer, int bw, int bh,
                                  const float *points, const float *border,
                                  const float *payload,
                                  int start_idx, int end_idx,
                                  int posx, int posy);

/* brush.c _brush_get_mask_roi: same per-segment falloff but respecting the
 * ROI buffer bounds [0,bw) x [0,bh). Segments entirely outside the ROI are
 * skipped (matching the C skip check on integer-truncated p0/p1). */
void darkroom_masks_brush_falloff_roi(float *buffer, int bw, int bh,
                                      const float *points, const float *border,
                                      const float *payload,
                                      int start_idx, int end_idx);

/* path.c _path_falloff: whole-pipe per-segment falloff for a path stroke.
 * p0/p1 are integer segment endpoints (float→int truncated by the C caller).
 * posx/posy offset the buffer origin in image coordinates; bw is the buffer
 * stride. No hardness/density — opacity is always 1.0 - i/l. Replaces the
 * _path_falloff call inside the falloff loop in _path_get_mask (path.c:3375).
 * The DT_INVALID_COORDINATE skip/dedup logic stays in C. */
void darkroom_masks_path_falloff(float *buffer, int bw, int bh,
                                 int p0x, int p0y, int p1x, int p1y,
                                 int posx, int posy);

/* path.c _path_falloff_roi: ROI-bounded per-segment falloff. segments is an
 * int array of [p0x, p0y, p1x, p1y] tuples; nsegments = dindex/4.
 * Replaces the DT_OMP_FOR loop at path.c:3918 that calls _path_falloff_roi
 * per segment. Coordinates are already int-truncated by the C caller. */
void darkroom_masks_path_falloff_roi(float *buffer, int bw, int bh,
                                     const int *segments, int nsegments);

/* path.c whole-pipe even-odd fill (path.c:3327). Toggles state on v == 1.0f,
 * writes 1.0 inside path. Replaces the DT_OMP_FOR() fill loop. */
void darkroom_masks_path_fill_plain(float *buffer, int wb, int hb);

/* path.c ROI even-odd fill (path.c:3835). Toggles state on v > 0.5f (not ==1.0f),
 * writes 1.0 inside path. Bounded to [xxmin..xxmax] x [yymin..yymax], stride
 * `width`. Replaces the DT_OMP_FOR(num_threads(...)) fill loop. */
void darkroom_masks_path_fill_plain_roi(float *buffer, int width,
                                        int xxmin, int xxmax,
                                        int yymin, int yymax);

/* object.c _mask_iou: intersection-over-union reduction of two float masks.
 * Counts pixels above `threshold` in each mask, returns inter/uni (or 0.0
 * if union is empty). Replaces the DT_OMP_FOR(reduction(+:inter,uni)) loop
 * at object.c:528. */
float darkroom_masks_object_mask_iou(const float *a, const float *b,
                                     size_t n, float threshold);

/* object.c peak-point exclusion: zero out pixels within a circular region of
 * radius sqrt(min_sep_sq) around (px,py) in the distance-transform buffer.
 * The bounding box [x0,x1]×[y0,y1] is pre-clamped to [0,w-1]×[0,h-1] by the
 * C caller. Replaces the DT_OMP_FOR(collapse(2)) loop at object.c:573 inside
 * _find_peak_point. Called once per exclude peak. */
void darkroom_masks_object_zero_peaks(float *dist, int w, int bh,
                                      int x0, int x1, int y0, int y1,
                                      float px, float py, float min_sep_sq);

/* group.c _combine_masks_* / inline copy: element-wise blend of dest and newmask.
 * op codes: 0=union, 1=intersect, 2=difference, 3=sum, 4=exclusion, 5=copy.
 * `inverted` selects 1-newmask vs newmask for the mask value. Replaces all six
 * DT_OMP_FOR_SIMD loops in _group_get_mask_roi (group.c:492–706). */
void darkroom_masks_group_combine(float *dest, const float *newmask,
                                  size_t npixels, float opacity,
                                  int inverted, int op);

void darkroom_masks_detail_scharr_luminance(const float *src, float *tmp,
                                            int width, int height, const float *wb);

void darkroom_masks_detail_scharr_gradient(const float *tmp, float *mask,
                                           int width, int height);

void darkroom_masks_detail_blend(const float *src, float *out,
                                 size_t msize, float threshold, int detail);

/*
 * Mask point-manipulation kernels — ports of the remaining DT_OMP_FOR loops
 * in src/develop/masks/{circle,ellipse,brush,path,gradient}.c. Each C loop
 * is a simple point-arithmetic loop (shift, circumference, bbox-reduction,
 * guide-curve generation).
 */

/* Shift all points by (dx,dy) starting at start_index. Replaces the four
 * identical shift loops in circle.c:744, ellipse.c:333, brush.c:1065, path.c:1511. */
void darkroom_masks_points_shift(float *points, size_t count,
                                 float dx, float dy, size_t start_index);

/* Generate circle circumference: center at [0], l points around the arc.
 * Points buffer must hold 2*(l+1) floats. */
void darkroom_masks_circle_circumference(float *points,
                                         float center_x, float center_y,
                                         float r, int l);

/* Generate ellipse circumference points at indices 5..l+5.
 * Points buffer must hold 2*(l+5) floats; caller sets indices 0–4. */
void darkroom_masks_ellipse_circumference(float *points,
                                          float x, float y,
                                          float a, float b,
                                          float cosv, float sinv, int l);

/* Bounding-box reduction over points[start_idx..count), optionally
 * also checking border at the same indices. */
void darkroom_masks_bbox_reduction(const float *points, const float *border,
                                   size_t count, size_t start_idx,
                                   float *x_min_out, float *x_max_out,
                                   float *y_min_out, float *y_max_out);

/* Generate gradient guide curve points. Caller sets indices 0..2 (3 control
 * points). Writes guide points starting at index 3, returns count written. */
size_t darkroom_masks_gradient_guide_points(float *points, size_t count,
                                            float x, float y,
                                            float wd, float ht,
                                            float scale,
                                            float cosv, float sinv,
                                            float curvature);

#ifdef __cplusplus
} /* extern "C" */
#endif
