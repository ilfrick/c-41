/*
    This file is part of darktable,
    Copyright (C) 2019-2024 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "common/fast_guided_filter.h"
#include "common/gaussian.h"
#include "rust_ffi/darkroom_core.h"

/***
 * DOCUMENTATION
 *
 * Exposure-Independent Guided Filter (EIGF)
 *
 * This filter is a modification of guided filter to make it exposure independent
 * As variance depends on the exposure, the original guided filter preserves
 * much better the edges in the highlights than in the shadows.
 * In particular doing:
 * (1) increase exposure by 1EV
 * (2) guided filtering
 * (3) decrease exposure by 1EV
 * is NOT equivalent to doing the guided filtering only.
 *
 * To overcome this, instead of using variance directly to determine "a",
 * we use a ratio:
 * variance / (pixel_value)^2
 * we tried also the following ratios:
 * - variance / average^2
 * - variance / (pixel_value * average)
 * we kept variance / (pixel_value)^2 as it seemed to behave a bit better than
 * the other (dividing by average^2 smoothed too much dark details surrounded
 * by bright pixels).
 *
 * This modification makes the filter exposure-independent.
 * However, due to the fact that the average advantages the bright pixels
 * compared to dark pixels if we consider that human eye sees in log,
 * we get strong bright halos.
 * These are due to the spatial averaging of "a" and "b" that is performed at
 * the end of the filter, especially due to the spatial averaging of "b".
 * We decided to remove this final spatial averaging, as it is very hard
 * to keep it without having either large unsmoothed regions or halos.
 * Although the filter may blur a bit less without it, it remains sufficiently
 * good at smoothing the image, and there are much less halos.
 *
 * The implementation EIGF uses downscaling to speed-up the filtering,
 * just like what is done in fast_guided_filter.h
**/

/* computes average and variance of guide and mask, and put them in out.
 * out has 4 channels:
 * - average of guide
 * - variance of guide
 * - average of mask
 * - covariance of mask and guide. */
static inline void eigf_variance_analysis(const float *const restrict guide, // I
                                    const float *const restrict mask, //p
                                    float *const restrict out,
                                    const size_t width, const size_t height,
                                    const float sigma)
{
  // We also use gaussian blurs instead of the square blurs of the guided filter
  const size_t Ndim = width * height;
  float *const restrict in = dt_alloc_align_float(Ndim * 4);

  // Seeds preserved from the former loop's initial reduction values, so the
  // (unreachable-in-practice) Ndim == 0 path behaves like the old C.
  dt_aligned_pixel_t min = { 10000000.0f, 10000000.0f, 10000000.0f, 10000000.0f };
  dt_aligned_pixel_t max = { 0.0f, 0.0f, 0.0f, 0.0f };
  // Ported to Rust FFI, replaces the former pack + min/max reduction loop
  darkroom_eigf_pack_variance_minmax_4c(in, guide, mask, Ndim, min, max);

  dt_gaussian_t *g = dt_gaussian_init(width, height, 4, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur_4c(g, in, out);
  dt_gaussian_free(g);

  // Ported to Rust FFI, replaces DT_OMP_FOR_SIMD loop
  darkroom_eigf_variance_correct_4c(out, Ndim);

  dt_free_align(in);
}

// same function as above, but specialized for the case where guide == mask
// for increased performance
static inline void eigf_variance_analysis_no_mask(const float *const restrict guide, // I
                                    float *const restrict out,
                                    const size_t width, const size_t height,
                                    const float sigma)
{
  // We also use gaussian blurs instead of the square blurs of the guided filter
  const size_t Ndim = width * height;
  float *const restrict in = dt_alloc_align_float(Ndim * 2);

  // Seeds preserved from the former loop's initial reduction values
  float min[2] = { 10000000.0f, 10000000.0f };
  float max[2] = { 0.0f, 0.0f };
  // Ported to Rust FFI, replaces the former pack + min/max reduction loop
  darkroom_eigf_pack_variance_minmax_2c(in, guide, Ndim, min, max);

  dt_gaussian_t *g = dt_gaussian_init(width, height, 2, max, min, sigma, 0);
  if(!g) return;
  dt_gaussian_blur(g, in, out);
  dt_gaussian_free(g);

  // Ported to Rust FFI, replaces DT_OMP_FOR_SIMD loop
  darkroom_eigf_variance_correct_2c(out, Ndim);

  dt_free_align(in);
}

void eigf_blending(float *const restrict image, const float *const restrict mask,
                  const float *const restrict av, const size_t Ndim,
                  const dt_iop_guided_filter_blending_t filter,
                  const float feathering)
{
  // Ported to Rust FFI, replaces the former element-wise loop
  darkroom_eigf_blending(image, mask, av, Ndim, (int)filter, feathering);
}

// same function as above, but specialized for the case where guide == mask
// for increased performance
void eigf_blending_no_mask(float *const restrict image,
                  const float *const restrict av, const size_t Ndim,
                  const dt_iop_guided_filter_blending_t filter,
                  const float feathering)
{
  // Ported to Rust FFI, replaces the former element-wise loop
  darkroom_eigf_blending_no_mask(image, av, Ndim, (int)filter, feathering);
}

__DT_CLONE_TARGETS__
static inline void fast_eigf_surface_blur(float *const restrict image,
                                      const size_t width, const size_t height,
                                      const float sigma, float feathering, const int iterations,
                                      const dt_iop_guided_filter_blending_t filter, const float scale,
                                      const float quantization, const float quantize_min, const float quantize_max)
{
  // Works in-place on a grey image
  // mostly similar with fast_surface_blur from fast_guided_filter.h

  // A down-scaling of 4 seems empirically safe and consistent no matter the image zoom level
  // see reference paper above for proof.
  const float scaling = fmaxf(fminf(sigma, 4.0f), 1.0f);
  const float ds_sigma = fmaxf(sigma / scaling, 1.0f);

  const size_t ds_height = height / scaling;
  const size_t ds_width = width / scaling;

  const size_t num_elem_ds = ds_width * ds_height;
  const size_t num_elem = width * height;

  float *const restrict mask = dt_alloc_align_float(num_elem);
  float *const restrict ds_image = dt_alloc_align_float(num_elem_ds);
  float *const restrict ds_mask = dt_alloc_align_float(num_elem_ds);
  // average - variance arrays: store the guide and mask averages and variances
  float *const restrict ds_av = dt_alloc_align_float(num_elem_ds * 4);
  float *const restrict av = dt_alloc_align_float(num_elem * 4);

  if(!ds_image || !ds_mask || !ds_av || !av)
  {
    dt_control_log(_("fast exposure independent guided filter failed to allocate memory, check your RAM settings"));
    goto clean;
  }

  // Iterations of filter models the diffusion, sort of
  for(int i = 0; i < iterations; i++)
  {
    // blend linear for all intermediate images
    dt_iop_guided_filter_blending_t blend = DT_GF_BLENDING_LINEAR;
    // use filter for last iteration
    if(i == iterations - 1)
      blend = filter;

    interpolate_bilinear(image, width, height, ds_image, ds_width, ds_height, 1);
    if(quantization != 0.0f)
    {
      // (Re)build the mask from the quantized image to help guiding
      quantize(image, mask, width * height, quantization, quantize_min, quantize_max);
      // Downsample the image for speed-up
      interpolate_bilinear(mask, width, height, ds_mask, ds_width, ds_height, 1);
      eigf_variance_analysis(ds_mask, ds_image, ds_av, ds_width, ds_height, ds_sigma);
      // Upsample the variances and averages
      interpolate_bilinear(ds_av, ds_width, ds_height, av, width, height, 4);
      // Blend the guided image
      eigf_blending(image, mask, av, num_elem, blend, feathering);
    }
    else
    {
      // no need to build a mask.
      eigf_variance_analysis_no_mask(ds_image, ds_av, ds_width, ds_height, ds_sigma);
      // Upsample the variances and averages
      interpolate_bilinear(ds_av, ds_width, ds_height, av, width, height, 2);
      // Blend the guided image
      eigf_blending_no_mask(image, av, num_elem, blend, feathering);
    }
  }

clean:
  dt_free_align(av);
  dt_free_align(ds_av);
  dt_free_align(ds_mask);
  dt_free_align(ds_image);
  dt_free_align(mask);
}
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

