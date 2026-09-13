/*
   This file is part of darktable,
   Copyright (C) 2021-2023 darktable developers.

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

#include "common/darktable.h"
#include "common/dwt.h"
#include "develop/openmp_maths.h"
#include "rust_ffi/darkroom_core.h"

// Define the following as TRUE to use nontemporal writes in the decomposition
// on a 32-core Threadripper, nt writes are 8% slower wiith one thread,
// break even at 8 threads, and are 8% faster with 32 and 64 threads
#define USE_NONTEMPORAL FALSE

// B spline filter
#define BSPLINE_FSIZE 5

// The B spline best approximate a Gaussian of standard deviation :
// see https://eng.aurelienpierre.com/2021/03/rotation-invariant-laplacian-for-2d-grids/
#define B_SPLINE_SIGMA 1.0553651328015339f

// Normalization scaling of the wavelet to approximate a laplacian
// from the function above for sigma = B_SPLINE_SIGMA as a constant
#define B_SPLINE_TO_LAPLACIAN 3.182727439285017f

static inline float equivalent_sigma_at_step(const float sigma, const unsigned int s)
{
  // If we stack several gaussian blurs of standard deviation sigma on top of each other,
  // this is the equivalent standard deviation we get at the end (after s steps)
  // First step is s = 0
  // see
  // https://eng.aurelienpierre.com/2021/03/rotation-invariant-laplacian-for-2d-grids/#Multi-scale-iterative-scheme
  if(s == 0)
    return sigma;
  else
    return sqrtf(sqf(equivalent_sigma_at_step(sigma, s - 1)) + sqf(exp2f((float)s) * sigma));
}

static inline unsigned int num_steps_to_reach_equivalent_sigma(const float sigma_filter, const float sigma_final)
{
  // The inverse of the above : compute the number of scales needed to reach the desired equivalent sigma_final
  // after sequential blurs of constant sigma_filter
  unsigned int s = 0;
  float radius = sigma_filter;
  while(radius < sigma_final)
  {
    ++s;
    radius = sqrtf(sqf(radius) + sqf((float)(1 << s) * sigma_filter));
  }
  return s + 1;
}


DT_OMP_DECLARE_SIMD(aligned(buf, indices, result:64))
static inline void sparse_scalar_product(const dt_aligned_pixel_t buf, const size_t indices[BSPLINE_FSIZE],
                                         dt_aligned_pixel_t result, const gboolean clip_negatives)
{
  // scalar product of 2 3×5 vectors stored as RGB planes and B-spline filter,
  // e.g. RRRRR - GGGGG - BBBBB
  static const float filter[BSPLINE_FSIZE] = { 1.0f / 16.0f,
                                               4.0f / 16.0f,
                                               6.0f / 16.0f,
                                               4.0f / 16.0f,
                                               1.0f / 16.0f };

  if(clip_negatives)
  {
    for_each_channel(c, aligned(buf,indices,result))
    {
      result[c] = MAX(0.0f, filter[0] * buf[indices[0] + c] +
                            filter[1] * buf[indices[1] + c] +
                            filter[2] * buf[indices[2] + c] +
                            filter[3] * buf[indices[3] + c] +
                            filter[4] * buf[indices[4] + c]);
    }
  }
  else
  {
    for_each_channel(c, aligned(buf,indices,result))
    {
      result[c] = filter[0] * buf[indices[0] + c] +
                  filter[1] * buf[indices[1] + c] +
                  filter[2] * buf[indices[2] + c] +
                  filter[3] * buf[indices[3] + c] +
                  filter[4] * buf[indices[4] + c];
    }
  }
}

DT_OMP_DECLARE_SIMD(aligned(in, temp))
static inline void _bspline_vertical_pass(const float *const restrict in, float *const restrict temp,
                                          size_t row, size_t width, size_t height, int mult, const gboolean clip_negatives)
{
  size_t DT_ALIGNED_ARRAY indices[BSPLINE_FSIZE];
  // compute the index offsets of the pixels of interest; since the offsets are the same for the entire row,
  // we only need to do this once and can then process the entire row
  indices[0] = 4 * width * MAX((int)row - 2 * mult, 0);
  indices[1] = 4 * width * MAX((int)row - mult, 0);
  indices[2] = 4 * width * row;
  indices[3] = 4 * width * MIN(row + mult, height-1);
  indices[4] = 4 * width * MIN(row + 2 * mult, height-1);
  for(size_t j = 0; j < width; j++)
  {
    // Compute the vertical blur of the current pixel and store it in the temp buffer for the row
    sparse_scalar_product(in + j * 4, indices, temp + j * 4, clip_negatives);
  }
}

DT_OMP_DECLARE_SIMD(aligned(temp, out))
static inline void _bspline_horizontal(const float *const restrict temp, float *const restrict out,
                                       size_t col, size_t width, int mult, const gboolean clip_negatives)
{
  // Compute the array indices of the pixels of interest; since the offsets will change near the ends of
  // the row, we need to recompute for each pixel
  size_t DT_ALIGNED_ARRAY indices[BSPLINE_FSIZE];
  indices[0] = 4 * MAX((int)col - 2 * mult, 0);
  indices[1] = 4 * MAX((int)col - mult,  0);
  indices[2] = 4 * col;
  indices[3] = 4 * MIN(col + mult, width-1);
  indices[4] = 4 * MIN(col + 2 * mult, width-1);
  // Compute the horizontal blur of the already vertically-blurred pixel and store the result at the proper
  //  row/column location in the output buffer
  sparse_scalar_product(temp, indices, out, clip_negatives);
}

DT_OMP_DECLARE_SIMD(aligned(in, out:64) aligned(tempbuf:16))
static inline void blur_2D_Bspline(const float *const restrict in,
                                   float *const restrict out,
                                   float *const restrict tempbuf,
                                   const size_t padded_size,
                                   const size_t width,
                                   const size_t height,
                                   const int mult,
                                   const gboolean clip_negatives)
{
  // A-trous B-spline blur ported to Rust (m4-192): the separable 5-tap
  // [1 4 6 4 1] / 16 vertical-then-horizontal pass now runs in
  // darkroom_blur_2d_bspline (non-nontemporal path only; USE_NONTEMPORAL
  // is FALSE). tempbuf and padded_size stay in the signature so the
  // filmicrgb and color-picker callers are unchanged; the Rust side owns
  // a private row scratch buffer instead.
  (void)tempbuf;
  (void)padded_size;
  darkroom_blur_2d_bspline(in, out, width, height, mult, clip_negatives);
}

DT_OMP_DECLARE_SIMD(aligned(in, HF, LF:64) aligned(tempbuf:16))
inline static void decompose_2D_Bspline(const float *const in,
                                        float *const HF,
                                        float *const restrict LF,
                                        const size_t width,
                                        const size_t height,
                                        const int mult,
                                        float *const tempbuf,
                                        const size_t padded_size)
{
  // Blur and decimated-wavelet split in one pass, ported to Rust (m4-193):
  // the double-clipped separable 5-tap blur into LF plus the unclipped
  // HF = in - LF residue now runs in darkroom_decompose_2d_bspline
  // (non-nontemporal path only; USE_NONTEMPORAL is FALSE). tempbuf and
  // padded_size stay in the signature so the diffuse and laplacian callers
  // are unchanged; the Rust side owns a private row scratch buffer instead.
  (void)tempbuf;
  (void)padded_size;
  darkroom_decompose_2d_bspline(in, HF, LF, width, height, mult);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
