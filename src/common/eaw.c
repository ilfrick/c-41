/*
    This file is part of darktable,
    Copyright (C) 2017-2024 darktable developers.

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

#include "common/eaw.h"
#include "common/math.h"
#include "control/control.h"     // needed by dwt.h
#include "common/dwt.h"          // for dwt_interleave_rows
#include "rust_ffi/darkroom_core.h" // for darkroom_eaw_synthesize/_dn_decompose

static inline void __attribute__((unused)) weight(const dt_aligned_pixel_t c1,
                              const dt_aligned_pixel_t c2,
                              const dt_aligned_pixel_t sharpen,
                              dt_aligned_pixel_t weight)
{
/* Computes the vector
 * (wl, wc, wc, 1)
 *
 * where:
 * wl = exp(-sharpen*SQR(c1[0] - c2[0]))
 * wc = exp(-sharpen*(SQR(c1[1] - c2[1]) + SQR(c1[2] - c2[2]))
 */
  dt_aligned_pixel_t square;
  for_each_channel(c) square[c] = c1[c] - c2[c];
  for_each_channel(c) square[c] = square[c] * square[c];	// { d1, d2, d3, ? }
  const dt_aligned_pixel_t square2 = { square[0], square[2], square[1], square[3] }; // { d1, d3, d2, ? }
  dt_aligned_pixel_t added;
  for_each_channel(c)
    added[c] = square[c] + square2[c];				// { d1+d1, d2+d3, d2+d3, ? }
  dt_aligned_pixel_t sharpened;
  for_each_channel(c)
    sharpened[c] = sharpen[c] * added[c];			// { -s*d1,  -s*(d2+d3), -s*(d2+d3), 0 }
  dt_vector_exp(sharpened, weight);				// { wl, wc, wc, 1 }
}

static inline void __attribute__((unused)) accumulate(dt_aligned_pixel_t accum,
                              const dt_aligned_pixel_t detail,
                              const dt_aligned_pixel_t thresh,
                              const dt_aligned_pixel_t boostval)
{
  // decrease the absolute magnitude of the detail by the threshold,
  // then add the result to the accumulator; copysignf does not
  // vectorize, but it turns out that just adding up two clamped
  // alternatives gives exactly the same result and DOES vectorize
  //    const float absamt = fmaxf(0.0f, (fabsf(detail[k + c]) - threshold[c]));
  //    const float amount = copysignf(absamt, detail[k + c]);
  // the below code is the vectorization of
  //   amount = MAX(detail - thresh, 0.0f) + MIN(detail + thresh, 0.0f);
  static const dt_aligned_pixel_t zero = { 0.0f, 0.0f, 0.0f, 0.0f };
  dt_aligned_pixel_t sum, diff;
  for_four_channels(c, aligned(sum, diff, detail, thresh))
  {
    sum[c] = detail[c] + thresh[c];
    diff[c] = detail[c] - thresh[c];
  }
  dt_vector_min(sum, sum, zero);
  dt_vector_max(diff, diff, zero);
  dt_aligned_pixel_t amount;
  for_four_channels(c, aligned(amount, sum, diff))
    amount[c] = sum[c] + diff[c];

  for_four_channels(c,aligned(accum,detail,thresh,boostval))
  {
    accum[c] += (boostval[c] * amount[c]);
  }
}

#define SUM_PIXEL_CONTRIBUTION						\
  do                                                                    \
  {                                                                     \
    dt_aligned_pixel_t wp;                                              \
    weight(px + 4*i, px2, vsharpen, wp);                                \
    dt_aligned_pixel_t w;                                               \
    const float f = filter[filter_idx++];                               \
    for_four_channels(c,aligned(px2))                                   \
    {                                                                   \
      w[c] = f * wp[c];                                                 \
      wgt[c] += w[c];                                                   \
      sum[c] += w[c] * px2[c];                                          \
    }                                                                   \
  } while(0)

#define SUM_PIXEL_PROLOGUE                                                                                   \
  dt_aligned_pixel_t sum = { 0.0f, 0.0f, 0.0f, 0.0f };                                                       \
  dt_aligned_pixel_t wgt = { 0.0f, 0.0f, 0.0f, 0.0f };							     \
  size_t filter_idx = 0;

#define SUM_PIXEL_EPILOGUE                                                                                   \
  dt_aligned_pixel_t det;										     \
  for_each_channel(c)      										     \
  {													     \
    sum[c] /= wgt[c];                                                   				     \
    det[c] = (px[4*i+c] - sum[c]);								     	     \
  }                                                                       				     \
  copy_pixel_nontemporal(pcoarse + 4*i,sum);                                  				     \
  accumulate(pdetail + 4*i, det, threshold, boost);							     \

void eaw_decompose_and_synthesize(float *const restrict out,
                                  const float *const restrict in,
                                  float *const restrict accum,
                                  const int scale,
                                  const float sharpen,
                                  const dt_aligned_pixel_t threshold,
                                  const dt_aligned_pixel_t boost,
                                  const ssize_t width,
                                  const ssize_t height)
{
  // Live data-parallel loop ported to Rust (m4-186): the edge-avoiding 5x5
  // B-spline smooth (expf-based `weight` vector taps, sharpen vector) plus
  // the per-channel threshold/boost `accumulate` now runs in
  // darkroom_eaw_decompose_and_synthesize, which reuses the safe
  // decompose_and_synthesize kernel (natural row order and unified clamping
  // instead of the C's dwt_interleave_rows order and 3-phase split; in-range
  // pixels match, degenerate narrow images clamp instead of the C's
  // row-bleeding left-edge phase). Signature unchanged for the atrous caller.
  // OpenCL code and everything else above/below are untouched.
  darkroom_eaw_decompose_and_synthesize(out, in, accum, scale, sharpen, threshold, boost, width, height);
  dt_omploop_sfence();
}

void eaw_synthesize(float *const out, const float *const in, const float *const restrict detail,
                    const float *const restrict threshold, const float *const restrict boost,
                    const int32_t width, const int32_t height)
{
  // Live data-parallel loop ported to Rust (m4-182): the soft-threshold
  // accumulate (MAX(detail-thresh,0)+MIN(detail+thresh,0), scaled by boost)
  // now runs in darkroom_eaw_synthesize. `in` stays in the signature for
  // eaw_synthesize_t compatibility (the body never read it). Decompose
  // helpers, OpenCL code, and everything else above/below are untouched.
  darkroom_eaw_synthesize(out, in, detail, threshold, boost, width, height);
  dt_omploop_sfence();
}

// =====================================================================================
// begin wavelet code from denoiseprofile.c
// =====================================================================================

static inline float __attribute__((unused)) dn_weight(const float *c1, const float *c2, const float inv_sigma2)
{
  // 3d distance based on color
  dt_aligned_pixel_t sqr;
  for_each_channel(c)
  {
    const float diff = c1[c] - c2[c];
    sqr[c] = diff * diff;
  }
  const float dot = (sqr[0] + sqr[1] + sqr[2]) * inv_sigma2;
  const float var
      = 0.02f; // FIXME: this should ideally depend on the image before noise stabilizing transforms!
  const float off2 = 9.0f; // (3 sigma)^2
  return fast_mexp2f(MAX(0, dot * var - off2));
}

#undef SUM_PIXEL_CONTRIBUTION
#define SUM_PIXEL_CONTRIBUTION	 		                                                             \
  do                                                                                                         \
  {                                                                                                          \
    const float f = filter[filter_idx++];                                                                    \
    const float wp = dn_weight(px, px2, inv_sigma2);                                                         \
    const float w = f * wp;                                                                                  \
    for_each_channel(c,aligned(px2))                                                                         \
    {                                                                                                        \
      wgt[c] += w;                                                                                           \
      sum[c] += w * px2[c];                                                                                  \
    }                                                                                                        \
  } while(0)

#undef SUM_PIXEL_EPILOGUE
#define SUM_PIXEL_EPILOGUE                                                                                   \
  dt_aligned_pixel_t det;									             \
  for_each_channel(c)      										     \
  {													     \
    sum[c] /= wgt[c];                                                   				     \
    pcoarse[c] = sum[c];                                                                                     \
    det[c] = (px[c] - sum[c]);									             \
    sum_sq[c] += (det[c]*det[c]);					                                     \
  }                                                                       				     \
  copy_pixel_nontemporal(pdetail, det);                                                                      \
  px += 4;                                                                                                   \
  pdetail += 4;                                                                                              \
  pcoarse += 4;

void eaw_dn_decompose(float *const restrict out, const float *const restrict in, float *const restrict detail,
                      dt_aligned_pixel_t sum_squared, const int scale, const float inv_sigma2,
                      const int32_t width, const int32_t height)
{
  // Live data-parallel loop ported to Rust (m4-185): the à-trous B-spline
  // smooth with edge-avoiding dn_weight taps plus the sum-of-squared-details
  // statistic now runs in darkroom_eaw_dn_decompose, which reuses the safe
  // dn_decompose kernel (no duplication; in-range pixels match, sum_sq order
  // and degenerate narrow-image clamping follow the kernel's documented
  // semantics). Signature unchanged for eaw_dn_decompose_t compatibility
  // (denoiseprofile passes this through process_wavelets). Decompose-and-
  // synthesize helpers, OpenCL code, and everything else above/below are
  // untouched.
  darkroom_eaw_dn_decompose(out, in, detail, sum_squared, scale, inv_sigma2, width, height);
  dt_omploop_sfence();
}

#undef SUM_PIXEL_CONTRIBUTION
#undef SUM_PIXEL_PROLOGUE
#undef SUM_PIXEL_EPILOGUE

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

