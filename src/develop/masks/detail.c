/*
    This file is part of darktable,
    Copyright (C) 2013-2025 darktable developers.

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

/* How are "detail masks" implemented?

  The detail masks (DM) are used by the dual demosaicer and as a
  further refinement step for shape / parametric masks.  They contain
  threshold weighed values of pixel-wise local signal changes so they
  can be understood as "areas with or without local detail".

  As the DM using algorithms (like dual demosaicing, sharpening ...)
  are all pixel peeping we want the "original data" from the sensor to
  calculate it.  (Calculating the mask from the modules roi might not
  detect such regions at all because of scaling / rotating artifacts,
  some blurring earlier in the pipeline, color changes ...)

  In all cases the user interface is pretty simple, we just pass a
  threshold value, which is in the range of -1.0 to 1.0 by an
  additional slider in the masks refinement section.  Positive values
  will select regions with lots of local detail, negatives select for
  flat areas.  (The dual demosaicer only wants positives as we always
  look for high frequency content.)  A threshold value of 0.0 means
  bypassing.

  So the first important point is:

  We make sure taking the input data for the DM right from the
  demosaicer for normal raws or from rawprepare in case of
  monochromes. This means some additional housekeeping for the
  pixelpipe.

  If any mask in any module selects a threshold of != 0.0 we leave a
  flag in the pipe struct telling a) we want a DM and b) we want it
  from either demosaic or from rawprepare.  If such a flag has not
  been previously set we will force a pipeline reprocessing.

  gboolean dt_dev_write_scharr_mask(dt_dev_pixelpipe_iop_t *piece,
                                     float *const rgb,
                                     const dt_iop_roi_t *const roi_in,
                                     const gboolean rawmode)
  or it's _cl equivalent write a preliminary mask holding signal-change values
  for every pixel. These mask values are calculated as
  a) get Y0 for every pixel
  b) apply a scharr operator on it

  This scharr mask (SM) is not scaled but only cropped to the roi
  of the writing module (demosaic or rawprepare).  The pipe gets roi
  copy of the writing module so we can later scale/distort the LM.

  Calculating the SM is done for performance and lower mem pressure
  reasons, so we don't have to pass full data to the module.

  If a mask uses the details refinement step it takes the scharr
  mask and calculates an intermediate mask (IM) which is still not
  scaled but has the roi of the writing module.

  For every pixel we calculate the IM value via a sigmoid function
  with the threshold and scharr as parameters.

  At last the IM is slightly blurred to avoid hard transitions, as
  there still is no scaling we can use a constant sigma.
  Now we have an unscaled detail mask which requires to be transformed
  through the pipeline using

  float *dt_dev_distort_detail_mask(const dt_dev_pixelpipe_t *pipe, float *src, const dt_iop_module_t *target_module)

  returning a pointer to a distorted mask (DT) with same size as used
  in the module wanting the refinement.  This DM is finally used to
  refine the original mask.

  All other refinements and parametric parameters are untouched.

  Some additional comments:

  1. intentionally this details mask refinement has only been
     implemented for raws. Especially for compressed inmages like
     jpegs or 8bit input the algo didn't work as good because of input
     precision and compression artifacts.

  2. In the gui the slider is above the rest of the refinemt sliders
     to emphasize that blurring & feathering use the mask corrected by
     detail refinement.

  3. Of course credit goes to Ingo @heckflosse from rt team for the
     original idea. (in the rt world this is known as details mask)

  4. Thanks to rawfiner for pointing out how to use Y0 and scharr for better maths.

  hanno@schwalm-bremen.de 21/04/29
*/

#include "common/debug.h"
#include "common/gaussian.h"
#include "common/imagebuf.h"
#include "develop/masks.h"
#include "rust_ffi/darkroom_core.h"

float *dt_masks_calc_scharr_mask(dt_dev_pixelpipe_t *pipe,
                                 float *const restrict src,
                                 const int width,
                                 const int height,
                                 const gboolean rawmode)
{
  float *mask = dt_iop_image_alloc(width, height, 1);
  float *tmp = dt_iop_image_alloc(width, height, 1);

  if(!tmp || !mask)
  {
    dt_free_align(tmp);
    dt_free_align(mask);
    return NULL;
  }

  dt_aligned_pixel_t wb = { 1.0f, 1.0f, 1.0f, 1.0f };
  if(pipe->dsc.temperature.enabled && rawmode)
    for(int i=0; i < 3; i++)
      wb[i] /= pipe->dsc.temperature.coeffs[i];

  // scharr luminance + gradient (ported to Rust FFI, replaces 2 OMP loops)
  darkroom_masks_detail_scharr_luminance(src, tmp, width, height, wb);
  darkroom_masks_detail_scharr_gradient(tmp, mask, width, height);
  dt_free_align(tmp);
  return mask;
}

void dt_masks_calc_detail_blend(float *const restrict src,
                                float *out,
                                const size_t msize,
                                const float threshold,
                                const gboolean detail)
{
  if(!src || !out) return;

  // blend factor sigmoid (ported to Rust FFI, replaces DT_OMP_FOR_SIMD loop)
  darkroom_masks_detail_blend(src, out, msize, threshold, detail);
}

float *dt_masks_calc_detail_mask(dt_dev_pixelpipe_iop_t *piece,
                                 const float threshold,
                                 const gboolean detail)
{
  dt_dev_pixelpipe_t *pipe = piece->pipe;
  dt_dev_detail_mask_t *details = &pipe->scharr;

  if(!details->data)
    return NULL;

  const size_t msize = (size_t) details->roi.width * details->roi.height;
  float *tmp = dt_alloc_align_float(msize);
  float *mask = dt_alloc_align_float(msize);
  if(!tmp || !mask)
  {
    dt_free_align(tmp);
    dt_free_align(mask);
    return NULL;
  }

  dt_masks_calc_detail_blend(details->data, tmp, msize, threshold, detail);
  dt_gaussian_fast_blur(tmp, mask, details->roi.width, details->roi.height, 2.0f, 0.0f, 1.0f, 1);
  dt_free_align(tmp);
  return mask;
}


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
