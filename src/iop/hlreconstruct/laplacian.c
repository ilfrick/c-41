/*
   This file is part of darktable,
   Copyright (C) 2010-2026 darktable developers.

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


typedef enum diffuse_reconstruct_variant_t
{
  DIFFUSE_RECONSTRUCT_RGB = 0,
  DIFFUSE_RECONSTRUCT_CHROMA
} diffuse_reconstruct_variant_t;


enum wavelets_scale_t
{
  ANY_SCALE   = 1 << 0, // any wavelets scale   : reconstruct += HF
  FIRST_SCALE = 1 << 1, // first wavelets scale : reconstruct = 0
  LAST_SCALE  = 1 << 2, // last wavelets scale  : reconstruct += residual
};


static unsigned int scale_type(const int s, const int scales)
{
  unsigned int scale = ANY_SCALE;
  if(s == 0) scale |= FIRST_SCALE;
  if(s == scales - 1) scale |= LAST_SCALE;
  return scale;
}

static void _interpolate_and_mask(const float *const restrict input,
                                  float *const restrict interpolated,
                                  float *const restrict clipping_mask,
                                  const dt_aligned_pixel_t clips,
                                  const dt_aligned_pixel_t wb,
                                  const uint32_t filters,
                                  const size_t width,
                                  const size_t height)
{
  // Bilinear CFA interpolation + clipping mask (Rust FFI)
  darkroom_highlights_interpolate_and_mask(input, interpolated, clipping_mask,
                                           clips, wb, filters, width, height);
}

static void _remosaic_and_replace(const float *const restrict input,
                                  const float *const restrict interpolated,
                                  const float *const restrict clipping_mask,
                                  float *const restrict output,
                                  const dt_aligned_pixel_t wb,
                                  const uint32_t filters,
                                  const size_t width,
                                  const size_t height)
{
  // remosaic + alpha-blend with original (Rust FFI)
  darkroom_highlights_remosaic_and_replace(input, interpolated, clipping_mask,
                                           output, wb, filters, width, height);
}

static inline void guide_laplacians(const float *const restrict high_freq,
                                    const float *const restrict low_freq,
                                    const float *const restrict clipping_mask,
                                    float *const restrict output,
                                    const size_t width,
                                    const size_t height,
                                    const int mult,
                                    const float noise_level,
                                    const int salt,
                                    const unsigned int scale,
                                    const float radius_sq)
{
  float *const restrict out = DT_IS_ALIGNED(output);
  const float *const restrict LF = DT_IS_ALIGNED(low_freq);
  const float *const restrict HF = DT_IS_ALIGNED(high_freq);

  // guided-laplacian RGB reconstruction, one wavelet scale (Rust FFI)
  darkroom_highlights_guide_laplacians(HF, LF, clipping_mask, out,
                                       width, height, mult,
                                       noise_level, salt, scale, radius_sq);
}


static inline void heat_PDE_diffusion(const float *const restrict high_freq,
                                      const float *const restrict low_freq,
                                      const float *const restrict clipping_mask,
                                      float *const restrict output,
                                      const size_t width,
                                      const size_t height,
                                      const int mult,
                                      const uint8_t scale,
                                      const float first_order_factor)
{
  // Simultaneous inpainting for image structure and texture using anisotropic heat transfer model
  // https://www.researchgate.net/publication/220663968
  // modified as follow :
  //  * apply it in a multi-scale wavelet setup : we basically solve it twice, on the wavelets LF and HF layers.
  //  * replace the manual texture direction/distance selection by an automatic detection similar to the structure one,
  //  * generalize the framework for isotropic diffusion and anisotropic weighted on the isophote direction
  //  * add a variance regularization to better avoid edges.
  // The sharpness setting mimics the contrast equalizer effect by simply multiplying the HF by some gain.

  float *const restrict out = DT_IS_ALIGNED(output);
  const float *const restrict LF = DT_IS_ALIGNED(low_freq);
  const float *const restrict HF = DT_IS_ALIGNED(high_freq);

  // anisotropic heat-PDE diffusion of the ratios, one wavelet scale (Rust FFI)
  darkroom_highlights_heat_pde_diffusion(HF, LF, clipping_mask, out,
                                         width, height, mult,
                                         scale, first_order_factor);
}

static inline gint wavelets_process(const float *const restrict in,
                                    float *const restrict reconstructed,
                                    const float *const restrict clipping_mask,
                                    const size_t width,
                                    const size_t height,
                                    const int scales,
                                    float *const restrict HF,
                                    float *const restrict LF_odd,
                                    float *const restrict LF_even,
                                    const diffuse_reconstruct_variant_t variant,
                                    const float noise_level,
                                    const int salt,
                                    const float first_order_factor)
{
  gint success = TRUE;

  // À trous decimated wavelet decompose
  // there is a paper from a guy we know that explains it : https://jo.dreggn.org/home/2010_atrous.pdf
  // the wavelets decomposition here is the same as the equalizer/atrous module,

  // allocate a one-row temporary buffer for the decomposition
  size_t padded_size;
  float *const tempbuf = dt_alloc_perthread_float(4 * width, &padded_size); //TODO: alloc in caller
  for(int s = 0; s < scales; ++s)
  {
    //dt_print(DT_DEBUG_ALWAYS, "CPU Wavelet decompose : scale %i", s);
    const int mult = 1 << s;

    const float *restrict buffer_in;
    float *restrict buffer_out;

    if(s == 0)
    {
      buffer_in = in;
      buffer_out = LF_odd;
    }
    else if(s % 2 != 0)
    {
      buffer_in = LF_odd;
      buffer_out = LF_even;
    }
    else
    {
      buffer_in = LF_even;
      buffer_out = LF_odd;
    }

    decompose_2D_Bspline(buffer_in, HF, buffer_out, width, height, mult, tempbuf, padded_size);

    unsigned int current_scale_type = scale_type(s, scales);
    const float radius = sqf(equivalent_sigma_at_step(B_SPLINE_SIGMA, s * DS_FACTOR));

    if(variant == DIFFUSE_RECONSTRUCT_RGB)
      guide_laplacians(HF, buffer_out, clipping_mask, reconstructed, width, height, mult, noise_level, salt, current_scale_type, radius);
    else
      heat_PDE_diffusion(HF, buffer_out, clipping_mask, reconstructed, width, height, mult, current_scale_type, first_order_factor);

    if(darktable.dump_pfm_module)
    {
      char name[64];
      sprintf(name, "scale-input-%i", s);
      dt_dump_pfm(name, buffer_in, width, height,  4 * sizeof(float), "highlights");

      sprintf(name, "scale-blur-%i", s);
      dt_dump_pfm(name, buffer_out, width, height,  4 * sizeof(float), "highlights");
    }
  }
  dt_free_align(tempbuf);

  return success;
}


static void process_laplacian_bayer(dt_iop_module_t *self,
                                    dt_dev_pixelpipe_iop_t *piece,
                                    const void *const restrict ivoid,
                                    void *const restrict ovoid,
                                    const dt_iop_roi_t *const roi_in,
                                    const dt_iop_roi_t *const roi_out,
                                    const dt_aligned_pixel_t clips)
{
  dt_iop_highlights_data_t *data = piece->data;

  const uint32_t filters = piece->filters;
  dt_aligned_pixel_t wb = { 1.f, 1.f, 1.f, 1.f };
  if(piece->pipe->dsc.temperature.coeffs[0] != 0.f)
  {
    wb[0] = piece->pipe->dsc.temperature.coeffs[0];
    wb[1] = piece->pipe->dsc.temperature.coeffs[1];
    wb[2] = piece->pipe->dsc.temperature.coeffs[2];
  }

  const size_t height = roi_in->height;
  const size_t width = roi_in->width;
  const size_t ds_height = height / DS_FACTOR;
  const size_t ds_width = width / DS_FACTOR;

  // [R, G, B, norm] for each pixel
  float *restrict interpolated, *restrict clipping_mask;
  // temp buffers for blurs. We will need to cycle between them for memory efficiency
  float *restrict LF_odd, *restrict LF_even, *restrict temp;
  // wavelets scales buffers
  float *restrict HF, *restrict ds_interpolated, *restrict ds_clipping_mask;

  if(!dt_iop_alloc_image_buffers(self, roi_in, roi_out,
                                 4 | DT_IMGSZ_INPUT, &interpolated,
                                 4 | DT_IMGSZ_INPUT, &clipping_mask,
                                 0, NULL))
  {
    dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
    return;
  }

  const dt_iop_roi_t roi_ds = { .x = 0, .y = 0, .height = ds_height, .width = ds_width };
  if(!dt_iop_alloc_image_buffers(self, &roi_ds, &roi_ds,
                                 4 | DT_IMGSZ_INPUT, &LF_odd,
                                 4 | DT_IMGSZ_INPUT, &LF_even,
                                 4 | DT_IMGSZ_INPUT, &temp,
                                 4 | DT_IMGSZ_INPUT, &HF,
                                 4 | DT_IMGSZ_INPUT, &ds_interpolated,
                                 4 | DT_IMGSZ_INPUT, &ds_clipping_mask,
                                 0, NULL))
  {
    dt_free_align(interpolated);
    dt_free_align(clipping_mask);
    dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
    return;
  }

  const float scale = fmaxf(DS_FACTOR * piece->iscale / (roi_in->scale), 1.f);
  const float final_radius = (float)((int)(1 << data->scales)) / scale;
  const int scales = CLAMP((int)ceilf(log2f(final_radius)), 1, MAX_NUM_SCALES);

  const float noise_level = data->noise_level / scale;

  const float *const restrict input = (const float *const restrict)ivoid;
  float *const restrict output = (float *const restrict)ovoid;

  _interpolate_and_mask(input, interpolated, clipping_mask, clips, wb, filters, width, height);
  dt_box_mean(clipping_mask, height, width, 4, 2, 1);

  // Downsample
  interpolate_bilinear(clipping_mask, width, height, ds_clipping_mask, ds_width, ds_height, 4);
  interpolate_bilinear(interpolated, width, height, ds_interpolated, ds_width, ds_height, 4);

  for(int i = 0; i < data->iterations; i++)
  {
    const int salt = (i == data->iterations - 1); // add noise on the last iteration only
    wavelets_process(ds_interpolated, temp, ds_clipping_mask, ds_width, ds_height, scales, HF, LF_odd,
                     LF_even, DIFFUSE_RECONSTRUCT_RGB, noise_level, salt, data->solid_color);
    wavelets_process(temp, ds_interpolated, ds_clipping_mask, ds_width, ds_height, scales, HF, LF_odd,
                     LF_even, DIFFUSE_RECONSTRUCT_CHROMA, noise_level, salt, data->solid_color);
  }

  // Upsample
  interpolate_bilinear(ds_interpolated, ds_width, ds_height, interpolated, width, height, 4);
  _remosaic_and_replace(input, interpolated, clipping_mask, output, wb, filters, width, height);

  if(darktable.dump_pfm_module)
  {
    dt_dump_pfm("interpolated", interpolated, width, height,  4 * sizeof(float), "highlights");
    dt_dump_pfm("clipping_mask", clipping_mask, width, height,  4 * sizeof(float), "highlights");
  }

  dt_free_align(interpolated);
  dt_free_align(clipping_mask);
  dt_free_align(temp);
  dt_free_align(LF_even);
  dt_free_align(LF_odd);
  dt_free_align(HF);
  dt_free_align(ds_interpolated);
  dt_free_align(ds_clipping_mask);
}

#ifdef HAVE_OPENCL
static inline cl_int wavelets_process_cl(const int devid,
                                         cl_mem in,
                                         cl_mem reconstructed,
                                         cl_mem clipping_mask,
                                         const int width,
                                         const int height,
                                         dt_iop_highlights_global_data_t *const gd,
                                         const int scales,
                                         cl_mem HF,
                                         cl_mem LF_odd,
                                         cl_mem LF_even,
                                         const diffuse_reconstruct_variant_t variant,
                                         const float noise_level,
                                         const int salt,
                                         const float solid_color)
{
  cl_int err = CL_SUCCESS;

  // À trous wavelet decompose
  // there is a paper from a guy we know that explains it : https://jo.dreggn.org/home/2010_atrous.pdf
  // the wavelets decomposition here is the same as the equalizer/atrous module,
  for(int s = 0; s < scales; ++s)
  {
    //dt_print(DT_DEBUG_ALWAYS, "GPU Wavelet decompose : scale %i", s);
    const int mult = 1 << s;

    cl_mem buffer_in;
    cl_mem buffer_out;

    if(s == 0)
    {
      buffer_in = in;
      buffer_out = LF_odd;
    }
    else if(s % 2 != 0)
    {
      buffer_in = LF_odd;
      buffer_out = LF_even;
    }
    else
    {
      buffer_in = LF_even;
      buffer_out = LF_odd;
    }

    // Compute wavelets low-frequency scales
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_filmic_bspline_horizontal, width, height,
          CLARG(buffer_in), CLARG(HF), CLARG(width), CLARG(height), CLARG(mult));
    if(err != CL_SUCCESS) return err;

    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_filmic_bspline_vertical, width, height,
          CLARG(HF), CLARG(buffer_out), CLARG(width), CLARG(height), CLARG(mult));
    if(err != CL_SUCCESS) return err;

    // Compute wavelets high-frequency scales and backup the maximum of texture over the RGB channels
    // Note : HF = detail - LF
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_filmic_wavelets_detail, width, height,
          CLARG(buffer_in), CLARG(buffer_out), CLARG(HF), CLARG(width), CLARG(height));
    if(err != CL_SUCCESS) return err;

    unsigned int current_scale_type = scale_type(s, scales);
    const float radius = sqf(equivalent_sigma_at_step(B_SPLINE_SIGMA, s * DS_FACTOR));

    // Compute wavelets low-frequency scales
    if(variant == DIFFUSE_RECONSTRUCT_RGB)
    {
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_highlights_guide_laplacians, width, height,
        CLARG(HF), CLARG(buffer_out), CLARG(clipping_mask),
        CLARG(reconstructed), // read-only
        CLARG(reconstructed), // write-only
        CLARG(width), CLARG(height), CLARG(mult), CLARG(noise_level), CLARG(salt), CLARG(current_scale_type), CLARG(radius));
      if(err != CL_SUCCESS) return err;
    }
    else // DIFFUSE_RECONSTRUCT_CHROMA
    {
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_highlights_diffuse_color, width, height,
        CLARG(HF), CLARG(buffer_out), CLARG(clipping_mask),
        CLARG(reconstructed), // read-only
        CLARG(reconstructed), // write-only
        CLARG(width), CLARG(height), CLARG(mult), CLARG(current_scale_type), CLARG(solid_color));
      if(err != CL_SUCCESS) return err;
    }
  }

  return err;
}

static cl_int process_laplacian_bayer_cl(dt_iop_module_t *self,
                                         dt_dev_pixelpipe_iop_t *piece,
                                         cl_mem dev_in,
                                         cl_mem dev_out,
                                         const dt_iop_roi_t *const roi_in,
                                         const dt_iop_roi_t *const roi_out,
                                         const dt_aligned_pixel_t clips)
{
  dt_iop_highlights_data_t *data = piece->data;
  dt_iop_highlights_global_data_t *gd = self->global_data;

  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const int ds_height = height / DS_FACTOR;
  const int ds_width = width / DS_FACTOR;

  const size_t sizes[2] = { ROUNDUPDWD(width, devid), ROUNDUPDHT(height, devid) };
  const size_t ds_sizes[2] = { ROUNDUPDWD(ds_width, devid), ROUNDUPDHT(ds_height, devid) };

  const uint32_t filters = piece->filters;

  dt_aligned_pixel_t wb = { 1.f, 1.f, 1.f, 1.f };
  if(piece->pipe->dsc.temperature.coeffs[0] != 0.f)
  {
    wb[0] = piece->pipe->dsc.temperature.coeffs[0];
    wb[1] = piece->pipe->dsc.temperature.coeffs[1];
    wb[2] = piece->pipe->dsc.temperature.coeffs[2];
  }

  const float scale = fmaxf(DS_FACTOR * piece->iscale / (roi_in->scale), 1.f);
  const float final_radius = (float)((int)(1 << data->scales)) / scale;
  const int scales = CLAMP((int)ceilf(log2f(final_radius)), 1, MAX_NUM_SCALES);
  const float noise_level = data->noise_level / scale;

  cl_mem interpolated = dt_opencl_alloc_device(devid, sizes[0], sizes[1], sizeof(float) * 4);  // [R, G, B, norm] for each pixel
  cl_mem clipping_mask = dt_opencl_alloc_device(devid, sizes[0], sizes[1], sizeof(float) * 4); // [R, G, B, norm] for each pixel

  // temp buffer for blurs. We will need to cycle between them for memory efficiency
  cl_mem LF_odd = dt_opencl_alloc_device(devid, ds_sizes[0], ds_sizes[1], sizeof(float) * 4);
  cl_mem LF_even = dt_opencl_alloc_device(devid, ds_sizes[0], ds_sizes[1], sizeof(float) * 4);
  cl_mem temp = dt_opencl_alloc_device(devid, sizes[0], sizes[1], sizeof(float) * 4); // need full size here for blurring

  // wavelets scales buffers
  cl_mem HF = dt_opencl_alloc_device(devid, ds_sizes[0], ds_sizes[1], sizeof(float) * 4);
  cl_mem ds_interpolated = dt_opencl_alloc_device(devid, ds_sizes[0], ds_sizes[1], sizeof(float) * 4);
  cl_mem ds_clipping_mask = dt_opencl_alloc_device(devid, ds_sizes[0], ds_sizes[1], sizeof(float) * 4);

  cl_mem clips_cl = dt_opencl_copy_host_to_device_constant(devid, 4 * sizeof(float), (float*)clips);
  cl_mem wb_cl = dt_opencl_copy_host_to_device_constant(devid, 4 * sizeof(float), (float*)wb);
  if(!interpolated || !clipping_mask || !LF_odd || !LF_even || !temp || !HF
      || !ds_interpolated || !ds_clipping_mask || !clips_cl || !wb_cl)
    goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_highlights_bilinear_and_mask, width, height,
    CLARG(dev_in), CLARG(interpolated), CLARG(temp),
    CLARG(clips_cl), CLARG(wb_cl), CLARG(filters), CLARG(roi_out->width), CLARG(roi_out->height));
  if(err != CL_SUCCESS) goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_highlights_box_blur, width, height,
    CLARG(temp), CLARG(clipping_mask),
    CLARG(roi_out->width), CLARG(roi_out->height));
  if(err != CL_SUCCESS) goto error;

  // Downsample
  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_interpolate_bilinear, ds_width, ds_height,
    CLARG(clipping_mask), CLARG(width), CLARG(height),
    CLARG(ds_clipping_mask), CLARG(ds_width), CLARG(ds_height));
  if(err != CL_SUCCESS) goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_interpolate_bilinear, ds_width, ds_height,
    CLARG(interpolated), CLARG(width), CLARG(height),
    CLARG(ds_interpolated), CLARG(ds_width), CLARG(ds_height));

  if(err != CL_SUCCESS) goto error;

  for(int i = 0; i < data->iterations; i++)
  {
    const int salt = (i == data->iterations - 1); // add noise on the last iteration only
    err = wavelets_process_cl(devid, ds_interpolated, temp, ds_clipping_mask, ds_width, ds_height, gd, scales, HF,
                              LF_odd, LF_even, DIFFUSE_RECONSTRUCT_RGB, noise_level, salt, data->solid_color);
    if(err != CL_SUCCESS) goto error;

    err = wavelets_process_cl(devid, temp, ds_interpolated, ds_clipping_mask, ds_width, ds_height, gd, scales, HF,
                              LF_odd, LF_even, DIFFUSE_RECONSTRUCT_CHROMA, noise_level, salt, data->solid_color);
    if(err != CL_SUCCESS) goto error;
  }

  // Upsample
  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_interpolate_bilinear, width, height,
    CLARG(ds_interpolated), CLARG(ds_width), CLARG(ds_height),
    CLARG(interpolated), CLARG(width), CLARG(height));
  if(err != CL_SUCCESS) goto error;

  // Remosaic
  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_highlights_remosaic_and_replace, width, height,
    CLARG(dev_in), CLARG(interpolated), CLARG(clipping_mask), CLARG(dev_out),
    CLARG(wb_cl), CLARG(filters), CLARG(width), CLARG(height));

error:
  dt_opencl_release_mem_object(wb_cl);
  dt_opencl_release_mem_object(clips_cl);
  dt_opencl_release_mem_object(interpolated);
  dt_opencl_release_mem_object(ds_clipping_mask);
  dt_opencl_release_mem_object(ds_interpolated);
  dt_opencl_release_mem_object(clipping_mask);

  dt_opencl_release_mem_object(temp);
  dt_opencl_release_mem_object(LF_even);
  dt_opencl_release_mem_object(LF_odd);
  dt_opencl_release_mem_object(HF);
  return err;
}

#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
