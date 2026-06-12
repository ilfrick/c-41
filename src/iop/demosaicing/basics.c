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

static void pre_median(float *out,
                       const float *const in,
                       const int width,
                       const int height,
                       const uint32_t filters,
                       const int num_passes,
                       const float threshold)
{
  darkroom_demosaic_pre_median(out, in, width, height, filters, num_passes, threshold);
}

static void color_smoothing(float *out,
                            const int width,
                            const int height,
                            const int num_passes)
{
  darkroom_demosaic_color_smoothing(out, width, height, num_passes);
}

static void green_equilibration_lavg(float *out,
                                     const float *const in,
                                     const int width,
                                     const int height,
                                     const uint32_t filters,
                                     const float thr)
{
  darkroom_demosaic_green_eq_lavg(out, in, width, height, filters, thr);
}

static void green_equilibration_favg(float *out,
                                     const float *const in,
                                     const int width,
                                     const int height,
                                     const uint32_t filters)
{
  darkroom_demosaic_green_eq_favg(out, in, width, height, filters);
}

#ifdef HAVE_OPENCL
// color smoothing step by multiple passes of median filtering
static int color_smoothing_cl(const dt_iop_module_t *self,
                              const dt_dev_pixelpipe_iop_t *piece,
                              cl_mem dev_in,
                              cl_mem dev_out,
                              const int width,
                              const int height,
                              const int passes)
{
  const dt_iop_demosaic_global_data_t *gd = self->global_data;

  const int devid = piece->pipe->devid;

  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  cl_mem dev_tmp = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
  if(dev_tmp == NULL) goto error;

  dt_opencl_local_buffer_t locopt
    = (dt_opencl_local_buffer_t){ .xoffset = 2*1, .xfactor = 1, .yoffset = 2*1, .yfactor = 1,
                                  .cellsize = 4 * sizeof(float), .overhead = 0,
                                  .sizex = 1 << 8, .sizey = 1 << 8 };

  err = dt_opencl_local_buffer_opt(devid, gd->kernel_color_smoothing, &locopt);
  if(err != CL_SUCCESS) goto error;

  // two buffer references for our ping-pong
  cl_mem dev_t1 = dev_out;
  cl_mem dev_t2 = dev_tmp;

  const size_t sizes[2] = { ROUNDUP(width, locopt.sizex), ROUNDUP(height, locopt.sizey) };
  const size_t local[2] = { locopt.sizex, locopt.sizey };
  for(int pass = 0; pass < passes; pass++)
  {
    err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_color_smoothing, sizes, local,
      CLARG(dev_t1), CLARG(dev_t2), CLARG(width),
      CLARG(height), CLLOCAL(sizeof(float) * 4 * (locopt.sizex + 2) * (locopt.sizey + 2)));
    if(err != CL_SUCCESS) goto error;

    // swap dev_t1 and dev_t2
    cl_mem t = dev_t1;
    dev_t1 = dev_t2;
    dev_t2 = t;
  }

  // after last step we find final output in dev_t1.
  // let's see if this is in dev_tmp1 and needs to be copied to dev_out
  if(dev_t1 == dev_tmp)
  {
    // copy data from dev_tmp -> dev_out
    const size_t region[2] = { width, height };
    err = dt_opencl_enqueue_copy_image(devid, dev_tmp, dev_out, CLIMG_ORIGIN, CLIMG_ORIGIN, region);
  }

error:
  dt_opencl_release_mem_object(dev_tmp);
  if(err != CL_SUCCESS)
    dt_print(DT_DEBUG_OPENCL, "[opencl_demosaic_color_smoothing] problem '%s'", cl_errstr(err));
  return err;
}

#define DT_REDUCESIZE_MIN 64
static int green_equilibration_cl(const dt_iop_module_t *self,
                                  const dt_dev_pixelpipe_iop_t *piece,
                                  cl_mem dev_in,
                                  cl_mem dev_out,
                                  const int width,
                                  const int height,
                                  const uint32_t filters)
{
  const dt_iop_demosaic_data_t *d = piece->data;
  const dt_iop_demosaic_global_data_t *gd = self->global_data;

  const int devid = piece->pipe->devid;

  cl_mem dev_tmp = NULL;
  cl_mem dev_m = NULL;
  cl_mem dev_r = NULL;
  cl_mem dev_in1 = NULL;
  cl_mem dev_out1 = NULL;
  cl_mem dev_in2 = NULL;
  cl_mem dev_out2 = NULL;
  float *sumsum = NULL;

  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  if(d->green_eq == DT_IOP_GREEN_EQ_BOTH)
  {
    dev_tmp = dt_opencl_alloc_device(devid, width, height, sizeof(float));
    if(dev_tmp == NULL) goto error;
  }

  switch(d->green_eq)
  {
    case DT_IOP_GREEN_EQ_FULL:
      dev_in1 = dev_in;
      dev_out1 = dev_out;
      break;
    case DT_IOP_GREEN_EQ_LOCAL:
      dev_in2 = dev_in;
      dev_out2 = dev_out;
      break;
    case DT_IOP_GREEN_EQ_BOTH:
      dev_in1 = dev_in;
      dev_out1 = dev_tmp;
      dev_in2 = dev_tmp;
      dev_out2 = dev_out;
      break;
    case DT_IOP_GREEN_EQ_NO:
    default:
      goto error;
  }

  if(d->green_eq == DT_IOP_GREEN_EQ_FULL || d->green_eq == DT_IOP_GREEN_EQ_BOTH)
  {
    dt_opencl_local_buffer_t flocopt
      = (dt_opencl_local_buffer_t){ .xoffset = 0, .xfactor = 1, .yoffset = 0, .yfactor = 1,
                                    .cellsize = 2 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 4, .sizey = 1 << 4 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_green_eq_favg_reduce_first, &flocopt);
    if(err != CL_SUCCESS) goto error;

    const size_t bwidth = ROUNDUP(width, flocopt.sizex);
    const size_t bheight = ROUNDUP(height, flocopt.sizey);

    const int bufsize = (bwidth / flocopt.sizex) * (bheight / flocopt.sizey);

    dev_m = dt_opencl_alloc_device_buffer(devid, sizeof(float) * 2 * bufsize);
    if(dev_m == NULL)
    {
      err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
      goto error;
    }

    const size_t fsizes[2] = { bwidth, bheight };
    const size_t flocal[2] = { flocopt.sizex, flocopt.sizey };
    err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_green_eq_favg_reduce_first, fsizes, flocal,
      CLARG(dev_in1), CLARG(width),
      CLARG(height), CLARG(dev_m), CLARG(filters),
      CLLOCAL(sizeof(float) * 2 * flocopt.sizex * flocopt.sizey));
    if(err != CL_SUCCESS) goto error;

    dt_opencl_local_buffer_t slocopt
      = (dt_opencl_local_buffer_t){ .xoffset = 0, .xfactor = 1, .yoffset = 0, .yfactor = 1,
                                    .cellsize = sizeof(float) * 2, .overhead = 0,
                                    .sizex = 1 << 16, .sizey = 1 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_green_eq_favg_reduce_second, &slocopt);
    if(err != CL_SUCCESS) goto error;

    const int reducesize = MIN(DT_REDUCESIZE_MIN, ROUNDUP(bufsize, slocopt.sizex) / slocopt.sizex);

    dev_r = dt_opencl_alloc_device_buffer(devid, sizeof(float) * 2 * reducesize);
    if(dev_r == NULL)
    {
      err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
      goto error;
    }

    const size_t ssizes[2] = { (size_t)reducesize * slocopt.sizex, 1 };
    const size_t slocal[2] = { slocopt.sizex, 1 };
    err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_green_eq_favg_reduce_second, ssizes, slocal,
      CLARG(dev_m), CLARG(dev_r),
      CLARG(bufsize), CLLOCAL(sizeof(float) * 2 * slocopt.sizex));
    if(err != CL_SUCCESS) goto error;

    sumsum = dt_alloc_align_float((size_t)2 * reducesize);
    if(sumsum == NULL)
    {
      err = DT_OPENCL_SYSMEM_ALLOCATION;
      goto error;
    }

    err = dt_opencl_read_buffer_from_device(devid, (void *)sumsum, dev_r, 0,
                                            sizeof(float) * 2 * reducesize, TRUE);
    if(err != CL_SUCCESS) goto error;

    double sum1 = 0.0;
    double sum2 = 0.0;
    for(int k = 0; k < reducesize; k++)
    {
      sum1 += (double)sumsum[2 * k];
      sum2 += (double)sumsum[2 * k + 1];
    }

    const float gr_ratio = (sum1 > 0.0 && sum2 > 0.0) ? (float)(sum2 / sum1) : 1.0f;

    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_green_eq_favg_apply, width, height,
      CLARG(dev_in1), CLARG(dev_out1), CLARG(width), CLARG(height), CLARG(filters),
      CLARG(gr_ratio));
    if(err != CL_SUCCESS) goto error;
  }

  if(d->green_eq == DT_IOP_GREEN_EQ_LOCAL || d->green_eq == DT_IOP_GREEN_EQ_BOTH)
  {
    const dt_image_t *img = &self->dev->image_storage;
    const float threshold = 0.0001f * img->exif_iso;

    dt_opencl_local_buffer_t locopt
      = (dt_opencl_local_buffer_t){ .xoffset = 2*2, .xfactor = 1, .yoffset = 2*2, .yfactor = 1,
                                    .cellsize = 1 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 8, .sizey = 1 << 8 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_green_eq_lavg, &locopt);
    if(err != CL_SUCCESS) goto error;

    const size_t sizes[2] = { ROUNDUP(width, locopt.sizex), ROUNDUP(height, locopt.sizey) };
    const size_t local[2] = { locopt.sizex, locopt.sizey };
    err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_green_eq_lavg, sizes, local,
      CLARG(dev_in2), CLARG(dev_out2),
      CLARG(width), CLARG(height), CLARG(filters),
      CLARG(threshold), CLLOCAL(sizeof(float) * (locopt.sizex + 4) * (locopt.sizey + 4)));
    if(err != CL_SUCCESS) goto error;
  }

error:
  dt_opencl_release_mem_object(dev_tmp);
  dt_opencl_release_mem_object(dev_m);
  dt_opencl_release_mem_object(dev_r);
  dt_free_align(sumsum);
  if(err != CL_SUCCESS)
    dt_print(DT_DEBUG_OPENCL, "[opencl_demosaic_green_equilibration] problem  '%s'", cl_errstr(err));
  return err;
}

static int process_default_cl(const dt_iop_module_t *self,
                              const dt_dev_pixelpipe_iop_t *piece,
                              cl_mem dev_in,
                              cl_mem dev_out,
                              cl_mem dev_xtrans,
                              const int width,
                              const int height,
                              const int demosaicing_method,
                              const uint32_t filters)
{
  const dt_iop_demosaic_data_t *d = piece->data;
  const dt_iop_demosaic_global_data_t *gd = self->global_data;

  const int devid = piece->pipe->devid;

  cl_mem dev_tmp = NULL;
  cl_mem dev_med = NULL;
  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  if(demosaicing_method == DT_IOP_DEMOSAIC_PASSTHROUGH_MONOCHROME)
  {
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_passthrough_monochrome, width, height,
        CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height));
  }
  else if(demosaicing_method == DT_IOP_DEMOSAIC_PASSTHROUGH_COLOR)
  {
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_passthrough_color, width, height,
        CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height),
        CLARG(filters), CLARG(dev_xtrans));
  }
  else if(demosaicing_method == DT_IOP_DEMOSAIC_PPG)
  {
    dev_tmp = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
    if(dev_tmp == NULL) goto error;

    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_border_interpolate, width, height,
          CLARG(dev_in), CLARG(dev_tmp), CLARG(width), CLARG(height), CLARG(filters));
    if(err != CL_SUCCESS) goto error;

    if(d->median_thrs > 0.0f)
    {
      dev_med = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
      if(dev_med == NULL)
      {
        err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
        goto error;
      }

      dt_opencl_local_buffer_t locopt
          = (dt_opencl_local_buffer_t){ .xoffset = 2*2, .xfactor = 1, .yoffset = 2*2, .yfactor = 1,
                                        .cellsize = 1 * sizeof(float), .overhead = 0,
                                        .sizex = 1 << 8, .sizey = 1 << 8 };

      err = dt_opencl_local_buffer_opt(devid, gd->kernel_pre_median, &locopt);
      if(err != CL_SUCCESS) goto error;

      const size_t sizes[2] = { ROUNDUP(width, locopt.sizex), ROUNDUP(height, locopt.sizey) };
      const size_t local[2] = { locopt.sizex, locopt.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_pre_median, sizes, local,
          CLARG(dev_in), CLARG(dev_med), CLARG(width),
          CLARG(height), CLARG(filters), CLARG(d->median_thrs), CLLOCAL(sizeof(float) * (locopt.sizex + 4) * (locopt.sizey + 4)));
      if(err != CL_SUCCESS) goto error;
      dev_in = dev_out;
    }
    else dev_med = dev_in;

    {
      dt_opencl_local_buffer_t locopt
          = (dt_opencl_local_buffer_t){ .xoffset = 2*3, .xfactor = 1, .yoffset = 2*3, .yfactor = 1,
                                        .cellsize = sizeof(float) * 1, .overhead = 0,
                                        .sizex = 1 << 8, .sizey = 1 << 8 };

      err = dt_opencl_local_buffer_opt(devid, gd->kernel_ppg_green, &locopt);
      if(err != CL_SUCCESS) goto error;

      const size_t sizes[2] = { ROUNDUP(width, locopt.sizex), ROUNDUP(height, locopt.sizey) };
      const size_t local[2] = { locopt.sizex, locopt.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_ppg_green, sizes, local,
          CLARG(dev_med), CLARG(dev_tmp), CLARG(width),
          CLARG(height), CLARG(filters), CLLOCAL(sizeof(float) * (locopt.sizex + 2*3) * (locopt.sizey + 2*3)),
          CLARGINT(100000));
      if(err != CL_SUCCESS) goto error;
    }

    {
      dt_opencl_local_buffer_t locopt
          = (dt_opencl_local_buffer_t){ .xoffset = 2*1, .xfactor = 1, .yoffset = 2*1, .yfactor = 1,
                                        .cellsize = 4 * sizeof(float), .overhead = 0,
                                        .sizex = 1 << 8, .sizey = 1 << 8 };

      err = dt_opencl_local_buffer_opt(devid, gd->kernel_ppg_redblue, &locopt);
      if(err != CL_SUCCESS) goto error;

      const size_t sizes[2] = { ROUNDUP(width, locopt.sizex), ROUNDUP(height, locopt.sizey) };
      const size_t local[2] = { locopt.sizex, locopt.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_ppg_redblue, sizes, local,
          CLARG(dev_tmp), CLARG(dev_out), CLARG(width),
          CLARG(height), CLARG(filters), CLLOCAL(sizeof(float) * 4 * (locopt.sizex + 2) * (locopt.sizey + 2)),
          CLARGINT(100000));
      if(err != CL_SUCCESS) goto error;
    }
  }

error:
  if(dev_med != dev_in) dt_opencl_release_mem_object(dev_med);
  dt_opencl_release_mem_object(dev_tmp);

 if(err != CL_SUCCESS)
    dt_print(DT_DEBUG_OPENCL, "[opencl_demosaic] basic kernel problem '%s'", cl_errstr(err));
  return err;
}

static int demosaic_box3_cl(dt_iop_module_t *self,
                            dt_dev_pixelpipe_iop_t *piece,
                            cl_mem dev_in,
                            cl_mem dev_out,
                            cl_mem dev_xtrans,
                            const int width,
                            const int height,
                            const uint32_t filters)
{
  const dt_iop_demosaic_global_data_t *gd = self->global_data;
  const cl_int err = dt_opencl_enqueue_kernel_2d_args(piece->pipe->devid, gd->kernel_demosaic_box3, width, height,
                      CLARG(dev_in), CLARG(dev_out),
                      CLARG(width), CLARG(height),
                      CLARG(filters), CLARG(dev_xtrans));
  if(err != CL_SUCCESS)
    dt_print(DT_DEBUG_OPENCL, "[opencl_demosaic] box3 problem '%s'", cl_errstr(err));
  return err;
}

#endif
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

