/*
    This file is part of darktable,
    Copyright (C) 2021-2026 darktable developers.

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

#include "bauhaus/bauhaus.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/drawingarea.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include "rust_ffi/darkroom_core.h"

// #include <fftw3.h> // one day, include FFT convolution
#include "common/gaussian.h"
#include "common/math.h"
#include "develop/tiling.h"
#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_blurs_params_t)

typedef enum dt_iop_blur_type_t
{
  DT_BLUR_LENS = 0,     // $DESCRIPTION: "lens"
  DT_BLUR_MOTION = 1,   // $DESCRIPTION: "motion"
  DT_BLUR_GAUSSIAN = 2, // $DESCRIPTION: "gaussian"
} dt_iop_blur_type_t;


typedef struct dt_iop_blurs_params_t
{
  dt_iop_blur_type_t type; // $DEFAULT: DT_BLUR_LENS $DESCRIPTION: "blur type"
  int radius;              // $MIN: 4 $MAX: 256 $DEFAULT: 8 $DESCRIPTION: "blur radius"

  // lens blur params
  int blades;              // $MIN: 3 $MAX: 11 $DEFAULT: 5 $DESCRIPTION: "diaphragm blades"
  float concavity;         // $MIN: 1. $MAX: 9.  $DEFAULT: 1. $DESCRIPTION: "concavity"
  float linearity;         // $MIN: 0. $MAX: 1.  $DEFAULT: 1. $DESCRIPTION: "linearity"
  float rotation;          // $MIN: -M_PI_2f $MAX: M_PI_2f $DEFAULT: 0.f $DESCRIPTION: "rotation"

  // motion blur params
  float angle;             // $MIN: -M_PI_F $MAX: M_PI_F $DEFAULT: 0.f $DESCRIPTION: "direction"
  float curvature;         // $MIN: -2.   $MAX: 2.   $DEFAULT: 0. $DESCRIPTION: "curvature"
  float offset;            // $MIN: -1.   $MAX: 1.   $DEFAULT: 0  $DESCRIPTION: "offset"

} dt_iop_blurs_params_t;


typedef struct dt_iop_blurs_gui_data_t
{
  GtkWidget *type, *radius, *blades, *concavity, *linearity, *rotation, *angle, *curvature, *offset;
  GtkDrawingArea *area;
  unsigned char *img;
  int img_cached;
  float img_width;
} dt_iop_blurs_gui_data_t;


typedef struct dt_iop_blurs_global_data_t
{
  int kernel_blurs_convolve;
  int kernel_blurs_convolve_sparse;
  int kernel_blurs_restore_alpha;
} dt_iop_blurs_global_data_t;


const char *name()
{
  return _("blurs");
}

const char *aliases()
{
  return _("blur|lens|motion");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
                                _("simulate physically-accurate lens and motion blurs"),
                                _("creative"), _("linear, RGB, scene-referred"), _("linear, RGB"),
                                _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}


int default_group()
{
  return IOP_GROUP_EFFECTS | IOP_GROUP_EFFECT;
}


dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}


void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);
}

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  const dt_iop_blurs_params_t *p = piece->data;
  const float scale = fmaxf(piece->iscale / roi_in->scale, 1.f);
  const int radius = MAX(roundf(p->radius / scale), 2);

  // Gaussian uses IIR which handles tiling internally (no halo needed).
  // Lens/motion need full radius overlap so the convolution neighbourhood is intact.
  tiling->factor = 2.5f;
  tiling->factor_cl = 3.5f;
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = (p->type == DT_BLUR_GAUSSIAN) ? 0 : radius;
  tiling->align = 1;
}

// B spline filter
#define FSIZE 5

inline static void _blur_2D_Bspline(const float *const restrict in,
                                    float *const restrict out,
                                    const size_t width,
                                    const size_t height)
{
  DT_OMP_FOR(collapse(2))
  for(size_t i = 0; i < height; i++)
  {
    for(size_t j = 0; j < width; j++)
    {
      const size_t index = (i * width + j);
      float acc = 0.f;

      for(size_t ii = 0; ii < FSIZE; ++ii)
        for(size_t jj = 0; jj < FSIZE; ++jj)
        {
          const size_t row = CLAMP((int)i + (int)(ii - (FSIZE - 1) / 2), (int)0, (int)height - 1);
          const size_t col = CLAMP((int)j + (int)(jj - (FSIZE - 1) / 2), (int)0, (int)width - 1);
          const size_t k_index = (row * width + col);

          static const float DT_ALIGNED_ARRAY filter[FSIZE]
              = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

          acc += filter[ii] * filter[jj] * in[k_index];
        }

      out[index] = acc;
    }
  }
}


static inline void _init_kernel(float *const restrict buffer,
                                const size_t width,
                                const size_t height)
{
  darkroom_blurs_init_kernel(buffer, height * width);
}


static inline void _create_lens_kernel(float *const restrict buffer,
                                      const size_t width,
                                      const size_t height,
                                      const float n,
                                      const float m,
                                      const float k,
                                      const float rotation)
{
  // n is number of diaphragm blades
  // m is the concavity, aka the number of vertices on straight lines (?)
  // k is the roundness vs. linearity factor
  //   see https://math.stackexchange.com/a/4160104/498090
  // buffer sizes need to be odd

  darkroom_blurs_lens_kernel(buffer, width, height, n, m, k, rotation);
}

static inline void _create_motion_kernel(float *const restrict buffer,
                                         const size_t width,
                                         const size_t height,
                                         const float angle,
                                         const float curvature,
                                         const float offset)
{
  const float A = curvature / 2.f;

  // Spatial coordinates rounding error
  const float eps = 1.f / (float)width;

  const float radius = (float)(width - 1) / 2.f - 1;
  const float corr_angle = -M_PI_4f - angle;

  // Matrix of rotation
  const float M[2][2] = { { cosf(corr_angle), -sinf(corr_angle) },
                          { sinf(corr_angle), cosf(corr_angle) } };

  const float rot_flat[4] = { M[0][0], M[0][1], M[1][0], M[1][1] };
  darkroom_blurs_motion_kernel(buffer, width, A, offset, rot_flat, radius, eps);
}


static inline void _create_gauss_kernel(float *const restrict buffer,
                                        const size_t width, 
                                        const size_t height)
{
  darkroom_blurs_gauss_kernel(buffer, width, height);
}

static inline void _build_gui_kernel(unsigned char *const buffer,
                                    const size_t width,
                                    const size_t height,
                                    const dt_iop_blurs_params_t *p)
{
  float *const restrict kernel_1 = dt_alloc_align_float(width * height);
  float *const restrict kernel_2 = dt_alloc_align_float(width * height);
  if(!kernel_1 || !kernel_2)
  {
    dt_print(DT_DEBUG_ALWAYS,"[blurs] out of memory, skipping build_gui_kernel");
    goto cleanup;
  }

  if(p->type == DT_BLUR_LENS)
  {
    _create_lens_kernel(kernel_1, width, height, p->blades, p->concavity, p->linearity, p->rotation);

    // anti-aliasing step
    _blur_2D_Bspline(kernel_1, kernel_2, width, height);
  }
  else if(p->type == DT_BLUR_MOTION)
  {
    _init_kernel(kernel_1, width, height);
    _create_motion_kernel(kernel_1, width, height, p->angle, p->curvature, p->offset);

    // anti-aliasing step
    _blur_2D_Bspline(kernel_1, kernel_2, width, height);
  }
  else if(p->type == DT_BLUR_GAUSSIAN)
  {
    _create_gauss_kernel(kernel_2, width, height);
  }

  darkroom_blurs_gui_rgba(kernel_2, buffer, height * width);
cleanup:
  dt_free_align(kernel_1);
  dt_free_align(kernel_2);
}


static inline float _compute_norm(const float *const buffer, const size_t width, const size_t height)
{
  return darkroom_blurs_compute_norm(buffer, width * height);
}


static inline void _normalize(float *const buffer, const size_t width, const size_t height, const float norm)
{
  darkroom_blurs_normalize(buffer, width * height, norm);
}


static inline void _build_pixel_kernel(float *const buffer,
                                       const size_t width,
                                       const size_t height,
                                       const dt_iop_blurs_params_t *p)
{
  float *const restrict kernel_1 = dt_alloc_align_float(width * height);
  if(!kernel_1)
  {
    dt_print(DT_DEBUG_ALWAYS,"[blurs] out of memory, skippping build_pixel_kernel");
    return;
  }

  if(p->type == DT_BLUR_LENS)
  {
    _create_lens_kernel(kernel_1, width, height, p->blades, p->concavity, p->linearity, p->rotation + M_PI_F);

    // anti-aliasing step
    _blur_2D_Bspline(kernel_1, buffer, width, height);
  }
  else if(p->type == DT_BLUR_MOTION)
  {
    _init_kernel(kernel_1, width, height);
    _create_motion_kernel(kernel_1, width, height, p->angle + M_PI_F, p->curvature, p->offset);

    // anti-aliasing step
    _blur_2D_Bspline(kernel_1, buffer, width, height);
  }
  else if(p->type == DT_BLUR_GAUSSIAN)
  {
    _create_gauss_kernel(buffer, width, height);
  }

  // normalize to respect the conservation of energy law
  const float norm = _compute_norm(buffer, width, height);
  _normalize(buffer, width, height, norm);

  dt_free_align(kernel_1);
}

#if 0

// This crashes on the FFT step - not sure why and no time to investigate opaque libs now
// FFT convolution should be faster for large blurs because it is o(N log2(N))
// where N is the width of the kernel
// TODO

static void process_fft(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                        const void *const restrict ivoid, void *const restrict ovoid,
                        const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_blurs_params_t *p = piece->data;
  const float scale = piece->iscale / roi_in->scale;

  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out))
    return;

  const float *const restrict in = __builtin_assume_aligned(ivoid, 64);
  //float *const restrict out = __builtin_assume_aligned(ovoid, 64);

  // FFT needs odd buffer sizes, so fix that here
  const int is_width_even = (roi_in->width % 2 == 0);
  const int is_height_even = (roi_in->height % 2 == 0);
  const size_t padded_width = roi_in->width + 1 * is_width_even;
  const size_t padded_height = roi_in->height + 1 * is_height_even;

  float *const restrict padded_in = dt_alloc_align_float(padded_width * padded_height * 4);
  float *const restrict padded_out = dt_alloc_align_float(padded_width * padded_height * 4);
  if(!padded_in || !padded_out)
  {
    dt_print(DT_DEBUG_ALWAYS,"[blurs] out of memory, skipping process_fft");
    goto cleanup;
  }

  darkroom_blurs_pad_image(in, padded_in, roi_in->height, roi_in->width, padded_width);
  if(padded_width > roi_in->width)
    darkroom_blurs_pad_right_edge(in, padded_in, roi_in->height, roi_in->width, padded_width);
  if(padded_height > roi_in->height)
    darkroom_blurs_pad_bottom_edge(in, padded_in, roi_in->height, roi_in->width,
                                    padded_width, padded_height);

  // Init the blur kernel
  const size_t radius = MAX(roundf(p->radius / scale), 1);
  const size_t kernel_width = 2 * radius + 1;

  float *const restrict kernel = dt_alloc_align_float(kernel_width * kernel_width);
  build_pixel_kernel(kernel, kernel_width, kernel_width, p);

  // Convert to padded kernel - copy kernel in the center
  float *const restrict padded_kernel = dt_alloc_align_float(padded_width * padded_height);
  const size_t offset_i = (padded_height - 1) / 2 - (kernel_width - 1) / 2;
  const size_t offset_j = (padded_width - 1) / 2 - (kernel_width - 1) / 2;

  darkroom_blurs_pad_kernel(kernel, padded_kernel, kernel_width, padded_width,
                             offset_i, offset_j);

  // Init the FFT transforms
  int threads = fftw_init_threads();
  fftw_plan_with_nthreads(threads);

  // TODO: things go well until this point

  // Plan the dimensions of the FFT
  // notice we use fftwf prefix to use the float 32 variant of fftw
  int rank = 2;     /* 2D FFT */
  int n[2] = { padded_width, padded_height };
  int howmany = 4; /* 4 channels : RGBa */
  int idist = 1;   /* channels are distanced by one float in memory */
  int odist = 1;   /* channels are distanced by one float in memory */
  int istride = 4; /* array is not contiguous in memory */
  int ostride = 4; /* array is not contiguous in memory */
  int *inembed = n;
  int *onembed = n;

  fftwf_complex *const restrict kernel_fft = fftwf_alloc_complex(padded_height * padded_height * 4);
  fftwf_complex *const restrict image_fft = fftwf_alloc_complex(padded_height * padded_height * 4);

  // FFT convert the kernel
  fftwf_plan kernel_plan = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                 padded_kernel, inembed,
                                 istride, idist,
                                 kernel_fft, onembed,
                                 ostride, odist,
                                 FFTW_ESTIMATE);
  fftwf_execute(kernel_plan);

  // Clean FFT
  fftwf_destroy_plan(kernel_plan);
  fftwf_free(kernel_fft);
  fftwf_free(image_fft);
  fftw_cleanup_threads();

  dt_free_align(kernel);
  dt_free_align(padded_kernel);
cleanup:
  dt_free_align(padded_in);
  dt_free_align(padded_out);
}
#endif

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid,
             void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  dt_iop_blurs_params_t *p = piece->data;
  const float scale = fmaxf(piece->iscale / roi_in->scale, 1.f);

  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out))
    return;

  const float *const restrict in = DT_IS_ALIGNED(ivoid);
  float *const restrict out = DT_IS_ALIGNED(ovoid);

  const int radius = MAX(roundf(p->radius / scale), 2);

  // ── Gaussian fast path: IIR separable filter, O(n) regardless of radius ──
  if(p->type == DT_BLUR_GAUSSIAN)
  {
    // sigma matched to the original expf(-4*(d/radius)²) kernel shape
    const float sigma = (float)(radius - 1) / (2.0f * sqrtf(2.0f));
    const float range = 1.0e9f;
    const dt_aligned_pixel_t maxv = { range, range, range, range };
    const dt_aligned_pixel_t minv = { -range, -range, -range, -range };

    dt_gaussian_t *g =
      dt_gaussian_init(roi_in->width, roi_in->height, 4, maxv, minv, sigma, DT_IOP_GAUSSIAN_ZERO);
    if(g)
    {
      dt_gaussian_blur_4c(g, in, out);
      dt_gaussian_free(g);
    }
    else
    {
      dt_iop_copy_image_roi(ovoid, ivoid, 4, roi_in, roi_out);
    }

    // dt_gaussian_blur_4c also blurs alpha; restore original pipeline mask
    const size_t npix = (size_t)roi_out->width * roi_out->height;
    darkroom_blurs_alpha_restore(in, out, npix);
  }
  else
  {

    // ── Lens / motion: sparse spatial convolution ──
    //
    // With tiling_callback declaring overlap = radius, the pipeline provides
    // roi_in with a halo of (at least) radius pixels on each side.
    // ox/oy are the position of roi_out's origin within the roi_in buffer.
    const int in_width = roi_in->width;
    const int in_height = roi_in->height;
    const int ox = roi_out->x - roi_in->x; // >= 0
    const int oy = roi_out->y - roi_in->y; // >= 0

    const size_t kernel_width = 2 * radius + 1;
    const size_t max_entries = kernel_width * kernel_width;

    float *const restrict kernel = dt_alloc_align_float(max_entries);
    ptrdiff_t *const restrict offsets = dt_alloc_aligned(max_entries * sizeof(ptrdiff_t));
    float *const restrict values = dt_alloc_align_float(max_entries);

    if(!kernel || !offsets || !values)
    {
      dt_free_align(kernel);
      dt_free_align(offsets);
      dt_free_align(values);
      dt_iop_copy_image_roi(ovoid, ivoid, 4, roi_in, roi_out);
    }
    else
    {

      _build_pixel_kernel(kernel, kernel_width, kernel_width, p);

      // Build sparse list: pixel-offset (in floats) from center, for the interior
      // fast path where no clamping is needed.
      int n_nonzero = 0;
      for(int l = -radius; l <= radius; l++)
        for(int m = -radius; m <= radius; m++)
        {
          const float k = kernel[(l + radius) * kernel_width + (m + radius)];
          if(k > 1e-6f)
          {
            offsets[n_nonzero] = ((ptrdiff_t)l * in_width + m) * 4;
            values[n_nonzero] = k;
            n_nonzero++;
          }
        }

      darkroom_blurs_sparse_convolve(in, out,
                                     (size_t)roi_out->width, (size_t)roi_out->height,
                                     (size_t)in_width, (size_t)in_height,
                                     radius, ox, oy,
                                     offsets, values, (size_t)n_nonzero,
                                     kernel);

      dt_free_align(kernel);
      dt_free_align(offsets);
      dt_free_align(values);
    }
  }
}


#if HAVE_OPENCL
int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_blurs_params_t *p = piece->data;
  const dt_iop_blurs_global_data_t *const gd = self->global_data;

  cl_int err = DT_OPENCL_SYSMEM_ALLOCATION;
  const int devid = piece->pipe->devid;
  const int width = roi_out->width;
  const int height = roi_out->height;

  const float scale = fmaxf(piece->iscale / roi_in->scale, 1.f);
  const int radius = MAX(roundf(p->radius / scale), 2);

  // ── Gaussian fast path: IIR separable filter ──
  if(p->type == DT_BLUR_GAUSSIAN)
  {
    const float sigma = (float)(radius - 1) / (2.0f * sqrtf(2.0f));
    const float range = 1.0e9f;
    const dt_aligned_pixel_t maxv = { range, range, range, range };
    const dt_aligned_pixel_t minv = { -range, -range, -range, -range };

    cl_mem dev_blurred = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
    dt_gaussian_cl_t *g =
      dt_gaussian_init_cl(devid, width, height, 4, maxv, minv, sigma, DT_IOP_GAUSSIAN_ZERO);
    if(dev_blurred && g)
    {
      err = dt_gaussian_blur_cl(g, dev_in, dev_blurred);
      if(err == CL_SUCCESS)
        err = dt_opencl_enqueue_kernel_2d_args(devid,
                                               gd->kernel_blurs_restore_alpha,
                                               width,
                                               height,
                                               CLARG(dev_in),
                                               CLARG(dev_blurred),
                                               CLARG(dev_out),
                                               CLARG(width),
                                               CLARG(height));
    }
    dt_gaussian_free_cl(g);
    dt_opencl_release_mem_object(dev_blurred);
    return err;
  }

  // ── Lens / motion: sparse convolution ──
  const size_t kernel_width = 2 * radius + 1;
  const size_t max_entries = kernel_width * kernel_width;

  float *const restrict kernel = dt_alloc_align_float(max_entries);
  int *const restrict offsets_x = dt_alloc_aligned(max_entries * sizeof(int));
  int *const restrict offsets_y = dt_alloc_aligned(max_entries * sizeof(int));
  float *const restrict values = dt_alloc_align_float(max_entries);

  cl_mem dev_offsets_x = NULL;
  cl_mem dev_offsets_y = NULL;
  cl_mem dev_values = NULL;
  cl_mem dev_kernel = NULL;

  if(!kernel || !offsets_x || !offsets_y || !values)
    goto cleanup_cl;

  _build_pixel_kernel(kernel, kernel_width, kernel_width, p);

  // Build sparse list
  int n_nonzero = 0;
  for(int l = -radius; l <= radius; l++)
    for(int m = -radius; m <= radius; m++)
    {
      const float k = kernel[(l + radius) * kernel_width + (m + radius)];
      if(k > 1e-6f)
      {
        offsets_x[n_nonzero] = m;
        offsets_y[n_nonzero] = l;
        values[n_nonzero] = k;
        n_nonzero++;
      }
    }

  // Try sparse path first; fall back to dense if allocation fails
  dev_offsets_x = dt_opencl_copy_host_to_device_constant(devid, n_nonzero * sizeof(int), offsets_x);
  dev_offsets_y = dt_opencl_copy_host_to_device_constant(devid, n_nonzero * sizeof(int), offsets_y);
  dev_values = dt_opencl_copy_host_to_device_constant(devid, n_nonzero * sizeof(float), values);

  if(dev_offsets_x && dev_offsets_y && dev_values)
  {
    err = dt_opencl_enqueue_kernel_2d_args(devid,
                                           gd->kernel_blurs_convolve_sparse,
                                           width,
                                           height,
                                           CLARG(dev_in),
                                           CLARG(dev_offsets_x),
                                           CLARG(dev_offsets_y),
                                           CLARG(dev_values),
                                           CLARG(dev_out),
                                           CLARG(width),
                                           CLARG(height),
                                           CLARG(n_nonzero));
  }
  else
  {
    // Dense fallback: kernel as read-only buffer (L2-cacheable on GPU)
    dev_kernel = dt_opencl_copy_host_to_device_constant(devid, max_entries * sizeof(float), kernel);
    if(dev_kernel)
    {
      const int kw = (int)kernel_width;
      err = dt_opencl_enqueue_kernel_2d_args(devid,
                                             gd->kernel_blurs_convolve,
                                             width,
                                             height,
                                             CLARG(dev_in),
                                             CLARG(dev_kernel),
                                             CLARG(dev_out),
                                             CLARG(width),
                                             CLARG(height),
                                             CLARG(radius),
                                             CLARG(kw));
    }
  }

cleanup_cl:
  dt_free_align(kernel);
  dt_free_align(offsets_x);
  dt_free_align(offsets_y);
  dt_free_align(values);
  dt_opencl_release_mem_object(dev_offsets_x);
  dt_opencl_release_mem_object(dev_offsets_y);
  dt_opencl_release_mem_object(dev_values);
  dt_opencl_release_mem_object(dev_kernel);
  return err;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 34; // blurs.cl
  dt_iop_blurs_global_data_t *gd = malloc(sizeof(dt_iop_blurs_global_data_t));
  self->data = gd;
  gd->kernel_blurs_convolve = dt_opencl_create_kernel(program, "convolve");
  gd->kernel_blurs_convolve_sparse = dt_opencl_create_kernel(program, "convolve_sparse");
  gd->kernel_blurs_restore_alpha = dt_opencl_create_kernel(program, "restore_alpha");
}


void cleanup_global(dt_iop_module_so_t *self)
{
  const dt_iop_blurs_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_blurs_convolve);
  dt_opencl_free_kernel(gd->kernel_blurs_convolve_sparse);
  dt_opencl_free_kernel(gd->kernel_blurs_restore_alpha);
  free(self->data);
  self->data = NULL;
}
#endif


void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  const dt_iop_blurs_params_t *p = self->params;
  const dt_iop_blurs_gui_data_t *g = self->gui_data;

  if(!w || w == g->type)
  {
    if(p->type == DT_BLUR_LENS)
    {
      gtk_widget_hide(g->angle);
      gtk_widget_hide(g->curvature);
      gtk_widget_hide(g->offset);

      gtk_widget_show(g->blades);
      gtk_widget_show(g->concavity);
      gtk_widget_show(g->rotation);
      gtk_widget_show(g->linearity);
    }
    else if(p->type == DT_BLUR_MOTION)
    {
      gtk_widget_show(g->angle);
      gtk_widget_show(g->curvature);
      gtk_widget_show(g->offset);

      gtk_widget_hide(g->blades);
      gtk_widget_hide(g->concavity);
      gtk_widget_hide(g->rotation);
      gtk_widget_hide(g->linearity);
    }
    else if(p->type == DT_BLUR_GAUSSIAN)
    {
      gtk_widget_hide(g->angle);
      gtk_widget_hide(g->curvature);
      gtk_widget_hide(g->offset);

      gtk_widget_hide(g->blades);
      gtk_widget_hide(g->concavity);
      gtk_widget_hide(g->rotation);
      gtk_widget_hide(g->linearity);
    }
  }

  // update kernel view
  if(g->img_cached)
  {
    _build_gui_kernel(g->img, g->img_width, g->img_width, p);
    gtk_widget_queue_draw(GTK_WIDGET(g->area));
  }
}

static gboolean _dt_iop_tonecurve_draw(GtkWidget *widget, cairo_t *crf, const dt_iop_module_t *self)
{
  dt_iop_blurs_gui_data_t *g = self->gui_data;
  const dt_iop_blurs_params_t *p = self->params;

  GtkAllocation allocation;
  GtkStyleContext *context = gtk_widget_get_style_context(widget);
  gtk_widget_get_allocation(widget, &allocation);
  gtk_render_background(context, crf, 0, 0, allocation.width, allocation.height);

  if(allocation.width != g->img_width)
  {
    // Widget size changed, flush the cache buffer and restart
    g->img_cached = FALSE;
    if(g->img) dt_free_align(g->img);
  }

  if(!g->img_cached)
  {
    g->img = dt_alloc_align_type(unsigned char, 4 * allocation.width * allocation.width);
    g->img_width = allocation.width;
    g->img_cached = TRUE;
    _build_gui_kernel(g->img, g->img_width, g->img_width, p);

    // Note: if params change, we silently recompute the img in the buffer
    // no need to flush the cache. Flush only if buffer size changes,
    // aka GUI widget gets resized.
  }

  // Paint the kernel
  const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, g->img_width);
  cairo_surface_t *cst = cairo_image_surface_create_for_data(g->img, CAIRO_FORMAT_ARGB32,
                            g->img_width, g->img_width, stride);

  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return TRUE;
}

void gui_update(dt_iop_module_t *self)
{
// FIXME check why needed
  gui_changed(self, NULL, NULL);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_blurs_gui_data_t *g = IOP_GUI_ALLOC(blurs);

  // Image buffer to store the kernel look
  // Don't recompute it in the drawing function, only when a param is changed
  // then serve it from cache to the drawing function.
  g->img_cached = FALSE;
  g->img = NULL;
  g->img_width = 0.f;

  g->area = GTK_DRAWING_AREA(dtgtk_drawing_area_new_with_height(0));
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_dt_iop_tonecurve_draw), self);
  self->widget = dt_gui_vbox(g->area);

  g->radius = dt_bauhaus_slider_from_params(self, "radius");
  dt_bauhaus_slider_set_format(g->radius, " px");

  // The current implementation's run time is proportional to the square of
  // the blur radius. Let's soft-limit it to protect users from unacceptably
  // long processing times for the full image (like when exporting), which
  // many users perceive as a hang.
  // NOTE: This is just quick damage control. The correct solution is to
  // replace the bad implementation with a more efficient one.
  dt_bauhaus_slider_set_soft_max(g->radius, 64);

  g->type = dt_bauhaus_combobox_from_params(self, "type");

  g->blades = dt_bauhaus_slider_from_params(self, "blades");
  g->concavity = dt_bauhaus_slider_from_params(self, "concavity");
  g->linearity = dt_bauhaus_slider_from_params(self, "linearity");

  g->rotation = dt_bauhaus_slider_from_params(self, "rotation");
  dt_bauhaus_slider_set_factor(g->rotation, RAD_2_DEG);
  dt_bauhaus_slider_set_format(g->rotation, "°");

  g->angle = dt_bauhaus_slider_from_params(self, "angle");
  dt_bauhaus_slider_set_factor(g->angle, -RAD_2_DEG);
  dt_bauhaus_slider_set_format(g->angle, "°");

  g->curvature = dt_bauhaus_slider_from_params(self, "curvature");
  g->offset = dt_bauhaus_slider_from_params(self, "offset");

  // add the tooltips
  gtk_widget_set_tooltip_markup(g->radius, _("size of the blur in pixels\n"
                                             "<b>caution</b>: doubling the radius quadruples the run-time!"));
  gtk_widget_set_tooltip_text(g->concavity, _("shifts towards a star shape as value approaches blades-1"));
  gtk_widget_set_tooltip_text(g->linearity,
                              _("adjust straightness of edges from 0=perfect circle\nto 1=completely straight"));
  gtk_widget_set_tooltip_text(g->rotation, _("set amount by which to rotate shape around its center"));

  gtk_widget_set_tooltip_text(g->angle, _("orientation of the motion's path"));
  gtk_widget_set_tooltip_text(g->curvature, _("amount to curve the motion relative\nto its overall orientation"));
  gtk_widget_set_tooltip_text(g->offset, _("select which portion of the path to use,\n"
                                           "allowing the path to become asymmetric"));
}

void gui_cleanup(dt_iop_module_t *self)
{
  const dt_iop_blurs_gui_data_t *g = self->gui_data;
  if(g->img) dt_free_align(g->img);
}


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

