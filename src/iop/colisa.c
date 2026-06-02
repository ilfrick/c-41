/*
  This file is part of darktable,
  Copyright (C) 2013-2026 darktable developers.

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
#include "common/debug.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/tiling.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "rust_ffi/darkroom_core.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <gtk/gtk.h>
#include <inttypes.h>

DT_MODULE_INTROSPECTION(1, dt_iop_colisa_params_t)

typedef struct dt_iop_colisa_params_t
{
  float contrast;   // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float brightness; // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float saturation; // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
} dt_iop_colisa_params_t;

typedef struct dt_iop_colisa_gui_data_t
{
  GtkWidget *contrast;
  GtkWidget *brightness;
  GtkWidget *saturation;
} dt_iop_colisa_gui_data_t;

typedef struct dt_iop_colisa_data_t
{
  float contrast;
  float brightness;
  float saturation;
  float ctable[0x10000];      // precomputed look-up table for contrast curve
  float cunbounded_coeffs[3]; // approximation for extrapolation of contrast curve
  float ltable[0x10000];      // precomputed look-up table for brightness curve
  float lunbounded_coeffs[3]; // approximation for extrapolation of brightness curve
} dt_iop_colisa_data_t;

typedef struct dt_iop_colisa_global_data_t
{
  int kernel_colisa;
} dt_iop_colisa_global_data_t;


const char *deprecated_msg()
{
  return _("this module is deprecated. please use colorbalance RGB module instead.");
}

const char *name()
{
  return _("contrast brightness saturation");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("adjust the look of the image"),
                                      _("creative"),
                                      _("non-linear, Lab, display-referred"),
                                      _("non-linear, Lab"),
                                      _("non-linear, Lab, display-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING
    | IOP_FLAGS_ALLOW_TILING | IOP_FLAGS_DEPRECATED;
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_GRADING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_LAB;
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_colisa_data_t *d = piece->data;
  dt_iop_colisa_global_data_t *gd = self->global_data;

  cl_int err = DT_OPENCL_DEFAULT_ERROR;
  const int devid = piece->pipe->devid;

  const int width = roi_in->width;
  const int height = roi_in->height;

  const float saturation = d->saturation;

  cl_mem dev_cm = NULL;
  cl_mem dev_ccoeffs = NULL;
  cl_mem dev_lm = NULL;
  cl_mem dev_lcoeffs = NULL;

  dev_cm = dt_opencl_copy_host_to_device(devid, d->ctable, 256, 256, sizeof(float));
  if(dev_cm == NULL) goto error;
  dev_ccoeffs = dt_opencl_copy_host_to_device_constant(devid, sizeof(float) * 3, d->cunbounded_coeffs);
  if(dev_ccoeffs == NULL) goto error;
  dev_lm = dt_opencl_copy_host_to_device(devid, d->ltable, 256, 256, sizeof(float));
  if(dev_lm == NULL) goto error;
  dev_lcoeffs = dt_opencl_copy_host_to_device_constant(devid, sizeof(float) * 3, d->lunbounded_coeffs);
  if(dev_lcoeffs == NULL) goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_colisa, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width),
    CLARG(height), CLARG(saturation), CLARG(dev_cm), CLARG(dev_ccoeffs), CLARG(dev_lm), CLARG(dev_lcoeffs));

error:
  dt_opencl_release_mem_object(dev_lcoeffs);
  dt_opencl_release_mem_object(dev_lm);
  dt_opencl_release_mem_object(dev_ccoeffs);
  dt_opencl_release_mem_object(dev_cm);
  return err;
}
#endif

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_colisa_data_t *data = piece->data;
  const size_t npixels = (size_t)roi_in->width * roi_in->height;
  darkroom_colisa_process((const float *)ivoid, (float *)ovoid, npixels,
                          data->ctable, data->cunbounded_coeffs,
                          data->ltable, data->lunbounded_coeffs,
                          data->saturation);
}


void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_colisa_params_t *p = (dt_iop_colisa_params_t *)p1;
  dt_iop_colisa_data_t *d = piece->data;

  d->contrast = p->contrast + 1.0f; // rescale from [-1;+1] to [0;+2] (zero meaning no contrast -> gray plane)
  d->brightness = p->brightness * 2.0f; // rescale from [-1;+1] to [-2;+2]
  d->saturation = p->saturation + 1.0f; // rescale from [-1;+1] to [0;+2] (zero meaning no saturation -> b&w)

  // generate precomputed contrast curve
  darkroom_colisa_build_contrast_lut(d->ctable, d->contrast);

  // now the extrapolation stuff for the contrast curve:
  const float xc[4] = { 0.7f, 0.8f, 0.9f, 1.0f };
  const float yc[4] = { d->ctable[CLAMP((int)(xc[0] * 0x10000ul), 0, 0xffff)],
                        d->ctable[CLAMP((int)(xc[1] * 0x10000ul), 0, 0xffff)],
                        d->ctable[CLAMP((int)(xc[2] * 0x10000ul), 0, 0xffff)],
                        d->ctable[CLAMP((int)(xc[3] * 0x10000ul), 0, 0xffff)] };
  dt_iop_estimate_exp(xc, yc, 4, d->cunbounded_coeffs);


  // generate precomputed brightness curve
  const float gamma = (d->brightness >= 0.0f) ? 1.0f / (1.0f + d->brightness) : (1.0f - d->brightness);

  darkroom_colisa_build_brightness_lut(d->ltable, gamma);

  // now the extrapolation stuff for the brightness curve:
  const float xl[4] = { 0.7f, 0.8f, 0.9f, 1.0f };
  const float yl[4] = { d->ltable[CLAMP((int)(xl[0] * 0x10000ul), 0, 0xffff)],
                        d->ltable[CLAMP((int)(xl[1] * 0x10000ul), 0, 0xffff)],
                        d->ltable[CLAMP((int)(xl[2] * 0x10000ul), 0, 0xffff)],
                        d->ltable[CLAMP((int)(xl[3] * 0x10000ul), 0, 0xffff)] };
  dt_iop_estimate_exp(xl, yl, 4, d->lunbounded_coeffs);
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_colisa_data_t *d = calloc(1, sizeof(dt_iop_colisa_data_t));
  piece->data = (void *)d;
  for(int k = 0; k < 0x10000; k++) d->ctable[k] = d->ltable[k] = 100.0f * k / 0x10000; // identity
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 2; // basic.cl, from programs.conf
  dt_iop_colisa_global_data_t *gd = malloc(sizeof(dt_iop_colisa_global_data_t));
  self->data = gd;
  gd->kernel_colisa = dt_opencl_create_kernel(program, "colisa");
}


void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_colisa_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_colisa);
  free(self->data);
  self->data = NULL;
}


void gui_init(dt_iop_module_t *self)
{
  dt_iop_colisa_gui_data_t *g = IOP_GUI_ALLOC(colisa);

  g->contrast = dt_bauhaus_slider_from_params(self, N_("contrast"));
  g->brightness = dt_bauhaus_slider_from_params(self, N_("brightness"));
  g->saturation = dt_bauhaus_slider_from_params(self, N_("saturation"));

  gtk_widget_set_tooltip_text(g->contrast, _("contrast adjustment"));
  gtk_widget_set_tooltip_text(g->brightness, _("brightness adjustment"));
  gtk_widget_set_tooltip_text(g->saturation, _("color saturation adjustment"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
