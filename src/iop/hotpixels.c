/*
    This file is part of darktable,
    Copyright (C) 2011-2026 darktable developers.

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
#include "common/imagebuf.h"
#include "control/control.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "dtgtk/resetlabel.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include "rust_ffi/darkroom_core.h"

#include <gtk/gtk.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_hotpixels_params_t)

typedef struct dt_iop_hotpixels_params_t
{
  float strength;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.25
  float threshold; // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.05
  gboolean markfixed;  // $DEFAULT: FALSE $DESCRIPTION: "mark fixed pixels"
  gboolean permissive; // $DEFAULT: FALSE $DESCRIPTION: "detect by 3 neighbors"
} dt_iop_hotpixels_params_t;

typedef struct dt_iop_hotpixels_gui_data_t
{
  GtkWidget *threshold, *strength;
  GtkToggleButton *markfixed;
  GtkToggleButton *permissive;
  GtkLabel *message;
  int pixels_fixed;
} dt_iop_hotpixels_gui_data_t;

typedef struct dt_iop_hotpixels_data_t
{
  uint32_t filters;
  float threshold;
  float multiplier;
  gboolean permissive;
  gboolean markfixed;
  gboolean monochrome;
  gboolean pure_monochrome;
} dt_iop_hotpixels_data_t;


const char *name()
{
  return _("hot pixels");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("remove abnormally bright pixels\n"
                                        "by dampening them with neighbors"),
                                      _("corrective"),
                                      _("linear, raw, scene-referred"),
                                      _("reconstruction, raw"),
                                      _("linear, raw, scene-referred"));
}


int default_group()
{
  return IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ONE_INSTANCE;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RAW;
}

/* Detect hot sensor pixels based on the 4 surrounding sites. Pixels
 * having 3 or 4 (depending on permissive setting) surrounding pixels that
 * than value*multiplier are considered "hot", and are replaced by the maximum of
 * the neighbour pixels. The permissive variant allows for
 * correcting pairs of hot pixels in adjacent sites. Replacement using
 * the maximum produces fewer artifacts when inadvertently replacing
 * non-hot pixels.
 * This is the Bayer sensor variant. */
static int _process_bayer(const dt_iop_hotpixels_data_t *data,
                          const void *const ivoid,
                          void *const ovoid,
                          const dt_iop_roi_t *const roi_out)
{
  const int min_neighbours = data->permissive ? 3 : 4;
  return darkroom_hotpixels_bayer((const float *)ivoid, (float *)ovoid,
                                  (size_t)roi_out->width,
                                  (size_t)roi_out->height,
                                  data->threshold,
                                  data->multiplier,
                                  min_neighbours,
                                  data->markfixed ? 1 : 0);
}

/* This is the monochrome sensor variant. */
static int _process_monochrome(const dt_iop_hotpixels_data_t *data,
                              const void *const ivoid,
                              void *const ovoid,
                              const dt_iop_roi_t *const roi_out,
                              const int planes)
{
  const int min_neighbours = data->permissive ? 3 : 4;
  return darkroom_hotpixels_monochrome((const float *)ivoid, (float *)ovoid,
                                       (size_t)roi_out->width,
                                       (size_t)roi_out->height,
                                       (size_t)planes,
                                       data->threshold,
                                       data->multiplier,
                                       min_neighbours,
                                       data->markfixed ? 1 : 0);
}

/* X-Trans sensor equivalent of _process_bayer(). */
static int _process_xtrans(const dt_iop_hotpixels_data_t *data,
                           const void *const ivoid,
                           void *const ovoid,
                           const dt_iop_roi_t *const roi_out,
                           const uint8_t (*const xtrans)[6])
{
  const int min_neighbours = data->permissive ? 3 : 4;
  return darkroom_hotpixels_xtrans((const float *)ivoid, (float *)ovoid,
                                   (size_t)roi_out->width,
                                   (size_t)roi_out->height,
                                   (const unsigned char *)xtrans,
                                   data->threshold,
                                   data->multiplier,
                                   min_neighbours,
                                   data->markfixed ? 1 : 0);
}

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_hotpixels_gui_data_t *g = self->gui_data;
  const dt_iop_hotpixels_data_t *data = piece->data;

  // The processing loop should output only a few pixels, so just copy everything first
  const int planes = data->pure_monochrome ? 4 : 1;
  dt_iop_image_copy_by_size(ovoid, ivoid, roi_out->width, roi_out->height, planes);

  int fixed = 0;

  if(data->monochrome || data->pure_monochrome)
  {
    fixed = _process_monochrome(data, ivoid, ovoid, roi_out, planes);
  }
  else if(piece->pipe->dsc.filters == 9u)
  {
    fixed = _process_xtrans(data, ivoid, ovoid, roi_out, piece->xtrans);
  }
  else
  {
    fixed = _process_bayer(data, ivoid, ovoid, roi_out);
  }

  if(g && self->dev->gui_attached && dt_pipe_is_full(piece->pipe))
  {
    dt_iop_gui_enter_critical_section(self);
    g->pixels_fixed = fixed;
    dt_iop_gui_leave_critical_section(self);
  }
}

void reload_defaults(dt_iop_module_t *self)
{
  const dt_image_t *img = &self->dev->image_storage;

  const gboolean monoraw = (img->flags & DT_IMAGE_S_RAW) && (img->flags & DT_IMAGE_MONOCHROME);
  const gboolean supported = dt_image_is_raw(img) || monoraw;
  // can't be switched on for non-raw images:
  self->hide_enable_button = !supported;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_hotpixels_params_t *p = (dt_iop_hotpixels_params_t *)params;
  dt_iop_hotpixels_data_t *d = piece->data;
  d->filters = piece->pipe->dsc.filters;
  d->multiplier = p->strength / 2.0;
  d->threshold = p->threshold;
  d->permissive = p->permissive;
  d->markfixed = p->markfixed && (!(dt_pipe_is_thumb(pipe) || dt_pipe_is_export(pipe)));

  const dt_image_t *img = &pipe->image;
  const gboolean monoraw = (img->flags & DT_IMAGE_S_RAW) && (img->flags & DT_IMAGE_MONOCHROME);
  const gboolean supported = dt_image_is_raw(img) || monoraw;
  d->monochrome = img->flags & DT_IMAGE_MONOCHROME_BAYER;
  d->pure_monochrome = monoraw;
  if(!supported || p->strength == 0.0) piece->enabled = FALSE;
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_hotpixels_data_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}


void gui_update(dt_iop_module_t *self)
{
  dt_iop_hotpixels_gui_data_t *g = self->gui_data;
  dt_iop_hotpixels_params_t *p = self->params;
  gtk_toggle_button_set_active(g->markfixed, p->markfixed);
  gtk_toggle_button_set_active(g->permissive, p->permissive);
  dt_iop_gui_enter_critical_section(self);
  g->pixels_fixed = -1;
  dt_iop_gui_leave_critical_section(self);
  gtk_label_set_text(g->message, "");

  const dt_image_t *img = &self->dev->image_storage;
  const gboolean monoraw = (img->flags & DT_IMAGE_S_RAW) && (img->flags & DT_IMAGE_MONOCHROME);
  const gboolean supported = dt_image_is_raw(img) || monoraw;
  // can't be switched on for non-raw images:
  self->hide_enable_button = !supported;

  gtk_stack_set_visible_child_name(GTK_STACK(self->widget), self->hide_enable_button ? "non_raw" : "raw");
}

static gboolean _draw(GtkWidget *widget, cairo_t *cr, dt_iop_module_t *self)
{
  dt_iop_hotpixels_gui_data_t *g = self->gui_data;
  DT_GUARD_GUI_UPDATE(FALSE);

  dt_iop_gui_enter_critical_section(self);
  const int pixels_fixed = g->pixels_fixed;
  g->pixels_fixed = -1;
  dt_iop_gui_leave_critical_section(self);

  if(pixels_fixed < 0) return FALSE;

  char *str = g_strdup_printf(ngettext("fixed %d pixel", "fixed %d pixels", pixels_fixed), pixels_fixed);

  DT_ENTER_GUI_UPDATE();
  gtk_label_set_text(g->message, str);
  DT_LEAVE_GUI_UPDATE();

  g_free(str);

  return FALSE;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_hotpixels_gui_data_t *g = IOP_GUI_ALLOC(hotpixels);

  g->pixels_fixed = -1;

  GtkWidget *box_raw = self->widget = dt_gui_vbox();
  g_signal_connect(G_OBJECT(box_raw), "draw", G_CALLBACK(_draw), self);

  g->threshold = dt_bauhaus_slider_from_params(self, N_("threshold"));
  dt_bauhaus_slider_set_digits(g->threshold, 4);
  gtk_widget_set_tooltip_text(g->threshold, _("lower threshold for hot pixel"));

  g->strength = dt_bauhaus_slider_from_params(self, N_("strength"));
  dt_bauhaus_slider_set_digits(g->strength, 4);
  gtk_widget_set_tooltip_text(g->strength, _("strength of hot pixel correction"));

  // 3 neighbours
  g->permissive = GTK_TOGGLE_BUTTON(dt_bauhaus_toggle_from_params(self, "permissive"));

  // mark fixed pixels
  GtkWidget *hbox = self->widget = dt_gui_hbox();
  g->markfixed = GTK_TOGGLE_BUTTON(dt_bauhaus_toggle_from_params(self, "markfixed"));
  g->message = GTK_LABEL(gtk_label_new("")); // This gets filled in by process
  dt_gui_box_add(hbox, g->message);
  dt_gui_box_add(box_raw, hbox);

  // start building top level widget
  self->widget = gtk_stack_new();
  gtk_stack_set_homogeneous(GTK_STACK(self->widget), FALSE);

  GtkWidget *label_non_raw = dt_ui_label_new(_("hot pixel correction\nonly works for raw images."));

  gtk_stack_add_named(GTK_STACK(self->widget), label_non_raw, "non_raw");
  gtk_stack_add_named(GTK_STACK(self->widget), box_raw, "raw");
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

