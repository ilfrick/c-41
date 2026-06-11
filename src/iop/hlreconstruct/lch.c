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

static void process_lch_bayer(dt_iop_module_t *self,
                              dt_dev_pixelpipe_iop_t *piece,
                              const void *const ivoid,
                              void *const ovoid,
                              const dt_iop_roi_t *const roi_in,
                              const dt_iop_roi_t *const roi_out,
                              const float clip)
{
  // LCH highlight reconstruction, Bayer (Rust FFI).
  darkroom_highlights_lch_bayer((const float *)ivoid, (float *)ovoid,
                                roi_out->width, roi_out->height,
                                piece->filters, clip);
}

static void process_lch_xtrans(dt_iop_module_t *self,
                               dt_dev_pixelpipe_iop_t *piece,
                               const void *const ivoid,
                               void *const ovoid,
                               const dt_iop_roi_t *const roi_in,
                               const dt_iop_roi_t *const roi_out,
                               const float clip)
{
  // LCH highlight reconstruction, X-Trans (Rust FFI). `in` rows are strided
  // by roi_in->width; `out` is a roi_out-sized plane.
  darkroom_highlights_lch_xtrans((const float *)ivoid, (float *)ovoid,
                                 roi_out->width, roi_out->height, roi_in->width,
                                 (const uint8_t *)piece->xtrans, clip);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
