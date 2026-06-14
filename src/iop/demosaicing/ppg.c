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

DT_OMP_DECLARE_SIMD(aligned(in, out:64))
static void demosaic_ppg(float *const out,
                         const float *const in,
                         const int width,
                         const int height,
                         const uint32_t filters,
                         const float thrs,
                         const int margin)
{
  // preliminary border interpolate for outermost 3 pixels
  float sum[8];
  for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
      if(i == 3 && j >= 3 && j < height - 3) i = width - 3;
      if(i == width) break;
      memset(sum, 0, sizeof(float) * 8);
      for(int y = j - 1; y != j + 2; y++)
        for(int x = i - 1; x != i + 2; x++)
        {
          if((y >= 0) && (x >= 0) && (y < height) && (x < width))
          {
            const int f = FC(y, x, filters);
            sum[f] += in[(size_t)y * width + x];
            sum[f + 4]++;
          }
        }
      const int f = FC(j, i, filters);
      for(int c = 0; c < 3; c++)
      {
        if(c != f && sum[c + 4] > 0.0f)
          out[4 * ((size_t)j * width + i) + c] = fmaxf(0.0f, sum[c] / sum[c + 4]);
        else
          out[4 * ((size_t)j * width + i) + c]
              = fmaxf(0.0f, in[(size_t)j * width + i]);
      }
    }
  const gboolean median = thrs > 0.0f;
  const float *input = in;
  if(median)
  {
    float *med_in = dt_alloc_align_float((size_t)height * width);
    pre_median(med_in, in, width, height, filters, 1, thrs);
    input = med_in;
  }

  // green interpolation (ring-aware, ring = margin+3) -- Rust FFI; the
  // original-input cursor switch after the ring skip is preserved inside
  darkroom_demosaic_ppg_green(out, input, in, width, height, filters, margin);

  // red/blue interpolation, in-place on out -- Rust FFI
  darkroom_demosaic_ppg_redblue(out, width, height, filters, margin);

  // _mm_sfence();
  if(median) dt_free_align((float *)input);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

