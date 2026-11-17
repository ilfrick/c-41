/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable. If not, see <http://www.gnu.org/licenses/>.

    Regression test for the deterministic two-pass capture-sharpen mask.
    Compare the 21-cell stencil against the prior single-threaded scatter
    semantics: each clipped/dark interior sample clears its 21-cell footprint,
    and the outer two-pixel border remains zero.
*/
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>

#include <cmocka.h>

#include "capture_mask_shared.h"

static void _old_scatter_reference(const float *raw, float *out,
                                   const int width, const int height)
{
  for(int row = 0; row < height; row++)
  {
    for(int col = 0; col < width; col++)
    {
      const int k = row * width + col;
      const int safe = row > 1 && col > 1 && row < height - 2 && col < width - 2;
      out[k] = safe ? 1.0f : 0.0f;
    }
  }

  for(int row = 2; row < height - 2; row++)
  {
    for(int col = 2; col < width - 2; col++)
    {
      const int k = row * width + col;
      if(raw[k] != 0.0f) continue;

      for(int dy = -2; dy <= 2; dy++)
      {
        const int dx_max = (dy == -2 || dy == 2) ? 1 : 2;
        for(int dx = -dx_max; dx <= dx_max; dx++)
          out[(row + dy) * width + col + dx] = 0.0f;
      }
    }
  }
}

static void _two_pass_reference(const float *raw, float *out,
                                const int width, const int height)
{
  for(int row = 0; row < height; row++)
  {
    for(int col = 0; col < width; col++)
    {
      const int k = row * width + col;
      const int interior = C41_CAPTURE_MASK_INTERIOR_21(col, row, width, height);
      out[k] = interior && C41_CAPTURE_MASK_SAFE_21(raw + k, width) ? 1.0f : 0.0f;
    }
  }
}

static void _assert_equivalent(const float *raw, const int width, const int height)
{
  float old_out[9 * 8];
  float new_out[9 * 8];
  _old_scatter_reference(raw, old_out, width, height);
  _two_pass_reference(raw, new_out, width, height);

  for(int k = 0; k < width * height; k++)
    assert_int_equal((int)old_out[k], (int)new_out[k]);
}

static void test_safe_mask_matches_scatter_reference(void **state)
{
  (void)state;
  enum { width = 9, height = 8, pixels = width * height };
  float raw[pixels];

  for(int k = 0; k < pixels; k++) raw[k] = 1.0f;
  _assert_equivalent(raw, width, height);

  /* Move one bad center over every interior position, covering every stencil
   * offset against both the zero border and interior output pixels. */
  for(int row = 2; row < height - 2; row++)
  {
    for(int col = 2; col < width - 2; col++)
    {
      const int k = row * width + col;
      raw[k] = 0.0f;
      _assert_equivalent(raw, width, height);
      raw[k] = 1.0f;
    }
  }

  /* Also cover every pair of overlapping bad-center footprints. */
  for(int first = 2 * width + 2; first < (height - 2) * width - 2; first++)
  {
    const int first_row = first / width;
    const int first_col = first % width;
    if(first_row < 2 || first_row >= height - 2 || first_col < 2 || first_col >= width - 2) continue;

    for(int second = first + 1; second < (height - 2) * width - 2; second++)
    {
      const int second_row = second / width;
      const int second_col = second % width;
      if(second_row < 2 || second_row >= height - 2 || second_col < 2 || second_col >= width - 2) continue;

      raw[first] = raw[second] = 0.0f;
      _assert_equivalent(raw, width, height);
      raw[first] = raw[second] = 1.0f;
    }
  }
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_safe_mask_matches_scatter_reference),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
