/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

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

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>

#include <cmocka.h>

#include "common/color_picker.h"
#include "common/colorspaces.h"
#include "develop/format.h"
#include "develop/pixelpipe.h"

static void test_large_area_mean_accumulates_rows_in_double(void **state)
{
  (void)state;

  enum { ROW_COUNT = 8192 };
  dt_aligned_pixel_t pixels[ROW_COUNT];
  for(size_t row = 0; row < ROW_COUNT; row++)
  {
    // Interleave large and small row sums so each OpenMP partition encounters
    // small contributions after its float total has grown beyond their precision.
    const float value = row % 2 == 0 ? 1000000.0f : 1.0f;
    for(int channel = 0; channel < 4; channel++) pixels[row][channel] = value;
  }

  const dt_iop_buffer_dsc_t dsc = {
    .channels = 4u,
    .datatype = TYPE_FLOAT,
    .cst = IOP_CS_RGB,
  };
  const dt_iop_roi_t roi = {
    .width = 1,
    .height = ROW_COUNT,
    .scale = 1.0f,
  };
  const int box[4] = { 0, 0, 1, ROW_COUNT };
  lib_colorpicker_stats stats;

  dt_color_picker_helper(&dsc, &pixels[0][0], &roi, box, FALSE, stats,
                         IOP_CS_RGB, IOP_CS_RGB, NULL);

  for(int channel = 0; channel < 4; channel++)
  {
    assert_float_equal(stats[DT_PICK_MEAN][channel], 500000.5f, 0.01f);
    assert_float_equal(stats[DT_PICK_MIN][channel], 1.0f, 0.0f);
    assert_float_equal(stats[DT_PICK_MAX][channel], 1000000.0f, 0.0f);
  }
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_large_area_mean_accumulates_rows_in_double),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
