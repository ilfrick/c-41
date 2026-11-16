/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
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

#include "develop/format.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static void test_late_correction_requires_white_balance(void **state)
{
  (void)state;
  dt_iop_buffer_dsc_t dsc = { 0 };

  assert_false(dt_dev_wb_late_correction_applies(TRUE, &dsc));
  dsc.temperature.enabled = TRUE;
  assert_true(dt_dev_wb_late_correction_applies(TRUE, &dsc));
  assert_false(dt_dev_wb_late_correction_applies(FALSE, &dsc));
}

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_late_correction_requires_white_balance),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
