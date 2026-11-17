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
#include <string.h>

#include <cmocka.h>

#include "iop/hazeremoval.c"

/* The selected element must equal the corresponding value from a full sort. */
static void test_quick_select_first_quantile_with_duplicates(void **state)
{
  (void)state;
  float values[] = { 0.0f, 0.0f, 1.0f };

  _quick_select(values, values, values + 3, FALSE);

  assert_float_equal(values[0], 0.0f, 0.0f);
}

static void test_quick_select_all_quantiles_with_duplicates(void **state)
{
  (void)state;
  const float input[] = { 4.0f, 1.0f, 4.0f, 2.0f, 1.0f, 3.0f };
  const float expected[] = { 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 4.0f };

  for(size_t nth = 0; nth < sizeof(input) / sizeof(input[0]); nth++)
  {
    float values[sizeof(input) / sizeof(input[0])];
    memcpy(values, input, sizeof(input));
    _quick_select(values, values + nth, values + sizeof(values) / sizeof(values[0]), FALSE);

    assert_float_equal(values[nth], expected[nth], 0.0f);
    for(size_t i = 0; i < nth; i++) assert_true(values[i] <= values[nth]);
    for(size_t i = nth + 1; i < sizeof(values) / sizeof(values[0]); i++)
      assert_true(values[i] >= values[nth]);
  }
}

int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_quick_select_first_quantile_with_duplicates),
    cmocka_unit_test(test_quick_select_all_quantiles_with_duplicates),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
