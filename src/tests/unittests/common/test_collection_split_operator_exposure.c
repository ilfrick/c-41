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
*/

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>

#include <cmocka.h>

#include "common/collection.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static void test_inverse_exposure_range(void **state)
{
  (void)state;

  char *number1 = NULL;
  char *number2 = NULL;
  char *operator = NULL;
  dt_collection_split_operator_exposure("[1/2;1/4]", &number1, &number2, &operator);

  const gboolean valid = g_strcmp0(number1, "1.0/2") == 0
                      && g_strcmp0(number2, "1.0/4") == 0
                      && g_strcmp0(operator, "[]") == 0;
  g_free(number1);
  g_free(number2);
  g_free(operator);

  assert_true(valid);
}

static void test_inverse_exposure_comparison(void **state)
{
  (void)state;

  char *number1 = NULL;
  char *number2 = NULL;
  char *operator = NULL;
  dt_collection_split_operator_exposure(">1/3", &number1, &number2, &operator);

  const gboolean valid = g_strcmp0(number1, "1.0/3") == 0
                      && number2 == NULL
                      && g_strcmp0(operator, ">") == 0;
  g_free(number1);
  g_free(number2);
  g_free(operator);

  assert_true(valid);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_inverse_exposure_range),
    cmocka_unit_test(test_inverse_exposure_comparison),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
