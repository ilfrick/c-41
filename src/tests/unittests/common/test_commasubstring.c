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

#include <cmocka.h>

#include "common/utility.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static void test_null_list_after_early_match_returns_false(void **state)
{
  (void)state;

  /* A match exits early from strtok(); the NULL-list guard must not continue
   * strtok's saved pointer into the freed list from the previous call. */
  assert_true(dt_str_commasubstring("first,second", "first"));
  assert_false(dt_str_commasubstring(NULL, "second"));
}

static void test_null_search_returns_false(void **state)
{
  (void)state;
  assert_false(dt_str_commasubstring("first,second", NULL));
}

static void test_search_matches_complete_tokens(void **state)
{
  (void)state;
  assert_true(dt_str_commasubstring("first,second", "second"));
  assert_false(dt_str_commasubstring("first,second", "cond"));
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_null_list_after_early_match_returns_false),
    cmocka_unit_test(test_null_search_returns_false),
    cmocka_unit_test(test_search_matches_complete_tokens),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
