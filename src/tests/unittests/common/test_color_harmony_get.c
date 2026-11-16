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
#include <sqlite3.h>

#include "common/color_harmony.h"
#include "common/database.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static sqlite3 *test_db;

// The database wrapper's concrete struct is intentionally opaque; substitute its
// accessor so this test can exercise the real color-harmony implementation using
// a small in-memory SQLite database.
sqlite3 *dt_database_get(const struct dt_database_t *db)
{
  (void)db;
  return test_db;
}

static int setup_database(void **state)
{
  (void)state;
  assert_int_equal(sqlite3_open(":memory:", &test_db), SQLITE_OK);
  assert_int_equal(sqlite3_exec(test_db,
                                "CREATE TABLE harmony_guide"
                                " (imgid INTEGER PRIMARY KEY, type INTEGER, rotation INTEGER, width INTEGER);"
                                " INSERT INTO harmony_guide VALUES (7, 3, 45, 2);",
                                NULL, NULL, NULL), SQLITE_OK);
  return 0;
}

static int teardown_database(void **state)
{
  (void)state;
  if(test_db)
  {
    const int result = sqlite3_close(test_db);
    test_db = NULL;
    assert_int_equal(result, SQLITE_OK);
  }
  return 0;
}

static void test_get_present_imgid_finalizes_statement(void **state)
{
  (void)state;
  dt_color_harmony_guide_t layout = {
    .type = DT_COLOR_HARMONY_NONE,
    .rotation = 0,
    .width = DT_COLOR_HARMONY_WIDTH_NORMAL,
    .custom_n = 2,
    .custom_angles = { 0.2f, 0.8f, 0.f, 0.f },
  };

  assert_true(dt_color_harmony_get(7, &layout));
  assert_int_equal(layout.type, 3);
  assert_int_equal(layout.rotation, 45);
  assert_int_equal(layout.width, 2);
  assert_int_equal(layout.custom_n, 0);
  for(size_t i = 0; i < G_N_ELEMENTS(layout.custom_angles); i++)
    assert_float_equal(layout.custom_angles[i], 0.f, 0.f);
  assert_null(sqlite3_next_stmt(test_db, NULL));
  assert_int_equal(sqlite3_close(test_db), SQLITE_OK);
  test_db = NULL;
}

static void test_get_missing_imgid_finalizes_statement(void **state)
{
  (void)state;
  dt_color_harmony_guide_t layout = {
    .type = DT_COLOR_HARMONY_TRIAD,
    .rotation = 123,
    .width = DT_COLOR_HARMONY_WIDTH_LARGE,
    .custom_n = 1,
    .custom_angles = { 0.4f, 0.f, 0.f, 0.f },
  };

  assert_false(dt_color_harmony_get(99, &layout));
  assert_int_equal(layout.type, DT_COLOR_HARMONY_TRIAD);
  assert_int_equal(layout.rotation, 123);
  assert_int_equal(layout.width, DT_COLOR_HARMONY_WIDTH_LARGE);
  assert_int_equal(layout.custom_n, 1);
  assert_float_equal(layout.custom_angles[0], 0.4f, 0.f);
  assert_null(sqlite3_next_stmt(test_db, NULL));
  assert_int_equal(sqlite3_close(test_db), SQLITE_OK);
  test_db = NULL;
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(test_get_present_imgid_finalizes_statement,
                                    setup_database,
                                    teardown_database),
    cmocka_unit_test_setup_teardown(test_get_missing_imgid_finalizes_statement,
                                    setup_database,
                                    teardown_database),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
