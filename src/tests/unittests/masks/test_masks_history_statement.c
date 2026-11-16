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

#include "common/database.h"
#include "develop/masks.h"

static sqlite3 *test_db;

// Substitute the opaque database accessor so the real writer can use a small
// in-memory database without initializing the full application or GTK.
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
                                "CREATE TABLE masks_history"
                                " (imgid INTEGER, num INTEGER, formid INTEGER, form INTEGER,"
                                " name VARCHAR(256), version INTEGER, points BLOB,"
                                " points_count INTEGER, source BLOB)",
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

static void test_writer_finalizes_statement_without_functions(void **state)
{
  (void)state;
  dt_masks_form_t form = { 0 };
  assert_null(form.functions);
  dt_masks_write_masks_history_item(1, 0, &form);

  assert_null(sqlite3_next_stmt(test_db, NULL));
  assert_int_equal(sqlite3_close(test_db), SQLITE_OK);
  test_db = NULL;
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(test_writer_finalizes_statement_without_functions,
                                    setup_database,
                                    teardown_database),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
