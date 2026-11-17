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
/*
 * CMocka regression tests for legacy highlights parameter conversion.
 * Linux-only: guard pages make old-blob overreads deterministic.
 */
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmocka.h>

#include "iop/highlights.c"

typedef struct test_resources_t
{
  void *mapping;
  size_t mapping_size;
  void *converted;
} test_resources_t;

static int setup(void **state)
{
  test_resources_t *resources = calloc(1, sizeof(*resources));
  if(!resources) return -1;
  *state = resources;
  return 0;
}

static int teardown(void **state)
{
  test_resources_t *resources = *state;
  if(!resources) return 0;

  int status = 0;
  if(resources->mapping && munmap(resources->mapping, resources->mapping_size) != 0)
    status = -1;
  free(resources->converted);
  free(resources);
  return status;
}

static void *guarded_blob(const void *bytes, const size_t size,
                          test_resources_t *resources)
{
  const long page_size = sysconf(_SC_PAGESIZE);
  if(page_size <= 0 || size > (size_t)page_size) return NULL;

  resources->mapping_size = 2 * (size_t)page_size;
  resources->mapping = mmap(NULL, resources->mapping_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if(resources->mapping == MAP_FAILED)
  {
    resources->mapping = NULL;
    return NULL;
  }

  char *guard_page = (char *)resources->mapping + page_size;
  if(mprotect(guard_page, (size_t)page_size, PROT_NONE) != 0)
  {
    munmap(resources->mapping, resources->mapping_size);
    resources->mapping = NULL;
    resources->mapping_size = 0;
    return NULL;
  }

  void *blob = guard_page - size;
  memcpy(blob, bytes, size);
  return blob;
}

static void test_v1_conversion_reads_exact_historical_blob(void **state)
{
  test_resources_t *resources = *state;
  const struct
  {
    dt_iop_highlights_mode_t mode;
    float blendL;
    float blendC;
    float blendh;
  } old_params = { DT_IOP_HIGHLIGHTS_OPPOSED, 0.25f, 0.5f, 0.75f };
  assert_int_equal(sizeof(old_params), 16);

  void *blob = guarded_blob(&old_params, sizeof(old_params), resources);
  assert_non_null(blob);

  int32_t converted_size = 0;
  int converted_version = 0;
  assert_int_equal(legacy_params(NULL, blob, 1, &resources->converted,
                                 &converted_size, &converted_version), 0);
  assert_non_null(resources->converted);
  assert_int_equal(converted_version, 4);
  assert_int_equal(converted_size, (int32_t)sizeof(dt_iop_highlights_params_t));

  const dt_iop_highlights_params_t *params = resources->converted;
  assert_int_equal(params->mode, old_params.mode);
  assert_float_equal(params->blendL, old_params.blendL, 1e-7f);
  assert_float_equal(params->blendC, old_params.blendC, 1e-7f);
  assert_float_equal(params->strength, 0.0f, 1e-7f);
  assert_float_equal(params->clip, 1.0f, 1e-7f);
}

static void test_v2_conversion_reads_exact_historical_blob(void **state)
{
  test_resources_t *resources = *state;
  const struct
  {
    dt_iop_highlights_mode_t mode;
    float blendL;
    float blendC;
    float blendh;
    float clip;
  } old_params = { DT_IOP_HIGHLIGHTS_OPPOSED, 0.25f, 0.5f, 0.75f, 1.25f };
  assert_int_equal(sizeof(old_params), 20);

  void *blob = guarded_blob(&old_params, sizeof(old_params), resources);
  assert_non_null(blob);

  int32_t converted_size = 0;
  int converted_version = 0;
  assert_int_equal(legacy_params(NULL, blob, 2, &resources->converted,
                                 &converted_size, &converted_version), 0);
  assert_non_null(resources->converted);
  assert_int_equal(converted_version, 4);
  assert_int_equal(converted_size, (int32_t)sizeof(dt_iop_highlights_params_t));

  const dt_iop_highlights_params_t *params = resources->converted;
  assert_int_equal(params->mode, old_params.mode);
  assert_float_equal(params->blendL, old_params.blendL, 1e-7f);
  assert_float_equal(params->blendC, old_params.blendC, 1e-7f);
  assert_float_equal(params->strength, 0.0f, 1e-7f);
  assert_float_equal(params->clip, old_params.clip, 1e-7f);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(test_v1_conversion_reads_exact_historical_blob,
                                    setup, teardown),
    cmocka_unit_test_setup_teardown(test_v2_conversion_reads_exact_historical_blob,
                                    setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
