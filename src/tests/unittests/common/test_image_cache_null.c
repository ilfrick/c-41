#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <cmocka.h>
#include "common/darktable.h"
#include "common/image_cache.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static dt_image_cache_t *previous_cache;

static int setup_null_cache(void **state)
{
  (void)state;
  previous_cache = darktable.image_cache;
  darktable.image_cache = NULL;
  return 0;
}

static int restore_cache(void **state)
{
  (void)state;
  darktable.image_cache = previous_cache;
  return 0;
}

static void test_image_cache_functions_handle_null_cache(void **state)
{
  (void)state;

  const dt_imgid_t imgid = 1;
  assert_null(dt_image_cache_get(imgid, 'r'));
  assert_null(dt_image_cache_testget(imgid, 'r'));

  dt_image_t img = { 0 };
  dt_image_cache_write_release_info(&img, DT_IMAGE_CACHE_RELAXED, "null-cache test");
  dt_image_cache_set_change_timestamp(imgid);
  dt_image_cache_set_change_timestamp_from_image(imgid, imgid);
  dt_image_cache_unset_change_timestamp(imgid);
  dt_image_cache_set_export_timestamp(imgid);
  dt_image_cache_set_print_timestamp(imgid);

}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(test_image_cache_functions_handle_null_cache,
                                    setup_null_cache,
                                    restore_cache),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
