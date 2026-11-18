#include <stddef.h>
#include <setjmp.h>
#include <stdarg.h>

#include "develop/pixelpipe_hb.h"
#include "develop/pixelpipe_cache.h"
#include "common/iop_profile.h"
#include <cmocka.h>

static void test_export_profile_changes_cache_hash(void **state)
{
  (void)state;
  dt_dev_pixelpipe_t pipe = { 0 };
  dt_iop_order_iccprofile_info_t profile_a = { 0 };
  dt_iop_order_iccprofile_info_t profile_b = { 0 };

  const dt_hash_t empty_hash = dt_dev_pixelpipe_cache_hash(NULL, &pipe, 0);
  pipe.export_profile_info = &profile_a;
  const dt_hash_t profile_a_hash = dt_dev_pixelpipe_cache_hash(NULL, &pipe, 0);
  pipe.export_profile_info = &profile_b;
  const dt_hash_t profile_b_hash = dt_dev_pixelpipe_cache_hash(NULL, &pipe, 0);

  assert_int_not_equal(empty_hash, profile_a_hash);
  assert_int_not_equal(profile_a_hash, profile_b_hash);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_export_profile_changes_cache_hash),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
