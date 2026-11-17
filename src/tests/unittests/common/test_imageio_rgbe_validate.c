/* Focused tests for RGBE dimensions before allocation and decoding. */
#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cmocka.h>

#include "imageio/imageio_rgbe_validate.h"

static void test_valid_dimensions(void **state)
{
  (void)state;
  size_t pixels = 0;
  assert_true(dt_imageio_rgbe_validate_dimensions(640, 480, &pixels));
  assert_int_equal(pixels, 640 * 480);
}

static void test_parse_valid_dimensions_line(void **state)
{
  (void)state;
  int height = 0;
  int width = 0;
  assert_true(dt_imageio_rgbe_parse_dimensions_line("-Y 480 +X 640", &height, &width));
  assert_int_equal(height, 480);
  assert_int_equal(width, 640);
}

static void test_parse_rejects_out_of_range_dimensions(void **state)
{
  (void)state;
  int height = 17;
  int width = 23;
  char line[64];
  const unsigned long long out_of_range = (unsigned long long)INT_MAX + 1ULL;

  snprintf(line, sizeof(line), "-Y %llu +X 640", out_of_range);
  assert_false(dt_imageio_rgbe_parse_dimensions_line(line, &height, &width));
  assert_int_equal(height, 17);
  assert_int_equal(width, 23);

  snprintf(line, sizeof(line), "-Y 480 +X %llu", out_of_range);
  assert_false(dt_imageio_rgbe_parse_dimensions_line(line, &height, &width));
  assert_int_equal(height, 17);
  assert_int_equal(width, 23);
}

static void test_zero_and_negative_dimensions(void **state)
{
  (void)state;
  size_t pixels = 1;
  assert_false(dt_imageio_rgbe_validate_dimensions(0, 1, &pixels));
  assert_false(dt_imageio_rgbe_validate_dimensions(1, 0, &pixels));
  assert_false(dt_imageio_rgbe_validate_dimensions(-1, 1, &pixels));
  assert_false(dt_imageio_rgbe_validate_dimensions(1, -1, &pixels));
  assert_false(dt_imageio_rgbe_validate_dimensions(1, 1, NULL));
  assert_false(dt_imageio_rgbe_validate_sizes(1, 1, NULL));
}

static void test_int_pixel_count_limit(void **state)
{
  (void)state;
  size_t pixels = 0;
  assert_false(dt_imageio_rgbe_validate_sizes((size_t)INT_MAX + 1, 1, &pixels));
}

static void test_size_product_overflow(void **state)
{
  (void)state;
  size_t pixels = 0;
  assert_false(dt_imageio_rgbe_validate_sizes(SIZE_MAX, 2, &pixels));
}

static void test_float_allocation_size_limit(void **state)
{
  (void)state;
  const size_t max_pixels = SIZE_MAX / (3 * sizeof(float));
  assert_true(dt_imageio_rgbe_float_allocation_fits(max_pixels));
  assert_false(dt_imageio_rgbe_float_allocation_fits(max_pixels + 1));
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_valid_dimensions),
    cmocka_unit_test(test_parse_valid_dimensions_line),
    cmocka_unit_test(test_parse_rejects_out_of_range_dimensions),
    cmocka_unit_test(test_zero_and_negative_dimensions),
    cmocka_unit_test(test_int_pixel_count_limit),
    cmocka_unit_test(test_size_product_overflow),
    cmocka_unit_test(test_float_allocation_size_limit),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
