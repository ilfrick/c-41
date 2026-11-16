#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cmocka.h>
#include "common/dng_opcode.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static void put32(uint8_t *p, uint32_t v)
{
  v = GUINT32_TO_BE(v);
  memcpy(p, &v, sizeof(v));
}

static void put64(uint8_t *p, uint64_t v)
{
  v = GUINT64_TO_BE(v);
  memcpy(p, &v, sizeof(v));
}

static void list3(uint8_t *b, uint32_t id, uint32_t size)
{
  memset(b, 0, 128);
  put32(b, 1);
  put32(b + 4, id);
  put32(b + 16, size);
}

static void test_short_list_and_opcode_headers(void **state)
{
  dt_image_t img = { 0 };
  uint8_t b[128] = { 0 };
  dt_dng_opcode_process_opcode_list_2(b, 3, &img);
  dt_dng_opcode_process_opcode_list_3(b, 0, &img);
  put32(b, 1);
  dt_dng_opcode_process_opcode_list_2(b, 4, &img);
  dt_dng_opcode_process_opcode_list_3(b, 4, &img);
}

static void test_truncated_lengths_and_opcode_sizes(void **state)
{
  dt_image_t img = { 0 };
  uint8_t b[128];
  list3(b, 99, UINT32_MAX);
  dt_dng_opcode_process_opcode_list_3(b, sizeof(b), &img);
  list3(b, 99, UINT32_MAX);
  dt_dng_opcode_process_opcode_list_2(b, sizeof(b), &img);
  list3(b, 1, 4);
  put32(b + 20, 1);
  dt_dng_opcode_process_opcode_list_3(b, 24, &img);
  list3(b, 3, 55);
  dt_dng_opcode_process_opcode_list_3(b, sizeof(b), &img);
}

static void test_valid_warp_and_vignette(void **state)
{
  uint8_t b[128];
  dt_image_t warp_img = { 0 };
  list3(b, 1, 68);
  put32(b + 20, 1);
  dt_dng_opcode_process_opcode_list_3(b, 88, &warp_img);
  assert_true(warp_img.exif_correction_data.dng.has_warp);

  dt_image_t vignette_img = { 0 };
  list3(b, 3, 56);
  dt_dng_opcode_process_opcode_list_3(b, 76, &vignette_img);
  assert_true(vignette_img.exif_correction_data.dng.has_vignette);
}

static void test_oversized_fixed_payloads(void **state)
{
  uint8_t b[128];
  dt_image_t warp_img = { 0 };
  list3(b, 1, 69);
  put32(b + 20, 1);
  dt_dng_opcode_process_opcode_list_3(b, 89, &warp_img);
  assert_false(warp_img.exif_correction_data.dng.has_warp);

  dt_image_t vignette_img = { 0 };
  list3(b, 3, 57);
  dt_dng_opcode_process_opcode_list_3(b, 77, &vignette_img);
  assert_false(vignette_img.exif_correction_data.dng.has_vignette);
}

static void test_gainmap_sizes_and_dimensions(void **state)
{
  dt_image_t img = { 0 };
  uint8_t b[128] = { 0 };
  put32(b, 1);
  put32(b + 4, 9);
  put32(b + 16, 75);
  dt_dng_opcode_process_opcode_list_2(b, 95, &img);
  put32(b + 16, 77);
  dt_dng_opcode_process_opcode_list_2(b, 97, &img);
  put32(b + 16, 80);
  put32(b + 4 + 16 + 32, 1);
  put32(b + 4 + 16 + 36, 1);
  put32(b + 4 + 16 + 72, 2);
  dt_dng_opcode_process_opcode_list_2(b, 100, &img);
  put32(b + 4 + 16 + 32, UINT32_MAX);
  put32(b + 4 + 16 + 36, UINT32_MAX);
  dt_dng_opcode_process_opcode_list_2(b, 100, &img);
  assert_null(img.dng_gain_maps);
}

static void test_valid_gainmap(void **state)
{
  dt_image_t img = { 0 };
  uint8_t b[100] = { 0 };
  put32(b, 1);
  put32(b + 4, 9);
  put32(b + 16, 80);
  put32(b + 20 + 8, 10);  // bottom
  put32(b + 20 + 12, 10); // right
  put32(b + 20 + 20, 1);  // planes
  put32(b + 20 + 24, 1);  // row pitch
  put32(b + 20 + 28, 1);  // column pitch
  put32(b + 20 + 32, 1);  // vertical points
  put32(b + 20 + 36, 1);  // horizontal points
  put64(b + 20 + 40, 0x3ff0000000000000); // vertical spacing = 1.0
  put64(b + 20 + 48, 0x3ff0000000000000); // horizontal spacing = 1.0
  put32(b + 20 + 72, 1);  // map planes
  put32(b + 20 + 76, 0x3f800000);
  dt_dng_opcode_process_opcode_list_2(b, sizeof(b), &img);
  assert_non_null(img.dng_gain_maps);
  dt_dng_gain_map_t *gm = img.dng_gain_maps->data;
  assert_int_equal(gm->map_points_v, 1);
  assert_int_equal(gm->map_points_h, 1);
  assert_float_equal(gm->map_gain[0], 1.0f, 1e-6f);
  g_list_free_full(img.dng_gain_maps, g_free);
}

int main(int argc, char *argv[])
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_short_list_and_opcode_headers),
    cmocka_unit_test(test_truncated_lengths_and_opcode_sizes),
    cmocka_unit_test(test_valid_warp_and_vignette),
    cmocka_unit_test(test_oversized_fixed_payloads),
    cmocka_unit_test(test_gainmap_sizes_and_dimensions),
    cmocka_unit_test(test_valid_gainmap),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
