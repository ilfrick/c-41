/*
 * CMocka regression coverage for historical denoiseprofile layouts.
 * Linux guard pages detect legacy source overreads and v11 destination overwrites.
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

#include "iop/denoiseprofile.c"

/* Independent, exact historical input layouts. */
typedef struct test_v8_t
{
  float radius, nbhood, strength, shadows, bias, scattering, central_pixel_weight, overshooting;
  float a[3], b[3];
  dt_iop_denoiseprofile_mode_t mode;
  float x[DT_DENOISE_PROFILE_NONE_V9][DT_IOP_DENOISE_PROFILE_V8_BANDS];
  float y[DT_DENOISE_PROFILE_NONE_V9][DT_IOP_DENOISE_PROFILE_V8_BANDS];
  gboolean wb_adaptive_anscombe, fix_anscombe_and_nlmeans_norm, use_new_vst;
} test_v8_t;
typedef struct test_v9_t
{
  float radius, nbhood, strength, shadows, bias, scattering, central_pixel_weight, overshooting;
  float a[3], b[3];
  dt_iop_denoiseprofile_mode_t mode;
  float x[DT_DENOISE_PROFILE_NONE_V9][DT_IOP_DENOISE_PROFILE_BANDS];
  float y[DT_DENOISE_PROFILE_NONE_V9][DT_IOP_DENOISE_PROFILE_BANDS];
  gboolean wb_adaptive_anscombe, fix_anscombe_and_nlmeans_norm, use_new_vst;
} test_v9_t;
typedef struct test_v10_t
{
  float radius, nbhood, strength, shadows, bias, scattering, central_pixel_weight, overshooting;
  float a[3], b[3];
  dt_iop_denoiseprofile_mode_t mode;
  float x[DT_DENOISE_PROFILE_NONE][DT_IOP_DENOISE_PROFILE_BANDS];
  float y[DT_DENOISE_PROFILE_NONE][DT_IOP_DENOISE_PROFILE_BANDS];
  gboolean wb_adaptive_anscombe, fix_anscombe_and_nlmeans_norm, use_new_vst;
  dt_iop_denoiseprofile_wavelet_mode_t wavelet_color_mode;
} test_v10_t;
typedef struct test_v11_t
{
  float radius, nbhood, strength, shadows, bias, scattering, central_pixel_weight, overshooting;
  float a[3], b[3];
  dt_iop_denoiseprofile_mode_t mode;
  float x[DT_DENOISE_PROFILE_NONE][DT_IOP_DENOISE_PROFILE_BANDS];
  float y[DT_DENOISE_PROFILE_NONE][DT_IOP_DENOISE_PROFILE_BANDS];
  gboolean wb_adaptive_anscombe, fix_anscombe_and_nlmeans_norm, use_new_vst;
  dt_iop_denoiseprofile_wavelet_mode_t wavelet_color_mode;
} test_v11_t;

typedef struct test_resources_t
{
  void *maps[2];
  size_t map_sizes[2];
  size_t map_count;
  void *converted;
} test_resources_t;

static int setup(void **state)
{
  *state = calloc(1, sizeof(test_resources_t));
  return *state ? 0 : -1;
}

static int teardown(void **state)
{
  test_resources_t *r = *state;
  if(!r) return 0;
  free(r->converted);
  int result = 0;
  for(size_t i = 0; i < r->map_count; i++)
    if(munmap(r->maps[i], r->map_sizes[i]) != 0) result = -1;
  free(r);
  return result;
}

static void *guarded_blob(const void *bytes, const size_t size, test_resources_t *r)
{
  const long page = sysconf(_SC_PAGESIZE);
  if(page <= 0 || size > (size_t)page || r->map_count >= 2) return NULL;
  const size_t map_size = 2 * (size_t)page;
  void *map = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if(map == MAP_FAILED) return NULL;
  const size_t index = r->map_count++;
  r->maps[index] = map;
  r->map_sizes[index] = map_size;
  void *guard = (char *)map + page;
  if(mprotect(guard, (size_t)page, PROT_NONE) != 0) return NULL;
  void *blob = (char *)guard - size;
  memcpy(blob, bytes, size);
  return blob;
}

static void check_conversion(const int version, const void *old, const float expected_curve,
                             void **state)
{
  test_resources_t *r = *state;
  dt_iop_module_t self = { 0 };
  dt_iop_denoiseprofile_params_t defaults = { 0 };
  defaults.radius = 1.0f;
  self.default_params = &defaults;

  const long page = sysconf(_SC_PAGESIZE);
  assert_true(page > 0);
  assert_true(sizeof(test_v11_t) < (size_t)page);
  void *map = mmap(NULL, 2 * (size_t)page, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert_true(map != MAP_FAILED);
  const size_t index = r->map_count++;
  r->maps[index] = map;
  r->map_sizes[index] = 2 * (size_t)page;
  assert_int_equal(mprotect((char *)map + page, (size_t)page, PROT_NONE), 0);
  test_v11_t *out = (test_v11_t *)((char *)map + page - sizeof(test_v11_t));
  memset(out, 0xa5, sizeof(*out));

  assert_int_equal(legacy_params_to11(&self, old, version, out, 11), 0);
  assert_float_equal(out->radius, 2.25f, 1e-7f);
  assert_float_equal(out->strength, 3.5f, 1e-7f);
  assert_float_equal(out->a[1], 12.0f, 1e-7f);
  assert_float_equal(out->x[0][DT_IOP_DENOISE_PROFILE_BANDS - 1], expected_curve, 1e-6f);
  assert_float_equal(out->y[0][DT_IOP_DENOISE_PROFILE_BANDS - 1], 0.73f, 1e-6f);
  assert_int_equal(out->wavelet_color_mode, MODE_RGB);

  int32_t output_size = 0;
  int output_version = 0;
  assert_int_equal(legacy_params(&self, old, version, &r->converted, &output_size, &output_version), 0);
  assert_int_equal(output_version, 11);
  assert_int_equal(output_size, (int32_t)sizeof(test_v11_t));
}

static void test_v8_layout(void **state)
{
  test_v8_t p = { 0 };
  p.radius = 2.25f; p.strength = 3.5f; p.a[1] = 12.0f;
  p.mode = MODE_WAVELETS;
  p.y[0][DT_IOP_DENOISE_PROFILE_V8_BANDS - 1] = 0.73f;
  const void *blob = guarded_blob(&p, sizeof(p), *state);
  assert_non_null(blob);
  check_conversion(8, blob, 1.0f, state);
}

static void test_v9_layout(void **state)
{
  test_v9_t p = { 0 };
  p.radius = 2.25f; p.strength = 3.5f; p.a[1] = 12.0f;
  p.mode = MODE_WAVELETS;
  p.x[0][DT_IOP_DENOISE_PROFILE_BANDS - 1] = 0.91f;
  p.y[0][DT_IOP_DENOISE_PROFILE_BANDS - 1] = 0.73f;
  const void *blob = guarded_blob(&p, sizeof(p), *state);
  assert_non_null(blob);
  check_conversion(9, blob, 0.91f, state);
}

static void test_v10_layout_and_destination_boundary(void **state)
{
  test_v10_t p = { 0 };
  p.radius = 2.25f; p.strength = 3.5f; p.a[1] = 12.0f;
  p.mode = MODE_WAVELETS;
  p.x[0][DT_IOP_DENOISE_PROFILE_BANDS - 1] = 0.91f;
  p.y[0][DT_IOP_DENOISE_PROFILE_BANDS - 1] = 0.73f;
  const void *blob = guarded_blob(&p, sizeof(p), *state);
  assert_non_null(blob);
  check_conversion(10, blob, 0.91f, state);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(test_v8_layout, setup, teardown),
    cmocka_unit_test_setup_teardown(test_v9_layout, setup, teardown),
    cmocka_unit_test_setup_teardown(test_v10_layout_and_destination_boundary, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
