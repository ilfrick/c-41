/* Reproducible pinned-darktable CPU golden generator for C-41.
 * This test includes the pinned IOP source to exercise its real init/commit/
 * ROI/process functions; see the adjacent README for build/run instructions.
 */
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "iop/contrastntexture.c"

#include <cmocka.h>

#define WIDTH 256
#define HEIGHT 256

static void test_golden(void **state)
{
  (void)state;
  const char *path = getenv("C41_GOLDEN_OUTPUT");
  const char *scale_text = getenv("C41_GOLDEN_ISCALE");
  assert_non_null(path);
  assert_non_null(scale_text);
  char *end = NULL;
  const float iscale = strtof(scale_text, &end);
  assert_non_null(end);
  assert_true(*end == '\0');
  assert_true(iscale == 1.0f || iscale == 2.0f);

  dt_iop_module_t module = { 0 };
  dt_dev_pixelpipe_t pipe = { 0 };
  dt_dev_pixelpipe_iop_t piece = { 0 };
  dt_iop_roi_t roi = { .x = 0, .y = 0, .width = WIDTH, .height = HEIGHT, .scale = 1.0f };
  dt_iop_contrastntexture_params_t params = { .gain_coarse = 0.7f, .gain_medium = 0.4f,
    .gain_fine = 0.2f, .detail_level = 5.0f, .halo_control = 0.0f,
    .noise_bias = 0.001f, .gain_shadows = 0.0f, .gain_highlights = 0.0f };
  float input[WIDTH * HEIGHT * 4], output[WIDTH * HEIGHT * 4];
  module.dev = calloc(1, sizeof(*module.dev));
  assert_non_null(module.dev);
  module.dev->gui_attached = FALSE;
  piece.iwidth = WIDTH;
  piece.iheight = HEIGHT;
  piece.iscale = iscale;
  piece.colors = 4;
  piece.pipe = &pipe;
  pipe.type = DT_DEV_PIXELPIPE_PREVIEW;

  for(int y = 0; y < HEIGHT; y++)
    for(int x = 0; x < WIDTH; x++)
    {
      const size_t i = ((size_t)y * WIDTH + x) * 4;
      const float fx = (float)x / (float)(WIDTH - 1);
      const float fy = (float)y / (float)(HEIGHT - 1);
      const int texture_index = (x * 17 + y * 31 + x * y * 7) % 251;
      const float texture = (float)texture_index / 250.0f;
      const float v = 0.08f + 0.72f * fx + 0.1f * texture;
      input[i] = v;
      input[i + 1] = v * (0.7f + 0.3f * fy);
      input[i + 2] = 0.15f + 0.75f * (0.5f * fx + 0.5f * texture);
      input[i + 3] = 0.25f + 0.75f * (float)((x + y) % 257) / 256.0f;
    }

  init_pipe(&module, &pipe, &piece);
  assert_non_null(piece.data);
  commit_params(&module, (dt_iop_params_t *)&params, &pipe, &piece);
  dt_iop_roi_t roi_in = roi;
  modify_roi_in(&module, &piece, &roi, &roi_in);
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  process(&module, &piece, input, output, &roi_in, &roi);
  GError *error = NULL;
  assert_true(g_file_set_contents(path, (const gchar *)output, sizeof(output), &error));
  assert_null(error);
  cleanup_pipe(&module, &pipe, &piece);
  free(module.dev);
}

int main(void)
{
  const struct CMUnitTest tests[] = { cmocka_unit_test(test_golden) };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
