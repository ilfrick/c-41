/* Reject TIFF files that omit mandatory PhotometricInterpretation. */
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <cmocka.h>
#include <glib.h>
#include <glib/gstdio.h>

#include "imageio/imageio_tiff.h"

static void test_missing_photometric_is_file_corrupted(void **state)
{
  (void)state;

  /* A minimal little-endian TIFF: 1x1, 8-bit, one uncompressed strip, but no
     mandatory PhotometricInterpretation tag. */
  static const uint8_t malformed_tiff[] = {
    'I', 'I', 42, 0, 8, 0, 0, 0,
    9, 0,
    0, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    2, 1, 3, 0, 1, 0, 0, 0, 8, 0, 0, 0,
    3, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    17, 1, 4, 0, 1, 0, 0, 0, 122, 0, 0, 0,
    21, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    22, 1, 4, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    23, 1, 4, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    28, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0,
    0
  };

  GError *error = NULL;
  gchar *directory = g_dir_make_tmp("c41-tiff-photometric-XXXXXX", &error);
  if(!directory)
  {
    gchar message[256];
    g_snprintf(message,
               sizeof(message),
               "failed to create temporary directory: %s",
               error ? error->message : "unknown error");
    g_clear_error(&error);
    fail_msg("%s", message);
    return;
  }
  g_clear_error(&error);

  gchar *filename = g_build_filename(directory, "missing-photometric.tif", NULL);
  if(!g_file_set_contents(filename,
                          (const gchar *)malformed_tiff,
                          (gssize)sizeof(malformed_tiff),
                          &error))
  {
    gchar message[256];
    g_snprintf(message,
               sizeof(message),
               "failed to write malformed TIFF fixture: %s",
               error ? error->message : "unknown error");
    g_clear_error(&error);
    g_remove(filename);
    g_rmdir(directory);
    g_free(filename);
    g_free(directory);
    fail_msg("%s", message);
    return;
  }
  g_clear_error(&error);

  dt_image_t image = { 0 };
  image.exif_inited = TRUE;
  const dt_imageio_retval_t result = dt_imageio_open_tiff(&image, filename, NULL);

  g_remove(filename);
  g_rmdir(directory);
  g_free(filename);
  g_free(directory);

  assert_int_equal(result, DT_IMAGEIO_FILE_CORRUPTED);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_missing_photometric_is_file_corrupted),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
