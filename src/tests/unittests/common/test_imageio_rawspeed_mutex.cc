/* A failed RawSpeed read must not leave the shared read mutex locked. */
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>

#include <glib.h>
#include <glib/gstdio.h>

#include "common/darktable.h"
#include "common/dtpthread.h"
#include "imageio/imageio_common.h"
#include "imageio/imageio_rawspeed.h"
#include "io/FileIOException.h"
#include "io/FileReader.h"

extern "C"
{
#include <cmocka.h>
}

struct test_fixture
{
  gchar *root;
  gchar *previous_dir;
  gchar *previous_datadir;
  gchar *test_datadir;
  gboolean changed_dir;
  gboolean changed_datadir;
  gboolean mutex_initialized;
  gboolean mutex_may_be_locked;
  gboolean mutex_held_by_test;
};

static int teardown(void **state);

static int setup(void **state)
{
  struct test_fixture *fixture = g_new0(struct test_fixture, 1);
  *state = fixture;

  GError *error = NULL;
  fixture->root = g_dir_make_tmp("c41-rawspeed-mutex-XXXXXX", &error);
  if(!fixture->root || error)
  {
    g_clear_error(&error);
    teardown(state);
    *state = NULL;
    return -1;
  }

  gchar *rawspeed = g_build_filename(fixture->root, "rawspeed", NULL);
  gchar *camera_source = g_strdup(RAWSPEED_CAMERAS_FILE);
  gchar *camera_target = g_build_filename(rawspeed, "cameras.xml", NULL);
  gchar *contents = NULL;
  gsize length = 0;
  gboolean ok = g_mkdir(rawspeed, 0700) == 0
                && g_file_get_contents(camera_source, &contents, &length, &error)
                && g_file_set_contents(camera_target, contents, length, &error);
  g_clear_error(&error);
  g_free(contents);
  g_free(camera_source);
  g_free(camera_target);
  g_free(rawspeed);
  if(!ok)
  {
    teardown(state);
    *state = NULL;
    return -1;
  }

  gchar *fixture_dir = g_build_filename(fixture->root, "fixture", NULL);
  gchar *raw_path = g_build_filename(fixture_dir, "raw", NULL);
  ok = g_mkdir(fixture_dir, 0700) == 0 && g_file_set_contents(raw_path, "", 0, &error);
  g_clear_error(&error);
  g_free(fixture_dir);
  g_free(raw_path);
  if(!ok)
  {
    teardown(state);
    *state = NULL;
    return -1;
  }

  fixture->previous_dir = g_get_current_dir();
  if(g_chdir(fixture->root) != 0)
  {
    teardown(state);
    *state = NULL;
    return -1;
  }
  fixture->changed_dir = TRUE;
  fixture->previous_datadir = darktable.datadir;
  fixture->test_datadir = g_strdup(fixture->root);
  fixture->changed_datadir = TRUE;
  darktable.datadir = fixture->test_datadir;
  if(dt_pthread_mutex_init(&darktable.readFile_mutex, NULL) != 0)
  {
    teardown(state);
    *state = NULL;
    return -1;
  }
  fixture->mutex_initialized = TRUE;
  return 0;
}

static int teardown(void **state)
{
  struct test_fixture *fixture = static_cast<struct test_fixture *>(*state);
  if(!fixture) return 0;

  if(fixture->changed_dir) g_chdir(fixture->previous_dir);
  if(fixture->mutex_initialized)
  {
    if(fixture->mutex_held_by_test) dt_pthread_mutex_unlock(&darktable.readFile_mutex);
    if(!fixture->mutex_may_be_locked) dt_pthread_mutex_destroy(&darktable.readFile_mutex);
  }
  if(fixture->changed_datadir)
  {
    g_free(fixture->test_datadir);
    darktable.datadir = fixture->previous_datadir;
  }

  if(fixture->root)
  {
    gchar *path = g_build_filename(fixture->root, "fixture", "raw", NULL);
    g_remove(path);
    g_free(path);
    path = g_build_filename(fixture->root, "fixture", NULL);
    g_rmdir(path);
    g_free(path);
    path = g_build_filename(fixture->root, "rawspeed", "cameras.xml", NULL);
    g_remove(path);
    g_free(path);
    path = g_build_filename(fixture->root, "rawspeed", NULL);
    g_rmdir(path);
    g_free(path);
    g_rmdir(fixture->root);
  }
  g_free(fixture->root);
  g_free(fixture->previous_dir);
  g_free(fixture);
  return 0;
}

static void assert_read_mutex_available(struct test_fixture *fixture)
{
  const int lock_result = dt_pthread_mutex_trylock(&darktable.readFile_mutex);
  if(lock_result != 0)
  {
    fixture->mutex_may_be_locked = TRUE;
    fail_msg("readFile_mutex remained unavailable after a failed read: %d", lock_result);
  }

  fixture->mutex_held_by_test = TRUE;
  const int unlock_result = dt_pthread_mutex_unlock(&darktable.readFile_mutex);
  if(unlock_result != 0)
  {
    fixture->mutex_may_be_locked = TRUE;
    fail_msg("could not unlock readFile_mutex after trylock: %d", unlock_result);
  }
  fixture->mutex_held_by_test = FALSE;
}

static void test_read_failure_releases_mutex(void **state)
{
  struct test_fixture *fixture = static_cast<struct test_fixture *>(*state);
  assert_true(g_file_test("fixture/raw", G_FILE_TEST_IS_REGULAR));
  rawspeed::FileReader reader("fixture/raw");
  bool caught_file_exception = false;
  try
  {
    (void)reader.readFile();
  }
  catch(const rawspeed::FileIOException &)
  {
    caught_file_exception = true;
  }
  assert_true(caught_file_exception);

  dt_image_t image = { 0 };
  image.exif_inited = TRUE;
  g_strlcpy(image.filename, "fixture/raw", sizeof(image.filename));
  assert_int_equal(dt_imageio_open_rawspeed(&image, "fixture/raw", NULL), DT_IMAGEIO_IOERROR);
  assert_read_mutex_available(fixture);
  assert_int_equal(dt_imageio_open_rawspeed(&image, "fixture/raw", NULL), DT_IMAGEIO_IOERROR);
  assert_read_mutex_available(fixture);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_read_failure_releases_mutex),
  };
  return cmocka_run_group_tests(tests, setup, teardown);
}
