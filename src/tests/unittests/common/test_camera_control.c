#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>

#include <cmocka.h>

#include "common/camera_control.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static void test_ptp_transient_errors_are_case_insensitive(void **state)
{
  (void)state;
  static const struct
  {
    const gchar *text;
    gboolean expected;
  } cases[] = {
    { "PTP Device Busy", TRUE },
    { "ptp device busy", TRUE },
    { "PTP error 0x2019", TRUE },
    { "PTP I/O error", FALSE },
    { "device busy", FALSE },
    { NULL, FALSE },
  };

  for(size_t i = 0; i < G_N_ELEMENTS(cases); i++)
    assert_int_equal(dt_camctl_ptp_error_is_transient(cases[i].text), cases[i].expected);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_ptp_transient_errors_are_case_insensitive),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
