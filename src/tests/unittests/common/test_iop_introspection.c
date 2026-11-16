#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <cmocka.h>
#include "common/introspection.h"
#include "develop/imageop.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

extern gboolean _iop_validate_params(dt_introspection_field_t *field,
                                      gpointer params,
                                      const gboolean report,
                                      const char *name);

typedef struct test_params_t
{
  int8_t link;
  char guard;
} test_params_t;

static dt_introspection_t test_introspection = {
  .size = sizeof(test_params_t),
};

static dt_introspection_field_t test_fields[] = {
  {
    .Char = {
      .header = {
        .type = DT_INTROSPECTION_TYPE_CHAR,
        .size = sizeof(char),
        .offset = offsetof(test_params_t, guard),
      },
      .Default = 0x5a,
    },
  },
  {
    .Int8 = {
      .header = {
        .type = DT_INTROSPECTION_TYPE_INT8,
        .size = sizeof(int8_t),
        .offset = offsetof(test_params_t, link),
      },
      .Min = INT8_MIN,
      .Max = INT8_MAX,
      .Default = (char)-1,
    },
  },
  {
    .header = {
      .type = DT_INTROSPECTION_TYPE_NONE,
    },
  },
};

static dt_introspection_t *test_get_introspection(void)
{
  return &test_introspection;
}

static dt_introspection_field_t *test_get_introspection_linear(void)
{
  return test_fields;
}

static void test_int8_default_does_not_overwrite_neighbor(void **state)
{
  (void)state;

  dt_iop_module_so_t so = { 0 };
  so.get_introspection = test_get_introspection;
  so.get_introspection_linear = test_get_introspection_linear;

  dt_iop_module_t module = { 0 };
  module.so = &so;

  dt_iop_default_init(&module);

  assert_non_null(module.default_params);
  const test_params_t *defaults = (const test_params_t *)module.default_params;
  assert_int_equal(defaults->link, -1);
  assert_int_equal(defaults->guard, 0x5a);

  dt_iop_default_cleanup(&module);
  assert_null(module.params);
  assert_null(module.default_params);
}

static void test_int8_validation_accepts_negative_link(void **state)
{
  (void)state;

  int8_t link = -1;
  dt_introspection_field_t field = {
    .Int8 = {
      .header = {
        .type = DT_INTROSPECTION_TYPE_INT8,
        .type_name = "int8_t",
        .name = "link",
        .size = sizeof(int8_t),
        .offset = 0,
      },
      .Min = INT8_MIN,
      .Max = INT8_MAX,
      .Default = (char)-1,
    },
  };

  assert_true(_iop_validate_params(&field, &link, FALSE, "fake-iop"));
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_int8_default_does_not_overwrite_neighbor),
    cmocka_unit_test(test_int8_validation_accepts_negative_link),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
