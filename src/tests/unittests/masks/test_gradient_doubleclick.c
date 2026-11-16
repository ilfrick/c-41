/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

// Exercise the gradient event callback directly: this path only reads or writes
// the supplied form/gui state and does not require initialized GTK globals.

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>

#include <cmocka.h>
#include <gdk/gdk.h>
#include "develop/masks.h"

static void _setup_empty_form(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui)
{
  memset(form, 0, sizeof(*form));
  form->type = DT_MASKS_GRADIENT;
  form->functions = &dt_masks_functions_gradient;

  memset(gui, 0, sizeof(*gui));
  gui->creation = TRUE;
  gui->posx = 10.f;
  gui->posy = 20.f;
}

static void _assert_creation_click_starts_drag(dt_masks_form_t *form,
                                                dt_masks_form_gui_t *gui,
                                                const int result)
{
  assert_int_equal(result, 0);
  assert_null(form->points);
  assert_true(gui->form_dragging);
  assert_float_equal(gui->posx_source, 10.f, 0.f);
  assert_float_equal(gui->posy_source, 20.f, 0.f);
}

static void test_doubleclick_during_creation_starts_drag(void **state)
{
  (void)state;
  dt_masks_form_t form;
  dt_masks_form_gui_t gui;
  _setup_empty_form(&form, &gui);

  const int result = dt_masks_functions_gradient.button_pressed(NULL, 0.5f, 0.5f, 1.0,
                                                                GDK_BUTTON_PRIMARY,
                                                                GDK_2BUTTON_PRESS, 0,
                                                                &form, 0, &gui, 0);
  _assert_creation_click_starts_drag(&form, &gui, result);
}

static void test_singleclick_during_creation_starts_drag(void **state)
{
  (void)state;
  dt_masks_form_t form;
  dt_masks_form_gui_t gui;
  _setup_empty_form(&form, &gui);

  const int result = dt_masks_functions_gradient.button_pressed(NULL, 0.5f, 0.5f, 1.0,
                                                                GDK_BUTTON_PRIMARY,
                                                                GDK_BUTTON_PRESS, 0,
                                                                &form, 0, &gui, 0);
  _assert_creation_click_starts_drag(&form, &gui, result);
}

static void test_doubleclick_on_empty_form_not_creating(void **state)
{
  (void)state;
  dt_masks_form_t form;
  dt_masks_form_gui_t gui;
  _setup_empty_form(&form, &gui);
  gui.creation = FALSE;

  const int result = dt_masks_functions_gradient.button_pressed(NULL, 0.5f, 0.5f, 1.0,
                                                                GDK_BUTTON_PRIMARY,
                                                                GDK_2BUTTON_PRESS, 0,
                                                                &form, 0, &gui, 0);
  assert_int_equal(result, 0);
  assert_null(form.points);
  assert_false(gui.form_dragging);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_doubleclick_during_creation_starts_drag),
    cmocka_unit_test(test_singleclick_during_creation_starts_drag),
    cmocka_unit_test(test_doubleclick_on_empty_form_not_creating)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}

// clang-format off
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: indent-width 2; replace-tabs on; indent-mode cstyle;
// clang-format on
