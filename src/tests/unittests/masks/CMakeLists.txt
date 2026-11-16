add_cmocka_mock_test(test_gradient_doubleclick
                     SOURCES test_gradient_doubleclick.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_gradient_doubleclick lib_darktable)
endif(WIN32)

# This test provides an executable-level dt_database_get() replacement for its
# in-memory SQLite fixture. Keep it on ELF platforms where symbol interposition
# is supported; the shared-library replacement is not portable to Windows.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_cmocka_mock_test(test_masks_history_statement
                       SOURCES test_masks_history_statement.c
                       LINK_LIBRARIES lib_darktable cmocka)
endif()
