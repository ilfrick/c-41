add_cmocka_mock_test(test_filmicrgb
                     SOURCES test_filmicrgb.c ../util/testimg.c
                     LINK_LIBRARIES lib_darktable cmocka
                     MOCKS dt_iop_color_picker_reset)
if(TARGET c41_core)
  # filmicrgb.c is included directly by this test and calls Rust FFI symbols.
  target_link_libraries(test_filmicrgb PRIVATE c41_core)
endif()

add_cmocka_mock_test(test_wb_late_correction
                     SOURCES test_wb_late_correction.c
                     LINK_LIBRARIES lib_darktable cmocka)

add_cmocka_mock_test(test_hazeremoval_quick_select
                     SOURCES test_hazeremoval_quick_select.c
                     LINK_LIBRARIES lib_darktable cmocka)
if(TARGET c41_core)
  # hazeremoval.c is included directly and calls Rust FFI kernels.
  target_link_libraries(test_hazeremoval_quick_select PRIVATE c41_core)
endif()

add_cmocka_mock_test(test_capture_mask
                     SOURCES test_capture_mask.c
                     LINK_LIBRARIES lib_darktable cmocka)
target_include_directories(test_capture_mask PRIVATE ${CMAKE_SOURCE_DIR}/data/kernels)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_cmocka_mock_test(test_highlights_legacy_params
                       SOURCES test_highlights_legacy_params.c
                       LINK_LIBRARIES lib_darktable cmocka)
  if(TARGET c41_core)
    target_link_libraries(test_highlights_legacy_params PRIVATE c41_core)
  endif()
endif()

# Windows: libs have to be copied next to the executable
if(WIN32)
    _copy_required_library(test_filmicrgb lib_darktable)
    _copy_required_library(test_wb_late_correction lib_darktable)
    _copy_required_library(test_hazeremoval_quick_select lib_darktable)
    _copy_required_library(test_capture_mask lib_darktable)
endif(WIN32)
