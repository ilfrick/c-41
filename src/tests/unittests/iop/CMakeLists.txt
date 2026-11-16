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

# Windows: libs have to be copied next to the executable
if(WIN32)
    _copy_required_library(test_filmicrgb lib_darktable)
    _copy_required_library(test_wb_late_correction lib_darktable)
endif(WIN32)
