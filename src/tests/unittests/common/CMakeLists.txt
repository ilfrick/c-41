add_cmocka_mock_test(test_dng_opcode
                     SOURCES test_dng_opcode.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_dng_opcode lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_iop_introspection
                     SOURCES test_iop_introspection.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_iop_introspection lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_image_cache_null
                     SOURCES test_image_cache_null.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_image_cache_null lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_color_harmony_get
                     SOURCES test_color_harmony_get.c ${CMAKE_SOURCE_DIR}/src/common/color_harmony.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_color_harmony_get lib_darktable)
endif(WIN32)
