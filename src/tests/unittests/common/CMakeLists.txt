add_cmocka_mock_test(test_imageio_rgbe_validate
                     SOURCES test_imageio_rgbe_validate.c
                     LINK_LIBRARIES cmocka)

add_cmocka_mock_test(test_imageio_tiff_photometric
                     SOURCES test_imageio_tiff_photometric.c
                     LINK_LIBRARIES lib_darktable cmocka)

add_cmocka_mock_test(test_imageio_rawspeed_mutex
                     SOURCES test_imageio_rawspeed_mutex.cc
                     LINK_LIBRARIES lib_darktable cmocka rawspeed)
target_include_directories(test_imageio_rawspeed_mutex PRIVATE
                           ${CMAKE_SOURCE_DIR}/src/external/rawspeed/src/librawspeed
                           $<TARGET_PROPERTY:rawspeed,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_features(test_imageio_rawspeed_mutex PRIVATE cxx_std_20)
target_compile_definitions(test_imageio_rawspeed_mutex PRIVATE
                           RAWSPEED_CAMERAS_FILE=\"${CMAKE_SOURCE_DIR}/src/external/rawspeed/data/cameras.xml\")
set_tests_properties(test_imageio_rawspeed_mutex PROPERTIES TIMEOUT 10)
if(WIN32)
    _copy_required_library(test_imageio_rawspeed_mutex lib_darktable)
endif(WIN32)

if(WIN32)
    _copy_required_library(test_imageio_tiff_photometric lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_dng_opcode
                     SOURCES test_dng_opcode.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_dng_opcode lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_commasubstring
                     SOURCES test_commasubstring.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_commasubstring lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_collection_split_operator_exposure
                     SOURCES test_collection_split_operator_exposure.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_collection_split_operator_exposure lib_darktable)
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

add_cmocka_mock_test(test_pixelpipe_cache_export_profile
                     SOURCES test_pixelpipe_cache_export_profile.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_pixelpipe_cache_export_profile lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_color_picker
                     SOURCES test_color_picker.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_color_picker lib_darktable)
endif(WIN32)

add_cmocka_mock_test(test_camera_control
                     SOURCES test_camera_control.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_camera_control lib_darktable)
endif(WIN32)
