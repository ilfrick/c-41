add_cmocka_mock_test(test_gradient_doubleclick
                     SOURCES test_gradient_doubleclick.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_gradient_doubleclick lib_darktable)
endif(WIN32)
