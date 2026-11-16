add_cmocka_mock_test(test_dng_opcode
                     SOURCES test_dng_opcode.c
                     LINK_LIBRARIES lib_darktable cmocka)

if(WIN32)
    _copy_required_library(test_dng_opcode lib_darktable)
endif(WIN32)
