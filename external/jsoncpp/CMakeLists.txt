find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

link_libraries(${JSONCPP_LIBRARIES})

target_link_libraries(cvops_inference ${JSONCPP_LIBRARIES})
target_link_libraries(C_API ${JSONCPP_LIBRARIES})
