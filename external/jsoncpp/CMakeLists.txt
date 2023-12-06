find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

link_libraries(${JSONCPP_LIBRARIES})

target_link_libraries(cvops_inference PRIVATE ${JSONCPP_LIBRARIES})
target_link_libraries(C_API PRIVATE ${JSONCPP_LIBRARIES})
