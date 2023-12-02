SET(OPENCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/root)
# SET(OPENCV_INSTALL_DIR ${GLOBAL_OUTPUT_PATH}/opencv)
# SET(OPENCV_SRC_DIR ${OPENCV_DIR}/src)
# SET(OPENCV_BUILD_DIR ${OPENCV_DIR}/build)
# SET(OPENCV_INCLUDE_DIR ${OPENCV_DIR}/include/opencv4)
# SET(OPENCV_LIB_DIR ${OPENCV_DIR}/lib)

# MESSAGE(STATUS "OPENCV_DIR: ${OPENCV_DIR}")
# MESSAGE(STATUS "OPENCV_INSTALL_DIR: ${OPENCV_INSTALL_DIR}")

# SET(OPENCV_LIBS
#     opencv_calib3d
#     opencv_core
#     opencv_dnn
#     opencv_features2d
#     opencv_flann
#     opencv_gapi
#     opencv_highgui
#     opencv_imgcodecs
#     opencv_imgproc
#     opencv_ml
#     opencv_objdetect
#     opencv_photo
#     opencv_stitching
#     opencv_video
#     opencv_videoio
# )

# SET(BUILD_BYPRODUCTS "")
# foreach(lib ${OPENCV_LIBS})
#     list(APPEND BUILD_BYPRODUCTS ${OPENCV_LIB_DIR}/${lib}.lib)
# endforeach()

# SET(CMAKE_BINARY_DIR ${OPENCV_DIR}/build)
# SET(CMAKE_SOURCE_DIR ${OPENCV_DIR}/src)
# SET(CMAKE_INSTALL_PREFIX ${OPENCV_DIR} CACHE INTERNAL "")
# SET(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
# SET(CMAKE_BUILD_TYPE Debug CACHE INTERNAL "")
# SET(BUILD_opencv_python3 OFF CACHE INTERNAL "")
# SET(BUILD_opencv_python_bindings_generator ON CACHE INTERNAL "")
# SET(BUILD_JAVA OFF CACHE INTERNAL "")
# SET(BUILD_FAT_JAVA_LIB OFF CACHE INTERNAL "")
# SET(BUILD_opencv_java_bindings_generator OFF CACHE INTERNAL "")
# SET(BUILD_opencv_js OFF CACHE INTERNAL "")
# SET(BUILD_opencv_python2 OFF CACHE INTERNAL "")
# SET(ENABLE_PYLINT OFF CACHE INTERNAL "")
# SET(ENABLE_FLAKE8 OFF CACHE INTERNAL "")
# # SET(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE INTERNAL "")
SET(FETCHCONTENT_QUIET OFF CACHE INTERNAL "")

# SET(OPENCV_CMAKE_SOURCE_DIR ${OPENCV_SRC_DIR} CACHE INTERNAL "")




include(FetchContent)

FetchContent_Declare(OpenCV
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.8.1
    OVERRIDE_FIND_PACKAGE
)
# FetchContent_GetProperties(OpenCV)

# # if(NOT OpenCV_POPULATED)
# #     FetchContent_Populate(
# #         OpenCV
# #         URL https://github.com/opencv/opencv/archive/refs/tags/4.8.1.tar.gz
# #         SOURCE_DIR ${OPENCV_SRC_DIR}
# #         BINARY_DIR ${OPENCV_BUILD_DIR}
# #         CMAKE_ARGS
# #             -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
# #             -DBUILD_SHARED_LIBS=ON
# #             -DCMAKE_BUILD_TYPE=Debug
# #             -DBUILD_opencv_python3=OFF
# #             -DBUILD_opencv_python_bindings_generator=ON
# #             -DBUILD_JAVA=OFF
# #             -DBUILD_FAT_JAVA_LIB=OFF
# #             -DBUILD_opencv_java_bindings_generator=OFF
# #             -DBUILD_opencv_js=OFF
# #             -DBUILD_opencv_python2=OFF
# #             -DENABLE_PYLINT=OFF
# #             -DENABLE_FLAKE8=OFF
# #             -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
# #             -DFETCHCONTENT_QUIET=OFF
# #     )
# #     message(STATUS "OpenCV_SOURCE_DIR: ${opencv_SOURCE_DIR}")
# #     message(STATUS "OpenCV_BINARY_DIR: ${opencv_BINARY_DIR}")
# #     add_subdirectory(${opencv_SOURCE_DIR} ${opencv_BINARY_DIR})
# # endif ()
# SET(OpenCV_CMAKE_BINARY_DIR ${OPENCV_BUILD_DIR} CACHE INTERNAL "")
# SET(CMAKE_INSTALL_PREFIX ${OPENCV_DIR} CACHE INTERNAL "")
SET(OpenCV_BUILD_SHARED_LIBS ON CACHE INTERNAL "")
SET(OpenCV_CMAKE_BUILD_TYPE Debug CACHE INTERNAL "")
SET(OpenCV_BUILD_opencv_python3 OFF CACHE INTERNAL "")
SET(OpenCV_BUILD_opencv_python_bindings_generator ON CACHE INTERNAL "")
SET(OpenCV_BUILD_JAVA OFF CACHE INTERNAL "")
SET(OpenCV_BUILD_FAT_JAVA_LIB OFF CACHE INTERNAL "")
SET(OpenCV_BUILD_opencv_java_bindings_generator OFF CACHE INTERNAL "")
SET(OpenCV_BUILD_opencv_js OFF CACHE INTERNAL "")
SET(OpenCV_BUILD_opencv_python2 OFF CACHE INTERNAL "")
SET(OpenCV_ENABLE_PYLINT OFF CACHE INTERNAL "")
SET(OpenCV_ENABLE_FLAKE8 OFF CACHE INTERNAL "")
SET(OpenCV_PREFIX ${GLOBAL_OUTPUT_PATH}/build CACHE INTERNAL "")
FetchContent_MakeAvailable(OpenCV)

# include(${GLOBAL_OUTPUT_PATH}/OpenCVConfig.cmake)

MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


# foreach(lib ${OPENCV_LIBS})
#     add_library(${lib} STATIC IMPORTED)
#     set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION ${OPENCV_LIB_DIR}/${lib}.lib)
# endforeach()


# include_directories( ${OPENCV_INCLUDE_DIR} )
# target_link_libraries(cvops_inference ${OPENCV_LIBS} )
# target_link_libraries(C_API ${OPENCV_LIBS} )