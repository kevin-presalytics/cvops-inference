cmake_minimum_required(VERSION 3.24)

# Compiler settings
SET (CMAKE_C_COMPILER             "/usr/bin/clang")
SET (CMAKE_C_FLAGS                "-Wall -std=c99")
SET (CMAKE_C_FLAGS_DEBUG          "-g")
SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS                "-Wall")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# SET (CMAKE_AR      "/usr/bin/llvm-ar")
SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
# SET (CMAKE_NM      "/usr/bin/llvm-nm")
# SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
# SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
set(CMAKE_CXX_FLAGS "-std=c++17")

# Global settings
set(GLOBAL_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Sets global output directory for single configuration (GCC)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(EXECUTABLE_OUTPUT_PATH ${GLOBAL_OUTPUT_PATH})
set(LIBRARY_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

project(cvops_inference)

set(HEADER_FILES 
    ${INCLUDE_DIR}/cvops_inference.h
    ${INCLUDE_DIR}/hungarian.h
    ${INCLUDE_DIR}/inference_manager_base.h
    ${INCLUDE_DIR}/inference_manager_factory.h
    ${INCLUDE_DIR}/inference_request.h
    ${INCLUDE_DIR}/inference_result.h
    ${INCLUDE_DIR}/inference_session_request.h
    ${INCLUDE_DIR}/kalman_filter.h
    ${INCLUDE_DIR}/model_platforms.h
    ${INCLUDE_DIR}/yolo_inference_manager.h
    ${INCLUDE_DIR}/metadata_parser.h
    ${INCLUDE_DIR}/image_utils.h
    ${INCLUDE_DIR}/tracking.h
)

set(LIBRARY_FILES
    ${LIB_DIR}/cvops_inference.cpp
    ${LIB_DIR}/hungarian.cpp
    ${LIB_DIR}/inference_manager_base.cpp
    ${LIB_DIR}/inference_manager_factory.cpp
    ${LIB_DIR}/yolo_inference_manager.cpp
    ${LIB_DIR}/inference_result.cpp
    ${LIB_DIR}/kalman_filter.cpp
    ${LIB_DIR}/metadata_parser.cpp
    ${LIB_DIR}/image_utils.cpp
    ${LIB_DIR}/inference_request.cpp
    ${LIB_DIR}/tracking.cpp
)

add_library(C_API SHARED
    ${LIBRARY_FILES}
    ${HEADER_FILES}
)

target_include_directories(C_API PUBLIC ${INCLUDE_DIR})

set_target_properties(C_API 
    PROPERTIES
        VERSION 1.0.0
        OUTPUT_NAME cvops
        PUBLIC_HEADER ${INCLUDE_DIR}/cvops_inference.h
)

add_executable(cvops_inference 
    ${SOURCE_DIR}/main.cpp
    ${LIBRARY_FILES}
    ${HEADER_FILES}
)

target_include_directories(cvops_inference PUBLIC ${INCLUDE_DIR})

set_target_properties(cvops_inference PROPERTIES
    OUTPUT_NAME cvops
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/opencv)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/onnx)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/jsoncpp)

install(
    TARGETS cvops_inference C_API
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

