cmake_minimum_required(VERSION 3.15)

# Compiler settings
set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")

# Global settings
set(GLOBAL_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

# Sets global output directory for single configuration (GCC)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(EXECUTABLE_OUTPUT_PATH ${GLOBAL_OUTPUT_PATH})
set(LIBRARY_OUTPUT_DIRECTORY ${GLOBAL_OUTPUT_PATH})
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

project(cvops_inference)

set(HEADER_FILES 
    ${INCLUDE_DIR}/c_api.h
    ${INCLUDE_DIR}/inference_manager_base.h
    ${INCLUDE_DIR}/inference_manager_factory.h
    ${INCLUDE_DIR}/inference_request.h
    ${INCLUDE_DIR}/inference_result.h
    ${INCLUDE_DIR}/inference_session_request.h
    ${INCLUDE_DIR}/model_platforms.h
    ${INCLUDE_DIR}/yolo_inference_manager.h
)

set(LIBRARY_FILES
    ${LIB_DIR}/c_api.cpp
    ${LIB_DIR}/inference_manager_base.cpp
    ${LIB_DIR}/inference_manager_factory.cpp
    ${LIB_DIR}/yolo_inference_manager.cpp
    ${LIB_DIR}/inference_result.cpp
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
        PUBLIC_HEADER ${INCLUDE_DIR}/c_api.h
)

add_executable(cvops_inference 
    ${SOURCE_DIR}/main.cpp
    ${LIBRARY_FILES}
    ${HEADER_FILES}
)

target_include_directories(cvops_inference PUBLIC ${INCLUDE_DIR})

set_target_properties(cvops_inference PROPERTIES
    OUTPUT_NAME cvops_inference.app
)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/opencv)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/numcpp)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/onnx)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/jsoncpp)

