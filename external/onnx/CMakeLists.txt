
include(FetchContent)

set(ONNX_DIR ${GLOBAL_OUTPUT_PATH}/onnx)
set(ONNX_DOWNLOAD_DIR ${ONNX_DIR}/download)
set(ONNX_PLATFORM "linux-x64")
set(ONNX_VERSION 1.12.1)
set(ONNX_RUNTIME_IDENTIFIER onnxruntime-${ONNX_PLATFORM}-${ONNX_VERSION})
set(ONNX_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_RUNTIME_IDENTIFIER}.tgz)




SET(FETCHCONTENT_QUIET FALSE CACHE INTERNAL "")
SET(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER CACHE INTERNAL "") 
FetchContent_Declare(
    onnxruntime_binary
    URL ${ONNX_URL}
    TLS_VERIFY TRUE
    DOWNLOAD_DIR ${ONNX_DOWNLOAD_DIR}
    SOURCE_DIR ${ONNX_DIR}/src
    BINARY_DIR ${ONNX_DIR}/bin
)

SET(ONNX_TARGET ${ONNX_DIR}/bin/${ONNX_RUNTIME_IDENTIFIER}/lib/libonnxruntime.so)

FetchContent_MakeAvailable(onnxruntime_binary)

set(ONNX_INCLUDE_DIR ${ONNX_DIR}/src/include)

add_library(ONNX_RUNTIME SHARED IMPORTED)

set_target_properties(ONNX_RUNTIME PROPERTIES 
    IMPORTED_LOCATION ${ONNX_DIR}/src/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INCLUDE_DIR}
)

target_link_libraries(C_API PRIVATE ONNX_RUNTIME)
target_link_libraries(cvops_inference PRIVATE ONNX_RUNTIME)

target_include_directories(cvops_inference PRIVATE ${ONNX_INCLUDE_DIR})
target_include_directories(C_API PRIVATE ${ONNX_INCLUDE_DIR})
