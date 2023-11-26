
include(FetchContent)

set(ONNX_DIR ${CMAKE_SOURCE_DIR}/build/external/onnxruntime)
set(ONNX_DOWNLOAD_DIR ${ONNX_DIR}/download)
set(ONNX_PLATFORM "linux-x64")
set(ONNX_VERSION 1.12.1)
set(ONNX_RUNTIME_IDENTIFIER onnxruntime-${ONNX_PLATFORM}-${ONNX_VERSION})
set(ONNX_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_RUNTIME_IDENTIFIER}.tgz)

FetchContent_Declare(
    onnxruntime_binary
    URL ${ONNX_URL}
    DOWNLOAD_NO_EXTRACT TRUE
    TLS_VERIFY TRUE
    DOWNLOAD_DIR ${ONNX_DOWNLOAD_DIR}
)

FetchContent_Populate(onnxruntime_binary)

set(ONNX_BIN_DIR ${ONNX_DIR}/bin)
set(ONNX_INCLUDE_DIR ${ONNX_BIN_DIR}/${ONNX_RUNTIME_IDENTIFIER}/include)


file(
    ARCHIVE_EXTRACT 
    INPUT ${ONNX_DOWNLOAD_DIR}/${ONNX_RUNTIME_IDENTIFIER}.tgz 
    DESTINATION ${ONNX_BIN_DIR}
)

add_library(ONNX_RUNTIME SHARED IMPORTED)

set_target_properties(ONNX_RUNTIME PROPERTIES 
    IMPORTED_LOCATION ${ONNX_BIN_DIR}/${ONNX_RUNTIME_IDENTIFIER}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INCLUDE_DIR}
)

target_link_libraries(C_API ONNX_RUNTIME)
target_link_libraries(RENDERING_API ONNX_RUNTIME)
target_link_libraries(cvops_inference ONNX_RUNTIME)

target_include_directories(cvops_inference PRIVATE ${ONNX_INCLUDE_DIR})
target_include_directories(C_API PRIVATE ${ONNX_INCLUDE_DIR})
target_include_directories(RENDERING_API PRIVATE ${ONNX_INCLUDE_DIR})

install(
    FILES 
        ${ONNX_BIN_DIR}/${ONNX_RUNTIME_IDENTIFIER}/lib/libonnxruntime.so
        ${ONNX_BIN_DIR}/${ONNX_RUNTIME_IDENTIFIER}/lib/libonnxruntime.so.${ONNX_VERSION}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${ONNX_BIN_DIR}/${ONNX_RUNTIME_IDENTIFIER}
)
