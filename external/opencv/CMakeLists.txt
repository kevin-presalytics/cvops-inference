
FIND_PACKAGE(
    OpenCV
    REQUIRED   
)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(cvops_inference ${OpenCV_LIBS} )
target_link_libraries(C_API ${OpenCV_LIBS} )
target_link_libraries(RENDERING_API ${OpenCV_LIBS} )