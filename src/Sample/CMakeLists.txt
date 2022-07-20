add_executable(upsampling_sample)

target_sources(upsampling_sample
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/upsampling.cpp
)

target_link_libraries(upsampling_sample
PRIVATE
    common
    ${OpenCV_LIBS}
)