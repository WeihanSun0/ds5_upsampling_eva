
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FGS)
aux_source_directory(FGS DIR_SRC)
add_library(ups ${DIR_SRC})


add_executable(fix_hand_0) # original upsampling without fix hand
target_sources(fix_hand_0
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_hand_0.cpp
)

target_link_libraries(fix_hand_0
PRIVATE
    common
    ups 
    ${OpenCV_LIBS}
)

add_executable(fix_hand_1) # fix hand by remove depth edge
target_sources(fix_hand_1
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_hand_1.cpp
)

target_link_libraries(fix_hand_1
PRIVATE
    common
    ups 
    ${OpenCV_LIBS}
)

add_executable(fix_hand_2) # fix hand by remove depth and guide edge
target_sources(fix_hand_2
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_hand_2.cpp
)

target_link_libraries(fix_hand_2
PRIVATE
    common
    ups 
    ${OpenCV_LIBS}
)

add_executable(fix_hand_3) # fix hand by remove depth and guide edge
target_sources(fix_hand_3
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_hand_3.cpp
)

target_link_libraries(fix_hand_3
PRIVATE
    common
    ups 
    ${OpenCV_LIBS}
)

add_executable(fix_hand_4) # fix hand by remove depth and guide edge
target_sources(fix_hand_4
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_hand_4.cpp
)

target_link_libraries(fix_hand_4
PRIVATE
    common
    ups 
    ${OpenCV_LIBS}
)