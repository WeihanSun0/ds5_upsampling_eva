add_executable(FGS_test) # eva app

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FGS)
aux_source_directory(FGS DIR_SRC)
add_library(FGS ${DIR_SRC})

target_sources(FGS_test
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fgs_test.cpp
)

target_link_libraries(FGS_test
PRIVATE
    common
    FGS
    ${OpenCV_LIBS}
)


add_executable(ds_reader) # read dsviewer output sample

target_sources(ds_reader
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ds_reader.cpp
)

target_link_libraries(ds_reader
PRIVATE
    common
    ${OpenCV_LIBS}
)

add_executable(test_fix_hand) # test fix hand shift by sim data

target_sources(test_fix_hand
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fix_hand.cpp
)

target_link_libraries(test_fix_hand
PRIVATE
    common
    ${OpenCV_LIBS}
)

