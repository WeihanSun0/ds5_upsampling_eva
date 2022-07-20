add_executable(FBS_test)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FBS)

aux_source_directory(FBS DIR_SRC)
add_library(FBS ${DIR_SRC})

target_sources(FBS_test
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/fbs_test.cpp
)

target_link_libraries(FBS_test
PRIVATE
    common
    FBS 
    ${OpenCV_LIBS}
)