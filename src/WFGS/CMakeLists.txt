add_executable(WFGS_test)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/openSAT)

aux_source_directory(openSAT DIR_SRC)
add_library(SAT ${DIR_SRC})

target_sources(WFGS_test
PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/wfgs_test.cpp
)

target_link_libraries(WFGS_test
PRIVATE
    common
    SAT
    ${OpenCV_LIBS}
)