add_executable(PF_test)

include_directories((${CMAKE_CURRENT_SOURCE_DIR}/PlanarFilter))
aux_source_directory(PlanarFilter DIR_SRC)
add_library(PlanarFilter ${DIR_SRC})

target_sources(PF_test
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pf_test.cpp
)

target_link_libraries(PF_test
PRIVATE
    common
    PlanarFilter
    ${OpenCV_LIBS}
)