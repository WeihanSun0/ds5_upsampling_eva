add_executable(eva_mk_pc) # make sparse pc from gt for test

include_directories(../common)

target_sources(eva_mk_pc
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/artifact/mk_sparse_pc.cpp
)

target_link_libraries(eva_mk_pc
PRIVATE
    ${OpenCV_LIBS}
)
