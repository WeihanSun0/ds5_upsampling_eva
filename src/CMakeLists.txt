project(upsampling)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(FGS)
add_subdirectory(WFGS) # comparision
add_subdirectory(FBS)
add_subdirectory(PlanarFilter) # comparision
add_subdirectory(Sample) # release to dsviewer
add_subdirectory(common)
add_subdirectory(fix_hand) # visual shift