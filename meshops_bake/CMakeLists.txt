#-------------------------------------------------------------------------
# Library
#-------------------------------------------------------------------------
set(PROJECT_NAME "meshops_bake")

# Sources and headers for the library
set(SOURCE_FILES
    meshops_bake_batch.hpp
    meshops_bake_batch.cpp
    meshops_bake_vk.hpp
    meshops_bake_vk.cpp
    meshops_bake.cpp
)

# Attempt to match MSVC warnings
if(UNIX)
    set_property(SOURCE ${SOURCE_FILES} APPEND_STRING PROPERTY COMPILE_FLAGS -Wconversion)
endif()

# Add library
add_library(${PROJECT_NAME} ${SOURCE_FILES})

# Add thirdparty  
target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_DIR})
 
# Dependencies
target_link_libraries (${PROJECT_NAME} meshops_core)

# We need this directory, and users of our library will need it too
#target_include_directories(${PROJECT_NAME} INTERFACE .)

#----------------------------------
# Adding shader files
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h ${COMMON_DIR}/*.glsl)
file(GLOB SHD_SRC  ${SHD_DIR}/*.rchit ${SHD_DIR}/*.rmiss ${SHD_DIR}/*.frag ${SHD_DIR}/*.rahit ${SHD_DIR}/*.rgen ${SHD_DIR}/*.vert ${SHD_DIR}/*.comp)
# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_SRC}
    HEADER_FILES ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen"
    VULKAN_TARGET "vulkan1.2"
    HEADER ON 
    DEPENDENCY ON
    FLAGS "-I${SHD_DIR}" "-I${COMMON_DIR}" -g
    )
target_sources(${PROJECT_NAME} PRIVATE ${SHD_SRC} ${SHD_HDR})
source_group("Shaders"         FILES ${SHD_SRC} ${SHD_HDR})
#----------------------------------

# Extra
install(TARGETS ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "meshops_bake")
