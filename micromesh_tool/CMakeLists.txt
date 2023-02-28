#-------------------------------------------------------------------------
# Library
#-------------------------------------------------------------------------
set(PROJECT_NAME "micromesh_tool_lib")

set(SOURCE_FILES
  src/baker_manager.cpp
  src/baker_manager.hpp
  src/tool_context.cpp
  src/tool_bary.cpp
  src/tool_scene.cpp
  src/tool_bake.cpp
  src/tool_bake_args.cpp
  src/tool_generate.cpp
  src/tool_pretessellate.cpp
  src/tool_remesh.cpp
  src/pullpush_filter.cpp
  src/pullpush_filter.hpp
  src/tool_merge.cpp
  src/tool_merge_args.cpp
  src/tool_meshops_objects.hpp
  src/tool_displacedtessellate.cpp
  src/tool_image.cpp
  src/tool_optimize.cpp
  src/mesh_view_conv.cpp
  src/gltf.cpp
  src/gltf/micromesh_util.cpp
)

file(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.*)

# Add library
add_library(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

source_group("public_include" FILES ${HEADER_FILES})

# Internal include directories
target_include_directories(${PROJECT_NAME} PRIVATE src)

# Add thirdparty
target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_DIR})

# Export include directories to use this library
target_include_directories(${PROJECT_NAME} PUBLIC include)

# Dependencies
target_link_libraries(${PROJECT_NAME} meshops_bake meshops_core meshops_remesher)

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
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "MicroMeshTool")

#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------
set(PROJECT_NAME "micromesh_tool")
set(SOURCE_FILES
  src/main.cpp
  src/tinygltf_impl.cpp
)

if(UNIX)
  set(CMAKE_EXECUTABLE_SUFFIX "") # remove nvpro_core default
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

configure_file(src/tool_version.h.in tool_version.h)

_add_package_VMA()

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_DIR})

# Adding extra source files from nvpro_core
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")
target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES})
source_group("nvpro_core"     FILES ${COMMON_SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} micromesh_tool_lib)

# Extra
install(TARGETS ${PROJECT_NAME})
