#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

file(GLOB SOURCE_FILES
    src/*.cpp
    src/*.hpp
    src/*.h
    src/aftermath/*.*
    src/elements/*.*
    src/hbao/*.*
    src/micromap/*.*
    src/ui/*.*
    )

configure_file(src/toolbox_version.h.in toolbox_version.h)
 
file(GLOB OTHER_SOURCE_FILES 
    ${COMMON_DIR}/*.cpp 
    ${COMMON_DIR}/*.hpp 
    ${COMMON_DIR}/*.h
    ${COMMON_DIR}/vulkan_nv/vk_nv_micromesh_prototypes.c
    )
list(APPEND COMMON_SOURCE_FILES ${OTHER_SOURCE_FILES})

#set(APP_ICON_RESOURCE_WINDOWS "${CMAKE_CURRENT_SOURCE_DIR}/resources/appicon.rc")


_add_package_KTX()
_add_package_NVML()
_add_package_VMA()


add_executable(${PROJECT_NAME} 
    WIN32 # Avoiding console on Windows
    ${SOURCE_FILES} 
    ${COMMON_SOURCE_FILES} 
    ${PACKAGE_SOURCE_FILES} 
    )


target_include_directories(${PROJECT_NAME} PRIVATE    
    ${COMMON_DIR} 
    ${ADVANCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

target_link_libraries (${PROJECT_NAME} 
    ${PLATFORM_LIBRARIES} 
    ${LIBRARIES_OPTIMIZED} 
    nvpro_core
    )

 # Micromesh
 target_link_libraries(${PROJECT_NAME} 
    micromesh_tool_lib
    micromesh_displacement_remeshing
    meshops_bake
    meshops_core
    meshops_remesher
    ) 


target_compile_definitions(${PROJECT_NAME} 
    PRIVATE PROJECT_NAME="${PROJECT_NAME}"
    )

# To used micromesh
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_VK_NV_DISPLACEMENT_MICROMESH)


source_group("Other"  
    FILES ${PACKAGE_SOURCE_FILES})
source_group("Common"  
    FILES ${COMMON_SOURCE_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${SOURCE_FILES})


# Setting multiple Defines used in the project: paths, names
_add_project_definitions(${PROJECT_NAME})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Aftermath
set(MICROMESH_TOOLBOX_USE_AFTERMATH OFF CACHE BOOL "Use NSight Aftermath if present")

if(MICROMESH_TOOLBOX_USE_AFTERMATH)
    get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    # Copy the Aftermath dll where the project excutable is
    macro(copy_dll dlls)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${dlls} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endmacro()

    # Searching for the Aftermath SDK
    set(AFTERMATH_SDK ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/aftermath_sdk)
    if (NOT DEFINED ENV{NSIGHT_AFTERMATH_SDK})
      set(ENV{NSIGHT_AFTERMATH_SDK} ${AFTERMATH_SDK})
    endif()
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
    find_package(NsightAftermath)

    if(NsightAftermath_FOUND)
        message(STATUS "Found Aftermath under:" ${NsightAftermath_LIBRARY})
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NSIGHT_AFTERMATH)
        target_link_libraries (${PROJECT_NAME} ${NsightAftermath_LIBRARIES})
        target_include_directories(${PROJECT_NAME} PRIVATE ${NsightAftermath_INCLUDE_DIRS})
        string(REPLACE ".lib" ".dll" NsightAftermath_DLL ${NsightAftermath_LIBRARY})
        copy_dll(${NsightAftermath_DLL})
        install(FILES ${NsightAftermath_DLL} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
        install(FILES ${NsightAftermath_DLL} CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
    else()
        message("\n\n")
        message(VERBOSE "============= WARNING =============")
        message(VERBOSE "Path to the Aftermath SDK is missing.")
        message(VERBOSE "Please follow steps in ")
        message(VERBOSE " thirdparty/aftermath_sdk/_readme.txt")
        message(VERBOSE "============= WARNING =============\n\n")
    endif(NsightAftermath_FOUND)

endif(MICROMESH_TOOLBOX_USE_AFTERMATH)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Local shaders
file(GLOB SHD_SRC # Shader sources
    ${SHD_DIR}/*.vert 
    ${SHD_DIR}/*.frag 
    ${SHD_DIR}/*.rgen 
    ${SHD_DIR}/*.rchit 
    ${SHD_DIR}/*.rahit 
    ${SHD_DIR}/*.rmiss 
    ${SHD_DIR}/*.comp
    ${SHD_DIR}/*.mesh
    ${SHD_DIR}/*.task
    ${SHD_DIR}/draw_compressed_basic.mesh.glsl
    ${SHD_DIR}/draw_compressed_basic.task.glsl
    ${SHD_DIR}/draw_compressed_basic.frag.glsl
    ${SHD_DIR}/draw_heightmap.mesh.glsl
    ${SHD_DIR}/draw_heightmap.task.glsl
    ${SHD_DIR}/draw_heightmap.frag.glsl
    )
    
file(GLOB SHD_HDR  # Shader headers
    ${SHD_DIR}/*.glsl 
    ${SHD_DIR}/*.h
    )



# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_SRC}
    HEADER_FILES ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
    VULKAN_TARGET "vulkan1.3"
    HEADER ON
    DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
    FLAGS -I${SHD_DIR} -I${NVPRO_CORE_DIR} -g
    )

target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
source_group("Shaders/src"   FILES ${GLSL_SOURCES} )
source_group("Shaders/hdr"   FILES ${GLSL_HEADERS} )

# Precompile headers 
# target_precompile_headers(${PROJECT_NAME} PRIVATE 
#     $<$<COMPILE_LANGUAGE:CXX>:glm/glm.hpp>
#     $<$<COMPILE_LANGUAGE:CXX>:string>
#     $<$<COMPILE_LANGUAGE:CXX>:filesystem>
#     $<$<COMPILE_LANGUAGE:CXX>:vk_mem_alloc.h>
#     $<$<COMPILE_LANGUAGE:CXX>:nvmath/nvmath.h>
#     $<$<COMPILE_LANGUAGE:CXX>:nvvk/memallocator_vma_vk.hpp>
#     )


# Copy binary
_finalize_target( ${PROJECT_NAME} )
