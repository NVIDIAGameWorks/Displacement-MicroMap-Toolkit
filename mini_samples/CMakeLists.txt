
function(DefaultBasicMakefile)

    # Executable
    get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(STATUS "-------------------------------")
    message(STATUS "Processing Project ${PROJECT_NAME}")
    add_executable(${PROJECT_NAME})

    # Sources for the project
    file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.h)
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES})  # Extra source from nvpro-core based on options
    target_sources(${PROJECT_NAME} PRIVATE ${PACKAGE_SOURCE_FILES}) # Extra source from nvpro-core based on options
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})
    source_group("Other" FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

    # Include paths
    target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR})

    # Linking with other libraries
    target_link_libraries (${PROJECT_NAME} ${PLATFORM_LIBRARIES})
    target_link_libraries (${PROJECT_NAME} nvpro_core)

    # Compile definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")

    # other properties
    set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")


    # Shaders in project
    set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
    file(GLOB SHD_SRC ${SHD_DIR}/*.vert ${SHD_DIR}/*.frag ${SHD_DIR}/*.rgen ${SHD_DIR}/*.rchit ${SHD_DIR}/*.rmiss)

    # Compiling shaders to Spir-V header
    compile_glsl(
        SOURCE_FILES ${SHD_SRC}
        HEADER_FILES ${SHD_HDR}
        DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
        VULKAN_TARGET "vulkan1.2"
        HEADER ON
        DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
        FLAGS "-I${SHD_DIR}" "-I${NVPRO_CORE_DIR}" -g
        )

    target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GLSL_SOURCES} ${GLSL_HEADERS})

    # Copy binary
    _finalize_target( ${PROJECT_NAME} )

endfunction()

add_subdirectory(dmm_displacement)

if(WIN32)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dx12_dmm_displacement")
       add_subdirectory(dx12_dmm_displacement)
    endif()
endif()
