DefaultBasicMakefile()

# Get name
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Adding the micromesh prototype
set(VULKAN_NV 
	${COMMON_DIR}/vulkan_nv/vk_nv_micromesh_prototypes.c 
	${COMMON_DIR}/vulkan_nv/vk_nv_micromesh_prototypes.h
	)
target_sources(${PROJECT_NAME} PRIVATE ${VULKAN_NV})
source_group(vulkan_nv FILES ${VULKAN_NV})

# Adding the dependency libraries
target_link_libraries(${PROJECT_NAME} micromesh_core)
target_link_libraries(${PROJECT_NAME} meshops_core)
target_link_libraries(${PROJECT_NAME} micromesh_displacement_compression)

# For array_view
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_DIR})
