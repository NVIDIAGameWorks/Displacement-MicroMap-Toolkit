
# All sources and headers
file(GLOB SOURCE_FILES src/*.*)
file(GLOB PUBLIC_HEADER_FILES include/imageio/*.* include/meshops/*.*  include/microutils/*.* include/mikktspace/*.*)
file(GLOB INTERNAL_HEADER_FILES include/meshops_internal/*.*)

source_group("public_include" FILES ${PUBLIC_HEADER_FILES})
source_group("internal_include" FILES ${INTERNAL_HEADER_FILES})
source_group("source" FILES ${SOURCE_FILES})

# Make an automatic library - will be static or dynamic based on user setting
add_library(meshops_core ${SOURCE_FILES} ${PUBLIC_HEADER_FILES} ${INTERNAL_HEADER_FILES})

# We need this directory, and users of our library will need it too
target_include_directories(meshops_core PUBLIC include)

target_include_directories(meshops_core PRIVATE $<TARGET_PROPERTY:micromesh_core,INTERFACE_INCLUDE_DIRECTORIES>)

# This depends on nvpro_core bary_core bary_utils micromesh_core
target_link_libraries(meshops_core PUBLIC nvpro_core bary_core bary_utils micromesh_core micromesh_displacement_compression png_static tinyexr)

set_target_properties(meshops_core PROPERTIES FOLDER "meshops_core")

# Add pretty printers for visual studio
# Alt: add_link_options(/NATVIS:file.natvis) if not for https://github.com/microsoft/vscode-cpptools/issues/10175
if (MSVC_IDE)
  target_sources(meshops_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshops/meshops_array_view.natvis
  )
endif()
