cmake_minimum_required(VERSION 3.5)

if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

#_add_project_definitions(${PROJNAME})

option(USE_PACKMAN "Enable to use packman dependencies where possible" OFF)

if (EXISTS "${BASE_DIRECTORY}/nvpro_core/OV")
  message(STATUS "Packman is available, toggle USE_PACKMAN to enable")
endif()

# If Pybind11 is not found, this project will be excluded
find_package(Pybind11)

if(USE_PACKMAN)
  set(Python3_ROOT_DIR "${BASE_DIRECTORY}/nvpro_core/OV/downloaded/python")
  # Stop looking fore newer versions in other locations
  set(Python3_FIND_STRATEGY LOCATION)
  # On Windows, ignore a potentially preinstalled newer version of Python
  set(Python3_FIND_REGISTRY NEVER)
endif()

find_package(Python3 COMPONENTS Development Interpreter)

if(USE_PACKMAN)
  # Install packman-packager
  execute_process(COMMAND "${BASE_DIRECTORY}/nvpro_core/OV/packman/${PACKMAN_COMMAND}" install packman-packager@0.10 -l ${CMAKE_CURRENT_BINARY_DIR}/packman-packager
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                  RESULT_VARIABLE USD_PACKMAN_RESULT)
  set(MICROMESH_PYTHON_VERSION 100.0.2 CACHE STRING "100.0.2")  # <bary version.major version.minor version>
endif()

set(PROJECT_NAME "micromesh_python")
if(Python3_FOUND AND EXISTS ${PYBIND11_INCLUDE_DIR})
  #-------------------------------------------------------------------------
  # Library
  #-------------------------------------------------------------------------

  # Sources and headers for the library
  file(GLOB SOURCE_FILES *.cpp)
  file(GLOB HEADER_FILES *.h *.hpp)

  # Attempt to match MSVC warnings
  if(UNIX)
      set_property(SOURCE ${SOURCE_FILES} APPEND_STRING PROPERTY COMPILE_FLAGS -Wconversion)
  endif()

  _add_package_VMA()
  #_add_package_VulkanSDK()

  # Add library
  add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

  # Udjust python lib name and suffix (if required)
  set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
  if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
  endif()

  # Add thirdparty  
  target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PRIVATE ${BASE_DIRECTORY}/nvpro_core/third_party/stb)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_tool/src)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_core/include)
  #target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_tool/in)

  target_link_directories(${PROJECT_NAME} PUBLIC "${Python3_RUNTIME_LIBRARY_DIRS}")

  # Dependencies
  target_link_libraries (${PROJECT_NAME} PUBLIC meshops_bake meshops_core micromesh_core micromesh_tool_lib bary_core bary_utils "${Python3_LIBRARIES}")

  if (WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /permissive /bigobj)
  else()
    target_compile_options(${PROJECT_NAME} PRIVATE -fpermissive)
  endif()

  # Adding extra source files from nvpro_core
  target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")
  target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES})
  source_group("nvpro_core"     FILES ${COMMON_SOURCE_FILES})

  install(TARGETS ${PROJECT_NAME})

  # Utility scripts
  if (WIN32)
    set(SET_ENV_CMD "set")
    set(PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}/Release;%PYTHONPATH%")
  else()
    set(SET_ENV_CMD "export")
    set(PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}:$PYTHONPATH")
  endif()
  configure_file("install_jupyterlab.cmd.in" "${CMAKE_CURRENT_BINARY_DIR}/install_jupyterlab.cmd" @ONLY)
  configure_file("start_jupyterlab.cmd.in" "${CMAKE_CURRENT_BINARY_DIR}/start_jupyterlab.cmd" @ONLY)

  # package target to create and publish a packman package (append -f to command below to force publish)
  if (WIN32)
    set(BUILD_PLATFORM "windows-x86_64")
    set(MODULE_EXT "pyd")
  else()
    set(BUILD_PLATFORM "linux-x86_64")
    set(MODULE_EXT "so")
  endif()

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/module)
  if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${PROJECT_NAME}.${MODULE_EXT}
              ${CMAKE_CURRENT_BINARY_DIR}/module/${PROJECT_NAME}.${MODULE_EXT}
    )
  else()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.${MODULE_EXT}
              ${CMAKE_CURRENT_BINARY_DIR}/module/${PROJECT_NAME}.${MODULE_EXT}
  )
  endif()

  set(PYTHON_VERSION py37)
  foreach(CONFIG_TYPE ${CMAKE_CONFIGURATION_TYPES})
    string(TOLOWER "${CONFIG_TYPE}" BUILD_TYPE)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/package.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/package.${CONFIG_TYPE}.py
      @ONLY
    )
  endforeach()

  if (USE_PACKMAN)
    set(PACKAGE_TARGET micromesh_python_package)
    add_custom_target(${PACKAGE_TARGET})
    set_target_properties(${PACKAGE_TARGET} PROPERTIES FOLDER packman-packages)
    add_custom_command(TARGET ${PACKAGE_TARGET}
      COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_BINARY_DIR}/package.$<CONFIG>.py -b ${CMAKE_CURRENT_BINARY_DIR}/package -r cloudfront
    )
  endif()
else()
  message(WARNING "Skipping project ${PROJECT_NAME} due to missing requirements (python and/or pybind11)")
endif()