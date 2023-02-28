# All sources and headers
file(GLOB SOURCE_FILES *.*)

# Make an automatic library - will be static or dynamic based on user setting
add_library(tinyexr ${SOURCE_FILES})

# We need this directory, and users of our library will need it too
target_include_directories(tinyexr PUBLIC .)

# Fix the miniz.c warning about large files with fopen() etc.
if(UNIX)
    target_compile_definitions(tinyexr PRIVATE _LARGEFILE64_SOURCE)
endif()