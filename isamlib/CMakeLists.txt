cmake_minimum_required(VERSION 2.6)

# find all source and header files
file(GLOB SRCS RELATIVE "${PROJECT_SOURCE_DIR}/isamlib" "*.cpp")

# isamlib
add_library(isamlib ${SRCS})
set_property(TARGET isamlib PROPERTY OUTPUT_NAME isam)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#        "~/tools/g2o/cmake_modules"
#        )
# simple way of finding CHOLMOD
#set(Cholmod_DIR /usr/share/doc/libsuitesparse-dev)
#find_package(Cholmod REQUIRED)
#include_directories(${CHOLMOD_INCLUDES})
#target_link_libraries(isamlib ${CHOLMOD_LIBRARIES})
include_directories("/usr/include/suitesparse")
target_link_libraries(isamlib cholmod)
# install library
install(TARGETS isamlib
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
# install header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/isam/ DESTINATION include/isam
          FILES_MATCHING PATTERN "*.h" PATTERN ".svn" EXCLUDE)

