include_directories(../hairLib)
add_executable(cudada WIN32 cudada.cpp)
target_link_libraries(cudada  hairLib ${OPENGL_LIBRARIES})
