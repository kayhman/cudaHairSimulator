include_directories(../hairLib)
add_executable(cudada WIN32 cudada.cpp objLoader.cpp objLoader.h)
target_link_libraries(cudada  hairLib ${OPENGL_LIBRARIES})
