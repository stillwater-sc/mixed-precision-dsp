add_executable(edge_detection edge_detection.cpp)
target_link_libraries(edge_detection PRIVATE sw::dsp)

add_executable(image_pipeline image_pipeline.cpp)
target_link_libraries(image_pipeline PRIVATE sw::dsp)

# Mixed-precision demo uses Universal types (bfloat16, integer<N>) that
# trigger std::bit_cast paths broken on MSVC. Skip until universal#710
# is fixed upstream.
if(NOT MSVC)
  add_executable(mixed_precision_image mixed_precision_image.cpp)
  target_link_libraries(mixed_precision_image PRIVATE sw::dsp)
endif()
