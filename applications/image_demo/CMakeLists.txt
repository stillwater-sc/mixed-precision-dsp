add_executable(edge_detection edge_detection.cpp)
target_link_libraries(edge_detection PRIVATE sw::dsp)

add_executable(image_pipeline image_pipeline.cpp)
target_link_libraries(image_pipeline PRIVATE sw::dsp)

add_executable(mixed_precision_image mixed_precision_image.cpp)
target_link_libraries(mixed_precision_image PRIVATE sw::dsp)
