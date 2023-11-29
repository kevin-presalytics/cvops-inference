#include "inference_result.h"

#include <iostream>

namespace cvops
{
    Box::Box() {}

    Box::Box(cv::Rect base_rect, int class_id, char* class_name, int object_id, float confidence) : cv::Rect(base_rect)
    {
        class_id = class_id;
        class_name = class_name;
        object_id = object_id;
        confidence = confidence;
    }

    InferenceResult::InferenceResult() {
        boxes = nullptr;
        image = nullptr;
        boxes_count = 0;
        image_size = 0;
        image_height = 0;
        image_width = 0;
        milliseconds = 0.0f;
    }

    InferenceResult::~InferenceResult() {
        if (image) {
            try {
                delete image;
            } catch (std::exception& ex) {
                std::cout << ex.what() << std::endl;
            }
        }
        if (boxes) {
            try {
                delete [] boxes;
            } catch (std::exception& ex) {
                std::cout << ex.what() << std::endl;
            }
        }
    }
}