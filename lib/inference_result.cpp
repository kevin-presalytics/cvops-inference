#include "inference_result.h"

#include <iostream>

namespace cvops
{
    InferenceResult::InferenceResult() {
        boxes = nullptr;
        image = nullptr;
        boxes_count = 0;
        image_size = 0;
        image_height = 0;
        image_width = 0;
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