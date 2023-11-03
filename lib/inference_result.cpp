#include "inference_result.h"

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
        if (image != nullptr) {
            delete image;
        }
        if (boxes != nullptr) {
            delete boxes;
        }
    }
}