#include "inference_result.h"

namespace cvops
{
    InferenceResult::InferenceResult() {
        boxes = std::vector<Box>();
        image = nullptr;
    }

    InferenceResult::~InferenceResult() {
        if (image != nullptr) {
            delete image;
        }
        if (boxes.size() > 0) {
            for (int i = 0; i < boxes.size(); i++) {
                delete boxes[i].x;
                delete boxes[i].y;
                delete boxes[i].width;
                delete boxes[i].height;
                delete boxes[i].class_id;
                delete boxes[i].class_name;
                delete boxes[i].object_id;
                delete boxes[i].confidence;
            }
        }
    }
}