#include "inference_result.h"

#include <iostream>

namespace cvops
{
    Box::Box(cv::Rect base_rect, int class_id, char* class_name, int object_id, float confidence) : cv::Rect(base_rect)
    {
        this->class_id = class_id;
        this->class_name = class_name;
        this->object_id = object_id;
        this->confidence = confidence;
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

    InferenceResult &InferenceResult::operator=(const InferenceResult &other) {
        if (this != &other) {
            this->boxes_count = other.boxes_count;
            this->image_size = other.image_size;
            this->image_height = other.image_height;
            this->image_width = other.image_width;
            this->milliseconds = other.milliseconds;
            void* boxes_ptr = operator new[](sizeof(Box) * other.boxes_count);
            this->boxes = static_cast<Box*>(boxes_ptr);
            memcpy(this->boxes, other.boxes, other.boxes_count * sizeof(Box));
            if (other.image) {
                this->image = new unsigned char[other.image_size];
                memcpy(this->image, other.image, other.image_size);
            } else {
                this->image = nullptr;
            }
        }
        return *this;
    }

    InferenceResult InferenceResult::clone() {
        InferenceResult result = InferenceResult();
        result.boxes_count = this->boxes_count;
        result.image_size = this->image_size;
        result.image_height = this->image_height;
        result.image_width = this->image_width;
        result.milliseconds = this->milliseconds;
        void* boxes_ptr = operator new[](sizeof(Box) * this->boxes_count);
        result.boxes = static_cast<Box*>(boxes_ptr);
        memcpy(result.boxes, this->boxes, this->boxes_count * sizeof(Box));
        if (this->image) {
            result.image = new unsigned char[this->image_size];
            memcpy(result.image, this->image, this->image_size);
        } else {
            result.image = nullptr;
        }
        return result;
    } 
}