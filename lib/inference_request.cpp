#include "inference_request.h"

#include <iostream>

namespace cvops
{
    InferenceRequest::InferenceRequest() {
        bytes = nullptr;
        name = nullptr;
        size = 0;
        draw_detections = false;
    }

    InferenceRequest::~InferenceRequest() {
        if (bytes) {
            try {
                delete [] bytes;
            } catch (std::exception& ex) {
                std::cout << ex.what() << std::endl;
            }
        }
        if (name) {
            try {
                delete [] name;
            } catch (std::exception& ex) {
                std::cout << ex.what() << std::endl;
            }
        }
    }
}