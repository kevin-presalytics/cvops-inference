#ifndef INFERENCE_RESULT_H
#define INFERENCE_RESULT_H

#include "inference_request.h"

#include <opencv2/opencv.hpp>

namespace cvops {
    struct Box : public cv::Rect2i {
        int class_id;
        char* class_name;
        int object_id;
        float confidence;
        Box();
        Box(cv::Rect base_rect, int class_id, char* class_name, int object_id, float confidence);
    };

    struct InferenceResult {
        Box* boxes;
        int boxes_count;
        unsigned char* image;
        int image_size;
        int image_width;
        int image_height;
        float milliseconds;
        InferenceResult();
        ~InferenceResult();
    };

    
}

#endif // INFERENCE_RESULT_H