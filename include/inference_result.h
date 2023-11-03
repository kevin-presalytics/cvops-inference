#ifndef INFERENCE_RESULT_H
#define INFERENCE_RESULT_H

#include "inference_request.h"

#include <vector>

namespace cvops {
    struct Box {
        int x;
        int y;
        int width;
        int height;
        int class_id;
        char* class_name;
        int object_id;
        float confidence;
    };

    struct InferenceResult {
        Box* boxes;
        int boxes_count;
        unsigned char* image;
        int image_size;
        int image_width;
        int image_height;
        InferenceResult();
        ~InferenceResult();
    };

    
}

#endif // INFERENCE_RESULT_H