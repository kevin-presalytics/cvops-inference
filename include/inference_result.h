#ifndef INFERENCE_RESULT_H
#define INFERENCE_RESULT_H

#include <vector>

namespace cvops {
    struct Box {
        int* x;
        int* y;
        int* width;
        int* height;
        int* class_id;
        char* class_name;
        int* object_id;
        float* confidence;
    };

    struct InferenceResult {
        std::vector<Box> boxes;
        char* image;
        InferenceResult();
        ~InferenceResult();
    };

    
}

#endif // INFERENCE_RESULT_H