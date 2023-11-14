#ifndef INFERENCE_SESSION_REQUEST_H
#define INFERENCE_SESSION_REQUEST_H

#include "model_platforms.h"

namespace cvops {
    struct InferenceSessionRequest{
        ModelPlatforms model_platform;
        char* model_path;
        char* metadata;
        float confidence_threshold;
        float iou_threshold;
    };
}

#endif // INFERENCE_SESSION_REQUEST_H