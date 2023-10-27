#ifndef INFERENCE_REQUEST_H
#define INFERENCE_REQUEST_H

namespace cvops
{
    struct InferenceRequest
    {
        char* bytes;
        char* name;
        int size;
        bool draw_detections;
    };
}

#endif // INFERENCE_REQUEST_H