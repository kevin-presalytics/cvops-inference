#ifndef INFERENCE_REQUEST_H
#define INFERENCE_REQUEST_H

namespace cvops
{
    struct InferenceRequest
    {
        unsigned char* bytes;
        char* name;
        int size;
        bool draw_detections;
        InferenceRequest();
        ~InferenceRequest();
    };
}

#endif // INFERENCE_REQUEST_H