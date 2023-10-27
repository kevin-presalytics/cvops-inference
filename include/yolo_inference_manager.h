#ifndef YOLO_INFERENCE_MANAGER_H
#define YOLO_INFERENCE_MANAGER_H

#include "inference_manager_base.h"
#include "inference_result.h"
#include "inference_request.h"

#include "onnxruntime_cxx_api.h"

namespace cvops 
{
    class YoloInferenceManager : public InferenceManagerBase {
        public:
            YoloInferenceManager() : InferenceManagerBase() {}
        protected:
            virtual InferenceResult post_process(std::vector<Ort::Value>* output_tensor);
            virtual void pre_process(InferenceRequest* inference_request, cv::Mat* input_image, std::vector<Ort::Value>* input_tensor);
    };
}

#endif // YOLO_INFERENCE_MANAGER_H
