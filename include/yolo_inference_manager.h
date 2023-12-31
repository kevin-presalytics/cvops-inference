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
            YoloInferenceManager();
            ~YoloInferenceManager();
        protected:
            virtual void post_process(std::vector<Ort::Value>* output_tensor, InferenceResult* inference_result);
            virtual void pre_process(InferenceRequest* inference_request, cv::Mat* input_image, std::shared_ptr<Ort::Value> input_tensor);
        private:
            float* blob;
    };
}

#endif // YOLO_INFERENCE_MANAGER_H
