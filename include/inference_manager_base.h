#ifndef INFERENCE_MANAGER_BASE_H
#define INFERENCE_MANAGER_BASE_H

#include "inference_session_request.h"
#include "inference_request.h"
#include "inference_manager_interface.h"

#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include <string>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace cvops {
    class InferenceManagerBase : public IInferenceManager {
        public:
            InferenceManagerBase() : IInferenceManager() {};
            virtual void start_session(InferenceSessionRequest* session_request_ptr);
            virtual InferenceResult* infer(InferenceRequest* request);
            virtual ~InferenceManagerBase();

        protected:
            // override in class implementations
            virtual void post_process(std::vector<Ort::Value>* output_tensor, InferenceResult* inference_result) = 0;
            virtual void pre_process(InferenceRequest* inference_request, cv::Mat* input_image, std::shared_ptr<Ort::Value> input_tensor) = 0;

            // class properties
            InferenceSessionRequest session_request;
            std::unique_ptr<Ort::Session> session;
            Json::Value metadata;
            std::vector<std::string> input_names_;
            std::vector<std::string> output_names_;
            std::vector<cv::Scalar> color_palette_;

        private:
            void get_input_names();
            void get_output_names();
            void get_metadata(char* metadata_json_ptr);
            void get_color_palette();
            void validate_session_request(InferenceSessionRequest* session_request_ptr);
    };
}

#endif // INFERENCE_MANAGER_BASE_H