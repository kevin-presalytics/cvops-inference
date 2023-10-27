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
            virtual void start_session(InferenceSessionRequest* session_request);
            virtual InferenceResult infer(InferenceRequest* request);
            virtual void end_session();
            virtual ~InferenceManagerBase();

        protected:
            // override in class implementations
            virtual InferenceResult post_process(std::vector<Ort::Value>* output_tensor) = 0;
            virtual void pre_process(InferenceRequest* inference_request, cv::Mat* input_image, std::vector<Ort::Value>* input_tensor) = 0;

            // Utility functions for derived classes
            virtual void decode_image(InferenceRequest* inference_request, cv::Mat* image);

            virtual void resize_and_letterbox_image(
                const cv::Mat& image, 
                cv::Mat& out_image,
                const cv::Size& target_size= cv::Size(640, 640),
                const cv::Scalar& color = cv::Scalar(114, 114, 114),
                int stride = 32
            );

            virtual void draw_detections(cv::Mat* image, InferenceResult* inference_result);
            virtual std::vector<std::string> get_class_names();
            virtual Ort::ModelMetadata get_model_metadata();
            
            
            // class properties
            InferenceSessionRequest* session_request_ptr_;
            std::unique_ptr<Ort::Session> session;
            std::vector<std::string> class_names;
            Json::Value metadata;

        private:
            std::vector<const char*> get_input_names();
            std::vector<const char*> get_output_names();
            void get_metadata(char* metadata_json_ptr);
    };
}

#endif // INFERENCE_MANAGER_BASE_H