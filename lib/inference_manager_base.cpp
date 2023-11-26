#include "inference_manager_base.h"
#include "inference_result.h"
#include "inference_request.h"
#include "image_utils.h"
#include "metadata_parser.h"

#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include <iostream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>
#include <chrono>

namespace cvops {
    void InferenceManagerBase::start_session(InferenceSessionRequest* session_request_ptr) {
        // std::cout << "Starting inference session..." << std::endl;
        validate_session_request(session_request_ptr);
        session_request = *session_request_ptr;
        this->ort_env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "InferenceManager"};
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::vector<std::string> eps = Ort::GetAvailableProviders();
        bool is_cuda_available = std::find(eps.begin(), eps.end(), "CUDAExecutionProvider") != eps.end();
        if (is_cuda_available) {
            OrtCUDAProviderOptions cudaOption;
            session_options.AppendExecutionProvider_CUDA(cudaOption);
        }
        const char* model_path = this->session_request.model_path;
        session = std::make_unique<Ort::Session>(this->ort_env, model_path, session_options);
        metadata = MetadataParser::parse_metadata(this->session_request.metadata);
        get_input_names();
        get_output_names();
        get_color_palette();
    }

    InferenceManagerBase::~InferenceManagerBase() {
        // For debugging
        // std::cout << "Inference Session Destroyed" << std::endl;
    }


    InferenceResult* InferenceManagerBase::infer(InferenceRequest* inference_request) {  // TODO: Add images to inference request
        cv::Mat image;
        try {
            // std::cout << "Decoding image to cv::Mat..." << std::endl;
            ImageUtils::decode_image(inference_request, &image);
        } catch (std::exception& ex) {
            std::string err_msg = "Unable to decode image inside infer method: " + std::string(ex.what());
            throw std::runtime_error(err_msg);
        }

        InferenceResult* inference_result = new InferenceResult();
        auto start = std::chrono::high_resolution_clock::now();
        inference_result->image_height = image.rows;
        inference_result->image_width = image.cols;

        std::shared_ptr<Ort::Value> input_tensor = std::make_shared<Ort::Value>(nullptr);

        // std::cout << "Preprocessing..." << std::endl;
        this->pre_process(inference_request, &image, input_tensor);

        std::vector<const char*> input_names_array;

        for (const auto& element : this->input_names_)
            input_names_array.push_back(element.c_str());

        const char* const* input_names = &input_names_array[0];

        std::vector<const char*> output_names_array;

        for (const auto& element : this->output_names_)
            output_names_array.push_back(element.c_str());

        const char* const* output_names = &output_names_array[0];
        
        // std::cout << "ONNX..." << std::endl;
        if (!input_tensor) {
            throw std::runtime_error("Input tensor is null");
        }
        std::vector<Ort::Value> output_tensor;
        try {
            output_tensor = session->Run(
                Ort::RunOptions{nullptr},
                input_names,
                input_tensor.get(),
                session->GetInputCount(),
                output_names,
                session->GetOutputCount()
            );
        } catch (Ort::Exception& ex) {
            std::string err_msg = "Unable to run inference session (code: " + std::to_string(ex.GetOrtErrorCode()) + "): " + std::string(ex.what());
            throw std::runtime_error(err_msg);
        }
        // std::cout << "Postprocessing..." << std::endl;
        this->post_process(&output_tensor, inference_result);
        if (inference_request->draw_detections) {
            ImageUtils::draw_detections(&image, inference_result, &this->color_palette_);
            std::vector<uchar> buf(10485760); // 10MB allocation for large images
            cv::imencode(".png", image, buf);
            size_t buf_size = buf.size();
            inference_result->image = new uchar[buf_size];
            std::copy(buf.begin(), buf.end(), inference_result->image);
            inference_result->image_size = (int)buf_size;
        } else {
            inference_result->image = nullptr;
        }
        auto end = std::chrono::high_resolution_clock::now();
        inference_result->milliseconds = std::chrono::duration<float, std::milli>(end - start).count();
        return inference_result;
    }

    void InferenceManagerBase::get_input_names() {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> input_names;
        size_t input_count = session->GetInputCount();
        for (size_t i = 0; i < input_count; i++)
        {
            Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
            std::string name_str = name.get();
            input_names_.push_back(name_str);
        }
    }

    void InferenceManagerBase::get_output_names() {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> output_names;
        size_t output_count = session->GetOutputCount();
        for (size_t i = 0; i < output_count; i++)
        {
            Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
            std::string name_str = name.get();
            output_names_.push_back(name_str);
        }
    }

    void InferenceManagerBase::get_color_palette()
    {
        if (this->metadata["color_palette"].empty())
            {
                for (int i = 0; i < this->metadata["classes"].size(); i++)
                {
                    cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
                    this->color_palette_.push_back(color);
                }
            }
        else
        {
            for (const auto& element : this->metadata["color_palette"])
            {
                cv::Scalar color(element[0].asInt(), element[1].asInt(), element[2].asInt());
                this->color_palette_.push_back(color);
            }
        }
    }

    void InferenceManagerBase::validate_session_request(InferenceSessionRequest* session_request_ptr) {
        if (session_request_ptr->model_path == nullptr) {
            throw std::invalid_argument("Model path is required");
        }
        if (session_request_ptr->metadata == nullptr) {
            throw std::invalid_argument("Metadata is required");
        }
        if (session_request_ptr->confidence_threshold < 0 || session_request_ptr->confidence_threshold > 1) {
            throw std::invalid_argument("Confidence threshold must be between 0 and 1");
        }
        if (session_request_ptr->iou_threshold < 0 || session_request_ptr->iou_threshold > 1) {
            throw std::invalid_argument("IOU threshold (for NMS Boxes Calculations) must be between 0 and 1");
        }
        std::filesystem::path model_path = session_request_ptr->model_path;
        if (!std::filesystem::exists(model_path)) {
            throw std::invalid_argument("User-supplied model path does not exist");
        }
    }
}
