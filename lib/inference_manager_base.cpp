#include "inference_manager_base.h"
#include "inference_result.h"
#include "inference_request.h"

#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include <iostream>
#include <memory>
#include <mutex>

namespace cvops {
    void InferenceManagerBase::start_session(InferenceSessionRequest* session_request) {
        std::cout << "Starting inference session..." << std::endl;
        session_request_ptr_ = session_request;
        static Ort::Env ort_env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "InferenceManager"};
        Ort::SessionOptions session_options; // Fix problem 2
        session_options.SetIntraOpNumThreads(1);
        const char* model_path = session_request_ptr_->model_path;
        session = std::make_unique<Ort::Session>(ort_env, model_path, session_options);
        get_metadata(session_request_ptr_->metadata);
        
        // model_metadata = session->GetModelMetadata();
        // if (session_request->class_names.size() == 0)
        // {
        //     for (int i = 0; i < model_metadata.GetOutputCount(); i++)
        //     {
        //         session_request->class_names.push_back(model_metadata.GetOutputName(i, ort_env));
        //     }
        // }
    }

    InferenceManagerBase::~InferenceManagerBase()
    {
        std::cout << "Destroying inference session..." << std::endl;
        session = std::unique_ptr<Ort::Session>(nullptr);
        delete session_request_ptr_;
    }

    void InferenceManagerBase::end_session() {
        std::cout << "Ending inference session..." << std::endl;
        session = std::unique_ptr<Ort::Session>(nullptr);
    }


    InferenceResult InferenceManagerBase::infer(InferenceRequest* inference_request) {  // TODO: Add images to inference request
        cv::Mat image;
        decode_image(inference_request, &image);

        std::vector<Ort::Value> input_tensor;
        pre_process(inference_request, &image, &input_tensor);
        
        std::vector<Ort::Value> output_tensor = session->Run(
            Ort::RunOptions{nullptr},
            get_input_names().data(),
            &input_tensor.data()[0],
            session->GetInputCount(),
            get_output_names().data(),
            session->GetOutputCount()
        );

        InferenceResult inference_result = post_process(&output_tensor);
        if (inference_request->draw_detections)
            draw_detections(&image, &inference_result);
        delete &image;
        delete &input_tensor;
        delete &output_tensor;
        return inference_result;
    }

    std::vector<const char*> InferenceManagerBase::get_input_names() {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> input_names;
        size_t input_count = session->GetInputCount();
        for (size_t i = 0; i < input_count; i++)
        {
            Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(name.get());
        }
        return input_names;
    }

    std::vector<const char*> InferenceManagerBase::get_output_names() {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> output_names;
        size_t output_count = session->GetOutputCount();
        for (size_t i = 0; i < output_count; i++)
        {
            Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(name.get());
        }
        return output_names;
    }

    Ort::ModelMetadata InferenceManagerBase:: get_model_metadata() {
        return this->session->GetModelMetadata();
    }

    void InferenceManagerBase::decode_image(InferenceRequest* inference_request, cv::Mat* image) {
        *image = cv::imdecode(cv::Mat(1, inference_request->size, CV_8UC1, inference_request->bytes), cv::IMREAD_UNCHANGED);
    }

    void InferenceManagerBase::resize_and_letterbox_image( 
        const cv::Mat& image, 
        cv::Mat& out_image,
        const cv::Size& target_size,
        const cv::Scalar& color,
        int stride)
    {
        cv::Size input_shape = image.size();
        cv::Size shape = image.size();


        float r = std::min((float)target_size.height / (float)input_shape.height,
                        (float)target_size.width / (float)input_shape.width);

        float ratio[2] {r, r};

        int pad[2] {(int)std::round((float)shape.width * r),
                        (int)std::round((float)shape.height * r)};

        auto dw = (float)(target_size.width - pad[0]);
        auto dh = (float)(target_size.height - pad[1]);

        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);

        dw /= 2.0f;
        dh /= 2.0f;

        if (shape.width != pad[0] && shape.height != pad[1])
        {
            cv::resize(image, out_image, cv::Size(pad[0], pad[1]));
        }

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        cv::copyMakeBorder(out_image, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    void InferenceManagerBase::draw_detections(cv::Mat* image_ptr, InferenceResult* inference_result_ptr)
    {
        // std::cout << "Drawing detections..." << std::endl;
        // cv::Mat image = *image_ptr;
        // InferenceResult inference_result = *inference_result_ptr;
        // // TODO: Create color palette on session starts
        // for (int i = 0; i < inference_result.boxes.size(); i++)
        // {
        //     cv::Rect box = inference_result.boxes[i]; // TODO: covert BoX to cv::Rect
        //     cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2);
        //     cv::putText(image, session_request_ptr_->class_names[i], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        // }
    }

    std::vector<std::string> InferenceManagerBase::get_class_names() {
        Ort::ModelMetadata model_metadata = this->session->GetModelMetadata();
        Ort::AllocatorWithDefaultOptions ort_alloc;
        Ort::AllocatedStringPtr search = model_metadata.LookupCustomMetadataMapAllocated("names", ort_alloc);
        std::vector<std::string> class_names;

        if (search != nullptr) {
            const std::array<const char*, 1> list_classes = { search.get() };
            //classNames = split(std::string(list_classes[0]), ",");
        //     for (int i = 0; i < classNames.size(); i++)
        //         std::cout << "\t" << i << " | " << classNames[i] << std::endl;
        }
        return class_names;
    }

    void InferenceManagerBase::get_metadata(char* metadata_json_ptr) {
        Json::CharReaderBuilder builder;
        Json::CharReader* reader = builder.newCharReader();
        std::string errors;
        bool parsingSuccessful = reader->parse(metadata_json_ptr, metadata_json_ptr + strlen(metadata_json_ptr), &metadata, &errors);
        if (!parsingSuccessful)
        {
            std::cout << "Failed to parse metadata JSON" << std::endl;
            std::cout << errors << std::endl;
        }
    }

}
