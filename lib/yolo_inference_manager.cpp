#include "yolo_inference_manager.h"
#include "inference_result.h"
#include "inference_request.h"

#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>


namespace cvops
{
    InferenceResult YoloInferenceManager::post_process(std::vector<Ort::Value>* output_tensor) { // TODO: write post processing code
        std::cout << "Post processing YOLO inference result..." << std::endl;
        InferenceResult inference_result  = InferenceResult{};
        return inference_result;
    }
    
    void YoloInferenceManager::pre_process(InferenceRequest* inference_request, cv::Mat* image, std::vector<Ort::Value>* input_tensor) { // TODO: write pre processing code
        std::cout << "Pre processing YOLO inference request..." << std::endl;
        cv::Mat resized_image, float_image, recolored_image;
        cv::cvtColor(*image, recolored_image, cv::COLOR_BGR2RGB);
        std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
        
        resize_and_letterbox_image(recolored_image, resized_image);

        inputTensorShape[2] = resized_image.rows;
        inputTensorShape[3] = resized_image.cols;

        resized_image.convertTo(float_image, CV_32FC3, 1 / 255.0);
        float* blob = new float[float_image.cols * float_image.rows * float_image.channels()];
        cv::Size floatImageSize {float_image.cols, float_image.rows};

        // hwc -> chw
        std::vector<cv::Mat> chw(float_image.channels());
        for (int i = 0; i < float_image.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
        }
        cv::split(float_image, chw);

        size_t input_tensor_size = 1;;

        for (const auto& element : inputTensorShape)
            input_tensor_size *= element;

        std::vector<float> inputTensorValues(blob, blob + input_tensor_size);


        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        input_tensor->push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, inputTensorValues.data(), input_tensor_size,
                inputTensorShape.data(), inputTensorShape.size()
        ));
    }
}