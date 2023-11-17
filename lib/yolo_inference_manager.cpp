#include "yolo_inference_manager.h"
#include "inference_result.h"
#include "inference_request.h"
#include "metadata_parser.h"
#include "image_utils.h"

#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>


template<typename T>
void image_to_blob(cv::Mat& image, T &blob) {
    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                blob[c * width * height + h * width + w] = image.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}


namespace cvops
{
    void YoloInferenceManager::post_process(std::vector<Ort::Value>* output_tensor, InferenceResult* inference_result) 
    {
        // For debugging
        // std::cout << "Post processing YOLO inference result..." << std::endl;

        auto* output_begin_ptr = (*output_tensor)[0].GetTensorData<float>();

        float* output_data = (*output_tensor)[0].GetTensorMutableData<float>();

        std::vector<int64_t> output_shape = (*output_tensor)[0].GetTensorTypeAndShapeInfo().GetShape();

        int output_rows = output_shape[2];
        int detection_resultant_size = output_shape[1];

        // Read tensor in to matrix and transpose (only for YOLOv8 models)
        cv::Mat output_matrix = cv::Mat(detection_resultant_size, output_rows, CV_32F, output_data).t();

        double confidence_threshold = (double)this->session_request.confidence_threshold;
        double nms_threshold = (double)this->session_request.iou_threshold;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        size_t class_count = (size_t)this->metadata["classes"].size();
        cv::Size current_image_size = MetadataParser::get_image_size(this->metadata);
        cv::Size target_image_size = cv::Size(inference_result->image_width, inference_result->image_height);
        

        for (int row = 0; row < output_rows; row++)
        {
            float* scores_ptr = &output_matrix.at<float>(row, 4);
            std::vector<float> scores(scores_ptr, scores_ptr + class_count);
            //
            //cv::Mat scores(1, (size_t)class_count, CV_32F, pdata + 4);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > confidence_threshold)
            {
                std::string object_class = this->metadata["classes"][std::to_string(class_id_point.x)].asString();
                // For Debugging
                // std::cout << object_class << ", Confidence: " << max_class_score << std::endl;
                std::vector<float> rect_data(&output_matrix.at<float>(row, 0), scores_ptr);
                int center_x = rect_data[0];
                int center_y = rect_data[1];
                int width = rect_data[2];
                int height = rect_data[3];
                int top = center_y - height / 2;
                int left = center_x - width / 2;

                cv::Rect rect_ = cv::Rect(left, top, width, height);

                ImageUtils::resize_rect(&rect_, &current_image_size, &target_image_size);
                boxes.push_back(rect_);
                confidences.push_back((float)max_class_score);
                class_ids.push_back(class_id_point.x);

     
            }

        }

        std::vector<int> nms_result;
        // Correct invalid user thresholds
        if (confidence_threshold < 0 || confidence_threshold > 1)
        {
            confidence_threshold = 0.5;
        }
        if (nms_threshold < 0 || nms_threshold > 1)
        {
            nms_threshold = 0.4;
        }
        try {
            // Removes overlapping boxes
            cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, nms_result);
            inference_result->boxes = new Box[nms_result.size()];
            inference_result->boxes_count = nms_result.size();
        } catch (std::exception& ex) {
            // NMBSBoxes throws an exception if no boxes are found
            std::cout << ex.what() << std::endl;
            inference_result->boxes = new Box[boxes.size()];
            inference_result->boxes_count = boxes.size();
        }
        

        for (int j = 0; j < inference_result->boxes_count; j++)
        {
            int idx = j;
            if (nms_result.size() > 0)
            {
                idx = nms_result[j];
            }
            std::string class_name = this->metadata["classes"][std::to_string(class_ids[idx])].asString();
            inference_result->boxes[j].x = boxes[idx].x;
            inference_result->boxes[j].y = boxes[idx].y;
            inference_result->boxes[j].width = boxes[idx].width;
            inference_result->boxes[j].height = boxes[idx].height;
            inference_result->boxes[j].class_id = class_ids[idx];
            inference_result->boxes[j].object_id = j;
            inference_result->boxes[j].confidence = confidences[idx];
            inference_result->boxes[j].class_name = (char*)this->metadata["classes"][std::to_string(class_ids[idx])].asCString();
            // for debugging

            // std::cout << "Class: " << class_name << ", Confidence: " << confidences[idx] << std::endl;
        }
    }
    
    void YoloInferenceManager::pre_process(InferenceRequest* inference_request, cv::Mat* image, std::shared_ptr<Ort::Value> input_tensor) { // TODO: write pre processing code
        // For Debugging
        // std::cout << "Pre processing YOLO inference request..." << std::endl;
        cv::Mat input_image;
        cv::Size model_default_size = MetadataParser::get_image_size(this->metadata);

        ImageUtils::resize_and_pad_image(*image, input_image, model_default_size);

        std::vector<int64_t> input_tensor_shape = {1, 3, model_default_size.height, model_default_size.width};

        size_t blob_size = input_image.cols * input_image.rows * 3;
        
        // 3 channels: model takes colored images, not grayscale
        std::unique_ptr<float[]> blob(new float[blob_size]);

        //std::vector<float> input = std::vector<float>(blob, blob + blob_size);

        // transfer image to blob
        image_to_blob(input_image, blob);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        *input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, 
            blob.get(), 
            blob_size,
            input_tensor_shape.data(), 
            input_tensor_shape.size()
        );
    }
}