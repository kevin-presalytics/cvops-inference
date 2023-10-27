#include "c_api.h"
#include "inference_session_request.h"
#include "inference_manager_interface.h"

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <filesystem>
#include <cstring>

void command_line_inference(std::string model_path, std::string image_dir, std::string output_path)
{
        // Start inference session
    cvops::InferenceSessionRequest session_request= cvops::InferenceSessionRequest();
    session_request.model_platform = cvops::ModelPlatforms::YOLO;
    session_request.model_path = (char*)model_path.c_str();
    cvops::IInferenceManager* mgr_ptr = start_inference_session(&session_request);

    // Run inference
    cvops::InferenceRequest request = cvops::InferenceRequest();
    request.bytes = 0;  //TODO: Add image bytes to request
    cvops::InferenceResult result = run_inference(mgr_ptr, &request);

    // Print results
    int objects = result.boxes.size();
    std::cout << "Number of objects detected: " << objects << std::endl;

    // Clean up
    delete mgr_ptr;
    delete &result;
    mgr_ptr = NULL;
}


int main(int argc, char** argv)
{
    try
    {
        if (argc != 4)
        {
            std::cout << "Usage: " << argv[0] << " <model_path> <image_dir> <output_path>" << std::endl;
            return 1;
        }
        // Get command line arguments
        std::string model_path = argv[1];
        std::string image_dir = argv[2];
        std::string output_path = argv[3];

        // Get list of images to run inference on
        std::vector<std::string> image_names;
        for (const auto &entry : std::filesystem::directory_iterator(image_dir))
            image_names.push_back(entry.path().string());

        // get class names from file

        char class_names = {};

        float confidence_threshold = 0.9f;
        float iou_threshold = 0.5f;
        char* model_path_ = new char[model_path.length() + 1];
        strcpy(model_path_, model_path.c_str());

        cvops::InferenceSessionRequest session_request = cvops::InferenceSessionRequest{};
        session_request.model_platform = cvops::ModelPlatforms::YOLO;
        session_request.model_path = model_path_;
        session_request.confidence_threshold = &confidence_threshold;
        session_request.iou_threshold = &iou_threshold;
        

        cvops::IInferenceManager* mgr_ptr = start_inference_session(&session_request);

        if (!mgr_ptr) {
            std::cout << "Error starting inference session" << std::endl;
            return 1;
        }


        // Run inference on each image
        for (std::string image_name : image_names)
        {
            std::ifstream input( image_name, std::ios::binary );

            // copies all data into buffer
            std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});

            char* image_name_ = new char[image_name.length() + 1];
            strcpy(image_name_, image_name.c_str());
            

            cvops::InferenceRequest request = cvops::InferenceRequest{
                .bytes = buffer.data(),
                .name = image_name_,
                .size = (int)buffer.size(),
                .draw_detections = true,
            };
            cvops::InferenceResult result = run_inference(mgr_ptr, &request);
        }
    } catch (const std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
