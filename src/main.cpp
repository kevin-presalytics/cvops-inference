#include "cvops_inference.h"
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
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

void command_line_inference(std::string model_path, std::string image_dir, std::string output_path)
{
        // Start inference session
    cvops::InferenceSessionRequest session_request= cvops::InferenceSessionRequest();
    session_request.model_platform = cvops::ModelPlatforms::YOLO;
    session_request.model_path = (char*)model_path.c_str();
    cvops::IInferenceManager* mgr_ptr = start_inference_session(&session_request);

    // Run inference
    cvops::InferenceRequest request = cvops::InferenceRequest();
    cvops::InferenceResult result = cvops::InferenceResult();

    request.bytes = 0;  //TODO: Add image bytes to request
    run_inference(mgr_ptr, &request, &result); 

    // Print results
    int objects = result.boxes_count;
    std::cout << "Number of objects detected: " << objects << std::endl;

    // Clean up
    delete mgr_ptr;
    delete &result;
    mgr_ptr = NULL;
}

size_t getFilesize(const char* filename) {
    struct stat st;
    if(stat(filename, &st) != 0) {
        return 0;
    }
    return st.st_size;   
}


int main(int argc, char** argv)
{
    try
    {
        if (argc != 5)
        {
            std::cout << "Usage: " << argv[0] << " <model_path> <image_dir> <output_path>" << std::endl;
            return 1;
        }
        // Get command line arguments
        std::string model_path = argv[1];
        std::string image_dir = argv[2];
        std::string metadata_path = argv[3];
        std::string output_path = argv[4];

        // Get list of images to run inference on
        std::vector<std::string> image_names;
        for (const auto &entry : std::filesystem::directory_iterator(image_dir))
            image_names.push_back(entry.path().string());

        struct stat sb;

        if (stat(output_path.c_str(), &sb) != 0)
        {
            std::filesystem::create_directory(output_path);
        }
        std::string metadata;
        std::ifstream metadata_file(metadata_path);
        if (metadata_file.is_open())
        {
            std::string line;
            while (std::getline(metadata_file, line))
                metadata += line + "\n";
        }
        else
        {
            std::cout << "Unable to open metadata file" << std::endl;
            return 1;
        }
        

        float confidence_threshold = 0.5f;
        float iou_threshold = 0.5f;
        char* model_path_ = new char[model_path.length() + 1];
        strcpy(model_path_, model_path.c_str());



        cvops::InferenceSessionRequest session_request = cvops::InferenceSessionRequest{};
        session_request.model_platform = cvops::ModelPlatforms::YOLO;
        session_request.model_path = model_path_;
        session_request.confidence_threshold = &confidence_threshold;
        session_request.iou_threshold = &iou_threshold;
        session_request.metadata = (char*)metadata.c_str();
        

        cvops::IInferenceManager* mgr_ptr = start_inference_session(&session_request);

        if (!mgr_ptr) {
            std::cout << "Error starting inference session" << std::endl;
            return 1;
        }


        // Run inference on each image
        for (std::string image_name : image_names)
        {
            try {
                cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
                std::vector<uchar> buf(10485760); // 10MB allocation for large images
                cv::imencode(".png", image, buf);
                size_t buf_size = buf.size();
                unsigned char* buffer = new u_char[buf_size];
                std::copy(buf.begin(), buf.end(), buffer);
                

                char* image_name_ = new char[image_name.length() + 1];
                strcpy(image_name_, image_name.c_str());

                

                cvops::InferenceRequest request = cvops::InferenceRequest{
                    .bytes = buffer,
                    .name = image_name_,
                    .size = (int)buf_size,
                    .draw_detections = true,
                };
                cvops::InferenceResult result = cvops::InferenceResult{};
                run_inference(mgr_ptr, &request, &result);

                // Write results to file
                size_t fname_start = image_name.find_last_of("//") + 1;
                size_t fname_length = image_name.find_last_of(".") - fname_start;
                std::string f_name = image_name.substr(fname_start, fname_length);
                std::string f_path = output_path + "/" + f_name + ".png";

                struct stat target_file;

                if (stat(f_path.c_str(), &target_file) == 0)
                    std::filesystem::remove(f_path);

                std::ofstream out_file;
                out_file.open (f_path);
                out_file.write(reinterpret_cast<const char*>(result.image), result.image_size);
                out_file.close();
                delete buffer;

                //for debugging
                std::cout << "Number of objects detected: " << result.boxes_count << std::endl;
            } catch (std::exception& ex) {
                std::cout << "Error running inference on image: " << image_name << std::endl;
                std::cout << ex.what() << std::endl;
                const char* err = error_message();
                std::string msg(err);
                if (msg.length() > 0)
                    std::cout << msg << std::endl;
            }
        }
    } catch (const std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
