#include "cvops_inference.h"
#include "inference_manager_factory.h"
#include "inference_manager_interface.h"
#include "model_platforms.h"
#include "inference_result.h"
#include "inference_request.h"
#include "inference_session_request.h"
#include "image_utils.h"
#include "metadata_parser.h"

#include <memory>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

    std::string err_msg;

    std::vector<std::shared_ptr<cvops::IInferenceManager>> managers;

    void wrap_exception(std::exception& ex) {
        err_msg = ex.what();
        std::cout << err_msg << std::endl;
    }

    cvops::IInferenceManager* start_inference_session(cvops::InferenceSessionRequest* request) { 
        try {
            std::unique_ptr<cvops::InferenceManagerFactory> factory = std::make_unique<cvops::InferenceManagerFactory>();
            std::shared_ptr<cvops::IInferenceManager> manager = factory->create_inference_manager(request);
            managers.push_back(manager);
            return manager.get();
        } catch (std::exception& ex) {
            wrap_exception(ex);
            return nullptr;
        }
    }

    cvops::InferenceResult* run_inference(cvops::IInferenceManager* inference_manager, cvops::InferenceRequest* inference_request) 
    {
        // std::cout << "Running inference..." << std::endl;
        cvops::InferenceResult* inference_result = nullptr;
        if (inference_request && inference_manager)
        {
            try {
                if (!(inference_request->bytes))
                    throw std::runtime_error("Inference request bytes are null");
                if (!(inference_request->name))
                    throw std::runtime_error("Inference request name is null");
                if (!(inference_request->size))
                    throw std::runtime_error("Inference request size is null");
                if (inference_request->size <= 0)
                    throw std::runtime_error("Inference request size is less than or equal to zero");
                
                // std::cout << "Running infer method..." << std::endl;
                inference_result = inference_manager->infer(inference_request);
            } catch (std::exception& ex) {
                wrap_exception(ex);
            }
        }
        return inference_result;
    }

    void end_inference_session(cvops::IInferenceManager* inference_manager) {
        try {
            // For debugging
            //std::cout << "Ending inference session..." << std::endl;
            size_t manager_count = managers.size();
            for (int i = 0; i < manager_count; i++)
            {
                std::shared_ptr<cvops::IInferenceManager> manager = managers[i];
                if (manager.get() == inference_manager) {
                    // Smart pointer deletes inference manager after this block
                    managers.erase(managers.begin() + i);
                    break;
                }
            }
        } catch (std::exception& ex) {
            wrap_exception(ex);
        }
    }

    std::shared_ptr<std::vector<cv::Scalar>> cv_colors = nullptr;

    void set_color_palette(char* color_palette) {
        std::vector<cv::Scalar> palette = cvops::MetadataParser::parse_color_palette(color_palette);
        cv_colors = std::make_shared<std::vector<cv::Scalar>>(palette);
    }

    void free_color_palette() {
        cv_colors->clear();
        cv_colors = nullptr;
    }

    void render_inference_result(cvops::InferenceResult* inference_result, void* image_data, int image_height, int image_width, int num_channels) 
    {
        int cv_data_type = CV_8UC3;
        if (num_channels == 1)
            cv_data_type = CV_8UC1;
        if (num_channels == 4)
            cv_data_type = CV_8UC4;
        cv::Mat raw_image(image_height, image_width, cv_data_type, image_data);
        cvops::ImageUtils::draw_detections(&raw_image, inference_result, cv_colors.get());
    }

    void dispose_inference_result(cvops::InferenceResult* inference_result) {
        try
        {
            if (inference_result)
                delete inference_result;
        } catch (std::exception& ex) {
            wrap_exception(ex);
        }
    }

    const char* error_message() {
        return err_msg.c_str();
    }

#ifdef __cplusplus
}
#endif