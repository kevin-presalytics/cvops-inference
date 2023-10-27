#include "inference_manager_factory.h"
#include "inference_manager_interface.h"
#include "model_platforms.h"
#include "inference_result.h"
#include "inference_request.h"
#include "inference_session_request.h"

#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

    std::string err_msg;

    void wrap_exception(std::exception& ex) {
        err_msg = ex.what();
        std::cout << err_msg << std::endl;
    }

    cvops::IInferenceManager* start_inference_session(cvops::InferenceSessionRequest* request) { 
        try {
            std::cout << "Starting inference session..." << std::endl;
            std::unique_ptr<cvops::InferenceManagerFactory> factory = std::make_unique<cvops::InferenceManagerFactory>();
            cvops::IInferenceManager* mgr_ptr = factory->create_inference_manager(request);
            return mgr_ptr;
        } catch (std::exception& ex) {
            wrap_exception(ex);
            return nullptr;
        }
    }

    cvops::InferenceResult run_inference(cvops::IInferenceManager* inference_manager, cvops::InferenceRequest* inference_request) {
        try {
            std::cout << "Running inference..." << std::endl;
            cvops::InferenceResult result = inference_manager->infer(inference_request);
            return result;
        } catch (std::exception& ex) {
            wrap_exception(ex);
            return cvops::InferenceResult();
        }
    }

    char error_message() {
        return *err_msg.c_str();
    }

#ifdef __cplusplus
}
#endif