#include "cvops_inference.h"
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

    std::vector<std::shared_ptr<cvops::IInferenceManager>> managers;

    void wrap_exception(std::exception& ex) {
        err_msg = ex.what();
        std::cout << err_msg << std::endl;
    }

    cvops::IInferenceManager* start_inference_session(cvops::InferenceSessionRequest* request) { 
        try {
            std::cout << "Starting inference session..." << std::endl;
            std::unique_ptr<cvops::InferenceManagerFactory> factory = std::make_unique<cvops::InferenceManagerFactory>();
            manager = factory->create_inference_manager(request);
            managers.push_back(manager);
            return manager.get();
        } catch (std::exception& ex) {
            wrap_exception(ex);
            return nullptr;
        }
    }

    void run_inference(cvops::IInferenceManager* inference_manager, cvops::InferenceRequest* inference_request, cvops::InferenceResult* inference_result) 
    {
        try {
            inference_manager->infer(inference_request, inference_result);
        } catch (std::exception& ex) {
            wrap_exception(ex);
        }
    }

    void end_inference_session(cvops::IInferenceManager* inference_manager) {
        try {
            std::cout << "Ending inference session..." << std::endl;
            for(const auto& manager : managers) {
                if (manager.get() == inference_manager) {
                    managers.erase(manager);
                    break;
                }
            }
            delete inference_manager;
        } catch (std::exception& ex) {
            wrap_exception(ex);
        }
    }

    char* error_message() {
        return err_msg.c_str();
    }

#ifdef __cplusplus
}
#endif