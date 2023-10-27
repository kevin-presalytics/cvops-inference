#include "inference_manager_base.h"
#include "model_platforms.h"
#include "inference_session_request.h"
#include "yolo_inference_manager.h"
#include "inference_manager_factory.h"

namespace cvops
{
    IInferenceManager* InferenceManagerFactory::create_inference_manager(InferenceSessionRequest* request)
    {
        IInferenceManager* mgr_ptr;
        switch (request->model_platform)
        {
            case ModelPlatforms::YOLO:
                mgr_ptr = new YoloInferenceManager();
                break;
            default:
                break;
        }
        mgr_ptr->start_session(request);
        return mgr_ptr;
    }
}