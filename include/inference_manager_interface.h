#ifndef I_INFERENCE_MANAGER_H
#define I_INFERENCE_MANAGER_H

#include "inference_result.h"
#include "inference_request.h"
#include "inference_session_request.h"

namespace cvops
{
    class IInferenceManager
    {
        public:
            IInferenceManager() {};
            virtual ~IInferenceManager() {};
            virtual void start_session(InferenceSessionRequest* session_request_ptr) = 0;
            virtual InferenceResult* infer(InferenceRequest* request) = 0;
    };
}

#endif // I_INFERENCE_MANAGER_H