#ifndef INFERENCE_MANAGER_FACTORY_H
#define INFERENCE_MANAGER_FACTORY_H

#include "inference_manager_base.h"
#include "inference_session_request.h"

namespace cvops
{
    class InferenceManagerFactory
    {
        public:
            InferenceManagerFactory() {};
            ~InferenceManagerFactory() {};
            virtual IInferenceManager* create_inference_manager(InferenceSessionRequest* session_request);
    };
}

#endif // INFERENCE_MANAGER_FACTORY_H