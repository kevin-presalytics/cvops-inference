#ifndef C_API_H
#define C_API_H

#include "inference_result.h"
#include "inference_request.h"
#include "inference_session_request.h"
#include "model_platforms.h"
#include "inference_manager_interface.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
    cvops::IInferenceManager* start_inference_session(cvops::InferenceSessionRequest* request);
    cvops::InferenceResult run_inference(cvops::IInferenceManager* inference_manager, cvops::InferenceRequest* inference_request);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // C_API_H