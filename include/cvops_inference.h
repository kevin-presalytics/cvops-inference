#ifndef C_API_H
#define C_API_H

#include <opencv2/opencv.hpp>

#include "inference_result.h"
#include "inference_request.h"
#include "inference_session_request.h"
#include "model_platforms.h"
#include "inference_manager_interface.h"
#include "tracking.h"
#include "image_utils.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
    cvops::IInferenceManager* start_inference_session(cvops::InferenceSessionRequest* request);
    cvops::InferenceResult* run_inference(cvops::IInferenceManager* inference_manager, cvops::InferenceRequest* inference_request);
    void end_inference_session(cvops::IInferenceManager* inference_manager);
    void dispose_inference_result(cvops::InferenceResult* inference_result);
    void render_inference_result(cvops::InferenceResult* inference_result, void* image_data, int image_height, int image_width, int num_channels);
    void free_color_palette();
    void set_color_palette(char* color_palette);
    cvops::Tracker* create_tracker(cvops::TrackerTypes tracker_type, void* image_data, int image_height, int image_width, int num_channels);
    void track_image(cvops::Tracker* tracker, void* image_data, int image_height, int image_width, int num_channels);
    void update_tracker(cvops::Tracker* tracker, cvops::InferenceResult* inference_result, void* image_data, int image_height, int image_width, int num_channels);
    cvops::TrackerState* get_tracker_state(cvops::Tracker* tracker);
    void dispose_tracker_state(cvops::TrackerState* tracker_state);
    void dispose_tracker(cvops::Tracker* tracker);
    const char* error_message();

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // C_API_H
