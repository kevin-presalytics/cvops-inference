#ifndef TRACKING_H
#define TRACKING_H

#include "inference_result.h"

#include <opencv2/opencv.hpp>
// This is for OpenCV 4.2
// for openCv > 4.2, use #include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <vector>

namespace cvops {

    enum class TrackerTypes {
        BOOSTING = 1,
        MIL = 2,
        KCF = 3,
        TLD = 4,
        MEDIANFLOW = 5,
        GOTURN = 6,
        MOSSE = 7,
        CSRT = 8
    };

    struct TrackerState {
        int boxes_count;
        Box* boxes;
        ~TrackerState();
        TrackerState();
    };
    
    
    /// @brief The Tracker class is used to track objects in a video.
    class Tracker {
        public:
            Tracker(TrackerTypes tracker_type);
            Tracker(TrackerTypes tracker_type, int result_history_size);
            ~Tracker();
            void update(cv::Mat& frame, InferenceResult inference_result);
            void update(cv::Mat& frame);
            TrackerState* get_state();
        private:
            // methods
            void init(cv::Mat& frame, InferenceResult inference_result);
            void add_new_result(InferenceResult inference_result);
            cv::Ptr<cv::Tracker> create_tracker();

            // properties
            bool is_initialized_;
            int result_history_size;
            cv::MultiTracker* multi_tracker;
            TrackerTypes tracker_type;
            std::vector<InferenceResult> result_history;
    };
}

#endif // TRACKING_H