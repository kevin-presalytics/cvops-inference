#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include "inference_result.h"

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

namespace cvops
{
    /// Kalman tracker implmentation adapted from https://github.com/mcximing/sort-cpp/blob/master/sort-c%2B%2B/KalmanTracker.h
    class KalmanTracker {
        public:
            // methods
            KalmanTracker(Box* dectected_rect, int object_id);
            ~KalmanTracker();
            Box predict();
            void update(cv::Rect new_observation);
            cv::Rect get_state();

            // properties
            int time_since_update;
            int hits;
            int hit_streak;
            int age;
            int id;
            

        private:
            void init();
            cv::Rect get_rect_xysr(float cx, float cy, float s, float r);
            cv::KalmanFilter filter;
            cv::Mat measurement;
            std::vector<cv::Rect2f> history;
            int measure_num_;
            int state_num_;
            Box* initial_box_;
            cv::Rect2f state_;
    };
}

#endif // KALMAN_FILTER_H