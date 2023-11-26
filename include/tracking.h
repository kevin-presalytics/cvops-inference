#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
// This is for OpenCV 4.2
// for openCv > 4.2, use #include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>

namespace cvops {

    enum class TrackerTypes {
        BOOSTING,
        MIL,
        KCF,
        TLD,
        MEDIANFLOW,
        GOTURN,
        MOSSE,
        CSRT
    };
    
    
    /// @brief The Tracker class is used to track objects in a video.
    class Tracker {
        public:
            Tracker(TrackerTypes tracker_type);
            ~Tracker();
            void init(cv::Mat& frame, std::vector<cv::Rect2d>& bounding_boxes);
            void update(cv::Mat& frame, std::vector<cv::Rect2d>& bounding_boxes);
        private:
            cv::MultiTracker* multi_tracker;
            TrackerTypes tracker_type;
            std::vector<cv::Rect2d> bounding_boxes;
    };
}

#endif // TRACKING_H