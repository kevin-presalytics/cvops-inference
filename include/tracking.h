#ifndef TRACKING_H
#define TRACKING_H

#include "inference_result.h"
#include "kalman_filter.h"
#include "hungarian.h"

// This is for OpenCV 4.2
// for openCv < 4.2, use #include <opencv2/tracking.hpp>
// #include <opencv2/tracking/tracking_legacy.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

namespace cvops {

    enum class TrackerTypes {
        SORT = 1,
    };

    struct MatchedPair {
        int tracker_index;
        int detection_index;
        MatchedPair(int tracker_index, int dectection_index);
    };
    
    
    /// @brief The Tracker class is used to track objects in a video.
    class MultiTracker {
        public:
            MultiTracker(std::shared_ptr<std::vector<cv::Scalar>> color_palette);
            ~MultiTracker();
            void init(cv::Mat& frame, const InferenceResult& inference_result);
            void update(cv::Mat& frame, const InferenceResult& inference_result);
            void update(cv::Mat& frame);
            InferenceResult* get_state();
        private:
            // methods
            void get_predictions();
            void get_iou_matrix(const InferenceResult& inference_result);
            void clear_results();
            void get_unmatched_predictions();
            void get_unmatched_dectections();
            void update_frame(cv::Mat& frame);
            void update_trackers();
            void create_new_trackers();
            void remove_dead_trackers(const cv::Mat& frame);
            void get_matches();
            int get_highest_object_id();


            // properties
            std::vector<KalmanTracker*> trackers;
            TrackerTypes tracker_type;
            HungarianAlgorithm hungarian_algoritm;
            // std::vector<cv::Rect2d> bounding_boxes;
            std::vector<Box> predictions;
            std::vector<std::vector<double>> iou_matrix;
            std::set<int> unmatched_detections;
            std::set<int> unmatched_predictions;
            std::set<int> all_items;
            std::set<int> matched_items;
            std::vector<MatchedPair> matched_pairs;
            std::vector<cv::Rect> tracking_results;
            std::vector<int> assignments;
            InferenceResult last_inference_result;
            double iou_threshold;
            int object_id_upper_bound;
            std::shared_ptr<std::vector<cv::Scalar>> color_palette;

        
    };
}

#endif // TRACKING_H