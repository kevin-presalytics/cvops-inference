#include "tracking.h"
#include "inference_result.h"
#include "image_utils.h"

#include <opencv2/opencv.hpp>

#include <vector>

namespace cvops {
    Tracker::Tracker(TrackerTypes tracker_type) {
        this->tracker_type = tracker_type;
        this->multi_tracker = cv::MultiTracker::create();
        this->result_history = std::vector<InferenceResult>();
        this->result_history_size = 1;
    }

    Tracker::Tracker(TrackerTypes tracker_type, int result_history_size) {
        this->tracker_type = tracker_type;
        this->multi_tracker = cv::MultiTracker::create();
        this->result_history = std::vector<InferenceResult>();
        this->result_history_size = result_history_size;
    }

    Tracker::~Tracker() {
        if (this->multi_tracker)
        {
            delete this->multi_tracker;
        }
        this->result_history.clear();
        this->result_history.shrink_to_fit();
    }

    void Tracker::add_new_result(InferenceResult inference_result)
    {
        if (this->result_history.size() >= this->result_history_size)
        {
            this->result_history.erase(this->result_history.end() - 1);
        }
        this->result_history.insert(this->result_history.begin(), inference_result);
    }

    void Tracker::init(cv::Mat& frame, InferenceResult inference_result)
    {
        std::vector<cv::Rect> bounding_boxes = std::vector<cv::Rect>();
        for (int i = 0; i < inference_result.boxes_count; i++)
        {
            cv::Rect bounding_box = ImageUtils::to_cv_rect(inference_result.boxes[i]);
            cv::Ptr<cv::Tracker> tracker = this->create_tracker();
            this->multi_tracker->add(tracker, frame, bounding_box);
            //bounding_boxes.push_back(ImageUtils::to_cv_rect(inference_result->boxes[i]));
        }
    }

    cv::Ptr<cv::Tracker> Tracker::create_tracker()
    {
        cv::Ptr<cv::Tracker> tracker;
        switch (this->tracker_type)
        {
            case TrackerTypes::BOOSTING:
                tracker = cv::TrackerBoosting::create();
                break;
            case TrackerTypes::MIL:
                tracker = cv::TrackerMIL::create();
                break;
            case TrackerTypes::KCF:
                tracker = cv::TrackerKCF::create();
                break;
            case TrackerTypes::TLD:
                tracker = cv::TrackerTLD::create();
                break;
            case TrackerTypes::MEDIANFLOW:
                tracker = cv::TrackerMedianFlow::create();
                break;
            case TrackerTypes::GOTURN:
                tracker = cv::TrackerGOTURN::create();
                break;
            case TrackerTypes::MOSSE:
                tracker = cv::TrackerMOSSE::create();
                break;
            case TrackerTypes::CSRT:
                tracker = cv::TrackerCSRT::create();
                break;
        }
        return tracker;
    }

    void Tracker::update(cv::Mat& frame, InferenceResult inference_result)
    {
        if (!this->is_initialized_)
        {
            this->init(frame, inference_result);
            this->is_initialized_ = true;
        }
        //  TODO:
        //  Use IoU calculation to match boxes of the same class from current tracker to new inference result
        //  If the distance is too large, then create a new tracker
        //  If the distance is small, then update the tracker with the new box dimensions (or replace the tracker with a new one)
        //  Remove trackers for off-screen objects
        this->update(frame);
    }

    // std::vector<cv::Rect> Tracker::merge_boxes(std::unique_ptr<InferenceResult> inference_result)
    // {
    //     std::vector<cv::Rect> new_boxes = std::vector<cv::Rect>();
    //     for (int i = 0; i < inference_result->boxes_count; i++)
    //     {
    //         new_boxes.push_back(ImageUtils::to_cv_rect(inference_result->boxes[i]));
    //     }
    //     return new_boxes;
    // }

    void Tracker::update(cv::Mat& frame)
    {
        if (!this->is_initialized_)
        {
            throw std::runtime_error("Tracker is not initialized");
        }
        this->multi_tracker->update(frame);
    }

    TrackerState* Tracker::get_state()
    {
        TrackerState* tracker_state = new TrackerState();
        tracker_state->boxes_count = this->multi_tracker->getObjects().size();
        tracker_state->boxes = new Box[tracker_state->boxes_count];
        std::vector<cv::Rect2d> bounding_boxes_d = this->multi_tracker->getObjects();
        for (int i = 0; i < tracker_state->boxes_count; i++)
        {   
            tracker_state->boxes[i].class_id = 0;
            tracker_state->boxes[i].confidence = 1.0f;
            tracker_state->boxes[i].class_name = nullptr;
            tracker_state->boxes[i].x = (int)bounding_boxes_d[i].x;
            tracker_state->boxes[i].y = (int)bounding_boxes_d[i].y;
            tracker_state->boxes[i].width = (int)bounding_boxes_d[i].width;
            tracker_state->boxes[i].height = (int)bounding_boxes_d[i].height;
        }
        return tracker_state;
    }

    TrackerState::~TrackerState()
    {
        delete[] this->boxes;
    }

    TrackerState::TrackerState()
    {
        this->boxes_count = 0;
        this->boxes = nullptr;
    }
}

