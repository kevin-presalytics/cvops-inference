#include "tracking.h"

#include <opencv2/opencv.hpp>

namespace cvops {
    Tracker::Tracker(TrackerTypes tracker_type) {
        this->tracker_type = tracker_type;
        this->multi_tracker = cv::MultiTracker::create();
        this->bounding_boxes = std::vector<cv::Rect2d>();
    }

    Tracker::~Tracker() {
        if (this->multi_tracker)
        {
            delete this->multi_tracker;
        }
    }

    void Tracker::init(cv::Mat& frame, std::vector<cv::Rect2d>& bounding_boxes)
    {
        for (int i = 0; i < bounding_boxes.size(); i++)
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
            this->multi_tracker->add(tracker, frame, bounding_boxes[i]);
            this->bounding_boxes.push_back(bounding_boxes[i]);
        }
    }

    void Tracker::update(cv::Mat& frame, std::vector<cv::Rect2d>& bounding_boxes)
    {
        this->multi_tracker->update(frame);
        bounding_boxes.clear();
        for (auto& object : this->multi_tracker->getObjects())
        {
            bounding_boxes.push_back(object);
        }
    }
}

