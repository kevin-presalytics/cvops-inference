#include "kalman_filter.h"
#include "inference_result.h"
#include "image_utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

/// Kalman tracker implmentation adapted from https://github.com/mcximing/sort-cpp/blob/master/sort-c%2B%2B/KalmanTracker.h

namespace cvops
{
	KalmanTracker::KalmanTracker(Box* box, int object_id)
	{
        initial_box_ = box;
        state_ = ImageUtils::to_cv_rect(*box); 
		time_since_update = 0;
		hits = 0;
		hit_streak = 0;
		age = 0;
		id = object_id;
        history = std::vector<cv::Rect2f>();
        state_num_ = 7;
        measure_num_ = 4;        
        filter = cv::KalmanFilter(state_num_, measure_num_, 0);
        this->init();
	}

	KalmanTracker::~KalmanTracker()
	{
		history.clear();
	}

    // initialize Kalman filter
    void KalmanTracker::init()
    {

        this->measurement = cv::Mat::zeros(this->measure_num_, 1, CV_32F);

        this->filter.transitionMatrix = (cv::Mat_<float>(this->state_num_, this->state_num_) <<
            1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1);

        

        cv::setIdentity(this->filter.measurementMatrix);
        cv::setIdentity(this->filter.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(this->filter.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(this->filter.errorCovPost, cv::Scalar::all(1));
        
        // initialize state vector with bounding box in [cx,cy,s,r] style
        this->filter.statePost.at<float>(0, 0) = this->state_.x + this->state_.width / 2;
        this->filter.statePost.at<float>(1, 0) = this->state_.y + this->state_.height / 2;
        this->filter.statePost.at<float>(2, 0) = this->state_.area();
        this->filter.statePost.at<float>(3, 0) = this->state_.width / this->state_.height;

        // this->filter.statePre.at<float>(0, 0) = this->state_.x + this->state_.width / 2;
        // this->filter.statePre.at<float>(1, 0) = this->state_.y + this->state_.height / 2;
        // this->filter.statePre.at<float>(2, 0) = this->state_.area();
        // this->filter.statePre.at<float>(3, 0) = this->state_.width / this->state_.height;
    }

    // Predict the estimated bounding box.
    Box KalmanTracker::predict()
    {
        // predict
        cv::Mat p = this->filter.predict();
        this->age += 1;

        if (this->time_since_update > 0)
            this->hit_streak = 0;
        this->time_since_update += 1;

        cv::Rect rect = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

        history.push_back(rect);
        return Box(
            history.back(), 
            this->initial_box_->class_id,
            this->initial_box_->class_name,
            this->initial_box_->object_id,
            this->initial_box_->confidence
        );
    }


    // Update the state vector with observed bounding box.
    void KalmanTracker::update(cv::Rect new_observation)
    {
        this->time_since_update = 0;
        this->history.clear();
        this->hits += 1;
        this->hit_streak += 1;

        // measurement
        this->measurement.at<float>(0, 0) = new_observation.x + new_observation.width / 2;
        this->measurement.at<float>(1, 0) = new_observation.y + new_observation.height / 2;
        this->measurement.at<float>(2, 0) = new_observation.area();
        this->measurement.at<float>(3, 0) = new_observation.width / new_observation.height;

        // update
        this->filter.correct(this->measurement);
    }


    // Return the current state vector
    Box KalmanTracker::get_state()
    {
        cv::Mat s = this->filter.statePost;
        cv::Rect rect = get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
        return Box(
            rect, 
            this->initial_box_->class_id,
            this->initial_box_->class_name,
            this->initial_box_->object_id,
            this->initial_box_->confidence
        );
    }

    // Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
    cv::Rect KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
    {
        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;

        return cv::Rect(x, y, w, h);
    }
}