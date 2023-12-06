#include "tracking.h"
#include "image_utils.h"
#include "kalman_filter.h"
#include "hungarian.h"

#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <algorithm>

namespace cvops {
    MatchedPair::MatchedPair(int tracker_index, int detection_index) {
        this->detection_index = detection_index;
        this->tracker_index = tracker_index;
    }

    MultiTracker::MultiTracker(std::shared_ptr<std::vector<cv::Scalar>> color_palette) {
        this->tracker_type = TrackerTypes::SORT;
        this->trackers = std::vector<std::unique_ptr<KalmanTracker>>();
        this->hungarian_algoritm = HungarianAlgorithm();
        this->unmatched_detections = std::set<int>();
        this->unmatched_predictions = std::set<int>();
        this->all_items = std::set<int>();
        this->matched_items = std::set<int>();
        this->matched_pairs = std::vector<MatchedPair>();
        this->tracking_results = std::vector<cv::Rect>();
        this->assignments = std::vector<int>();
        this->predictions = std::vector<Box>();
        this->iou_threshold = 0.3;
        this->object_id_upper_bound = 1000000;  // TODO: Make this configurable, probbly need to redo algo
        this->color_palette = color_palette;
        this->min_height = 10;  // TODO: Make these configurable
        this->min_width = 10; // TODO: Make these 
    }

    MultiTracker::~MultiTracker() { 
        this->clear_results();
        this->trackers.clear();
    }

    void MultiTracker::clear_results()
    {
        this->unmatched_detections.clear();
        this->unmatched_predictions.clear();
        this->all_items.clear();
        this->matched_items.clear();
        this->matched_pairs.clear();
        this->tracking_results.clear();
        this->assignments.clear();
        this->predictions.clear();
        this->iou_matrix.clear();
    }

    void MultiTracker::init(cv::Mat& frame, const InferenceResult& inference_result)
    {
        for (int i = 0; i < inference_result.boxes_count; i++)
        {
            this->trackers.emplace_back(std::make_unique<KalmanTracker>(&inference_result.boxes[i], i));
        }
    }

    void MultiTracker::update(cv::Mat& frame, const InferenceResult& inference_result)
    {
        this->clear_results();
        this->last_inference_result = inference_result;
        if (this->trackers.size() == 0) {
            // Initialize
            this->init(frame, inference_result);
        }
        this->get_predictions();
        this->get_iou_matrix(inference_result);
        this->hungarian_algoritm.Solve(this->iou_matrix, this->assignments);
        this->get_unmatched_dectections();
        this->get_unmatched_predictions();
        this->get_matches();
        this->update_trackers();
        this->create_new_trackers();
        this->remove_dead_trackers(frame);
        std::vector<Box> filtered_predictions = this->get_filtered_predictions();
        this->update_frame(frame, filtered_predictions);
    }

    void MultiTracker::update(cv::Mat& frame)
    {
        this->get_predictions();
        std::vector<Box> filtered_predictions = this->get_filtered_predictions();
        this->update_frame(frame, filtered_predictions);
    }

    void MultiTracker::get_predictions()
    {
        size_t prediction_count = this->trackers.size();
        for (int i = 0; i < prediction_count; i++)
        {
            this->predictions.push_back(this->trackers[i]->predict());
        }
    }

    void MultiTracker::get_iou_matrix(const InferenceResult& inference_result)
    {
        int num_trackers = (int)this->predictions.size();

        int num_detections = inference_result.boxes_count;

        iou_matrix.resize(num_trackers);

        for (int t = 0; t < num_trackers; t++)
        {
            iou_matrix[t].resize(num_detections);
            for (int d = 0; d < num_detections; d++)
            {
                cv::Rect detected_box = ImageUtils::to_cv_rect(inference_result.boxes[d]);
                // this is really the complement of IoU -> Hungarian is a min cost algorithm
                iou_matrix[t][d] = 1 - ImageUtils::get_iou(this->predictions[t], detected_box);
            }
        }
    }

    void MultiTracker::get_unmatched_predictions()
    {
        int detection_count = this->last_inference_result.boxes_count;
        int tracker_count = (int)this->trackers.size();
        if (detection_count < tracker_count)
        {
            for (int i = 0; i < tracker_count; i++)
            {
                if (this->assignments[i] == -1)
                {
                    this->unmatched_predictions.insert(i);
                }
            }
        }
    }

    void MultiTracker::get_unmatched_dectections()
    {
        int detection_count = this->last_inference_result.boxes_count;
        int tracker_count = (int)this->trackers.size();
        if (detection_count > tracker_count)
        {

            for (unsigned int n = 0; n < detection_count; n++)
				this->all_items.insert(n);

			for (unsigned int i = 0; i < tracker_count; ++i)
				this->matched_items.insert(this->assignments[i]);

			std::set_difference(this->all_items.begin(), this->all_items.end(),
				this->matched_items.begin(), this->matched_items.end(),
				std::insert_iterator<std::set<int>>(this->unmatched_detections, this->unmatched_detections.begin()));
        }
    }

    void MultiTracker::get_matches()
    {
        int tracker_count = (int)this->trackers.size();
        for (int i = 0; i < tracker_count; i++)
        {
            if (this->assignments[i] == -1) continue;
            if (1 - this->iou_matrix[i][this->assignments[i]] < this->iou_threshold)
			{
				this->unmatched_predictions.insert(i);
				this->unmatched_detections.insert(this->assignments[i]);
			}
			else
				this->matched_pairs.push_back(MatchedPair(i, this->assignments[i]));
        }
    }



    void MultiTracker::update_trackers() 
    {
        int match_count = (int)this->matched_pairs.size();
        for (int i = 0; i < match_count; i++)
        {
            MatchedPair match = this->matched_pairs[i];
            this->trackers[match.tracker_index]->update(this->last_inference_result.boxes[match.detection_index]);
        } 
    }

    int MultiTracker::get_highest_object_id()
    {
        size_t num_trackers = this->trackers.size();
        std::vector<int> ids(num_trackers); 
        std::vector<int>::iterator it_max;
        for (int i = 0; i < num_trackers; i++)
            ids.push_back(this->trackers[i]->id);
        it_max = std::max_element(ids.begin(), ids.end());
        int max = *it_max;
        if (max > this->object_id_upper_bound) max = 1;
        return max;
    }


    void MultiTracker::create_new_trackers()
    {
        int object_id = this->get_highest_object_id();
        // TODO: Create method to filter dection false positives prior to adding a new tracker
        for (int unmatched_index : this->unmatched_detections)
		{
            object_id++;
            Box* box = &(this->last_inference_result.boxes[unmatched_index]);
			this->trackers.emplace_back(std::make_unique<KalmanTracker>(box, object_id));
            
		}    
    }


    void MultiTracker::update_frame(cv::Mat& frame, std::vector<Box> boxes)
    {
        ImageUtils::draw_detections(&frame, boxes, this->color_palette.get());
    }

    void MultiTracker::remove_dead_trackers(const cv::Mat& frame)
    {
        int tracker_count = (int)this->trackers.size();
        int image_width = frame.cols;
        int image_height = frame.rows;

        for (int i = 0; i < tracker_count; i++)
        {
            // TODO: create method to filter out boxes that are no longer being detected
            cv::Rect2d box = this->trackers[i]->get_state();
            cv::Point box_center = cv::Point(box.x + box.width / 2, box.y + box.height / 2);
            if (box_center.x < 0 || box_center.y < 0 || box_center.x > image_width || box_center.y > image_height)
            {
                // Deletes box if out of frame
                this->trackers.erase(this->trackers.begin() + i);
            }
        }
    }

    InferenceResult* MultiTracker::get_state()
    {
        int tracker_count = (int)this->trackers.size();
        InferenceResult* result = new InferenceResult();
        result->boxes_count = tracker_count;
        void* boxes_ptr = operator new[](sizeof(Box) * tracker_count);
        result->boxes = static_cast<Box*>(boxes_ptr);
        for (int i = 0; i < tracker_count; i++)
        {
            result->boxes[i] = this->trackers[i]->get_state();
        }
        return result;
    }

    // filter predictions to those should be rendered
    std::vector<Box> MultiTracker::get_filtered_predictions()
    {
        std::vector<Box> filtered_predictions;
        int num_trackers = (int)this->trackers.size();
        for (int i = 0; i < num_trackers; i++)
        {

            Box box = this->trackers[i]->get_state();
            // Test box for min height and width
            bool too_small = box.height < this->min_height || box.width < this->min_width;
            // Test box for min age to filter false positives
            // First hit just comes from the initialization process of the tracker
            bool too_young = this->trackers[i]->hits <= 1; 

            if (!too_small && !too_young)
                filtered_predictions.push_back(box);
        }

        return filtered_predictions;
    }
}
