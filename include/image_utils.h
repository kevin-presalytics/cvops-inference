#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "inference_result.h"
#include "inference_request.h"

#include <opencv2/opencv.hpp>

#include <vector>

namespace cvops
{
    class ImageUtils
    {
        public:
            static void resize_and_letterbox_image(
                const cv::Mat& image, 
                cv::Mat& out_image,
                cv::Size target_size,
                const cv::Scalar& color = cv::Scalar(114, 114, 114),
                int stride = 32
            );
            static void resize_and_pad_image(
                const cv::Mat& input_image, 
                cv::Mat& resized_image,
                cv::Size target_size
            );
            static void decode_image(InferenceRequest* inference_request, cv::Mat* image);
            static void draw_detections(cv::Mat* image, InferenceResult* inference_result, std::vector<cv::Scalar>* color_palette_ptr);
            static void draw_detections(cv::Mat* image, std::vector<Box> boxes, std::vector<cv::Scalar>* color_palette_ptr);
            static void resize_rect(cv::Rect* rect, cv::Size* current_image_size, cv::Size* target_image_size);
            static cv::Rect to_cv_rect(const Box& box);

            static void write_to_file(const std::string& filename, const cv::Mat& image);
            static void write_to_file(const std::string& filename, const char* image, int image_size);

            static float get_iou(const cv::Rect& rect_1, const cv::Rect& rect_2);

            ImageUtils() = delete;
    };
}

#endif // IMAGE_UTILS_H