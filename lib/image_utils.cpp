#include "image_utils.h"
#include "inference_request.h"
#include "inference_result.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>
#include <iostream>

namespace cvops
{
    void ImageUtils::resize_and_letterbox_image( 
        const cv::Mat& image, 
        cv::Mat& out_image,
        cv::Size target_size,
        const cv::Scalar& color,
        int stride)
    {
        cv::Size input_shape = image.size();

        float r = std::min((float)target_size.height / (float)input_shape.height,
                        (float)target_size.width / (float)input_shape.width);

        float ratio[2] {r, r};

        int pad[2] {(int)std::round((float)input_shape.width * r),
                        (int)std::round((float)input_shape.height * r)};

        if (input_shape.width != pad[0] && input_shape.height != pad[1])
        {
            cv::resize(image, out_image, cv::Size(pad[0], pad[1]));
        } else {
            out_image = image;
        }

        auto dw = (float)(target_size.width - pad[0]);
        auto dh = (float)(target_size.height - pad[1]);

        dw /= 2.0f;
        dh /= 2.0f;

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        cv::copyMakeBorder(out_image, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    void ImageUtils::draw_detections(
        cv::Mat* image_ptr, 
        InferenceResult* inference_result_ptr,
        std::vector<cv::Scalar>* color_palette_ptr)
    {

        int detection_count = inference_result_ptr->boxes_count;
        for (int b = 0; b < detection_count; b++)
        {
            Box box = inference_result_ptr->boxes[b];
            cv::Scalar color = (*color_palette_ptr)[box.class_id];
            cv::Rect rect = ImageUtils::to_cv_rect(box);
            cv::rectangle(*image_ptr, rect, color, 2); // "2" is the line thickness of the bounding box

            int baseline = 0;
            std::string label(box.class_name);
            std::transform(label.begin(), label.end(), label.begin(), std::ptr_fun<int, int>(std::toupper));
            int confidence = (int)std::round(box.confidence * 100);
            label += ": " + std::to_string(confidence) + "%";

            cv::Size size = cv::getTextSize(label, cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);

            cv::rectangle(*image_ptr,
                        cv::Point(box.x, box.y - 25), cv::Point(box.x + size.width, box.y),
                        color, -1);

            cv::putText(*image_ptr, label,
                        cv::Point(box.x, box.y - 3), 
                        cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(25, 25, 25), 2);
        }

    }

    void ImageUtils::resize_rect(cv::Rect*rect, cv::Size* current_image_size, cv::Size* target_image_size)
    {
        if (current_image_size->height == target_image_size->height && current_image_size->width == target_image_size->width) return;

        float r = std::max((float)target_image_size->height / (float)current_image_size->height,
                        (float)target_image_size->width / (float)current_image_size->width);


        float pad_x = std::round(((float)target_image_size->width - (float)current_image_size->width * r) * 0.5f);
        float pad_y = std::round(((float)target_image_size->height - (float)current_image_size->height * r) * 0.5f);

        rect->x = (int)(rect->x * r + pad_x);
        rect->y = (int)(rect->y * r + pad_y);
        rect->width = (int)(rect->width * r);
        rect->height = (int)(rect->height * r);
    }

    void ImageUtils::decode_image(InferenceRequest* inference_request, cv::Mat* image) {
        *image = cv::imdecode(cv::Mat(1, inference_request->size, CV_8UC1, inference_request->bytes), cv::IMREAD_UNCHANGED);
    }

    cv::Rect ImageUtils::to_cv_rect(const Box& box)
    {
        return cv::Rect(box.x, box.y, box.width, box.height);
    }

    void ImageUtils::write_to_file(const std::string& filename, const cv::Mat& image)
    {
        cv::imwrite(filename, image);
    }

    void ImageUtils::write_to_file(const std::string& filename, const char* image, int image_size)
    {
        std::vector<uchar> buf(image_size);
        std::copy(image, image + image_size, buf.begin());
        cv::Mat mat = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
        cv::imwrite(filename, mat);
    }

    void ImageUtils::resize_and_pad_image(const cv::Mat& input_image, cv::Mat& resized_image, cv::Size target_size)
    {
        float resizeScales = 1.0;
        if (input_image.channels() == 3)
        {
            resized_image = input_image.clone();
            cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
        }
        else
        {
            cv::cvtColor(input_image, resized_image, cv::COLOR_GRAY2RGB);
        }

        if (input_image.cols >= input_image.rows)
        {
            resizeScales = input_image.cols / (float)target_size.width;
            cv::resize(resized_image, resized_image, cv::Size(target_size.width, int(input_image.rows / resizeScales)));
        }
        else
        {
            resizeScales = input_image.rows / (float)target_size.height;
            cv::resize(resized_image, resized_image, cv::Size(int(input_image.cols / resizeScales), target_size.height));
        }
        cv::Mat tempImg = cv::Mat::zeros(target_size, CV_8UC3);
        resized_image.copyTo(tempImg(cv::Rect(0, 0, resized_image.cols, resized_image.rows)));
        resized_image = tempImg;
    }
}