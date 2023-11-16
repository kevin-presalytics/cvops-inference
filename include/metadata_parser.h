#ifndef METADATA_PARSER_H
#define METADATA_PARSER_H

#include <map>
#include <memory>

#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>

namespace cvops
{
    class MetadataParser
    {
        public:
            static std::map<int, std::string> get_class_names(Json::Value metadata);
            static cv::Size get_image_size(Json::Value metadata);
            static Json::Value parse_metadata(char* data);
            static std::vector<cv::Scalar> parse_color_palette(char* data);
            static std::vector<cv::Scalar> parse_color_palette(Json::Value metadata);
            MetadataParser() = delete;

    };
}

#endif // METADATA_PARSER_H