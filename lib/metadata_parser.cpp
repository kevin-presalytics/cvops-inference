#include "metadata_parser.h"

#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>

#include <map>
#include <memory>
#include <string>
#include <stdexcept>

namespace cvops
{
    std::map<int, std::string> MetadataParser::get_class_names(Json::Value metadata)
    {
        std::map<int, std::string> class_names;
        Json::Value classes = metadata["classes"];
        for (int i = 0; i < classes.size(); i++)
        {
            std::string key_ = std::to_string(i);
            if (classes.isMember(key_))
            {
                class_names[i] = classes[key_].asString();
            }

        }
        return class_names;
    }

    cv::Size MetadataParser::get_image_size(Json::Value metadata)
    {
        Json::Value sizes = metadata["image_size"];
        if (sizes.size() != 2)
        {
            throw std::length_error("image_size must be a 2-element array");
        }
        cv::Size image_size(sizes[0].asInt(), sizes[1].asInt());
        return image_size;
    }

    Json::Value MetadataParser::parse_metadata(char* data)
    {
        Json::Value metadata;
        Json::CharReaderBuilder builder;
        Json::CharReader* reader = builder.newCharReader();
        std::string errors;
        size_t length = strlen(data);
        bool parsingSuccessful = reader->parse(data, data + length, &metadata, &errors);
        delete reader;
        if (!parsingSuccessful)
        {
            std::string message = "Failed to parse metadata JSON: " + errors;
            throw std::runtime_error(message);
        }
        return metadata;
    }
}