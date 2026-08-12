#include "number_classifier.hpp"
#include "../include/armor.hpp"
#include <fmt/format.h>
#include <algorithm>

namespace auto_aim {

NumberClassifier::NumberClassifier(const std::string & model_path,
                                   const std::string & label_path,
                                   const double thre,
                                   const std::vector<std::string> & ignore_classes)
: threshold(thre), ignore_classes_(ignore_classes) {
    net_ = cv::dnn::readNetFromONNX(model_path);
    
    std::ifstream label_file(label_path);
    if (!label_file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + label_path);
    }
    
    std::string line;
    while (std::getline(label_file, line)) {
        // 去除可能的换行符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        class_names_.push_back(line);
    }
    label_file.close();
    
    std::cout << "NumberClassifier loaded with " << class_names_.size() 
              << " classes: ";
    for (const auto & name : class_names_) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
}

cv::Mat NumberClassifier::extractNumber(const cv::Mat & src, const Armor & armor) {
    // Light length in image
    static const int light_length = 12;
    // Image size after warp
    static const int warp_height = 28;
    static const int small_armor_width = 32;
    static const int large_armor_width = 54;
    // Number ROI size
    static const cv::Size roi_size(20, 28);
    static const cv::Size input_size(28, 28);

    // Warp perspective transform
    cv::Point2f lights_vertices[4] = {
        armor.left.bottom, 
        armor.left.top, 
        armor.right.top, 
        armor.right.bottom
    };

    const int top_light_y = (warp_height - light_length) / 2 - 1;
    const int bottom_light_y = top_light_y + light_length;
    const int warp_width = (armor.type == ArmorType::big) ? large_armor_width : small_armor_width;
    
    cv::Point2f target_vertices[4] = {
        cv::Point2f(0, bottom_light_y),
        cv::Point2f(0, top_light_y),
        cv::Point2f(warp_width - 1, top_light_y),
        cv::Point2f(warp_width - 1, bottom_light_y),
    };
    
    cv::Mat number_image;
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // Get ROI
    number_image = number_image(cv::Rect(
        cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    // Binarize
    cv::cvtColor(number_image, number_image, cv::COLOR_BGR2GRAY);
    cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::resize(number_image, number_image, input_size);
    
    return number_image;
}

void NumberClassifier::classify(Armor & armor) {
    if (armor.number_img.empty()) {
        armor.confidence = 0.0;
        armor.name = ArmorName::not_armor;
        return;
    }
    
    // Normalize
    cv::Mat input;
    armor.number_img.convertTo(input, CV_32F, 1.0 / 255.0);

    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob);

    // Set the input blob for the neural network
    std::lock_guard<std::mutex> lock(mutex_);
    net_.setInput(blob);

    // Forward pass
    cv::Mat outputs = net_.forward().clone();

    // Decode the output
    double confidence;
    cv::Point class_id_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    int label_id = class_id_point.x;

    armor.confidence = confidence;
    
    // 映射 label_id 到 ArmorName
    if (label_id >= 0 && label_id < static_cast<int>(class_names_.size())) {
        std::string class_name = class_names_[label_id];
        
        // 映射类名到 ArmorName
        if (class_name == "1" || class_name == "one") armor.name = ArmorName::one;
        else if (class_name == "2" || class_name == "two") armor.name = ArmorName::two;
        else if (class_name == "3" || class_name == "three") armor.name = ArmorName::three;
        else if (class_name == "4" || class_name == "four") armor.name = ArmorName::four;
        else if (class_name == "5" || class_name == "five") armor.name = ArmorName::five;
        else if (class_name == "sentry") armor.name = ArmorName::sentry;
        else if (class_name == "outpost") armor.name = ArmorName::outpost;
        else if (class_name == "base") armor.name = ArmorName::base;
        else armor.name = ArmorName::not_armor;
    } else {
        armor.name = ArmorName::not_armor;
    }
}

void NumberClassifier::eraseIgnoreClasses(std::list<Armor> & armors) {
    armors.remove_if([this](const Armor & armor) {
        // 置信度低于阈值
        if (armor.confidence < threshold) {
            return true;
        }
        
        // 忽略类别
        std::string class_name = ARMOR_NAMES[armor.name];
        for (const auto & ignore_class : ignore_classes_) {
            if (class_name == ignore_class) {
                return true;
            }
        }
        
        // 类型不匹配检查
        bool mismatch = false;
        if (armor.type == ArmorType::big) {
            mismatch = (armor.name == ArmorName::outpost || 
                       armor.name == ArmorName::two ||
                       armor.name == ArmorName::sentry);
        } else if (armor.type == ArmorType::small) {
            mismatch = (armor.name == ArmorName::one || 
                       armor.name == ArmorName::base);
        }
        return mismatch;
    });
}

} // namespace auto_aim
