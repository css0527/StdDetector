#ifndef DETECTOR__NUMBER_CLASSIFIER_HPP
#define DETECTOR__NUMBER_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <fstream>

namespace auto_aim {

// 前向声明
struct Armor;

class NumberClassifier {
public:
    NumberClassifier(const std::string & model_path,
                     const std::string & label_path,
                     double threshold = 0.7,
                     const std::vector<std::string> & ignore_classes = {"negative"});
    
    // 提取数字区域
    cv::Mat extractNumber(const cv::Mat & src, const Armor & armor);
    
    // 分类
    void classify(Armor & armor);
    
    // 删除忽略的类别
    void eraseIgnoreClasses(std::list<Armor> & armors);
    
    // 阈值
    double threshold;

private:
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    std::vector<std::string> ignore_classes_;
    std::mutex mutex_;
};

} // namespace auto_aim

#endif
