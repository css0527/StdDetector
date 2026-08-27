#ifndef DETECTOR__NUMBER_CLASSIFIER_HPP
#define DETECTOR__NUMBER_CLASSIFIER_HPP

#include <fstream>
#include <list>  // 添加
#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../include/armor.hpp"  // 添加，确保 Armor 类型可见

namespace auto_aim
{

class NumberClassifier
{
public:
  NumberClassifier(
    const std::string & model_path, const std::string & label_path, double threshold = 0.7,
    const std::vector<std::string> & ignore_classes = {"negative"});

  cv::Mat extractNumber(const cv::Mat & src, const Armor & armor);
  void classify(Armor & armor);
  void eraseIgnoreClasses(std::list<Armor> & armors);

  double threshold;

private:
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
  std::mutex mutex_;
};

}  // namespace auto_aim

#endif
