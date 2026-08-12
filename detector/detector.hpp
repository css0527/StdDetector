#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <memory>
#include <opencv2/opencv.hpp>

#include "../include/armor.hpp"
#include "light_corner_corrector.hpp"
#include "pnp_optimizer.hpp"
#include "number_classifier.hpp"  // 添加这行

namespace auto_aim
{
class Detector
{
public:
  Detector();
  
  std::list<Armor> detect(const cv::Mat & bgr_img);
  
  cv::Mat binary_img;
  
  void setUsePca(bool use) { use_pca_ = use; }
  void setUsePnpOptimize(bool use) { use_pnp_optimize_ = use; }
  double getOptimizedYaw() const { return optimized_yaw_; }
  
  // 添加分类器
  std::unique_ptr<NumberClassifier> classifier;

private:
  bool check_geometry(const Lightbar & lightbar);
  bool check_geometry(const Armor & armor);
  bool check_name(const Armor & armor);

  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour);
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor);

  void classify(Armor & armor);
  
  bool use_pca_ = true;
  bool use_pnp_optimize_ = true;
  double optimized_yaw_ = 0.0;
  
  std::unique_ptr<LightCornerCorrector> corner_corrector_;
  std::unique_ptr<PnPOptimizer> pnp_optimizer_;
};

}  // namespace auto_aim

#endif
