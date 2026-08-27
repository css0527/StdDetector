#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "../include/armor.hpp"
#include "light_corner_corrector.hpp"
#include "number_classifier.hpp"

namespace auto_aim
{

struct DetectorParams
{
  // 二值化参数
  int binary_thres = 120;

  // 灯条参数
  struct LightParams
  {
    double min_ratio = 0.08;
    double max_ratio = 0.4;
    double max_angle = 40.0;
    int color_diff_thresh = 25;
  } light_params;

  // 装甲板参数
  struct ArmorParams
  {
    double min_light_ratio = 0.6;
    double min_small_center_distance = 0.8;
    double max_small_center_distance = 3.2;
    double min_large_center_distance = 3.2;
    double max_large_center_distance = 5.0;
    double max_angle = 35.0;
  } armor_params;

  // 分类器参数
  double classifier_threshold = 0.7;
  std::vector<std::string> ignore_classes = {"negative"};

  // 优化开关
  bool use_pca = true;
  bool use_pnp_optimize = true;
};

class Detector
{
public:
  explicit Detector(const DetectorParams & params = DetectorParams());
  ~Detector() = default;

  std::list<Armor> detect(const cv::Mat & bgr_img);

  void setEnemyColor(Color color) { enemy_color_ = color; }

  cv::Mat getBinaryImage() const { return binary_img_; }
  cv::Mat getGrayImage() const { return gray_img_; }

private:
  cv::Mat preprocessImage(const cv::Mat & rgb_img);

  std::vector<Lightbar> findLights(const cv::Mat & rgb_img, const cv::Mat & binary_img);
  bool isLight(const Lightbar & light) const;
  Color getColor(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;

  std::vector<Armor> matchLights(const std::vector<Lightbar> & lights);
  bool containLight(int i, int j, const std::vector<Lightbar> & lights) const;
  ArmorType judgeArmorType(const Lightbar & light1, const Lightbar & light2) const;

  DetectorParams params_;
  Color enemy_color_ = Color::red;

  cv::Mat gray_img_;
  cv::Mat binary_img_;
  std::vector<Lightbar> lights_;
  std::vector<Armor> armors_;

  std::unique_ptr<NumberClassifier> classifier_;
  std::unique_ptr<LightCornerCorrector> corner_corrector_;
};

}  // namespace auto_aim

#endif
