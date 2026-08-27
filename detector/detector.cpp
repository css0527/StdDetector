#include "detector.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <execution>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

Detector::Detector(const DetectorParams & params) : params_(params)
{
  // 初始化分类器
  try {
    std::string model_path = "/home/scurm/StdDetector/model/lenet.onnx";
    std::string label_path = "/home/scurm/StdDetector/model/label.txt";

    classifier_ = std::make_unique<NumberClassifier>(
      model_path, label_path, params_.classifier_threshold, params_.ignore_classes);
    tools::logger()->info("Number classifier initialized");
  } catch (const std::exception & e) {
    tools::logger()->error("Failed to init classifier: {}", e.what());
  }

  // 初始化角点矫正器
  if (params_.use_pca) {
    corner_corrector_ = std::make_unique<LightCornerCorrector>();
  }
}

std::list<Armor> Detector::detect(const cv::Mat & bgr_img)
{
  // 1. 预处理
  binary_img_ = preprocessImage(bgr_img);

  // 2. 检测灯条
  lights_ = findLights(bgr_img, binary_img_);

  // 3. 匹配装甲板
  armors_ = matchLights(lights_);

  // 4. 分类识别数字
  if (!armors_.empty() && classifier_) {
    for (auto & armor : armors_) {
      armor.number_img = classifier_->extractNumber(bgr_img, armor);
      classifier_->classify(armor);
    }

    // 将 vector 转为 list 进行过滤
    std::list<Armor> armor_list(armors_.begin(), armors_.end());
    classifier_->eraseIgnoreClasses(armor_list);
    armors_.assign(armor_list.begin(), armor_list.end());
  }

  // 5. PCA 角点矫正
  if (params_.use_pca && corner_corrector_ && !armors_.empty()) {
    for (auto & armor : armors_) {
      auto left_top = armor.left.top;
      auto left_bottom = armor.left.bottom;
      auto right_top = armor.right.top;
      auto right_bottom = armor.right.bottom;

      corner_corrector_->correctCorners(armor, gray_img_);

      float top_width = cv::norm(armor.left.top - armor.right.top);
      float bottom_width = cv::norm(armor.left.bottom - armor.right.bottom);
      float left_height = cv::norm(armor.left.top - armor.left.bottom);
      float right_height = cv::norm(armor.right.top - armor.right.bottom);

      float avg_width = (top_width + bottom_width) / 2;
      float avg_height = (left_height + right_height) / 2;
      float ratio = avg_width / std::max(avg_height, 1.0f);

      if (ratio < 0.5 || ratio > 8.0 || avg_height < 5.0f || avg_width < 5.0f) {
        armor.left.top = left_top;
        armor.left.bottom = left_bottom;
        armor.right.top = right_top;
        armor.right.bottom = right_bottom;
      }

      armor.points.clear();
      armor.points = {armor.left.top, armor.right.top, armor.right.bottom, armor.left.bottom};
    }
  }

  return std::list<Armor>(armors_.begin(), armors_.end());
}

cv::Mat Detector::preprocessImage(const cv::Mat & rgb_img)
{
  cv::cvtColor(rgb_img, gray_img_, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img_, binary_img, params_.binary_thres, 255, cv::THRESH_BINARY);

  return binary_img;
}

std::vector<Lightbar> Detector::findLights(const cv::Mat & rgb_img, const cv::Mat & binary_img)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::vector<Lightbar> lights;
  size_t id = 0;

  for (const auto & contour : contours) {
    if (contour.size() < 6) continue;

    auto rotated_rect = cv::minAreaRect(contour);
    Lightbar light(rotated_rect, id++);

    if (!isLight(light)) continue;

    int sum_r = 0, sum_b = 0;
    for (const auto & pt : contour) {
      sum_r += rgb_img.at<cv::Vec3b>(pt.y, pt.x)[2];
      sum_b += rgb_img.at<cv::Vec3b>(pt.y, pt.x)[0];
    }

    if (
      std::abs(sum_r - sum_b) / static_cast<int>(contour.size()) >
      params_.light_params.color_diff_thresh) {
      light.color = (sum_r > sum_b) ? Color::red : Color::blue;
    }

    lights.emplace_back(light);
  }

  std::sort(lights.begin(), lights.end(), [](const Lightbar & a, const Lightbar & b) {
    return a.center.x < b.center.x;
  });

  return lights;
}

bool Detector::isLight(const Lightbar & light) const
{
  float ratio = light.width / light.length;
  bool ratio_ok = params_.light_params.min_ratio < ratio && ratio < params_.light_params.max_ratio;
  bool angle_ok = light.angle_error * 57.3 < params_.light_params.max_angle;
  return ratio_ok && angle_ok;
}

Color Detector::getColor(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const
{
  int sum_r = 0, sum_b = 0;
  for (const auto & pt : contour) {
    sum_r += bgr_img.at<cv::Vec3b>(pt.y, pt.x)[2];
    sum_b += bgr_img.at<cv::Vec3b>(pt.y, pt.x)[0];
  }
  return (sum_b > sum_r) ? Color::blue : Color::red;
}

std::vector<Armor> Detector::matchLights(const std::vector<Lightbar> & lights)
{
  std::vector<Armor> armors;

  for (size_t i = 0; i < lights.size(); i++) {
    if (lights[i].color != enemy_color_) continue;

    for (size_t j = i + 1; j < lights.size(); j++) {
      if (lights[j].color != enemy_color_) continue;

      if (containLight(i, j, lights)) continue;

      auto type = judgeArmorType(lights[i], lights[j]);
      if (type == ArmorType::INVALID) continue;

      Armor armor(lights[i], lights[j]);
      armor.type = type;
      armors.emplace_back(armor);
    }
  }

  return armors;
}

bool Detector::containLight(int i, int j, const std::vector<Lightbar> & lights) const
{
  const auto & light1 = lights[i];
  const auto & light2 = lights[j];

  std::vector<cv::Point2f> points = {light1.top, light1.bottom, light2.top, light2.bottom};
  auto bounding_rect = cv::boundingRect(points);

  double avg_length = (light1.length + light2.length) / 2.0;
  double avg_width = (light1.width + light2.width) / 2.0;

  for (int k = i + 1; k < j; k++) {
    const auto & test_light = lights[k];

    if (test_light.width > 2 * avg_width) continue;
    if (test_light.length < 0.5 * avg_length) continue;

    if (
      bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
      bounding_rect.contains(test_light.center)) {
      return true;
    }
  }
  return false;
}

ArmorType Detector::judgeArmorType(const Lightbar & light1, const Lightbar & light2) const
{
  float length_ratio =
    light1.length < light2.length ? light1.length / light2.length : light2.length / light1.length;
  if (length_ratio < params_.armor_params.min_light_ratio) {
    return ArmorType::INVALID;
  }

  float avg_length = (light1.length + light2.length) / 2;
  float center_dist = cv::norm(light1.center - light2.center) / avg_length;

  bool is_small =
    (params_.armor_params.min_small_center_distance <= center_dist &&
     center_dist < params_.armor_params.max_small_center_distance);
  bool is_large =
    (params_.armor_params.min_large_center_distance <= center_dist &&
     center_dist < params_.armor_params.max_large_center_distance);

  cv::Point2f diff = light1.center - light2.center;
  float angle = std::abs(std::atan2(diff.y, diff.x)) * 180.0 / CV_PI;
  if (angle > params_.armor_params.max_angle) {
    return ArmorType::INVALID;
  }

  if (is_large) return ArmorType::LARGE;
  if (is_small) return ArmorType::SMALL;

  return ArmorType::INVALID;
}

}  // namespace auto_aim
