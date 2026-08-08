#include "detector.hpp"
#include "light_corner_corrector.hpp"
#include "pnp_optimizer.hpp"

#include <fmt/format.h>

#include "tools/img_tools.hpp"
using namespace std;
using namespace cv;

namespace auto_aim
{
Detector::Detector() {}
std::list<Armor> Detector::detect(const cv::Mat & bgr_img)
{
  // 彩色图转灰度图
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  // 进行二值化
  binary_img = gray_img.clone();  // 保存灰度图供 PCA 使用
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 120, 255, cv::THRESH_BINARY);

  // 获取轮廓点
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // 获取灯条
  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    auto lightbar = Lightbar(rotated_rect, lightbar_id);

    if (!check_geometry(lightbar)) continue;

    lightbar.color = get_color(bgr_img, contour);
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  // 将灯条从左到右排序
  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // 获取装甲板
  std::list<Armor> armors;

  for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
    for (auto right = std::next(left); right != lightbars.end(); right++) {
      if (left->color != right->color) continue;

      auto armor = Armor(*left, *right);
      if (!check_geometry(armor)) continue;

      armor.pattern = get_pattern(bgr_img, armor);

      classify(armor);
      if (!check_name(armor)) continue;

      armors.emplace_back(armor);
    }
  }

  // PCA 角点矫正 - 在所有装甲板检测完成后执行
  if (!armors.empty()) {
      if (corner_corrector_ == nullptr) {
        corner_corrector_ = std::make_unique<LightCornerCorrector>();
      }
      
      if (use_pca_ && corner_corrector_) {
        for (auto& armor : armors) {
          // 保存原始角点
          auto left_top_orig = armor.left.top;
          auto left_bottom_orig = armor.left.bottom;
          auto right_top_orig = armor.right.top;
          auto right_bottom_orig = armor.right.bottom;
          
          corner_corrector_->correctCorners(armor, gray_img);
          
          // 验证矫正后的角点是否合理
          // 检查装甲板的四个角点是否构成合理的四边形
          std::vector<cv::Point2f> new_points = {
              armor.left.top, armor.right.top, 
              armor.right.bottom, armor.left.bottom
          };
          
          // 计算新的宽高比
          float top_width = cv::norm(armor.left.top - armor.right.top);
          float bottom_width = cv::norm(armor.left.bottom - armor.right.bottom);
          float left_height = cv::norm(armor.left.top - armor.left.bottom);
          float right_height = cv::norm(armor.right.top - armor.right.bottom);
          
          float avg_width = (top_width + bottom_width) / 2;
          float avg_height = (left_height + right_height) / 2;
          float ratio = avg_width / std::max(avg_height, 1.0f);
          
          // 如果矫正后的形状不合理，恢复原始角点
          if (ratio < 0.5 || ratio > 8.0 || avg_height < 5.0f || avg_width < 5.0f) {
              armor.left.top = left_top_orig;
              armor.left.bottom = left_bottom_orig;
              armor.right.top = right_top_orig;
              armor.right.bottom = right_bottom_orig;
          }
          
          // 更新装甲板的角点
          armor.points.clear();
          armor.points.emplace_back(armor.left.top);
          armor.points.emplace_back(armor.right.top);
          armor.points.emplace_back(armor.right.bottom);
          armor.points.emplace_back(armor.left.bottom);
        }
      }
  }

  return armors;
}

bool Detector::check_geometry(const Lightbar & lightbar)
{
  auto angle_ok = (lightbar.angle_error * 57.3) < 45;  //degree
  auto ratio_ok = lightbar.ratio > 1.5 && lightbar.ratio < 20;
  auto length_ok = lightbar.length > 8;
  return angle_ok && ratio_ok && length_ok;
}

bool Detector::check_geometry(const Armor & armor)
{
  auto ratio_ok = armor.ratio > 1 && armor.ratio < 5;
  auto side_ratio_ok = armor.side_ratio < 1.5;
  auto rectangular_error_ok = (armor.rectangular_error * 57.3) < 25;
  return ratio_ok && side_ratio_ok && rectangular_error_ok;
}

bool Detector::check_name(const Armor & armor)
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > 0.8;

  return name_ok && confidence_ok;
}

Color Detector::get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour)
{
  int red_sum = 0, blue_sum = 0;

  for (const auto & point : contour) {
    red_sum += bgr_img.at<cv::Vec3b>(point)[2];
    blue_sum += bgr_img.at<cv::Vec3b>(point)[0];
  }

  return blue_sum > red_sum ? Color::blue : Color::red;
}

cv::Mat Detector::get_pattern(const cv::Mat & bgr_img, const Armor & armor)
{
  // 延长灯条获得装甲板角点
  // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
  auto tl = armor.left.center - armor.left.top2bottom * 1.125;
  auto bl = armor.left.center + armor.left.top2bottom * 1.125;
  auto tr = armor.right.center - armor.right.top2bottom * 1.125;
  auto br = armor.right.center + armor.right.top2bottom * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);

  return bgr_img(roi);
}

void Detector::classify(Armor & armor)
{
  cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/c/AutoAim/StdDetector/tiny_resnet.onnx");
  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  net.setInput(blob);
  cv::Mat outputs = net.forward();

  // softmax
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  if (confidence > 0.5 && confidence < 0.8) std::cout << confidence << std::endl;
  armor.name = static_cast<ArmorName>(label_id);
}

}  // namespace auto_aim
