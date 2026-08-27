#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{
enum Color
{
  red,
  blue,
  purple
};
const std::vector<std::string> COLORS = {"red", "blue", "purple"};

enum ArmorType
{
  INVALID,
  SMALL,
  LARGE
};
const std::vector<std::string> ARMOR_TYPES = {"invalid", "small", "large"};

enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

enum ArmorPriority
{
  first = 1,
  second,
  third,
  forth,
  fifth
};

struct Lightbar
{
  std::size_t id;
  Color color;
  cv::Point2f center, top, bottom, top2bottom;
  std::vector<cv::Point2f> points;
  double angle, angle_error, length, ratio, width;

  Lightbar() : id(0), color(Color::red), angle(0), angle_error(0), length(0), ratio(0), width(0) {}

  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id_)
  {
    id = id_;
    std::vector<cv::Point2f> corners(4);
    rotated_rect.points(&corners[0]);
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
      return a.y < b.y;
    });

    center = rotated_rect.center;
    top = (corners[0] + corners[1]) / 2;
    bottom = (corners[2] + corners[3]) / 2;
    top2bottom = bottom - top;

    points.emplace_back(top);
    points.emplace_back(bottom);

    width = cv::norm(corners[0] - corners[1]);
    angle = std::atan2(top2bottom.y, top2bottom.x);
    angle_error = std::abs(angle - CV_PI / 2);
    length = cv::norm(top2bottom);
    ratio = length / width;
  };
};

struct Armor
{
  Color color;
  Lightbar left, right;
  cv::Point2f center;
  cv::Point2f center_norm;
  std::vector<cv::Point2f> points;

  double ratio;
  double side_ratio;
  double rectangular_error;

  ArmorType type;
  ArmorName name;
  ArmorPriority priority;
  cv::Mat pattern;
  cv::Mat number_img;
  double confidence;
  bool duplicated;

  double yaw_raw;

  Armor()
  : color(Color::red),
    type(ArmorType::INVALID),
    name(ArmorName::not_armor),
    priority(ArmorPriority::first),
    confidence(0),
    duplicated(false),
    yaw_raw(0)
  {
  }

  Armor(const Lightbar & left_, const Lightbar & right_) : left(left_), right(right_)
  {
    color = left.color;
    center = (left.center + right.center) / 2;

    points.emplace_back(left.top);
    points.emplace_back(right.top);
    points.emplace_back(right.bottom);
    points.emplace_back(left.bottom);

    auto left2right = right.center - left.center;
    auto width = cv::norm(left2right);
    auto max_lightbar_length = std::max(left.length, right.length);
    auto min_lightbar_length = std::min(left.length, right.length);
    ratio = width / max_lightbar_length;
    side_ratio = max_lightbar_length / min_lightbar_length;

    auto roll = std::atan2(left2right.y, left2right.x);
    auto left_rectangular_error = std::abs(left.angle - roll - CV_PI / 2);
    auto right_rectangular_error = std::abs(right.angle - roll - CV_PI / 2);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

    type = ArmorType::SMALL;
    name = ArmorName::not_armor;
    confidence = 0.0;
  };
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP
