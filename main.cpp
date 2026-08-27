#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "detector/detector.hpp"
#include "fmt/format.h"
#include "io/camera.hpp"
#include "io/serial_cboard.hpp"
#include "predict/kalman_filter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

using namespace cv;
using namespace auto_aim;
using namespace std;

DetectorParams loadDetectorParams(const YAML::Node & yaml)
{
  DetectorParams params;

  // 传统方法参数 - 使用带默认值的 read
  params.binary_thres = tools::read<int>(yaml, "threshold", 120);
  params.light_params.max_angle = tools::read<double>(yaml, "max_angle_error", 45.0);
  params.light_params.min_ratio = 1.0 / tools::read<double>(yaml, "max_lightbar_ratio", 20.0);
  params.light_params.max_ratio = 1.0 / tools::read<double>(yaml, "min_lightbar_ratio", 1.5);

  params.armor_params.min_light_ratio = 0.6;
  params.armor_params.min_small_center_distance = tools::read<double>(yaml, "min_armor_ratio", 1.0);
  params.armor_params.max_small_center_distance = tools::read<double>(yaml, "max_armor_ratio", 5.0);
  params.armor_params.max_angle = tools::read<double>(yaml, "max_rectangular_error", 25.0);

  params.classifier_threshold = tools::read<double>(yaml, "min_confidence", 0.8);

  return params;
}

int main()
{
  // 初始化日志
  tools::set_logger();
  tools::logger()->info("AutoAim System Starting...");

  // 加载配置
  std::string config_path = "/home/scurm/StdDetector/config/config.yaml";
  auto yaml = tools::load(config_path);

  // 读取敌方颜色
  std::string enemy_color_str = tools::read<std::string>(yaml, "enemy_color", "red");
  Color enemy_color = (enemy_color_str == "red") ? Color::red : Color::blue;
  tools::logger()->info("Enemy color: {}", enemy_color_str);

  // 读取相机参数
  std::vector<double> camera_matrix_data = tools::read<std::vector<double>>(yaml, "camera_matrix");
  std::vector<double> distort_coeffs_data =
    tools::read<std::vector<double>>(yaml, "distort_coeffs");

  cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) << camera_matrix_data[0], camera_matrix_data[1], camera_matrix_data[2],
     camera_matrix_data[3], camera_matrix_data[4], camera_matrix_data[5], camera_matrix_data[6],
     camera_matrix_data[7], camera_matrix_data[8]);

  cv::Mat distort_coeffs =
    (cv::Mat_<double>(1, 5) << distort_coeffs_data[0], distort_coeffs_data[1],
     distort_coeffs_data[2], distort_coeffs_data[3], distort_coeffs_data[4]);

  // 读取装甲板参数
  float armor_width = static_cast<float>(tools::read<double>(yaml, "armor_width", 0.135));
  float lightbar_length = static_cast<float>(tools::read<double>(yaml, "lightbar_length", 0.056));

  // 构建3D物体点
  std::vector<cv::Point3f> object_points = {
    {-armor_width / 2, -lightbar_length / 2, 0},
    {armor_width / 2, -lightbar_length / 2, 0},
    {armor_width / 2, lightbar_length / 2, 0},
    {-armor_width / 2, lightbar_length / 2, 0}};

  // 初始化检测器
  DetectorParams detector_params = loadDetectorParams(yaml);
  Detector detector(detector_params);
  detector.setEnemyColor(enemy_color);

  // 读取视频/相机配置
  auto video_yaml = yaml["video"];
  bool use_camera = tools::read<bool>(video_yaml, "use_camera", false);
  std::string video_path = tools::read<std::string>(video_yaml, "video_path", "");

  // 初始化视频/相机
  VideoCapture video;
  std::unique_ptr<io::Camera> camera;

  if (use_camera) {
    try {
      camera = std::make_unique<io::Camera>(config_path);
      tools::logger()->info("Camera initialized");
    } catch (const std::exception & e) {
      tools::logger()->error("Failed to init camera: {}", e.what());
      tools::logger()->info("Falling back to video file");
      video.open(video_path);
    }
  } else {
    video.open(video_path);
    if (!video.isOpened()) {
      tools::logger()->error("Cannot open video: {}", video_path);
      return -1;
    }
    tools::logger()->info("Video opened: {}", video_path);
  }

  // 卡尔曼滤波器
  predict::KalmanFilter kf_yaw, kf_pitch, kf_roll;

  // 主循环
  auto last_time = std::chrono::steady_clock::now();
  int frame_count = 0;

  while (true) {
    Mat bgr_img;
    std::chrono::steady_clock::time_point timestamp;

    if (use_camera && camera) {
      camera->read(bgr_img, timestamp);
    } else {
      video >> bgr_img;
      timestamp = std::chrono::steady_clock::now();
    }

    if (bgr_img.empty()) {
      tools::logger()->warn("Empty frame");
      if (!use_camera) break;
      continue;
    }

    // 计算FPS
    frame_count++;
    auto current_time = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(current_time, last_time);
    if (dt >= 1.0) {
      double fps = frame_count / dt;
      tools::logger()->debug("FPS: {:.1f}", fps);
      frame_count = 0;
      last_time = current_time;
    }

    // 检测装甲板
    auto armors = detector.detect(bgr_img);

    Mat draw_img = bgr_img.clone();

    if (!armors.empty()) {
      auto target = armors.front();

      // 绘制装甲板
      tools::draw_points(draw_img, target.points);

      // PnP求解
      std::vector<cv::Point2f> img_points = {
        target.left.top, target.right.top, target.right.bottom, target.left.bottom};

      Mat rvec, tvec, rmat;
      bool pnp_success = cv::solvePnP(
        object_points, img_points, camera_matrix, distort_coeffs, rvec, tvec, false,
        cv::SOLVEPNP_IPPE);

      if (pnp_success) {
        double distance = cv::norm(tvec);

        cv::Rodrigues(rvec, rmat);
        double yaw = atan2(rmat.at<double>(0, 2), rmat.at<double>(2, 2));
        double pitch = -asin(rmat.at<double>(1, 2));
        double roll = atan2(rmat.at<double>(1, 0), rmat.at<double>(1, 1));

        // 卡尔曼滤波
        float filtered_yaw = kf_yaw.KalmanUpdate(yaw * 57.3f, 0.0f);
        float filtered_pitch = kf_pitch.KalmanUpdate(pitch * 57.3f, 0.0f);
        float filtered_roll = kf_roll.KalmanUpdate(roll * 57.3f, 0.0f);

        tools::draw_text(
          draw_img, fmt::format("Dist: {:.2f}m", distance), cv::Point(10, 30),
          cv::Scalar(0, 255, 255), 0.7, 2);

        tools::draw_text(
          draw_img,
          fmt::format(
            "Angles: Yaw {:.2f} Pitch {:.2f} Roll {:.2f}", yaw * 57.3, pitch * 57.3, roll * 57.3),
          cv::Point(10, 60), cv::Scalar(0, 255, 255), 0.7, 2);

        tools::draw_text(
          draw_img,
          fmt::format(
            "Filtered: Yaw {:.2f} Pitch {:.2f} Roll {:.2f}", filtered_yaw, filtered_pitch,
            filtered_roll),
          cv::Point(10, 90), cv::Scalar(255, 0, 0), 0.7, 2);

        // 绘制预测方向
        float predicted_yaw = kf_yaw.getPreAngle();
        float yaw_change = predicted_yaw - filtered_yaw;
        float pixel_scale = 20.0f;
        cv::Point2f predicted_center(target.center.x + yaw_change * pixel_scale, target.center.y);

        cv::circle(draw_img, predicted_center, 5, cv::Scalar(255, 0, 0), -1);
        cv::circle(draw_img, target.center, 5, cv::Scalar(0, 0, 255), -1);
        cv::line(draw_img, target.center, predicted_center, cv::Scalar(0, 255, 255), 2);

        cv::drawFrameAxes(draw_img, camera_matrix, distort_coeffs, rvec, tvec, 0.1);
      }

      // 绘制所有装甲板信息
      for (const auto & armor : armors) {
        std::string name = ARMOR_NAMES[armor.name];
        std::string color_str = COLORS[armor.color];
        tools::draw_text(
          draw_img, fmt::format("{}{}{:.2f}", color_str, name, armor.confidence), armor.left.top,
          cv::Scalar(0, 255, 0), 0.5, 1);
      }
    }

    cv::imshow("Armor Detection", draw_img);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      tools::logger()->info("User exit");
      break;
    }

    if (key == 'b') {
      cv::imshow("Binary", detector.getBinaryImage());
    }
  }

  cv::destroyAllWindows();
  tools::logger()->info("AutoAim System stopped");
  return 0;
}
