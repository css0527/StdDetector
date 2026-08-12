#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "detector/detector.hpp"
#include "io/camera.hpp"
#include "io/serial_cboard.hpp"
#include "predict/kalman_filter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "fmt/format.h"

using namespace cv;
using namespace auto_aim;
using namespace std;

// 从 config.yaml 读取装甲板物理参数
struct ArmorConfig {
    double armor_width;
    double lightbar_length;
};

ArmorConfig load_armor_config(const std::string & config_path) {
    auto yaml = tools::load(config_path);
    ArmorConfig config;
    config.armor_width = tools::read<double>(yaml, "armor_width");
    config.lightbar_length = tools::read<double>(yaml, "lightbar_length");
    return config;
}

// 获取装甲板3D点
std::vector<cv::Point3f> get_object_points(double armor_width, double lightbar_length) {
    return {
        {static_cast<float>(-armor_width / 2), static_cast<float>(-lightbar_length / 2), 0},
        {static_cast<float>(armor_width / 2), static_cast<float>(-lightbar_length / 2), 0},
        {static_cast<float>(armor_width / 2), static_cast<float>(lightbar_length / 2), 0},
        {static_cast<float>(-armor_width / 2), static_cast<float>(lightbar_length / 2), 0}
    };
}

int main() {
    // 加载配置
    std::string config_path = "/home/c/AutoAim/StdDetector/config/config.yaml";
    auto yaml = tools::load(config_path);
    
    // 读取装甲板参数
    auto armor_config = load_armor_config(config_path);
    auto object_points = get_object_points(armor_config.armor_width, armor_config.lightbar_length);
    
    // 读取相机参数
    std::vector<double> camera_matrix_data = tools::read<std::vector<double>>(yaml, "camera_matrix");
    std::vector<double> distort_coeffs_data = tools::read<std::vector<double>>(yaml, "distort_coeffs");
    
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
        camera_matrix_data[0], camera_matrix_data[1], camera_matrix_data[2],
        camera_matrix_data[3], camera_matrix_data[4], camera_matrix_data[5],
        camera_matrix_data[6], camera_matrix_data[7], camera_matrix_data[8]);
    
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << 
        distort_coeffs_data[0], distort_coeffs_data[1], distort_coeffs_data[2], 
        distort_coeffs_data[3], distort_coeffs_data[4]);
    
    // 读取视频/相机配置
    auto video_yaml = yaml["video"];
    bool use_camera = tools::read<bool>(video_yaml, "use_camera");
    std::string video_path = tools::read<std::string>(video_yaml, "video_path");
    
        // 初始化检测器
    Detector detector;
    
    // 初始化数字分类器
    std::string model_path = "/home/c/AutoAim/StdDetector/model/lenet.onnx";
    std::string label_path = "/home/c/AutoAim/StdDetector/model/label.txt";
    double classify_threshold = 0.7;
    std::vector<std::string> ignore_classes = {"negative"};
    
    try {
        detector.classifier = std::make_unique<auto_aim::NumberClassifier>(
            model_path, label_path, classify_threshold, ignore_classes);
        tools::logger()->info("Number classifier initialized with LeNet");
    } catch (const std::exception & e) {
        tools::logger()->error("Failed to init classifier: {}", e.what());
        tools::logger()->warn("Falling back to tiny_resnet classification");
    }

    
    // 初始化卡尔曼滤波器
    predict::KalmanFilter kf_yaw;
    predict::KalmanFilter kf_pitch;
    predict::KalmanFilter kf_roll;
    
    // 初始化视频/相机输入
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
            // 选择最近的目标（按优先级）
            auto_aim::Armor target = armors.front();
            
            // 绘制装甲板
            tools::draw_points(draw_img, target.points);
            
            // PnP求解
            std::vector<cv::Point2f> img_points{
                target.left.top, target.right.top, 
                target.right.bottom, target.left.bottom
            };
            
            Mat rvec, tvec, rmat;
            bool pnp_success = cv::solvePnP(
                object_points, img_points, camera_matrix, distort_coeffs, rvec, tvec);
            
            if (pnp_success) {
                // 绘制位姿信息 - 修正 draw_text 参数顺序
                tools::draw_text(draw_img, 
                    fmt::format("tvec: x{: .2f} y{: .2f} z{: .2f}", 
                        tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)),
                    cv::Point2f(10, 30), cv::Scalar(0, 255, 255), 0.7, 2);
                
                // 计算欧拉角
                cv::Rodrigues(rvec, rmat);
                double yaw = atan2(rmat.at<double>(0, 2), rmat.at<double>(2, 2));
                double pitch = -asin(rmat.at<double>(1, 2));
                double roll = atan2(rmat.at<double>(1, 0), rmat.at<double>(1, 1));
                
                // 卡尔曼滤波
                float filtered_yaw = kf_yaw.KalmanUpdate(yaw * 57.3f, 0.0f);
                float filtered_pitch = kf_pitch.KalmanUpdate(pitch * 57.3f, 0.0f);
                float filtered_roll = kf_roll.KalmanUpdate(roll * 57.3f, 0.0f);
                
                tools::draw_text(draw_img,
                    fmt::format("angles: yaw{: .2f} pitch{: .2f} roll{: .2f}", 
                        yaw * 57.3, pitch * 57.3, roll * 57.3),
                    cv::Point2f(10, 60), cv::Scalar(0, 255, 255), 0.7, 2);
                
                tools::draw_text(draw_img,
                    fmt::format("filtered: yaw{: .2f} pitch{: .2f} roll{: .2f}", 
                        filtered_yaw, filtered_pitch, filtered_roll),
                    cv::Point2f(10, 90), cv::Scalar(255, 0, 0), 0.7, 2);
                
                // 绘制预测点
                float predicted_yaw = kf_yaw.getPreAngle();
                float yaw_change = predicted_yaw - filtered_yaw;
                float pixel_scale = 20.0f;
                cv::Point2f predicted_center(
                    target.center.x + yaw_change * pixel_scale, 
                    target.center.y);
                
                cv::circle(draw_img, predicted_center, 3, cv::Scalar(255, 0, 0), -1);
                cv::circle(draw_img, target.center, 3, cv::Scalar(0, 0, 255), -1);
                cv::line(draw_img, target.center, predicted_center, 
                    cv::Scalar(0, 255, 255), 2);
            }
            
            // 绘制所有装甲板信息
            for (const auto & armor : armors) {
                tools::draw_points(draw_img, armor.points);
                tools::draw_text(draw_img,
                    fmt::format("{},{}{:.2f}", 
                        COLORS[armor.color], ARMOR_NAMES[armor.name], armor.confidence),
                    armor.left.top, cv::Scalar(0, 255, 0), 0.5, 1);
            }
        }
        
        // 显示
        cv::imshow("Armor Detection", draw_img);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            tools::logger()->info("User exit");
            break;
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}
