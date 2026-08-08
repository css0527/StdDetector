#include "pnp_optimizer.hpp"
#include <cmath>
#include <Eigen/Geometry>

namespace auto_aim {

PnPOptimizer::PnPOptimizer(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs)
    : camera_matrix_(camera_matrix.clone())
    , distort_coeffs_(distort_coeffs.clone()) {}

void PnPOptimizer::setImagePoints(const std::vector<cv::Point2f>& img_points) {
    img_points_ = img_points;
}

double PnPOptimizer::optimizeYaw(const cv::Mat& tvec, 
                                  double init_yaw,
                                  double armor_width, 
                                  double armor_height,
                                  double pitch) {
    // 构造装甲板 3D 点（以装甲板中心为原点）
    object_points_.clear();
    object_points_.emplace_back(-armor_width/2, -armor_height/2, 0);
    object_points_.emplace_back(armor_width/2, -armor_height/2, 0);
    object_points_.emplace_back(armor_width/2, armor_height/2, 0);
    object_points_.emplace_back(-armor_width/2, armor_height/2, 0);
    
    // 在初始 yaw ±45° 范围内搜索
    double yaw_min = init_yaw - CV_PI / 4;
    double yaw_max = init_yaw + CV_PI / 4;
    
    return trisectionSearch(tvec, pitch, object_points_, yaw_min, yaw_max);
}

double PnPOptimizer::computeReprojectionError(
    double yaw, 
    const cv::Mat& tvec,
    double pitch,
    const std::vector<cv::Point3f>& object_points) {
    
    // 构造旋转矩阵：roll=0, pitch=15°, yaw=variable
    // R = Rz(yaw) * Ry(pitch) * Rx(0)
    cv::Mat rvec;
    // 简化：直接用欧拉角构造
    Eigen::AngleAxisd rollAngle(0, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d R = q.matrix();
    
    cv::Mat rmat = (cv::Mat_<double>(3,3) << 
        R(0,0), R(0,1), R(0,2),
        R(1,0), R(1,1), R(1,2),
        R(2,0), R(2,1), R(2,2));
    cv::Rodrigues(rmat, rvec);
    
    // 重投影
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, 
                     distort_coeffs_, projected_points);
    
    // 计算重投影误差（加权：垂直于灯条方向权重更高）
    double error = 0;
    for (size_t i = 0; i < projected_points.size(); i++) {
        cv::Point2f diff = img_points_[i] - projected_points[i];
        error += diff.x * diff.x + diff.y * diff.y;
    }
    
    return error;
}

double PnPOptimizer::trisectionSearch(const cv::Mat& tvec, 
                                       double pitch,
                                       const std::vector<cv::Point3f>& object_points,
                                       double yaw_min, 
                                       double yaw_max) {
    constexpr int MAX_ITER = 30;
    constexpr double TOL = 1e-5;
    
    double left = yaw_min;
    double right = yaw_max;
    
    for (int i = 0; i < MAX_ITER; i++) {
        if (right - left < TOL) break;
        
        double mid1 = left + (right - left) / 3;
        double mid2 = right - (right - left) / 3;
        
        double f1 = computeReprojectionError(mid1, tvec, pitch, object_points);
        double f2 = computeReprojectionError(mid2, tvec, pitch, object_points);
        
        if (f1 < f2) {
            right = mid2;
        } else {
            left = mid1;
        }
    }
    
    return (left + right) / 2;
}

} // namespace auto_aim
