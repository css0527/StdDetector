#ifndef DETECTOR__PNP_OPTIMIZER_HPP
#define DETECTOR__PNP_OPTIMIZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

namespace auto_aim {

class PnPOptimizer {
public:
    PnPOptimizer(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs);
    
    /**
     * @brief 降自由度优化：固定 pitch/roll，优化 yaw
     * @param tvec PnP 解出的平移向量
     * @param init_yaw 初始 yaw 角
     * @param armor_width 装甲板宽度
     * @param armor_height 装甲板高度
     * @param pitch 固定 pitch 角（弧度），默认 15°
     * @return 优化后的 yaw 角（弧度）
     */
    double optimizeYaw(const cv::Mat& tvec, 
                       double init_yaw,
                       double armor_width, 
                       double armor_height,
                       double pitch = 15.0 * CV_PI / 180.0);

    /**
     * @brief 设置图像角点（用于计算重投影误差）
     */
    void setImagePoints(const std::vector<cv::Point2f>& img_points);

private:
    /**
     * @brief 计算给定 yaw 的重投影误差
     */
    double computeReprojectionError(double yaw, 
                                    const cv::Mat& tvec,
                                    double pitch,
                                    const std::vector<cv::Point3f>& object_points);

    /**
     * @brief 三分法搜索最优 yaw
     */
    double trisectionSearch(const cv::Mat& tvec, 
                            double pitch,
                            const std::vector<cv::Point3f>& object_points,
                            double yaw_min, 
                            double yaw_max);

    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
    std::vector<cv::Point2f> img_points_;
    std::vector<cv::Point3f> object_points_;
};

} // namespace auto_aim

#endif
