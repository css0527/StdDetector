#include "light_corner_corrector.hpp"
#include <numeric>
#include <cmath>

namespace auto_aim {

void LightCornerCorrector::correctCorners(Armor& armor, const cv::Mat& gray_img) {
    // 提高最小宽度阈值，避免在小灯条上出错
    // width = length / ratio，所以用 ratio 来判断灯条宽度
    // ratio 越小，灯条越细长（宽度越小）
    constexpr double MIN_RATIO_FOR_CORRECT = 5.0;  // ratio > 5 说明灯条太细，不适合矫正
    
    // 检查灯条是否足够大且形状合理
    // ratio = length / width，所以 ratio > MIN_RATIO 意味着 width < length/MIN_RATIO
    bool left_ok = armor.left.length > 15 && armor.left.ratio < MIN_RATIO_FOR_CORRECT;
    bool right_ok = armor.right.length > 15 && armor.right.ratio < MIN_RATIO_FOR_CORRECT;
    
    // 如果两个灯条都不够大，完全跳过矫正
    if (!left_ok && !right_ok) {
        return;
    }
    
    // 处理左灯条
    if (left_ok) {
        SymmetryAxis left_axis = findSymmetryAxis(gray_img, armor.left);
        
        // 验证 PCA 结果的有效性
        if (left_axis.direction.y < -0.3 && left_axis.direction.y > -3.0) {
            cv::Point2f t = findCorner(gray_img, armor.left, left_axis, "top");
            cv::Point2f b = findCorner(gray_img, armor.left, left_axis, "bottom");
            
            // 只有当两个角点都有效，且距离合理时才更新
            if (t.x >= 0 && b.x >= 0) {
                float corner_distance = cv::norm(t - b);
                float ratio = corner_distance / armor.left.length;
                
                // 新角点距离应该在原长度的 0.7-1.3 倍之间
                if (ratio > 0.7 && ratio < 1.3) {
                    armor.left.top = t;
                    armor.left.bottom = b;
                    armor.left.center = (t + b) * 0.5f;
                }
            }
        }
    }
    
    // 处理右灯条
    if (right_ok) {
        SymmetryAxis right_axis = findSymmetryAxis(gray_img, armor.right);
        
        // 验证 PCA 结果的有效性
        if (right_axis.direction.y < -0.3 && right_axis.direction.y > -3.0) {
            cv::Point2f t = findCorner(gray_img, armor.right, right_axis, "top");
            cv::Point2f b = findCorner(gray_img, armor.right, right_axis, "bottom");
            
            // 只有当两个角点都有效，且距离合理时才更新
            if (t.x >= 0 && b.x >= 0) {
                float corner_distance = cv::norm(t - b);
                float ratio = corner_distance / armor.right.length;
                
                // 新角点距离应该在原长度的 0.7-1.3 倍之间
                if (ratio > 0.7 && ratio < 1.3) {
                    armor.right.top = t;
                    armor.right.bottom = b;
                    armor.right.center = (t + b) * 0.5f;
                }
            }
        }
    }
}

SymmetryAxis LightCornerCorrector::findSymmetryAxis(const cv::Mat& gray_img, 
                                                      const Lightbar& light) {
    constexpr float MAX_BRIGHTNESS = 25;
    
    // 根据灯条大小动态调整扩展区域
    float expand_ratio = std::min(static_cast<float>(light.length), 50.0f) / 50.0f;
    int expand_x = static_cast<int>(light.length * 0.3 * expand_ratio);
    int expand_y = static_cast<int>(light.length * 0.3 * expand_ratio);
    
    // 扩展灯条区域
    cv::Rect light_box(
        std::max(0, static_cast<int>(light.center.x) - expand_x),
        std::max(0, static_cast<int>(light.center.y) - expand_y),
        std::min(gray_img.cols - std::max(0, static_cast<int>(light.center.x) - expand_x), expand_x * 2),
        std::min(gray_img.rows - std::max(0, static_cast<int>(light.center.y) - expand_y), expand_y * 2)
    );
    
    // 检查区域是否有效
    if (light_box.width <= 0 || light_box.height <= 0 || 
        light_box.width > gray_img.cols || light_box.height > gray_img.rows) {
        return {light.center, cv::Point2f(0, -1), 128.0f};
    }
    
    cv::Mat roi = gray_img(light_box);
    if (roi.empty()) {
        return {light.center, cv::Point2f(0, -1), 128.0f};
    }
    
    float mean_val = cv::mean(roi)[0];
    roi.convertTo(roi, CV_32F);
    cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);
    
    // 计算质心
    cv::Moments moments = cv::moments(roi, false);
    if (moments.m00 < 1.0) {
        return {light.center, cv::Point2f(0, -1), mean_val};
    }
    
    cv::Point2f centroid = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00) +
                          cv::Point2f(light_box.x, light_box.y);
    
    // PCA 获取对称轴 - 使用更少的采样点以提高稳定性
    std::vector<cv::Point2f> points;
    points.reserve(roi.rows * roi.cols / 2);  // 预分配内存
    
    for (int i = 0; i < roi.rows; i++) {
        for (int j = 0; j < roi.cols; j++) {
            int weight = std::round(roi.at<float>(i, j));
            // 只采样亮度较高的点
            if (weight > 0) {
                for (int k = 0; k < std::min(weight, 3); k++) {  // 限制最大采样数
                    points.emplace_back(cv::Point2f(j, i));
                }
            }
        }
    }
    
    if (points.size() < 10) {  // 提高最小点数要求
        return {centroid, cv::Point2f(0, -1), mean_val};
    }
    
    cv::Mat points_mat = cv::Mat(points).reshape(1);
    auto pca = cv::PCA(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);
    
    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), 
                     pca.eigenvectors.at<float>(0, 1));
    
    // 验证特征值的合理性
    if (pca.eigenvalues.at<float>(0) < 1.0f) {
        return {centroid, cv::Point2f(0, -1), mean_val};
    }
    
    axis = axis / cv::norm(axis);
    if (axis.y > 0) axis = -axis;  // 确保轴朝上
    
    return {centroid, axis, mean_val};
}

cv::Point2f LightCornerCorrector::findCorner(const cv::Mat& gray_img,
                                              const Lightbar& light,
                                              const SymmetryAxis& axis,
                                              const std::string& order) {
    // 更保守的搜索范围
    constexpr float START = 0.35f;
    constexpr float END = 0.65f;
    
    auto inImage = [&gray_img](const cv::Point& point) -> bool {
        return point.x >= 0 && point.x < gray_img.cols && 
               point.y >= 0 && point.y < gray_img.rows;
    };
    
    int oper = (order == "top") ? 1 : -1;
    float L = light.length;
    float dx = axis.direction.x * oper;
    float dy = axis.direction.y * oper;
    
    // 检查方向是否合理
    if (std::abs(dy) < 0.1f) {
        return cv::Point2f(-1, -1);
    }
    
    float x0 = axis.centroid.x + L * START * dx;
    float y0 = axis.centroid.y + L * START * dy;
    
    if (!inImage(cv::Point(x0, y0))) {
        return cv::Point2f(-1, -1);
    }
    
    cv::Point2f prev(x0, y0);
    cv::Point2f corner(x0, y0);
    float max_brightness_diff = 0;
    bool has_corner = false;
    
    // 减小步长，提高精度
    float step_size = 0.3f;
    int max_steps = static_cast<int>(L * (END - START) / step_size);
    
    for (int i = 0; i < max_steps; i++) {
        float x = axis.centroid.x + L * (START + i * step_size / L) * dx;
        float y = axis.centroid.y + L * (START + i * step_size / L) * dy;
        cv::Point2f cur(x, y);
        
        if (!inImage(cv::Point(cur))) break;
        
        float prev_val = gray_img.at<uchar>(prev);
        float cur_val = gray_img.at<uchar>(cur);
        float brightness_diff = prev_val - cur_val;
        
        // 亮度差必须足够大，且前一个点要足够亮
        if (brightness_diff > max_brightness_diff && 
            brightness_diff > 10.0f &&  // 最小亮度差要求
            prev_val > axis.mean_val * 0.8f) {  // 前一个点要足够亮
            max_brightness_diff = brightness_diff;
            corner = prev;
            has_corner = true;
        }
        
        prev = cur;
    }
    
    return has_corner ? corner : cv::Point2f(-1, -1);
}

} // namespace auto_aim
