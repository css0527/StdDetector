#ifndef DETECTOR__LIGHT_CORNER_CORRECTOR_HPP
#define DETECTOR__LIGHT_CORNER_CORRECTOR_HPP

#include <opencv2/opencv.hpp>
#include "../include/armor.hpp"

namespace auto_aim {

struct SymmetryAxis {
    cv::Point2f centroid;
    cv::Point2f direction;
    float mean_val;
};

class LightCornerCorrector {
public:
    void correctCorners(Armor& armor, const cv::Mat& gray_img);
    
private:
    SymmetryAxis findSymmetryAxis(const cv::Mat& gray_img, 
                                   const Lightbar& light);
    cv::Point2f findCorner(const cv::Mat& gray_img,
                           const Lightbar& light,
                           const SymmetryAxis& axis,
                           const std::string& order);
};

} // namespace auto_aim

#endif
