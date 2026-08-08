#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>

// 检测文件夹是否存在
bool dirExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

// 获取文件夹下所有图片
std::vector<std::string> getImagesInFolder(const std::string& folder) {
    std::vector<std::string> images;
    DIR* dir = opendir(folder.c_str());
    if (!dir) return images;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        // 只支持 jpg, jpeg, png, bmp
        if (name.find(".jpg") != std::string::npos ||
            name.find(".jpeg") != std::string::npos ||
            name.find(".png") != std::string::npos ||
            name.find(".bmp") != std::string::npos) {
            images.push_back(folder + "/" + name);
        }
    }
    closedir(dir);
    return images;
}

std::vector<cv::Point3f> centers_3d(const cv::Size& pattern_size, const float center_distance) {
    std::vector<cv::Point3f> centers_3d;
    for (int i = 0; i < pattern_size.height; i++)
        for (int j = 0; j < pattern_size.width; j++)
            centers_3d.push_back({j * center_distance, i * center_distance, 0});
    return centers_3d;
}

void print_yaml(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs, double error) {
    YAML::Emitter result;
    std::vector<double> camera_matrix_data(camera_matrix.begin<double>(), camera_matrix.end<double>());
    std::vector<double> distort_coeffs_data(distort_coeffs.begin<double>(), distort_coeffs.end<double>());

    result << YAML::BeginMap;
    result << YAML::Comment(fmt::format("重投影误差: {:.4f}px", error));
    result << YAML::Key << "camera_matrix";
    result << YAML::Value << YAML::Flow << camera_matrix_data;
    result << YAML::Key << "distort_coeffs";
    result << YAML::Value << YAML::Flow << distort_coeffs_data;
    result << YAML::EndMap;

    fmt::print("\n{}\n", result.c_str());
}

int main(int argc, char* argv[]) {
    std::string input_folder;
    std::string config_path = "config/calibration.yaml";
    
    // 如果没有传入参数，交互式询问
    if (argc < 2) {
        std::cout << "========================================\n";
        std::cout << "   相机标定工具 Camera Calibration\n";
        std::cout << "========================================\n\n";
        std::cout << "请输入标定图片所在文件夹路径: ";
        std::getline(std::cin, input_folder);
        
        // 去掉首尾空格
        input_folder.erase(0, input_folder.find_first_not_of(" \t\n\r"));
        input_folder.erase(input_folder.find_last_not_of(" \t\n\r") + 1);
        
        // 如果用户直接回车，使用默认路径
        if (input_folder.empty()) {
            input_folder = "assets/calib_images";
            std::cout << "使用默认路径: " << input_folder << "\n";
        }
        
        std::cout << "配置文件路径 (直接回车使用默认 config/calibration.yaml): ";
        std::string input_config;
        std::getline(std::cin, input_config);
        if (!input_config.empty()) {
            config_path = input_config;
        }
    } else {
        input_folder = argv[1];
        if (argc >= 3) {
            config_path = argv[2];
        }
    }
    
    // 检查文件夹是否存在
    if (!dirExists(input_folder)) {
        std::cerr << "错误: 文件夹不存在: " << input_folder << "\n";
        std::cerr << "请确保路径正确，或使用: ./calibrate_camera /path/to/images\n";
        return -1;
    }
    
    // 检查配置文件
    std::ifstream config_file(config_path);
    if (!config_file.good()) {
        std::cerr << "警告: 配置文件不存在: " << config_path << "，使用默认参数\n";
        config_file.close();
    } else {
        config_file.close();
    }
    
    // 读取yaml参数
    YAML::Node yaml;
    int pattern_cols = 11, pattern_rows = 8;
    double center_distance_mm = 180.0;
    
    try {
        yaml = YAML::LoadFile(config_path);
        pattern_cols = yaml["pattern_cols"].as<int>(11);
        pattern_rows = yaml["pattern_rows"].as<int>(8);
        center_distance_mm = yaml["center_distance_mm"].as<double>(180.0);
        std::cout << "标定板参数: " << pattern_cols << "x" << pattern_rows 
                  << ", 方格大小: " << center_distance_mm << "mm\n";
    } catch (const std::exception& e) {
        std::cout << "使用默认标定板参数: " << pattern_cols << "x" << pattern_rows 
                  << ", 方格大小: " << center_distance_mm << "mm\n";
    }
    
    cv::Size pattern_size(pattern_cols, pattern_rows);
    
    // 获取所有图片
    auto image_paths = getImagesInFolder(input_folder);
    if (image_paths.empty()) {
        std::cerr << "错误: 文件夹中没有图片: " << input_folder << "\n";
        return -1;
    }
    std::cout << "找到 " << image_paths.size() << " 张图片\n";
    
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;
    cv::Size img_size;
    int valid_count = 0;
    
    std::cout << "\n开始检测标定板... (按 ESC 跳过当前图片, 按 q 退出)\n";
    
    for (size_t i = 0; i < image_paths.size(); i++) {
        auto img = cv::imread(image_paths[i]);
        if (img.empty()) continue;
        
        img_size = img.size();
        std::vector<cv::Point2f> centers_2d;
        bool success = cv::findChessboardCorners(img, pattern_size, centers_2d, cv::CALIB_CB_SYMMETRIC_GRID);
        
        // 显示识别结果
        auto drawing = img.clone();
        cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
        
        // 显示进度
        std::string info = fmt::format("[{}/{}] {}", i+1, image_paths.size(), success ? "✓ 成功" : "✗ 失败");
        cv::putText(drawing, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, 
                    success ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        
        cv::resize(drawing, drawing, {}, 0.6, 0.6);
        cv::imshow("标定板检测 (按 ESC 跳过, q 退出)", drawing);
        
        int key = cv::waitKey(100);
        if (key == 'q' || key == 'Q') {
            std::cout << "用户退出\n";
            break;
        }
        if (key == 27) { // ESC
            std::cout << "跳过: " << image_paths[i] << "\n";
            continue;
        }
        
        if (success) {
            // 亚像素细化
            cv::cornerSubPix(img, centers_2d, cv::Size(11, 11), cv::Size(-1, -1),
                            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            
            obj_points.emplace_back(centers_3d(pattern_size, static_cast<float>(center_distance_mm)));
            img_points.emplace_back(centers_2d);
            valid_count++;
            std::cout << "✓ 成功: " << image_paths[i] << "\n";
        }
    }
    cv::destroyAllWindows();
    
    if (valid_count < 3) {
        std::cerr << "错误: 有效标定图片不足 (需要至少3张, 当前 " << valid_count << " 张)\n";
        return -1;
    }
    
    std::cout << "\n有效图片: " << valid_count << " 张，开始标定...\n";
    
    // 相机标定
    cv::Mat camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    auto criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);
    
    cv::calibrateCamera(obj_points, img_points, img_size, camera_matrix, distort_coeffs, 
                        rvecs, tvecs, cv::CALIB_FIX_K3, criteria);
    
    // 重投影误差
    double error_sum = 0;
    size_t total_points = 0;
    for (size_t i = 0; i < obj_points.size(); i++) {
        std::vector<cv::Point2f> reprojected_points;
        cv::projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);
        
        for (size_t j = 0; j < reprojected_points.size(); j++)
            error_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
        total_points += reprojected_points.size();
    }
    double error = error_sum / total_points;
    
    // 输出结果
    std::cout << "\n========================================\n";
    std::cout << "标定完成!\n";
    std::cout << "========================================\n";
    std::cout << "重投影误差: " << error << " px\n\n";
    
    print_yaml(camera_matrix, distort_coeffs, error);
    
    // 询问是否保存到配置文件
    std::cout << "\n是否将结果保存到 config/config.yaml? (y/n): ";
    std::string answer;
    std::getline(std::cin, answer);
    
    if (answer == "y" || answer == "Y") {
        try {
            // 读取现有配置
            YAML::Node config = YAML::LoadFile("config/config.yaml");
            
            // 提取数据
            std::vector<double> cam_data(camera_matrix.begin<double>(), camera_matrix.end<double>());
            std::vector<double> dist_data(distort_coeffs.begin<double>(), distort_coeffs.end<double>());
            
            // 更新配置
            config["camera_matrix"] = cam_data;
            config["distort_coeffs"] = dist_data;
            
            // 保存
            std::ofstream fout("config/config.yaml");
            fout << config;
            fout.close();
            
            std::cout << "✓ 已保存到 config/config.yaml\n";
        } catch (const std::exception& e) {
            std::cerr << "保存失败: " << e.what() << "\n";
            std::cerr << "请手动将上面的 camera_matrix 和 distort_coeffs 复制到 config/config.yaml\n";
        }
    }
    
    return 0;
}
