#ifndef IO__SERIALBOARD_HPP
#define IO__SERIALBOARD_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "io/command.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"
#include "io/public_param.hpp"

namespace io
{
class SerialBoard
{
public:
  double bullet_speed;
  Mode mode;
  ShootMode shoot_mode;
  double ft_angle;

  explicit SerialBoard(const std::string & config_path);
  ~SerialBoard();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
  void send(io::Command command) const;

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  tools::ThreadSafeQueue<IMUData> queue_;

  int fd_;  // 串口文件描述符
  std::thread read_thread_;
  std::atomic<bool> running_;

  std::string port_;
  int baudrate_;

  // 串口通信协议参数
  uint8_t tx_header_;      // 发送帧头
  uint8_t tx_cmd_id_;      // 发送指令ID
  uint8_t tx_data_length_; // 发送数据长度
  uint8_t rx_header_;      // 接收帧头
  uint8_t rx_footer_;      // 接收帧尾
  size_t tx_frame_length_;  // 发送帧长度
  size_t rx_frame_length_;  // 接收帧长度

  IMUData data_ahead_;
  IMUData data_behind_;

  void readLoop();
  void parseFrame(const std::vector<uint8_t> & frame);
  std::string read_yaml(const std::string & config_path);
};

}  // namespace io

#endif  // IO__SERIALBOARD_HPP
