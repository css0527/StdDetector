#include "io/serial_cboard.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
SerialBoard::SerialBoard(const std::string & config_path)
: mode(Mode::idle),
  shoot_mode(ShootMode::left_shoot),
  bullet_speed(0),
  ft_angle(0),
  queue_(5000),
  fd_(-1),
  running_(true)
{
  tools::logger()->info("初始化 SerialBoard ...");

  // 读取配置文件
  auto dev_path = read_yaml(config_path);

  // 打开串口
  fd_ = open(dev_path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    throw std::runtime_error("[SerialBoard] 无法打开串口: " + dev_path);
  }

  // 配置串口参数
  struct termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    throw std::runtime_error("[SerialBoard] 无法获取串口属性");
  }

  if (baudrate_ == 115200) {
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
  } else if (baudrate_ == 921600) {
    cfsetospeed(&tty, B921600);
    cfsetispeed(&tty, B921600);
  } else {
    tools::logger()->warn("不支持的波特率 {}, 使用 115200 作为默认值", baudrate_);
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
  }

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_iflag &= ~IGNBRK;
  tty.c_lflag = 0;
  tty.c_oflag = 0;
  tty.c_cc[VMIN] = 1;
  tty.c_cc[VTIME] = 1;
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | PARODD);
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    throw std::runtime_error("[SerialBoard] 无法设置串口属性");
  }

  tools::logger()->info("[SerialBoard] 成功打开串口: {}", dev_path);

  // 启动读线程
  read_thread_ = std::thread(&SerialBoard::readLoop, this);
}

SerialBoard::~SerialBoard()
{
  running_ = false;
  if (read_thread_.joinable()) read_thread_.join();
  if (fd_ >= 0) close(fd_);
}

void SerialBoard::readLoop()
{
  tools::logger()->info("启动串口读取线程...");

  std::vector<uint8_t> recv_buf;
  recv_buf.reserve(256);

  uint8_t byte = 0;

  while (running_) {
    int n = read(fd_, &byte, 1);
    if (n <= 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    recv_buf.push_back(byte);

    if (recv_buf.size() == 1 && recv_buf[0] != rx_header_) {
      recv_buf.clear();
      continue;
    }

    if (recv_buf.size() == rx_frame_length_) {
      if (recv_buf.front() == rx_header_ && recv_buf.back() == rx_footer_) {
        parseFrame(recv_buf);
      } else {
        tools::logger()->warn("[SerialBoard] 异常帧（帧头/帧尾错误）");
      }
      recv_buf.clear();
    }
  }
}

void SerialBoard::parseFrame(const std::vector<uint8_t> & recv_buf)
{
  if (fd_ < 0) {
    tools::logger()->warn("[SerialBoard] 串口未打开");
    return;
  }

  float pitch = 0.0f, yaw = 0.0f;
  int16_t bullet_speed = 0;
  uint8_t color = 0, buff = 0;

  memcpy(&pitch, &recv_buf[1], sizeof(float));
  memcpy(&yaw, &recv_buf[5], sizeof(float));
  memcpy(&bullet_speed, &recv_buf[9], sizeof(int16_t));
  color = recv_buf[11];
  buff = recv_buf[12];

  auto deg2rad = [](float deg) { return deg * static_cast<float>(M_PI) / 180.0f; };

  float yaw_rad = deg2rad(yaw);
  float pitch_rad = deg2rad(pitch);

  double cy = cos(yaw_rad * 0.5);
  double sy = sin(yaw_rad * 0.5);
  double cp = cos(pitch_rad * 0.5);
  double sp = sin(pitch_rad * 0.5);

  Eigen::Quaterniond q;
  q.w() = cy * cp;
  q.x() = sp * cy;
  q.y() = 0.0;
  q.z() = sy * cp;
  q.normalize();

  queue_.push({q, std::chrono::steady_clock::now()});

  // 使用 info 级别输出接收数据
  tools::logger()->info(
    "[SerialBoard] 接收数据: pitch={:.3f}, yaw={:.3f}, speed={}, color={}, buff={}", pitch, yaw,
    bullet_speed, color, buff);
}

Eigen::Quaterniond SerialBoard::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  if (queue_.empty()) {
    tools::logger()->warn("[SerialBoard] IMU队列为空，返回默认四元数");
    return Eigen::Quaterniond::Identity();
  }
  if (data_behind_.timestamp < timestamp) data_ahead_ = data_behind_;
  while (true) {
    queue_.pop(data_behind_);
    if (data_behind_.timestamp > timestamp) break;
    data_ahead_ = data_behind_;
  }

  Eigen::Quaterniond q_a = data_ahead_.q.normalized();
  Eigen::Quaterniond q_b = data_behind_.q.normalized();

  auto t_a = data_ahead_.timestamp;
  auto t_b = data_behind_.timestamp;
  std::chrono::duration<double> t_ab = t_b - t_a;
  std::chrono::duration<double> t_ac = timestamp - t_a;
  double k = (t_ab.count() == 0) ? 0.0 : std::clamp(t_ac.count() / t_ab.count(), 0.0, 1.0);

  Eigen::Vector3d euler_a = q_a.toRotationMatrix().eulerAngles(2, 1, 0);
  Eigen::Vector3d euler_b = q_b.toRotationMatrix().eulerAngles(2, 1, 0);

  Eigen::Vector3d euler_interp = euler_a + k * (euler_b - euler_a);

  Eigen::AngleAxisd yawAngle(euler_interp[0], Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchAngle(euler_interp[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rollAngle(euler_interp[2], Eigen::Vector3d::UnitX());
  Eigen::Quaterniond q_interp = yawAngle * pitchAngle * rollAngle;

  return q_interp.normalized();
}

void SerialBoard::send(io::Command command) const
{
  if (fd_ < 0) {
    tools::logger()->warn("[SerialBoard] 串口未打开，无法发送数据");
    return;
  }

  // 打印发送前信息
  tools::logger()->info(
    "[SerialBoard] before send -> control={}, yaw_deg={:.2f}, pitch_deg={:.2f}, shoot={}",
    command.control ? "true" : "false", command.yaw * 180.0 / M_PI, command.pitch * 180.0 / M_PI,
    command.shoot ? "true" : "false");

  // 构造发送帧
  uint8_t frame[12] = {0};

  // 帧头
  frame[0] = tx_header_;

  // 指令ID (控制指令)
  frame[1] = tx_ctrl_id_;

  // 数据长度 (4字节 yaw + 4字节 pitch + 2字节控制标志 = 10字节)
  frame[2] = 0x0A;

  // Yaw (float, 小端)
  float yaw_rad = command.yaw;
  memcpy(&frame[3], &yaw_rad, sizeof(float));

  // Pitch (float, 小端)
  float pitch_rad = command.pitch;
  memcpy(&frame[7], &pitch_rad, sizeof(float));

  // 控制标志: bit0 = shoot, bit1 = control
  uint16_t flags = 0;
  if (command.shoot) flags |= 0x01;
  if (command.control) flags |= 0x02;
  memcpy(&frame[11], &flags, sizeof(uint16_t));

  // 校验和 (帧头 + 指令ID + 数据长度 + 数据)
  uint8_t checksum = 0;
  for (int i = 0; i < 13; i++) {
    checksum += frame[i];
  }
  frame[13] = checksum;

  // 发送数据
  ssize_t written = write(fd_, frame, 14);

  if (written != 14) {
    perror("[SerialBoard] 串口发送失败");
  } else {
    // 打印发送成功信息
    tools::logger()->info(
      "[SerialBoard] 发送成功 -> yaw={:.2f}, pitch={:.2f}, shoot=0x{:02X}, mode=0x{:02X}",
      command.yaw * 180.0 / M_PI, command.pitch * 180.0 / M_PI, command.shoot ? 0x01 : 0x00,
      command.control ? 0x01 : 0x00);
  }

  // 打印十六进制帧
  std::cout << "[Serial] Send frame: ";
  for (int i = 0; i < 14; i++) {
    printf("%02X ", frame[i]);
  }
  std::cout << std::endl;
}

std::string SerialBoard::read_yaml(const std::string & config_path)
{
  tools::logger()->info("读取配置文件中...");
  auto yaml = tools::load(config_path);

  if (!yaml["serial_port"]) {
    throw std::runtime_error("缺少 'serial_port' 配置项");
  }
  if (!yaml["baudrate"]) {
    throw std::runtime_error("缺少 'baudrate' 配置项");
  }

  port_ = yaml["serial_port"].as<std::string>();
  baudrate_ = yaml["baudrate"].as<int>();

  // 读取串口协议参数
  if (auto serial_protocol = yaml["serial_protocol"]) {
    tx_header_ = serial_protocol["tx_header"].as<uint8_t>(0xED);
    tx_cmd_id_ = serial_protocol["tx_cmd_id"].as<uint8_t>(0x01);
    tx_data_length_ = serial_protocol["tx_data_length"].as<uint8_t>(0x04);
    tx_ctrl_id_ = serial_protocol["tx_ctrl_id"].as<uint8_t>(0x02);
    tx_shoot_id_ = serial_protocol["tx_shoot_id"].as<uint8_t>(0x03);

    rx_header_ = serial_protocol["rx_header"].as<uint8_t>(0x78);
    rx_footer_ = serial_protocol["rx_footer"].as<uint8_t>(0x76);

    tx_frame_length_ = serial_protocol["tx_frame_length"].as<size_t>(8);
    rx_frame_length_ = serial_protocol["rx_frame_length"].as<size_t>(14);

    tools::logger()->info(
      "[SerialBoard] 协议配置: TX[0x{:02X}]({} bytes), RX[0x{:02X}...0x{:02X}]({} bytes)",
      (int)tx_header_, (int)tx_frame_length_, (int)rx_header_, (int)rx_footer_, rx_frame_length_);
  } else {
    tx_header_ = 0xED;
    tx_cmd_id_ = 0x01;
    tx_data_length_ = 0x04;
    tx_ctrl_id_ = 0x02;
    tx_shoot_id_ = 0x03;
    rx_header_ = 0x78;
    rx_footer_ = 0x76;
    tx_frame_length_ = 8;
    rx_frame_length_ = 14;

    tools::logger()->warn("[SerialBoard] 使用默认协议配置");
  }

  return port_;
}

}  // namespace io
