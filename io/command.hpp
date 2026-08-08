#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

namespace io
{
struct Command
{
  bool control;
  bool shoot;
  double yaw;
  double pitch;
  double horizon_distance = 0;      // 无人机专有
  double motor_rotations = 0.0;     // 电机转圈数（控制发射装置左右移动）
};

}  // namespace io

#endif  // IO__COMMAND_HPP