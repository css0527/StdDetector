#include "kalman_filter.hpp"

namespace predict {

KalmanFilter::KalmanFilter()
{
  Q_angle = 0.001f;
  Q_gyro = 0.001f;
  R_angle = 0.03f;

  angle = 0.0f;
  Q_bias = 0.0f;

  P[0][0] = 0.0f;
  P[0][1] = 0.0f;
  P[1][0] = 0.0f;
  P[1][1] = 0.0f;

  dt = 0.0f;
  measure = 0.0f;
  predicted_angle = 0.0f;
}

float KalmanFilter::KalmanUpdate(float newAngle, float newGyro)
{
  predicted_angle = angle - Q_bias * dt + newGyro * dt;

  P[0][0] = P[0][0] + Q_angle - (P[0][1] - P[1][0]) * dt;
  P[0][1] = P[0][1] - P[1][1] * dt;
  P[1][0] = P[1][0] - P[1][1] * dt;
  P[1][0] = P[1][0] + Q_gyro;

  measure = newAngle;

  float K0 = (P[0][0]) / (P[0][0] + R_angle);
  float K1 = (P[1][0]) / (P[0][0] + R_angle);

  angle = angle + K0 * (newAngle - angle);
  Q_bias = Q_bias + K1 * (newAngle - angle);

  P[0][0] = P[0][0] - K0 * P[0][0];
  P[0][1] = P[0][1] - K0 * P[0][1];
  P[1][0] = P[1][0] - K1 * P[0][0];
  P[1][0] = P[1][0] - K1 * P[0][1];

  return angle;
}

} // namespace predict
