#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

namespace predict {

class KalmanFilter
{
private:
  float angle;
  float Q_bias;
  float dt;
  float P[2][2];
  float Q_angle;
  float Q_gyro;
  float R_angle;
  float measure;
  float predicted_angle;

public:
  KalmanFilter();
  float KalmanUpdate(float newAngle, float newGyro);
  float getPreAngle() const { return predicted_angle; }
};

} // namespace predict

#endif // KALMAN_FILTER_HPP
