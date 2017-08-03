#include "Eigen/Dense"
#include "ukf.hpp"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // initialized?
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2; // 1; 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3; // 0.1; 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // timestamp in microseconds
  time_us_ = 0.0;

  // lidar measurement dimension
  n_z_laser_ = 2;

  // radar measurement dimension
  n_z_radar_ = 3;

  // state dimension
  n_x_ = 5;

  // augmented state dimension
  n_aug_ = 7;

  // sigma points dimension
  n_sig_x_ = 2 * n_x_ + 1;

  // augmented sigma points dimension
  n_sig_aug_ = 2 * n_aug_ + 1;

  // spreading parameters
  lambda_x_ = 3 - n_x_;
  lambda_aug_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(0, 0) = 0.15;
  P_(1, 1) = 0.15;

  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_aug_);

  Q_ = MatrixXd(2, 2);
  Q_ << (std_a_ * std_a_), 0, 0, (std_yawdd_ * std_yawdd_);

  R_lidar_ = MatrixXd(n_z_laser_, n_z_laser_);
  R_lidar_ << (std_laspx_ * std_laspx_), 0, 0, (std_laspy_ * std_laspy_);

  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << (std_radr_ * std_radr_), 0, 0, 0, (std_radphi_ * std_radphi_), 0,
      0, 0, (std_radrd_ * std_radrd_);

  weights_ = VectorXd::Constant(n_sig_aug_, 1 / (2 * (lambda_aug_ + n_aug_)));
  weights_(0) = lambda_aug_ / (lambda_aug_ + n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0),
          meas_package.raw_measurements_(1), 0, 0, 0;
    } else {
      double rho = meas_package.raw_measurements_(0);
      double theta = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      x_ << (rho * cos(theta)), (rho * sin(theta)), rho_dot, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  // calculate delta_t in seconds!
  float delta_t = (meas_package.timestamp_ - time_us_) / 1e+6;
  // skip if sensor_type_ should be ignored
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {
    time_us_ = meas_package.timestamp_;

    // predict
    Prediction(delta_t);

    // update
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // cout << "prediction done; start udpate radar" << endl;
      UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // cout << "prediction done; start udpate lidar" << endl;
      UpdateLidar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance
  matrix.
  */
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_aug_);
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

void UKF::PredictionChase(double delta_t) {
  VectorXd stored_x = x_;
  MatrixXd stored_P = P_;

  while (delta_t > 0.1) {
    Prediction(0.1);
    delta_t -= 0.1;
  }
  Prediction(delta_t);

  predicted_x_ = x_;
  x_ = stored_x;
  P_ = stored_P;

  return;
}

/**
 * Updates the state and the state covariance matrix using a laser
 * measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the
  object's
  position. Modify the state vector, x_, and covariance, P_.
  */

  MatrixXd H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0;

  // create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_lidar_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar
 * measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use radar data to update the belief about the
  object's
  position. Modify the state vector, x_, and covariance, P_.
  */
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_aug_);
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);

  PredictRadarMeasurement(&Zsig, &z_pred, &S);

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
  UpdateState(meas_package, Zsig, z_pred, S, Tc);
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {
  // Augmented Mean State
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  // Augmented Covariance Matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner<2, 2>() << Q_;

  // square root of P_aug
  MatrixXd A_aug = P_aug.llt().matrixL();

  // set values in output matrix
  Xsig_out->col(0) = x_aug;
  // calculate sigma points
  A_aug *= sqrt(lambda_aug_ + n_aug_);
  for (int i = 0; i < n_aug_; i++) {
    Xsig_out->col(1 + i) = x_aug + A_aug.col(i);
    Xsig_out->col(1 + n_aug_ + i) = x_aug - A_aug.col(i);
  }
}

void UKF::SigmaPointPrediction(const MatrixXd Xsig_aug, double delta_t) {
  Xsig_pred_ = MatrixXd(n_x_, n_sig_aug_);

  double dt2 = delta_t * delta_t;
  double half_dt = 0.5 * dt2;
  for (int i = 0; i < n_sig_aug_; ++i) {
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yr = Xsig_aug(4, i);
    double v_a = Xsig_aug(5, i);
    double v_yr = Xsig_aug(6, i);
    double cos_yaw = cos(yaw);
    double sin_yaw = sin(yaw);

    if (fabs(yr) < 0.001) {
      Xsig_pred_(0, i) = px + v * cos_yaw * delta_t + half_dt * cos_yaw * v_a;
      Xsig_pred_(1, i) = py + v * sin_yaw * delta_t + half_dt * sin_yaw * v_a;
    } else {
      Xsig_pred_(0, i) = px + (v / yr) * (sin(yaw + yr * delta_t) - sin_yaw) +
                         half_dt * cos_yaw * v_a;
      Xsig_pred_(1, i) = py + (v / yr) * (-cos(yaw + yr * delta_t) + cos_yaw) +
                         half_dt * sin_yaw * v_a;
    }
    Xsig_pred_(2, i) = v + delta_t * v_a;
    Xsig_pred_(3, i) = yaw + yr * delta_t + half_dt * v_yr;
    Xsig_pred_(4, i) = yr + delta_t * v_yr;
  }
}

void UKF::PredictMeanAndCovariance() {
  x_ = VectorXd(5);
  x_.fill(0.0);
  for (int j = 0; j < n_sig_aug_; j++) {
    x_ += weights_(j) * Xsig_pred_.col(j);
  }

  P_ = MatrixXd(5, 5);
  P_.fill(0.0);
  for (int j = 0; j < n_sig_aug_; j++) {
    VectorXd diff = Xsig_pred_.col(j) - x_;
    Tools::NormalizeAngle(diff(3));
    P_ += weights_(j) * diff * diff.transpose();
  }
}

void UKF::PredictRadarMeasurement(MatrixXd *Zsig_out, VectorXd *z_out,
                                  MatrixXd *S_out) {
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_aug_);
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  S += R_radar_;

  for (int i = 0; i < n_sig_aug_; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double yr = Xsig_pred_(4, i);

    double rho = sqrt(px * px + py * py);
    double psi = atan2(py, px);
    double rho_d = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;

    if (fabs(rho) < 0.0001) {
      cout << "Division by zero error" << endl;
      rho = 1.0;
    }

    Zsig(0, i) = rho;
    Zsig(1, i) = psi;
    Zsig(2, i) = rho_d;

    z_pred += weights_(i) * Zsig.col(i);
  }

  for (int i = 0; i < n_sig_aug_; ++i) {
    VectorXd residual = Zsig.col(i) - z_pred;
    Tools::NormalizeAngle(residual(1));
    S += weights_(i) * residual * residual.transpose();
  }

  // write results
  *Zsig_out = Zsig;
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateState(const MeasurementPackage measPackage, const MatrixXd Zsig,
                      const VectorXd z_pred, const MatrixXd S, MatrixXd &Tc) {
  // calculate cross correlation matrix
  for (int i = 0; i < n_sig_aug_; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tools::NormalizeAngle(x_diff(3));

    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tools::NormalizeAngle(z_diff(1));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd z = VectorXd(3);
  z(0) = measPackage.raw_measurements_(0);
  z(1) = measPackage.raw_measurements_(1);
  z(2) = measPackage.raw_measurements_(2);

  VectorXd z_diff = z - z_pred;
  Tools::NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
