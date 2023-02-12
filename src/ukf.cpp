#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
cout << "--initialize--" << endl;
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// state vector dimension
	n_x_ = 5;

	// initial state vector
	x_ = VectorXd(n_x_);

	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_);
	P_ << 1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1;

	// Process noise standard deviation longitudinal acceleration in m/s^2
	// Need to be tuned!!
	std_a_ = 1.6;

	// Process noise standard deviation yaw acceleration in rad/s^2
	// Need to be tuned!!
	std_yawdd_ = 0.6;

	//DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
	//No need to modify anything!!!!!
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
	//DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

	/**
	TODO:

	Complete the initialization. See ukf.h for other member properties.

	Hint: one or more values initialized above might be wildly off...
	*/
	cout << "--initial user parameter--" << endl;

	lambda_ = 3 - n_x_;

	///* Augmented state dimension
	n_aug_ = n_x_ + 2;

	///* Sigma points dimension
	n_sig_ = 2 * n_aug_ + 1;

	///* X sigma point prediction dimension definition.
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Initialize weights.
	weights_ = VectorXd(n_sig_);
	cout << "set weights " << endl;
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	weights_(0) = weight_0;
	for (int i = 1; i < n_sig_; i++)
	{//2n+1 weights
		weights_(i) = .5 / (lambda_ + n_aug_);
	}
	cout << weights_ << endl;
	// Initialize measurement noice covarieance matrix
	//R_lidar_ defined in ukf.h file, lidar has 2 dimensions.
	R_lidar_ = MatrixXd(2, 2);
	R_lidar_ << std_laspx_ * std_laspx_, 0,
		0, std_laspy_*std_laspy_;

	//R_radar_ defined in ukf.h file, radar has 3 dimensions.
	R_radar_ = MatrixXd(3, 3);
	R_radar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_*std_radrd_;

	cout << "initialization finished" << endl;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
 /**
	TODO:

	Complete this function! Make sure you switch between lidar and radar
	measurements.
	*/
	if (!is_initialized_) {
		cout << "initialize lidar and radar" << endl;
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			double ro = meas_package.raw_measurements_(0);
			double phi = meas_package.raw_measurements_(1);
			double ro_dot = meas_package.raw_measurements_(2);
			double px = ro * cos(phi);
			double py = ro * sin(phi);
			double vx = ro_dot * cos(phi);
			double vy = ro_dot * sin(phi);
			double v = sqrt(vx*vx + vy * vy);
			x_ << px, py, v, 0, 0;
		}
		else {
			x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
		}

		// Saving first timestamp in seconds
		time_us_ = meas_package.timestamp_;
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	// Calculate dt
	cout << "delta t calculation " << endl;
	double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;
	// Prediction step
	cout << "delta time is " << endl;
	cout << dt << endl;

	Prediction(dt);

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		cout << "Update Radar in process measurement step " << endl;
		UpdateRadar(meas_package);
	}
	if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		cout << "Update Lidar in process measurement step" << endl;
		UpdateLidar(meas_package);
	}

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
/**
	TODO:

	Complete this function! Estimate the object's location. Modify the state
	vector, x_. Predict sigma points, the state, and the state covariance matrix.
	*/
	cout << "start prediction step" << endl;
	// 1. Generate sigma points.
	//create augmented mean vector
	VectorXd x_augument = VectorXd(n_aug_);
	x_augument.head(5) = x_;
	x_augument(5) = 0;
	x_augument(6) = 0;

	//create augmented state covariances
	MatrixXd P_augument_ = MatrixXd(n_aug_, n_aug_);
	P_augument_.fill(0.0);
	P_augument_.topLeftCorner(n_x_, n_x_) = P_;
	P_augument_(5, 5) = std_a_ * std_a_;
	P_augument_(6, 6) = std_yawdd_ * std_yawdd_;

	// Create sigma points of augmented states.
	cout << "create augmented sigma points " << endl;
	MatrixXd Xsig_aug_ = GenerateSigmaPoints(x_augument, P_augument_, lambda_, n_sig_);
	// 2. Predict Sigma Points.
	Xsig_pred_ = PredictSigmaPoints(Xsig_aug_, delta_t, n_x_, n_sig_, std_a_, std_yawdd_);
	// 3. Predict Mean and Covariance
	//predicted state mean
	x_ = Xsig_pred_ * weights_;

	//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points

										// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		if (x_diff(3) > M_PI) {
			x_diff(3) -= 2. * M_PI;
		}
		else if (x_diff(3) < -M_PI) {
			x_diff(3) += 2. * M_PI;
		}
		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
	cout << "+++++++++++++++++++++++++++finish prediction one time+++++++++++++++++++++++++++" << endl;


}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Use lidar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the lidar NIS.
	*/
	cout << "start update lidar---------------------------" << endl;
	// 1. Predit measurement
	int n_z_ = 2;
	
	// define the Zsig_ points equal to Xsig_pred_ variable.
	MatrixXd Zsig_ = Xsig_pred_.block(0, 0, n_z_, n_sig_);

	//mean predicted measurement
	VectorXd z_prediction_ = VectorXd(n_z_);
	z_prediction_.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {
		z_prediction_ = z_prediction_ + weights_(i) * Zsig_.col(i);
	}

	cout << "calculate measurement covariance matrix S" << endl;
	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
		//residual
		VectorXd z_diff = Zsig_.col(i) - z_prediction_;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S = S + R_lidar_;

	// 2. Update state
	// Incoming radar measurement
	cout << "update measurement with sensor data" << endl;
	VectorXd z = VectorXd(n_z_);
	z(0) = meas_package.raw_measurements_(0);
	z(1) = meas_package.raw_measurements_(1);

	//create matrix for cross correlation Tc
	cout << "calculation the cross correlation Tc laser" << endl;
	MatrixXd Tc = MatrixXd(n_x_, n_z_);

	Tc.fill(0.0);
	cout<<"calculation tc with x error and z error"<<endl;
	for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

		//residual
		VectorXd z_diff = Zsig_.col(i) - z_prediction_;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;k=tc*s^(-1)
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_prediction_;

	//update state mean and covariance matrix
	cout << "update laser state mean covariance matrix" << endl;
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S*K.transpose();

	//NIS Lidar Update
	cout << "calculation the NIS_Laser" << endl;
	NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
	cout << "finish  NIS lidar calculation" << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
 /**
	TODO:

	Complete this function! Use radar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the radar NIS.
	*/
	cout << "update radar----------------------------" << endl;
	// Radar measument dimension
	int n_z_ = 3;
	// 1. Predict measurement
	MatrixXd Zsig_ = MatrixXd(n_z_, n_sig_);
	//transform sigma points into measurement space
	for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

										// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig_(0, i) = sqrt(p_x*p_x + p_y * p_y);                        //r
		Zsig_(1, i) = atan2(p_y, p_x);                                 //phi
		Zsig_(2, i) = (p_x*v1 + p_y * v2) / sqrt(p_x*p_x + p_y * p_y);   //r_dot
	}

	//mean predicted measurement
	VectorXd z_prediction_ = VectorXd(n_z_);
	z_prediction_.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {
		z_prediction_ = z_prediction_ + weights_(i) * Zsig_.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
										//residual
		VectorXd z_diff = Zsig_.col(i) - z_prediction_;

		//angle normalization
		if (z_diff(1) > M_PI) {
			z_diff(1) -= 2. * M_PI;
		}
		else if (z_diff(1) < -M_PI) {
			z_diff(1) += 2. * M_PI;
		}

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S = S + R_radar_;

	// 2. Update state
	cout << "update UKF with radar new measurements " << endl;
	// new radar measurement
	//VectorXd z = meas_package.raw_measurements_;
	VectorXd z = VectorXd(n_z_);
	z(0) = meas_package.raw_measurements_(0);
	z(1) = meas_package.raw_measurements_(1);
	z(2) = meas_package.raw_measurements_(2);
	cout << " create radar matrix for cross correlation Tc" << endl;
	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z_);

	Tc.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

										//residual
		VectorXd z_diff = Zsig_.col(i) - z_prediction_;
		//angle normalization
		if (z_diff(1) > M_PI) {
			z_diff(1) -= 2. * M_PI;
		}
		else if (z_diff(1) < -M_PI) {
			z_diff(1) += 2. * M_PI;
		}

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		if (x_diff(3) > M_PI) {
			x_diff(3) -= 2. * M_PI;
		}
		else if (x_diff(3) < -M_PI) {
			x_diff(3) += 2. * M_PI;
		}

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_prediction_;

	//angle normalization
	if (z_diff(1) > M_PI) {
		z_diff(1) -= 2. * M_PI;
	}
	else if (z_diff(1) < -M_PI) {
		z_diff(1) += 2. * M_PI;
	}

	cout << "update radar x_ and P_ " << endl;
	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S*K.transpose();

	//NIS Update
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
	cout << "finish NIS_radar calculation" << endl;
}


/**
 * Predits sigma points.
 * @param Xsig : Sigma points to predict.
 * @param delta_t : Time between k and k+1 in s
 * @param n_x : State dimension.
 * @param n_sig : Sigma points dimension.
 * @param nu_am : Process noise standard deviation longitudinal acceleration in m/s^2
 * @param nu_yawdd : Process noise standard deviation yaw acceleration in rad/s^2
 */
MatrixXd UKF::PredictSigmaPoints(MatrixXd Xsig, double delta_t, int n_x, int n_sig, double nu_am, double nu_yawdd) {
  MatrixXd Xsig_pred = MatrixXd(n_x, n_sig);
  //predict sigma points
  for (int i = 0; i< n_sig; i++)
  {
    //extract values for better readability
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double nu_a = Xsig(5,i);
    double nu_yawdd = Xsig(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  return Xsig_pred;
}

/**
 *   Generate sigma points:
 *  @param x : State vector.
 *  @param P : Covariance matrix.
 *  @param lambda: Sigma points spreading parameter.
 *  @param n_sig: Sigma points dimension.
 */
MatrixXd UKF::GenerateSigmaPoints(VectorXd x, MatrixXd P, double lambda, int n_sig) {
  int n = x.size();
  //create sigma point matrix
  MatrixXd Xsig = MatrixXd( n, n_sig );

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

  Xsig.col(0) = x;

  double lambda_plue_n_x_sqrt = sqrt(lambda + n);
  for (int i = 0; i < n; i++){
      Xsig.col( i + 1 ) = x + lambda_plue_n_x_sqrt * A.col(i);
      Xsig.col( i + 1 + n ) = x - lambda_plue_n_x_sqrt * A.col(i);
  }
  return Xsig;
}
