#ifndef TOOLS_H_
#define TOOLS_H_
#include "Eigen/Dense"
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                         const vector<VectorXd> &ground_truth);

  static void NormalizeAngle(double &angle);

  static double CalculateDistance(VectorXd pointA, VectorXd pointB);
  static double CalculateAngle(VectorXd pointA, VectorXd pointB);
};

#endif /* TOOLS_H_ */
