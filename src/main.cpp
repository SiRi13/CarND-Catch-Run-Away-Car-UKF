#include "json.hpp"
#include "tools.hpp"
#include "ukf.hpp"
#include <iostream>
#include <math.h>
#include <uWS/uWS.h>

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != std::string::npos) {
    return "";
  } else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  // Create a UKF instance
  UKF ukf;

  double target_x = 0.0;
  double target_y = 0.0;
  double target_v = 0.0;
  double remaining_t = 0.0;
  long long previous_t;

  h.onMessage([&ukf, &target_x, &target_y, &target_v, &remaining_t,
               &previous_t](uWS::WebSocket<uWS::SERVER> ws, char *data,
                            size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(std::string(data));
      if (s != "") {

        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          VectorXd hunter = VectorXd(3);
          hunter << double(std::stod(j[1]["hunter_x"].get<std::string>())),
              double(std::stod(j[1]["hunter_y"].get<std::string>())),
              double(std::stod(j[1]["hunter_heading"].get<std::string>()));

          string lidar_measurment = j[1]["lidar_measurement"];

          MeasurementPackage meas_package_L;
          istringstream iss_L(lidar_measurment);
          long long timestamp_L;

          // reads first element from the current line
          string sensor_type_L;
          iss_L >> sensor_type_L;

          // read measurements at this timestamp
          meas_package_L.sensor_type_ = MeasurementPackage::LASER;
          meas_package_L.raw_measurements_ = VectorXd(2);
          float px;
          float py;
          iss_L >> px;
          iss_L >> py;
          meas_package_L.raw_measurements_ << px, py;
          iss_L >> timestamp_L;
          meas_package_L.timestamp_ = timestamp_L;

          ukf.ProcessMeasurement(meas_package_L);

          string radar_measurment = j[1]["radar_measurement"];

          MeasurementPackage meas_package_R;
          istringstream iss_R(radar_measurment);
          long long timestamp_R;

          // reads first element from the current line
          string sensor_type_R;
          iss_R >> sensor_type_R;

          // read measurements at this timestamp
          meas_package_R.sensor_type_ = MeasurementPackage::RADAR;
          meas_package_R.raw_measurements_ = VectorXd(3);
          float ro;
          float theta;
          float ro_dot;
          iss_R >> ro;
          iss_R >> theta;
          iss_R >> ro_dot;
          meas_package_R.raw_measurements_ << ro, theta, ro_dot;
          iss_R >> timestamp_R;
          meas_package_R.timestamp_ = timestamp_R;

          ukf.ProcessMeasurement(meas_package_R);

          target_x = ukf.x_[0];
          target_y = ukf.x_[1];
          target_v = ukf.x_[2];

          double new_heading = 0.0;
          double distance = 0.0;
          double turn_correction = 0.0;

          // calculate angle with respect to x-axis
          new_heading = Tools::CalculateAngle(ukf.x_, hunter);
          Tools::NormalizeAngle(new_heading);
          // turn towards the target
          turn_correction = new_heading - hunter(2);
          Tools::NormalizeAngle(turn_correction);

          // calculate distance to target
          distance = Tools::CalculateDistance(ukf.x_, hunter);

          // wait for certain amount of data
          if (ukf.P_.maxCoeff() < 0.1) {
            // run prediction up to 5 [s] into the future
            for (double delta_t = 0.1; delta_t < 5; delta_t += 0.05) {
              ukf.PredictionChase(delta_t);

              // get distance from hunter to the point where the target might be
              double hunter_prediction =
                  Tools::CalculateDistance(ukf.predicted_x_, hunter);

              // compute distance where hunter catches target
              double hunter_future_target =
                  hunter_prediction - delta_t * target_v;

              // hunter should be closer to predicted point than target
              if (hunter_future_target < 0) {
                remaining_t = delta_t;
                // recompute heading angle
                new_heading = Tools::CalculateAngle(ukf.predicted_x_, hunter);
                Tools::NormalizeAngle(new_heading);
                cout << "Catching Car in\t" << remaining_t << "[s]\nat\t"
                     << new_heading * 180 / M_PI << " [deg]" << endl;

                turn_correction = new_heading - hunter(2);
                Tools::NormalizeAngle(turn_correction);

                distance = hunter_prediction * 0.3;
                break;
              }
            }
          }

          // smooth turning of hunter
          turn_correction *= 0.1;

          //   cout << "Turning:\t" << turn_correction << "\nDistance:\t" <<
          //   distance
          //        << endl;

          json msgJson;
          msgJson["turn"] = turn_correction;
          msgJson["dist"] = distance;
          auto msg = "42[\"move_hunter\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }

  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
