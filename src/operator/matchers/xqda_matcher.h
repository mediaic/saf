// Copyright 2018 The SAF Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Multi-target matching using XQDA
// Reference: Liao, Shengcai, et al. "Person re-identification by local maximal
// occurrence representation and metric learning." Proceedings of the IEEE
// Conference on Computer Vision and Pattern Recognition. 2015.

#ifndef SAF_OPERATOR_MATCHERS_XQDA_MATCHER_H_
#define SAF_OPERATOR_MATCHERS_XQDA_MATCHER_H_

#include <cmath>
#include <fstream>

#include <Eigen/Dense>

#include "common/context.h"
#include "model/model.h"
#include "operator/matchers/object_matcher.h"

class XQDAMatcher : public BaseMatcher {
 public:
  XQDAMatcher(const ModelDesc& model_desc) : model_desc_(model_desc) {}
  virtual ~XQDAMatcher() {}
  virtual bool Init() {
    // Load "W.txt" & "M_xqda.txt"
    std::string model_file = model_desc_.GetModelParamsPath();
    std::string weights_file = model_desc_.GetModelDescPath();
    W = read_matrix_file(model_file);
    M = read_matrix_file(weights_file);

    return true;
  }
  virtual double Match(const std::vector<double>& feat1,
                       const std::vector<double>& feat2) {
    CHECK(feat1.size() == feat2.size());
    std::vector<double> feat1_copy(feat1);
    std::vector<double> feat2_copy(feat2);
    double* feat1_ptr = &feat1_copy[0];
    double* feat2_ptr = &feat2_copy[0];
    Eigen::Map<Eigen::VectorXd> Xg(feat1_ptr, W.rows());
    Eigen::Map<Eigen::VectorXd> Xp(feat2_ptr, W.rows());

    Xg = Xg / Xg.norm();
    Xp = Xp / Xp.norm();

    Eigen::MatrixXd Xg_m = (Xg.transpose() * W).transpose();
    Eigen::MatrixXd Xp_m = (Xp.transpose() * W).transpose();
    double u = ((Xg_m.transpose() * M) * Xg_m)(0, 0);
    double v = ((Xp_m.transpose() * M) * Xp_m)(0, 0);
    double w = ((Xg_m.transpose() * M) * Xp_m)(0, 0);

    return u + v - 2 * w;
  }

 private:
  Eigen::MatrixXd W;
  Eigen::MatrixXd M;
  ModelDesc model_desc_;

  Eigen::MatrixXd read_matrix_file(const std::string& fname) {
   std::ifstream in(fname);
   std::string line;
   std::vector<float> buff(565248);
   size_t rows = 0, cols = 0;

   if (in.is_open()) {
     while (std::getline(in, line)) {
       char* ptr = (char*)line.c_str();
       size_t len = line.length();
       size_t temp_cols = 0;
       char* start = ptr;
       for (size_t i = 0; i < len; i++) {
         if (ptr[i] == ',') {
           buff[rows*cols+temp_cols++] = atof(start);
           start = ptr + i + 1;
         }
       }
       buff[rows*cols+temp_cols++] = atof(start);
       cols = temp_cols;
       rows++;
     }
     in.close();
   } else {
     LOG(FATAL) << "Could not load matrix file" << fname;
   }

   Eigen::MatrixXd result(rows, cols);
   for (size_t i = 0; i < rows; i++) {
     for (size_t j = 0; j < cols; j++) {
       result(i, j) = buff[i*cols+j];
     }
   }

   return result;
  }
};

#endif  // SAF_OPERATOR_MATCHERS_XQDA_MATCHER_H_
