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

// An example application of tracking workload.

#include <csignal>

#include <boost/program_options.hpp>

#include "saf.h"

namespace po = boost::program_options;

/////// Global vars
std::vector<std::shared_ptr<Camera>> cameras;
std::shared_ptr<ObjectDetector> object_detector;
std::vector<std::shared_ptr<Operator>> trackers;
std::shared_ptr<Operator> feature_extractor;
std::shared_ptr<Operator> object_matcher;
std::vector<StreamReader*> output_readers;
std::shared_ptr<Operator> sender;

cv::Mat ReadMatrixFile(const std::string& fname) {
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

  cv::Mat result(rows, cols, CV_64FC1);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result.at<double>(i, j) = buff[i*cols+j];
    }
  }

  return result;
}

cv::Point ImageToMap(const cv::Point pt, const cv::Mat intr_m, 
                     const cv::Mat extr_m, const cv::Mat distort_coef) {
  cv::Mat original_pt(1,1, CV_64FC2);
  original_pt.at<cv::Vec3d>(0,0)[0] = pt.x;
  original_pt.at<cv::Vec3d>(0,0)[1] = pt.y;
  cv::Mat undistort_pt;
  cv::undistortPoints(original_pt, undistort_pt, intr_m, distort_coef, cv::noArray(), intr_m);
  cv::Mat undistort_pt_ = cv::Mat::ones(3, 1, CV_64FC1);
  undistort_pt_.at<double>(0,0) = undistort_pt.at<cv::Vec3d>(0,0)[0];
  undistort_pt_.at<double>(1,0) = undistort_pt.at<cv::Vec3d>(0,0)[1];
  cv::Mat trfm;
  cv::hconcat(extr_m(cv::Rect(0,0,2,3)), extr_m(cv::Rect(3,0,1,3)), trfm); // 3*3
  cv::Mat world_pt = (intr_m * trfm).inv() * undistort_pt_; // 3*1
  double world_x = world_pt.at<double>(0,0) / world_pt.at<double>(2,0);
  double world_y = world_pt.at<double>(1,0) / world_pt.at<double>(2,0);
  return cv::Point(world_x, world_y);
}

cv::Point MapToScreen(const cv::Point pt, const cv::Mat display) {
  return cv::Point(pt.x, display.rows-pt.y);
}

std::vector<cv::Mat> Get10Icons(const std::string& icon_folder) {
  std::vector<cv::Mat> icons;
  int select_id[10] = {25,1,4,7,10,19,39,60,100,147};
  for (int i = 0; i < 10; ++i) {
    char file_name[1024];
    std::sprintf(file_name, "%s/%d.png", icon_folder.c_str(), select_id[i]);
    icons.push_back(cv::imread(file_name, cv::IMREAD_UNCHANGED));
  }
  return icons;
}

std::vector<cv::Scalar> Get10Colors() {
  std::vector<cv::Scalar> colors;
  cv::Scalar select_color[10] = {
    cv::Scalar(177,253,253),
    cv::Scalar(177,256,201),
    cv::Scalar(177,215,253),
    cv::Scalar(253,253,177),
    cv::Scalar(201,253,175),
    cv::Scalar(253,177,215),
    cv::Scalar(253,177,253),
    cv::Scalar(253,209,177),
    cv::Scalar(253,177,179),
    cv::Scalar(253,177,177)
  };
  for (int i = 0; i < 10; ++i) {
    colors.push_back(select_color[i]);
  }
  return colors;
}

void OverlayFootstep(cv::Mat& map_display, cv::Point prev_pt, cv::Point pt, 
                     cv::Scalar color, size_t step_size, size_t step_n, 
                     const std::string& footstep_img) {
  cv::Mat foot = cv::imread(footstep_img, cv::IMREAD_UNCHANGED);
  cv::Point motion_vec = pt - prev_pt;
  if (motion_vec.x == 0 && motion_vec.y == 0) return;
  CHECK(motion_vec.x != 0 || motion_vec.y != 0);
  float angle = cv::fastAtan2(motion_vec.y, motion_vec.x);
  cv::Mat R = cv::getRotationMatrix2D(cv::Point(foot.cols/2, foot.rows/2), 180+90-angle, 1);
  cv::Mat foot_r;
  cv::warpAffine(foot, foot_r, R, foot.size());
  cv::Mat rgba[4];
  cv::split(foot_r, rgba);
  cv::Mat foot_overlay(foot_r.rows, foot_r.cols, CV_8UC3, color);
  if (step_n % (2*step_size) == 0) {
    // Draw right foot
    cv::Point top_left_pt;
    if (abs(motion_vec.y) >= abs(motion_vec.x) && motion_vec.y > 0 )
      top_left_pt = cv::Point(pt.x - foot_overlay.cols, pt.y - foot_overlay.rows);
    else if (abs(motion_vec.y) >= abs(motion_vec.x) && motion_vec.y <= 0 )
      top_left_pt = cv::Point(pt.x, pt.y);
    else if (abs(motion_vec.y) < abs(motion_vec.x) && motion_vec.x > 0 )
      top_left_pt = cv::Point(pt.x - foot_overlay.cols, pt.y);
    else 
      top_left_pt = cv::Point(pt.x, pt.y - foot_overlay.rows);
    cv::Rect roi(top_left_pt.x, top_left_pt.y, foot_overlay.cols, foot_overlay.rows);
    if (!InsideImage(roi, map_display)) return;
    foot_overlay.copyTo(map_display(roi), rgba[3]);
  } else if (step_n % (2*step_size) == step_size) {
    // Draw left foot
    cv::Point top_left_pt;
    if (abs(motion_vec.y) >= abs(motion_vec.x) && motion_vec.y > 0 )
      top_left_pt = cv::Point(pt.x, pt.y - foot_overlay.rows);
    else if (abs(motion_vec.y) >= abs(motion_vec.x) && motion_vec.y <= 0 )
      top_left_pt = cv::Point(pt.x - foot_overlay.cols, pt.y);
    else if (abs(motion_vec.y) < abs(motion_vec.x) && motion_vec.x > 0 )
      top_left_pt = cv::Point(pt.x - foot_overlay.cols, pt.y - foot_overlay.rows);
    else 
      top_left_pt = cv::Point(pt.x, pt.y);
    cv::Rect roi(top_left_pt.x, top_left_pt.y,
                 foot_overlay.cols, foot_overlay.rows);
    if (!InsideImage(roi, map_display)) return;
    foot_overlay.copyTo(map_display(roi), rgba[3]);
    
  }
}

void OverlayIcon(cv::Mat& display, const cv::Point pos, const cv::Mat& overlay_icon, 
                 const cv::Scalar color, const cv::Size size) {
  cv::Mat resized_icon, rgb, rgba[4];
  cv::rectangle(display, pos, pos + cv::Point(size.width, size.height),
                color, CV_FILLED);
  cv::resize(overlay_icon, resized_icon, size, 0, 0, CV_INTER_LINEAR);
  cv::split(resized_icon, rgba);
  cv::cvtColor(resized_icon, rgb, cv::COLOR_RGBA2RGB);
  cv::Rect roi(pos.x, pos.y, size.width, size.height);
  if (InsideImage(roi, display)) {
    rgb.copyTo(display(roi), rgba[3]);
  }
}

void CleanUp() {
  if (sender != nullptr && sender->IsStarted()) sender->Stop();

  for (auto reader : output_readers) {
    reader->UnSubscribe();
  }

  if (object_matcher != nullptr && object_matcher->IsStarted())
    object_matcher->Stop();

  if (feature_extractor != nullptr && feature_extractor->IsStarted())
    feature_extractor->Stop();

  for (auto tracker : trackers) {
    if (tracker->IsStarted()) tracker->Stop();
  }

  if (object_detector != nullptr && object_detector->IsStarted())
    object_detector->Stop();

  for (auto camera : cameras) {
    if (camera->IsStarted()) camera->Stop();
  }
}

void SignalHandler(int) {
  std::cout << "Received SIGINT, try to gracefully exit" << std::endl;
  exit(0);
}

void Run(const std::vector<std::string>& camera_names,
         const std::string& detector_type, const std::string& detector_model,
         bool display, bool display_map, float detector_confidence_threshold,
         float detector_idle_duration, const std::string& detector_targets,
         int face_min_size, const std::string& tracker_type,
         const std::string& extractor_type, const std::string& extractor_model,
         const std::string& matcher_type, float matcher_distance_threshold,
         const std::string& matcher_model, const std::string& sender_endpoint,
         const std::string& sender_package_type, int frames,
         const std::string& map, const std::string& icon_folder,
         const std::string& footstep_img) {
  // Silence complier warning sayings when certain options are turned off.
  (void)detector_confidence_threshold;
  (void)detector_targets;

  std::cout << "Run tracker_obj demo" << std::endl;

  std::signal(SIGINT, SignalHandler);

  size_t batch_size = camera_names.size();
  CameraManager& camera_manager = CameraManager::GetInstance();
  ModelManager& model_manager = ModelManager::GetInstance();

  // Check options
  CHECK(model_manager.HasModel(detector_model))
      << "Model " << detector_model << " does not exist";
  for (auto camera_name : camera_names) {
    CHECK(camera_manager.HasCamera(camera_name))
        << "Camera " << camera_name << " does not exist";
  }

  ////// Start cameras, operators

  for (auto camera_name : camera_names) {
    auto camera = camera_manager.GetCamera(camera_name);
    cameras.push_back(camera);
  }

  // detector
  auto model_descs = model_manager.GetModelDescs(detector_model);
  auto t = SplitString(detector_targets, ",");
  std::set<std::string> targets;
  for (const auto& m : t) {
    if (!m.empty()) targets.insert(m);
  }
  object_detector = std::make_shared<ObjectDetector>(
      detector_type, model_descs, batch_size, detector_confidence_threshold,
      detector_idle_duration, targets, face_min_size);
  object_detector->SetBlockOnPush(true);
  for (size_t i = 0; i < batch_size; i++) {
    object_detector->SetSource("input" + std::to_string(i),
                               cameras[i]->GetSink("output"));
  }

  // tracker
  for (size_t i = 0; i < batch_size; i++) {
    std::shared_ptr<Operator> tracker(new ObjectTracker(tracker_type, cameras[i]->GetMask()));
    tracker->SetSource("input",
                       object_detector->GetSink("output" + std::to_string(i)));
    trackers.push_back(tracker);
  }

  // extractor
  if (!extractor_type.empty()) {
    auto model_desc = model_manager.GetModelDesc(extractor_model);
    feature_extractor = std::make_shared<FeatureExtractor>(
        model_desc, batch_size, extractor_type);
    for (size_t i = 0; i < batch_size; i++) {
      feature_extractor->SetSource(FeatureExtractor::GetSourceName(i),
                                   trackers[i]->GetSink("output"));
    }
  }

  // matcher
  if (!matcher_type.empty()) {
    auto model_desc = model_manager.GetModelDesc(matcher_model);
    object_matcher = std::make_shared<ObjectMatcher>(
        matcher_type, batch_size, matcher_distance_threshold, model_desc);
    for (size_t i = 0; i < batch_size; i++) {
      if (!extractor_type.empty()) {
        object_matcher->SetSource(
            "input" + std::to_string(i),
            feature_extractor->GetSink(FeatureExtractor::GetSinkName(i)));
      } else {
        object_matcher->SetSource("input" + std::to_string(i),
                                  trackers[i]->GetSink("output"));
      }
    }
  }

  // output_readers
  for (size_t i = 0; i < batch_size; i++) {
    if (!matcher_type.empty()) {
      output_readers.push_back(
          object_matcher->GetSink(ObjectMatcher::GetSinkName(i))->Subscribe());
    } else {
      output_readers.push_back(trackers[i]->GetSink("output")->Subscribe());
    }
  }

  // sender
  if (!sender_endpoint.empty()) {
    sender = std::make_shared<Sender>(sender_endpoint, sender_package_type,
                                      batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      if (!matcher_type.empty()) {
        sender->SetSource(
            Sender::GetSourceName(i),
            object_matcher->GetSink(ObjectMatcher::GetSinkName(i)));
      } else {
        if (!extractor_type.empty()) {
          sender->SetSource(
              Sender::GetSourceName(i),
              feature_extractor->GetSink(FeatureExtractor::GetSinkName(i)));
        } else {
          sender->SetSource(Sender::GetSourceName(i),
                            trackers[i]->GetSink("output"));
        }
      }
    }
  }

  for (auto camera : cameras) {
    if (!camera->IsStarted()) {
      camera->Start();
    }
  }

  object_detector->Start();

  for (auto tracker : trackers) {
    tracker->Start();
  }

  if (!extractor_type.empty()) {
    feature_extractor->Start();
  }

  if (!matcher_type.empty()) {
    object_matcher->Start();
  }

  if (!sender_endpoint.empty()) {
    sender->Start();
  }

  //////// Operator started, display the results

  if (display) {
    for (std::string camera_name : camera_names) {
      cv::namedWindow(camera_name, cv::WINDOW_NORMAL);
    }
  }
  if (display) {
    for (size_t i = 0; i < batch_size; i++) {
      cv::resizeWindow(camera_names[i], 927, 500);
      if (i % 3 == 0) {
        cv::moveWindow(camera_names[i], 66, 24);
      } else if (i % 3 == 1) {
        cv::moveWindow(camera_names[i], 66, 24+528);
      } else {
        cv::moveWindow(camera_names[i], 66+927, 24+528);
      } 
    }
  }

  std::vector<cv::Mat> extr_mats;
  std::vector<cv::Mat> intr_mats;
  std::vector<cv::Mat> distort_coefs;
  if (display_map) {
    cv::namedWindow("Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Map", 927, 500);
    cv::moveWindow("Map", 66+927, 24);
    for (size_t i = 0; i < batch_size; i++) {
      auto extr_mat = ReadMatrixFile(cameras[i]->GetExtrinsicMat());
      auto intr_mat = ReadMatrixFile(cameras[i]->GetIntrinsicMat());
      auto distort_coef = ReadMatrixFile(cameras[i]->GetDistortCoef());
      extr_mats.push_back(extr_mat);
      intr_mats.push_back(intr_mat);
      distort_coefs.push_back(distort_coef);
    }
  }

  size_t map_mode = 0, step_size = 10;
  int max_color_count = 10;
  const std::vector<cv::Scalar> colors = Get10Colors();
  const std::vector<cv::Mat> icons = Get10Icons(icon_folder);
  int color_count = 0;
  std::map<std::string, int> ids_colors;
  std::unordered_map<std::string,
                     boost::circular_buffer<boost::optional<cv::Point>>>
      cam_trajectories[camera_names.size()]; // For display
  std::unordered_map<std::string,
                     boost::circular_buffer<boost::optional<cv::Point>>>
                     map_tracks; // For map display
  while (true) {
    auto map_display = cv::imread(map, CV_LOAD_IMAGE_COLOR);
    for (size_t i = 0; i < camera_names.size(); i++) {
      auto reader = output_readers[i];
      auto frame = reader->PopFrame();
      // Get Frame Rate
      double rate = reader->GetPushFps();
      std::ostringstream label;
      label.precision(2);
      label << rate << " FPS";
      auto label_string = label.str();
      if (display || display_map) {
        auto image = frame->GetValue<cv::Mat>("original_image");
        auto image_display = image.clone();
        auto bboxes = frame->GetValue<std::vector<Rect>>("bounding_boxes");
        if (frame->Count("face_landmarks") > 0) {
          auto face_landmarks =
              frame->GetValue<std::vector<FaceLandmark>>("face_landmarks");
          for (const auto& m : face_landmarks) {
            for (int j = 0; j < 5; j++)
              cv::circle(image_display, cv::Point(m.x[j], m.y[j]), 1,
                         cv::Scalar(255, 255, 0), 5);
          }
        }

        CHECK(frame->Count("ids") > 0);
        auto ids = frame->GetValue<std::vector<std::string>>("ids");
        auto tags = frame->GetValue<std::vector<std::string>>("tags");

        // Update trajectories
        std::set<size_t> used_id_index;
        auto& trajectories = cam_trajectories[i];
        for (auto& m : trajectories) {
          bool found = false;
          for (size_t j = 0; j < ids.size(); ++j) {
            if (m.first == ids[j]) {
              cv::Point pt(bboxes[j].px + bboxes[j].width / 2,
                           bboxes[j].py + bboxes[j].height);
              m.second.push_back(pt);
              CHECK(map_tracks.find(ids[j]) != map_tracks.end());
              map_tracks[m.first].push_back(
                ImageToMap(pt, intr_mats[i],
                           extr_mats[i], distort_coefs[i]));
              used_id_index.insert(j);
              found = true;
              break;
            }
          }
          if (!found) {
            m.second.push_back(boost::optional<cv::Point>());
          }
        }
        for (size_t j = 0; j < ids.size(); ++j) {
          if (used_id_index.find(j) == used_id_index.end()) {
            cv::Point pt(bboxes[j].px + bboxes[j].width / 2,
                         bboxes[j].py + bboxes[j].height);
            trajectories.insert(std::make_pair(
                ids[j],
                boost::circular_buffer<boost::optional<cv::Point>>(4096)));
            trajectories.at(ids[j]).push_back(pt);
            map_tracks.insert(std::make_pair(
                ids[j],
                boost::circular_buffer<boost::optional<cv::Point>>(4096)));
            map_tracks.at(ids[j]).push_back(
                ImageToMap(pt, intr_mats[i],
                           extr_mats[i], distort_coefs[i]));
          }
        }

        for (size_t j = 0; j < ids.size(); ++j) {
          // Get the color
          int color_index;
          auto it = ids_colors.find(ids[j]);
          if (it == ids_colors.end()) {
            ids_colors.insert(std::make_pair(ids[j], color_count++));
            if (color_count >= max_color_count) 
              color_count = 0;
            color_index = ids_colors.find(ids[j])->second;
          } else {
            color_index = it->second;
          }
          const cv::Scalar& color = colors[color_index];

          // Overlay trajectories for display
          auto trajectories_it = trajectories.find(ids[j]);
          if (trajectories_it != trajectories.end()) {
            const auto& cb = trajectories_it->second;
            auto prev_it = cb.begin();
            for (auto it = cb.begin(); it != cb.end(); ++it) {
              if (it != cb.begin()) {
                if ((*it) && (*prev_it)) {
                  if (display) {
                    cv::line(image_display, **prev_it, **it, color, 5);
                  }
                }
              }
              prev_it = it;
            }
          }

          // Overlay trajectories for map_display
          auto tracks_it = map_tracks.find(ids[j]);
          if (tracks_it != map_tracks.end()) {
            const auto& cb = tracks_it->second;

            auto prev_it = cb.begin();
            size_t n = 0;
            for (auto it = cb.begin(); it != cb.end(); ++it) {
              if (it != cb.begin()) {
                if ((*it) && (*prev_it)) {
                  if (map_mode == 1) {
                    cv::line(map_display, MapToScreen(**prev_it, map_display), 
                             MapToScreen(**it, map_display), color, 3);
                    if (n == cb.size()-1) {
                      OverlayIcon(map_display, MapToScreen(**it, map_display)-cv::Point(20,15),
                                  icons[color_index], color, cv::Size(40,30));
                    }
                  } else {
                    if (n >= cb.size()-1-(step_size+2)) {
                      OverlayFootstep(map_display, MapToScreen(**prev_it, map_display), 
                                      MapToScreen(**it, map_display), color, step_size, n, footstep_img);
                    }
                    if (n == cb.size()-1) {
                      OverlayIcon(map_display, MapToScreen(**it, map_display)+cv::Point(24,0),
                                  icons[color_index], color, cv::Size(40,30));
                    }
                  }
                }
              }
              prev_it = it;
              n++;
            }
          }

          if (display) {
            // Draw bboxes
            cv::Point top_left_pt(bboxes[j].px, bboxes[j].py);
            cv::Point bottom_right_pt(bboxes[j].px + bboxes[j].width,
                                      bboxes[j].py + bboxes[j].height);
            cv::Point bottom_left_pt(bboxes[j].px,
                                     bboxes[j].py + bboxes[j].height);
            cv::rectangle(image_display, top_left_pt, bottom_right_pt, color, 4);
            cv::Size icon_size(80, 60);
            OverlayIcon(image_display, bottom_left_pt - cv::Point(0, icon_size.height), 
                        icons[color_index], color, icon_size);
          }
        }

        if (display) {
          // Overlay FPS
          auto font_scale = 2.0;
          cv::Point label_point(25, 50);
          cv::Scalar label_color(200, 200, 250);
          cv::Scalar outline_color(0, 0, 0);
          cv::putText(image_display, label_string, label_point,
                      CV_FONT_HERSHEY_PLAIN, font_scale, outline_color, 8, CV_AA);
          cv::putText(image_display, label_string, label_point,
                      CV_FONT_HERSHEY_PLAIN, font_scale, label_color, 2, CV_AA);
          cv::imshow(camera_names[i], image_display);
        }

      }
    }

    if (display_map) {
      cv::imshow("Map", map_display);
    }
    if (display || display_map) {
      char key = cv::waitKey(10);
      if (key == 'q') break;
      else if (key == 'm') {
        if (map_mode == 0) map_mode = 1;
        else map_mode = 0;
      }
    }
  }

  LOG(INFO) << "Done";

  //////// Clean up

  CleanUp();
  cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
  // FIXME: Use more standard arg parse routine.
  // Set up glog
  gst_init(&argc, &argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  po::options_description desc("Multi-camera end to end video ingestion demo");
  desc.add_options()("help,h", "print the help message");
  desc.add_options()("display,d", "Enable display or not");
  desc.add_options()("display_map", "Display all trajectories on a global map");
  desc.add_options()("device", po::value<int>()->default_value(-1),
                     "which device to use, -1 for CPU, > 0 for GPU device");
  desc.add_options()("config_dir,C", po::value<std::string>(),
                     "The directory to find SAF's configurations");
  desc.add_options()("camera,c", po::value<std::string>()->required(),
                     "The name of the camera to use, if there are multiple "
                     "cameras to be used, separate with ,");
  desc.add_options()("detector_type", po::value<std::string>()->required(),
                     "The name of the detector type to run");
  desc.add_options()("detector_model,m", po::value<std::string>()->required(),
                     "The name of the detector model to run");
  desc.add_options()("detector_confidence_threshold",
                     po::value<float>()->default_value(0.5),
                     "detector confidence threshold");
  desc.add_options()("detector_idle_duration",
                     po::value<float>()->default_value(1.0),
                     "detector idle duration");
  desc.add_options()("detector_targets",
                     po::value<std::string>()->default_value(""),
                     "The name of the target to detect, separate with ,");
  desc.add_options()("face_min_size", po::value<int>()->default_value(40),
                     "Face min size for mtcnn");
  desc.add_options()("tracker_type",
                     po::value<std::string>()->default_value("dlib"),
                     "The name of the tracker type to run");
  desc.add_options()("extractor_type",
                     po::value<std::string>()->default_value(""),
                     "The name of the extractor type to run");
  desc.add_options()("extractor_model",
                     po::value<std::string>()->default_value(""),
                     "The name of the extractor model to run");
  desc.add_options()("matcher_type",
                     po::value<std::string>()->default_value(""),
                     "The name of the matcher type to run");
  desc.add_options()("matcher_model",
                     po::value<std::string>()->default_value(""),
                     "The name of the matcher model to run");
  desc.add_options()(
      "matcher_distance_threshold",
      po::value<float>()->default_value(std::numeric_limits<float>::max()),
      "matcher distance threshold");
  desc.add_options()("sender_endpoint",
                     po::value<std::string>()->default_value(""),
                     "The remote endpoint address");
  desc.add_options()("sender_package_type",
                     po::value<std::string>()->default_value("thumbnails"),
                     "The sender package type");
  desc.add_options()("frames", po::value<int>()->default_value(-1),
                     "Total running frames");
  desc.add_options()("map", po::value<std::string>()->default_value(""),
                     "The global map to display trajectories");
  desc.add_options()("icon_folder", po::value<std::string>()->default_value(""),
                     "The folder of displayed icons");
  desc.add_options()("footstep_img", po::value<std::string>()->default_value(""),
                     "The image of displayed foot step");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << e.what() << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  ///////// Parse arguments
  if (vm.count("config_dir")) {
    Context::GetContext().SetConfigDir(vm["config_dir"].as<std::string>());
  }
  // Init SAF context, this must be called before using SAF.
  Context::GetContext().Init();
  int device_number = vm["device"].as<int>();
  Context::GetContext().SetInt(DEVICE_NUMBER, device_number);

  bool display = vm.count("display") != 0;
  bool display_map = vm.count("display_map") != 0;
  auto camera_names = SplitString(vm["camera"].as<std::string>(), ",");
  auto detector_type = vm["detector_type"].as<std::string>();
  auto detector_model = vm["detector_model"].as<std::string>();
  auto detector_confidence_threshold =
      vm["detector_confidence_threshold"].as<float>();
  auto detector_idle_duration = vm["detector_idle_duration"].as<float>();
  auto detector_targets = vm["detector_targets"].as<std::string>();
  auto face_min_size = vm["face_min_size"].as<int>();
  auto tracker_type = vm["tracker_type"].as<std::string>();
  auto extractor_type = vm["extractor_type"].as<std::string>();
  auto extractor_model = vm["extractor_model"].as<std::string>();
  auto matcher_type = vm["matcher_type"].as<std::string>();
  auto matcher_model = vm["matcher_model"].as<std::string>();
  auto matcher_distance_threshold =
      vm["matcher_distance_threshold"].as<float>();
  auto sender_endpoint = vm["sender_endpoint"].as<std::string>();
  auto sender_package_type = vm["sender_package_type"].as<std::string>();
  auto frames = vm["frames"].as<int>();
  auto map = vm["map"].as<std::string>();
  auto icon_folder = vm["icon_folder"].as<std::string>();
  Run(camera_names, detector_type, detector_model, display, display_map,
      detector_confidence_threshold, detector_idle_duration, detector_targets,
      face_min_size, tracker_type, extractor_type, extractor_model,
      matcher_type, matcher_distance_threshold, matcher_model, sender_endpoint,
      sender_package_type, frames, map, icon_folder, footstep_img);

  return 0;
}
