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

// Multi-target matcher

#include "operator/matchers/object_matcher.h"

#include "camera/camera.h"
#include "common/context.h"
#include "model/model_manager.h"
#include "operator/matchers/euclidean_matcher.h"
#include "operator/matchers/xqda_matcher.h"
#include "utils/string_utils.h"
#include "utils/time_utils.h"

using std::cout;
using std::endl;

ObjectMatcher::ObjectMatcher(const std::string& type, 
                             const std::string& summarization_mode,
                             size_t batch_size,
                             float distance_threshold,
                             const ModelDesc& model_desc)
    : Operator(OPERATOR_TYPE_OBJECT_MATCHER, {}, {}),
      type_(type),
      summarization_mode_(summarization_mode),
      batch_size_(batch_size),
      distance_threshold_(distance_threshold),
      model_desc_(model_desc),
      reid_is_running(true),
      reid_thread_run_(true) {
  for (size_t i = 0; i < batch_size; i++) {
    sources_.insert({GetSourceName(i), nullptr});
    sinks_.insert({GetSinkName(i), std::shared_ptr<Stream>(new Stream)});
    camera_track_buffers_.push_back(std::map<std::string, TrackInfoWeakPtr>());
  }
}

std::shared_ptr<ObjectMatcher> ObjectMatcher::Create(
    const FactoryParamsType& params) {
  auto type = params.at("type");
  auto summarization_mode = params.at("summarization_mode");
  auto batch_size = StringToSizet(params.at("batch_size"));
  auto distance_threshold = atof(params.at("distance_threshold").c_str());

  ModelManager& model_manager = ModelManager::GetInstance();
  auto model_name = params.at("model");
  CHECK(model_manager.HasModel(model_name));
  auto model_desc = model_manager.GetModelDesc(model_name);

  return std::make_shared<ObjectMatcher>(type, summarization_mode, batch_size, 
                                         distance_threshold, model_desc);
}

bool ObjectMatcher::Init() {
  bool result = false;
  if (type_ == "euclidean") {
    matcher_ = std::make_unique<EuclideanMatcher>();
    result = matcher_->Init();
  } else if (type_ == "xqda") {
    matcher_ = std::make_unique<XQDAMatcher>(model_desc_);
    result = matcher_->Init();
  } else {
    LOG(FATAL) << "Matcher type " << type_ << " not supported.";
  }

  reid_thread_ = std::thread(&ObjectMatcher::ReIDThread, this);
  return result;
}

bool ObjectMatcher::OnStop() {
  if (reid_thread_.joinable()) {
    reid_thread_run_ = false;
    reid_cv_.notify_all();
    reid_thread_.join();
  }
  return true;
}

void ObjectMatcher::Process() {
  Timer timer;
  timer.Start();

  for (size_t i = 0; i < batch_size_; i++) {
    auto frame = GetFrame(GetSourceName(i));
    if (!frame) continue;

    auto camera_name = frame->GetValue<std::string>("camera_name");
    auto ids = frame->GetValue<std::vector<std::string>>("ids");
    auto timestamp = GetTimeSinceEpochMillis();
    auto tags = frame->GetValue<std::vector<std::string>>("tags");
    auto features =
        frame->GetValue<std::vector<std::vector<double>>>("features");
    CHECK(ids.size() == tags.size());
    CHECK(ids.size() == features.size());

    // Search track_buffer_ for id and return mapped_id
    std::vector<std::string> mapped_ids(ids.size(), std::string());
    size_t mapped_count = 0;
    for (size_t j = 0; j < ids.size(); j++) {
      const auto& id = ids[j];
      auto it = track_buffer_.find(id);
      if (it != track_buffer_.end()) {
        mapped_ids[j] = it->second->GetIDMapped();
        mapped_count++;
      } else {
        mapped_ids[j] = id;
        mapped_count++;
      }
    }

    if (!reid_is_running) {
      {
        std::lock_guard<std::mutex> guard(reid_lock_);
        reid_data = std::make_unique<ReIDData>();
        reid_data->source_idx = i;
        reid_data->camera_name = camera_name;
        reid_data->ids = ids;
        reid_data->timestamp = timestamp;
        reid_data->tags = tags;
        reid_data->features = features;
      }
      reid_cv_.notify_all();
    }

    CHECK(mapped_count == ids.size());
    frame->SetValue("ids", mapped_ids);
    PushFrame(GetSinkName(i), std::move(frame));
  }

  LOG(INFO) << "ObjectMatcher took " << timer.ElapsedMSec() << " ms";
}

std::string short_id(std::string s) {
  cout << "OOO: " << s << endl;
  if (s.length() <= 2 || s.empty()) {
    cout << "WTF: " << s << endl;
    return s;
  } else {
    return s.substr(s.length()-2);
  }
}

void ObjectMatcher::ReIDThread() {
  while (reid_thread_run_) {
    // Delete outdated tracks (inactive for 3600 sec.) in track_database_
    auto now = GetTimeSinceEpochMillis();
    for (auto it = track_database_.begin(); it != track_database_.end();) {
      if (abs((long)(now - it->second->GetLastTimestamp())) > (3600 * 1000)) {
        track_database_.erase(it++);
      } else {
        it++;
      }
    }

    reid_data.reset();
    reid_is_running = false;
    std::unique_lock<std::mutex> lk(reid_lock_);
    reid_cv_.wait(lk, [this] {
      // Stop waiting when reid_data available
      return reid_data || !reid_thread_run_;
    });

    if (!reid_data) continue;

    reid_is_running = true;
    auto source_idx = reid_data->source_idx;
    auto camera_name = reid_data->camera_name;
    auto ids = reid_data->ids;
    auto timestamp = reid_data->timestamp;
    auto tags = reid_data->tags;
    auto features = reid_data->features;

    std::vector<std::string> mapped_ids(ids.size(), std::string()); // key:id_idx, value:uuid
    size_t mapped_count = 0;

    for (auto& m : track_buffer_) {
      m.second->SetActive(false);
      m.second->SetMapped(false);
    }

    // Phase 1: Update features in track_buffer_ for known id
    // Phase 2: Create new track in track_buffer_ for unmapped id
    for (size_t j = 0; j < ids.size(); j++) {
      const auto& id = ids[j];
      const auto& tag = tags[j];
      const auto& feature = features[j];
      auto it = track_buffer_.find(id);
      if (it != track_buffer_.end()) {
        mapped_ids[j] = id;
        mapped_count++;
        it->second->UpdateFeature(source_idx, timestamp, feature);
        it->second->SetActive(true);
      } else {
        mapped_ids[j] = id;
        mapped_count++;

        // Create new track into track_buffer
        auto new_track_info = std::make_shared<TrackInfo>(
            camera_name, id, tag, summarization_mode_);
        track_buffer_[id] = new_track_info;
        new_track_info->UpdateFeature(source_idx, timestamp, feature);
        new_track_info->SetActive(true);
      }
    }
    CHECK(mapped_count == ids.size());

    // Phase 3: Move inactive tracks in track_buffer_ to track_database_
    for (auto it = track_buffer_.begin(); it != track_buffer_.end();) {
      if (it->second->GetActive() == false) {
        auto mapped_id = it->second->GetIDMapped();
        auto feature = it->second->GetFeature();
        auto matched_history = track_database_.find(mapped_id);
        if (matched_history != track_database_.end()) {
          matched_history->second->UpdateFeature(source_idx, timestamp, feature);
          matched_history->second->SetIDMapped(mapped_id);
        } else {
          track_database_[it->second->GetID()] = it->second;
        }
        track_buffer_.erase(it++);
      } else {
        it++;
      }
    }


    // Phase 4: Match tracks in track_buffer_ with tracks in track_database_
    auto mapping = GetSortedMapping(ids, mapped_ids, features);
    for (decltype(mapping.size()) j = 0; j < mapping.size(); j++) {
      auto buffer_track_info = std::get<0>(mapping[j]);
      auto database_track_info = std::get<1>(mapping[j]);
      if (buffer_track_info->GetMapped() == false) {
        buffer_track_info->SetIDMapped(database_track_info->GetID());
        buffer_track_info->SetMapped(true);
      }
    }
    for (auto m: track_buffer_) {
      if (m.second->GetMapped() == false) {
        m.second->SetIDMapped(m.second->GetID());
        m.second->SetMapped(true);
      }
    }

    //==========================
    /*cout << ">>>>>>>>>>>>>>" << endl;
    cout << "Input frame:" << endl;
    for (size_t j=0; j < ids.size(); j++) {
        cout << short_id(ids[j]) << endl;
    }
    cout << "Track_buffer_:" << endl;
    for (auto m: track_buffer_) {
        cout << "ID: " << short_id(m.second->GetID()) << ", Mapped ID: " << short_id(m.second->GetIDMapped()) << endl;
    }
    cout << "Track_database_:" << endl;
    for (auto m: track_database_) {
        cout << "ID: " << short_id(m.second->GetID()) << ", Mapped ID: " << short_id(m.second->GetIDMapped()) << endl;
    }
    cout << "<<<<<<<<<<<<<<<" << endl; */
    //==========================

  }
}

std::vector<std::tuple<TrackInfoPtr, TrackInfoPtr, double>>
ObjectMatcher::GetSortedMapping(
    const std::vector<std::string>& ids,
    const std::vector<std::string>& mapped_ids,
    const std::vector<std::vector<double>>& features) {
  //std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
  std::vector<std::tuple<TrackInfoPtr, TrackInfoPtr, double>> mapping;
  CHECK(ids.size() == features.size());
  CHECK(mapped_ids.size() == features.size());

  for (auto buffer_it: track_buffer_) {
    auto buffer_track_info = buffer_it.second;
    //std::cout << ">>>>> Trying to match " << short_id(buffer_track_info->GetID()) << " to previous id" << std::endl;
    for (auto database_it: track_database_) {
      auto database_track_info = database_it.second;
      auto dist = matcher_->Match(buffer_track_info->GetFeature(), database_track_info->GetFeature());
      //std::cout << "================= " << short_id(buffer_track_info->GetID()) << "<->" << short_id(database_track_info->GetID()) << " =================" << std::endl;
      //std::cout << dist << std::endl;
      //std::cout << "===========================================" << std::endl;
      mapping.push_back(std::make_tuple(buffer_track_info, database_track_info, dist));
    }
  }

  std::sort(mapping.begin(), mapping.end(),
            [](std::tuple<TrackInfoPtr, TrackInfoPtr, double>& t1,
               std::tuple<TrackInfoPtr, TrackInfoPtr, double>& t2) {
              auto dist1 = std::get<2>(t1);
              auto dist2 = std::get<2>(t2);
              return dist1 < dist2;
            });

  boost::optional<size_t> found;
  for (size_t i = 0; i < mapping.size(); i++) {
    auto dist = std::get<2>(mapping[i]);
    if (dist < distance_threshold_) {
      found = i;
    } else {
      break;
    }
  }

  return found ? std::vector<std::tuple<TrackInfoPtr, TrackInfoPtr, double>>(
                     mapping.begin(), mapping.begin() + (*found) + 1)
               : std::vector<std::tuple<TrackInfoPtr, TrackInfoPtr, double>>();
}
