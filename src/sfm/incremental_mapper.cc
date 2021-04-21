// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "sfm/incremental_mapper.h"

#include <fstream>
#include <cmath>
#include <queue>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>


#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/pose.h"
#include "util/bitmap.h"
#include "util/misc.h"
using namespace std;

namespace colmap {
namespace {

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
  std::sort(image_ranks.begin(), image_ranks.end(),
            [](const std::pair<image_t, float>& image1,
               const std::pair<image_t, float>& image2) {
              return image1.second > image2.second;
            });

  sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
  for (const auto& image : image_ranks) {
    sorted_images_ids->push_back(image.first);
  }

  image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D());
}

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D()) /
         static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) {
  return static_cast<float>(image.Point3DVisibilityScore());
}

}  // namespace

bool IncrementalMapper::Options::Check() const {
  CHECK_OPTION_GT(init_min_num_inliers, 0);
  CHECK_OPTION_GT(init_max_error, 0.0);
  CHECK_OPTION_GE(init_max_forward_motion, 0.0);
  CHECK_OPTION_LE(init_max_forward_motion, 1.0);
  CHECK_OPTION_GE(init_min_tri_angle, 0.0);
  CHECK_OPTION_GE(init_max_reg_trials, 1);
  CHECK_OPTION_GT(abs_pose_max_error, 0.0);
  CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
  CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(local_ba_num_images, 2);
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GE(max_reg_trials, 1);
  return true;
}

IncrementalMapper::IncrementalMapper(/*const*/ DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId) {}

void IncrementalMapper::BeginReconstruction(Reconstruction* reconstruction) {
  CHECK(reconstruction_ == nullptr);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  //HERE
  reconstruction_->SetUp(&database_cache_->SceneGraph());
  //reconstruction_->SetUp(&database_cache_->SG());
  triangulator_.reset(new IncrementalTriangulator(
      //&database_cache_->SceneGraph(), reconstruction));
      &database_cache_->SG(), reconstruction));
  sg = &(database_cache_->SG());

  num_shared_reg_images_ = 0;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  prev_init_image_pair_id_ = kInvalidImagePairId;
  prev_init_two_view_geometry_ = TwoViewGeometry();

  refined_cameras_.clear();
  filtered_images_.clear();
  num_reg_trials_.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
  CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  reconstruction_->TearDown();
  reconstruction_ = nullptr;
  triangulator_.reset();
}

//initial image pair
//improve so, that it allows only pairs of images from the first set, which will be provided as hashset of image ids (this will have to be found in the higher function)
bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             image_t* image_id1,
                                             image_t* image_id2) {
  CHECK(options.Check());

  std::vector<image_t> image_ids1;
  if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
    // Only *image_id1 provided.
    if (!database_cache_->ExistsImage(*image_id1)) {
      return false;
    }
    image_ids1.push_back(*image_id1);
  } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
    // Only *image_id2 provided.
    if (!database_cache_->ExistsImage(*image_id2)) {
      return false;
    }
    image_ids1.push_back(*image_id2);
  } else {
    // No initial seed image provided.
    image_ids1 = FindFirstInitialImage(options);
  }

  // Try to find good initial pair.
  for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
    *image_id1 = image_ids1[i1];

    const std::vector<image_t> image_ids2 =
        FindSecondInitialImage(options, *image_id1);

    for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
      *image_id2 = image_ids2[i2];

      const image_pair_t pair_id =
          Database::ImagePairToPairId(*image_id1, *image_id2);

      // Try every pair only once.
      if (init_image_pairs_.count(pair_id) > 0) {
        continue;
      }

      init_image_pairs_.insert(pair_id);

      if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
        return true;
      }
    }
  }

  // No suitable pair found in entire dataset.
  *image_id1 = kInvalidImageId;
  *image_id2 = kInvalidImageId;

  return false;
}

bool IncrementalMapper::FindInitFirstSet(const Options& options, image_t* image_id1, image_t* image_id2, std::unordered_set<image_t> first_set)
{
	CHECK(options.Check());
	std::vector<image_t> image_ids1;

	//this part is same as in the original function
	if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId)
	{
		// Only *image_id1 provided.
		if (!database_cache_->ExistsImage(*image_id1))
			return false;
		image_ids1.push_back(*image_id1);
	}
	else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId)
	{
		// Only *image_id2 provided.
		if (!database_cache_->ExistsImage(*image_id2))
			return false;
		image_ids1.push_back(*image_id2);
	}
	else
	{
		// No initial seed image provided.
		image_ids1 = FindFirstInitialImage(options);
  	}

	// Try to find good initial pair.
	for (size_t i1 = 0; i1 < image_ids1.size(); ++i1)
	{
		std::cout << i1 << " " << image_ids1.size() << "\n";
		if(!first_set.count(image_ids1[i1])) continue;
		std::cout << "K\n";
		//std::cout << i1 << " " << image_ids1.size() << "\n";
    	*image_id1 = image_ids1[i1];

		const std::vector<image_t> image_ids2 = FindSecondInitialImage(options, *image_id1);

		for (size_t i2 = 0; i2 < image_ids2.size(); ++i2)
		{
			std::cout << i1 << " " << image_ids1.size() << " " << i2 << " " << image_ids2.size() << "\n";
			if(!first_set.count(image_ids2[i2])) continue;
			*image_id2 = image_ids2[i2];
			const image_pair_t pair_id = Database::ImagePairToPairId(*image_id1, *image_id2);
			// Try every pair only once.
			if (init_image_pairs_.count(pair_id) > 0)
				continue;
			init_image_pairs_.insert(pair_id);

			if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2))
			{
        		return true;
      		}
		}
	}

	/*for (size_t i1 = 0; i1 < image_ids1.size(); ++i1)
	{
		//if(!first_set.count(image_ids1[i1])) continue;
    	*image_id1 = image_ids1[i1];

		const std::vector<image_t> image_ids2 = FindSecondInitialImage(options, *image_id1);

		for (size_t i2 = 0; i2 < image_ids2.size(); ++i2)
		{
			if(!first_set.count(image_ids2[i2])) continue;
			*image_id2 = image_ids2[i2];
			const image_pair_t pair_id = Database::ImagePairToPairId(*image_id1, *image_id2);
			// Try every pair only once.
			if (init_image_pairs_.count(pair_id) > 0)
				continue;
			init_image_pairs_.insert(pair_id);

			if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2))
			{
        		return true;
      		}
		}
	}*/

	// No suitable pair found in entire dataset.
	*image_id1 = kInvalidImageId;
	*image_id2 = kInvalidImageId;
	return false;
}

std::vector<image_t> IncrementalMapper::FindNextImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }

    // Only consider images with a sufficient number of visible points.
    if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

std::vector<image_t> IncrementalMapper::FindNextImagesSecondSet(const Options& options, std::unordered_set<image_t> first_set)
{
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }
    if(image.second.IsSeqRegistered()) continue;

	//get name of the image and compare it with legal names
	
    if(first_set.count(image.first)) continue;

    std::cout << "Seqreg " << image.first << "\n";
    std::cout << image.second.NumVisiblePoints3D() << " " << options.abs_pose_min_num_inliers << "\n"; 

    // Only consider images with a sufficient number of visible points.
    /*if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }*/

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

std::vector<image_t> IncrementalMapper::FindNextImagesFirstSet(const Options& options, std::unordered_set<image_t> first_set)
{
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }

	//get name of the image and compare it with legal names
	
    if(!first_set.count(image.first)) continue;

    // Only consider images with a sufficient number of visible points.
    if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

bool IncrementalMapper::RegisterInitialImagePair(const Options& options,
                                                 const image_t image_id1,
                                                 const image_t image_id2) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_EQ(reconstruction_->NumRegImages(), 0);

  CHECK(options.Check());

  init_num_reg_trials_[image_id1] += 1;
  init_num_reg_trials_[image_id2] += 1;
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  Image& image1 = reconstruction_->Image(image_id1);
  const Camera& camera1 = reconstruction_->Camera(image1.CameraId());

  Image& image2 = reconstruction_->Image(image_id2);
  const Camera& camera2 = reconstruction_->Camera(image2.CameraId());

  //////////////////////////////////////////////////////////////////////////////
  // Estimate two-view geometry
  //////////////////////////////////////////////////////////////////////////////

  if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
    return false;
  }

  image1.Qvec() = ComposeIdentityQuaternion();
  image1.Tvec() = Eigen::Vector3d(0, 0, 0);
  image2.Qvec() = prev_init_two_view_geometry_.qvec;
  image2.Tvec() = prev_init_two_view_geometry_.tvec;

  const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
  const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
  const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
  const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

  //////////////////////////////////////////////////////////////////////////////
  // Update Reconstruction
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  const SceneGraph& scene_graph = database_cache_->SceneGraph();
  const std::vector<std::pair<point2D_t, point2D_t>>& corrs =
      scene_graph.FindCorrespondencesBetweenImages(image_id1, image_id2);

  const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);

  // Add 3D point tracks.
  Track track;
  track.Reserve(2);
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;
  for (size_t i = 0; i < corrs.size(); ++i) {
    const point2D_t point2D_idx1 = corrs[i].first;
    const point2D_t point2D_idx2 = corrs[i].second;
    const Eigen::Vector2d point1_N =
        camera1.ImageToWorld(image1.Point2D(point2D_idx1).XY());
    const Eigen::Vector2d point2_N =
        camera2.ImageToWorld(image2.Point2D(point2D_idx2).XY());
    const Eigen::Vector3d& xyz =
        TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);
    const double tri_angle =
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
    if (tri_angle >= min_tri_angle_rad &&
        HasPointPositiveDepth(proj_matrix1, xyz) &&
        HasPointPositiveDepth(proj_matrix2, xyz)) {
      track.Element(0).point2D_idx = point2D_idx1;
      track.Element(1).point2D_idx = point2D_idx2;
      reconstruction_->AddPoint3D(xyz, track);
    }
  }

  return true;
}

bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);
  Camera& camera = reconstruction_->Camera(image.CameraId());

  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  num_reg_trials_[image_id] += 1;

  // Check if enough 2D-3D correspondences.
  if (image.NumVisiblePoints3D() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Search for 2D-3D correspondences
  //////////////////////////////////////////////////////////////////////////////

  const int kCorrTransitivity = 1;

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;

  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    const SceneGraph& scene_graph = database_cache_->SceneGraph();
    const std::vector<SceneGraph::Correspondence> corrs =
        scene_graph.FindTransitiveCorrespondences(image_id, point2D_idx,
                                                  kCorrTransitivity);

    std::unordered_set<point3D_t> point3D_ids;

    for (const auto corr : corrs) {
      const Image& corr_image = reconstruction_->Image(corr.image_id);
      if (!corr_image.IsRegistered()) {
        continue;
      }

      const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // Avoid duplicate correspondences.
      if (point3D_ids.count(corr_point2D.Point3DId()) > 0) {
        continue;
      }

      const Camera& corr_camera =
          reconstruction_->Camera(corr_image.CameraId());

      // Avoid correspondences to images with bogus camera parameters.
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.Point3DId());

      tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
      point3D_ids.insert(corr_point2D.Point3DId());
      tri_points2D.push_back(point2D.XY());
      tri_points3D.push_back(point3D.XYZ());
    }
  }

  // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
  // can only differ, when there are images with bogus camera parameters, and
  // hence we skip some of the 2D-3D correspondences.
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D estimation
  //////////////////////////////////////////////////////////////////////////////

  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera
  // parameters)

  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  // Use high confidence to avoid preemptive termination of P3P RANSAC
  // - too early termination may lead to bad registration.
  abs_pose_options.ransac_options.min_num_trials = 30;
  abs_pose_options.ransac_options.confidence = 0.9999;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (refined_cameras_.count(image.CameraId()) > 0) {
    // Camera already refined from another image with the same camera.
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-refine.
      refined_cameras_.erase(image.CameraId());
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
    }
  } else {
    // Camera not refined before.
    abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_refinement_options.refine_focal_length = true;
    abs_pose_refinement_options.refine_extra_params = true;
  }

  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }

  size_t num_inliers;
  std::vector<char> inlier_mask;

  if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                            &image.Qvec(), &image.Tvec(), &camera, &num_inliers,
                            &inlier_mask)) {
    return false;
  }

  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                          tri_points2D, tri_points3D, &image.Qvec(),
                          &image.Tvec(), &camera)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const point2D_t point2D_idx = tri_corrs[i].first;
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const point3D_t point3D_id = tri_corrs[i].second;
        const TrackElement track_el(image_id, point2D_idx);
        reconstruction_->AddObservation(point3D_id, track_el);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Update data
  //////////////////////////////////////////////////////////////////////////////

  refined_cameras_.insert(image.CameraId());

  return true;
}

int IncrementalMapper::SeqRegisterImage(const Options& options, const image_t image_id)
{
	CHECK_NOTNULL(reconstruction_);
	CHECK_GE(reconstruction_->NumRegImages(), 2);

	std::cout << "SQ1\n";

	CHECK(options.Check());

	Image& image = reconstruction_->Image(image_id);
	Camera& camera = reconstruction_->Camera(image.CameraId());

	CHECK(!image.IsSeqRegistered()) << "Image cannot be sequentially registered multiple times";

	num_reg_trials_[image_id] += 1;

	// Check if enough 2D-3D correspondences.
	if (image.NumVisiblePoints3D() < static_cast<size_t>(options.abs_pose_min_num_inliers))
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Search for 2D-3D correspondences
	//////////////////////////////////////////////////////////////////////////////

	const int kCorrTransitivity = 1;

	std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
	std::vector<Eigen::Vector2d> tri_points2D;
	std::vector<Eigen::Vector3d> tri_points3D;

	for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
	{
		const Point2D& point2D = image.Point2D(point2D_idx);
		const SceneGraph& scene_graph = database_cache_->SceneGraph();
		const std::vector<SceneGraph::Correspondence> corrs = scene_graph.FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);

		std::unordered_set<point3D_t> point3D_ids;

		for (const auto corr : corrs)
		{
			const Image& corr_image = reconstruction_->Image(corr.image_id);
			if (!corr_image.IsRegistered())
				continue;

			const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
			if (!corr_point2D.HasPoint3D())
				continue;

			// Avoid duplicate correspondences.
			if (point3D_ids.count(corr_point2D.Point3DId()) > 0)
				continue;

			const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

			// Avoid correspondences to images with bogus camera parameters.
			if (corr_camera.HasBogusParams(options.min_focal_length_ratio, options.max_focal_length_ratio, options.max_extra_param))
				continue;

			const Point3D& point3D = reconstruction_->Point3D(corr_point2D.Point3DId());

			tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
			point3D_ids.insert(corr_point2D.Point3DId());
			tri_points2D.push_back(point2D.XY());
			tri_points3D.push_back(point3D.XYZ());
		}
	}

	// The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
	// can only differ, when there are images with bogus camera parameters, and
	// hence we skip some of the 2D-3D correspondences.
	if (tri_points2D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers))
		return 0;

	//////////////////////////////////////////////////////////////////////////////
  	// 2D-3D estimation
  	//////////////////////////////////////////////////////////////////////////////

  //EXTRACT OPTIONS AND SET THEM
  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera
  // parameters)

  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  // Use high confidence to avoid preemptive termination of P3P RANSAC
  // - too early termination may lead to bad registration.
  abs_pose_options.ransac_options.min_num_trials = 30;
  abs_pose_options.ransac_options.confidence = 0.9999;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (refined_cameras_.count(image.CameraId()) > 0) {
    // Camera already refined from another image with the same camera.
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-refine.
      refined_cameras_.erase(image.CameraId());
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
    }
  } else {
    // Camera not refined before.
    abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_refinement_options.refine_focal_length = true;
    abs_pose_refinement_options.refine_extra_params = true;
  }

  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }

  std::cout << "SQ2\n";


	//SEQUENTIAL PNP IS APPLIED HERE
	int ret = 0;
	std::vector<std::pair<point2D_t, point3D_t>> remaining_tri_corrs = tri_corrs;
	std::vector<Eigen::Vector2d> remaining_tri_points2D = tri_points2D;
	std::vector<Eigen::Vector3d> remaining_tri_points3D = tri_points3D;
	//image.seq_
	while(true)
	{
		std::cout << "A\n";
		size_t num_inliers;
  		std::vector<char> inlier_mask;
  		Eigen::Vector4d qvec;
		Eigen::Vector3d tvec;
  		if (!EstimateAbsolutePose(abs_pose_options, remaining_tri_points2D, remaining_tri_points3D,
        		&qvec, &tvec, &camera, &num_inliers, &inlier_mask))
					break;
		std::cout << "Pose estimated\n";
		if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers))
			break;
		if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, remaining_tri_points2D,
				remaining_tri_points3D, &qvec, &tvec, &camera))
					break;

		//std::cout << "QV: " << qvec << " TV: " << tvec << "\n";
		//add tentative pose to the image
		image.AddPose(qvec, tvec);
		image.seq_inliers.push_back(num_inliers);
		std::cout << "Number of inliers: " << num_inliers << "\n";
		
		//add inliers to the image
		//set outliers as remaining corrs
		std::vector<std::pair<point2D_t, point3D_t>> inlier_tri_corrs;
		std::vector<std::pair<point2D_t, point3D_t>> next_tri_corrs;
		std::vector<Eigen::Vector2d> next_tri_points2D;
		std::vector<Eigen::Vector3d> next_tri_points3D;
		
		for (size_t i = 0; i < inlier_mask.size(); ++i)
		{
			if (inlier_mask[i])
			{
				inlier_tri_corrs.push_back(remaining_tri_corrs[i]);
				/*const point2D_t point2D_idx = tri_corrs[i].first;
      			const Point2D& point2D = image.Point2D(point2D_idx);
      			if (!point2D.HasPoint3D())
      			{
					const point3D_t point3D_id = tri_corrs[i].second;
					const TrackElement track_el(image_id, point2D_idx);
					reconstruction_->AddObservation(point3D_id, track_el);
				}*/
			}
			else
			{
				next_tri_corrs.push_back(remaining_tri_corrs[i]);
				next_tri_points2D.push_back(remaining_tri_points2D[i]);
				next_tri_points3D.push_back(remaining_tri_points3D[i]);
			}
		}
		remaining_tri_corrs = next_tri_corrs;
		remaining_tri_points2D = next_tri_points2D;
		remaining_tri_points3D = next_tri_points3D;
		image.AddInlierCorrs(inlier_tri_corrs);

		ret++;
		//if(ret >= 2) break;
		std::cout << "B\n";
	}

	std::cout << "SQ3\n";

	if(ret==2)
		image.SetSeqRegistered(true);

	refined_cameras_.insert(image.CameraId());
	
	return ret;
}

//uses a determined already used camera instead of a new one
int IncrementalMapper::SeqRegisterImage2(const Options& options, const image_t image_id, const camera_t cam)
{
	CHECK_NOTNULL(reconstruction_);
	CHECK_GE(reconstruction_->NumRegImages(), 2);

	CHECK(options.Check());


	Image& image = reconstruction_->Image(image_id);
	image.SetCameraId(cam);
	Camera& camera = reconstruction_->Camera(image.CameraId());

	CHECK(!image.IsSeqRegistered()) << "Image cannot be sequentially registered multiple times";

	num_reg_trials_[image_id] += 1;

	// Check if enough 2D-3D correspondences.
	if (image.NumVisiblePoints3D() < static_cast<size_t>(options.abs_pose_min_num_inliers))
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Search for 2D-3D correspondences
	//////////////////////////////////////////////////////////////////////////////

	const int kCorrTransitivity = 1;

	std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
	std::vector<Eigen::Vector2d> tri_points2D;
	std::vector<Eigen::Vector3d> tri_points3D;

	for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
	{
		const Point2D& point2D = image.Point2D(point2D_idx);
		const SceneGraph& scene_graph = database_cache_->SceneGraph();
		const std::vector<SceneGraph::Correspondence> corrs = scene_graph.FindTransitiveCorrespondences(image_id, point2D_idx, kCorrTransitivity);

		std::unordered_set<point3D_t> point3D_ids;

		for (const auto corr : corrs)
		{
			const Image& corr_image = reconstruction_->Image(corr.image_id);
			if (!corr_image.IsRegistered())
				continue;

			const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
			if (!corr_point2D.HasPoint3D())
				continue;

			// Avoid duplicate correspondences.
			if (point3D_ids.count(corr_point2D.Point3DId()) > 0)
				continue;

			const Camera& corr_camera = reconstruction_->Camera(corr_image.CameraId());

			// Avoid correspondences to images with bogus camera parameters.
			if (corr_camera.HasBogusParams(options.min_focal_length_ratio, options.max_focal_length_ratio, options.max_extra_param))
				continue;

			const Point3D& point3D = reconstruction_->Point3D(corr_point2D.Point3DId());

			tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
			point3D_ids.insert(corr_point2D.Point3DId());
			tri_points2D.push_back(point2D.XY());
			tri_points3D.push_back(point3D.XYZ());
		}
	}

	// The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
	// can only differ, when there are images with bogus camera parameters, and
	// hence we skip some of the 2D-3D correspondences.
	if (tri_points2D.size() < static_cast<size_t>(options.abs_pose_min_num_inliers))
		return 0;

	//////////////////////////////////////////////////////////////////////////////
  	// 2D-3D estimation
  	//////////////////////////////////////////////////////////////////////////////

  //EXTRACT OPTIONS AND SET THEM
  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera
  // parameters)
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  // Use high confidence to avoid preemptive termination of P3P RANSAC
  // - too early termination may lead to bad registration.
  abs_pose_options.ransac_options.min_num_trials = 30;
  abs_pose_options.ransac_options.confidence = 0.9999;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (refined_cameras_.count(image.CameraId()) > 0) {
    // Camera already refined from another image with the same camera.
    /*if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-refine.
      refined_cameras_.erase(image.CameraId());
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
    }*/
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }
  else
  {
    // Camera not refined before.
    std::cout << image.CameraId() << "\n";
    refined_cameras_.insert(image.CameraId());
    //abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;
  }

  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }


	//SEQUENTIAL PNP IS APPLIED HERE
	int ret = 0;
	std::vector<std::pair<point2D_t, point3D_t>> remaining_tri_corrs = tri_corrs;
	std::vector<Eigen::Vector2d> remaining_tri_points2D = tri_points2D;
	std::vector<Eigen::Vector3d> remaining_tri_points3D = tri_points3D;
	//image.seq_
	while(true)
	{
		size_t num_inliers;
  		std::vector<char> inlier_mask;
  		Eigen::Vector4d qvec;
		Eigen::Vector3d tvec;
  		if (!EstimateAbsolutePose(abs_pose_options, remaining_tri_points2D, remaining_tri_points3D,
        		&qvec, &tvec, &camera, &num_inliers, &inlier_mask))
					break;
		if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers))
			break;
		if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask, remaining_tri_points2D,
				remaining_tri_points3D, &qvec, &tvec, &camera))
					break;

		//std::cout << "QV: " << qvec << " TV: " << tvec << "\n";
		//add tentative pose to the image
		image.AddPose(qvec, tvec);
		image.seq_inliers.push_back(num_inliers);
		std::cout << "Number of inliers: " << num_inliers << "\n";
		
		//add inliers to the image
		//set outliers as remaining corrs
		std::vector<std::pair<point2D_t, point3D_t>> inlier_tri_corrs;
		std::vector<std::pair<point2D_t, point3D_t>> next_tri_corrs;
		std::vector<Eigen::Vector2d> next_tri_points2D;
		std::vector<Eigen::Vector3d> next_tri_points3D;
		
		for (size_t i = 0; i < inlier_mask.size(); ++i)
		{
			if (inlier_mask[i])
			{
				inlier_tri_corrs.push_back(remaining_tri_corrs[i]);
				/*const point2D_t point2D_idx = tri_corrs[i].first;
      			const Point2D& point2D = image.Point2D(point2D_idx);
      			if (!point2D.HasPoint3D())
      			{
					const point3D_t point3D_id = tri_corrs[i].second;
					const TrackElement track_el(image_id, point2D_idx);
					reconstruction_->AddObservation(point3D_id, track_el);
				}*/
			}
			else
			{
				next_tri_corrs.push_back(remaining_tri_corrs[i]);
				next_tri_points2D.push_back(remaining_tri_points2D[i]);
				next_tri_points3D.push_back(remaining_tri_points3D[i]);
			}
		}
		remaining_tri_corrs = next_tri_corrs;
		remaining_tri_points2D = next_tri_points2D;
		remaining_tri_points3D = next_tri_points3D;
		image.AddInlierCorrs(inlier_tri_corrs);

		ret++;
		//if(ret >= 2) break;
	}

	if(ret==2)
		image.SetSeqRegistered(true);

	refined_cameras_.insert(image.CameraId());
	
	return ret;
}

bool IncrementalMapper::FinishRegistration(const Options& options, const image_t image_id, const std::vector<std::pair<point2D_t, point3D_t>> tri_corrs)
{
	std::cout << "Finish registration\n";
	std::cout << image_id << "\n";
	if(!tri_corrs.size()) return false;
	reconstruction_->RegisterImage(image_id);
	RegisterImageEvent(image_id);
	Image& image = reconstruction_->Image(image_id);
	std::cout << image.NumPoints2D() << "\n";

	//////////////////////////////////////////////////////////////////////////////
	// Continue tracks
	//////////////////////////////////////////////////////////////////////////////

	for(size_t i=0;i<tri_corrs.size();++i)
	{
		const point2D_t point2D_idx = tri_corrs[i].first;
		std::cout << point2D_idx << "\n";
		if(point2D_idx >= image.NumPoints2D())
			continue;
		const Point2D& point2D = image.Point2D(point2D_idx);
		std::cout << "K\n";
		if (!point2D.HasPoint3D())
		{
			const point3D_t point3D_id = tri_corrs[i].second;
			const TrackElement track_el(image_id, point2D_idx);
			reconstruction_->AddObservation(point3D_id, track_el);
		}
	}
	
	return true;
}

void IncrementalMapper::InitImage(const image_t image_id, const image_t original, std::vector<image_pair_t> old_pairs, std::vector<image_pair_t> new_pairs)
{
	sg->AddImage(image_id, reconstruction_->Image(image_id).NumPoints2D());
	sg->CopyCorrespondences(original, image_id, old_pairs, new_pairs);
}

size_t IncrementalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t IncrementalMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  LocalBundleAdjustmentReport report;

  // Find images that have most 3D points with given image in common.
  const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

  // Do the bundle adjustment only if there is any connected images.
  if (local_bundle.size() > 0) {
    BundleAdjustmentConfig ba_config;
    ba_config.AddImage(image_id);
    for (const image_t local_image_id : local_bundle) {
      ba_config.AddImage(local_image_id);
    }

    // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
    if (local_bundle.size() == 1) {
      ba_config.SetConstantPose(local_bundle[0]);
      ba_config.SetConstantTvec(image_id, {0});
    } else if (local_bundle.size() > 1) {
      ba_config.SetConstantPose(local_bundle[local_bundle.size() - 1]);
      ba_config.SetConstantTvec(local_bundle[local_bundle.size() - 2], {0});
    }

    // Make sure, we refine all new and short-track 3D points, no matter if
    // they are fully contained in the local image set or not. Do not include
    // long track 3D points as they are usually already very stable and adding
    // to them to bundle adjustment and track merging/completion would slow
    // down the local bundle adjustment significantly.
    std::unordered_set<point3D_t> variable_point3D_ids;
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction_->Point3D(point3D_id);
      const size_t kMaxTrackLength = 15;
      if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // Adjust the local bundle.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_);

    report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;

    // Merge refined tracks with other existing points.
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    // Complete tracks that may have failed to triangulate before refinement
    // of camera pose and calibration in bundle-adjustment. This may avoid
    // that some points are filtered and it helps for subsequent image
    // registrations.
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // Filter both the modified images and all changed 3D points to make sure
  // there are no outlier points in the model. This results in duplicate work as
  // many of the provided 3D points may also be contained in the adjusted
  // images, but the filtering is not a bottleneck at this point.
  std::unordered_set<image_t> filter_image_ids;
  filter_image_ids.insert(image_id);
  filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
  report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      filter_image_ids);
  report.num_filtered_observations += reconstruction_->FilterPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      point3D_ids);

  return report;
}

bool IncrementalMapper::AdjustGlobalBundle(
    const BundleAdjustmentOptions& ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }
  ba_config.SetConstantPose(reg_image_ids[0]);
  ba_config.SetConstantTvec(reg_image_ids[1], {0});

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  return true;
}

bool IncrementalMapper::AdjustParallelGlobalBundle(
    const BundleAdjustmentOptions& ba_options,
    const ParallelBundleAdjuster::Options& parallel_ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2)
      << "At least two images must be registered for global bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Run bundle adjustment.
  ParallelBundleAdjuster bundle_adjuster(parallel_ba_options, ba_options,
                                         ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  return true;
}

size_t IncrementalMapper::FilterImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids = reconstruction_->FilterImages(
      options.min_focal_length_ratio, options.max_focal_length_ratio,
      options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  return image_ids.size();
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());
  return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                            options.filter_min_tri_angle);
}

const Reconstruction& IncrementalMapper::GetReconstruction() const {
  CHECK_NOTNULL(reconstruction_);
  return *reconstruction_;
}

size_t IncrementalMapper::NumTotalRegImages() const {
  return num_total_reg_images_;
}

size_t IncrementalMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> IncrementalMapper::FindFirstInitialImage(
    const Options& options) const {
  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    image_t num_correspondences;
  };

  const size_t init_max_reg_trials =
      static_cast<size_t>(options.init_max_reg_trials);

  // Collect information of all not yet registered images with
  // correspondences.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto& image : reconstruction_->Images()) {
    // Only images with correspondences can be registered.
    if (image.second.NumCorrespondences() == 0) {
      continue;
    }

    // Only use images for initialization a maximum number of times.
    if (init_num_reg_trials_.count(image.first) &&
        init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
      continue;
    }

    // Only use images for initialization that are not registered in any
    // of the other reconstructions.
    if (num_registrations_.count(image.first) > 0 &&
        num_registrations_.at(image.first) > 0) {
      continue;
    }

    const class Camera& camera =
        reconstruction_->Camera(image.second.CameraId());
    ImageInfo image_info;
    image_info.image_id = image.first;
    image_info.prior_focal_length = camera.HasPriorFocalLength();
    image_info.num_correspondences = image.second.NumCorrespondences();
    image_infos.push_back(image_info);
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindSecondInitialImage(
    const Options& options, const image_t image_id1) const {
  const SceneGraph& scene_graph = database_cache_->SceneGraph();

  // Collect images that are connected to the first seed image and have
  // not been registered before in other reconstructions.
  const class Image& image1 = reconstruction_->Image(image_id1);
  std::unordered_map<image_t, point2D_t> num_correspondences;
  for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
       ++point2D_idx) {
    const std::vector<SceneGraph::Correspondence>& corrs =
        scene_graph.FindCorrespondences(image_id1, point2D_idx);
    for (const SceneGraph::Correspondence& corr : corrs) {
      if (num_registrations_.count(corr.image_id) == 0 ||
          num_registrations_.at(corr.image_id) == 0) {
        num_correspondences[corr.image_id] += 1;
      }
    }
  }

  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    point2D_t num_correspondences;
  };

  const size_t init_min_num_inliers =
      static_cast<size_t>(options.init_min_num_inliers);

  // Compose image information in a compact form for sorting.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto elem : num_correspondences) {
    if (elem.second >= init_min_num_inliers) {
      const class Image& image = reconstruction_->Image(elem.first);
      const class Camera& camera = reconstruction_->Camera(image.CameraId());
      ImageInfo image_info;
      image_info.image_id = elem.first;
      image_info.prior_focal_length = camera.HasPriorFocalLength();
      image_info.num_correspondences = elem.second;
      image_infos.push_back(image_info);
    }
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  CHECK(options.Check());

  const Image& image = reconstruction_->Image(image_id);
  CHECK(image.IsRegistered());

  // Extract all images that have at least one 3D point with the query image
  // in common, and simultaneously count the number of common 3D points.
  std::unordered_map<image_t, size_t> num_shared_observations;
  for (const Point2D& point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      const Point3D& point3D = reconstruction_->Point3D(point2D.Point3DId());
      for (const TrackElement& track_el : point3D.Track().Elements()) {
        if (track_el.image_id != image_id) {
          num_shared_observations[track_el.image_id] += 1;
        }
      }
    }
  }

  std::vector<std::pair<image_t, size_t>> local_bundle(
      num_shared_observations.begin(), num_shared_observations.end());

  // The local bundle is composed of the given image and its most connected
  // neighbor images, hence the subtraction of 1.
  const size_t num_images =
      static_cast<size_t>(options.local_ba_num_images - 1);
  const size_t num_eff_images = std::min(num_images, local_bundle.size());

  // Sort according to number of common 3D points.
  std::partial_sort(local_bundle.begin(), local_bundle.begin() + num_eff_images,
                    local_bundle.end(),
                    [](const std::pair<image_t, size_t>& image1,
                       const std::pair<image_t, size_t>& image2) {
                      return image1.second > image2.second;
                    });

  // Extract most connected images.
  std::vector<image_t> image_ids(num_eff_images);
  std::transform(local_bundle.begin(), local_bundle.begin() + num_eff_images,
                 image_ids.begin(),
                 [](const std::pair<image_t, size_t>& image_num_shared) {
                   return image_num_shared.first;
                 });

  return image_ids;
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    num_total_reg_images_ += 1;
  } else if (num_regs_for_image > 1) {
    num_shared_reg_images_ += 1;
  }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    num_total_reg_images_ -= 1;
  } else if (num_regs_for_image > 0) {
    num_shared_reg_images_ -= 1;
  }
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
    const Options& options, const image_t image_id1, const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);

  if (prev_init_image_pair_id_ == image_pair_id) {
    return true;
  }

  const Image& image1 = database_cache_->Image(image_id1);
  const Camera& camera1 = database_cache_->Camera(image1.CameraId());

  const Image& image2 = database_cache_->Image(image_id2);
  const Camera& camera2 = database_cache_->Camera(image2.CameraId());

  const SceneGraph& scene_graph = database_cache_->SceneGraph();
  const std::vector<std::pair<point2D_t, point2D_t>>& corrs =
      scene_graph.FindCorrespondencesBetweenImages(image_id1, image_id2);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.XY());
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.XY());
  }

  FeatureMatches matches(corrs.size());
  for (size_t i = 0; i < corrs.size(); ++i) {
    matches[i].point2D_idx1 = corrs[i].first;
    matches[i].point2D_idx2 = corrs[i].second;
  }

  TwoViewGeometry two_view_geometry;
  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  two_view_geometry.EstimateWithRelativePose(
      camera1, points1, camera2, points2, matches, two_view_geometry_options);

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
          options.init_min_num_inliers &&
      std::abs(two_view_geometry.tvec.z()) < options.init_max_forward_motion &&
      two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
    prev_init_image_pair_id_ = image_pair_id;
    prev_init_two_view_geometry_ = two_view_geometry;
    return true;
  }

  return false;
}

std::vector<cam_s> load_cams()
{
	cout << "Loading Cams\n";
	
	ifstream cams;
	cams.open("cams.txt");
	ifstream cent;
	cent.open("center.txt");
	int id;
	std::vector<cam_s> ret;
	while(cams >> id)
	{
		//int cam_id;
		int second;
		int anchor;
		cams >> second;
		cams >> anchor;
		Eigen::Vector4d q;
		Eigen::Vector3d t;
		cams >> q(0);
		cams >> q(1);
		cams >> q(2);
		cams >> q(3);
		cams >> t(0);
		cams >> t(1);
		cams >> t(2);
		double size_x;
		double size_y;
		double f;
		double px;
		double py;
		int inl;
		int id_2;
		cams >> size_x;
		cams >> size_y;
		cams >> f;
		cams >> px;
		cams >> py;
		cams >> inl;
		cams >> id_2;
		Eigen::Vector3d center;
		cent >> center(0);
		cent >> center(1);
		cent >> center(2);
		//cout << cam_id << " " << second << " " << anchor << "\n";
		//cout << q(0) << " " << q(1) << "\n";
		//break;

		cam_s ncam;
		ncam.id = id;
		ncam.second = second;
		ncam.anchor = anchor;
		//ncam.q = q;
		ncam.R = QuaternionToRotationMatrix(q);
		ncam.t = t;
		ncam.size_x = size_x;
		ncam.size_y = size_y;
		ncam.f = f;
		ncam.px = px;
		ncam.py = py;
		ncam.inl = inl;
		ncam.id_2 = id_2;
		ncam.center = center;
		ret.push_back(ncam);
	}
	cams.close();
	cent.close();

	return ret;

}

int count_takes(std::vector<cam_s> C)
{
	int ret = 0;
	for(unsigned int i=0;i<C.size();i++)
	{
		if(C[i].second > ret) ret = C[i].second;
		if(C[i].anchor > ret) ret = C[i].anchor;
	}
	return ret;
}

std::vector<basis_t> find_bases(std::vector<cam_s> C)
{
	std::cout << "Finding Bases\n";
	std::vector<basis_t> ret;
	for(unsigned int i=0;i<C.size();i++)
	{
		//check if the cam is from anchor take
		cam_s c1 = C[i];
		if(c1.anchor != c1.second)
			continue;
		Eigen::Matrix3d R1 = c1.R;
		//Eigen::Vector3d C1 = (-R1.transpose()) * c1.t;
		//cout << c1.t << "\n";
		for(unsigned int j=0;j<C.size();j++)
		{
			cam_s c2 = C[j];
			if(c1.id != c2.id || c1.anchor == c2.anchor)
				continue;
			Eigen::Matrix3d R2 = c2.R;
			//Eigen::Vector3d C2 = (-R2.transpose()) * c2.t;
			Eigen::Matrix3d A = R2.transpose() * R1;
			//cout << c1.anchor << " " << c2.anchor << "\n";
			//cout << A << "\n\n";
			//cout << R2 << "\n\n" << R2.transpose() << "\n\n\n";
			basis_t nb;
			nb.init = c1.anchor;
			nb.final = c2.anchor;
			nb.cam_id = c1.id;
			
			if(c1.anchor < c2.anchor)
			{
				nb.c1 = i;
				nb.c2 = j;
				nb.A = A;
			}
			else
			{
				nb.c1 = j;
				nb.c2 = i;
				nb.A = A.transpose();
			}
			ret.push_back(nb);
		}
	}
	
	return ret;
}

Eigen::Vector3d r2a(Eigen::Matrix3d R)
{
	Eigen::Vector3d ret;

	double phi = acos(0.5 * (R.trace() - 1));
	Eigen::Vector3d r;
	r(0) = R(2,1) - R(1,2);
	r(1) = R(0,2) - R(2,0);
	r(2) = R(1,0) - R(0,1);
	//cout << r << "\n\n";
	//cout << sin(phi) << "\n";
	if(sin(phi) == 0)
	{
		ret(0) = 0;
		ret(1) = 0;
		ret(2) = 0;
	}
	else
	{
		ret = r/sin(phi);
		//cout << "RET " << ret << "\n\n";
	}
	ret = 0.5 * phi * ret;
	//cout << "RET2 " << ret <<  "\n\n";
	return ret;
}

Eigen::Matrix3d a2r(Eigen::Vector3d a)
{
	Eigen::Matrix3d ret;
	if( a(0) == 0 && a(1) == 0 && a(2) == 0 )
	{
		ret.setIdentity();
		return ret;
	}
	double phi = a.norm();
	double v1 = a(0)/phi;
	double v2 = a(1)/phi;
	double v3 = a(2)/phi;

	Eigen::Matrix3d R1;
	R1 << v1*v1, v1*v2, v1*v3, v2*v1, v2*v2, v2*v3, v3*v1, v3*v2, v3*v3;
	R1 = (1-cos(phi)) * R1;
	Eigen::Matrix3d R2;
	R2.setIdentity();
	R2 = cos(phi) * R2;
	Eigen::Matrix3d R3;
	R3 << 0, -v3, v2, v3, 0, -v1, -v2, v1, 0;
	R3 = sin(phi) * R3;
	ret = R1 + R2 + R3;
	return ret;
}

vector<vector<basis_t>> ransac_bases(std::vector<basis_t> G, std::vector<cam_s> C, double sigma)
{
	vector<vector<basis_t>> ret;
	vector<bool> grouped = vector<bool>(G.size(), 0);
	int cl = 0;

	while(1)
	{
		cl++;
		cout << "CLUSTER " << cl << "\n";
		vector<bool> best = vector<bool>(G.size(), 0);
		int best_count = 0;
		for(unsigned int i=0;i<G.size();i++)
		{
			if(grouped[i]) continue;
			Eigen::Matrix3d R1 = G[i].A;
			//variables necessary to compute translation
			Eigen::Vector3d ts1 = C[G[i].c1].t;
			Eigen::Matrix3d Rt1 = C[G[i].c2].R;
			Eigen::Vector3d tt1 = C[G[i].c2].t;
			Eigen::MatrixXd A1(3, 4);
			A1 << Rt1, -ts1;
			//cout << A1 << "\n\n";

			for(unsigned int j=i+1;j<G.size();j++)
			{
				if(grouped[j]) continue;
				Eigen::Matrix3d R2 = G[j].A;
				//variables necessary to compute translation
				Eigen::Vector3d ts2 = C[G[j].c1].t;
				Eigen::Matrix3d Rt2 = C[G[j].c2].R;
				Eigen::Vector3d tt2 = C[G[j].c2].t;
				Eigen::MatrixXd A2(3, 4);
				A2 << Rt2, -ts2;

				Eigen::Matrix3d R = R2 * R1.transpose();
				Eigen::Vector3d k3 = r2a(R);
				double n = k3.norm();

				//cout << R << "\n\n";
				//cout << k3 << "\n\n";
				//cout << n << "\n\n";

				//the condition here does not have to bee too strong, as passing it would lead only to longer evaluation but not to errors
				if(n > sigma * 5 * 0.0247)
				{
					continue;
				}

				//compute the translation hypothesis
				Eigen::MatrixXd A(6, 4);
				A << A1, A2;
				Eigen::MatrixXd b(6, 1);
				b << -tt1, -tt2;
				Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

				//cout << x << "\n\n";

				//check the inliers
				vector<bool> inliers = vector<bool>(G.size(), 0);
				int count = 0;
				for(unsigned int k=0;k<G.size();k++)
				{
					if(grouped[k]) continue;
					Eigen::Matrix3d R3 = C[G[k].c1].R;
					Eigen::Vector3d t3 = C[G[k].c1].t;
					Eigen::Vector3d c3 = -R3.transpose() * t3;

					Eigen::Vector3d c3t;
					c3t(0) = x(0);
					c3t(1) = x(1);
					c3t(2) = x(2);
					Eigen::Vector3d c3_trans = x(3) * R1 * c3 + c3t;

					Eigen::Matrix3d R4 = C[G[k].c2].R;
					Eigen::Vector3d t4 = C[G[k].c2].t;
					Eigen::Vector3d c4 = -R4.transpose() * t4;

					//rotational consistency
					Eigen::Matrix3d R34 = R4.transpose() * R3;
					Eigen::Matrix3d R134 = R1 * R34.transpose();
					Eigen::Vector3d a34 = r2a(R134);
					//cout << "R" << a34.norm() << "\n";
					if(a34.norm() > 5*sigma*0.0247)
						continue;

					//translational consistency
					Eigen::Vector3d point = C[G[k].c2].center;
					Eigen::Vector3d v1 = c3_trans - point;
					Eigen::Vector3d v2 = c4 - point;
					double angle = (v1.dot(v2))/(v1.norm() * v2.norm());

					if(angle >= 0.99)
					{
						inliers[k] = 1;
						count++;
					}
				
				}
				if(count > best_count)
				{
					best = inliers;
					best_count = count;
				}			
			}
		}

		cout << "size " << best_count << "\n";
		if(best_count == 0)
			break;

		vector<basis_t> ND = vector<basis_t>(best_count);
		int j=0;
		for(unsigned int i=0;i<G.size();i++)
		{
			if(best[i])
			{
				ND[j] = G[i];
				j++;
				grouped[i] = 1;
			}
		}
		ret.push_back(ND);
		if(G.size() == (unsigned int)best_count) break;
	}

	return ret;
}

std::vector<std::vector<std::vector<basis_t>>> divide_bases(std::vector<basis_t> B, std::vector<cam_s> C, double sigma)
{
	std::vector<std::vector<std::vector<basis_t>>> ret;
	cout << "Clustering bases\n";
	int n=0;
	vector<bool> grouped = vector<bool>(B.size(), 0);
	while(1)
	{
		n++;
		cout << "Motion " << n << "\n";
		vector<basis_t> GR;
		int init = -1;
		int final = -1;
		//find the group corresponding to the new motion
		for(unsigned int i=0;i<B.size();i++)
		{
			if(GR.size() == 0 && !grouped[i])
			{
				init = B[i].init;
				final = B[i].final;
				grouped[i] = 1;
				basis_t nb = B[i];
				GR.push_back(nb);
			}
			else if(!grouped[i] && ( (B[i].init == init && B[i].final == final ) || (B[i].init == final && B[i].final == init) ))
			{
				grouped[i] = 1;
				basis_t nb = B[i];
				GR.push_back(nb);
			}
		}
		if(GR.size() == 0)
			break;
		cout << GR.size() << "\n";

		//cluster the camera pairs in the group
		vector<vector<basis_t>> CL = ransac_bases(GR, C, sigma);
		ret.push_back(CL);
	}
	
	return ret;
}

double med(std::vector<double> v, int first, int last, int desired)
{
	if(first == last)
		return v[first];
	double pivot = v[first];
	int low = first+1;
	int high = last;
	while(1)
	{
		while(v[low] <= pivot && low <= last) low++;
		while(v[high] >= pivot && high >= first) high--;
		if(low >= high) break;
		double bz = v[low];
		v[low] = v[high];
		v[high] = bz;
	}
	if(desired >= first && desired < high)
		return med(v, first+1, high, desired+1);
	if(desired >= low && desired <= last)
		return med(v, low, last, desired);
	return pivot;
}

int med(std::vector<int> v, int first, int last, int desired)
{
	if(first == last)
		return v[first];
	int pivot = v[first];
	int low = first+1;
	int high = last;
	while(1)
	{
		while(v[low] <= pivot && low <= last) low++;
		while(v[high] >= pivot && high >= first) high--;
		if(low >= high) break;
		int bz = v[low];
		v[low] = v[high];
		v[high] = bz;
	}
	if(desired >= first && desired < high)
		return med(v, first+1, high, desired+1);
	if(desired >= low && desired <= last)
		return med(v, low, last, desired);
	return pivot;
}

Eigen::Vector3d median(std::vector<Eigen::Vector3d> CL)
{
	Eigen::Vector3d ret;
	//double * v = malloc(sizeof(double) * CL.size());
	std::vector<double> v = vector<double>(CL.size());
	for(int i=0;i<3;i++)
	{
		for(unsigned int j=0;j<CL.size();j++)
		{
			v[j] = CL[j](i);
		}
		double m = med(v, 0, CL.size()-1, CL.size()/2);
		ret(i) = m;
	}
	return ret;
}

Eigen::Vector3i median(std::vector<Eigen::Vector3i> CL)
{
	Eigen::Vector3i ret;
	//double * v = malloc(sizeof(double) * CL.size());
	std::vector<int> v = vector<int>(CL.size());
	for(int i=0;i<3;i++)
	{
		for(unsigned int j=0;j<CL.size();j++)
		{
			v[j] = CL[j](i);
		}
		double m = med(v, 0, CL.size()-1, CL.size()/2);
		ret(i) = m;
	}
	return ret;
}

std::vector<std::vector<clust_b>> meanclust(std::vector<std::vector<std::vector<basis_t>>> C, std::vector<cam_s> cams)
{
	cout << "Meanclust\n";
	std::vector<std::vector<clust_b>> ret;
	for(unsigned int i=0;i<C.size();i++)
	{
		std::vector<clust_b> r;
		for(unsigned int j=0;j<C[i].size();j++)
		{
			vector<basis_t> cl = C[i][j];
			vector<Eigen::Vector3d> R = vector<Eigen::Vector3d>(cl.size());
			Eigen::MatrixXd A;
			Eigen::MatrixXd b;
			Eigen::MatrixXd E;
			Eigen::MatrixXd f;
			
			for(unsigned int k=0;k<cl.size();k++)
			{
				R[k] = r2a(cl[k].A);
				//cout << r2a(cl[k].A) << "\n\n";
				Eigen::Vector3d ts1 = cams[cl[k].c1].t;
				Eigen::Matrix3d Rt1 = cams[cl[k].c2].R;
				Eigen::Vector3d tt1 = cams[cl[k].c2].t;

				Eigen::MatrixXd A1(3, 4);
				A1 << Rt1, -ts1;
				Eigen::Vector3d b1 = -tt1;

				Eigen::Vector3d ts2 = cams[cl[k].c2].t;
				Eigen::Matrix3d Rt2 = cams[cl[k].c1].R;
				Eigen::Vector3d tt2 = cams[cl[k].c1].t;

				Eigen::MatrixXd A2(3, 4);
				A2 << Rt2, -ts2;
				Eigen::Vector3d b2 = -tt2;

				if(k==0)
				{
					A = A1;
					b = b1;
					E = A2;
					f = b2;
				}
				else
				{
					Eigen::MatrixXd nA = Eigen::MatrixXd(3*(k+1), 4);
					nA << A, A1;
					A = nA;
					Eigen::MatrixXd nb = Eigen::MatrixXd(3*(k+1), 1);
					nb << b, b1;
					b = nb;
					Eigen::MatrixXd nE = Eigen::MatrixXd(3*(k+1), 4);
					nE << E, A2;
					E = nE;
					Eigen::MatrixXd nf = Eigen::MatrixXd(3*(k+1), 1);
					nf << f, b2;
					f = nf;
				}
			}
			Eigen::Vector3d MR = median(R);
			Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);
			Eigen::Vector4d x2 = E.colPivHouseholderQr().solve(f);
			//cout << "MEDIAN " << MR << "\n\n\n\n";
			//cout << x << "\n\n";
			clust_b nc;
			nc.size = cl.size();
			if(cl[0].init < cl[0].final)
			{
				nc.init = cl[0].init;
				nc.final = cl[0].final;
			}
			else
			{
				nc.final = cl[0].init;
				nc.init = cl[0].final;
			}
			nc.R = a2r(MR);
			
			Eigen::Vector3d mt;
			mt(0) = x(0);
			mt(1) = x(1);
			mt(2) = x(2);
			nc.t1 = mt;
			nc.sigma1 = x(3);

			Eigen::Vector3d mt2;
			mt2(0) = x2(0);
			mt2(1) = x2(1);
			mt2(2) = x2(2);
			nc.t2 = mt2;
			nc.sigma2 = x2(3);

			r.push_back(nc);
		}
		ret.push_back(r);
	}

	return ret;
}

std::vector<cycle_3> find3cycles(std::vector<std::vector<clust_b>> M)
{
	std::vector<cycle_3> ret;
	for(unsigned int i=0;i<M.size();i++)
	{
		if(!M[i].size()) continue;
		for(unsigned int j=0;j<M.size();j++)
		{
			if(!M[j].size()) continue;
			if(M[i][0].final != M[j][0].init) continue;
			for(unsigned int k=0;k<M.size();k++)
			{
				if(!M[k].size()) continue;
				if(M[i][0].init != M[k][0].init || M[j][0].final != M[k][0].final) continue;
				//cout << M[i][0].init << " " << M[i][0].final << " " << M[j][0].final << "\n";
				for(unsigned int a=0;a<M[i].size();a++)
				{
					Eigen::Matrix3d R12 = M[i][a].R;
					Eigen::Vector3d o12 = M[i][a].t1;
					double s12 = M[i][a].sigma1;
					for(unsigned int b=0;b<M[j].size();b++)
					{
						Eigen::Matrix3d R23 = M[j][b].R;
						Eigen::Vector3d o23 = M[j][b].t1;
						double s23 = M[j][b].sigma1;
						for(unsigned int c=0;c<M[k].size();c++)
						{
							Eigen::Matrix3d R31 = M[k][c].R.transpose();
							Eigen::Vector3d o31 = M[k][c].t2;
							double s31 = M[i][a].sigma2;

							Eigen::Vector3d ev = r2a(R31 * R23 * R12);
							double i1 = ev.norm();

							double i2 = s12 * s23 * s31;

							Eigen::Vector3d i3 = s31*s23*R31*R23*o12 + s31*R31*o23 + o31;
							//cout << R31 * R23 * R12 << "\n\n";
							//cout << ev.norm() << "\n\n";
							
							cycle_3 ncl;
							ncl.t1 = i;
							ncl.cl1 = a;
							ncl.t2 = j;
							ncl.cl2 = b;
							ncl.t3 = k;
							ncl.cl3 = c;

							ncl.i1 = i1;
							ncl.i2 = i2;
							ncl.i3 = i3;

							ret.push_back(ncl);
						}
					}
				}
			}
		}
	}

	return ret;
}

std::vector<int> cluster_cycles(std::vector<cycle_3> C, double thr, std::vector<cam_s> cams, std::vector<std::vector<std::vector<basis_t>>> CL, std::vector<double> * bzs)
{
	vector<int> ret(C.size());
	vector<int> pos(C.size());
	vector<cycle_3> NC;
	cout << "Clustering cycles\n";
	for(unsigned int i=0;i<C.size();i++)
	{
		if(C[i].i1 <= thr)
		{
			vector<basis_t> cc = CL[C[i].t3][C[i].cl3];
			double bz = 1;
			for(unsigned int j=0;j<cc.size();j++)
			{
				Eigen::Matrix3d R = cams[cc[j].c1].R;
				Eigen::Vector3d center = -R.transpose() * cams[cc[j].c1].t;
				Eigen::Vector3d point = cams[cc[j].c1].center;
				Eigen::Vector3d point2 = point + C[i].i3;
				Eigen::Vector3d line1 = point-center;
				Eigen::Vector3d line2 = point2-center;
				double angle = line1.dot(line2)/(line1.norm() * line2.norm() );
				if(angle < bz)
					bz = angle;
			}
			//cout << C[i].i3 << "\n\n";
			//cout << bz << "\n\n";
			//cout << C[i].i2 << "\n\n";
			// ? 0.99 probably better than 0.98 ?
			if(bz >= 0.98)
			{
				NC.push_back(C[i]);
				ret[i] = 0;
				pos[i] = NC.size()-1;
				(*bzs)[i] = bz;
			}
			else { ret[i] = -1; pos[i] = -1; (*bzs)[i] = 0;}
		}
		else { ret[i] = -1; pos[i] = -1; (*bzs)[i] = 0;}
		//cout << C[i].i1 << " " << ret[i] << "\n";
	}

	//find connected components
	Eigen::MatrixXd D = Eigen::MatrixXd(NC.size(), NC.size());
	D.setZero();
	Eigen::MatrixXi B = Eigen::MatrixXi(NC.size(), NC.size());
	B.setZero();
	for(unsigned int i=0;i<NC.size();i++)
	{
		double sum = 0;
		for(unsigned int j=0;j<NC.size();j++)
		{
			if(i==j) continue;

			//find if the cycles share a cluster
			bool ended = 0;
			if(NC[i].t1 == NC[j].t1 && NC[i].cl1 == NC[j].cl1)
				ended = 1;
			else if(NC[i].t1 == NC[j].t2 && NC[i].cl1 == NC[j].cl2)
				ended = 1;
			else if(NC[i].t1 == NC[j].t3 && NC[i].cl1 == NC[j].cl3)
				ended = 1;
			else if(NC[i].t2 == NC[j].t1 && NC[i].cl2 == NC[j].cl1)
				ended = 1;
			else if(NC[i].t2 == NC[j].t2 && NC[i].cl2 == NC[j].cl2)
				ended = 1;
			else if(NC[i].t2 == NC[j].t3 && NC[i].cl2 == NC[j].cl3)
				ended = 1;
			else if(NC[i].t3 == NC[j].t1 && NC[i].cl3 == NC[j].cl1)
				ended = 1;
			else if(NC[i].t3 == NC[j].t2 && NC[i].cl3 == NC[j].cl2)
				ended = 1;
			else if(NC[i].t3 == NC[j].t3 && NC[i].cl3 == NC[j].cl3)
				ended = 1;

			if(ended)
			{
				D(i,j) = -exp(NC[i].i1)-exp(NC[j].i1);
				B(i,j) = 1;
				sum = sum + D(i,j);
			}
		}
		D(i,i) = -sum;
	}

	//find the connected components
	int gr = 1;
	while(1)
	{
		vector<int> q;
		for(unsigned int i=0;i<ret.size();i++)
		{
			if(ret[i] == 0)
			{
				ret[i] = gr;
				q.push_back(i);
				break;
			}
		}
		if(q.size() == 0) break;
		while(q.size())
		{
			int cur = q[q.size()-1];
			q.pop_back();
			for(unsigned int i=0;i<C.size();i++)
			{
				if(B(pos[cur], pos[i]) && !ret[i])
				{
					ret[i] = gr;
					q.push_back(i);
				}
			}
		}
		gr++;
	}

	
	return ret;
}

std::pair<std::pair<Eigen::MatrixXi, Eigen::MatrixXi>, Eigen::MatrixXd> select_bases(std::vector<std::vector<clust_b>> M, std::vector<cycle_3> CY, std::vector<int> CL2, int takes, std::vector<double> bzs, double thr1, double thr2)
{
	double best_sum = 5;
	int best_count = 0;
	Eigen::MatrixXi best_if2cl;
	Eigen::MatrixXi best_sm;
	Eigen::MatrixXd best_cons;
	for(int cl=1;;cl++)
	{
		//find the component
		std::vector<cycle_3> NBC;
		vector<double> nbz;
		for(unsigned int i=0;i<CL2.size();i++)
		{
			if(CL2[i] == cl)
			{
				NBC.push_back(CY[i]);
				nbz.push_back(bzs[i]);
				//cout << "BZ " << bzs[i] << "\n";
			}
		}
		if(!NBC.size()) break;

		//initialize the structures
		//shows which motion clusters are between the takes
		//somewhere here is a problem
		Eigen::MatrixXi if2cl = Eigen::MatrixXi::Ones(takes, takes);
		if2cl = -1*if2cl;
		//shows which motion between the takes has been selected
		Eigen::MatrixXi sm = Eigen::MatrixXi::Ones(takes, takes);
		sm = -1*sm;
		Eigen::MatrixXd consistency = Eigen::MatrixXd::Ones(takes, takes);
		consistency = -1*consistency;
		for(unsigned int i=0;i<M.size();i++)
		{
			if(!M[i].size()) continue;
			int init = M[i][0].init;
			int final = M[i][0].final;
			if2cl(init-1, final-1) = i;
			if2cl(final-1, init-1) = i;
		}
		vector<bool> comp(takes);
		for(int i=0;i<takes;i++)
			comp[i] = 0;

		//for each motion cluster find its consistency as the 
		//save to a vector of vectors with a structure similar to the M
		vector<vector<double>> rc(M.size());
		vector<vector<double>> tc(M.size());
		for(unsigned int i=0;i<M.size();i++)
		{
			rc[i] = vector<double>(M[i].size());
			tc[i] = vector<double>(M[i].size());
			for(unsigned int j=0;j<M[i].size();j++)
			{
				rc[i][j] = 1;
				tc[i][j] = 0;
			}
		}
		for(unsigned int i=0;i<NBC.size();i++)
		{
			double rc_cur = NBC[i].i1;
			double tc_cur = nbz[i];
			
			int t=NBC[i].t1;
			int c=NBC[i].cl1;
			if(rc_cur < rc[t][c])
				rc[t][c] = rc_cur;
			if(tc_cur > tc[t][c])
				tc[t][c] = tc_cur;

			t=NBC[i].t2;
			c=NBC[i].cl2;
			if(rc_cur < rc[t][c])
				rc[t][c] = rc_cur;
			if(tc_cur > tc[t][c])
				tc[t][c] = tc_cur;

			t=NBC[i].t3;
			c=NBC[i].cl3;
			if(rc_cur < rc[t][c])
				rc[t][c] = rc_cur;
			if(tc_cur > tc[t][c])
				tc[t][c] = tc_cur;
		}
		
		//select the first motion cluster (and pair of takes)
		//take into account only the consistency and the number of the cameras
		double best_score = 3;
		int best_t = -1;
		int best_c = -1;
		for(unsigned int i=0;i<M.size();i++)
		{
			for(unsigned int j=0;j<M[i].size();j++)
			{
				if(rc[i][j] > thr1 || tc[i][j] < thr2 || M[i][j].size < 1)
					continue;

				double score = rc[i][j]/thr1 + (1-tc[i][j])/thr2 + 1/((double)M[i][j].size);
				//cout << rc[i][j] << " " << 1-tc[i][j] << " " << 1/((double)M[i][j].size + 1) << "\n";
				if(score < best_score)
				{
					best_score = score;
					best_t = i;
					best_c = j;
				}
			}
		}

		//add other motion clusters, apart from the previous take into account the number of common observed points (maybe in a form of classes, take the best from the highest class, or add it to the score (somehow))
		int i1 = M[best_t][best_c].init;
		int i2 = M[best_t][best_c].final;
		comp[i1 - 1] = 1;
		comp[i2 - 1] = 1;
		sm(i1-1, i2-1) = best_c;
		sm(i2-1, i1-1) = best_c;
		consistency(i1-1, i2-1) = best_score;
		consistency(i2-1, i1-1) = best_score;
		int count = 2;
		double sum = best_score;
		//TODO
		//find the list of tracks observed by the clusters in the spanning tree
		//use the list to add the best next cluster
		while(1)
		{
			double best_score = 4;
			bool found = false;
			int best_t = -1;
			int best_c = -1;
			for(unsigned int i=0;i<M.size();i++)
			{
				//only those clusters which have one take in the component and one outside of it
				//cout << M[i].size() << "\n";
				if(!M[i].size()) continue;
				bool comp1 = comp[(M[i][0].init)-1];
				bool comp2 = comp[(M[i][0].final)-1];
				if((comp1 && comp2) || ((!comp1) && (!comp2))) continue;
				for(unsigned int j=0;j<M[i].size();j++)
				{
					if(rc[i][j] > thr1 || tc[i][j] < thr2 || M[i][j].size < 1);
					//	continue;

					double score = rc[i][j]/thr1 + (1-tc[i][j])/thr2 + 1/((double)M[i][j].size);
					//cout << rc[i][j] << " " << 1-tc[i][j] << " " << 1/((double)M[i][j].size + 1) << "\n";
					if(score < best_score)
					{
						best_score = score;
						best_t = i;
						best_c = j;
						found = 1;
					}
				}
			}
			if(!found) break;

			int i1 = M[best_t][best_c].init;
			int i2 = M[best_t][best_c].final;
			comp[i1 - 1] = 1;
			comp[i2 - 1] = 1;
			sm(i1-1, i2-1) = best_c;
			sm(i2-1, i1-1) = best_c;
			consistency(i1-1, i2-1) = best_score;
			consistency(i2-1, i1-1) = best_score;
			count++;
			sum+=best_score;
			
		}
		
		//evaluate the quality of the whole spanning tree
		//cout << if2cl << "\n\n\n\n" << sm << "\n";
		if(count > best_count || (count == best_count && sum < best_sum))
		{
			best_count = count;
			best_sum = sum;
			best_sm = sm;
			best_if2cl = if2cl;
			best_cons = consistency;
		}
	}
	//cout << best_sm << "\n";
	std::pair<Eigen::MatrixXi, Eigen::MatrixXi> ret (best_if2cl, best_sm);
	std::pair<std::pair<Eigen::MatrixXi, Eigen::MatrixXi>, Eigen::MatrixXd> ret2(ret, best_cons);
	return ret2;
}

int find_reference(Eigen::MatrixXd st, int takes)
{
	int reference = 0;
	int best_deg = 0;
	double best_w_deg = 0;
	for(int i=0;i<takes;i++)
	{
		int deg = 0;
		double w_deg = 0;
		for(int j=0;j<takes;j++)
		{
			if(st(i,j) > -0.5)
			{
				deg++;
				w_deg = w_deg - st(i,j);
			}
		}
		if(deg > best_deg || (deg==best_deg && w_deg > best_w_deg))
		{
			reference = i;
			best_deg = deg;
			best_w_deg = w_deg;
		}
	}
	return reference;
}

std::vector<std::vector<trans_s>> find_transformations(std::pair<Eigen::MatrixXi, Eigen::MatrixXi> st, int reference, std::vector<std::vector<clust_b>> M, int takes)
{
	std::vector<std::vector<trans_s>> ret(takes);
	vector<bool> found(takes);
	for(int i=0;i<takes;i++)
	{
		if(i==reference)
			found[i] = 1;
		else
			found[i] = 0;
	}
	cout << "Finding transformations\n";
	vector<pair<int, int>> Q;
	for(int i=0;i<takes;i++)
	{
		for(int j=0;j<takes;j++)
		{
			if(st.second(i,j) > -1)
			{
				pair<int, int> ne(i,j);
				Q.push_back(ne);
			}
		}
	}
	queue<int> acc;
	acc.push(reference);
	while(!acc.empty())
	{
		int t = acc.front();
		acc.pop();

		for(unsigned int i=0;i<Q.size();i++)
		{
			if(Q[i].first == t && !found[Q[i].second])
			{
				int mov = st.first(t,Q[i].second);
				int cl = st.second(t,Q[i].second);
				cout << mov << " " << cl << "\n";
				if(Q[i].second > t)
				{
					trans_s nt;
					nt.R = M[mov][cl].R.transpose();
					nt.o = M[mov][cl].t2;
					nt.s = M[mov][cl].sigma2;
					ret[Q[i].second].push_back(nt);
				}
				else
				{
					trans_s nt;
					nt.R = M[mov][cl].R;
					nt.o = M[mov][cl].t1;
					nt.s = M[mov][cl].sigma1;
					ret[Q[i].second].push_back(nt);
				}
				for(unsigned int j=0;j<ret[t].size();j++)
					ret[Q[i].second].push_back(ret[t][j]);
				acc.push(Q[i].second);
				found[Q[i].second] = 1;
			}
		}
	}
	

	return ret;
}

std::vector<motion_t> find_motion(std::vector<cam_s> C)
{
	std::vector<motion_t> ret;

	for(unsigned int i=0;i<C.size();i++)
	{
		cam_s c1 = C[i];
		Eigen::Matrix3d R1 = c1.R;
		Eigen::Vector3d C1 = (-R1.transpose()) * c1.t;
		//cout << c1.t << "\n";
		for(unsigned int j=i+1;j<C.size();j++)
		{
			if(i==j) continue;
			cam_s c2 = C[j];
			if(c1.id != c2.id || c1.anchor != c2.anchor)
				continue;
			Eigen::Matrix3d R2 = c2.R;
			Eigen::Vector3d C2 = (-R2.transpose()) * c2.t;
			Eigen::Matrix3d A = R1.transpose() * R2;
			Eigen::Vector3d T = C1 - A * C2;
			Eigen::Vector3d T2 = C2 - A.transpose() * C1;
			
			motion_t nb;
			nb.init = c1.second;
			nb.final = c1.anchor;
			nb.cam_id = c1.id;
			
			if(c1.second < c1.anchor)
			{
				nb.c1 = i;
				nb.c2 = j;
				nb.A = A;
				nb.t = T;
			}
			else
			{
				nb.c1 = j;
				nb.c2 = i;
				nb.A = A.transpose();
				nb.t = T2;
			}
			ret.push_back(nb);
		}
	}

	return ret;
}

std::vector<motion_t> transform_motion(std::vector<motion_t> C, std::vector<std::vector<trans_s>> transform)
{
	std::vector<motion_t> D;
	for(unsigned int i=0;i<C.size();i++)
	{
		vector<trans_s> tr = transform[C[i].final-1];
		if(!tr.size())
		{
			D.push_back(C[i]);
			continue;
		}
		Eigen::Matrix3d R = C[i].A;
		Eigen::Vector3d t = C[i].t;
		for(unsigned int j=0;j<tr.size();j++)
		{
			R = tr[j].R * R * tr[j].R.transpose();
			t = tr[j].s * tr[j].R * t + tr[j].o - R*tr[j].o;
		}
		motion_t nb;
		nb.init = C[i].init;
		nb.final = C[i].final;
		nb.cam_id = C[i].cam_id;
		nb.c1 = C[i].c1;
		nb.c2 = C[i].c2;
		nb.A = R;
		nb.t = t;
		D.push_back(nb);
	}
	return D;
}

Eigen::MatrixXd distance3d(vector<motion_t> C)
{
	Eigen::MatrixXd D = Eigen::MatrixXd::Zero(C.size(), C.size()); 
	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=0;j<C.size();j++)
		{
			if(i==j) continue;
			Eigen::Matrix3d R = C[i].A.transpose() * C[j].A;
			Eigen::Vector3d r = r2a(R);
			D(i,j) = r.norm();
		}
	}
	Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(C.size(), C.size());
	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=0;j<C.size();j++)
		{
			ret(i,j) = 0.5*(D(i,j) + D(j,i));
		}
	}
	return ret;
}

Eigen::MatrixXd distance3d_trans(vector<motion_t> C)
{
	Eigen::MatrixXd D = Eigen::MatrixXd::Zero(C.size(), C.size()); 
	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=0;j<C.size();j++)
		{
			if(i==j) continue;
			Eigen::Vector3d t = C[i].t - C[j].t;
			D(i,j) = t.norm();
		}
	}
	Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(C.size(), C.size());
	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=0;j<C.size();j++)
		{
			ret(i,j) = 0.5*(D(i,j) + D(j,i));
		}
	}
	return ret;
}

Eigen::MatrixXd dist2laplace(Eigen::MatrixXd D, int size)
{
	Eigen::MatrixXd L = Eigen::MatrixXd::Zero(size, size);
	for(int i=0;i<size;i++)
	{
		double sum = 0;
		for(int j=0;j<size;j++)
		{
			if(i==j) continue;
			if(D(i,j) == 0)
			{
				L(i,j) = -100000000;
				sum = sum + 100000000;
			}
			else
			{
				L(i,j) = -1/(D(i,j));
				sum = sum + 1/(D(i,j));
			}
		}
		L(i,i) = sum;
	}

	Eigen::MatrixXd L_ = Eigen::MatrixXd::Zero(size, size);
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			L_(i,j) = L(i,j)/(sqrt(L(i,i)) * sqrt(L(j,j)));
		}
	}
	return L_;
}

std::vector<int> kmeans(std::vector<Eigen::VectorXd> A, int k, int dim)
{
	std::vector<int> ret(A.size());
	for(unsigned int i=0;i<A.size();i++)
		ret[i] = -1;
	srand(time(NULL));
	int first = rand() % A.size();
	vector<Eigen::VectorXd> means;
	means.push_back(A[first]);
	for(int i=1;i<k;i++)
	{
		//select the center which is the furthest from the nearest point
		double max_dist = 0;
		int next = 0;
		for(unsigned int j=0;j<A.size();j++)
		{
			double min_dist = (A[j] - means[0]).norm();
			for(int l=1;l<i;l++)
			{
				double dist = (A[j] - means[l]).norm();
				if(dist < min_dist) min_dist = dist;
			}
			if(min_dist > max_dist)
			{
				max_dist = min_dist;
				next = j;
			}
		}
		means.push_back(A[next]);
	}

	/*for(int i=0;i<k;i++)
	{
		cout << "M" << means[i] << "\n\n";
	}
	cout << "\n\n";*/

	bool i1 = 1;
	double q = 0;
	while(1)
	{
		//assign mean to each point
		std::vector<int> n_ret(A.size());
		double nq = 0;
		for(unsigned int i=0;i<A.size();i++)
		{
			double min_dist = (A[i] - means[0]).norm();
			int sm = 0;
			for(int j=1;j<k;j++)
			{
				double dist = (A[i] - means[j]).norm();
				if(dist < min_dist)
				{
					sm = j;
					min_dist = dist;
				}
			}
			nq += min_dist;
			n_ret[i] = sm;
		}

		//detect if something has changed
		if(!i1)
		{
			bool ch = 0;
			for(unsigned int i=0;i<A.size();i++)
			{
				if(ret[i] == n_ret[i])
				{
					ch=1;
					break;
				}
			}
			if(!ch) break;
			ret = n_ret;
		}
		else
		{
			ret = n_ret;
		}

		if(i1 || nq < q)
			q = nq;
		else
			break;

		//recompute the means, find out if reinitialisation is necessary
		for(int i=0;i<k;i++)
		{
			Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
			int count = 0;
			for(unsigned int j=0;j<A.size();j++)
			{
				if(ret[j] == i)
				{
					count++;
					mean = mean + A[j];
				}
			}
			if(!count)
			{
				//reinitialise the mean
				int np = rand() % A.size();
				means[i] = A[np];
			}
			else
			{
				mean = mean / count;
				means[i] = mean;
			}
		}

		i1 = 0;
	}
	/*for(int i=1;i<k;i++)
	{
		//cout << means[i] << "\n\n";
	}*/

	return ret;
}

std::vector<motion_t> remove_id(std::vector<motion_t> C, double pd, double thr1, double thr2)
{
	std::vector<motion_t> ret;
	cout << "Removing zero motions\n";
	int n=0;
	vector<bool> grouped = vector<bool>(C.size(), 0);
	while(1)
	{
		n++;
		cout << "Motion " << n << "\n";
		vector<motion_t> B;
		int init = -1;
		int final = -1;
		//find the group corresponding to the new motion
		for(unsigned int i=0;i<C.size();i++)
		{
			if(B.size() == 0 && !grouped[i])
			{
				init = C[i].init;
				final = C[i].final;
				grouped[i] = 1;
				motion_t nb = C[i];
				B.push_back(nb);
			}
			else if(!grouped[i] && ( (C[i].init == init && C[i].final == final ) || (C[i].init == final && C[i].final == init) ))
			{
				grouped[i] = 1;
				motion_t nb = C[i];
				B.push_back(nb);
			}
		}
		if(B.size() == 0)
			break;
		cout << B.size() << "\n";

		//find distance between rotations and translations
		Eigen::MatrixXd D1 = distance3d(B);
		Eigen::MatrixXd D2 = distance3d_trans(B);
		//compute the graph laplacian
		Eigen::MatrixXd L1 = dist2laplace(D1, B.size());
		Eigen::MatrixXd L2 = dist2laplace(D2, B.size());
		Eigen::MatrixXd L = L1+L2;

		//perform the spectral clustering
		Eigen::EigenSolver<Eigen::MatrixXd> es(L);
		Eigen::MatrixXcd lambda = es.eigenvalues();
		Eigen::MatrixXcd V = es.eigenvectors();
		//cout << lambda << "\n\n";
		int max_i = 1;
		for(unsigned int i=1;i<B.size();i++)
		{
			if(lambda(i).real() < 1.2)
			{
				max_i = i+1;
				//break;
			}
			//cout << lambda(i).real() << "\n";
		}

		//find the vectors
		vector<Eigen::VectorXd> a;
		for(unsigned int j=0;j<B.size();j++)
		{
			Eigen::VectorXd nv(max_i);
			for(int k=0;k<max_i;k++)
			{
				nv(k) = V(j,k).real();
			}
			a.push_back(nv);
		}

		//double chtg = 10 + B.size();
		double chtg = INFINITY;
		double best_S = -chtg;
		//cout << 10 + B.size() << " " << chtg << "\n";
		vector<int> clust;
		for(unsigned int i=max_i;i<=B.size();i++)
		{
			
			//perform the kmeans
			vector<int> c = kmeans(a, i, max_i);
			//cout << c.size() << "\n";

			//compute the silhouette and evaluate the clustering
			double s = 0;
			for(unsigned int j=0;j<B.size();j++)
			{
				vector<double> d(i);
				vector<int> n(i);
				for(unsigned int bz = 0;bz<i;bz++)
				{
					d[bz] = 0;
					n[bz] = 0;
				}
				//cout << c[j] << "\n";
				for(unsigned int k=0;k<B.size();k++)
				{
					n[c[k]]++;
					d[c[k]] += D1(j,k) + D2(j,k);
				}

				double as = 0;
				double bs = INFINITY;
				if(n[c[j]] > 1)
					as = d[c[j]]/(n[c[j]]-1);

				for(unsigned int k=0;k<i;k++)
				{
					if(k==(unsigned int)c[j]) continue;
					double bb = d[k]/(n[k]-1);
					if(bb < bs)
						bs = bb;
				}
				double S;
				if(i==1)
					S = 0;
				else if(n[c[j]] == 1)
					S = 0;
				else
				{
					double mx = bs;
					if(as > bs) mx = as;
					S = (bs-as)/mx;
				}
				s = s+S;
			}
			//cout << s << "\n";
			//cout << best_S << "\n";
			if(s > best_S)
			{
				clust = c;
				best_S = s;
			}
			else
			{
				if(!clust.size()) clust = c;
				break;
			}
		}

		
		//detect if the vectors are zero vectors and eventually remove them
		/*cout << "CS " << clust.size() << "\n";
		for(unsigned int i=0;i<clust.size();i++)
		{
			cout << clust[i] << "\n";
		}
		cout << "\n";*/
		for(int i=0;;i++)
		{
			vector<Eigen::Vector3d> rot;
			vector<Eigen::Vector3d> trans;
			for(unsigned int j=0;j<B.size();j++)
			{
				if(clust[j] == i)
				{
					rot.push_back(r2a(B[j].A));
					trans.push_back(B[j].t);
				}
			}
			if(!rot.size()) break;

			Eigen::Vector3d med_rot = median(rot);
			Eigen::Vector3d med_trans = median(trans);
			if(med_rot.norm() < 0.0247 * thr1)
			{
				if(med_trans.norm() < thr2 * 0.007 * pd)
				{
					cout << "IDENTITY REMOVED\n";
				}
				else
				{
					cout << "OK\n";
					for(unsigned int j=0;j<B.size();j++)
					{
						if(clust[j] == i)
							ret.push_back(B[j]);
					}
				}
				//cout << med_trans.norm() / pd << "\n";
			}
			else
			{
				cout << "OK\n";
				for(unsigned int j=0;j<B.size();j++)
				{
					if(clust[j] == i)
						ret.push_back(B[j]);
				}
				//cout << med_rot.norm() << " " << med_trans.norm() << " ";
			}
			
		}
		
	}

	/*for(int i=0;i<ret.size();i++)
	{
		cout << (r2a(ret[i].A)).norm() << "\n";
	}*/
	
	return ret;
}

double princ_dist(std::vector<cam_s> C, int take)
{
	double ret = 0;
	double cnt = 0;
	for(unsigned int i=0;i<C.size();i++)
	{
		if(C[i].anchor == take && C[i].second == take)
		{
			Eigen::Matrix3d R = C[i].R;
			Eigen::Vector3d C1 = (-R.transpose()) * C[i].t;
			Eigen::Vector3d C2 = C[i].center;
			ret += (C1 - C2).norm();
			cnt += 1;
		}
	}
	return ret/cnt;
}

std::vector<cam_s> change_basis_cams(std::vector<cam_s> C, std::vector<std::vector<trans_s>> T)
{
	std::vector<cam_s> ret;
	for(unsigned int i=0;i<C.size();i++)
	{
		vector<trans_s> tr = T[C[i].anchor-1];
		if(!tr.size())
		{
			ret.push_back(C[i]);
			continue;
		}

		cam_s nc;
		nc.id = C[i].id;
		nc.second = C[i].second;
		nc.anchor = C[i].anchor;
		Eigen::Matrix3d R = C[i].R;
		Eigen::Vector3d t = C[i].t;
		Eigen::Vector3d c = -R.transpose() * t;
		Eigen::Vector3d p = C[i].center;

		for(unsigned int j=0;j<tr.size();j++)
		{
			R = R * tr[j].R.transpose();
			c = tr[j].s * tr[j].R * c + tr[j].o;
			p = tr[j].s * tr[j].R * p + tr[j].o;
		}
		nc.R = R;
		nc.t = t;
		nc.center = p;
		nc.size_x = C[i].size_x;
		nc.size_y = C[i].size_y;
		nc.f = C[i].f;
		nc.px = C[i].px;
		nc.py = C[i].py;
		nc.inl = C[i].inl;
		ret.push_back(nc);
	}
	return ret;
}

std::vector<std::vector<motion_t>> ransac_motions(std::vector<motion_t> G, std::vector<cam_s> C, double sigma, double pd, double thr2)
{
	std::vector<std::vector<motion_t>> D;


	vector<vector<basis_t>> ret;
	vector<bool> grouped = vector<bool>(G.size(), 0);
	int cl = 0;

	while(1)
	{
		cl++;
		cout << "CLUSTER " << cl << "\n";
		vector<bool> best = vector<bool>(G.size(), 0);
		int best_count = 0;
		for(unsigned int i=0;i<G.size();i++)
		{
			if(grouped[i]) continue;
			Eigen::Matrix3d R1 = G[i].A;
			Eigen::Vector3d t1 = G[i].t;

			//check the inliers
			vector<bool> inliers = vector<bool>(G.size(), 0);
			int count = 0;
			for(unsigned int j=0;j<G.size();j++)
			{
				if(grouped[j]) continue;
				Eigen::Matrix3d R2 = G[j].A;
				Eigen::Vector3d t2 = G[j].t;

				Eigen::Matrix3d R3 = R2 * R1.transpose();
				Eigen::Vector3d k3 = r2a(R3);
				double n = k3.norm();
				if(n > sigma * 0.0247)
					continue;

				//check translation
				Eigen::Vector3d t3 = t2 - t1;
				double n3 = t3.norm();
				//cout << n3 << " " << pd << "\n";

				if(n3 <= 0.015 * pd * thr2)
				{
					inliers[j] = 1;
					count++;
				}
			}
			if(count > best_count)
			{
				best = inliers;
				best_count = count;
			}
		}

		cout << "size " << best_count << "\n";
		if(best_count == 0)
			break;

		vector<motion_t> ND = vector<motion_t>(best_count);
		int j=0;
		for(unsigned int i=0;i<G.size();i++)
		{
			if(best[i])
			{
				ND[j] = G[i];
				j++;
				grouped[i] = 1;
			}
		}
		D.push_back(ND);
		if(G.size() == (unsigned int)best_count) break;
	}

	return D;
}

std::vector<std::vector<std::vector<motion_t>>> divide_motions(std::vector<motion_t> B, std::vector<cam_s> C, double sigma, double pd, double thr2)
{
	std::vector<std::vector<std::vector<motion_t>>> ret;
	cout << "Clustering motions\n";
	int n=0;
	vector<bool> grouped = vector<bool>(B.size(), 0);
	while(1)
	{
		n++;
		cout << "Motion " << n << "\n";
		vector<motion_t> GR;
		int init = -1;
		int final = -1;
		//find the group corresponding to the new motion
		for(unsigned int i=0;i<B.size();i++)
		{
			if(GR.size() == 0 && !grouped[i])
			{
				init = B[i].init;
				final = B[i].final;
				grouped[i] = 1;
				motion_t nb = B[i];
				GR.push_back(nb);
			}
			else if(!grouped[i] && ( (B[i].init == init && B[i].final == final ) || (B[i].init == final && B[i].final == init) ))
			{
				grouped[i] = 1;
				motion_t nb = B[i];
				GR.push_back(nb);
			}
		}
		if(GR.size() == 0)
			break;
		cout << GR.size() << "\n";

		//cluster the camera pairs in the group
		vector<vector<motion_t>> CL = ransac_motions(GR, C, sigma, pd, thr2);
		ret.push_back(CL);
	}
	
	return ret;
}

std::vector<std::vector<clust_m>> meanclust_motions(std::vector<std::vector<std::vector<motion_t>>> C)
{
	std::vector<std::vector<clust_m>> D;

	for(unsigned int i=0;i<C.size();i++)
	{
		std::vector<clust_m> ND;
		unsigned int max_size = 0;
		for(unsigned int j=0;j<C[i].size();j++)
		{
			if(C[i][j].size() > max_size)
				max_size = C[i][j].size();
		}
		for(unsigned int j=0;j<C[i].size();j++)
		{
			//TODO
			//experimental
			if((C[i][j].size() < 2 && max_size >= 2) || (C[i][j].size() < 3 && max_size >= 6))
			{
				continue;
			}

			std::vector<Eigen::Vector3d> R;
			std::vector<Eigen::Vector3d> t;
			for(unsigned int k=0;k<C[i][j].size();k++)
			{
				R.push_back(r2a(C[i][j][k].A));
				t.push_back(C[i][j][k].t);
			}
			clust_m nm;
			nm.size = C[i][j].size();

			if(C[i][j][0].init < C[i][j][0].final)
			{
				nm.init = C[i][j][0].init;
				nm.final = C[i][j][0].final;
			}
			else
			{
				nm.final = C[i][j][0].init;
				nm.init = C[i][j][0].final;
			}
			nm.R = a2r(median(R));
			nm.t = median(t);

			ND.push_back(nm);
			
		}
		D.push_back(ND);
	}

	return D;
}

std::pair<std::vector<std::vector<std::vector<motion_t>>>, std::vector<std::vector<clust_m>>> remove_identity(std::vector<std::vector<clust_m>> C, std::vector<std::vector<std::vector<motion_t>>> CL , double pd, double thr1, double thr2)
{
	cout << "REMOVING IDENTITY\n";
	std::vector<std::vector<std::vector<motion_t>>> NCL;
	std::vector<std::vector<clust_m>> M;

	for(unsigned int i=0;i<C.size();i++)
	{
		vector<clust_m> C1;
		vector<vector<motion_t>> C2;
		for(unsigned int j=0;j<C[i].size();j++)
		{
			//cout << r2a(C[i][j].R).norm() << " " << C[i][j].t.norm() << "\n";
			if(r2a(C[i][j].R).norm() < thr1 * 0.0247)
			{
				if(C[i][j].t.norm() >= pd * 0.025)
				{
					C1.push_back(C[i][j]);
					C2.push_back(CL[i][j]);	
				}
				//else cout << "NK\n";
			}
			else
			{
				C1.push_back(C[i][j]);
				C2.push_back(CL[i][j]);
			}
		}
		NCL.push_back(C2);
		M.push_back(C1);
	}
	
	std::pair<std::vector<std::vector<std::vector<motion_t>>>, std::vector<std::vector<clust_m>>> ret;
	ret.first = NCL;
	ret.second = M;
	return ret;
}

std::vector<imgs_s> load_imgs(int takes)
{
	//HERE_
	cout << "LOADING IMAGES\n";
	std::vector<imgs_s> ret;
	for(int i=1;i<=takes;i++)
	{
		imgs_s img;
		cout << "Take " << i << "\n";
		bool succ = 0;
		string line;
		string p2 = to_string(i) + "/images.txt";
		ifstream imgs;
		imgs.open(p2);

		for(int j=0;j<4;j++)
			getline(imgs,line);

		int pos = 0;
		while(getline(imgs, line))
		{
			int id;
			istringstream str(line);
			str >> id;
			double num;
			for(int j=0;j<8;j++)
				str >> num;
			string name;
			str >> name;
			img.id.push_back(id);
			img.map[id] = pos;

			vector<int> n_obs;
			vector<int> n_obs2;
			vector<Eigen::Vector2d> n_feat;
			getline(imgs,line);
			istringstream str2(line);

			while(str2 >> num)
			{
				double num2;
				str2 >> num2;
				Eigen::Vector2d cf;
				cf(0) = num;
				cf(1) = num2;
				n_feat.push_back(cf);
				int pnt_id;
				str2 >> pnt_id;
				n_obs2.push_back(pnt_id);
				succ = 1;
				if (pnt_id >= 0)
				{
					//cout << pnt_id << "\n";
					n_obs.push_back(pnt_id);
				}
			}
			//std::sort(n_obs.begin(), n_obs.end());
			img.obs.push_back(n_obs);
			img.features.push_back(n_feat);
			img.obs_all.push_back(n_obs2);
			pos++;
		}
		ret.push_back(img);
		cout << succ << "\n";
	}

	return ret;
}

std::vector<pnts_s> load_pnts(int takes)
{
	cout << "LOADING POINTS\n";
	std::vector<pnts_s> ret;

	for(int i=1;i<=takes;i++)
	{
		cout << "Take " << i << "\n";
		pnts_s pnt;

		ifstream pnts;
		string p1 = to_string(i) + "/points3D.txt";
		pnts.open(p1);
		string line;
		for(int j=0;j<3;j++)
			getline(pnts,line);

		while(getline(pnts,line))
		{
			//cout << line << "\n";
			istringstream str(line);
			int id;
			str >> id;
			pnt.ID.push_back(id);
			double x;
			double y;
			double z;
			int r;
			int g;
			int b;
			str >> x;
			str >> y;
			str >> z;
			str >> r;
			str >> g;
			str >> b;
			Eigen::Vector3d point;
			point(0) = x;
			point(1) = y;
			point(2) = z;
			pnt.points.push_back(point);

			Eigen::Vector3i color;
			color(0) = r;
			color(1) = g;
			color(2) = b;
			pnt.color.push_back(color);
			
			pnt.ID_map[id] = pnt.ID.size()-1;
			//if(id < 10)
			//	cout << id << "\n";
		}
		
		pnts.close();
		ret.push_back(pnt);
	}

	return ret;
}

std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::unordered_map<int, int>>> load_tracks(std::vector<pnts_s> P, int takes)
{
	cout << "LOADING TRACKS\n";
	std::vector<std::vector<std::pair<int, int>>> T;
	std::vector<std::unordered_map<int, int>> P2T(takes);
	/*for(int i=0;i<takes;i++)
	{
		P2T[i] = std::vector<int>(P[i].ID.size());
	}*/

	ifstream ft;
	ft.open("tracks.txt");
	string line;
	while(getline(ft, line))
	{
		std::vector<std::pair<int, int>> track;
		int t;
		int p;	
		istringstream str(line);
		while(str >> t)
		{
			str >> p;
			std::pair<int, int> np;
			np.first = t;
			np.second = p;
			track.push_back(np);
			P2T[t-1][P[t-1].ID_map[p]] = T.size();
		}
		T.push_back(track);
	}
	ft.close();

	ifstream fg;
	fg.open("graph.txt");
	while(getline(fg, line))
	{
		std::vector<std::pair<int, int>> track;
		int t;
		int p;	
		istringstream str(line);
		while(str >> t)
		{
			str >> p;
			std::pair<int, int> np;
			np.first = t;
			np.second = p;
			track.push_back(np);
			//P2T[t-1][P[t-1].ID_map[p]] = T.size();
			//cout << t << " " << p << " " ;
		}
		//T.push_back(track);
		if(!track.size()) continue;
		//cout << "\n";

		//prepare the laplacian of the graph
		Eigen::MatrixXd G(track.size(), track.size());
		for(unsigned int i=0;i<track.size();i++)
		{
			for(unsigned int j=0;j<track.size();j++)
			{
				double nn;
				fg >> nn;
				G(i,j) = nn;
			}
		}
		Eigen::MatrixXd L(track.size(), track.size());
		L = -1 * G;
		for(unsigned int i=0;i<track.size();i++)
		{
			double sum = 0;
			for(unsigned int j=0;j<track.size();j++)
			{
				if(i==j) continue;
				sum += G(i,j);
			}
			L(i,i) = sum;
		}
		Eigen::MatrixXd L1(track.size(), track.size());
		for(unsigned int i=0;i<track.size();i++)
		{
			for(unsigned int j=0;j<track.size();j++)
			{
				L1(i,j) = L(i,j)/(sqrt(L(i,i))*sqrt(L(j,j)));
			}
		}
		Eigen::EigenSolver<Eigen::MatrixXd> es(L1);
		Eigen::MatrixXcd V = es.eigenvectors();
		//cout << "F" << "\n";
		int last_cons = 0;
		//perform the kmeans
		for(unsigned int i=2;i<track.size();i++)
		{
			//cout << i << "\n";
			int dim = i;
			if(i>10) dim = 10;
			
			vector<Eigen::VectorXd> a;
			for(unsigned int j=0;j<track.size();j++)
			{
				Eigen::VectorXd nv(i);
				for(unsigned int k=0;k<i;k++)
				{
					nv(k) = V(j,k).real();
				}
				a.push_back(nv);
			}
		
			std::vector<int> c = kmeans(a, i, dim);

			//check the consistency
			bool consistent = 1;
			int cons = 0;
			int incons = 0;
			for(unsigned int j=0;j<i;j++)
			{
				std::vector<std::pair<int, int>> CC;
				for(unsigned int k=0;k<track.size();k++)
				{
					if((unsigned int)c[k] == j)
					{
						CC.push_back(track[k]);
					}
				}
				bool con = 1;
				for(unsigned int k=0;k<CC.size();k++)
				{
					for(unsigned int l=k+1;l<CC.size();l++)
					{
						if(CC[k].first == CC[l].first)
						{
							consistent = 0;
							con = 0;
							break;
						}
					}	
					//if(!consistent) break;
				}
				if(con) cons++;
				else	incons++;
				//if(!consistent) break;
			}

			//cout << cons << " " << incons << "\n";

			if(consistent || i==(track.size()-1))
			{
				for(unsigned int j=0;j<i;j++)
				{
					std::vector<std::pair<int, int>> CCC;
					for(unsigned int k=0;k<track.size();k++)
					{
						if((unsigned int)c[k] == j)
						{
							CCC.push_back(track[k]);
							int t = track[k].first;
							int p = track[k].second;
							P2T[t-1][P[t-1].ID_map[p]] = T.size();
						}
					}
					T.push_back(CCC);
				}
				break;
			}
			else if((last_cons < cons && i >= 10) || i >= 20)
			{
				//take the consistent clusters together and the inconsistent ones as single points
				for(unsigned int j=0;j<i;j++)
				{
					std::vector<std::pair<int, int>> CC;
					for(unsigned int k=0;k<track.size();k++)
					{
						if((unsigned int)c[k] == j)
						{
							CC.push_back(track[k]);
						}
					}
					bool con = 1;
					for(unsigned int k=0;k<CC.size();k++)
					{
						for(unsigned int l=k+1;l<CC.size();l++)
						{
							if(CC[k].first == CC[l].first)
							{
								con = 0;
								break;
							}
						}
					}
					//TODO
					//this is only an experimental feature 
					if(con)
					{
						//cout << "C " << CC.size() << "\n";
						for(unsigned int k=0;k<CC.size();k++)
						{
							int t = CC[k].first;
							int p = CC[k].second;
							P2T[t-1][P[t-1].ID_map[p]] = T.size();
						}
						T.push_back(CC);
					}
					else
					{
						//cout << "I " << CC.size() << "\n";
						for(unsigned int k=0;k<CC.size();k++)
						{
							std::vector<std::pair<int, int>> CCC;
							CCC.push_back(CC[k]);
							int t = CC[k].first;
							int p = CC[k].second;
							P2T[t-1][P[t-1].ID_map[p]] = T.size();
							T.push_back(CCC);
						}
					}
				}

				
				break;
			}
		}
		//getline(fg, line);
	}
	fg.close();
	cout << "TRACKS LOADED\n";

	std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::unordered_map<int, int>>> ret;
	ret.first = T;
	ret.second = P2T;
	return ret;
}

std::vector<std::vector<int>> observed_tracks(std::vector<cam_s> C, std::vector<imgs_s> I, std::vector<pnts_s> P, std::vector<std::unordered_map<int, int>> P2T)
{
	cout << "FINDING TRACKS OBSERVED BY THE CAMERAS\n";
	std::vector<std::vector<int>> ret;
	for(unsigned int i=0;i<C.size();i++)
	{
		std::cout << i << "\n";
		if(C[i].anchor == C[i].second)
		{
			std::vector<int> tr;
			ret.push_back(tr);
			continue;
		}
		int rec = C[i].anchor;
		int cam = C[i].id_2;
		std::cout << rec << " " << cam << "\n";
		int pos = I[rec-1].map[cam];
		//std::cout << "POS " << pos << "\n";
		//std::cout << I[rec-1].obs.size() << "\n";
		if(pos >= I[rec-1].obs.size())
		{
			std::vector<int> tr;
			ret.push_back(tr);
			continue;
		}
		std::vector<int> obs = I[rec-1].obs[pos];
		//cout << pos << " " << I[rec-1].obs_all.size() << " " << I[rec-1].obs.size() << "\n";
		//std::cout << "OBS " << pos << "\n";
		std::vector<int> tr(obs.size());
		for(unsigned int j=0;j<obs.size();j++)
		{
			int pnt_p = P[rec-1].ID_map.at(obs[j]);
			int track = P2T[rec-1].at(pnt_p);
			tr[j] = track;
		}
		std::sort(tr.begin(), tr.end());
		ret.push_back(tr);
	}
	cout << "FOUND\n";

	return ret;
}

std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> observed_tracks_2(std::vector<cam_s> C, std::vector<imgs_s> I, std::vector<pnts_s> P, std::vector<std::unordered_map<int, int>> P2T)
{
	std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> ret;
	for(unsigned int i=0;i<C.size();i++)
	{		
		//cout << "A\n";
		int rec = C[i].anchor;
		int cam = C[i].id_2;
		//cout << rec << " " << cam << "\n";
		int pos = I[rec-1].map[cam];
		if(!I[rec-1].obs_all.size())
		{
			std::pair<std::vector<int>, std::vector<Eigen::Vector2d>> tr;
			ret.push_back(tr);
			continue;
		}
		//cout << pos << " " << I[rec-1].obs_all.size() << " " << I[rec-1].obs.size() << "\n";
		std::vector<int> obs = I[rec-1].obs_all[pos];
		//cout << "T1\n";
		std::vector<Eigen::Vector2d> feat = I[rec-1].features[pos];
		//cout << "T1\n";
		std::vector<int> tr(obs.size());
		//cout << "B\n";
		for(unsigned int j=0;j<obs.size();j++)
		{
			if(obs[j] >= 0)
			{
				int pnt_p = P[rec-1].ID_map.at(obs[j]);
				int track = P2T[rec-1].at(pnt_p);
				tr[j] = track;
			}
			else
			{
				tr[j] = -1;
			}
		}
		//cout << "C\n";
		std::pair<std::vector<int>, std::vector<Eigen::Vector2d>> nr;
		nr.first = tr;
		nr.second = feat;
		ret.push_back(nr);
		//cout << "D\n";
	}

	return ret;
}

std::vector<std::vector<pair_t>> group_ot(std::vector<std::vector<int>> O, std::vector<cam_s> C, int takes)
{
	cout << "CREATING PAIRS OF CAMERAS\n";
	
	int cur_rec = -1;
	std::vector<std::vector<pair_t>> G(takes);
	for(int i=0;i<takes;i++)	
	{
		std::vector<pair_t> cG;
		G[i] = cG;
	}
	
	for(int i=0;i<C.size();i++)
	{
		if(C[i].anchor != cur_rec)
		{
			cur_rec = C[i].anchor;
		}

		if(C[i].anchor == C[i].second) continue;

		int cur_cam = C[i].id;
		for(int j=i+1;j<C.size();j++)
		{
			if(cur_cam != C[j].id || C[j].anchor != C[i].anchor)
				break;

			std::pair<std::vector<int>, std::vector<int>> cur_div;
			cur_div.first = O[i];
			cur_div.second = O[j];
			std::vector<int> cur_label(2);
			cur_label[0] = i;
			cur_label[1] = j;
			pair_t cn;
			cn.div = cur_div;
			cn.label = cur_label;
			cn.size = 1;

			G[cur_rec-1].push_back(cn);
		}
	}
	return G;
}

//std::vector<std::vector<pair_t>> filter_groups(std::vector<std::vector<pair_t>> G, std::vector<std::vector<std::pair<int, int>>> T)
std::vector<std::vector<pair_t>> filter_groups(std::vector<std::vector<pair_t>> G, std::vector<std::vector<std::pair<int, int>>> T, std::vector<pnts_s> P)
{
	std::cout << "FILTER GROUPS\n";
	std::cout << G.size() << "\n";
	std::vector<std::vector<pair_t>> ret(G.size());
	for(int i=0;i<G.size();i++)
	{
		std::vector<pair_t> nv(G[i].size());
		for(int j=0;j<G[i].size();j++)
		{
			pair_t np;
			np.label = G[i][j].label;
			np.size = G[i][j].size;
			std::vector<int> div1;
			std::vector<int> div2;
			

			//find the nearest 5 classified neighbors for every point
			for(int k=0;k<G[i][j].div.first.size();k++)
			{
				bool ok = 1;
				int track_ix = G[i][j].div.first[k];
				for(int a=0;a<T[track_ix].size();a++)
				{
				
					if(T[track_ix][a].first == i+1)
					{
						int pos = T[track_ix][a].second;
						int pos2 = P[i].ID_map[pos];
						Eigen::Vector3d pnt = P[i].points[pos2];

						//try all other points in the division and find the Euclidean distances from the current point
						std::vector<double> dists1;
						for(int l=0;l<G[i][j].div.first.size();l++)
						{
							if(l==k) continue;
							int track_ix2 = G[i][j].div.first[l];
							for(int b=0;b<T[track_ix2].size();b++)
							{
								if(T[track_ix2][b].first == i+1)
								{
									int pos_ = T[track_ix2][b].second;
									int pos2_ = P[i].ID_map[pos_];
									Eigen::Vector3d pnt_ = P[i].points[pos2_];

									//find the Euclidean distance
									double norm = (pnt-pnt_).norm();
									dists1.push_back(norm);
								}
							}
						}
						std::sort(dists1.begin(),dists1.end());
						std::vector<double> dists2;
						for(int l=0;l<G[i][j].div.second.size();l++)
						{
							//if(l==k) continue;
							int track_ix2 = G[i][j].div.second[l];
							for(int b=0;b<T[track_ix2].size();b++)
							{
								if(T[track_ix2][b].first == i+1)
								{
									int pos_ = T[track_ix2][b].second;
									int pos2_ = P[i].ID_map[pos_];
									Eigen::Vector3d pnt_ = P[i].points[pos2_];

									//find the Euclidean distance
									double norm = (pnt-pnt_).norm();
									dists2.push_back(norm);
								}
							}
						}
						std::sort(dists2.begin(),dists2.end());
						if(dists1.size()>0 && dists2.size()>4)
							if(dists1[0] > dists2[4])
								ok = 0;
					}
					//int pos2 = P[i].ID_map[pos];
					//Eigen::Vector3d pnt = P[i].points[pos2];
					//std::cout << pnt.transpose() << "\n";
				}
				if(ok)
					div1.push_back(track_ix);
			}
			for(int k=0;k<G[i][j].div.second.size();k++)
			{
				bool ok = 1;
				int track_ix = G[i][j].div.second[k];
				for(int a=0;a<T[track_ix].size();a++)
				{
					int pos = T[track_ix][a].second;
					if(T[track_ix][a].first == i+1)
					{
						int pos = T[track_ix][a].second;
						int pos2 = P[i].ID_map[pos];
						Eigen::Vector3d pnt = P[i].points[pos2];

						//try all other points in the division and find the Euclidean distances from the current point
						std::vector<double> dists1;
						for(int l=0;l<G[i][j].div.first.size();l++)
						{
							//if(l==k) continue;
							int track_ix2 = G[i][j].div.first[l];
							for(int b=0;b<T[track_ix2].size();b++)
							{
								if(T[track_ix2][b].first == i+1)
								{
									int pos_ = T[track_ix2][b].second;
									int pos2_ = P[i].ID_map[pos_];
									Eigen::Vector3d pnt_ = P[i].points[pos2_];

									//find the Euclidean distance
									double norm = (pnt-pnt_).norm();
									dists1.push_back(norm);
								}
							}
						}
						std::sort(dists1.begin(),dists1.end());
						std::vector<double> dists2;
						for(int l=0;l<G[i][j].div.second.size();l++)
						{
							if(l==k) continue;
							int track_ix2 = G[i][j].div.second[l];
							for(int b=0;b<T[track_ix2].size();b++)
							{
								if(T[track_ix2][b].first == i+1)
								{
									int pos_ = T[track_ix2][b].second;
									int pos2_ = P[i].ID_map[pos_];
									Eigen::Vector3d pnt_ = P[i].points[pos2_];

									//find the Euclidean distance
									double norm = (pnt-pnt_).norm();
									dists2.push_back(norm);
								}
							}
						}
						std::sort(dists2.begin(),dists2.end());
						if(dists1.size()>4 && dists2.size()>0)
							if(dists2[0] > dists1[4])
								ok = 0;
					}
					//int pos2 = P[i].ID_map[pos];
					//Eigen::Vector3d pnt = P[i].points[pos2];
					//std::cout << pnt.transpose() << "\n";
				}
				if(ok)
					div2.push_back(track_ix);
			}
			std::pair<std::vector<int>, std::vector<int>> div;
			div.first = div1;
			div.second = div2;
			np.div = div;
			nv[j] = np;
			std::cout << i << " " << div1.size() << " " << div2.size() << "\n";
		}
		ret[i] = nv;
	}
	return ret;
}

int intersect_size(std::vector<int> O1, std::vector<int> O2)
{
	int sz = 0;
	int pos1 = 0;
	int pos2 = 0;
	while(pos1 < O1.size() && pos2 < O2.size())
	{
		if(O1[pos1] == O2[pos2])
		{
			++pos1;
			++pos2;
			++sz;
		}
		else if(O1[pos1] < O2[pos2])
		{
			++pos1;
		}
		else
		{
			++pos2;
		}
	}
	return sz;
}

std::vector<int> join(std::vector<int> O1, std::vector<int> O2)
{
	int sz = 0;
	int pos1 = 0;
	int pos2 = 0;
	std::vector<int> ret;
	while(pos1 < O1.size() && pos2 < O2.size())
	{
		if(O1[pos1] == O2[pos2])
		{
			ret.push_back(O1[pos1]);
			++pos1;
			++pos2;
			++sz;
		}
		else if(O1[pos1] < O2[pos2])
		{
			ret.push_back(O1[pos1]);
			++pos1;
		}
		else
		{
			ret.push_back(O2[pos2]);
			++pos2;
		}
	}
	return ret;
}

std::vector<std::vector<pair_t>> linkage(std::vector<std::vector<pair_t>> G)
{
	cout << "GROUPING THE PAIRS OF CAMERAS\n";
	std::vector<std::vector<pair_t>> NG(G.size());
	const double block = 0.02;

	for(int i=0;i<G.size();i++)
	{
		cout << i << "\n";
		std::vector<pair_t> CG = G[i];

		//initialize the similarity matrices
		std::vector<std::vector<int>> sim11(CG.size());
		std::vector<std::vector<int>> sim12(CG.size());
		std::vector<std::vector<int>> sim21(CG.size());
		std::vector<std::vector<int>> sim22(CG.size());
		for(int j=0;j<CG.size();j++)
		{
			std::vector<int> csim11(CG.size());
			sim11[j] = csim11;

			std::vector<int> csim12(CG.size());
			sim12[j] = csim12;

			std::vector<int> csim21(CG.size());
			sim21[j] = csim21;

			std::vector<int> csim22(CG.size());
			sim22[j] = csim22;
			
			for(int k=0;k<CG.size();k++)
			{
				sim11[j][k] = 0;
				sim12[j][k] = 0;
				sim21[j][k] = 0;
				sim22[j][k] = 0;
			}
		}

		//cout << "A\n";

		//find the initial similarity
		for(int a=0;a<CG.size();a++)
		{
			for(int b=(a+1);b<CG.size();b++)
			{
				//find the intersection of the observations of pairs a and b
				int c11 = intersect_size(CG[a].div.first, CG[b].div.first);
				int c12 = intersect_size(CG[a].div.first, CG[b].div.second);
				int c21 = intersect_size(CG[a].div.second, CG[b].div.first);
				int c22 = intersect_size(CG[a].div.second, CG[b].div.second);

				sim11[a][b] = c11;
				sim12[a][b] = c12;
				sim21[a][b] = c21;
				sim22[a][b] = c22;
			}
		}

		//cout << "B\n";

		//perform the linkage
		while(1)
		{
			//find the best pair to be joined
			int max = 0;
			int best_a = 0;
			int best_b = 0;
			bool swapped = 0;
			for(int a=0;a<CG.size();a++)
			{
				if(!CG[a].size) continue;
				int sz1 = CG[a].div.first.size() + CG[a].div.second.size();
				for(int b=(a+1);b<CG.size();b++)
				{
					if(!CG[b].size) continue;
					int sz2 = CG[b].div.first.size() + CG[b].div.second.size();
					//TODO change the threshold for the sim2 to join all relevant pairs
					if(sim11[a][b] > max && sim22[a][b] > max && ( (sim12[a][b] <= 2 && sim21[a][b] <= 2) || (sim12[a][b] <= block*CG[a].div.first.size() && sim12[a][b] <= block*CG[b].div.second.size() && sim21[a][b] <= block*CG[a].div.second.size() && sim21[a][b] <= block*CG[b].div.first.size()) ))
					{
						if(sim11[a][b] < sim22[a][b])
							max = sim11[a][b];
						else
							max = sim22[a][b];
						swapped = 0;
						best_a = a;
						best_b = b;
					}
					else if( sim12[a][b] > max && sim21[a][b] > max && ( (sim11[a][b] <= 2 && sim22[a][b] <= 2 ) || (sim11[a][b] <= block*CG[a].div.first.size() && sim11[a][b] <= block*CG[b].div.first.size() && sim22[a][b] <= block*CG[a].div.second.size() && sim22[a][b] <= block*CG[b].div.second.size()) ) )
					{
						if(sim12[a][b] < sim21[a][b])
							max = sim12[a][b];
						else
							max = sim21[a][b];
						swapped = 1;
						best_a = a;
						best_b = b;
					}
				}
			}

			//cout << "C\n";

			if(!max) break;

			//join the pair
			if(!swapped)
			{
				vector<int> div1 = join(CG[best_a].div.first, CG[best_b].div.first);
				vector<int> div2 = join(CG[best_a].div.second, CG[best_b].div.second);
				CG[best_a].div.first = div1;
				CG[best_a].div.second = div2;
				vector<int> empty1;
				vector<int> empty2;
				CG[best_b].div.first = empty1;
				CG[best_b].div.second = empty2;
				CG[best_a].size = CG[best_a].size + CG[best_b].size;
				CG[best_b].size = 0;

				for(int a=0;a<CG[best_b].label.size();a++)
					CG[best_a].label.push_back(CG[best_b].label[a]);
			}
			else
			{
				vector<int> div1 = join(CG[best_a].div.first, CG[best_b].div.second);
				vector<int> div2 = join(CG[best_a].div.second, CG[best_b].div.first);
				CG[best_a].div.first = div1;
				CG[best_a].div.second = div2;
				vector<int> empty1;
				vector<int> empty2;
				CG[best_b].div.first = empty1;
				CG[best_b].div.second = empty2;

				for(int a=0;a<CG[best_b].label.size();a++)
				{
					if(a%2)
						CG[best_a].label.push_back(CG[best_b].label[a-1]);
					else
						CG[best_a].label.push_back(CG[best_b].label[a+1]);
				}
			}

			//cout << "D\n";

			//change the values in the similarity matrices
			for(int a=0;a<CG.size();a++)
			{
				if(a==best_a) continue;
				int c11 = intersect_size(CG[a].div.first, CG[best_a].div.first);
				int c12 = intersect_size(CG[a].div.first, CG[best_a].div.second);
				int c21 = intersect_size(CG[a].div.second, CG[best_a].div.first);
				int c22 = intersect_size(CG[a].div.second, CG[best_a].div.second);

				if(a<best_a)
				{
					sim11[a][best_a] = c11;
					sim12[a][best_a] = c12;
					sim21[a][best_a] = c21;
					sim22[a][best_a] = c22;
				}
				else
				{
					sim11[best_a][a] = c11;
					sim12[best_a][a] = c12;
					sim21[best_a][a] = c21;
					sim22[best_a][a] = c22;
				}
			}
			for(int a=0;a<CG.size();a++)
			{
				if(a<best_b)
				{
					sim11[a][best_b] = 0;
					sim12[a][best_b] = 0;
					sim21[a][best_b] = 0;
					sim22[a][best_b] = 0;
				}
				else
				{
					sim11[best_b][a] = 0;
					sim12[best_b][a] = 0;
					sim21[best_b][a] = 0;
					sim22[best_b][a] = 0;
				}
			}

			//cout << "E\n";
		}
		NG[i] = CG;
	}

	return NG;
}

std::vector<int> merge(std::vector<int> O1, std::vector<int> O2)
{
	std::vector<int> O;
	unsigned int p1 = 0;
	unsigned int p2 = 0;
	while(1)
	{
		if(p1 >= O1.size() && p2 >= O2.size())
		{
			break;
		}
		else if(p1 >= O1.size())
		{
			for(;p2<O2.size();p2++)
			{
				O.push_back(O2[p2]);
			}
			break;
		}
		else if(p2 >= O2.size())
		{
			for(;p1<O1.size();p1++)
			{
				O.push_back(O1[p1]);
			}
			break;
		}

		if(O1[p1] < O2[p2])
		{
			O.push_back(O1[p1]);
			p1++;
		}
		else if(O1[p1] > O2[p2])
		{
			O.push_back(O2[p2]);
			p2++;
		}
		else
		{
			O.push_back(O1[p1]);
			p1++;
			p2++;
		}
	}
	return O;
}

int common(std::vector<int> O1, std::vector<int> O2)
{
	int ret = 0;
	unsigned int p1 = 0;
	unsigned int p2 = 0;
	while(1)
	{
		if(p1 >= O1.size() || p2 >= O2.size())
		{
			break;
		}

		if(O1[p1] < O2[p2])
		{
			p1++;
		}
		else if(O1[p1] > O2[p2])
		{
			p2++;
		}
		else
		{
			ret++;
			p1++;
			p2++;
		}
	}
	return ret;
}

std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> observed_by_cluster(std::vector<std::vector<std::vector<motion_t>>> CL, std::vector<std::vector<int>> O)
{
	std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> ret(CL.size());
	for(unsigned int i=0;i<CL.size();i++)
	{
		std::vector<std::vector<motion_t>> CL2 = CL[i];
		std::vector<std::pair<std::vector<int>, std::vector<int>>> D2(CL2.size());
		for(unsigned int j=0;j<CL2.size();j++)
		{
			std::vector<motion_t> CL3 = CL2[j];
			std::pair<std::vector<int>, std::vector<int>> obs;
			vector<int> obs1;
			vector<int> obs2;
			//cout << "NC\n";
			for(unsigned int k=0;k<CL3.size();k++)
			{
				int cam1 = CL3[k].c1;
				std::vector<int> o1 = O[cam1];
				int cam2 = CL3[k].c2;
				std::vector<int> o2 = O[cam2];
				//cout << o1.size() << " " << o2.size() << " " << common(o1, o2) << "\n";
				if(CL3[k].init < CL3[k].final)
				{
					obs1 = merge(obs1, o1);
					obs2 = merge(obs2, o2);
				}
				else
				{
					obs1 = merge(obs1, o2);
					obs2 = merge(obs2, o1);
				}
			}
			obs.first = obs1;
			obs.second = obs2;
			D2[j] = obs;
		}
		ret[i] = D2;
	}
	return ret;
}

std::vector<std::vector<std::pair<int, int>>> chordal_completion( std::vector<std::vector<clust_m>> CL, std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> O, double thr1, double thr2, double pd )
{
	std::vector<std::pair<int, int>> num2cl;
	std::vector<clust_m> C;
	for(unsigned int i=0;i<CL.size();i++)
	{
		std::vector<clust_m> T = CL[i];
		for(unsigned int j=0;j<T.size();j++)
		{
			C.push_back(T[j]);
			pair<int, int> np;
			np.first = i;
			np.second = j;
			num2cl.push_back(np);
		}
	}

	//create the structure of the edges
	vector<edge_s> E;
	Eigen::MatrixXi C2E = Eigen::MatrixXi::Ones(C.size(), C.size());
	C2E = -C2E;

	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=i+1;j<C.size();j++)
		{
			if(C[i].init == C[j].init && C[i].final == C[j].final)
				continue;
			C2E(i,j) = E.size();
			C2E(j,i) = E.size();
			edge_s ne;
			if( (C[i].init == C[j].init) || (C[i].init == C[j].final) || (C[i].final == C[j].init) || (C[i].final == C[j].final) )
			{
				
				ne.cl1 = i;
				ne.cl2 = j;
				ne.pred = -1;
				ne.grade = 0;
				ne.rc = 0;
				ne.tc = 0;
				
			}
			else
			{
				//cout << "NF\n";
				ne.cl1 = i;
				ne.cl2 = j;
				ne.pred = -1;
				ne.grade = -1;
				ne.rc = 0;
				ne.tc = 0;
			}
			std::vector<int> o11 = O[num2cl[i].first][num2cl[i].second].first;
			std::vector<int> o21 = O[num2cl[j].first][num2cl[j].second].first;

			std::vector<int> o12 = O[num2cl[i].first][num2cl[i].second].second;
			std::vector<int> o22 = O[num2cl[j].first][num2cl[j].second].second;
			int com1 = common(o11, o21);
			int com2 = common(o12, o22);
			int com3 = common(o11, o22);
			int com4 = common(o12, o21);
			//cout << i << " " << j << " " << " " << common << "\n";
			ne.common = com1+com2-com3-com4;
			if(com3+com4==0)
				ne.ratio = INFINITY;
			else
				ne.ratio = (double)(com1+com2)/(double)(com3+com4);
			//cout << num2cl[i].first << " " << num2cl[i].second << " " << num2cl[j].first << " " << num2cl[j].second << " " << ne.common << " " << ne.ratio << "\n";
			E.push_back(ne);
		}
	}

	//create the structure of the cycles
	vector<pair<Eigen::Vector3i, int>> CY;
	vector<pair<Eigen::Vector3i, int>> Q;
	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=0;j<C.size();j++)
		{
			if(C[i].init == C[j].init && C[i].final == C[j].final) continue;
			for(unsigned int k=0;k<C.size();k++)
			{
				if(C[i].init == C[k].init && C[i].final == C[k].final) continue;
				if(C[j].init == C[k].init && C[j].final == C[k].final) continue;
				//if(C[i].final == C[j].init && C[j].final == C[k].final && C[i].init == C[k].init)
				{
					Eigen::Vector3i ijk;
					ijk(0) = i;
					ijk(1) = j;
					ijk(2) = k;
					pair<Eigen::Vector3i, int> nc;
					nc.first = ijk;
					nc.second = 0;
					pair<Eigen::Vector3i, int> nq;
					nq.first = ijk;
					nq.second = CY.size();
					CY.push_back(nc);
					Q.push_back(nq);
				}
			}
		}
	}

	//perform the chordal completion
	vector<int> verified_edges;
	vector<int> verified_edges_2;
	while(Q.size())
	{
		vector<pair<Eigen::Vector3i, int>> NQ;
		for(unsigned int i=0;i<Q.size();i++)
		{
			pair<Eigen::Vector3i, int> cur = Q[i];
			int e1 = E[C2E(cur.first(0), cur.first(1))].grade;
			int e2 = E[C2E(cur.first(0), cur.first(2))].grade;
			int e3 = E[C2E(cur.first(1), cur.first(2))].grade;
			//if(e1==-1 || e2 == -1 || e3 == -1)
			//	cout << e1 << " " << e2 << " " << e3 << "\n";
			if((e1==-1 && e2==-1) || (e1==-1 && e3==-1) || (e2==-1 && e3==-1))
			{
				//too many edges are missing
				NQ.push_back(cur);
			}
			else if(e1 == -1)
			{
				//e1 is missing, e2, e3 are present, add a chord
				int grade;
				if(e2 > e3)
					grade = e2+1;
				else
					grade = e3+1;
				edge_s ce = E[C2E(cur.first(0), cur.first(1))];
				ce.grade = grade;
				ce.pred = cur.second;
				E[C2E(cur.first(0), cur.first(1))] = ce;
			}
			else if(e2 == -1)
			{
				//e2 is missing, e1, e3 are present, add a chord
				int grade;
				if(e1 > e3)
					grade = e1+1;
				else
					grade = e3+1;
				edge_s ce = E[C2E(cur.first(0), cur.first(2))];
				ce.grade = grade;
				ce.pred = cur.second;
				E[C2E(cur.first(0), cur.first(2))] = ce;
			}
			else if(e3 == -1)
			{
				//e3 is missing, e1, e2 are present, add a chord
				int grade;
				if(e1 > e2)
					grade = e1+1;
				else
					grade = e2+1;
				edge_s ce = E[C2E(cur.first(1), cur.first(2))];
				ce.grade = grade;
				ce.pred = cur.second;
				E[C2E(cur.first(1), cur.first(2))] = ce;
			}
			else
			{
				//the cycle is complete
				//find the original cycle and verify it
				//cout << "FOUND" << "\n";
				if(e1==0 && e2==0 && e3==0 && 0)
				{
					//original cycle, not necessary to discover
					//may be implemented alone
				}
				else
				{
					//discover the cycle with depth first search
					vector<int> edges;
					vector<int> cycles;
					cycles.push_back(cur.second);
					stack<int> st;
					st.push(C2E(cur.first(0), cur.first(2)));
					st.push(C2E(cur.first(1), cur.first(2)));
					st.push(C2E(cur.first(0), cur.first(1)));
					while(st.size())
					{
						int cur_c = st.top();
						st.pop();
						if(!E[cur_c].grade)
						{
							edges.push_back(cur_c);
						}
						else
						{
							int pred = E[cur_c].pred;
							cycles.push_back(pred);
							pair<Eigen::Vector3i, int> cycle = CY[pred];
							int ne1 = C2E(cycle.first(0), cycle.first(1));
							int ne2 = C2E(cycle.first(0), cycle.first(2));
							int ne3 = C2E(cycle.first(1), cycle.first(2));
							if(ne3 != cur_c)
								st.push(ne3);
							if(ne2 != cur_c)
								st.push(ne2);
							if(ne1 != cur_c)
								st.push(ne1);
						}
					}

					//check whether it is a cycle
					bool rep = 0;
					vector<int> se(edges.size());
					for(unsigned int el=0;el<se.size();el++)
					{
						//if the cycle is long enough, at one moment there holds true el=011
						se[el] = edges[el];
					}
					std::sort(se.begin(), se.end());
					for(unsigned int j=1;j<se.size();j++)
					{
						if(se[j-1]==se[j])
						{
							rep=1;
							break;
						}
					}
					if(!rep)
					{
						//build the cycle
						vector<int> cycle;
						vector<int> order;
						vector<bool> mark;
						vector<int> marked;
						cycle.push_back(E[edges[0]].cl1);
						cycle.push_back(E[edges[0]].cl2);
						order.push_back(edges[0]);
						mark.push_back(0);
						marked.push_back(E[edges[0]].cl1);
						vector<bool> taken(edges.size());
						taken[0] = 1;
						for(unsigned int j=1;j<edges.size();j++)
						{
							taken[j] = 0;
						}
						while(cycle.size() < edges.size())
						{
							for(unsigned int j=0;j<edges.size();j++)
							{
								if(taken[j]) continue;
								if(cycle[cycle.size()-1] == E[edges[j]].cl1)
								{
									taken[j] = 1;
									cycle.push_back(E[edges[j]].cl2);
									mark.push_back(0);
									marked.push_back(E[edges[j]].cl1);
									order.push_back(edges[j]);
								}
								else if(cycle[cycle.size()-1] == E[edges[j]].cl2)
								{
									taken[j] = 1;
									cycle.push_back(E[edges[j]].cl1);
									mark.push_back(1);
									marked.push_back(E[edges[j]].cl2);
									order.push_back(edges[j]);
								}
							}
						}
						/*for(unsigned int j=0;j<cycle.size();j++)
						{
							cout << cycle[j] << " ";
						}
						cout << "\n";*/

						//check the conditions for the cycle
						if(1)
						{
							Eigen::Matrix3d R;
							vector<int> takes;
							takes.push_back(C[cycle[0]].init);
							takes.push_back(C[cycle[0]].final);
							R = C[marked[0]].R;
							Eigen::Vector3d t = C[marked[0]].t;
							Eigen::Vector3d t2 = -R.transpose() * C[marked[0]].t;
							for(unsigned int j=1;j<(cycle.size()-1);j++)
							{
								if(takes[takes.size()-1] == C[cycle[j]].init)
								{
									takes.push_back(C[cycle[j]].final);
									R = C[cycle[j]].R * R;
									t = C[cycle[j]].R * t + C[cycle[j]].t;
									t2 = C[cycle[j]].R.transpose() * (t2 - C[cycle[j]].t);
								}
								else
								{
									takes.push_back(C[cycle[j]].init);
									R = C[cycle[j]].R.transpose() * R;
									t = C[cycle[j]].R.transpose() * (t - C[cycle[j]].t);
									t2 = C[cycle[j]].R * t2 + C[cycle[j]].t;
								}
							}
							bool palindrom = 0;
							bool palindrom2 = 0;
							/*for(unsigned int j=0;j<takes.size();j++)
							{
								//cout << takes[j] << " ";
								if(takes[j] != takes[takes.size()-2-j])
								{
									palindrom = 0;
									break;
								}
								if(j>takes.size()/2)
									break;
							}
							bool palindrom2 = 1;
							for(unsigned int j=0;j<takes.size();j++)
							{
								if(takes[j+1] != takes[takes.size()-1-j])
								{
									palindrom2 = 0;
									break;
								}
								if(j>takes.size()/2)
									break;
							}*/
							vector<int> st(takes);
							for(unsigned int j=0;j<takes.size();j++)
							{
								st[j] = takes[j];
							}
							std::sort(st.begin(), st.end());
							int back = 0;
							for(unsigned int j=0;j<takes.size()-1;j++)
							{
								if(st[j]==st[j+1])
									back++;
							}

							//verify the cycle
							bool verified = 0;
							if(!palindrom && !palindrom2 && back <= 1)
							{
								double rc = r2a(R).norm();
								//cout << rc << " " << cycle.size()-1 << "\n";
								if(rc <= 0.03 * thr1 * (double)(cycle.size()-1))
								{
									//cout << "K\n";
									//cout << rc << "\n";
									//double tc = t.norm();
									double tc2 = t2.norm();
									//cout << rc << " " << tc2 << " " << cycle.size()-1 << "\n";
									if(tc2 <= 1 * 0.0125 * pd * thr2 * (double)(cycle.size()-1))
									{
										//cout << "K\n";
										//cout << t2 << "\n";
										//cout << tc2 << " " << (cycle.size()-1) << " " << pd << "\n";
										/*for(int j=0;j<cycle.size();j++)
										{
											cout << cycle[j] << " ";
										}
										cout << "\n";*/
										verified = 1;

										//check if the cycle contains contradictory clusters
										if(verified)
										{
											double worst = INFINITY;
											int w2 = INT_MAX;
											for(unsigned int j=0;j<cycle.size();j++)
											{
												for(unsigned int k=j+1;k<cycle.size();k++)
												{
													if(cycle[j] == cycle[k]) continue;
													edge_s ce = E[C2E(cycle[j],cycle[k])];
													//cout << "COMMON " << ce.common << " " << ce.ratio << "\n";
													if(ce.ratio < worst)
														worst = ce.ratio;
													if(ce.common < w2)
														w2 = ce.common;
												}
											}
											if(worst >= 1)
											{
												//cout << "K\n";
												//no contradictory cluster
												verified_edges = merge(verified_edges, se);
												for(unsigned int j=0;j<se.size();j++)
												{
													if(E[se[j]].rc < 1/rc)
													{
														E[se[j]].rc = 1/rc;
													}
													if(E[se[j]].tc < 1/tc2)
													{
														E[se[j]].tc = 1/tc2;
													}														
												}
											}
											else if(worst >= 0.5 || w2 >= -20)
											{
												cout << "?\n";
												//no hard contradictory cluster
												verified_edges_2 = merge(verified_edges_2, se);
												for(unsigned int j=0;j<se.size();j++)
												{
													if(E[se[j]].rc < 1/(2*rc))
													{
														E[se[j]].rc = 1/(2*rc);
													}
													if(E[se[j]].tc < 1/(2*tc2))
													{
														E[se[j]].tc = 1/(2*tc2);
													}														
												}
											}
										}
									}
								}
							}		
						}
					}
				}
			}
		}
		if(NQ.size() < Q.size())
			Q = NQ;
		else
			break;
	}

	//find the neighbouring clusters
	vector<vector<int>> nc(C.size());
	for(unsigned int i=0;i<verified_edges.size();i++)
	{
		nc[E[verified_edges[i]].cl1].push_back(E[verified_edges[i]].cl2);
		nc[E[verified_edges[i]].cl2].push_back(E[verified_edges[i]].cl1);
	}

	//find the connected components
	int cluster = 0;
	vector<int> cc(C.size());
	for(unsigned int i=0;i<cc.size();i++)
	{
		cc[i] = -1;
	}
	while(1)
	{
		bool f = 0;
		stack<int> Q;
		for(unsigned int i=0;i<C.size();i++)
		{
			if(cc[i] == -1)
			{
				f = 1;
				Q.push(i);
				break;
			}
		}
		if(!f) break;
		while(!Q.empty())
		{
			int cur = Q.top();
			Q.pop();
			if(cc[cur] == -1)
			{
				cc[cur] = cluster;
				for(unsigned int i=0;i<nc[cur].size();i++)
				{
					Q.push(nc[cur][i]);
				}
			}
		}
		cluster++;
	}

	//check the consistency of the components
	vector<vector<int>> CCCC;
	for(int cl=0;cl<cluster;cl++)
	{
		vector<int> CC;
		for(unsigned int i=0;i<cc.size();i++)
		{
			if(cc[i] == cl)
				CC.push_back(i);
		}
		bool consistent = 1;
		cout << "GROUP " << cl << " size " << CC.size() << "\n";
		int ic = 0;
		for(unsigned int i=0;i<CC.size();i++)
		{
			
			for(unsigned int j=i+1;j<CC.size();j++)
			{
				if(C[CC[i]].init == C[CC[j]].init && C[CC[i]].final == C[CC[j]].final)
				{
					Eigen::Matrix3d R1 = C[CC[i]].R;
					Eigen::Matrix3d R2 = C[CC[j]].R;
					Eigen::Matrix3d R = R2.transpose() * R1;
					//cout << r2a(R).norm() << "\n";
					//cout << R << "\n";
					//if(isnan(r2a(R).norm()))
					if(r2a(R).norm() > 0.1)
					{
						//cout << "NK\n";
						//consistent = 0;
					}
				}
				else
				{
					int common = E[C2E(CC[i], CC[j])].common;
					double ratio = E[C2E(CC[i], CC[j])].ratio;
					if(ratio < 0.5 && common < -20)
					{
						consistent = 0;
						//cout << "NK2\n";
						ic++;
						//cout << common << " " << ratio << "\n";
					}
					//cout << common << " " << ratio << "\n";
				}
			}
		}
		cout << consistent << "\n";
		if(consistent)
		{
			CCCC.push_back(CC);
		}
		else
		{
			CCCC.push_back(CC);
			//TODO
			//use the star approach instead of this
			//or start removing the smallest clusters until it is consistent
			/*cout << ic << "\n";
			Eigen::MatrixXd r_dist = Eigen::MatrixXd::Zero(CC.size(), CC.size());
			Eigen::MatrixXd t_dist = Eigen::MatrixXd::Zero(CC.size(), CC.size());
			Eigen::MatrixXd c_dist = Eigen::MatrixXd::Zero(CC.size(), CC.size());
			double w1 = 0;
			double w2 = 0;
			double w3 = 0;
			for(int i=0;i<CC.size();i++)
			{
				for(int j=i+1;j<CC.size();j++)
				{
					int edge = C2E(CC[i], CC[j]);
					if(edge != -1)
					{
						r_dist(i,j) = E[edge].rc;
						r_dist(j,i) = E[edge].rc;
						if(w1 < E[edge].rc)
							w1 = E[edge].rc;
						t_dist(i,j) = E[edge].tc;
						if(w2 < E[edge].tc)
							w2 = E[edge].tc;
						t_dist(j,i) = E[edge].tc;
						if(E[edge].common > 0)
						{
							double coeff = 1-1/E[edge].ratio;
							//cout << coeff << "\n";
							c_dist(i,j) = coeff*E[edge].common;
							c_dist(j,i) = coeff*E[edge].common;
							//c_dist(j,i) = 1;//coeff;
							//c_dist(i,j) = 1;//coeff;
							if(w3 < coeff*E[edge].common)
								w3 = coeff*E[edge].common;
						}
					}
					//cout << "E " << edge << "\n";
				}
			}
			if(w1==0) w1=1;
			if(w2==0) w2=1;
			if(w3==0) w3=1;
			//cout << c_dist << "\n";
			Eigen::MatrixXd L_1 = -1 * r_dist;
			Eigen::MatrixXd L_2 = -1 * t_dist;
			Eigen::MatrixXd L_3 = -1 * c_dist;
			//Eigen::MatrixXd L1 = Eigen::MatrixXd::Zero(CC.size(), CC.size());
			//Eigen::MatrixXd L2 = Eigen::MatrixXd::Zero(CC.size(), CC.size());
			Eigen::MatrixXd L3 = Eigen::MatrixXd::Zero(CC.size(), CC.size());
			for(int i=0;i<CC.size();i++)
			{
				double sum = 0;
				double sum2 = 0;
				double sum3 = 0;
				for(int j=0;j<CC.size();j++)
				{
					if(i==j) continue;
					sum+=r_dist(i,j);
					sum2+=t_dist(i,j);
					sum3+=c_dist(i,j);
				}
				L_1(i,i) = sum;
				L_2(i,i) = sum2;
				L_3(i,i) = sum3;
			}
			for(int i=0;i<CC.size();i++)
			{
				for(int j=0;j<CC.size();j++)
				{
					L3(i,j) = L_3(i,j)/(sqrt(L_3(i,i))*sqrt(L_3(j,j)));
				}
			}
			L_1 = (1/w1)*L_1;
			L_2 = (1/w2)*L_2;
			L_3 = (1/w3)*L_3;
			Eigen::MatrixXd L = L_1+L_2+L_3;
			//Eigen::MatrixXd L = L3;
			Eigen::EigenSolver<Eigen::MatrixXd> es(L);
			Eigen::MatrixXcd lambda = es.eigenvalues();
			Eigen::MatrixXcd V = es.eigenvectors();
			for(int i=2;i<CC.size();i++)
			{
				cout << "CONSISTENCY SPLITTING\n";
				int dim = i;
				if(i>10) dim=10;
				vector<Eigen::VectorXd> a;
				for(unsigned int j=0;j<CC.size();j++)
				{
					Eigen::VectorXd nv(dim);
					for(int k=0;k<dim;k++)
					{
						nv(k) = V(j,k).real();
					}
					a.push_back(nv);
				}
				vector<int> c = kmeans(a, i, dim);
				//for(int j=0;j<c.size();j++) cout << c[j] << "\n";
				//cout << "G\n";

				//check consistency of the subgraphs
				bool ccons = 1;
				for(int kl=0;kl<i;kl++)
				{
					int ic2 = 0;
					vector<int> CCC;
					for(int j=0;j<CC.size();j++)
					{
						if(c[j] == kl)
							CCC.push_back(j);
					}
					for(int k=0;k<CCC.size();k++)
					{
						for(int j=k+1;j<CCC.size();j++)
						{
							if(C[CCC[k]].init == C[CCC[j]].init && C[CCC[k]].final == C[CCC[j]].final)
							{
								Eigen::Matrix3d R1 = C[CCC[k]].R;
								Eigen::Matrix3d R2 = C[CCC[j]].R;
								Eigen::Matrix3d R = R2.transpose() * R1;
								//cout << r2a(R).norm() << "\n";
								//cout << R << "\n";
								//if(isnan(r2a(R).norm()))
								if(r2a(R).norm() > 0.1)
								{
									//cout << "NK\n";
									//ccons = 0;
								}
							}
							else
							{
								int common = E[C2E(CCC[k], CCC[j])].common;
								double ratio = E[C2E(CCC[k], CCC[j])].ratio;
								if(ratio < 0.5 && common < -20)
								{
									ccons = 0;
									ic2++;
								}
								//cout << common << " " << ratio << "\n";
							}
						}
					}
					cout << ic2 << "\n";
				}
				cout << ccons << "\n";
				

				if(ccons || i>10)
				{
					for(int kl=0;kl<i;kl++)
					{
						vector<int> CCC;
						for(int j=0;j<CC.size();j++)
						{
							if(c[j] == kl)
								CCC.push_back(j);
						}
						CCCC.push_back(CCC);
					}
					break;
				}
			}*/
		}
	}

	/*for(unsigned int i=0;i<CCCC.size();i++)
	{
		if(CCCC[i].size() <= 1) continue;
		for(unsigned int j=0;j<CCCC.size();j++)
		{
			if(i==j) continue;
			for(unsigned int a=0;a<CCCC[i].size();a++)
			{
				for(unsigned int b=0;b<CCCC[j].size();b++)
				{
					int common = E[C2E(CCCC[i][a], CCCC[j][b])].common;
					double ratio = E[C2E(CCCC[i][a], CCCC[j][b])].ratio;
					cout << "AA " << common << " " << ratio << "\n";
				}
			}
		}
		cout << "\n";
	}*/

	std::vector<std::vector<std::pair<int, int>>> ret;
	for(unsigned int i=0;i<CCCC.size();i++)
	{
		vector<pair<int,int>> nr;
		for(unsigned int j=0;j<CCCC[i].size();j++)
		{
			std::pair<int, int> pos = num2cl[CCCC[i][j]];
			nr.push_back(pos);
		}
		ret.push_back(nr);
	}
	return ret;
}

std::vector<std::vector<std::pair<int, int>>> split_cams(std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> O)
{
	std::vector<std::pair<int, int>> num2cl;
	for(unsigned int i=0;i<O.size();i++)
	{
		for(unsigned int j=0;j<O[i].size();j++)
		{
			//check whether the cluster is contradictory
			int com = common(O[i][j].first, O[i][j].second);
			//cout << "COMMON " << com << " " << O[i][j].first.size() + O[i][j].second.size() << "\n";
			if((unsigned int)(20*com) <= O[i][j].first.size() + O[i][j].second.size())
			{
				pair<int, int> np;
				np.first = i;
				np.second = j;
				num2cl.push_back(np);
			}
		}
	}

	//find the adjacency matrix
	Eigen::MatrixXd b_mat = Eigen::MatrixXd::Zero(num2cl.size(), num2cl.size());
	for(unsigned int i=0;i<num2cl.size();i++)
	{
		for(unsigned int j=0;j<num2cl.size();j++)
		{
			std::vector<int> o11 = O[num2cl[i].first][num2cl[i].second].first;
			std::vector<int> o21 = O[num2cl[j].first][num2cl[j].second].first;

			std::vector<int> o12 = O[num2cl[i].first][num2cl[i].second].second;
			std::vector<int> o22 = O[num2cl[j].first][num2cl[j].second].second;
			
			int com1 = common(o11, o21);
			int com2 = common(o12, o22);
			int com3 = common(o11, o22);
			int com4 = common(o12, o21);

			if(com1 > 0 && com2 > 0 && com3 <= 0 && com4 <= 0)
				b_mat(i,j) = 1;
			else if(com1+com2-com3-com4 < 0)
				b_mat(i,j) = -1;
			else
				b_mat(i,j) = 0;
		}
		b_mat(i,i) = 1;
	}

	//find the connected components
	vector<vector<int>> ccomp;
	vector<bool> in(num2cl.size());
	for(unsigned int i=0;i<in.size();i++)
		in[i] = 0;
	unsigned int in_c = 0;
	while(in_c < in.size())
	{
		vector<int> comp;
		queue<int> q;
		for(unsigned int i=0;i<in.size();i++)
		{
			if(!in[i])
			{
				in[i] = 1;
				comp.push_back(i);
				q.push(i);
				in_c++;
				break;
			}
		}
		while(!q.empty())
		{
			int cur = q.front();
			q.pop();

			for(unsigned int i=0;i<in.size();i++)
			{
				if((b_mat(cur, i) > 0.5 || b_mat(cur, i) < -0.5) && !in[i])
				{
					in[i] = 1;
					comp.push_back(i);
					q.push(i);
					in_c++;
				}
			}
		}
		ccomp.push_back(comp);
	}

	//divide every connected component
	//we expect existence of two objects and no false motions, if there are more of them, it may cause a mess
	//this may however be forced
	std::vector<std::vector<std::pair<int, int>>> ret;
	for(unsigned int i=0;i<ccomp.size();i++)
	{
		vector<int> comp  = ccomp[i];
		Eigen::MatrixXd c_mat = Eigen::MatrixXd::Zero(comp.size(), comp.size());
		Eigen::MatrixXd d_mat = Eigen::MatrixXd::Zero(comp.size(), comp.size());
		for(unsigned int j=0;j<comp.size();j++)
		{
			int d=0;
			for(unsigned int k=0;k<comp.size();k++)
			{
				c_mat(j,k) = b_mat(comp[j], comp[k]);
				if(j!=k && (c_mat(j,k) > 0.5 || c_mat(j,k) < -0.5))
				{
					d++;
				}
			}
			d_mat(j,j) = d;
		}
		//cout << c_mat << "\n\n";
		Eigen::MatrixXd ZA = d_mat.inverse() * c_mat;
		Eigen::EigenSolver<Eigen::MatrixXd> es(ZA);
		Eigen::MatrixXcd lambda = es.eigenvalues();
		Eigen::MatrixXcd V = es.eigenvectors();
		
		//cout << lambda << "\n\n";
		int best = -1;
		double best_d = INFINITY;
		for(unsigned int j=0;j<comp.size();j++)
		{
			double dist = fabs(lambda(j).real() - 1);
			//cout << dist << " " << lambda(j).real() << "\n";
			if(dist < best_d)
			{
				best_d = dist;
				best = j;
			}
		}

		if(best >= 0)
		{
			//segment the clusters according to the eigenvector
			Eigen::VectorXd nv(comp.size());
			for(unsigned int k=0;k<comp.size();k++)
			{
				//nv(k) = V(best,k).real();
				nv(k) = V(k,best).real();
			}
			/*cout << "EV " << lambda(best) << "\n\n";
			cout << nv << "\n\n";*/
			std::vector<std::pair<int, int>> g1;
			std::vector<std::pair<int, int>> g2;
			for(unsigned int k=0;k<comp.size();k++)
			{
				pair<int, int> e = num2cl[comp[k]];
				if(nv(k) > 0)
				{
					g1.push_back(e);
				}
				else if(nv(k) < 0)
				{
					g2.push_back(e);
				}
			}
			ret.push_back(g1);
			ret.push_back(g2);
		}
		else
		{
			//put everything in one group
			std::vector<std::pair<int, int>> g1;
			for(unsigned int k=0;k<comp.size();k++)
			{
				pair<int, int> e = num2cl[comp[k]];
				g1.push_back(e);
			}
			ret.push_back(g1);
		}
	}
	return ret;
}

int select_group(std::vector<std::vector<std::pair<int, int>>> CLCL, std::vector<std::vector<clust_m>> CL)
{
	int ret = 0;
	int count = 0;
	for(unsigned int i=0;i<CLCL.size();i++)
	{
		int cur_c = 0;
		for(unsigned int j=0;j<CLCL[i].size();j++)
		{
			pair<int, int> cur = CLCL[i][j];
			cur_c += CL[cur.first][cur.second].size;
		}
		if(cur_c > count)
		{
			count = cur_c;
			ret = i;
		}
	}
	return ret;
}

int select_group_o(std::vector<std::vector<std::pair<int, int>>> CLCL, std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> O)
{
	int ret = 0;
	int count = 0;
	for(unsigned int i=0;i<CLCL.size();i++)
	{
		int cur_bg = 0;
		int cur_fg = 0;
		for(unsigned int j=0;j<CLCL[i].size();j++)
		{
			pair<int, int> cur = CLCL[i][j];
			cur_bg += O[cur.first][cur.second].first.size();
			cur_fg += O[cur.first][cur.second].second.size();
		}
		int cur_c = cur_fg;
		if(cur_bg < cur_fg)
			cur_c = cur_bg;
		if(cur_c > count)
		{
			count = cur_c;
			ret = i;
		}
	}
	return ret;
}

std::pair<std::vector<int>, std::vector<int>> group_cams(std::vector<std::pair<int, int>> CL, std::vector<std::vector<std::vector<motion_t>>> MM)
{
	vector<int> OB;
	vector<int> OO;
	for(unsigned int i=0;i<CL.size();i++)
	{
		pair<int, int> c = CL[i];
		vector<motion_t> M = MM[c.first][c.second];
		for(unsigned int j=0;j<M.size();j++)
		{
			//maybe sort and remove duplicities afterwards, this does not remove all of them
			if(M[j].init < M[j].final)
			{
				//if(!OB.size() || OB[OB.size()-1] != M[j].c1)
					OB.push_back(M[j].c1);
				//if(!OO.size() || OO[OO.size()-1] != M[j].c2)
					OO.push_back(M[j].c2);
			}
			else
			{
				//if(!OB.size() || OB[OB.size()-1] != M[j].c2)
					OB.push_back(M[j].c2);
				//if(!OO.size() || OO[OO.size()-1] != M[j].c1)
					OO.push_back(M[j].c1);
			}
		}
	}
	pair<vector<int>, vector<int>> ret;
	std::sort(OB.begin(), OB.end());
	std::sort(OO.begin(), OO.end());
	vector<int> OB2;
	if(OB.size())
	{
		OB2.push_back(OB[0]);
		for(unsigned int i=1;i<OB.size();i++)
		{
			if(OB[i] != OB[i-1])
				OB2.push_back(OB[i]);
		}
	}
	vector<int> OO2;
	if(OO.size())
	{
		OO2.push_back(OO[0]);
		for(unsigned int i=1;i<OO.size();i++)
		{
			if(OO[i] != OO[i-1])
				OO2.push_back(OO[i]);
		}
	}
	ret.first = OB2;
	ret.second = OO2;
	return ret;
}

std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<int>> split_tracks(std::pair<std::vector<int>, std::vector<int>> Q, std::vector<std::vector<int>> O, int size)
{
	vector<int> score(size);
	for(int i=0;i<size;i++)
		score[i] = 0;
	vector<int> bck = Q.first;
	for(unsigned int i=0;i<bck.size();i++)
	{
		vector<int> obs = O[bck[i]];
		for(unsigned int j=0;j<obs.size();j++)
		{
			score[obs[j]]++;
		}
	}
	/*for(int i=0;i<bck.size();i++)
	{
		vector<int> obs = O[bck[i]];
		for(int j=0;j<obs.size();j++)
		{
			score[obs[j]]++;
		}
	}*/
	int col = 0;
	vector<int> obj = Q.second;
	for(unsigned int i=0;i<obj.size();i++)
	{
		vector<int> obs = O[obj[i]];
		for(unsigned int j=0;j<obs.size();j++)
		{
			if(score[obs[j]] > 0)
				col++;
			score[obs[j]]--;
		}
	}
	cout << "COLISIONS " << col << "\n";

	vector<int> bt;
	vector<int> ot;
	vector<int> ut;
	for(int i=0;i<size;i++)
	{
		if(score[i] > 0)
			bt.push_back(i);
		else if(score[i] < 0)
			ot.push_back(i);
		else
			ut.push_back(i);
	}

	pair<vector<int>, vector<int>> ret;
	ret.first = bt;
	ret.second = ot;
	std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<int>> ret2;
	ret2.first = ret;
	ret2.second = ut;
	return ret2;
}

std::vector<int> order_fold(std::vector<std::vector<std::pair<int, int>>> T, std::pair<std::vector<int>, std::vector<int>> OT, int takes)
{
	vector<int> b = OT.first;
	vector<int> o = OT.second;
	Eigen::MatrixXd b_mat = Eigen::MatrixXd::Zero(takes, takes);
	Eigen::MatrixXd o_mat = Eigen::MatrixXd::Zero(takes, takes);
	//common points for background
	for(unsigned int i=0;i<b.size();i++)
	{
		vector<pair<int, int>> tr = T[b[i]];
		if(tr.size() > 1)
		{
			for(unsigned int q=0;q<tr.size();q++)
			{
				for(unsigned int r=q+1;r<tr.size();r++)
				{
					if(tr[q].first != tr[r].first)
					{
						b_mat(tr[q].first-1, tr[r].first-1) = b_mat(tr[q].first-1, tr[r].first-1) + 1;
						b_mat(tr[r].first-1, tr[q].first-1) = b_mat(tr[q].first-1, tr[r].first-1) + 1;
					}
				}
			}
		}
	}
	//common points for object
	for(unsigned int i=0;i<o.size();i++)
	{
		vector<pair<int, int>> tr = T[o[i]];
		if(tr.size() > 1)
		{
			for(unsigned int q=0;q<tr.size();q++)
			{
				for(unsigned int r=q+1;r<tr.size();r++)
				{
					if(tr[q].first != tr[r].first)
					{
						o_mat(tr[q].first-1, tr[r].first-1) = o_mat(tr[q].first-1, tr[r].first-1) + 1;
						o_mat(tr[r].first-1, tr[q].first-1) = o_mat(tr[q].first-1, tr[r].first-1) + 1;
					}
				}
			}
		}
	}
	Eigen::MatrixXd min_mat = o_mat;
	for(int i=0;i<takes;i++)
	{
		for(int j=0;j<takes;j++)
		{
			if(b_mat(i,j) < o_mat(i,j))
				min_mat(i,j) = b_mat(i,j);
		}
	}
	//find the ordering and the central take
	vector<int> ret;
	double strongest = 0;
	int best_i = 0;
	int best_j = 0;
	for(int i=0;i<takes;i++)
	{
		for(int j=i+1;j<takes;j++)
		{
			if(min_mat(i,j) > strongest)
			{
				strongest = min_mat(i,j);
				best_i = i;
				best_j = j;
			}
		}
	}
	ret.push_back(best_i);
	ret.push_back(best_j);

	while(ret.size() < (unsigned int)takes)
	{
		for(int i=0;i<takes;i++)
		{
			min_mat(i,ret[0]) = min_mat(i,ret[0])+min_mat(i,ret[ret.size()-1]);
			min_mat(ret[0], i) = min_mat(i, ret[0]);
			min_mat(i, ret[ret.size()-1]) = 0;
			min_mat(ret[ret.size()-1], i) = 0;
		}
		min_mat(ret[0], ret[0]) = 0;
		strongest = 0;
		best_i = 0;
		for(int i=0;i<takes;i++)
		{
			if(min_mat(i, ret[0]) > strongest)
			{
				strongest = min_mat(i, ret[0]);
				best_i = i;
			}
		}
		ret.push_back(best_i);
	}

	return ret;
}

trans_s transform_points_N(vector<Eigen::Vector3d> points1, vector<Eigen::Vector3d> points2)
{
	Eigen::Vector3d P1_centroid = points1[0];
	Eigen::Vector3d P2_centroid = points2[0];
	for(unsigned int i=1;i<points1.size();i++)
	{
		P1_centroid = P1_centroid + points1[i];
		P2_centroid = P2_centroid + points2[i];
	}
	P1_centroid(0) = P1_centroid(0) / points1.size();
	P1_centroid(1) = P1_centroid(1) / points1.size();
	P1_centroid(2) = P1_centroid(2) / points1.size();
	P2_centroid(0) = P2_centroid(0) / points1.size();
	P2_centroid(1) = P2_centroid(1) / points1.size();
	P2_centroid(2) = P2_centroid(2) / points1.size();

	vector<Eigen::Vector3d> P1c;
	vector<Eigen::Vector3d> P2c;
	for(unsigned int i=0;i<points2.size();i++)
	{
		P1c.push_back(points1[i] - P1_centroid);
		P2c.push_back(points2[i] - P2_centroid);
	}

	//estimate scale
	vector<double> sc;
	for(unsigned int i=0;i<P1c.size();i++)
	{
		sc.push_back(P2c[i].norm() / P2c[i].norm());
	}
	double s = med(sc, 0, sc.size()-1, sc.size()/2);

	vector<Eigen::Vector3d> P1s;
	for(unsigned int i=0;i<P1c.size();i++)
	{
		P1s.push_back(P1c[i] * s);
	}
	//cout << P1s[0].transpose() << "\n";

	//estimate rotation
	Eigen::Matrix3d H;
	H.setZero();
	for(unsigned int i=0;i<P1s.size();i++)
	{
		H = H + P1s[i] * P2c[i].transpose();
	}
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

	//estimate translation;
	vector<Eigen::Vector3d> P1rs;
	for(unsigned int i=0;i<points1.size();i++)
	{
		P1rs.push_back(s * R * points1[i]);
	}
	vector<Eigen::Vector3d> diff;
	for(unsigned int i=0;i<P1rs.size();i++)
	{
		diff.push_back(points2[i] - P1rs[i]);
	}
	Eigen::Vector3d t = median(diff);
	
	trans_s ret;
	ret.R = R;
	ret.o = t;
	ret.s = s;
	return ret;
}

trans_s transform_points_R_N(vector<Eigen::Vector3d> pnts1, vector<Eigen::Vector3d> pnts2, int ssample, int trials)
{
	srand(time(NULL));
	//int best_c = 0;
	double best_v = 100;
	trans_s best;
	for(int i=0;i<trials;i++)
	{
		//build the sample
		vector<Eigen::Vector3d> points1;
		vector<Eigen::Vector3d> points2;
		vector<int> prev_s;
		for(int j=0;j<ssample;j++)
		{
			while(1)
			{
				int ns = rand() % pnts1.size();
				bool ok = true;
				for(unsigned int k=0;k<prev_s.size();k++)
				{
					if(ns == prev_s[k])
					{
						ok = 0;
						break;
					}
				}
				if(ok)
				{
					points1.push_back(pnts1[ns]);
					points2.push_back(pnts2[ns]);
					prev_s.push_back(ns);
					break;
				}
			}
		}

		//find the hypothesis
		Eigen::Vector3d P1_centroid = points1[0];
		Eigen::Vector3d P2_centroid = points2[0];
		for(unsigned int i=1;i<points1.size();i++)
		{
			P1_centroid = P1_centroid + points1[i];
			P2_centroid = P2_centroid + points2[i];
		}
		P1_centroid(0) = P1_centroid(0) / points1.size();
		P1_centroid(1) = P1_centroid(1) / points1.size();
		P1_centroid(2) = P1_centroid(2) / points1.size();
		P2_centroid(0) = P2_centroid(0) / points1.size();
		P2_centroid(1) = P2_centroid(1) / points1.size();
		P2_centroid(2) = P2_centroid(2) / points1.size();

		vector<Eigen::Vector3d> P1c;
		vector<Eigen::Vector3d> P2c;
		for(unsigned int i=0;i<points2.size();i++)
		{
			P1c.push_back(points1[i] - P1_centroid);
			P2c.push_back(points2[i] - P2_centroid);
		}

		//estimate scale
		vector<double> sc;
		for(unsigned int i=0;i<P1c.size();i++)
		{
			sc.push_back(P2c[i].norm() / P2c[i].norm());
		}
		double s = med(sc, 0, sc.size()-1, sc.size()/2);

		vector<Eigen::Vector3d> P1s;
		for(unsigned int i=0;i<P1c.size();i++)
		{
			P1s.push_back(P1c[i] * s);
		}
		//cout << P1s[0].transpose() << "\n";

		//estimate rotation
		Eigen::Matrix3d H;
		H.setZero();
		for(unsigned int i=0;i<P1s.size();i++)
		{
			H = H + P1s[i] * P2c[i].transpose();
		}
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

		//estimate translation;
		vector<Eigen::Vector3d> P1rs;
		for(unsigned int i=0;i<points1.size();i++)
		{
			P1rs.push_back(s * R * points1[i]);
		}
		vector<Eigen::Vector3d> diff;
		for(unsigned int i=0;i<P1rs.size();i++)
		{
			diff.push_back(points2[i] - P1rs[i]);
		}
		Eigen::Vector3d t = median(diff);

		//find the support
		//int count = 0;
		std::vector<double> err;
		for(unsigned int k=0;k<pnts1.size();k++)
		{
			Eigen::Vector3d diff = pnts2[k] - (s * R * pnts1[k] + t);
			err.push_back(diff.norm());
			//cout << diff.norm() << " ";
		}
		double er = med(err, 0, err.size()-1, err.size()/2);
		if(er < best_v)
		{
			best_v = er;
			best.R = R;
			best.o = t;
			best.s = s;
		}
		//cout << er << "\n";
		
	}
	cout << best_v << "\n";
	vector<Eigen::Vector3d> points1;
	vector<Eigen::Vector3d> points2;
	for(unsigned int k=0;k<pnts1.size();k++)
	{
		Eigen::Vector3d diff = pnts2[k] - (best.s * best.R * pnts1[k] + best.o);
		if(diff.norm() <= best_v)
		{
			points1.push_back(pnts1[k]);
			points2.push_back(pnts2[k]);
		}
	}
	
	
	
	trans_s ret;
	ret = transform_points_N(points1, points2);
	/*ret.R = R;
	ret.o = t;
	ret.s = s;*/
	return ret;
}

trans_s transform_points_2(vector<Eigen::Vector3d> points1, vector<Eigen::Vector3d> points2, double s)
{
	Eigen::Vector3d P1_centroid = points1[0];
	Eigen::Vector3d P2_centroid = points2[0];
	for(unsigned int i=1;i<points1.size();i++)
	{
		P1_centroid = P1_centroid + points1[i];
		P2_centroid = P2_centroid + points2[i];
	}
	P1_centroid(0) = P1_centroid(0) / points1.size();
	P1_centroid(1) = P1_centroid(1) / points1.size();
	P1_centroid(2) = P1_centroid(2) / points1.size();
	P2_centroid(0) = P2_centroid(0) / points1.size();
	P2_centroid(1) = P2_centroid(1) / points1.size();
	P2_centroid(2) = P2_centroid(2) / points1.size();

	vector<Eigen::Vector3d> P1c;
	vector<Eigen::Vector3d> P2c;
	for(unsigned int i=0;i<points2.size();i++)
	{
		P1c.push_back(points1[i] - P1_centroid);
		P2c.push_back(points2[i] - P2_centroid);
	}

	//estimate scale
	/*vector<double> sc;
	for(int i=0;i<P1c.size();i++)
	{
		sc.push_back(P2c[i].norm() / P2c[i].norm());
	}
	double s = med(sc, 0, sc.size()-1, sc.size()/2);*/

	vector<Eigen::Vector3d> P1s;
	for(unsigned int i=0;i<P1c.size();i++)
	{
		P1s.push_back(P1c[i] * s);
	}
	//cout << P1s[0].transpose() << "\n";

	//estimate rotation
	Eigen::Matrix3d H;
	H.setZero();
	for(unsigned int i=0;i<P1s.size();i++)
	{
		H = H + P1s[i] * P2c[i].transpose();
	}
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

	//estimate translation;
	vector<Eigen::Vector3d> P1rs;
	for(unsigned int i=0;i<points1.size();i++)
	{
		P1rs.push_back(s * R * points1[i]);
	}
	vector<Eigen::Vector3d> diff;
	for(unsigned int i=0;i<P1rs.size();i++)
	{
		diff.push_back(points2[i] - P1rs[i]);
	}
	Eigen::Vector3d t = median(diff);
	
	trans_s ret;
	ret.R = R;
	ret.o = t;
	ret.s = s;
	return ret;
}

std::vector<int> merge_obs(std::vector<std::vector<int>> obs)
{
	std::vector<int> ret;
	if(!obs.size()) return ret;
	for(unsigned int i=0;i<obs[0].size();i++)
	{
		int val = -1;
		int count = 0;
		for(unsigned int j=0;j<obs.size();j++)
		{
			if(!obs[j].size()) continue;
			int cur_count = 0;
			int cur_val = obs[j][i];
			if(cur_val == -1) continue;
			for(unsigned int k=0;k<obs.size();k++)
			{
				if(!obs[k].size()) continue;
				if(cur_val == obs[k][i])
				{
					cur_count++;
				}
			}
			if(cur_count > count)
			{
				count = cur_count;
				val = cur_val;
			}
		}
		ret.push_back(val);
	}
	return ret;
}

std::pair<std::pair<pnts_s, pnts_s>, std::vector<std::pair<trans_s,trans_s>>> merge_reconstructions(std::vector<pnts_s> P, std::vector<std::vector<std::pair<int, int>>> T, std::pair<std::vector<int>, std::vector<int>> D, std::vector<int> order)
//std::pair< std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>> , std::vector<std::pair<trans_s,trans_s>> >  merge_reconstructions(std::vector<pnts_s> P, std::vector<std::vector<std::pair<int, int>>> T, std::pair<std::vector<int>, std::vector<int>> D, std::vector<int> order)
{
	int ref = order[0];
	//the structure into which the final points are saved (according to the tracks)
	vector<vector<Eigen::Vector3d>> points;
	vector<vector<Eigen::Vector3i>> colors;
	for(unsigned int i=0;i<T.size();i++)
	{
		vector<Eigen::Vector3d> np;
		points.push_back(np);
		vector<Eigen::Vector3i> nc;
		colors.push_back(nc);
	}
	vector<pair<trans_s, trans_s>> motions(P.size());
	vector<bool> take_exists(P.size());
	for(unsigned int i=0;i<P.size();i++)
	{
		if(i==(unsigned int)ref)
			take_exists.push_back(1);
		else
			take_exists.push_back(0);
	}
	//copy points from the reference take to the structure
	for(unsigned int i=0;i<D.first.size();i++)
	{
		vector<pair<int,int>> track = T[D.first[i]];
		for(unsigned int j=0;j<track.size();j++)
		{
			if(track[j].first == ref+1)
			{
				points[D.first[i]].push_back(P[ref].points[P[ref].ID_map[track[j].second]]);
				cout << P[ref].points[P[ref].ID_map[track[j].second]].transpose() << "\n";
				colors[D.first[i]].push_back(P[ref].color[P[ref].ID_map[track[j].second]]);
				break;
			}
		}
	}
	cout << "\n\n";
	for(unsigned int i=0;i<D.second.size();i++)
	{
		vector<pair<int,int>> track = T[D.second[i]];
		for(unsigned int j=0;j<track.size();j++)
		{
			if(track[j].first == ref+1)
			{
				points[D.second[i]].push_back(P[ref].points[P[ref].ID_map[track[j].second]]);
				cout << P[ref].points[P[ref].ID_map[track[j].second]].transpose() << "\n";
				colors[D.second[i]].push_back(P[ref].color[P[ref].ID_map[track[j].second]]);
				break;
			}
		}
	}

	//iterate through the rest of the reconstructions in the given order and add the points to the final reconstruction
	for(unsigned int rr=1;rr<order.size();rr++)
	{
		if(order[rr] < 0) continue;
		int r = order[rr];
		take_exists[r] = 1;

		//collect points from the background which contain both the points from the merged reconstruction and from the current take
		vector<Eigen::Vector3d> group1;
		vector<Eigen::Vector3d> group2;
		for(unsigned int i=0;i<D.first.size();i++)
		{
			if(!points[D.first[i]].size()) continue;
			vector<pair<int, int>> track = T[D.first[i]];
			for(unsigned int j=0;j<track.size();j++)
			{
				if(track[j].first == r+1)
				{
					group1.push_back(P[r].points[P[r].ID_map[track[j].second]]);
					group2.push_back(median(points[D.first[i]]));
				}
			}
		}
		//TODO
		//also try RANSAC to find the transformation
		//test both versions
		if(!group1.size() || !group2.size()) continue;
		trans_s tb = transform_points_N(group1, group2);
		//trans_s tb = transform_points_R_N(group1, group2, 3, 2000);

		//find the points from the background, transform them and add to the points
		for(unsigned int i=0;i<D.first.size();i++)
		{
			vector<pair<int, int>> track = T[D.first[i]];
			for(unsigned int j=0;j<track.size();j++)
			{
				if(track[j].first == r+1)
				{
					Eigen::Vector3d p = P[r].points[P[r].ID_map[track[j].second]];
					Eigen::Vector3d new_p = tb.s * tb.R * p + tb.o;
					points[D.first[i]].push_back(new_p);
					colors[D.first[i]].push_back(P[r].color[P[r].ID_map[track[j].second]]);
					
				}
			}
		}

		//do the same with the object
		trans_s to;
		vector<Eigen::Vector3d> gr1;
		vector<Eigen::Vector3d> gr2;
		for(unsigned int i=0;i<D.second.size();i++)
		{
			if(!points[D.second[i]].size()) continue;
			vector<pair<int, int>> track = T[D.second[i]];
			for(unsigned int j=0;j<track.size();j++)
			{
				if(track[j].first == r+1)
				{
					gr1.push_back(P[r].points[P[r].ID_map[track[j].second]]);
					gr2.push_back(median(points[D.second[i]]));
				}
			}
		}
		cout << gr1.size() << " " << gr2.size() << "\n";
		to = transform_points_2(gr1, gr2, tb.s);

		for(unsigned int i=0;i<D.second.size();i++)
		{
			vector<pair<int, int>> track = T[D.second[i]];
			for(unsigned int j=0;j<track.size();j++)
			{
				if(track[j].first == r+1)
				{
					Eigen::Vector3d p = P[r].points[P[r].ID_map[track[j].second]];
					Eigen::Vector3d new_p = to.s * to.R * p + to.o;
					points[D.second[i]].push_back(new_p);
					colors[D.second[i]].push_back(P[r].color[P[r].ID_map[track[j].second]]);
				}
			}
		}
		pair<trans_s, trans_s> nm;
		nm.first = tb;
		nm.second = to;
		motions[r]=nm;
	}

	//extract the points
	/*std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>> ret;
	std::vector<Eigen::Vector3d> rpnts;
	std::vector<Eigen::Vector3i> rcol;
	for(int i=0;i<points.size();i++)
	{
		Eigen::Vector3d pnt;
		Eigen::Vector3i col;
		if(points[i].size())
		{
			pnt = median(points[i]);
			col = median(colors[i]);
		}
		else
		{
			pnt(0) = 0;
			pnt(1) = 0;
			pnt(2) = 0;
			col(0) = -1;
			col(1) = -1;
			col(2) = -1;
		}
		rpnts.push_back(pnt);
		rcol.push_back(col);
	}
	ret.first = rpnts;
	ret.second = rcol;

	std::pair< std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>> , std::vector<std::pair<trans_s,trans_s>> > ret2;
	ret2.first = ret;
	ret2.second = motions;
	return ret2;*/
	//HERE_
	vector<Eigen::Vector3d> pb;
	vector<Eigen::Vector3i> cb;
	std::vector<int> IDb;
	std::unordered_map<int, int> IDb_map;
	vector<Eigen::Vector3d> po;
	vector<Eigen::Vector3i> co;
	std::vector<int> IDo;
	std::unordered_map<int, int> IDo_map;
	for(unsigned int i=0;i<D.first.size();i++)
	{
		if(points[D.first[i]].size())
		{
			IDb.push_back(D.first[i]);
			IDb_map[D.first[i]] = pb.size();
			Eigen::Vector3d np = median(points[D.first[i]]);
			pb.push_back(np);
			Eigen::Vector3i nc = median(colors[D.first[i]]);
			cb.push_back(nc);
		}
	}
	for(unsigned int i=0;i<D.second.size();i++)
	{
		if(points[D.second[i]].size())
		{
			IDo.push_back(D.second[i]);
			IDo_map[D.second[i]] = po.size();
			Eigen::Vector3d np = median(points[D.second[i]]);
			po.push_back(np);
			Eigen::Vector3i nc = median(colors[D.second[i]]);
			co.push_back(nc);
		}
	}

	std::pair<std::pair<pnts_s, pnts_s>, std::vector<std::pair<trans_s,trans_s>>> ret;
	//cout << "TEST " << pb.size() << " " << 
	pnts_s pnts_b;
	pnts_b.points = pb;
	pnts_b.ID = IDb;
	pnts_b.ID_map = IDb_map;
	pnts_b.color = cb;
	pnts_s pnts_o;
	pnts_o.points = po;
	pnts_o.ID = IDo;
	pnts_o.ID_map = IDo_map;
	pnts_o.color = co;
	pair<pnts_s, pnts_s> pnts;
	pnts.first = pnts_b;
	pnts.second = pnts_o;
	ret.first = pnts;
	ret.second = motions;
	return ret;
}

void save_model_ply(std::pair<pnts_s, pnts_s> M, int mode)
{
	ofstream ply;
	ply.open("model.txt");
	for(unsigned int i=0;i<M.first.points.size();i++)
	{
		Eigen::Vector3d cp = M.first.points[i];
		ply << cp(0) << " " << cp(1) << " " << cp(2) << " 255 0 0\n";
	}
	for(unsigned int i=0;i<M.second.points.size();i++)
	{
		Eigen::Vector3d cp = M.second.points[i];
		ply << cp(0) << " " << cp(1) << " " << cp(2) << " 0 255 0\n";
	}
	ply.close();
}

void save_points(std::pair<pnts_s, pnts_s> P)
{
	ofstream pnts;
	pnts.open("model1.txt");
	for(unsigned int i=0;i<P.first.points.size();i++)
	{
		pnts << " " << P.first.points[i](0) << " " << P.first.points[i](1) << " " << P.first.points[i](2) << "\n";
	}
	pnts.close();
	
	pnts.open("model2.txt");
	for(unsigned int i=0;i<P.second.points.size();i++)
	{
		pnts << " " << P.second.points[i](0) << " " << P.second.points[i](1) << " " << P.second.points[i](2) << "\n";
	}
	pnts.close();
}

std::pair<std::vector<img_s>, std::vector<trans_s>> merge_cameras(std::vector<cam_s> C, std::vector<std::pair<trans_s,trans_s>> motions, int ref, std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> O, std::pair<std::vector<int>, std::vector<int>> Q, std::pair<std::vector<int>, std::vector<int>> D)
{
	cout << "MERGING CAMERAS\n";
	//classify the cameras
	vector<int> SC(C.size());
	for(unsigned int i=0;i<C.size();i++)
	{
		if(C[i].anchor == C[i].second)
			SC[i] = 0;
		else
			SC[i] = -1;
	}
	for(unsigned int i=0;i<Q.first.size();i++)
	{
		SC[Q.first[i]] = 1;
	}
	for(unsigned int i=0;i<Q.second.size();i++)
	{
		SC[Q.second[i]] = 2;
	}

	//bring the cameras to the reference system
	vector<cam_s> moved_cams;
	vector<int> mc_ids;
	for(unsigned int i=0;i<C.size();i++)
	{
		if((SC[i] == 0 || SC[i] == 1) && C[i].anchor == ref+1)
		{
			//cameras from the reference reconstruction through background
			//these are not moved
        	//we want center of the camera and angle axis
        	Eigen::Matrix3d R = C[i].R;
        	Eigen::Vector3d t = C[i].t;
        	Eigen::Vector3d c = -R.transpose() * t;
        	
        	cam_s ncam;
        	ncam.R = R;
        	//t will here serve as center of the camera
        	ncam.t = c;
        	ncam.id = C[i].id;
			ncam.second = C[i].second;
			ncam.anchor = C[i].anchor;
			ncam.size_x = C[i].size_x;
			ncam.size_y = C[i].size_y;
			ncam.f = C[i].f;
			ncam.px = C[i].px;
			ncam.py = C[i].py;
			ncam.inl = C[i].inl;
			ncam.id_2 = C[i].id_2;
			ncam.center = C[i].center;
			moved_cams.push_back(ncam);
			mc_ids.push_back(i);
		}
		else if(SC[i] == 0 || SC[i] == 1)
		{
			//camera registered towards the background ot from the anchor take
			//move according to the background motion
			pair<trans_s,trans_s> M = motions[C[i].anchor-1];
			Eigen::Matrix3d R = C[i].R;
			Eigen::Vector3d t = C[i].t;
        	Eigen::Vector3d c = -R.transpose() * t;

        	Eigen::Matrix3d nR = R * M.first.R.transpose();
        	Eigen::Vector3d nc = M.first.s * M.first.R * c + M.first.o;

        	cam_s ncam;
        	ncam.R = nR;
        	//t will here serve as center of the camera
        	ncam.t = nc;
        	ncam.id = C[i].id;
			ncam.second = C[i].second;
			ncam.anchor = C[i].anchor;
			ncam.size_x = C[i].size_x;
			ncam.size_y = C[i].size_y;
			ncam.f = C[i].f;
			ncam.px = C[i].px;
			ncam.py = C[i].py;
			ncam.inl = C[i].inl;
			ncam.id_2 = C[i].id_2;
			ncam.center = C[i].center;
			moved_cams.push_back(ncam);
			mc_ids.push_back(i);
		}
		else if(SC[i] == 2 && C[i].second == ref+1)
		{
			//cameras registered towards the object which come from the reference take
			pair<trans_s,trans_s> M = motions[C[i].anchor-1];
			Eigen::Matrix3d R = C[i].R;
			Eigen::Vector3d t = C[i].t;
        	Eigen::Vector3d c = -R.transpose() * t;

        	Eigen::Matrix3d nR = R * M.second.R.transpose();
        	Eigen::Vector3d nc = M.second.s * M.second.R * c + M.second.o;

        	cam_s ncam;
        	ncam.R = nR;
        	//t will here serve as center of the camera
        	ncam.t = nc;
        	ncam.id = C[i].id;
			ncam.second = C[i].second;
			ncam.anchor = C[i].anchor;
			ncam.size_x = C[i].size_x;
			ncam.size_y = C[i].size_y;
			ncam.f = C[i].f;
			ncam.px = C[i].px;
			ncam.py = C[i].py;
			ncam.inl = C[i].inl;
			ncam.id_2 = C[i].id_2;
			ncam.center = C[i].center;
			moved_cams.push_back(ncam);
			mc_ids.push_back(i);
		}
		else if(SC[i] == 2 && C[i].anchor != ref+1 && C[i].second != ref+1)
		{
			pair<trans_s,trans_s> M1 = motions[C[i].anchor-1];
			pair<trans_s,trans_s> M2 = motions[C[i].second-1];
			Eigen::Matrix3d R = C[i].R;
			Eigen::Vector3d t = C[i].t;
        	Eigen::Vector3d c = -R.transpose() * t;

        	Eigen::Matrix3d R12bo = M1.second.R * M1.first.R.transpose();
        	Eigen::Matrix3d R32bo = M2.second.R * M2.first.R.transpose();
        	Eigen::Vector3d t12bo = M1.second.o - (R12bo * M1.first.o);
        	Eigen::Vector3d t32bo = M2.second.o - (R32bo * M2.first.o);

        	Eigen::Matrix3d nR1 = R * M1.first.R.transpose();
        	Eigen::Vector3d nc1 = (M1.first.s * M1.first.R * c) + M1.first.o;

        	Eigen::Matrix3d nR = nR1 * R12bo.transpose() * R32bo;
        	Eigen::Vector3d nc = R32bo.transpose() * (R12bo * nc1 + t12bo - t32bo);

        	cam_s ncam;
        	ncam.R = nR;
        	//t will here serve as center of the camera
        	ncam.t = nc;
        	ncam.id = C[i].id;
			ncam.second = C[i].second;
			ncam.anchor = C[i].anchor;
			ncam.size_x = C[i].size_x;
			ncam.size_y = C[i].size_y;
			ncam.f = C[i].f;
			ncam.px = C[i].px;
			ncam.py = C[i].py;
			ncam.inl = C[i].inl;
			ncam.id_2 = C[i].id_2;
			ncam.center = C[i].center;
			moved_cams.push_back(ncam);
			mc_ids.push_back(i);
		}
		else if(C[i].anchor == ref+1 && SC[i] == 2 && C[i].second != ref+1)
		{
			pair<trans_s,trans_s> M = motions[C[i].second-1];
			Eigen::Matrix3d R = C[i].R;
			Eigen::Vector3d t = C[i].t;
        	Eigen::Vector3d c = -R.transpose() * t;

        	Eigen::Matrix3d R12bo = M.second.R * M.first.R.transpose();
        	Eigen::Vector3d t12bo = M.second.o - (R12bo * M.first.o);

        	Eigen::Matrix3d nR = R * R12bo;
        	Eigen::Vector3d nc = R12bo.transpose() * (c - t12bo);

        	cam_s ncam;
        	ncam.R = nR;
        	//t will here serve as center of the camera
        	ncam.t = nc;
        	ncam.id = C[i].id;
			ncam.second = C[i].second;
			ncam.anchor = C[i].anchor;
			ncam.size_x = C[i].size_x;
			ncam.size_y = C[i].size_y;
			ncam.f = C[i].f;
			ncam.px = C[i].px;
			ncam.py = C[i].py;
			ncam.inl = C[i].inl;
			ncam.id_2 = C[i].id_2;
			ncam.center = C[i].center;
			moved_cams.push_back(ncam);
			mc_ids.push_back(i);
		}
	}

	/*for(int i=0;i<moved_cams.size();i++)
	{
		EPNPEstimator residualCheck;
		std::vector<Eigen::Vector2d> points2D0;
		std::vector<Eigen::Vector3d> points3D0;
		Eigen::Matrix3x4d p2;
		p2.leftCols<3>() = moved_cams[i].R;
		p2.rightCols<1>() = -1 * moved_cams[i].R * moved_cams[i].t;
		Eigen::Matrix3d K;
		//cout << K << "\n";
		K.setZero();
		K(0,0) = moved_cams[i].f;
		K(1,1) = moved_cams[i].f;
		K(2,2) = 1;
		K(0,2) = moved_cams[i].px;
		K(1,2) = moved_cams[i].py;
		p2 = K*p2;
		std::vector<double> residuals_2;
		vector<int> obs = O[mc_ids[i]].first;
		vector<Eigen::Vector2d> feat = O[mc_ids[i]].second;
		for(int j=0;j<obs.size();j++)
		{
			if(obs[j] >= 0)
			{
				points2D0.push_back(feat[j]);
				Eigen::Vector3d pnt = T[obs[j]];
				points3D0.push_back( pnt );
			}
		}
		residualCheck.Residuals(points2D0, points3D0, p2, &residuals_2);
		for(int i=0;i<residuals_2.size();i++) cout << residuals_2[i] << " ";
		cout << "\n";
	}*/


	//compute motions between object and background
	vector<trans_s> motion;
	for(unsigned int i=0;i<motions.size();i++)
	{
		trans_s m;
		if(i==(unsigned int)ref)
		{
			motion.push_back(m);
			continue;
		}
		pair<trans_s,trans_s> M = motions[i];
		Eigen::Matrix3d R12bo = M.second.R * M.first.R.transpose();
    	Eigen::Vector3d t12bo = M.second.o - (R12bo * M.first.o);
    	
    	m.R = R12bo;
    	m.o = t12bo;
    	m.s = 1;
    	motion.push_back(m);
	}
	vector<int> motion_leader(motions.size());


	//create hash sets of the tracks in the object and in the background
	//HERE_
	unordered_set<int> T1m;
	for(unsigned int i=0;i<D.first.size();i++)
	{
		T1m.insert(D.first[i]);
	}
	
	unordered_set<int> T2m;
	for(unsigned int i=0;i<D.second.size();i++)
	{
		T2m.insert(D.second[i]);
	}

	/*for(unsigned int i=0;i<D.second.size();i++)
	{
		cout << T2m.count(D.second[i]);
	}*/

	

	//group the cameras and merge their observations
	vector<img_s> cams;
	int cam_size = 0;
	for(unsigned int i=0;i<moved_cams.size();i++)
	{
		if(moved_cams[i].id > cam_size)
			cam_size = moved_cams[i].id;
	}
	for(int i=1;i<=cam_size;i++)
	{
		img_s cam;
		int take = 0;
		vector<int> background_obs;
		vector<int> object_obs;
		vector<int> unknown_obs;
		vector<vector<int>> obs;
		vector<Eigen::Vector2d> features;
		double size_x;
		double size_y;
		double f = 0;
		double f2 = 0;
		double px;
		double py;
		vector<Eigen::Vector3d> Rs;
		vector<Eigen::Vector3d> ts;
		for(unsigned int j=0;j<moved_cams.size();j++)
		{
			if(moved_cams[j].id != i) continue;
			if(!take)
			{
				take = moved_cams[j].second;
				size_x = moved_cams[j].size_x;
				size_y = moved_cams[j].size_y;
				px = moved_cams[j].px;
				py = moved_cams[j].py;
				//f2 = motions[moved_cams[j].anchor].second.s * moved_cams[j].f;
				f2 = moved_cams[j].f;
				features = O[mc_ids[j]].second;
			}
			if(moved_cams[j].anchor == moved_cams[j].second)
				f = moved_cams[j].f;
			obs.push_back(O[mc_ids[j]].first);
			Rs.push_back(r2a(moved_cams[j].R));
			ts.push_back(moved_cams[j].t);
		}
		cout << obs.size() << "\n";
		vector<int> m_obs = merge_obs(obs);

		//divide the observations between the object and the background
		for(unsigned int j=0;j<m_obs.size();j++)
		{
			if(T1m.count(m_obs[j]))
			{
				background_obs.push_back(m_obs[j]);
				object_obs.push_back(-1);
				unknown_obs.push_back(-1);
			}
			else if(T2m.count(m_obs[j]))
			{
				background_obs.push_back(-1);
				object_obs.push_back(m_obs[j]);
				unknown_obs.push_back(-1);
			}
			else
			{
				background_obs.push_back(-1);
				object_obs.push_back(-1);
				unknown_obs.push_back(m_obs[j]);
			}
		}

		if(!Rs.size()) continue;
		Eigen::Matrix3d R = a2r(median(Rs));
		Eigen::Vector3d t = median(ts);
		cam.take = take;
		cam.R = R;
		cam.c = t;
		cam.size_x = size_x;
		cam.size_y = size_y;
		cam.px = px;
		cam.py = py;
		cam.features = features;
		if(f > 0)
			cam.f = f;
		else
			cam.f = f2;
		cam.b_obs = background_obs;
		cam.o_obs = object_obs;
		cam.u_obs = unknown_obs;

		cams.push_back(cam);
	}

	cout << "FINISHED\n";
	
	std::pair<std::vector<img_s>, std::vector<trans_s>> ret;
	ret.first = cams;
	ret.second = motion;
	return ret;
}

void save_model(std::pair<pnts_s, pnts_s> P, std::vector<img_s> C, std::vector<trans_s> motion, int mode, int ref)
{
	/*for(int i=0;i<P.first.ID.size();i++)
	{
		cout << i << " " << P.first.ID[i] << " " << P.first.ID_map.at(P.first.ID[i]) << " " << P.first.points[i].transpose() << "\n";
	}*/
	/*for(int i=0;i<motion.size();i++)
	{
		cout << motion[i].R << "\n\n" << motion[i].o << "\n\n";
	}*/
	cout << "Saving cameras\n";

	int img_count = 0;
	ofstream cams;
	string p1 = "model" + to_string(mode) + "/cameras.txt";
	cams.open(p1);
	cams << "# Camera list with one line of data per camera:\n";
	cams << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
	cams << "# Number of cameras: " << C.size() << "\n";
	for(unsigned int i=0;i<C.size();i++)
	{
		if(C[i].take == ref+1 || (mode%4) == 2 || (mode%4) == 3)
			img_count++;
		else
			img_count += 2;
		cams << i+1 << " SIMPLE_RADIAL " << C[i].size_x << " " << C[i].size_y << " " << C[i].f << " " << C[i].px << " " << C[i].py << " -0.03\n";
	}
	cams.close();
	cout << "Saving images\n";

	//also save the observations of each track which will be saved to the points3d.txt
	//use a map
	unordered_map<int, vector<pair<int, int>>> p2tr;
	ofstream imgs;
	string p2 = "model" + to_string(mode) + "/images.txt";
	imgs.open(p2);
	imgs << "# Image list with two lines of data per image:\n";
	imgs << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
	imgs << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
	imgs << "# Number of images: " << img_count << ", mean observations per image: 683.904\n";
	int pos = 1;
	EPNPEstimator residualCheck;
	ofstream c2m;
	string p3 = "model" + to_string(mode) + "/cam2mot.txt";
	c2m.open(p3);
	for(unsigned int i=0;i<C.size();i++)
	{
		Eigen::Vector4d q = RotationMatrixToQuaternion(C[i].R);
		Eigen::Vector3d t = -1 * C[i].R * C[i].c;
		if(C[i].take == ref+1)
		{
			imgs << pos << " " << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << " " << t(0) << " " << t(1) << " " << t(2) << " " << i+1 << " " << i+1 << ".JPG\n";
			for(unsigned int j=0;j<C[i].features.size();j++)
			{
				imgs << C[i].features[j](0) << " " << C[i].features[j](1) << " ";
				if(C[i].b_obs[j] != -1)
				{
					imgs << C[i].b_obs[j]+1 << " ";
					vector<pair<int, int>> tr;
					pair<int, int> n_ob;
					n_ob.first = pos;
					n_ob.second = j+1;
					if(p2tr.count(C[i].b_obs[j]))
					{
						tr = p2tr[C[i].b_obs[j]];
					}
					tr.push_back(n_ob);
					p2tr[C[i].b_obs[j]] = tr;
				}
				else if(C[i].o_obs[j] != -1)
				{
					imgs << C[i].o_obs[j]+1 << " ";
					vector<pair<int, int>> tr;
					pair<int, int> n_ob;
					n_ob.first = pos;
					n_ob.second = j+1;
					if(p2tr.count(C[i].o_obs[j]))
					{
						tr = p2tr[C[i].o_obs[j]];
					}
					tr.push_back(n_ob);
					p2tr[C[i].o_obs[j]] = tr;
				}
				else
					imgs << -1 << " ";
			}
			imgs << "\n";
			c2m << "-1\n";
			pos++;
		}
		else
		{
			if((mode%4) != 3)
			{
				std::vector<Eigen::Vector2d> points2D0;
				std::vector<Eigen::Vector3d> points3D0;
				Eigen::Matrix3x4d p2;
				p2.leftCols<3>() = C[i].R;
				p2.rightCols<1>() = t;
				Eigen::Matrix3d K;
				K.setZero();
				K(0,0) = C[i].f;
				K(1,1) = C[i].f;
				K(2,2) = 1;
				K(0,2) = C[i].px;
				K(1,2) = C[i].py;
				p2 = K*p2;
				std::vector<double> residuals_2;
			
				imgs << pos << " " << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << " " << t(0) << " " << t(1) << " " << t(2) << " " << i+1 << " " << i+1 << ".JPG\n";
				for(unsigned int j=0;j<C[i].features.size();j++)
				{
					imgs << C[i].features[j](0) << " " << C[i].features[j](1) << " ";
					if(C[i].b_obs[j] != -1)
					{
						imgs << C[i].b_obs[j]+1 << " ";
						vector<pair<int, int>> tr;
						pair<int, int> n_ob;
						n_ob.first = pos;
						n_ob.second = j+1;
						if(p2tr.count(C[i].b_obs[j]))
						{
							tr = p2tr[C[i].b_obs[j]];
						}
						tr.push_back(n_ob);
						p2tr[C[i].b_obs[j]] = tr;
						points2D0.push_back(C[i].features[j]);
						points3D0.push_back(P.first.points[ P.first.ID_map.at( C[i].b_obs[j] ) ]);
					}
					else
						imgs << -1 << " ";
				}

				//residualCheck.Residuals(points2D0, points3D0, p2, &residuals_2);
				/*for(int i=0;i<residuals_2.size();i++)
					cout << residuals_2[i] << " ";*/
				
				imgs << "\n";
				c2m << "-1\n";
				pos++;
			}
			if((mode%4) != 2)
			{
				trans_s M = motion[(C[i].take)-1];
				Eigen::Matrix3d R = C[i].R * M.R;//.transpose();
				Eigen::Vector3d c = M.R.transpose() * (C[i].c - M.o);//M.R * C[i].c + M.o;
				Eigen::Vector4d q2 = RotationMatrixToQuaternion(R);
				Eigen::Vector3d t2 = -R * c;

				std::vector<Eigen::Vector2d> points2D0;
				std::vector<Eigen::Vector3d> points3D0;
				Eigen::Matrix3x4d p2;
				p2.leftCols<3>() = R;
				p2.rightCols<1>() = t2;
				Eigen::Matrix3d K;
				//cout << K << "\n";
				K.setZero();
				K(0,0) = C[i].f;
				K(1,1) = C[i].f;
				K(2,2) = 1;
				K(0,2) = C[i].px;
				K(1,2) = C[i].py;
				p2 = K*p2;
				std::vector<double> residuals_2;
				
				//cout << M.o.transpose() << "\n";
				imgs << pos << " " << q2(0) << " " << q2(1) << " " << q2(2) << " " << q2(3) << " " << t2(0) << " " << t2(1) << " " << t2(2) << " " << i+1 << " " << i+1 << ".JPG\n";
				for(unsigned int j=0;j<C[i].features.size();j++)
				{
					imgs << C[i].features[j](0) << " " << C[i].features[j](1) << " ";
					if(C[i].o_obs[j] != -1)
					{
						imgs << C[i].o_obs[j]+1 << " ";
						vector<pair<int, int>> tr;
						pair<int, int> n_ob;
						n_ob.first = pos;
						n_ob.second = j+1;
						if(p2tr.count(C[i].o_obs[j]))
						{
							tr = p2tr[C[i].o_obs[j]];
						}
						tr.push_back(n_ob);
						p2tr[C[i].o_obs[j]] = tr;
						points2D0.push_back(C[i].features[j]);
						points3D0.push_back(P.second.points[ P.second.ID_map.at( C[i].o_obs[j] ) ]);
					}
					else
						imgs << -1 << " ";
				}
				//residualCheck.Residuals(points2D0, points3D0, p2, &residuals_2);
				//for(int i=0;i<residuals_2.size();i++) cout << residuals_2[i] << " ";
				//cout << "\n";
				pos++;
				imgs << "\n";
				if(C[i].take < ref+1)
					c2m << C[i].take << "\n";
				else
					c2m << C[i].take-1 << "\n";
			}
		}
	}

	imgs.close();
	c2m.close();

	cout << "Saving motions\n";

	ofstream mot;
	string p4 = "model" + to_string(mode) + "/motions.txt";
	mot.open(p4);
	for(unsigned int i=0;i<motion.size();i++)
	{
		if(i!=(unsigned int)ref)
			mot << r2a(motion[i].R).transpose() << " " << motion[i].o.transpose() << "\n";
	}
	mot.close();

	cout << "Saving points\n";
	/*int bo = 0;
	int vbo = 0;*/

	ofstream pnts;
	string p5 = "model" + to_string(mode) + "/points3d.txt";
	pnts.open(p5);
	pnts << "# 3D point list with one line of data per point:\n";
	pnts << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
	pnts << "# Number of points: " << P.first.points.size() + P.second.points.size() <<", mean track length: 11.69\n";
	if((mode%4) != 3)
	{
		for(unsigned int i=0;i<P.first.points.size();i++)
		{
			int r = 255;
			int g = 0;
			int b = 0;
			if((mode%4))
			{
				r = P.first.color[i](0);
				g = P.first.color[i](1);
				b = P.first.color[i](2);
			}
			pnts << P.first.ID[i]+1 << " " << P.first.points[i](0) << " " << P.first.points[i](1) << " " << P.first.points[i](2) << " " << r << " " << g << " " << b << " 0 ";
			if(p2tr.count(P.first.ID[i]))
			{
				vector<pair<int, int>> tr = p2tr[P.first.ID[i]];
				for(unsigned int j=0;j<tr.size();j++)
				{
					pnts << tr[j].first << " " << tr[j].second << " ";
				}
				//TODO
				//experimental
				//pnts << "1 1 2 2 3 3 ";
			}
			//experimental
			//pnts << "1 1 2 2 ";
			pnts << "\n";
		}
	}
	if((mode%4) != 2)
	{
		for(unsigned int i=0;i<P.second.points.size();i++)
		{
			int r = 0;
			int g = 255;
			int b = 0;
			if((mode%4))
			{
				r = P.second.color[i](0);
				g = P.second.color[i](1);
				b = P.second.color[i](2);
			}
			/*if(!p2tr.count(P.second.ID[i]) || p2tr[P.second.ID[i]].size() <= 2)
			{
				r = 0;
				g = 0;
				b = 255;
				bo++;
				if(!p2tr.count(P.second.ID[i]) || p2tr[P.second.ID[i]].size() <= 1)
				{
					vbo++;
					r = 255;
					g = 0;
					b = 255;
				}
			}*/
			pnts << P.second.ID[i]+1 << " " << P.second.points[i](0) << " " << P.second.points[i](1) << " " << P.second.points[i](2) << " " << r << " " << g << " " << b << " 1 ";
			if(p2tr.count(P.second.ID[i]))
			{
				vector<pair<int, int>> tr = p2tr[P.second.ID[i]];
				for(unsigned int j=0;j<tr.size();j++)
				{
					pnts << tr[j].first << " " << tr[j].second << " ";
				}
				//TODO
				//experimental
				//pnts << "1 1 2 2 3 3 ";
			}
			//HERE_
			//experimental
			pnts << "1 1 ";
			pnts << "\n";
		}
	}
	//cout << "BLUE POINTS " << bo << "\n";
	//cout << "VERY BLUE POINTS " << vbo << "\n";
	
	pnts.close();

	//for(int i=0;i<P)
}

void step1()
{
	//this will be added to the colmap
	ifstream pivot;
	pivot.open("pivot.txt");
	int takes = 0;
	pivot >> takes;
	pivot.close();
	cout << takes << "\n";

	//create files cams and center
	ofstream cams;
	cams.open("cams.txt");
	ofstream center;
	center.open("center.txt");

	if(takes == 0)
		takes = 1;
	for(int i=1;i<=takes;i++)
	{
		cout << i << "\n";
		//work with the i-th folder (and reconstruction)

		//load points3D
		unordered_map<int,Eigen::Vector3d> points;
		ifstream pnts;
		string p1 = to_string(i) + "/points3D.txt";
		//cout << p1 << "\n";
		pnts.open(p1);

		string line;
		while(getline(pnts,line))
		{
			//cout << line << "\n";
			istringstream str(line);
			int id;
			str >> id;
			
			Eigen::Vector3d p;
			str >> p(0);
			str >> p(1);
			str >> p(2);
			//cout << p.x << " " << p.y << " " << p.z << "\n";
			points[id] = p;
		}

		pnts.close();

		//load images
		unordered_map<int,Eigen::Vector3d> centerpoints;
		string p2 = to_string(i) + "/images.txt";
		ifstream imgs;
		imgs.open(p2);

		for(int j=0;j<4;j++)
			getline(imgs,line);

		while(getline(imgs,line))
		{
			int id;
			istringstream str(line);
			str >> id;
			//cout << id << "\n";
			getline(imgs,line);
			istringstream str2(line);
			double num;
			double x_sum = 0;
			double y_sum = 0;
			double z_sum = 0;
			double count = 0;
			while(str2 >> num)
			{
				str2 >> num;
				int pnt_id;
				str2 >> pnt_id;
				if(pnt_id == -1)
					continue;
				//cout << pnt_id << "\n";
				count++;
				Eigen::Vector3d pnt = points[pnt_id];
				x_sum += pnt(0);
				y_sum += pnt(1);
				z_sum += pnt(2);
				//cout << x_sum << "\n";
			}
			//cout << "\n";
			//cout << x_sum << "\n";
			//cout << "\n\n";
			//cout << count << "\n";
			Eigen::Vector3d centerpoint;
			centerpoint(0) = (x_sum/count);
			centerpoint(1) = (y_sum/count);
			centerpoint(2) = (z_sum/count);
			centerpoints[id] = centerpoint;
			//cout << centerpoint.x << "\n\n\n";
		}
		imgs.close();

		//copy the cams file and add centerpoints to file center.txt
		ifstream c;
		string p3 = to_string(i) + "/cams.txt";
		c.open(p3);
		while(getline(c,line))
		{
			cams << line << "\n";
			istringstream str(line);
			double num;
			for(int j=0;j<16;j++)
			{
				str >> num;
			}
			int id;
			str >> id;
			//cout << id << "\n";
			Eigen::Vector3d cp = centerpoints[id];
			//cout << cp.x << "\n";
			center << cp(0) << " " << cp(1) << " " << cp(2) << "\n";
		}
		c.close();
		
	}
	cams.close();
	center.close();

	//CREATE TRACKS OF 3D POINTS
	//for each 3D point find neighbouring points, which share an observation with the 3D point
	vector<node_s> nodes;
	vector<unordered_map<int,int>> id2pos;

	//load 3D points from a file
	for(int i=1;i<=takes;i++)
	{
		ifstream pnts;
		string p1 = to_string(i) + "/points3D.txt";
		pnts.open(p1);
		unordered_map<int,int> cur_id2pos;

		string line;
		for(int j=0;j<3;j++)
			getline(pnts,line);
		while(getline(pnts,line))
		{
			//cout << line << "\n";
			istringstream str(line);
			int id;
			str >> id;
			
			//cout << i << " " << id << "\n";
			//cout << nodes.size();
			point_id nid;
			nid.take = i;
			nid.id = id;
			nid.position = nodes.size();
			node_s nnd;
			nnd.id = nid;
			nnd.track = 0;

			cur_id2pos[id] = nodes.size();
			nodes.push_back(nnd);
		}
		
		pnts.close();
		id2pos.push_back(cur_id2pos);
	}

	//for each image and take load observations
	vector<vector<vector<int>>> images;
	unordered_map<string,int> img_id2pos;
	vector<int> obs_size;
	for(int i=1;i<=takes;i++)
	{
		cout << i << "\n";
		string line;
		string p2 = to_string(i) + "/images.txt";
		ifstream imgs;
		imgs.open(p2);

		for(int j=0;j<4;j++)
			getline(imgs,line);

		while(getline(imgs,line))
		{
			int id;
			/*if(id%20)
				cout << id << "\n";*/
			istringstream str(line);
			str >> id;
			double num;
			for(int j=0;j<8;j++)
				str >> num;
			string name;
			str >> name;

			if(!img_id2pos.count(name))
			{
				vector<vector<int>> ni;
				images.push_back(ni);
				img_id2pos[name] = images.size()-1;
				obs_size.push_back(0);
			}

			int pos = img_id2pos[name];
			
			while(images.at(pos).size() < (unsigned int)i)
			{
				vector<int> obs;
				images.at(pos).push_back(obs);
			}

			
			getline(imgs,line);
			istringstream str2(line);

			//cout << images.at(pos).size() << " " << i << "\n";
			
			if(images.at(pos).at(i-1).size())
			{
				int position = 0;
				while(str2 >> num)
				{
					str2 >> num;
					int pnt_id;
					str2 >> pnt_id;
					if (pnt_id >= 0)
						images.at(pos).at(i-1).at(position) = pnt_id;
					position++;
				}
			}
			else
			{
				int position = 0;
				//vector<int> obs;
				while(str2 >> num)
				{
					str2 >> num;
					int pnt_id;
					str2 >> pnt_id;
					images.at(pos).at(i-1).push_back(pnt_id);
					//obs.push_back(pnt_id);
					position++;
				}
				obs_size.at(pos) = position;
			}
		}
		imgs.close();
	}

	//for each point find its neighbours
	for(unsigned int i=0;i<images.size();i++)
	{
		if(!(i%10))
			cout << i << "\n";
		/*cout << images.at(i).at(0).size() << "\n";
		cout << obs_size.at(i) << "\n";*/
		for(int j=0;j<obs_size.at(i);j++)
		{
			
			for(int t=0;t<takes;t++)
			{
				if(images.at(i).size() <= (unsigned int)t)
					break;
				if(images.at(i).at(t).size() == 0)
					continue;
				if(images.at(i).at(t).at(j) == -1)
					continue;
				for(int u=0;u<takes;u++)
				{
					if(images.at(i).size() <= (unsigned int)u)
						break;
					if(images.at(i).at(u).size() == 0)
						continue;
					if(t==u)
						continue;
					if(images.at(i).at(u).at(j) == -1)
						continue;
					//cout << t << " " << images.at(i).at(t).at(j) << " " << u << " " << images.at(i).at(u).at(j) << "\n";
					point_id npt;
					npt.id = images.at(i).at(u).at(j);
					npt.take = u+1;
					npt.position = id2pos.at(u)[npt.id];
					npt.strength = 1;
					const int sz = nodes.at(id2pos.at(t)[images.at(i).at(t).at(j)]).neighbours.size();
					bool found = 0;
					for(int k=0;k<sz;k++)
					{
						if(nodes.at(id2pos.at(t)[images.at(i).at(t).at(j)]).neighbours.at(k).position == npt.position)
						{
							nodes.at(id2pos.at(t)[images.at(i).at(t).at(j)]).neighbours.at(k).strength = nodes.at(id2pos.at(t)[images.at(i).at(t).at(j)]).neighbours.at(k).strength+1;
							found = 1;
							break;
						}
					}
					if(!found)
						nodes.at(id2pos.at(t)[images.at(i).at(t).at(j)]).neighbours.push_back(npt);
				}
				/*cout << t << "\n";
				cout << images.at(i).at(t).at(j) << "\n";*/
			}
		}
	}

	//merge the neighbouring 3D points into tracks
	ofstream trac;
	trac.open("tracks.txt");
	ofstream grap;
	grap.open("graph.txt");
	int track_id = 1;
	//int neg_track_id = -1;
	for(unsigned int i=0;i<nodes.size();i++)
	{
		vector<int> cur_track;
		if(!(track_id % 1000))
			cout << track_id << "\n";
		if(nodes.at(i).track)
		{
			continue;
		}
		nodes.at(i).track = track_id;
		queue<int> q;
		//cout << nodes.at(i).id.position << "\n";
		q.push(i);
		cur_track.push_back(i);
		while(!q.empty())
		{
			int cur = q.front();
			q.pop();
			for(unsigned int j=0;j<nodes.at(cur).neighbours.size();j++)
			{
				int nw = nodes.at(cur).neighbours.at(j).position;
				if(!nodes.at(nw).track)
				{
					//cout << "T" << nodes.at(nw).track << "\n";
					nodes.at(nw).track = track_id;
					q.push(nw);
					cur_track.push_back(nw);
				}
			}
		}
		//check track consistency
		bool consistent = 1;
		for(unsigned int j=0;j<cur_track.size();j++)
		{
			for(unsigned int k=j+1;k<cur_track.size();k++)
			{
				if(nodes.at(cur_track.at(j)).id.take == nodes.at(cur_track.at(k)).id.take)
				{
					consistent = 0;
					//cout << "INCONSISTENT\n";
					break;
				}
			}
			if(!consistent)
				break;
		}
		if(consistent)
		{
			for(unsigned int j=0;j<cur_track.size();j++)
			{
				trac << nodes.at(cur_track.at(j)).id.take << " " << nodes.at(cur_track.at(j)).id.id << " ";
			}
			trac << "\n";
		}
		else
		{
			for(unsigned int j=0;j<cur_track.size();j++)
			{
				grap << nodes.at(cur_track.at(j)).id.take << " " << nodes.at(cur_track.at(j)).id.id << " ";
			}
			grap << "\n";
			for(unsigned int j=0;j<cur_track.size();j++)
			{
				int tj = cur_track.at(j);
				for(unsigned int k=0;k<cur_track.size();k++)
				{
					int tk = cur_track.at(k);
					if(tj==tk)
						grap << "0 ";
					else
					{
						bool found = 0;
						for(unsigned int a=0;a<nodes.at(tj).neighbours.size();a++)
						{
							if(nodes.at(tj).neighbours.at(a).position == nodes.at(tk).id.position)
							{
								grap << nodes.at(tj).neighbours.at(a).strength << " ";
								found = 1;
								break;
							}
						}
						if(!found)
							grap << "0 ";
					}
				}
				grap << "\n";
			}
			grap << "\n";
		}
		
		track_id++;
	}

	trac.close();
	grap.close();
}

void check(std::vector<imgs_s> I, std::vector<pnts_s> P, std::vector<cam_s> C, std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::unordered_map<int, int>>> T, std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> O)
{
	for(unsigned int i=0;i<C.size();i++)
	{
		EPNPEstimator residualCheck;
		std::vector<Eigen::Vector2d> points2D0;
		std::vector<Eigen::Vector3d> points3D0;
		Eigen::Matrix3x4d p2;
		p2.leftCols<3>() = C[i].R;
		p2.rightCols<1>() = C[i].t;
		Eigen::Matrix3d K;
		K.setZero();
		cout << K << "\n";
		K(0,0) = C[i].f;
		K(1,1) = C[i].f;
		K(2,2) = 1;
		K(0,2) = C[i].px;
		K(1,2) = C[i].py;
		K(0,1) = -0.04;
		p2 = K*p2;
		std::vector<double> residuals_2;
		cout << K << "\n\n" << p2 << "\n\n";
		int take = C[i].anchor;
		cout << i << " " << C[i].anchor << " " << C[i].second << " " << C[i].id << " " << C[i].id_2 << "\n";

		vector<int> obs = O[i].first;
		vector<Eigen::Vector2d> feat = O[i].second;

		for(unsigned int j=0;j<obs.size();j++)
		{
			if(obs[j] >= 0)
			{
				points2D0.push_back(feat[j]);
				//Eigen::Vector3d pnt = P[take-1].points[ P[take-1].ID_map.at(obs[j]) ];
				Eigen::Vector3d pnt;
				std::vector<std::pair<int, int>> tr = T.first[obs[j]];
				for(unsigned int k=0;k<tr.size();k++)
				{
					if(take == tr[k].first)
					{
						pnt = P[tr[k].first-1].points[ P[take-1].ID_map.at(tr[k].second) ];
						break;
					}
					if(k == tr.size()-1) cout << "YZYZ\n\n";
				}
				points3D0.push_back( pnt );
				//cout << feat[j].transpose() << " " << pnt.transpose() << " ";
			}
		}
		cout << "\n\n";
		residualCheck.Residuals(points2D0, points3D0, p2, &residuals_2);
		for(unsigned int i=0;i<residuals_2.size();i++) cout << residuals_2[i] << " ";
		cout << "\n";
	}
	/*for(int i=0;i<C.size();i++)
	{
		EPNPEstimator residualCheck;
		std::vector<Eigen::Vector2d> points2D0;
		std::vector<Eigen::Vector3d> points3D0;
		Eigen::Matrix3x4d p2;
		p2.leftCols<3>() = C[i].R;
		p2.rightCols<1>() = C[i].t;
		Eigen::Matrix3d K;
		K.setZero();
		cout << K << "\n";
		K(0,0) = C[i].f;
		K(1,1) = C[i].f;
		K(2,2) = 1;
		K(0,2) = C[i].px;
		K(1,2) = C[i].py;
		p2 = K*p2;
		std::vector<double> residuals_2;
		cout << K << "\n\n" << p2 << "\n\n";

		int take = C[i].anchor;
		int id = C[i].id_2;

		cout << id << " " << I[take-1].map.at(id) << "\n"; 
		
		vector<int> obs = I[take-1].obs_all[ I[take-1].map.at(id) ];
		vector<Eigen::Vector2d> feat = I[take-1].features[ I[take-1].map.at(id) ];
		for(int j=0;j<obs.size();j++)
		{
			if(obs[j] >= 0)
			{
				points2D0.push_back(feat[j]);
				Eigen::Vector3d pnt = P[take-1].points[ P[take-1].ID_map.at(obs[j]) ];
				points3D0.push_back( pnt );
				//cout << feat[j] << " " << pnt.transpose() << " ";
			}
		}
		residualCheck.Residuals(points2D0, points3D0, p2, &residuals_2);
		for(int i=0;i<residuals_2.size();i++) cout << residuals_2[i] << " ";
		cout << "\n";
		
	}*/
}

struct CostFunctor
{
	CostFunctor(double obs_x, double obs_y) : obs_x(obs_x), obs_y(obs_y) {}
	template <typename T>
	bool operator()(const T* const camera, const T* const point, T* residuals) const
	{
		T p[3];
		ceres::AngleAxisRotatePoint(camera, point, p);
		p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
		T xp = p[0] / p[2];
	    T yp = p[1] / p[2];
	    const T& focal = camera[6];
		T predicted_x = focal * xp;
		T predicted_y = focal * yp;
		residuals[0] = predicted_x - T(obs_x);
	    residuals[1] = predicted_y - T(obs_y);
	    return true;
	}
	double obs_x;
	double obs_y;
};

struct CostFunctor2
{
	CostFunctor2(double obs_x, double obs_y) : obs_x(obs_x), obs_y(obs_y) {}
	template <typename T>
	bool operator()(const T* const camera, const T* const point, const T* const motion, T* residuals) const
	{
		T p0[3];
		ceres::AngleAxisRotatePoint(motion, point, p0);
		p0[0] += motion[3]; p0[1] += motion[4]; p0[2] += motion[5];
		T p[3];
		ceres::AngleAxisRotatePoint(camera, p0, p);
		p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
		T xp = p[0] / p[2];
	    T yp = p[1] / p[2];
	    const T& focal = camera[6];
		T predicted_x = focal * xp;
		T predicted_y = focal * yp;
		residuals[0] = predicted_x - T(obs_x);
	    residuals[1] = predicted_y - T(obs_y);
	    return true;
	}
	double obs_x;
	double obs_y;
};

double test1(const double* const camera, const double* const point, double obs_x, double obs_y)
{
	double p[3];
	ceres::AngleAxisRotatePoint(camera, point, p);
	p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
	double xp = /*-*/ p[0] / p[2];
    double yp = /*-*/ p[1] / p[2];
    const double& focal = camera[6];
	double predicted_x = focal * xp;
	double predicted_y = focal * yp;
	return (predicted_x - obs_x)*(predicted_x - obs_x) + (predicted_y - obs_y)*(predicted_y - obs_y);
}

double test2(const double* const camera, const double* const point, const double* const motion, double obs_x, double obs_y)
{
	double p0[3];
	ceres::AngleAxisRotatePoint(motion, point, p0);
	p0[0] += motion[3]; p0[1] += motion[4]; p0[2] += motion[5];
	double p[3];
	ceres::AngleAxisRotatePoint(camera, p0, p);
	p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
	double xp = /*-*/ p[0] / p[2];
    double yp = /*-*/ p[1] / p[2];
    const double& focal = camera[6];
	double predicted_x = focal * xp;
	double predicted_y = focal * yp;
	//cout << motion[0] << " " << motion[1] << " " << motion[2] << " " << motion[3] << " " << motion[4] << " " << motion[5] << "\n";
	return (predicted_x - obs_x)*(predicted_x - obs_x) + (predicted_y - obs_y)*(predicted_y - obs_y);
}

void perform_BA(std::pair<pnts_s, pnts_s> &P, std::vector<img_s> &C, std::vector<trans_s> &motion, int mode, int ref)
{
	//2 cfs will be necessary, one for the normal cams and the other one for the object cams (with parameters of the camera + the motions)
	
	//create a list (array, probably a classic one) with parameters
	//fill it with the parameters
	//cameras, points
	vector<double> cams;
	vector<double> motions;
	vector<double> points1;
	vector<double> points2;
	for(unsigned int i=0;i<C.size();i++)
	{
		//aa(3), t(3), f(1), /*px(1), py(1)*/ - fix at least the px, py, nothing to change there, maybe also the focal length
		Eigen::Vector3d aa = r2a(C[i].R);
		cams.push_back(aa(0));
		cams.push_back(aa(1));
		cams.push_back(aa(2));
		Eigen::Vector3d t = -1 * C[i].R * C[i].c;
		cams.push_back(t(0));
		cams.push_back(t(1));
		cams.push_back(t(2));
		cams.push_back(C[i].f);
		//cams.push_back(C[i].px);
		//cams.push_back(C[i].py);
	}
	for(unsigned int i=0;i<motion.size();i++)
	{
		//aa(3), t(3)
		Eigen::Vector3d aa = r2a(motion[i].R.transpose());
		motions.push_back(aa(0));
		motions.push_back(aa(1));
		motions.push_back(aa(2));
		Eigen::Vector3d t = -1 * motion[i].R.transpose() * motion[i].o;
		motions.push_back(t(0));
		motions.push_back(t(1));
		motions.push_back(t(2));
	}
	for(unsigned int i=0;i<P.first.points.size();i++)
	{
		//point(3)
		Eigen::Vector3d p = P.first.points[i];
		points1.push_back(p(0));
		points1.push_back(p(1));
		points1.push_back(p(2));
	}
	for(unsigned int i=0;i<P.second.points.size();i++)
	{
		//point(3)
		Eigen::Vector3d p = P.second.points[i];
		points2.push_back(p(0));
		points2.push_back(p(1));
		points2.push_back(p(2));
	}
	double * cams_ = cams.data();
	double * motions_ = motions.data();
	double * points1_ = points1.data();
	double * points2_ = points2.data();

	ceres::Problem problem;
	//for each observation (camera, feature, point) add a residual block
	//a cost function is necessary
	//subtract px, py from the observations
	for(unsigned int i=0;i<C.size();i++)
	{
		//background observations (use the simple reprojection error)
		vector<Eigen::Vector2d> feat = C[i].features;
		vector<int> obs = C[i].b_obs;
		double * camera = cams_ + 7*i;
		for(unsigned int j=0;j<obs.size();j++)
		{
			if(obs[j] == -1) continue;
			if(!P.first.ID_map.count(obs[j])) {cout << "NK\n"; continue;}
			int pos = P.first.ID_map[obs[j]];
			double * point = points1_ + 3*pos;
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 2, 7, 3>(new CostFunctor(feat[j](0)-C[i].px, feat[j](1)-C[i].py));
			ceres::LossFunction* loss_function = new ceres::HuberLoss(30.0);
			problem.AddResidualBlock(cost_function, loss_function, camera, point);
			//cout << test1(camera, point, feat[j](0)-C[i].px, feat[j](1)-C[i].py) << "\n";
		}

		//object observations (if the take is not reference, use the complex reprojection error)
		obs = C[i].o_obs;
		if(C[i].take == ref+1)
		{
			for(unsigned int j=0;j<obs.size();j++)
			{
				if(obs[j] == -1) continue;
				if(!P.second.ID_map.count(obs[j])) {cout << "NK\n"; continue;}
				int pos = P.second.ID_map[obs[j]];
				double * point = points2_ + 3*pos;
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 2, 7, 3>(new CostFunctor(feat[j](0)-C[i].px, feat[j](1)-C[i].py));
				ceres::LossFunction* loss_function = new ceres::HuberLoss(30.0);
				problem.AddResidualBlock(cost_function, loss_function, camera, point);
				//test1(camera, point, feat[j](0)-C[i].px, feat[j](1)-C[i].py);
			}
		}
		else
		{
			double * mot = motions_ + 6 * (C[i].take-1);
			for(unsigned int j=0;j<obs.size();j++)
			{
				if(obs[j] == -1) continue;
				if(!P.second.ID_map.count(obs[j])) {cout << "NK\n"; continue;}
				int pos = P.second.ID_map[obs[j]];
				double * point = points2_ + 3*pos;
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor2, 2, 7, 3, 6>(new CostFunctor2(feat[j](0)-C[i].px, feat[j](1)-C[i].py));
				ceres::LossFunction* loss_function = new ceres::HuberLoss(30.0);
				problem.AddResidualBlock(cost_function, loss_function, camera, point, mot);
				//test2(camera, point, mot, feat[j](0)-C[i].px, feat[j](1)-C[i].py);
			}
		}
		
	}
	//ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor(11,11));
	//problem.AddResidualBlock(cost_function, NULL, &x);

	//solve the problem (use a sparse solver)
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	//options.max_num_iterations = 1;
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	//save the points back
	for(unsigned int i=0;i<C.size();i++)
	{
		double * camera = cams_ + 7*i;
		Eigen::Vector3d aa;
		aa(0) = camera[0];
		aa(1) = camera[1];
		aa(2) = camera[2];
		Eigen::Matrix3d R = a2r(aa);
		C[i].R = R;
		Eigen::Vector3d t;
		t(0) = camera[3];
		t(1) = camera[4];
		t(2) = camera[5];
		C[i].c = -1 * R.transpose() * t;
		C[i].f = camera[6];
		
	}
	for(unsigned int i=0;i<motion.size();i++)
	{
		double * mot = motions_ + 6 * i;
		Eigen::Vector3d aa;
		aa(0) = mot[0];
		aa(1) = mot[1];
		aa(2) = mot[2];
		motion[i].R = a2r(aa);
		Eigen::Vector3d t;
		t(0) = mot[3];
		t(1) = mot[4];
		t(2) = mot[5];
		motion[i].o = t;
	}
	for(unsigned int i=0;i<P.first.points.size();i++)
	{
		double * point = points1_ + 3*i;
		Eigen::Vector3d pnt;
		pnt(0) = point[0];
		pnt(1) = point[1];
		pnt(2) = point[2];
		P.first.points[i] = pnt;
	}
	for(unsigned int i=0;i<P.second.points.size();i++)
	{
		double * point = points2_ + 3*i;
		Eigen::Vector3d pnt;
		pnt(0) = point[0];
		pnt(1) = point[1];
		pnt(2) = point[2];
		P.second.points[i] = pnt;
	}
}

void perform_BA_alter(std::pair<pnts_s, pnts_s> &P, std::vector<img_s> &C, std::vector<trans_s> &motion, int mode, int ref)
{
	//2 cfs will be necessary, one for the normal cams and the other one for the object cams (with parameters of the camera + the motions)
	
	//create a list (array, probably a classic one) with parameters
	//fill it with the parameters
	//cameras, points
	vector<double> cams;
	vector<double> motions;
	vector<double> points1;
	vector<double> points2;
	for(unsigned int i=0;i<C.size();i++)
	{
		//aa(3), t(3), f(1), /*px(1), py(1)*/ - fix at least the px, py, nothing to change there, maybe also the focal length
		Eigen::Vector3d aa = r2a(C[i].R);
		cams.push_back(aa(0));
		cams.push_back(aa(1));
		cams.push_back(aa(2));
		Eigen::Vector3d t = -1 * C[i].R * C[i].c;
		cams.push_back(t(0));
		cams.push_back(t(1));
		cams.push_back(t(2));
		cams.push_back(C[i].f);
		//cams.push_back(C[i].px);
		//cams.push_back(C[i].py);
	}
	for(unsigned int i=0;i<motion.size();i++)
	{
		//aa(3), t(3)
		Eigen::Vector3d aa = r2a(motion[i].R.transpose());
		motions.push_back(aa(0));
		motions.push_back(aa(1));
		motions.push_back(aa(2));
		Eigen::Vector3d t = -1 * motion[i].R.transpose() * motion[i].o;
		motions.push_back(t(0));
		motions.push_back(t(1));
		motions.push_back(t(2));
	}
	for(unsigned int i=0;i<P.first.points.size();i++)
	{
		//point(3)
		Eigen::Vector3d p = P.first.points[i];
		points1.push_back(p(0));
		points1.push_back(p(1));
		points1.push_back(p(2));
	}
	for(unsigned int i=0;i<P.second.points.size();i++)
	{
		//point(3)
		Eigen::Vector3d p = P.second.points[i];
		points2.push_back(p(0));
		points2.push_back(p(1));
		points2.push_back(p(2));
	}
	double * cams_ = cams.data();
	double * motions_ = motions.data();
	double * points1_ = points1.data();
	double * points2_ = points2.data();
	vector<bool> pf1(points1.size());
	
	ceres::Problem problem;
	//problem.AddParameterBlock(cams_, /*cams.size()**/7);
	//problem.AddParameterBlock(motions_, /*motions.size()**/6);
	//problem.AddParameterBlock(points1_, /*points1.size()**/3);
	//problem.AddParameterBlock(points2_, /*points2.size()**/3);
	//for each observation (camera, feature, point) add a residual block
	//a cost function is necessary
	//subtract px, py from the observations
	//fix the points
	/*int c1 = 0;
	int c2 = 0;
	int c3 = 0;*/
	for(unsigned int i=0;i<C.size();i++)
	{
		//background observations (use the simple reprojection error)
		vector<Eigen::Vector2d> feat = C[i].features;
		vector<int> obs = C[i].b_obs;
		double * camera = cams_ + 7*i;
		//problem2.AddParameterBlock(camera, 7);
		//problem2.SetParameterBlockConstant(camera);
		for(unsigned int j=0;j<obs.size();j++)
		{
			if(obs[j] == -1) continue;
			if(!P.first.ID_map.count(obs[j])) {cout << "NK\n"; continue;}
			int pos = P.first.ID_map[obs[j]];
			double * point = points1_ + 3*pos;
			pf1[pos] = 1;
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 2, 7, 3>(new CostFunctor(feat[j](0)-C[i].px, feat[j](1)-C[i].py));
			ceres::LossFunction* loss_function = new ceres::HuberLoss(30.0);
			//problem.AddParameterBlock(camera,7);
			//double cerr = test1(camera, point, feat[j](0)-C[i].px, feat[j](1)-C[i].py);
			//if(cerr < 10000) 
				problem.AddResidualBlock(cost_function, loss_function, camera, point);
		}

		//object observations (if the take is not reference, use the complex reprojection error)
		obs = C[i].o_obs;
		if(C[i].take == ref+1)
		{
			for(unsigned int j=0;j<obs.size();j++)
			{
				if(obs[j] == -1) continue;
				if(!P.second.ID_map.count(obs[j])) {cout << "NK\n"; continue;}
				int pos = P.second.ID_map[obs[j]];
				double * point = points2_ + 3*pos;
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 2, 7, 3>(new CostFunctor(feat[j](0)-C[i].px, feat[j](1)-C[i].py));
				ceres::LossFunction* loss_function = new ceres::HuberLoss(30.0);
				//double cerr = test1(camera, point, feat[j](0)-C[i].px, feat[j](1)-C[i].py);
				//if(cerr < 10000)
					problem.AddResidualBlock(cost_function, loss_function, camera, point);
			}
		}
		else
		{
			double * mot = motions_ + 6 * (C[i].take-1);
			for(unsigned int j=0;j<obs.size();j++)
			{
				if(obs[j] == -1) continue;
				if(!P.second.ID_map.count(obs[j])) {cout << "NK\n"; continue;}
				int pos = P.second.ID_map[obs[j]];
				double * point = points2_ + 3*pos;
				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor2, 2, 7, 3, 6>(new CostFunctor2(feat[j](0)-C[i].px, feat[j](1)-C[i].py));
				ceres::LossFunction* loss_function = new ceres::HuberLoss(30.0);
				//double cerr = test2(camera, point, mot, feat[j](0)-C[i].px, feat[j](1)-C[i].py);
				//if(cerr < 10000) 
					problem.AddResidualBlock(cost_function, loss_function, camera, point, mot);
			}
		}
		
	}

	for(unsigned int i=0;i<P.first.points.size();i++)
	{
		double * point = points1_ + 3*i;
		problem.AddParameterBlock(point, 3);
	}
	for(unsigned int i=0;i<P.second.points.size();i++)
	{
		double * point = points2_ + 3*i;
		problem.AddParameterBlock(point, 3);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 1;
	ceres::Solver::Summary summary;

	for(int k=0;k<20;k++)
	{
		cout << "iteration " << k << "\n";
		for(unsigned int i=0;i<C.size();i++)
		{
			double * camera = cams_ + 7*i;
			problem.SetParameterBlockVariable(camera);
			
		}
		for(unsigned int i=0;i<motion.size();i++)
		{
			if(i==(unsigned int)ref) continue;
			double * mot = motions_ + 6*i;
			problem.SetParameterBlockVariable(mot);
			
		}
		for(unsigned int i=0;i<P.first.points.size();i++)
		{
			double * point = points1_ + 3*i;
			problem.SetParameterBlockConstant(point);
		}
		for(unsigned int i=0;i<P.second.points.size();i++)
		{
			double * point = points2_ + 3*i;
			problem.SetParameterBlockConstant(point);
		}
		Solve(options, &problem, &summary);
		
		for(unsigned int i=0;i<C.size();i++)
		{
			double * camera = cams_ + 7*i;
			problem.SetParameterBlockConstant(camera);
			
		}
		for(unsigned int i=0;i<motion.size();i++)
		{
			if(i==(unsigned int)ref) continue;
			double * mot = motions_ + 6*i;
			problem.SetParameterBlockConstant(mot);
		}
		for(unsigned int i=0;i<P.first.points.size();i++)
		{
			double * point = points1_ + 3*i;
			problem.SetParameterBlockVariable(point);
		}
		for(unsigned int i=0;i<P.second.points.size();i++)
		{
			double * point = points2_ + 3*i;
			problem.SetParameterBlockVariable(point);
		}
		//ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor(11,11));
		//problem.AddResidualBlock(cost_function, NULL, &x);

		//solve the problem (use a sparse solver)
		
		Solve(options, &problem, &summary);
	}

	cout << "K\n";

	//save the points back
	for(unsigned int i=0;i<C.size();i++)
	{
		double * camera = cams_ + 7*i;
		Eigen::Vector3d aa;
		aa(0) = camera[0];
		aa(1) = camera[1];
		aa(2) = camera[2];
		Eigen::Matrix3d R = a2r(aa);
		C[i].R = R;
		Eigen::Vector3d t;
		t(0) = camera[3];
		t(1) = camera[4];
		t(2) = camera[5];
		C[i].c = -1 * R.transpose() * t;
		C[i].f = camera[6];
		
	}
	cout << "K\n";
	for(unsigned int i=0;i<motion.size();i++)
	{
		double * mot = motions_ + 6 * i;
		Eigen::Vector3d aa;
		aa(0) = mot[0];
		aa(1) = mot[1];
		aa(2) = mot[2];
		motion[i].R = a2r(aa);
		Eigen::Vector3d t;
		t(0) = mot[3];
		t(1) = mot[4];
		t(2) = mot[5];
		motion[i].o = t;
	}
	cout << "K\n";
	for(unsigned int i=0;i<P.first.points.size();i++)
	{
		double * point = points1_ + 3*i;
		Eigen::Vector3d pnt;
		pnt(0) = point[0];
		pnt(1) = point[1];
		pnt(2) = point[2];
		P.first.points[i] = pnt;
	}
	cout << "K\n";
	for(unsigned int i=0;i<P.second.points.size();i++)
	{
		double * point = points2_ + 3*i;
		Eigen::Vector3d pnt;
		pnt(0) = point[0];
		pnt(1) = point[1];
		pnt(2) = point[2];
		P.second.points[i] = pnt;
	}
	cout << "F\n";
}

std::vector<Eigen::Vector3d> ransac_points(std::vector<Eigen::Vector3d> P, double pd, std::vector<int> tk)
{
	//TODO
	//utilize multi body RANSAC (however it seems the single body will be enough here, the difference is negligible)
	std::vector<Eigen::Vector3d> ret;
	int BC = 0;
	std::vector<Eigen::Vector3d> best;
	for(unsigned int i=0;i<P.size();i++)
	{
		int count = 0;
		std::vector<Eigen::Vector3d> cur;
		for(unsigned int j=0;j<P.size();j++)
		{
			if(tk[i] == tk[j]) continue;
			Eigen::Vector3d Q = P[i]-P[j];
			if(Q.norm() <= 0.16 * pd)
			{
				cur.push_back(P[j]);
				count++;
			}
			/*if(i==0 && j==1)
				cout << Q.norm() << " " << pd << "\n";*/
		}
		if(count > BC)
		{
			BC = count;
			best = cur;
		}
	}
	if(BC > 0)
		ret = best;
	return ret;
}

void new_points(std::vector<std::vector<std::pair<int, int>>> T, std::vector<pnts_s> P, std::vector<int> D, std::vector<std::pair<trans_s,trans_s>> M, int r, double pd)
{
	cout << "UNKNOWN " << D.size() << "\n";
	int count = 0;
	for(unsigned int i=0;i<D.size();i++)
	{
		if(T[D[i]].size() > 1)
		{
			//transform the points in the track to the reference take according to the background
			
			vector<Eigen::Vector3d> bg;
			vector<Eigen::Vector3d> fg;
			vector<int> tk;
			for(unsigned int j=0;j<T[D[i]].size();j++)
			{
				std::pair<int, int> pos = T[D[i]][j];
				Eigen::Vector3d OP = P[pos.first-1].points[P[pos.first-1].ID_map[pos.second]];
				Eigen::Vector3d NPB;
				Eigen::Vector3d NPF;
				trans_s M1 = M[pos.first-1].first;
				trans_s M2 = M[pos.first-1].second;
				if(r == pos.first)
				{
					NPB = OP;
					NPF = OP;
				}
				else
				{
					NPB = M1.s * M1.R * OP + M1.o;
					NPF = M2.s * M2.R * OP + M2.o;
				}
				/*cout << "NP\n";
				cout << pos.first << "\n";
				cout << NPB.transpose() << "\n";
				cout << NPF.transpose() << "\n";*/
				bg.push_back(NPB);
				fg.push_back(NPF);
				tk.push_back(pos.first);
			}
			//cout << "B\n";
			vector<Eigen::Vector3d> NB = ransac_points(bg, pd, tk);
			//cout << "F\n";
			vector<Eigen::Vector3d> NF = ransac_points(fg, pd, tk);
			if(NF.size()) count++;
		}
	}
	cout << "UNKNOWN " << count << "\n";
}

std::vector<std::vector<std::pair<int, int>>> load_2tracks(std::vector<img_s> &C, std::pair<pnts_s, pnts_s> &R, std::vector<trans_s> motion)
{
	vector<vector<pair<int, int>>> ret;
	vector<vector<pair<int, int>>> tracks;
	ifstream tf;
	tf.open("1/tracks.txt");
	int ccount;
	tf >> ccount;
	int count;
	tf >> count;
	int cnt = 0;
	for(int i=0;i<count;i++)
	{
		vector<pair<int, int>> track;
		vector<int> take;
		while(1)
		{
			int cam;
			tf >> cam;
			if(cam == -1) break;
			int feat;
			tf >> feat;
			pair<int, int> e;
			e.first = cam;
			e.second = feat;
			track.push_back(e);
			take.push_back(C[cam-1].take);
		}
		if(track.size() > 2)
		{
			//if it should be perspective, the track has to be long enough, from different takes and not triangulated (at least not in the final reconstruction)
			ret.push_back(track);
			bool ok = false;
			for(unsigned int j=1;j<track.size();j++)
			{
				if(take[j] != take[0])
				{
					ok = true;
					break;
				}
			}
			if(!ok) continue;

			vector<pair<int, int>> track2;
			//cout << "NT\n";
			//strategy 1: find unobserving tracks
			//strategy 2: select the parts of tracks which do not observe a point, filter those where only one take remains
			for(unsigned int j=0;j<track.size();j++)
			{
				pair<int, int> e = track[j];
				int ob = C[e.first-1].b_obs[e.second-1];
				int of = C[e.first-1].o_obs[e.second-1];
				if(ob >= 0 || of >= 0)
				{
					ok = false;
				}
				else
				{
					track2.push_back(e);
				}
				//cout << ob << " " << of << " ";
			}
			//cout << "\n";
			if(ok)
			{
				tracks.push_back(track);
			}
			else
			{
				bool ok2 = false;
				for(unsigned int j=1;j<track.size();j++)
				{
					if(track[j].first != track[0].first)
					{
						ok2 = true;
						break;
					}
				}
				if(ok2)
					tracks.push_back(track2);
			}
			cnt++;
		}
	}
	cout << "TRACKS " << cnt << "\n";
	tf.close();
	cnt = 0;

	//find max id of a point
	int max = 0;
	for(unsigned int i=0;i<R.first.ID.size();i++)
	{
		if(R.first.ID[i] > max)
			max = R.first.ID[i];
	}
	for(unsigned int i=0;i<R.second.ID.size();i++)
	{
		if(R.second.ID[i] > max)
			max = R.second.ID[i];
	}
	max++;

	//move to another function
	for(unsigned int t=0;t<tracks.size();t++)
	{
		//try triangulating the new point and find its support among other elements of the track
		//if the features from which the point is triangulated are from different takes, count every inlier
		//otherwise only those which are from different takes
		//try both foreground and background
		vector<pair<int, int>> track = tracks[t];
		Eigen::Vector3d best;
		int best_count = 0;
		double best_err = 0;
		bool best_obj = 0;
		vector<pair<int, int>> best_inl;

		//background
		for(unsigned int i=0;i<track.size();i++)
		{
			pair<int, int> e1 = track[i];
			Eigen::Vector2d feat_i1 = C[e1.first-1].features[e1.second-1];
			Eigen::Vector2d feat_w1;
			feat_w1(0) = (feat_i1(0) - C[e1.first-1].px)/C[e1.first-1].f;
			feat_w1(1) = (feat_i1(1) - C[e1.first-1].py)/C[e1.first-1].f;
			//compose the matrix according to the background
			Eigen::Matrix3d R1B = C[e1.first-1].R;
			Eigen::Vector3d t1B = -R1B * C[e1.first-1].c;
			Eigen::Matrix3x4d P1B;
			P1B.leftCols<3>() = R1B;
			P1B.rightCols<1>() = t1B;

			//compose the matrix according to the foreground
			trans_s M = motion[(C[e1.first-1].take)-1];
			Eigen::Matrix3d R1F = R1B * M.R.transpose();
			Eigen::Vector3d c1F = M.R * C[e1.first-1].c + M.o;
			Eigen::Vector3d t1F = -R1F * c1F;
			Eigen::Matrix3x4d P1F;
			P1F.leftCols<3>() = R1F;
			P1F.rightCols<1>() = t1F;
  				
			for(unsigned int j=0;j<track.size();j++)
			{
				if(i==j) continue;
				pair<int, int> e2 = track[j];
				/*bool diff_tk = 1;
				if(C[e1.first-1].take == C[e2.first-1].take)
					diff_tk = 0;*/
				//obtain the world coordinates
				Eigen::Vector2d feat_i2 = C[e2.first-1].features[e2.second-1];
				Eigen::Vector2d feat_w2;
				feat_w2(0) = (feat_i2(0) - C[e2.first-1].px)/C[e2.first-1].f;
				feat_w2(1) = (feat_i2(1) - C[e2.first-1].py)/C[e2.first-1].f;

				//triangulate the point (according to the background)
				//compose the matrix according to the background
				Eigen::Matrix3d R2B = C[e2.first-1].R;
				Eigen::Vector3d t2B = -R2B * C[e2.first-1].c;
				Eigen::Matrix3x4d P2B;
  				P2B.leftCols<3>() = R2B;
  				P2B.rightCols<1>() = t2B;

  				//triangulate the point according to the background
  				//cout << "B ";
  				Eigen::Vector3d TPB = TriangulatePoint(P1B, P2B, feat_w1, feat_w2);
  				int countB = 0;
				double errB = 0;
				vector<pair<int, int>> inlB;

				//compose the matrix according to the foreground
				trans_s M2 = motion[(C[e2.first-1].take)-1];
				Eigen::Matrix3d R2F = R2B * M2.R.transpose();
				Eigen::Vector3d c2F = M2.R * C[e2.first-1].c + M2.o;
				Eigen::Vector3d t2F = -R2F * c2F;
				Eigen::Matrix3x4d P2F;
				P2F.leftCols<3>() = R2F;
				P2F.rightCols<1>() = t2F;

				//triangulate the point according to the foreground
				Eigen::Vector3d TPF = TriangulatePoint(P1F, P2F, feat_w1, feat_w2);
  				int countF = 0;
				double errF = 0;
				vector<pair<int, int>> inlF;

				bool bgok = 1;
				bool fgok = 1;
  				if(!HasPointPositiveDepth(P1B, TPB) || !HasPointPositiveDepth(P2B, TPB)) bgok = 0;
  				if(!HasPointPositiveDepth(P1F, TPF) || !HasPointPositiveDepth(P2F, TPF)) fgok = 0;
  				if(!bgok && !fgok) continue;
  				for(unsigned int k=0;k<track.size();k++)
  				{
  					pair<int, int> e3 = track[k];
  					//if(!diff_tk && C[e1.first-1].take == C[e3.first-1].take) continue;

  					//obtain the feature
  					Eigen::Vector2d feat_i3 = C[e3.first-1].features[e3.second-1];
					Eigen::Vector2d feat_w3;
					feat_w3(0) = (feat_i3(0) - C[e3.first-1].px);
					feat_w3(1) = (feat_i3(1) - C[e3.first-1].py);

					if(bgok)
					{
	  					//background cam
	  					//compose the matrix according to the background
						Eigen::Matrix3d R3B = C[e3.first-1].R;
						Eigen::Vector3d t3B = -R3B * C[e3.first-1].c;
						Eigen::Matrix3x4d P3B;
		  				P3B.leftCols<3>() = R3B;
		  				P3B.rightCols<1>() = t3B;

		  				//project the triangulated point to the camera
		  				Eigen::Vector4d TPB2;
		  				TPB2(0) = TPB(0);
		  				TPB2(1) = TPB(1);
		  				TPB2(2) = TPB(2);
		  				TPB2(3) = 1;
		  				Eigen::Vector3d proj = P3B * TPB2;
		  				Eigen::Vector2d nproj;
		  				nproj(0) = proj(0)/proj(2);
		  				nproj(1) = proj(1)/proj(2);
		  				nproj = nproj * C[e3.first-1].f;

		  				//check the cheirality
		  				if(HasPointPositiveDepth(P3B, TPB))
		  				{
			  				Eigen::Vector2d diff = nproj - feat_w3;
			  				if(diff.norm() < 20)
			  				{
			  					countB++;
			  					errB +=diff.norm();
			  					inlB.push_back(e3);
				  				//cout << diff.norm() << " ";
			  				}
		  				}
	  				}

	  				if(fgok)
	  				{
	  					//foreground cam
	  					//compose the matrix according to the foreground
	  					trans_s M3 = motion[(C[e3.first-1].take)-1];
						Eigen::Matrix3d R3F = C[e3.first-1].R * M3.R.transpose();
						Eigen::Vector3d c3F = M3.R * C[e3.first-1].c + M3.o;
						Eigen::Vector3d t3F = -R3F * c3F;
						Eigen::Matrix3x4d P3F;
						P3F.leftCols<3>() = R3F;
						P3F.rightCols<1>() = t3F;

						//project the triangulated point to the camera
		  				Eigen::Vector4d TPF2;
		  				TPF2(0) = TPF(0);
		  				TPF2(1) = TPF(1);
		  				TPF2(2) = TPF(2);
		  				TPF2(3) = 1;
		  				Eigen::Vector3d proj = P3F * TPF2;
		  				Eigen::Vector2d nproj;
		  				nproj(0) = proj(0)/proj(2);
		  				nproj(1) = proj(1)/proj(2);
		  				nproj = nproj * C[e3.first-1].f;

		  				//check the cheirality
		  				if(HasPointPositiveDepth(P3F, TPF))
		  				{
			  				Eigen::Vector2d diff = nproj - feat_w3;
			  				if(diff.norm() < 20)
			  				{
			  					countF++;
			  					errF +=diff.norm();
			  					inlF.push_back(e3);
				  				//cout << diff.norm() << " ";
			  				}
		  				}

	  				}
	  				
  				}
  				//cout << countB << "\n";
  				if(countB > best_count || (countB == best_count && errB < best_err && countB > 0))
  				{
  					bool ok = 0;
  					for(unsigned int q=1;q<inlB.size();q++)
  					{
  						if(inlB[q].first != inlB[0].first)
  						{
  							ok = 1;
  							break;
  						}
  					}
  					//take any inliers but at the end confirm only if at least one take is different
  					if(ok)
  					{
	  					best_count = countB;
  						best_err = errB;
  						best_obj = 0;
  						best = TPB;
  						best_inl = inlB;
					}
  				}
  				if(countF > best_count || (countF == best_count && errF < best_err && countF > 0))
  				{
  					bool ok = 0;
  					for(unsigned int q=1;q<inlF.size();q++)
  					{
  						if(inlF[q].first != inlF[0].first)
  						{
  							ok = 1;
  							break;
  						}
  					}
  					//take any inliers but at the end confirm only if at least one take is different
  					if(ok)
  					{
	  					best_count = countF;
  						best_err = errF;
  						best_obj = 1;
  						best = TPF;
  						best_inl = inlF;
					}
  				}
  				
			}
		}
		if(best_count > 2)
		{
			if(!best_obj)
			{
				R.first.points.push_back(best);
				R.first.ID.push_back(max);
				R.first.ID_map[max] = R.first.color.size();
				
				Eigen::Vector3i color;
				color(0) = 255;
				color(1) = 0;
				color(2) = 0;
				R.first.color.push_back(color);

				for(unsigned int i=0;i<best_inl.size();i++)
				{
					pair<int, int> e2 = best_inl[i];
					C[e2.first-1].b_obs[e2.second-1] = max;
				}
				
			}
			else
			{
				R.second.points.push_back(best);
				R.second.ID.push_back(max);
				R.second.ID_map[max] = R.first.color.size();
				
				Eigen::Vector3i color;
				color(0) = 0;
				color(1) = 255;
				color(2) = 0;
				R.second.color.push_back(color);

				for(unsigned int i=0;i<best_inl.size();i++)
				{
					pair<int, int> e2 = best_inl[i];
					C[e2.first-1].o_obs[e2.second-1] = max;
				}
				cnt++;
			}
			max++;
			
		}
		//cout << "\n";
	}
	cout << "TRACKS " << cnt << "\n";
	return ret;
}

std::pair<std::vector<int>, std::vector<int>> split_tracks2(std::vector<img_s> &C, int k, int ts)
{
	vector<int> b_cam;
	vector<int> f_cam;
	vector<vector<int>> O;
	for(unsigned int i=0;i<C.size();i++)
	{
		vector<Eigen::Vector2d> train;
		vector<Eigen::Vector2d> test;
		vector<int> test_id;
		vector<bool> label;
		//sort the observations between B, F, U
		for(unsigned int j=0;j<C[i].features.size();j++)
		{
			if(C[i].b_obs[j] > -1)
			{
				train.push_back(C[i].features[j]);
				label.push_back(0);
			}
			else if(C[i].o_obs[j] > -1)
			{
				train.push_back(C[i].features[j]);
				label.push_back(1);
			}
			else if(C[i].u_obs[j] > -1)
			{
				test.push_back(C[i].features[j]);
				test_id.push_back(j);
			}
		}
		/*cout << train.size() << " " << test.size() << "\n";
		for(int j=0;j<label.size();j++)
		{
			cout << label[j] << " ";
		}
		cout << "\n";*/

		//classify the unknown observations with knn
		int b_cnt = 0;
		int f_cnt = 0;
		vector<int> b_obs;
		vector<int> o_obs;
		for(unsigned int j=0;j<test.size();j++)
		{
			bool same = 1;
			vector<int> nearest;
			for(int a=0;a<k;a++)
			{
				double dist = 2*(C[i].px + C[i].py);
				int best = -1;
				for(unsigned int b=0;b<train.size();b++)
				{
					double cd = (test[j] - train[b]).norm();
					if(cd < dist)
					{
						bool ok = 1;
						for(unsigned int c=0;c<nearest.size();c++)
						{
							if(b==(unsigned int)nearest[c])
							{
								ok=0;
								break;
							}
						}
						if(ok)
						{
							dist = cd;
							best = (int)b;
						}
					}
				}
				if(best >= 0)
				{
					nearest.push_back(best);
					if(label[nearest[0]] != label[best])
					{
						same = 0;
					}
				}
			}
			if(same && nearest.size())
			{
				//observation classified
				if(label[nearest[0]])
				{
					f_cnt++;
					o_obs.push_back(C[i].u_obs[test_id[j]]);
				}
				else
				{
					b_cnt++;
					b_obs.push_back(C[i].u_obs[test_id[j]]);
				}
			}
		}
		//cout << b_cnt << " " << f_cnt << "\n";
		b_cam.push_back(O.size());
		O.push_back(b_obs);
		f_cam.push_back(O.size());
		O.push_back(o_obs);
	}
	std::pair<std::vector<int>, std::vector<int>> Q;
	Q.first = b_cam;
	Q.second = f_cam; 
	pair<pair<vector<int>, vector<int>>, vector<int>> D = split_tracks(Q, O, ts);
	std::pair<std::vector<int>, std::vector<int>> ret = D.first;

	
	
	return ret;

}

std::pair<std::vector<int>, std::vector<int>> filter_points2(std::pair<std::vector<int>, std::vector<int>> D2, std::pair<std::vector<int>, std::vector<int>> D, std::vector<img_s> &C, std::vector<pnts_s> P, int k, std::vector<std::vector<std::pair<int, int>>> T)
{
	cout << "Filtering points\n";
	//find the points from the background and the foreground
	vector<vector<Eigen::Vector3d>> b_pnts(P.size());
	vector<vector<Eigen::Vector3d>> o_pnts(P.size());
	for(unsigned int i=0;i<D.first.size();i++)
	{
		vector<pair<int, int>> track = T[D.first[i]];
		for(unsigned int j=0;j<track.size();j++)
		{
			pair<int, int> e = track[j];
			Eigen::Vector3d OP = P[e.first-1].points[P[e.first-1].ID_map[e.second]];
			b_pnts[e.first-1].push_back(OP);
		}
	}
	for(unsigned int i=0;i<D.second.size();i++)
	{
		vector<pair<int, int>> track = T[D.second[i]];
		for(unsigned int j=0;j<track.size();j++)
		{
			pair<int, int> e = track[j];
			Eigen::Vector3d OP = P[e.first-1].points[P[e.first-1].ID_map[e.second]];
			o_pnts[e.first-1].push_back(OP);
		}
	}

	//find the typical distance between the points
	vector<double> d_b(P.size());
	vector<double> d_o(P.size());
	for(unsigned int i=0;i<P.size();i++)
	{
		//cout << i << "\n";
		d_b[i] = 0;
		for(unsigned int j=0;j<b_pnts[i].size();j++)
		{
			Eigen::Vector3d pnt1 = b_pnts[i][j];
			double dist = INFINITY;
			for(unsigned int k=0;k<b_pnts[i].size();k++)
			{
				if(j==k) continue;
				Eigen::Vector3d pnt2 = b_pnts[i][k];
				double cur_dist = (pnt1-pnt2).norm();
				if(cur_dist < dist)
					dist = cur_dist;
			}
			d_b[i] = d_b[i] + dist;
		}
		if(b_pnts[i].size())
			d_b[i] = d_b[i] / (double)(b_pnts[i].size());
	}
	for(unsigned int i=0;i<P.size();i++)
	{
		//cout << i << "\n";
		d_o[i] = 0;
		for(unsigned int j=0;j<o_pnts[i].size();j++)
		{
			Eigen::Vector3d pnt1 = o_pnts[i][j];
			double dist = INFINITY;
			for(unsigned int k=0;k<o_pnts[i].size();k++)
			{
				if(j==k) continue;
				Eigen::Vector3d pnt2 = o_pnts[i][k];
				double cur_dist = (pnt1-pnt2).norm();
				if(cur_dist < dist)
					dist = cur_dist;
			}
			d_o[i] = d_o[i] + dist;
		}
		if(b_pnts[i].size())
			d_o[i] = d_o[i] / (double)(o_pnts[i].size());
	}

	//for each newly classified point find the k nearest points from the same object
	std::vector<int> b_ret;
	for(unsigned int i=0;i<D2.first.size();i++)
	{
		vector<pair<int, int>> track = T[D2.first[i]];
		vector<double> values;
		for(unsigned int j=0;j<track.size();j++)
		{
			pair<int, int> e = track[j];
			Eigen::Vector3d OP = P[e.first-1].points[P[e.first-1].ID_map[e.second]];

			vector<int> ids;
			for(int a=0;a<k;a++)
			{
				double dist = INFINITY;
				int best = -1;
				for(unsigned int b=0;b<b_pnts[e.first-1].size();b++)
				{
					Eigen::Vector3d P = b_pnts[e.first-1][b];
					double c_dist = (OP - P).norm();
					if(c_dist < dist)
					{
						bool ok = 1;
						for(unsigned int c=0;c<ids.size();c++)
						{
							if(ids[c] == (int)b)
							{
								ok = 0;
								break;
							}
						}
						if(ok)
						{
							dist = c_dist;
							best = b;
						}
					}
				}
				ids.push_back(best);
				values.push_back(dist / d_b[e.first-1]);
			}
		}
		int cnt = 0;
		for(unsigned int j=0;j<values.size();j++)
		{
			if(values[j] <= 10) cnt++;
			//cout << values[j] << " ";
		}
		if(cnt >= k)
			b_ret.push_back(D2.first[i]);
		//cout << "\n";
	}

	//for each newly classified point find the k nearest points from the same object
	std::vector<int> o_ret;
	for(unsigned int i=0;i<D2.second.size();i++)
	{
		vector<pair<int, int>> track = T[D2.second[i]];
		vector<double> values;
		for(unsigned int j=0;j<track.size();j++)
		{
			pair<int, int> e = track[j];
			Eigen::Vector3d OP = P[e.first-1].points[P[e.first-1].ID_map[e.second]];

			vector<int> ids;
			for(int a=0;a<k;a++)
			{
				double dist = INFINITY;
				int best = -1;
				for(unsigned int b=0;b<o_pnts[e.first-1].size();b++)
				{
					Eigen::Vector3d P = o_pnts[e.first-1][b];
					double c_dist = (OP - P).norm();
					if(c_dist < dist)
					{
						bool ok = 1;
						for(unsigned int c=0;c<ids.size();c++)
						{
							if((unsigned  int)ids[c] == b)
							{
								ok = 0;
								break;
							}
						}
						if(ok)
						{
							dist = c_dist;
							best = (int)b;
						}
					}
				}
				ids.push_back(best);
				values.push_back(dist / d_o[e.first-1]);
			}
		}
		int cnt = 0;
		for(unsigned int j=0;j<values.size();j++)
		{
			if(values[j] <= 10) cnt++;
			//cout << values[j] << " ";
		}
		if(cnt >= k)
			o_ret.push_back(D2.second[i]);
		//cout << "\n";
	}
	std::pair<std::vector<int>, std::vector<int>> ret;
	ret.first = b_ret;
	ret.second = o_ret;

	cout << "F\n";

	//add the observations of the newly separated tracks to the lists of the observations
	unordered_set<int> is_bg;
	unordered_set<int> is_fg;
	for(unsigned int i=0;i<ret.first.size();i++)
	{
		is_bg.insert(ret.first[i]);
	}
	for(unsigned int i=0;i<ret.second.size();i++)
	{
		is_fg.insert(ret.second[i]);
	}

	for(unsigned int i=0;i<C.size();i++)
	{
		for(unsigned int j=0;j<C[i].u_obs.size();j++)
		{
			if(is_bg.count(C[i].u_obs[j]))
			{
				C[i].b_obs[j] = C[i].u_obs[j];
			}
			else if(is_fg.count(C[i].u_obs[j]))
			{
				C[i].o_obs[j] = C[i].u_obs[j];
			}
		}
	}

	cout << "K\n";
	
	return ret;
}

void add_points2(std::pair<std::vector<int>, std::vector<int>> D, std::vector<std::vector<std::pair<int, int>>> T, std::vector<pnts_s> P, std::pair<pnts_s, pnts_s> &R, std::vector<std::pair<trans_s,trans_s>> M, int ref)
{
	vector<int> b_obs = D.first;
	vector<int> f_obs = D.second;
	
	//find the points of the background and the foreground
	//2 strategies -> nearest from one take, nearest among all takes
	
	//iterate through the new observations
	for(unsigned int i=0;i<b_obs.size();i++)
	{
		//find the track
		vector<pair<int, int>> track = T[b_obs[i]];

		//transform the points in the track
		vector<Eigen::Vector3d> npnt;
		vector<Eigen::Vector3i> ncol;
		//bool near = 0;
		//std::vector
		for(unsigned int j=0;j<track.size();j++)
		{
			pair<int, int> e = track[j];
			Eigen::Vector3d OP = P[e.first-1].points[P[e.first-1].ID_map[e.second]];
			Eigen::Vector3i NC = P[e.first-1].color[P[e.first-1].ID_map[e.second]];
			//cout << OP.transpose() << " ";
			trans_s mot = M[e.first-1].first;
			Eigen::Vector3d NP = mot.s * mot.R * OP + mot.o;
			if(ref == (e.first-1))
				NP = OP;
			//cout << NP.transpose() << " " << mot.s << " " << e.first << "\n";
			npnt.push_back(NP);
			ncol.push_back(NC);

			//check whether a point exists which is near enough to the 
			//and remember the found points, so we can enforce the minimum count of the found points
			
		}

		//find the median point
		Eigen::Vector3d pnt = median(npnt);
		//cout << "P " << pnt.transpose() << "\n";
		Eigen::Vector3i col = median(ncol);
		
		//add the point to the reconstruction (together with everything)
		R.first.ID.push_back(b_obs[i]);
		R.first.points.push_back(pnt);
		R.first.color.push_back(col);
		R.first.ID_map[b_obs[i]] = R.first.ID.size()-1;
	}

	//iterate through the new observations
	for(unsigned int i=0;i<f_obs.size();i++)
	{
		//find the track
		vector<pair<int, int>> track = T[f_obs[i]];

		//transform the points in the track
		vector<Eigen::Vector3d> npnt;
		vector<Eigen::Vector3i> ncol;
		for(unsigned int j=0;j<track.size();j++)
		{
			pair<int, int> e = track[j];
			Eigen::Vector3d OP = P[e.first-1].points[P[e.first-1].ID_map[e.second]];
			trans_s mot = M[e.first-1].second;
			Eigen::Vector3d NP = mot.s * mot.R * OP + mot.o;
			if(ref == (e.first-1))
				NP = OP;
			npnt.push_back(NP);
			Eigen::Vector3i NC = P[e.first-1].color[P[e.first-1].ID_map[e.second]];
			ncol.push_back(NC);
		}

		//find the median point
		Eigen::Vector3d pnt = median(npnt);
		Eigen::Vector3i col = median(ncol);
		
		//add the point to the reconstruction (together with everything)
		R.second.ID.push_back(f_obs[i]);
		R.second.points.push_back(pnt);
		R.second.color.push_back(col);
		R.second.ID_map[f_obs[i]] = R.second.ID.size()-1;
	}
}

std::pair<std::vector<int>, std::vector<int>> split_tracks3(std::pair<std::vector<int>, std::vector<int>> D, int k, int ts, std::vector<pnts_s> P, std::vector<std::unordered_map<int, int>> P2T)
{
	vector<int> score(ts);
	for(int i=0;i<ts;i++) score[i] = 0;
	vector<int> f_cam;
	vector<vector<int>> O;
	unordered_set<int> b_ids;
	unordered_set<int> f_ids;
	for(unsigned int i=0;i<D.first.size();i++)
	{
		b_ids.insert(D.first[i]);
	}
	for(unsigned int i=0;i<D.second.size();i++)
	{
		f_ids.insert(D.second[i]);
	}
	for(unsigned int t=0;t<P.size();t++)
	{
		vector<Eigen::Vector3d> train;
		vector<Eigen::Vector3d> test;
		vector<int> test_id;
		vector<bool> label;
		for(unsigned int i=0;i<P[t].points.size();i++)
		{

			//sort the points between B, F, U
			if(b_ids.count(P2T[t][i]))
			{
				train.push_back(P[t].points[i]);
				label.push_back(0);
			}
			else if(f_ids.count(P2T[t][i]))
			{
				train.push_back(P[t].points[i]);
				label.push_back(1);
			}
			else
			{
				test.push_back(P[t].points[i]);
				test_id.push_back(P2T[t][ i ]);
			}
		}

		for(unsigned int j=0;j<test.size();j++)
		{
			bool same = 1;
			vector<int> nearest;
			for(int a=0;a<k;a++)
			{
				double dist = INFINITY;
				int best = -1;
				for(unsigned int b=0;b<train.size();b++)
				{
					double cd = (test[j] - train[b]).norm();
					if(cd < dist)
					{
						bool ok = 1;
						for(unsigned int c=0;c<nearest.size();c++)
						{
							if(b==(unsigned int)nearest[c])
							{
								ok=0;
								break;
							}
						}
						if(ok)
						{
							dist = cd;
							best = (int)b;
						}
					}
				}
				if(best >= 0)
				{
					nearest.push_back(best);
					if(label[nearest[0]] != label[best])
					{
						same = 0;
					}
				}
			}
			if(same && nearest.size())
			{
				//observation classified
				if(label[nearest[0]])
				{
					score[test_id[j]] = score[test_id[j]]-1;
				}
				else
				{
					score[test_id[j]] = score[test_id[j]]+1;
				}
			}
		}
	}
	
	std::pair<std::vector<int>, std::vector<int>> Q;
	for(int i=0;i<ts;i++)
	{
		if(score[i] > 0) Q.first.push_back(i);
		else if(score[i] < 0) Q.second.push_back(i);
	}
	 
	std::pair<std::vector<int>, std::vector<int>> ret = Q;
	
	return ret;

}

}  // namespace colmap
