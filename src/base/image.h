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

#ifndef COLMAP_SRC_BASE_IMAGE_H_
#define COLMAP_SRC_BASE_IMAGE_H_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/point2d.h"
#include "base/visibility_pyramid.h"
#include "util/alignment.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/types.h"

namespace colmap {

// Class that holds information about an image. An image is the product of one
// camera shot at a certain location (parameterized as the pose). An image may
// share a camera with multiple other images, if its intrinsics are the same.
class Image {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Image();

  // Setup / tear down the image and necessary internal data structures before
  // and after being used in reconstruction.
  void SetUp(const Camera& camera);
  void TearDown();

  // Access the unique identifier of the image.
  inline image_t ImageId() const;
  inline void SetImageId(const image_t image_id);

  // Access the name of the image.
  inline const std::string& Name() const;
  inline std::string& Name();
  inline void SetName(const std::string& name);

  // Access the unique identifier of the camera. Note that multiple images
  // might share the same camera.
  inline camera_t CameraId() const;
  inline void SetCameraId(const camera_t camera_id);
  // Check whether identifier of camera has been set.
  inline bool HasCamera() const;

  //access image group
  inline void SetGroup(int g);
  inline int Group();
  //inline void SetGroup(int g) const;
  inline int Group() const;

  //access corresponding image
  inline void SetCorr(int c);
  inline int Corr();

  // Check if image is registered.
  inline bool IsRegistered() const;
  inline void SetRegistered(const bool registered);

  inline bool IsSeqRegistered() const;
  inline void SetSeqRegistered(const bool registered);

  // Get the number of image points.
  inline point2D_t NumPoints2D() const;

  // Get the number of triangulations, i.e. the number of points that
  // are part of a 3D point track.
  inline point2D_t NumPoints3D() const;

  // Get the number of observations, i.e. the number of image points that
  // have at least one correspondence to another image.
  inline point2D_t NumObservations() const;
  inline void SetNumObservations(const point2D_t num_observations);

  // Get the number of correspondences for all image points.
  inline point2D_t NumCorrespondences() const;
  inline void SetNumCorrespondences(const point2D_t num_observations);

  /*// Access 2D points
  //already exists
  inline std::vector<class Point2D> Points2D();
  inline void SetPoints2D(std::vector<class Point2D> points);*/

  // Get the number of observations that see a triangulated point, i.e. the
  // number of image points that have at least one correspondence to a
  // triangulated point in another image.
  inline point2D_t NumVisiblePoints3D() const;

  // Get the score of triangulated observations. In contrast to
  // `NumVisiblePoints3D`, this score also captures the distribution
  // of triangulated observations in the image. This is useful to select
  // the next best image in incremental reconstruction, because a more
  // uniform distribution of observations results in more robust registration.
  inline size_t Point3DVisibilityScore() const;

  // Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
  // pose which is defined as the transformation from world to image space.
  inline const Eigen::Vector4d& Qvec() const;
  inline Eigen::Vector4d& Qvec();
  inline double Qvec(const size_t idx) const;
  inline double& Qvec(const size_t idx);
  inline void SetQvec(const Eigen::Vector4d& qvec);

  // Quaternion prior, e.g. given by EXIF gyroscope tag.
  inline const Eigen::Vector4d& QvecPrior() const;
  inline Eigen::Vector4d& QvecPrior();
  inline double QvecPrior(const size_t idx) const;
  inline double& QvecPrior(const size_t idx);
  inline bool HasQvecPrior() const;
  inline void SetQvecPrior(const Eigen::Vector4d& qvec);

  // Access quaternion vector as (tx, ty, tz) specifying the translation of the
  // pose which is defined as the transformation from world to image space.
  inline const Eigen::Vector3d& Tvec() const;
  inline Eigen::Vector3d& Tvec();
  inline double Tvec(const size_t idx) const;
  inline double& Tvec(const size_t idx);
  inline void SetTvec(const Eigen::Vector3d& tvec);

  // Quaternion prior, e.g. given by EXIF GPS tag.
  inline const Eigen::Vector3d& TvecPrior() const;
  inline Eigen::Vector3d& TvecPrior();
  inline double TvecPrior(const size_t idx) const;
  inline double& TvecPrior(const size_t idx);
  inline bool HasTvecPrior() const;
  inline void SetTvecPrior(const Eigen::Vector3d& tvec);

  // Access tentative poses
  // add new pose
  inline void AddPose(const Eigen::Vector4d qvec, const Eigen::Vector3d tvec);
  inline std::vector<Eigen::Vector4d> Qvecs();
  inline std::vector<Eigen::Vector3d> Tvecs();

  inline void AddInlierCorrs(const std::vector<std::pair<point2D_t, point3D_t>> ic);
  inline std::vector<std::vector<std::pair<point2D_t, point3D_t>>> InlierCorrs();

  // Access the coordinates of image points.
  inline const class Point2D& Point2D(const point2D_t point2D_idx) const;
  inline class Point2D& Point2D(const point2D_t point2D_idx);
  inline const std::vector<class Point2D>& Points2D() const;
  void SetPoints2D(const std::vector<Eigen::Vector2d>& points);
  void SetPoints2D(const std::vector<class Point2D>& points);

  inline void ChangePoint2D(point2D_t id ,Eigen::Vector2d point);

  // Set the point as triangulated, i.e. it is part of a 3D point track.
  void SetPoint3DForPoint2D(const point2D_t point2D_idx,
                            const point3D_t point3D_id);

  // Set the point as not triangulated, i.e. it is not part of a 3D point track.
  void ResetPoint3DForPoint2D(const point2D_t point2D_idx);

  // Check whether an image point has a correspondence to an image point in
  // another image that has a 3D point.
  inline bool IsPoint3DVisible(const point2D_t point2D_idx) const;

  // Check whether one of the image points is part of the 3D point track.
  bool HasPoint3D(const point3D_t point3D_id) const;

  // Indicate that another image has a point that is triangulated and has
  // a correspondence to this image point. Note that this must only be called
  // after calling `SetUp`.
  void IncrementCorrespondenceHasPoint3D(const point2D_t point2D_idx);

  // Indicate that another image has a point that is not triangulated any more
  // and has a correspondence to this image point. This assumes that
  // `IncrementCorrespondenceHasPoint3D` was called for the same image point
  // and correspondence before. Note that this must only be called
  // after calling `SetUp`.
  void DecrementCorrespondenceHasPoint3D(const point2D_t point2D_idx);

  // Normalize the quaternion vector.
  void NormalizeQvec();

  // Compose the projection matrix from world to image space.
  Eigen::Matrix3x4d ProjectionMatrix() const;

  // Compose the inverse projection matrix from image to world space
  Eigen::Matrix3x4d InverseProjectionMatrix() const;

  // Compose rotation matrix from quaternion vector.
  Eigen::Matrix3d RotationMatrix() const;

  // Extract the projection center in world space.
  Eigen::Vector3d ProjectionCenter() const;

  // Extract the viewing direction of the image.
  Eigen::Vector3d ViewingDirection() const;

  // The number of levels in the 3D point multi-resolution visibility pyramid.
  static const int kNumPoint3DVisibilityPyramidLevels;

  std::vector<int> seq_inliers;

 private:
  // Identifier of the image, if not specified `kInvalidImageId`.
  image_t image_id_;

  // The name of the image, i.e. the relative path.
  std::string name_;

  // The identifier of the associated camera. Note that multiple images might
  // share the same camera. If not specified `kInvalidCameraId`.
  camera_t camera_id_;

  // Whether the image is successfully registered in the reconstruction.
  bool registered_;

  bool seq_registered;

  // Group of the image
  // 0: first set, 1: second set true, 2: second set phantom
  int group;

  // Corresponding image from the other group
  image_t corresponding;

  // The number of 3D points the image observes, i.e. the sum of its `points2D`
  // where `point3D_id != kInvalidPoint3DId`.
  point2D_t num_points3D_;

  // The number of image points that have at least one correspondence to
  // another image.
  point2D_t num_observations_;

  // The sum of correspondences per image point.
  point2D_t num_correspondences_;

  // The number of 2D points, which have at least one corresponding 2D point in
  // another image that is part of a 3D point track, i.e. the sum of `points2D`
  // where `num_tris > 0`.
  point2D_t num_visible_points3D_;

  // The pose of the image, defined as the transformation from world to image.
  Eigen::Vector4d qvec_;
  Eigen::Vector3d tvec_;

  // Tentative poses computed from sequential pnp
  std::vector<Eigen::Vector4d> qvecs;
  std::vector<Eigen::Vector3d> tvecs;

  // Inliers of the tentative poses
  std::vector<std::vector<std::pair<point2D_t, point3D_t>>> inlier_corrs;

  // The pose prior of the image, e.g. extracted from EXIF tags.
  Eigen::Vector4d qvec_prior_;
  Eigen::Vector3d tvec_prior_;

  // All image points, including points that are not part of a 3D point track.
  std::vector<class Point2D> points2D_;

  // Per image point, the number of correspondences that have a 3D point.
  std::vector<image_t> num_correspondences_have_point3D_;

  // Data structure to compute the distribution of triangulated correspondences
  // in the image. Note that this structure is only usable after `SetUp`.
  VisibilityPyramid point3D_visibility_pyramid_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

image_t Image::ImageId() const { return image_id_; }

void Image::SetImageId(const image_t image_id) { image_id_ = image_id; }

const std::string& Image::Name() const { return name_; }

std::string& Image::Name() { return name_; }

void Image::SetName(const std::string& name) { name_ = name; }

inline camera_t Image::CameraId() const { return camera_id_; }

inline void Image::SetCameraId(const camera_t camera_id) {
  CHECK_NE(camera_id, kInvalidCameraId);
  camera_id_ = camera_id;
}

inline bool Image::HasCamera() const { return camera_id_ != kInvalidCameraId; }

bool Image::IsRegistered() const { return registered_; }

void Image::SetRegistered(const bool registered) { registered_ = registered; }

bool Image::IsSeqRegistered() const { return seq_registered; }

void Image::SetSeqRegistered(const bool seq_reg) { seq_registered = seq_reg; }

point2D_t Image::NumPoints2D() const {
  return static_cast<point2D_t>(points2D_.size());
}

point2D_t Image::NumPoints3D() const { return num_points3D_; }

point2D_t Image::NumObservations() const { return num_observations_; }

void Image::SetNumObservations(const point2D_t num_observations) {
  num_observations_ = num_observations;
}

point2D_t Image::NumCorrespondences() const { return num_correspondences_; }

void Image::SetNumCorrespondences(const point2D_t num_correspondences) {
  num_correspondences_ = num_correspondences;
}

point2D_t Image::NumVisiblePoints3D() const { return num_visible_points3D_; }

size_t Image::Point3DVisibilityScore() const {
  return point3D_visibility_pyramid_.Score();
}

const Eigen::Vector4d& Image::Qvec() const { return qvec_; }

Eigen::Vector4d& Image::Qvec() { return qvec_; }

inline double Image::Qvec(const size_t idx) const { return qvec_(idx); }

inline double& Image::Qvec(const size_t idx) { return qvec_(idx); }

void Image::SetQvec(const Eigen::Vector4d& qvec) { qvec_ = qvec; }

const Eigen::Vector4d& Image::QvecPrior() const { return qvec_prior_; }

Eigen::Vector4d& Image::QvecPrior() { return qvec_prior_; }

inline double Image::QvecPrior(const size_t idx) const {
  return qvec_prior_(idx);
}

inline double& Image::QvecPrior(const size_t idx) { return qvec_prior_(idx); }

inline bool Image::HasQvecPrior() const { return !IsNaN(qvec_prior_.sum()); }

void Image::SetQvecPrior(const Eigen::Vector4d& qvec) { qvec_prior_ = qvec; }

const Eigen::Vector3d& Image::Tvec() const { return tvec_; }

Eigen::Vector3d& Image::Tvec() { return tvec_; }

inline double Image::Tvec(const size_t idx) const { return tvec_(idx); }

inline double& Image::Tvec(const size_t idx) { return tvec_(idx); }

void Image::SetTvec(const Eigen::Vector3d& tvec) { tvec_ = tvec; }

const Eigen::Vector3d& Image::TvecPrior() const { return tvec_prior_; }

Eigen::Vector3d& Image::TvecPrior() { return tvec_prior_; }

inline double Image::TvecPrior(const size_t idx) const {
  return tvec_prior_(idx);
}

inline double& Image::TvecPrior(const size_t idx) { return tvec_prior_(idx); }

inline bool Image::HasTvecPrior() const { return !IsNaN(tvec_prior_.sum()); }

void Image::SetTvecPrior(const Eigen::Vector3d& tvec) { tvec_prior_ = tvec; }

inline void Image::AddPose(const Eigen::Vector4d qvec, const Eigen::Vector3d tvec)
{
	qvecs.push_back(qvec);
	tvecs.push_back(tvec);
}

inline std::vector<Eigen::Vector4d> Image::Qvecs() {return qvecs;}
inline std::vector<Eigen::Vector3d> Image::Tvecs() {return tvecs;}

inline void Image::AddInlierCorrs(const std::vector<std::pair<point2D_t, point3D_t>> ic) {inlier_corrs.push_back(ic);}
inline std::vector<std::vector<std::pair<point2D_t, point3D_t>>> Image::InlierCorrs() {return inlier_corrs;}

const class Point2D& Image::Point2D(const point2D_t point2D_idx) const {
  return points2D_.at(point2D_idx);
}

class Point2D& Image::Point2D(const point2D_t point2D_idx) {
  return points2D_.at(point2D_idx);
}

const std::vector<class Point2D>& Image::Points2D() const { return points2D_; }

void Image::ChangePoint2D(point2D_t id ,Eigen::Vector2d point)
{
	points2D_.at(id).SetXY(point);
}

bool Image::IsPoint3DVisible(const point2D_t point2D_idx) const {
  return num_correspondences_have_point3D_.at(point2D_idx) > 0;
}

inline void Image::SetGroup(int g) { group = g;}
inline int Image::Group() {return group;}
//inline void Image::SetGroup(int g) const { group = g;}
inline int Image::Group() const {return group;}

inline void Image::SetCorr(int c) {corresponding = c;}
inline int Image::Corr() {return corresponding;}

/*inline std::vector<class Point2D> Image::Points2D() {return points2D_;}
inline void Image::SetPoints2D(std::vector<class Point2D> points) {points2D_ = points;}*/

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::Image)

#endif  // COLMAP_SRC_BASE_IMAGE_H_
