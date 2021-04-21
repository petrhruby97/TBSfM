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

#ifndef COLMAP_SRC_SFM_INCREMENTAL_MAPPER_H_
#define COLMAP_SRC_SFM_INCREMENTAL_MAPPER_H_

#include "base/database.h"
#include "base/database_cache.h"
#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"
#include "sfm/incremental_triangulator.h"
#include "util/alignment.h"
#include <utility>
#include <unordered_map>
#include "estimators/absolute_pose.h"

namespace colmap {

// Class that provides all functionality for the incremental reconstruction
// procedure. Example usage:
//
//  IncrementalMapper mapper(&database_cache);
//  mapper.BeginReconstruction(&reconstruction);
//  CHECK(mapper.FindInitialImagePair(options, image_id1, image_id2));
//  CHECK(mapper.RegisterInitialImagePair(options, image_id1, image_id2));
//  while (...) {
//    const auto next_image_ids = mapper.FindNextImages(options);
//    for (const auto image_id : next_image_ids) {
//      CHECK(mapper.RegisterNextImage(options, image_id));
//      if (...) {
//        mapper.AdjustLocalBundle(...);
//      } else {
//        mapper.AdjustGlobalBundle(...);
//      }
//    }
//  }
//  mapper.EndReconstruction(false);
//
class IncrementalMapper {
 public:
  struct Options {
    // Minimum number of inliers for initial image pair.
    int init_min_num_inliers = 100;

    // Maximum error in pixels for two-view geometry estimation for initial
    // image pair.
    double init_max_error = 4.0;

    // Maximum forward motion for initial image pair.
    double init_max_forward_motion = 0.95;

    // Minimum triangulation angle for initial image pair.
    double init_min_tri_angle = 16.0;

    // Maximum number of trials to use an image for initialization.
    int init_max_reg_trials = 2;

    // Maximum reprojection error in absolute pose estimation.
    double abs_pose_max_error = 12.0;

    // Minimum number of inliers in absolute pose estimation.
    int abs_pose_min_num_inliers = 30;

    // Minimum inlier ratio in absolute pose estimation.
    double abs_pose_min_inlier_ratio = 0.25;

    // Whether to estimate the focal length in absolute pose estimation.
    bool abs_pose_refine_focal_length = true;

    // Whether to estimate the extra parameters in absolute pose estimation.
    bool abs_pose_refine_extra_params = true;

    // Number of images to optimize in local bundle adjustment.
    int local_ba_num_images = 6;

    // Thresholds for bogus camera parameters. Images with bogus camera
    // parameters are filtered and ignored in triangulation.
    double min_focal_length_ratio = 0.1;  // Opening angle of ~130deg
    double max_focal_length_ratio = 10;   // Opening angle of ~5deg
    double max_extra_param = 1;

    // Maximum reprojection error in pixels for observations.
    double filter_max_reproj_error = 4.0;

    // Minimum triangulation angle in degrees for stable 3D points.
    double filter_min_tri_angle = 1.5;

    // Maximum number of trials to register an image.
    int max_reg_trials = 3;

    // Number of threads.
    int num_threads = -1;

    // Method to find and select next best image to register.
    enum class ImageSelectionMethod {
      MAX_VISIBLE_POINTS_NUM,
      MAX_VISIBLE_POINTS_RATIO,
      MIN_UNCERTAINTY,
    };
    ImageSelectionMethod image_selection_method =
        ImageSelectionMethod::MIN_UNCERTAINTY;

    bool Check() const;
  };

  struct LocalBundleAdjustmentReport {
    size_t num_merged_observations = 0;
    size_t num_completed_observations = 0;
    size_t num_filtered_observations = 0;
    size_t num_adjusted_observations = 0;
  };

  // Create incremental mapper. The database cache must live for the entire
  // life-time of the incremental mapper.
  explicit IncrementalMapper(/*const*/ DatabaseCache* database_cache);

  // Prepare the mapper for a new reconstruction, which might have existing
  // registered images (in which case `RegisterNextImage` must be called) or
  // which is empty (in which case `RegisterInitialImagePair` must be called).
  void BeginReconstruction(Reconstruction* reconstruction);

  // Cleanup the mapper after the current reconstruction is done. If the
  // model is discarded, the number of total and shared registered images will
  // be updated accordingly.
  void EndReconstruction(const bool discard);

  // Find initial image pair to seed the incremental reconstruction. The image
  // pairs should be passed to `RegisterInitialImagePair`. This function
  // automatically ignores image pairs that failed to register previously.
  bool FindInitialImagePair(const Options& options, image_t* image_id1,
                            image_t* image_id2);

  bool FindInitFirstSet(const Options& options, image_t* image_id1, image_t* image_id2, std::unordered_set<image_t> first_set);

  // Find best next image to register in the incremental reconstruction. The
  // images should be passed to `RegisterNextImage`. This function automatically
  // ignores images that failed to registered for `max_reg_trials`.
  std::vector<image_t> FindNextImages(const Options& options);

  std::vector<image_t> FindNextImagesFirstSet(const Options& options, std::unordered_set<image_t> first_set);

  std::vector<image_t> FindNextImagesSecondSet(const Options& options, std::unordered_set<image_t> first_set);

  // Attempt to seed the reconstruction from an image pair.
  bool RegisterInitialImagePair(const Options& options, const image_t image_id1,
                                const image_t image_id2);

  // Attempt to register image to the existing model. This requires that
  // a previous call to `RegisterInitialImagePair` was successful.
  bool RegisterNextImage(const Options& options, const image_t image_id);

  int SeqRegisterImage(const Options& options, const image_t image_id);

  int SeqRegisterImage2(const Options& options, const image_t image_id, const camera_t cam);

  bool FinishRegistration(const Options& options, const image_t image_id, const std::vector<std::pair<point2D_t, point3D_t>> tri_corrs);

  // Triangulate observations of image.
  size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                          const image_t image_id);

  // Retriangulate image pairs that should have common observations according to
  // the scene graph but don't due to drift, etc. To handle drift, the employed
  // reprojection error thresholds should be relatively large. If the thresholds
  // are too large, non-robust bundle adjustment will break down; if the
  // thresholds are too small, we cannot fix drift effectively.
  size_t Retriangulate(const IncrementalTriangulator::Options& tri_options);

  // Complete tracks by transitively following the scene graph correspondences.
  // This is especially effective after bundle adjustment, since many cameras
  // and point locations might have improved. Completion of tracks enables
  // better subsequent registration of new images.
  size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options);

  // Merge tracks by using scene graph correspondences. Similar to
  // `CompleteTracks`, this is effective after bundle adjustment and improves
  // the redundancy in subsequent bundle adjustments.
  size_t MergeTracks(const IncrementalTriangulator::Options& tri_options);

  // Adjust locally connected images and points of a reference image. In
  // addition, refine the provided 3D points. Only images connected to the
  // reference image are optimized. If the provided 3D points are not locally
  // connected to the reference image, their observing images are set as
  // constant in the adjustment.
  LocalBundleAdjustmentReport AdjustLocalBundle(
      const Options& options, const BundleAdjustmentOptions& ba_options,
      const IncrementalTriangulator::Options& tri_options,
      const image_t image_id, const std::unordered_set<point3D_t>& point3D_ids);

  // Global bundle adjustment using Ceres Solver or PBA.
  bool AdjustGlobalBundle(const BundleAdjustmentOptions& ba_options);
  bool AdjustParallelGlobalBundle(
      const BundleAdjustmentOptions& ba_options,
      const ParallelBundleAdjuster::Options& parallel_ba_options);

  // Filter images and point observations.
  size_t FilterImages(const Options& options);
  size_t FilterPoints(const Options& options);

  const Reconstruction& GetReconstruction() const;

  // Number of images that are registered in at least on reconstruction.
  size_t NumTotalRegImages() const;

  // Number of shared images between current reconstruction and all other
  // previous reconstructions.
  size_t NumSharedRegImages() const;

  // Get changed 3D points, since the last call to `ClearModifiedPoints3D`.
  const std::unordered_set<point3D_t>& GetModifiedPoints3D();

  // Clear the collection of changed 3D points.
  void ClearModifiedPoints3D();

  void InitImage(const image_t image_id, const image_t original, std::vector<image_pair_t> old_pairs, std::vector<image_pair_t> new_pairs);

  

 private:
  // Find seed images for incremental reconstruction. Suitable seed images have
  // a large number of correspondences and have camera calibration priors. The
  // returned list is ordered such that most suitable images are in the front.
  std::vector<image_t> FindFirstInitialImage(const Options& options) const;

  // For a given first seed image, find other images that are connected to the
  // first image. Suitable second images have a large number of correspondences
  // to the first image and have camera calibration priors. The returned list is
  // ordered such that most suitable images are in the front.
  std::vector<image_t> FindSecondInitialImage(const Options& options,
                                              const image_t image_id1) const;

  // Find local bundle for given image in the reconstruction. The local bundle
  // is defined as the images that are most connected, i.e. maximum number of
  // shared 3D points, to the given image.
  std::vector<image_t> FindLocalBundle(const Options& options,
                                       const image_t image_id) const;

  // Register / De-register image in current reconstruction and update
  // the number of shared images between all reconstructions.
  void RegisterImageEvent(const image_t image_id);
  void DeRegisterImageEvent(const image_t image_id);

  bool EstimateInitialTwoViewGeometry(const Options& options,
                                      const image_t image_id1,
                                      const image_t image_id2);

  // Class that holds all necessary data from database in memory.
  /*const*/ DatabaseCache* database_cache_;

  // Class that holds data of the reconstruction.
  Reconstruction* reconstruction_;

  // the scene graph
  SceneGraph * sg;

  // Class that is responsible for incremental triangulation.
  std::unique_ptr<IncrementalTriangulator> triangulator_;

  // Number of images that are registered in at least on reconstruction.
  size_t num_total_reg_images_;

  // Number of shared images between current reconstruction and all other
  // previous reconstructions.
  size_t num_shared_reg_images_;

  // Estimated two-view geometry of last call to `FindFirstInitialImage`,
  // used as a cache for a subsequent call to `RegisterInitialImagePair`.
  image_pair_t prev_init_image_pair_id_;
  TwoViewGeometry prev_init_two_view_geometry_;

  // Images and image pairs that have been used for initialization. Each image
  // and image pair is only tried once for initialization.
  std::unordered_map<image_t, size_t> init_num_reg_trials_;
  std::unordered_set<image_pair_t> init_image_pairs_;

  // Cameras whose parameters have been refined in pose refinement. Used
  // to avoid duplicate refinement of camera parameters or degradation of
  // already refined camera parameters when multiple images share intrinsics.
  std::unordered_set<camera_t> refined_cameras_;

  // The number of reconstructions in which images are registered.
  std::unordered_map<image_t, size_t> num_registrations_;

  // Images that have been filtered in current reconstruction.
  std::unordered_set<image_t> filtered_images_;

  // Number of trials to register image in current reconstruction. Used to set
  // an upper bound to the number of trials to register an image.
  std::unordered_map<image_t, size_t> num_reg_trials_;
};

typedef struct
{
	int id;
	int second;
	int anchor;
	//Eigen::Vector4d q;
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	double size_x;
	double size_y;
	double f;
	double px;
	double py;
	int inl;
	int id_2;
	Eigen::Vector3d center;
} cam_s;

typedef struct
{
	int init;
	int final;
	int cam_id;
	int c1;
	int c2;
	Eigen::Matrix3d A;
	//no translation, as to compute it two camera pairs are necessary
	//however this struct may be useful for other purposes where the translation may be known
} basis_t;

typedef struct
{
	int init;
	int final;
	int cam_id;
	int c1;
	int c2;
	Eigen::Matrix3d A;
	Eigen::Vector3d t;
	//no translation, as to compute it two camera pairs are necessary
	//however this struct may be useful for other purposes where the translation may be known
} motion_t;

typedef struct
{
	std::pair<std::vector<int>, std::vector<int>> div;
	std::vector<int> label;
	int size;
} pair_t;

typedef struct
{
	int size;
	int init;
	int final;
	Eigen::Matrix3d R;
	Eigen::Vector3d t1;
	double sigma1;
	Eigen::Vector3d t2;
	double sigma2;
} clust_b;

typedef struct
{
	int size;
	int init;
	int final;
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
} clust_m;

typedef struct
{
	int t1;
	int cl1;
	int t2;
	int cl2;
	int t3;
	int cl3;
	double i1;
	double i2;
	Eigen::Vector3d i3;
} cycle_3;

typedef struct
{
	Eigen::Matrix3d R;
	Eigen::Vector3d o;
	double s;
} trans_s;

typedef struct
{
	std::vector<int> id;
	std::vector<std::vector<int>> obs;
	std::unordered_map<int,int> map;
	std::vector<std::vector<Eigen::Vector2d>> features;
	std::vector<std::vector<int>> obs_all;
} imgs_s;

typedef struct
{
	std::vector<int> ID;
	std::unordered_map<int, int> ID_map;
	std::vector<Eigen::Vector3d> points;
	std::vector<Eigen::Vector3i> color;
} pnts_s;

typedef struct
{
	int cl1;
	int cl2;
	int pred;
	int grade;
	double rc;
	double tc;
	int common;
	double ratio;
} edge_s;

typedef struct
{
	Eigen::Matrix3d R;
	Eigen::Vector3d c;
	int take;
	std::vector<int> b_obs;
	std::vector<int> o_obs;
	std::vector<int> u_obs;
	std::vector<Eigen::Vector2d> features;
	double size_x;
	double size_y;
	double f;
	double px;
	double py;
} img_s;

typedef struct
{
	int take;
	int position;
	int id;
	int strength;
} point_id;

typedef struct
{
	point_id id;
	int track;
	std::vector<point_id> neighbours;
} node_s;


std::vector<basis_t> find_bases(std::vector<cam_s> C);

std::vector<cam_s> load_cams();

int count_takes(std::vector<cam_s> C);

std::vector<std::vector<std::vector<basis_t>>> divide_bases(std::vector<basis_t> B, std::vector<cam_s> C, double sigma);

std::vector<std::vector<clust_b>> meanclust(std::vector<std::vector<std::vector<basis_t>>> C, std::vector<cam_s> cams);

std::vector<cycle_3> find3cycles(std::vector<std::vector<clust_b>> M);

std::vector<int> cluster_cycles(std::vector<cycle_3> C, double thr, std::vector<cam_s> cams, std::vector<std::vector<std::vector<basis_t>>> CL, std::vector<double> * bzs);

std::pair<std::pair<Eigen::MatrixXi, Eigen::MatrixXi>, Eigen::MatrixXd> select_bases(std::vector<std::vector<clust_b>> M, std::vector<cycle_3> CY, std::vector<int> CL2, int takes, std::vector<double> bzs, double thr1, double thr2);

int find_reference(Eigen::MatrixXd st, int takes);

std::vector<std::vector<trans_s>> find_transformations(std::pair<Eigen::MatrixXi, Eigen::MatrixXi> st, int reference, std::vector<std::vector<clust_b>> M, int takes	);

std::vector<motion_t> find_motion(std::vector<cam_s> C);

std::vector<motion_t> transform_motion(std::vector<motion_t> C, std::vector<std::vector<trans_s>> transform);

std::vector<motion_t> remove_id(std::vector<motion_t>, double pd, double thr1, double thr2);

double princ_dist(std::vector<cam_s> C, int take);

std::vector<cam_s> change_basis_cams(std::vector<cam_s> C, std::vector<std::vector<trans_s>> T);

std::vector<std::vector<std::vector<motion_t>>> divide_motions(std::vector<motion_t> B, std::vector<cam_s> C, double sigma, double pd, double thr2);

std::vector<std::vector<clust_m>> meanclust_motions(std::vector<std::vector<std::vector<motion_t>>> C);

std::pair<std::vector<std::vector<std::vector<motion_t>>>, std::vector<std::vector<clust_m>>> remove_identity(std::vector<std::vector<clust_m>> C, std::vector<std::vector<std::vector<motion_t>>> CL , double pd, double thr1, double thr2);

std::vector<imgs_s> load_imgs(int takes);

std::vector<pnts_s> load_pnts(int takes);

std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::unordered_map<int, int>>> load_tracks(std::vector<pnts_s> P, int takes);

std::vector<std::vector<int>> observed_tracks(std::vector<cam_s> C, std::vector<imgs_s> I, std::vector<pnts_s> P, std::vector<std::unordered_map<int, int>> P2T);

std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> observed_tracks_2(std::vector<cam_s> C, std::vector<imgs_s> I, std::vector<pnts_s> P, std::vector<std::unordered_map<int, int>> P2T);

std::vector<std::vector<pair_t>> group_ot(std::vector<std::vector<int>> O, std::vector<cam_s> C, int takes);

//std::vector<std::vector<pair_t>> filter_groups(std::vector<std::vector<pair_t>> G, std::vector<std::vector<std::pair<int, int>>> T);
std::vector<std::vector<pair_t>> filter_groups(std::vector<std::vector<pair_t>> G, std::vector<std::vector<std::pair<int, int>>> T, std::vector<pnts_s> P);

int intersect_size(std::vector<int> O1, std::vector<int> O2);

std::vector<std::vector<pair_t>> linkage(std::vector<std::vector<pair_t>> G);

std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> observed_by_cluster(std::vector<std::vector<std::vector<motion_t>>> CL, std::vector<std::vector<int>> O);

std::vector<std::vector<std::pair<int, int>>> chordal_completion( std::vector<std::vector<clust_m>> CL, std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> O, double thr1, double thr2, double pd );

int select_group(std::vector<std::vector<std::pair<int, int>>> CLCL, std::vector<std::vector<clust_m>> CL);

int select_group_o(std::vector<std::vector<std::pair<int, int>>> CLCL, std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> O);

std::pair<std::vector<int>, std::vector<int>> group_cams(std::vector<std::pair<int, int>> CL, std::vector<std::vector<std::vector<motion_t>>> MM);

std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<int>> split_tracks(std::pair<std::vector<int>, std::vector<int>> Q, std::vector<std::vector<int>> O, int size);

std::vector<int> order_fold(std::vector<std::vector<std::pair<int, int>>> T, std::pair<std::vector<int>, std::vector<int>> OT, int takes);

std::pair<std::pair<pnts_s, pnts_s>, std::vector<std::pair<trans_s,trans_s>>> merge_reconstructions(std::vector<pnts_s> P, std::vector<std::vector<std::pair<int, int>>> T, std::pair<std::vector<int>, std::vector<int>> D, std::vector<int> order);
//std::pair< std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>> , std::vector<std::pair<trans_s,trans_s>> > merge_reconstructions(std::vector<pnts_s> P, std::vector<std::vector<std::pair<int, int>>> T, std::pair<std::vector<int>, std::vector<int>> D, std::vector<int> order);

void save_model_ply(std::pair<pnts_s, pnts_s> M, int mode);

std::pair<std::vector<img_s>, std::vector<trans_s>> merge_cameras(std::vector<cam_s> C, std::vector<std::pair<trans_s,trans_s>> motions, int ref, std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> O, std::pair<std::vector<int>, std::vector<int>> Q, std::pair<std::vector<int>, std::vector<int>> D);

void save_points(std::pair<pnts_s, pnts_s> P);

void save_model(std::pair<pnts_s, pnts_s> P, std::vector<img_s> C, std::vector<trans_s> motion, int mode, int ref);

void perform_BA(std::pair<pnts_s, pnts_s> &P, std::vector<img_s> &C, std::vector<trans_s> &motion, int mode, int ref);

void perform_BA_alter(std::pair<pnts_s, pnts_s> &P, std::vector<img_s> &C, std::vector<trans_s> &motion, int mode, int ref);

void step1();

void check(std::vector<imgs_s> I, std::vector<pnts_s> P, std::vector<cam_s> C, std::pair<std::vector<std::vector<std::pair<int, int>>>, std::vector<std::unordered_map<int, int>>> T, std::vector<std::pair<std::vector<int>, std::vector<Eigen::Vector2d>>> O);

void new_points(std::vector<std::vector<std::pair<int, int>>> T, std::vector<pnts_s> P, std::vector<int> D, std::vector<std::pair<trans_s,trans_s>> M, int r, double pd);

std::pair<std::vector<int>, std::vector<int>> split_tracks2(std::vector<img_s> &C, int k, int ts);

void add_points2(std::pair<std::vector<int>, std::vector<int>> D, std::vector<std::vector<std::pair<int, int>>> T, std::vector<pnts_s> P, std::pair<pnts_s, pnts_s> &R, std::vector<std::pair<trans_s,trans_s>> M, int ref);

std::vector<std::vector<std::pair<int, int>>> load_2tracks(std::vector<img_s> &C, std::pair<pnts_s, pnts_s> &R, std::vector<trans_s> motion);

std::pair<std::vector<int>, std::vector<int>> filter_points2(std::pair<std::vector<int>, std::vector<int>> D2, std::pair<std::vector<int>, std::vector<int>> D, std::vector<img_s> &C, std::vector<pnts_s> P, int k, std::vector<std::vector<std::pair<int, int>>> T);

std::pair<std::vector<int>, std::vector<int>> split_tracks3(std::pair<std::vector<int>, std::vector<int>> D, int k, int ts, std::vector<pnts_s> P, std::vector<std::unordered_map<int, int>> P2T);

std::vector<std::vector<std::pair<int, int>>> split_cams(std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> O);

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_INCREMENTAL_MAPPER_H_
