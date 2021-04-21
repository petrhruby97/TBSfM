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

#include "controllers/incremental_mapper.h"

#include "util/misc.h"
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "estimators/absolute_pose.h"
#include <assert.h>
#include <exception>

namespace colmap {
namespace {

size_t TriangulateImage(const IncrementalMapperOptions& options,
                        const Image& image, IncrementalMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.Triangulation(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

void AdjustGlobalBundle(const IncrementalMapperOptions& options,
                        IncrementalMapper* mapper) {
  BundleAdjustmentOptions custom_options = options.GlobalBundleAdjustment();

  const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

  // Use stricter convergence criteria for first registered images.
  const size_t kMinNumRegImages = 10;
  if (num_reg_images < kMinNumRegImages) {
    custom_options.solver_options.function_tolerance /= 10;
    custom_options.solver_options.gradient_tolerance /= 10;
    custom_options.solver_options.parameter_tolerance /= 10;
    custom_options.solver_options.max_num_iterations *= 2;
    custom_options.solver_options.max_linear_solver_iterations = 200;
  }

  PrintHeading1("Global bundle adjustment");
  if (options.ba_global_use_pba && num_reg_images >= kMinNumRegImages &&
      ParallelBundleAdjuster::IsSupported(custom_options,
                                          mapper->GetReconstruction())) {
    mapper->AdjustParallelGlobalBundle(
        custom_options, options.ParallelGlobalBundleAdjustment());
  } else {
    mapper->AdjustGlobalBundle(custom_options);
  }
}

void IterativeLocalRefinement(const IncrementalMapperOptions& options,
                              const image_t image_id,
                              IncrementalMapper* mapper) {
  auto ba_options = options.LocalBundleAdjustment();
  for (int i = 0; i < options.ba_local_max_refinements; ++i) {
    const auto report = mapper->AdjustLocalBundle(
        options.Mapper(), ba_options, options.Triangulation(), image_id,
        mapper->GetModifiedPoints3D());
    std::cout << "  => Merged observations: " << report.num_merged_observations
              << std::endl;
    std::cout << "  => Completed observations: "
              << report.num_completed_observations << std::endl;
    std::cout << "  => Filtered observations: "
              << report.num_filtered_observations << std::endl;
    const double changed =
        (report.num_merged_observations + report.num_completed_observations +
         report.num_filtered_observations) /
        static_cast<double>(report.num_adjusted_observations);
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < options.ba_local_max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    ba_options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  mapper->ClearModifiedPoints3D();
}

void IterativeGlobalRefinement(const IncrementalMapperOptions& options,
                               IncrementalMapper* mapper) {
  PrintHeading1("Retriangulation");
  CompleteAndMergeTracks(options, mapper);
  std::cout << "  => Retriangulated observations: "
            << mapper->Retriangulate(options.Triangulation()) << std::endl;

  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    const size_t num_observations =
        mapper->GetReconstruction().ComputeNumObservations();
    size_t num_changed_observations = 0;
    AdjustGlobalBundle(options, mapper);
    num_changed_observations += CompleteAndMergeTracks(options, mapper);
    num_changed_observations += FilterPoints(options, mapper);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                              reconstruction->Image(image_id).Name().c_str(),
                              image_path.c_str())
              << std::endl;
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::string& snapshot_path) {
  PrintHeading1("Creating snapshot");
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const std::string path =
      JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
  CreateDirIfNotExists(path);
  std::cout << "  => Writing to " << path << std::endl;
  reconstruction.Write(path);
}

}  // namespace

size_t FilterPoints(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.Mapper());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IncrementalMapperOptions& options,
                              IncrementalMapper* mapper) {
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.Triangulation());
  std::cout << "  => Merged observations: " << num_completed_observations
            << std::endl;
  const size_t num_merged_observations =
      mapper->MergeTracks(options.Triangulation());
  std::cout << "  => Completed observations: " << num_merged_observations
            << std::endl;
  return num_completed_observations + num_merged_observations;
}

IncrementalMapper::Options IncrementalMapperOptions::Mapper() const {
  IncrementalMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  return options;
}

IncrementalTriangulator::Options IncrementalMapperOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.solver_options.num_threads = num_threads;
  options.solver_options.num_linear_solver_threads = num_threads;
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.loss_function_scale = 1.0;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::CAUCHY;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = true;
  options.solver_options.num_threads = num_threads;
  options.solver_options.num_linear_solver_threads = num_threads;
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  return options;
}

ParallelBundleAdjuster::Options
IncrementalMapperOptions::ParallelGlobalBundleAdjustment() const {
  ParallelBundleAdjuster::Options options;
  options.max_num_iterations = ba_global_max_num_iterations;
  options.print_summary = true;
  options.gpu_index = ba_global_pba_gpu_index;
  options.num_threads = num_threads;
  return options;
}

bool IncrementalMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_num_images, 2);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_images_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_images_freq, 0);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

IncrementalMapperController::IncrementalMapperController(
    const IncrementalMapperOptions* options, const std::string& image_path,
    const std::string& database_path,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(reconstruction_manager) {
  CHECK(options_->Check());
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void IncrementalMapperController::Run() {
  if (!LoadDatabase()) {
    return;
  }

  IncrementalMapper::Options init_mapper_options = options_->Mapper();
  Reconstruct(init_mapper_options);

  const size_t kNumInitRelaxations = 2;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    if (reconstruction_manager_->Size() > 0 || IsStopped()) {
      break;
    }

    std::cout << "  => Relaxing the initialization constraints." << std::endl;
    init_mapper_options.init_min_num_inliers /= 2;
    Reconstruct(init_mapper_options);

    if (reconstruction_manager_->Size() > 0 || IsStopped()) {
      break;
    }

    std::cout << "  => Relaxing the initialization constraints." << std::endl;
    init_mapper_options.init_min_tri_angle /= 2;
    Reconstruct(init_mapper_options);
  }

  std::cout << std::endl;
  GetTimer().PrintMinutes();
}

bool IncrementalMapperController::LoadDatabase() {
  PrintHeading1("Loading database");

  Database database(database_path_);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  database_cache_.Load(database, min_num_matches, options_->ignore_watermarks,
                       options_->image_names);
  std::cout << std::endl;
  timer.PrintMinutes();

  std::cout << std::endl;

  if (database_cache_.NumImages() == 0) {
    std::cout << "WARNING: No images with matches found in the database."
              << std::endl
              << std::endl;
    return false;
  }

  return true;
}

void IncrementalMapperController::Reconstruct(
    const IncrementalMapper::Options& init_mapper_options) {
  const bool kDiscardReconstruction = true;

  //////////////////////////////////////////////////////////////////////////////
  // Main loop
  //////////////////////////////////////////////////////////////////////////////

  IncrementalMapper mapper(&database_cache_);

  // Is there a sub-model before we start the reconstruction? I.e. the user
  // has imported an existing reconstruction.
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                  "single reconstruction, but "
                                                  "multiple are given.";

  for (int num_trials = 0; num_trials < options_->init_num_trials;
       ++num_trials) {
    BlockIfPaused();
    if (IsStopped()) {
      break;
    }

    size_t reconstruction_idx;
    if (!initial_reconstruction_given || num_trials > 0) {
      reconstruction_idx = reconstruction_manager_->Add();
    } else {
      reconstruction_idx = 0;
    }

    Reconstruction& reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    mapper.BeginReconstruction(&reconstruction);

    //find out which set to reconstruct
	int recon_set = 1;
	std::ifstream pivot;
	pivot.open("pivot.txt");
	pivot >> recon_set;
	pivot.close();
	std::cout << "ANCHOR TAKE " << recon_set << "\n";
	std::unordered_set<image_t> main_set;
	std::unordered_set<image_t> rest_images;

    size_t imgs = reconstruction.NumImages();
    std::cerr << imgs << "\n";
    //scene graph saved in reconstruction or in database
    //HERE
    const SceneGraph scene_graph = database_cache_.SceneGraph();
    //SceneGraph scene_graph = database_cache_.SG();
    /*std::cout << "finding tracks\n";
    std::vector<std::unordered_set<point2D_t>> hash(imgs);
    std::vector<std::vector<SceneGraph::Correspondence>> tracks;
    for(size_t i=1;i<imgs;i++)
    {
    	//std::cerr << i << "\n";
    	Image cur_img = reconstruction.Image(i);
    	point2D_t points = cur_img.NumPoints2D();
		//std::cerr << i << " " << points << "\n";
		for(point2D_t j=0;j<points;j++)
		{
			if(hash[i-1].count(j)) continue;
			//std::vector<SceneGraph::Correspondence> track = scene_graph.FindTransitiveCorrespondences(i, j, imgs);
			std::vector<SceneGraph::Correspondence> track = scene_graph.FindTransitiveCorrespondences(i, j, 1);
			if(!track.size()) continue;
			track.emplace_back(i, j);
			for(SceneGraph::Correspondence corr:track)
			{
				hash[corr.image_id-1].insert(corr.point2D_idx);
				//std::cerr << corr.image_id << " " << corr.point2D_idx << " ";
			}
			tracks.push_back(track);
		}
    }
    //write out total number of created tracks
    std::ofstream file;
    file.open(JoinPaths(std::to_string(recon_set),"tracks.txt"));
    file << imgs << "\n";
    file << tracks.size() << "\n";
    for(std::vector<SceneGraph::Correspondence> track : tracks)
    {
    	for(SceneGraph::Correspondence corr : track)
    	{
    		Point2D p = reconstruction.Image(corr.image_id).Point2D(corr.point2D_idx);
    		//file << corr.image_id << " " << corr.point2D_idx << " ";
    		file << corr.image_id << " " << p.X() << " " << p.Y() << " ";
    	}
    	file << "-1\n";
    }
    file.close();
    //return;

    std::cout << "tracks found\n";*/

    //get ids of images which belong to the first set according to the included file
	std::unordered_set<std::string> fs;
	std::ifstream fset;
	fset.open("takes.txt");
	std::string name;
	int cur_id;
	std::unordered_map<std::string,int> take_map;
	//std::
	while(fset >> name)
	{
		fset >> cur_id;
		//fs.insert(name);
		//std::cout << "NAME " << name << " " << cur_id << "\n";
		//take_map.insert(std::make_pair<std::string,int>(name,cur_id));
		take_map.insert({name, cur_id});
	}
	fset.close();

	for (const auto& image : reconstruction.Images())
	{
		//std::cout << "ID " << image.first << " " << image.second.Name() << " " << (take_map[image.second.Name()]==recon_set) << "\n";
		if((take_map[image.second.Name()]==recon_set))
		{
			main_set.insert(image.first);
		}
		else
		{
			rest_images.insert(image.first);
		}
	}
	//return;
	/*for(image_t i=1;i<=imgs;i++)
	{
		//check in which take the i-th image is
		Image cur_img = reconstruction.Image(i);
		std::string name = cur_img.Name();
		int take = take_map[name];
		std::cout << name << "\n";
		std::cout << "New image: " << i << " " << take << " " << imgs << "\n";
		if(take == recon_set)
		{
			main_set.insert(i);
			std::cout << "MAIN" << i << "\n";
		}
		else
		{
			rest_images.insert(i);
		}f
	}*/

	std::cout << "init pair registration\n";
	

	
    ////////////////////////////////////////////////////////////////////////////
    // Register initial pair
    ////////////////////////////////////////////////////////////////////////////

    if (reconstruction.NumRegImages() == 0) {
      image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
      image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

      // Try to find good initial pair.
      if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
        //const bool find_init_success = mapper.FindInitialImagePair(init_mapper_options, &image_id1, &image_id2);
        const bool find_init_success = mapper.FindInitFirstSet(init_mapper_options, &image_id1, &image_id2, main_set);
        if (!find_init_success) {
          std::cout << "  => No good initial image pair found." << std::endl;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          break;
        }
      } else {
        if (!reconstruction.ExistsImage(image_id1) ||
            !reconstruction.ExistsImage(image_id2)) {
          std::cout << StringPrintf(
                           "  => Initial image pair #%d and #%d do not exist.",
                           image_id1, image_id2)
                    << std::endl;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          return;
        }
      }

      PrintHeading1(StringPrintf("Initializing with image pair #%d and #%d",
                                 image_id1, image_id2));
      const bool reg_init_success = mapper.RegisterInitialImagePair(
          init_mapper_options, image_id1, image_id2);
      if (!reg_init_success) {
        std::cout << "  => Initialization failed - possible solutions:"
                  << std::endl
                  << "     - try to relax the initialization constraints"
                  << std::endl
                  << "     - manually select an initial image pair"
                  << std::endl;
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        break;
      }

      AdjustGlobalBundle(*options_, &mapper);
      FilterPoints(*options_, &mapper);
      FilterImages(*options_, &mapper);

      // Initial image pair failed to register.
      if (reconstruction.NumRegImages() == 0 ||
          reconstruction.NumPoints3D() == 0) {
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        // If both initial images are manually specified, there is no need for
        // further initialization trials.
        if (options_->init_image_id1 != -1 && options_->init_image_id2 != -1) {
          break;
        } else {
          continue;
        }
      }

      if (options_->extract_colors) {
        ExtractColors(image_path_, image_id1, &reconstruction);
      }
    }

    Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

    ////////////////////////////////////////////////////////////////////////////
    // Incremental mapping
    ////////////////////////////////////////////////////////////////////////////

    size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
    size_t ba_prev_num_reg_images = reconstruction.NumRegImages();
    size_t ba_prev_num_points = reconstruction.NumPoints3D();

    bool reg_next_success = true;
    bool prev_reg_next_success = true;

    //incremental mapping loop
    //this loop will be used for first set only
    //under this loop will be another which will sequentially register the second set
    while (reg_next_success) {
      BlockIfPaused();
      if (IsStopped()) {
        break;
      }

      reg_next_success = false;

      //const std::vector<image_t> next_images = mapper.FindNextImages(options_->Mapper());
      const std::vector<image_t> next_images = mapper.FindNextImagesFirstSet(options_->Mapper(), main_set);

      if (next_images.empty()) {
        break;
      }

      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        const image_t next_image_id = next_images[reg_trial];
        const Image& next_image = reconstruction.Image(next_image_id);

        PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                   reconstruction.NumRegImages() + 1));

        std::cout << StringPrintf("  => Image sees %d / %d points",
                                  next_image.NumVisiblePoints3D(),
                                  next_image.NumObservations())
                  << std::endl;

        reg_next_success =
            mapper.RegisterNextImage(options_->Mapper(), next_image_id);

        if (reg_next_success) {
          //next_image.SetGroup(0);
          TriangulateImage(*options_, next_image, &mapper);
          IterativeLocalRefinement(*options_, next_image_id, &mapper);

          if (reconstruction.NumRegImages() >=
                  options_->ba_global_images_ratio * ba_prev_num_reg_images ||
              reconstruction.NumRegImages() >=
                  options_->ba_global_images_freq + ba_prev_num_reg_images ||
              reconstruction.NumPoints3D() >=
                  options_->ba_global_points_ratio * ba_prev_num_points ||
              reconstruction.NumPoints3D() >=
                  options_->ba_global_points_freq + ba_prev_num_points) {
            IterativeGlobalRefinement(*options_, &mapper);
            ba_prev_num_points = reconstruction.NumPoints3D();
            ba_prev_num_reg_images = reconstruction.NumRegImages();
          }

          if (options_->extract_colors) {
            ExtractColors(image_path_, next_image_id, &reconstruction);
          }

          if (options_->snapshot_images_freq > 0 &&
              reconstruction.NumRegImages() >=
                  options_->snapshot_images_freq +
                      snapshot_prev_num_reg_images) {
            snapshot_prev_num_reg_images = reconstruction.NumRegImages();
            WriteSnapshot(reconstruction, options_->snapshot_path);
          }

          Callback(NEXT_IMAGE_REG_CALLBACK);

          break;
        } else {
          std::cout << "  => Could not register, trying another image."
                    << std::endl;

          // If initial pair fails to continue for some time,
          // abort and try different initial pair.
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              reconstruction.NumRegImages() <
                  static_cast<size_t>(options_->min_model_size)) {
            break;
          }
        }
      }

      const size_t max_model_overlap =
          static_cast<size_t>(options_->max_model_overlap);
      if (mapper.NumSharedRegImages() >= max_model_overlap) {
        break;
      }

      // If no image could be registered, try a single final global iterative
      // bundle adjustment and try again to register one image. If this fails
      // once, then exit the incremental mapping.
      if (!reg_next_success && prev_reg_next_success) {
        reg_next_success = true;
        prev_reg_next_success = false;
        IterativeGlobalRefinement(*options_, &mapper);
      } else {
        prev_reg_next_success = reg_next_success;
      }
      //next_images.clear();
    }

    if (IsStopped()) {
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
      break;
    }

    // Only run final global BA, if last incremental BA was not global.
    if (reconstruction.NumRegImages() >= 2 &&
        reconstruction.NumRegImages() != ba_prev_num_reg_images &&
        reconstruction.NumPoints3D() != ba_prev_num_points) {
      IterativeGlobalRefinement(*options_, &mapper);
    }

    std::ofstream ph_out;
    ph_out.open(JoinPaths(std::to_string(recon_set/*num_trials*/),"phantom.txt"));
    std::ofstream real_out;
    real_out.open(JoinPaths(std::to_string(recon_set/*num_trials*/),"real.txt"));
    std::ofstream camera_log;
    camera_log.open(JoinPaths(std::to_string(recon_set/*num_trials*/),"cams.txt"));

	Camera mc = Camera();
	camera_t mc_id = reconstruction.NumCameras()+1;
	mc.SetCameraId(mc_id);
	std::vector<double> flengths;
    //print out the first set of cameras
    bool cam_found = false;
    for(const auto& image : reconstruction.Images())
	{
		//check whether i-th image is in the second set
		int i = image.first;
		Image cur_img = reconstruction.Image(i);
		if(!cur_img.IsRegistered()) continue;
		Camera cam = reconstruction.Camera(cur_img.CameraId());
		Eigen::Vector4d q = cur_img.Qvec();
		Eigen::Vector3d t = cur_img.Tvec();
		camera_log << i << " " << recon_set << " " << recon_set /**//* << " " << num_trials /**/ << " " << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << " "
				<< t(0) << " " << t(1) << " " << t(2) << " " << cam.Width() << " " << cam.Height() << " " << cam.FocalLength() << " "
				<< cam.PrincipalPointX() << " " << cam.PrincipalPointY() << " -1 " << i << "\n";
		flengths.push_back(cam.FocalLength());
		if(!cam_found)
		{
			cam_found = true;
			int m = cam.ModelId();
			std::cout << m << "\n";
			mc.SetModelId(cam.ModelId());
			mc.SetWidth(cam.Width());
			mc.SetHeight(cam.Height());
			mc.SetPrincipalPointX(cam.PrincipalPointX());
			mc.SetPrincipalPointY(cam.PrincipalPointY());
		}
	}
	std::sort(flengths.begin(), flengths.end());
	mc.SetFocalLength(flengths[flengths.size()/2]);
	reconstruction.AddCamera(mc);
	std::cout << "Median camera created\n";

    //here the first set is reconstructed and bundle adjusted
    //we can start to find poses of the second set cameras
    EPNPEstimator residualCheck;
	reg_next_success = true;
	while(reg_next_success)
	{
		std::cout << "Sequential registration\n";
		//find possible next images, omit those which have already been reconstructed although they have not been trully reconstructed, only their two possible poses have been found
		//next images will have to be from the original images, not from the newly added ones (maybe they will be added after this procedure)
		reg_next_success = false;
		const std::vector<image_t> next_images = mapper.FindNextImagesSecondSet(options_->Mapper(), main_set);
		std::cout << next_images.size() << " images" << "\n";
		std::vector<image_t> good_images;
		std::vector<std::vector<image_t>> images_1;
		std::vector<std::vector<image_t>> images_2;
		for(size_t i=0;i<next_images.size();++i)
		{
			//here will be the whole registration process
			const image_t next_image_id = next_images[i];
			//const Image& next_image = reconstruction.Image(next_image_id);

			PrintHeading1(StringPrintf("Sequentially registering image #%d (%d)", next_image_id, reconstruction.NumRegImages() + 1));
			std::cout << mc_id << "\n";
			//int found_poses = mapper.SeqRegisterImage2(options_->Mapper(), next_image_id,mc_id);
			int found_poses = 0;
			try
			{
				found_poses = mapper.SeqRegisterImage(options_->Mapper(), next_image_id);
				//throw 20;
			}
			catch(const int e)
			{
				std::cout << "ERROR\n";
				//throw 20;
			}
			catch(std::bad_function_call& e)
			{
				std::cout << "ERROR 2\n";
			}
			std::cout << "SEQ REG RESULT: " << found_poses << "\n";
			//TODO
			//this is experimental
			if(found_poses >= 2)
				good_images.push_back(next_image_id);
			
		}
		
		for(image_t img : good_images)
		{
			//prepare for adding new images
			Image& next_image = reconstruction.Image(img);
			std::vector<class Point2D> points_orig = next_image.Points2D();
			//copy vector of 2D points and add the copy to the phantom image (if there are more phantoms, make more copies)
			std::vector<class Point2D> points_copy;
			for(Point2D pnt : points_orig)
			{
				points_copy.push_back(pnt);
			}
			Camera cam = reconstruction.Camera(next_image.CameraId());
			
			std::vector<Eigen::Vector4d> qvecs = reconstruction.Image(img).Qvecs();
			std::vector<Eigen::Vector3d> tvecs = reconstruction.Image(img).Tvecs();
			//inlier corrs are already in the image but we don't know which of them use, so it will be safer to give them as argument (they won't be in the second image)
			std::vector<std::vector<std::pair<point2D_t, point3D_t>>> tri_corrs = next_image.InlierCorrs();
			//set found pose as the official pose of the image
			next_image.SetQvec(qvecs[0]);
			next_image.SetTvec(tvecs[0]);
			next_image.SetGroup(1);
			//finish registration of the first image
			//triangulate and do bundle adjustment (but both later), do the same with the phantom image
			mapper.FinishRegistration(options_->Mapper(), img, tri_corrs[0]);
			//add a camera to the camera list
			std::string name = next_image.Name();
			int take = take_map[name];
			//set = takes_map
			/*if(first_set.count(img))
				set = 1;
			else if(second_set.count(img))
				set = 2;
			else
				set = 3;*/
			std::cout << "CAMS\n";
			camera_log << img << " " << take << " " << recon_set << " " << qvecs[0](0) << " " << qvecs[0](1) << " " << qvecs[0](2) << " " << qvecs[0](3) << " "
				<< tvecs[0](0) << " " << tvecs[0](1) << " " << tvecs[0](2) << " " << cam.Width() << " " << cam.Height() << " " << cam.FocalLength() << " "
				<< cam.PrincipalPointX() << " " << cam.PrincipalPointY() << " " << next_image.seq_inliers[0] << " " << img << "\n";
			
			for(size_t i=1;i<qvecs.size();i++)
			{
				//create second 'phantom' image and register it (it will have the same camera (not just params but also id) and the same image name)
				image_t id_p = reconstruction.NumImages()+1;
				next_image.SetCorr(id_p);
				Image phantom;
				phantom.SetImageId(id_p);
				phantom.SetCameraId(next_image.CameraId());
				phantom.SetUp(cam);
				phantom.SetName(next_image.Name());
				phantom.SetQvec(qvecs[i]);
				phantom.SetTvec(tvecs[i]);
				phantom.SetGroup(i+1);
				phantom.SetCorr(img);
				phantom.SetPoints2D(points_orig);
				std::cout << id_p << " " << next_image.Points2D().size() << "\n";
				reconstruction.AddImage(phantom);
				std::vector<image_pair_t> old_pairs; 
				std::vector<image_pair_t> new_pairs;
				reconstruction.InitNewImage(id_p, img, &old_pairs, &new_pairs);
				//HERE
				//scene_graph.AddNewImage(id_p, reconstruction.Image(id_p).NumPoints2D(),img);
				//scene_graph.CopyCorrespondences(id_p, img, old_pairs, new_pairs);
				mapper.InitImage(id_p, img, old_pairs, new_pairs);
				mapper.FinishRegistration(options_->Mapper(), id_p, tri_corrs[i]);
				//also triangulate both images
				if(i==1)
				{
					for(std::pair<point2D_t, point3D_t> corr : tri_corrs[1])
					{
						ph_out << corr.second << "\n";
					}
				}
				camera_log << img << " " << take << " " << recon_set << " " << qvecs[i](0) << " " << qvecs[i](1) << " " << qvecs[i](2) << " " << qvecs[i](3) << " "
					<< tvecs[i](0) << " " << tvecs[i](1) << " " << tvecs[i](2) << " " << cam.Width() << " " << cam.Height() << " " << cam.FocalLength() << " "
					<< cam.PrincipalPointX() << " " << cam.PrincipalPointY() << " " << next_image.seq_inliers[i] << " " << id_p << "\n";
			}
			for(std::pair<point2D_t, point3D_t> corr : tri_corrs[0])
			{
				real_out << corr.second << "\n";
			}
		}
		//maybe will also be something under the loop
		//bundle adjustment
		//IterativeGlobalRefinement(*options_, &mapper);
		//AdjustGlobalBundle(*options_, &mapper);
	}

	ph_out.close();
	real_out.close();
	camera_log.close();

    // If the total number of images is small then do not enforce the minimum
    // model size so that we can reconstruct small image collections.
    const size_t min_model_size =
        std::min(database_cache_.NumImages(),
                 static_cast<size_t>(options_->min_model_size));
    if ((options_->multiple_models &&
         reconstruction.NumRegImages() < min_model_size) ||
        reconstruction.NumRegImages() == 0) {
      mapper.EndReconstruction(kDiscardReconstruction);
      reconstruction_manager_->Delete(reconstruction_idx);
    } else {
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
    }

    Callback(LAST_IMAGE_REG_CALLBACK);

    const size_t max_num_models = static_cast<size_t>(options_->max_num_models);
    if (initial_reconstruction_given || !options_->multiple_models ||
        reconstruction_manager_->Size() >= max_num_models ||
        mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {
      break;
    }
  }
  //delete &mapper;
}

}  // namespace colmap
