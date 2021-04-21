TBSfM
=====

The work is a modified version of SfM pipeline COLMAP https://colmap.github.io, which can handle data described in the main paper.
the most important improvements are in files:
	./src/exe/colmap.cc
	./src/controllers/incremental_mapper.cc
	./src/sfm/incremental_mapper.cc
	./src/sfm/incremental_mapper.h

Install
-------
Please, follow instructions on https://colmap.github.io/install.html to install TBSfM.
We have tested the installation on Linux. If you use Windows, you may use Windows subsystem for Linux.


Run
---
1. in folder ./images create subdirectories called 1, ..., k where k is the number of takes
2. place images from every take to a distinct subdirectory (such as ./images/1)
3. run colmap preprocessor
4. run colmap feature_extractor --database_path ./database.db --image_path ./images
5. run colmap exhaustive_matcher database_path $DATASET_PATH/database.db
6. run colmap mapper --database_path ./database.db --image_path ./images/ --export_path . --Mapper.init_max_reg_trials 5 --Mapper.init_num_trials 400 --Mapper.abs_pose_min_inlier_ratio 0.02
7. run colmap postprocessor

Results
-------
In directory model0 you can find the model of both objects where the background is red and the foreground is green
In directory model1 you can find the model of both objects in real color
In directory model2 you can find the model of the background in real color
In directory model3 you can find the model of the foreground in real color
