/*****************************************************************//**
 * \file    EstimatePose.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once

#include "../Devices/Camera.h"
#include "ImageUtils.h"
#include "EigenType.h"
#include "G2OTypes.h"
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <Eigen/Eigen>

namespace EstimatePose {

enum class Solver {
	Ceres,
	G2O
};

struct DirectMethodConfiguration {
	cv::Mat firstImage, secondImage;
	std::vector<Eigen::Vector2d> keyPointsInFirstImage;
	std::vector<Eigen::Vector3d> corresponding3DKeyPointsInFirstImage;

	//Ĭ�����firstImage��secondImage��λ�ˣ�����ҪsecondImage��firstImage��λ�ˣ�������Ϊtrue
	bool isReverseOutput = false;
};

struct LKMethodConfiguration {
	cv::Mat firstImage, secondImage;
	std::vector<Eigen::Vector2d> keyPointsInFirstImage;
	std::vector<Eigen::Vector2d> correspondingKeyPointsInSecondImage;

	//Ĭ�����firstImage��secondImage��λ�ˣ�����ҪsecondImage��firstImage��λ�ˣ�������Ϊtrue
	bool isReverseOutput = false;
};



template <Solver SolverType>
void CalculateRelativePose(const DirectMethodConfiguration &config, const Camera &camera, Sophus::SE3d &output_pose);

template <Solver SolverType>
void CalculateRelativePose(const LKMethodConfiguration &config, const Camera &camera, Sophus::SE3d &output_pose);

bool Triangulation(const std::vector<Sophus::SE3d> &poses, const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &pt_world);

}