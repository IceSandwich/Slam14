/*****************************************************************//**
 * \file    main.cpp
 * \brief   Bundle Adjustment
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include <iostream>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include "common.h"
#include "rotation.h"

class CostFunction {
protected:
	CostFunction(double x, double y) : x(x), y(y) {

	}

public:

	// camera[9]: [0-2]: angle-axis rotation, [3-5]: translation, [6-8]: focal length, radial distoration k1 and k2
	template <typename T>
	bool operator ()(const T *const camera, const T* const point, T *residuals) const { // 数组维度：9, 3, 2
		T predictions[2];

		CamProjectionWithDistortion(camera, point, predictions);

		residuals[0] = predictions[0] - T(x);
		residuals[1] = predictions[1] - T(y);

		return true;
	}

	template <typename T>
	static inline bool CamProjectionWithDistortion(const T camera[9], const T point[3], T predictions[2]) {
		T p[3];

		// p = R * point + t
		AngleAxisRotatePoint(camera, point, p);
		p[0] += camera[3];
		p[1] += camera[4];
		p[2] += camera[5];

		// 归一化
		T xp = -p[0] / p[2];
		T yp = -p[1] / p[2];

		// 去畸变
		T r2 = xp * xp + yp * yp;
		T distortion = T(1.0) + r2 * (camera[7] + camera[8] * r2);
		T undistortionX = xp * distortion;
		T undistortionY = yp * distortion;

		// 转换到图像坐标系, 该数据集tx, ty 都为0
		predictions[0] = camera[6] * undistortionX;
		predictions[1] = camera[6] * undistortionY;

		return true;
	}

	static ceres::CostFunction *Create(double x, double y) {
		return new ceres::AutoDiffCostFunction<CostFunction, 2, 9, 3>(
			new CostFunction(x, y),
			ceres::Ownership::TAKE_OWNERSHIP
		);
	}

private:
	double x, y;
};

void SolveBA(BALProblem &dataset) {
	const int point_block_size = dataset.point_block_size();
	const int camera_block_size = dataset.camera_block_size();
	double *points = dataset.mutable_points();
	double *cameras = dataset.mutable_cameras();
	const double *observations = dataset.observations();

	ceres::Problem problem;
	for (int i = 0; i < dataset.num_observations(); ++i) {
		// 2：输出的维度为2个，9：优化的变量之一是9x1，3：优化的变量之二是3x1
		ceres::CostFunction *cost_function = CostFunction::Create(observations[2 * i + 0], observations[2 * i + 1]);

		double *camera = cameras + camera_block_size * dataset.camera_index()[i];
		double *point = points + point_block_size * dataset.point_index()[i];
		problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), camera, point);
	}

	std::cout << "Solving ceres BA..." << std::endl;
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;
}

int main() {
	BALProblem dataset{ "Data/problem-16-22106-pre.txt" };
	dataset.Normalize();
	dataset.Perturb(0.1, 0.5, 0.5);
	dataset.WriteToPLYFile("initial.ply");
	SolveBA(dataset);
	dataset.WriteToPLYFile("final.ply");
	return 0;
}
