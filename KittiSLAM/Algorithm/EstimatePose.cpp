/*****************************************************************//**
 * \file    EstimatePose.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "EstimatePose.h"
#include <g2o/core/robust_kernel_impl.h>

namespace EstimatePose {

namespace Impl {
class DirectMethodCostFunction : public ceres::SizedCostFunction<1, 6> {
private:
	Eigen::Vector2d m_2d;
	Eigen::Vector3d m_3d;
	const Camera &m_camera;

	cv::Mat m_first;
	cv::Mat m_second;
	const int m_halfPatchSize;
public:
	DirectMethodCostFunction(cv::Mat firstImage, Eigen::Vector2d keypointLocation, Eigen::Vector3d corresponsed3DPoint, cv::Mat secondImage, const Camera &camera, int halfPatchSize = 1) :
		m_2d{ keypointLocation }, m_3d{ corresponsed3DPoint }, m_camera{ camera }, m_halfPatchSize{ halfPatchSize } {

	}

	DirectMethodCostFunction(cv::Mat firstImage, Eigen::Vector2d keypointLocation, cv::Mat depthImage, cv::Mat secondImage, const Camera &camera) :
		DirectMethodCostFunction(firstImage, keypointLocation, camera.PixelToPoint3D(keypointLocation, ImageUtils::GetDepth(depthImage, keypointLocation)), secondImage, camera) {

	}

	// 通过 SizedCostFunction 继承
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
		const Sophus::SE3d T21 = Sophus::SE3d::exp(Eigen::Map<const Vector6d>{ parameters[0]});
		const Eigen::Vector3d transformedPoint = T21 * m_3d;
		const Eigen::Vector2d reprojectPoint = m_camera.Point3DToPixel(transformedPoint);

		const Matrix26d du_dxi = ImageUtils::Jaccobian_Pixel_To_DeltaSE3(m_camera.K(), transformedPoint);

		double &residual = residuals[0]; //I1-I2=I1(p)-I2(pi(TKp))
		residual = 0;

		Matrix16d J = Matrix16d::Zero();
		for (int dy = -m_halfPatchSize; dy <= m_halfPatchSize; ++dy) {
			for (int dx = -m_halfPatchSize; dx <= m_halfPatchSize; ++dx) {
				const Eigen::Vector2d offset{ dx, dy };
				const Eigen::Vector2d reprojPlusOffset = reprojectPoint + offset;

				double e = ImageUtils::GetGrayColor(m_first, m_2d + offset) - ImageUtils::GetGrayColor(m_second, reprojPlusOffset);

				J += -ImageUtils::GetGradient(m_second, reprojPlusOffset).transpose() * du_dxi;

				residual += e * e;
			}
		}

		if (jacobians == nullptr) return true;
		if (jacobians[0] == nullptr) return true;
		Eigen::Map<Matrix16d>{jacobians[0]} = J;
		return true;
	}

};
}


template <>
void CalculateRelativePose<Solver::Ceres>(const DirectMethodConfiguration &config, const Camera &camera, Sophus::SE3d &output_pose) {
	assert(config.keyPointsInFirstImage.size() == config.corresponding3DKeyPointsInFirstImage.size());

	ceres::Problem problem;
	Vector6d pose = output_pose.log();
	for (int i = 0; i < config.keyPointsInFirstImage.size(); ++i) {
		problem.AddResidualBlock(new Impl::DirectMethodCostFunction{
			config.firstImage, config.keyPointsInFirstImage[i], config.corresponding3DKeyPointsInFirstImage[i], config.secondImage, camera
			}, new ceres::HuberLoss(0.5), pose.data());
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	output_pose = Sophus::SE3d::exp(pose);
}

template <>
void CalculateRelativePose<Solver::G2O>(const DirectMethodConfiguration &config, const Camera &camera, Sophus::SE3d &output_pose) {
	auto solver = new g2o::OptimizationAlgorithmLevenberg(
		std::make_unique<g2o::BlockSolver_6_3>(
			std::make_unique< g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType> >()
		)
	);

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	VertexSE3Pose *vertex_pose = new VertexSE3Pose();
	vertex_pose->setId(0);
	vertex_pose->setEstimate(output_pose);
	optimizer.addVertex(vertex_pose);

	std::vector<EdgeProjectionPose *> edges;
	for (int i = 0; i < config.keyPointsInFirstImage.size(); ++i) {
		EdgeProjectionPose *edge = edges.emplace_back();
		edge->setId(i);
		edge->setVertex(0, vertex_pose);
		edge->setMeasurement(config.keyPointsInFirstImage[i]);
		edge->setInformation(Eigen::Matrix2d::Identity());
		edge->setRobustKernel(new g2o::RobustKernelHuber);
		optimizer.addEdge(edge);
	}

	int cnt_outlier = 0;
	for (int iter = 0; iter < 4; ++iter) {
		optimizer.initializeOptimization();
		optimizer.optimize(10);
		cnt_outlier = 0;

		for (size_t i = 0; i < edges.size(); ++i) {
			auto e = edges[i];

		}
	}
}

template <>
void CalculateRelativePose<Solver::G2O>(const LKMethodConfiguration &config, const Camera &camera, Sophus::SE3d &output_pose) {

}

bool Triangulation(const std::vector<Sophus::SE3d> &poses, const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &pt_world) {
	Eigen::MatrixX<double> A{ 2 * poses.size(), 4 };
	Eigen::VectorX<double> b{ 2 * poses.size() };

	b.setZero();
	for (int i = 0; i < poses.size(); ++i) {
		Matrix34d m = poses[i].matrix3x4();
		A.block<1, 4>(2 * i, 0) = points[i].x() * m.row(2) - m.row(0);
		A.block<1, 4>(2 * i + 1, 0) = points[i].y() * m.row(2) - m.row(1);
	}
	auto svd = A.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>();
	pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

	if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
		// 解质量不好，放弃
		return true;
	}
	return false;
}

}

