/*****************************************************************//**
 * \file    PnP.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "PnP.h"
#include "CameraUtils.h"
#include "TumDataset.h"
#include "../EigenExample/GuardTimer.h"
#include "../PangolinExample/PangolinUtils.h"
#include "../PangolinExample/PointCloud.h"
#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <iostream>

#pragma region Gaussian Newton

Sophus::SE3d GaussianNewtonPnP(const std::vector< std::pair<Eigen::Vector3d, Eigen::Vector2d> > &samples, const Eigen::Matrix3d &K) {
	//const double &fx = K(0, 0), &fy = K(1, 1), &cx = K(0, 2), &cy = K(1, 2);

	typedef Eigen::Matrix<double, 6, 1> Vector6d;
	typedef Eigen::Matrix<double, 6, 6> Matrix6d;
	Sophus::SE3d pose;
	double cost = std::numeric_limits<double>::infinity();

	double weight = 1.0; //模拟退火
	static constexpr bool UseAnnel = false;
	for (int iter = 0; iter < 100; ++iter) {
		Matrix6d H = Matrix6d::Zero();
		Vector6d b = Vector6d::Zero();

		cost = 0.0;
		//for (int i = 0; i < samples.size(); ++i) {
		//	const auto &[P, u] = samples[i]; // P是三维点，u是对应的二维点
		for (const auto &[P, u] : samples) {
			//const double &Xpi = P[0], &Ypi = P[1], &Zpi = P[2];

			Eigen::Vector3d pc = pose * P;
			//const double &Xpi = pc[0], &Ypi = pc[1], &Zpi = pc[2];
			//double invZpi = 1.0 / Zpi, invZpi2 = invZpi * invZpi;
			Eigen::Vector2d proj = Point3DToPixel(K, pc);
			Eigen::Vector2d e = u - proj;
			cost += e.squaredNorm();

			//Eigen::Vector4d P_{ P[0], P[1], P[2], 1.0 };
			//Eigen::Vector4d TP_ = pose.matrix() * P_;
			//Eigen::Vector3d TP{ TP_[0] / TP_[3], TP_[1] / TP_[3], TP_[2] / TP_[3] };
			//Eigen::Vector3d KTP_ = K * TP;
			//Eigen::Vector2d KTP{ KTP_[0] / KTP_[2], KTP_[1] / KTP_[2] };
			//Eigen::Vector2d e = (u - KTP);

			//Eigen::Matrix<double, 2, 6> J;
			//J << fx * invZpi,         0.0, -fx * Xpi * invZpi2,          fx *Xpi *Ypi *invZpi2, fx + fx * Xpi * Xpi * invZpi2, -fx * Ypi * invZpi,
			//	         0.0, fy * invZpi, -fy * Ypi * invZpi2, -fy - fy * Ypi * Ypi * invZpi2,         fy *Xpi *Ypi *invZpi2,    fy *Xpi *invZpi;

			Matrix26d J = Jaccobian_Pixel_To_DeltaSE3(K, pc);

			H += J.transpose() * J; // (-J)^T(-J)
			b += J.transpose() * e; // - (-J^T)e
		}

		Vector6d delta = H.ldlt().solve(b);

		if (std::isnan(delta[0])) {
			std::cerr << "result is nan!" << std::endl;
			break;
		}

		if constexpr (UseAnnel) {
			pose = Sophus::SE3d::exp(delta * weight) * pose;
		} else {
			pose = Sophus::SE3d::exp(delta) * pose;
		}

		std::cout << "at iter: " << iter << " got cost: " << cost << " weight: " << weight << std::endl;
		if (delta.norm() < std::numeric_limits<double>::epsilon()) { // coverge
			break;
		}

		if constexpr (UseAnnel) { // 模拟退火
			double weight_tmp = iter / 50.0 / 2.0;
			weight = std::exp(-weight_tmp * weight_tmp);
		}
	}
	std::cout << "Pose: " << pose.matrix() << std::endl;
	std::cout << "Last cost: " << cost << std::endl;
	return pose;
}

#pragma endregion

#pragma region Ceres
#include <ceres/ceres.h>
#include "SE3Manifold.h"

class PnPCostFunction : public ceres::SizedCostFunction<2, 6> {
protected:
	Eigen::Vector3d p3d; Eigen::Vector2d p2d;
	const Eigen::Matrix3d &K;
public:
	PnPCostFunction(Eigen::Vector3d point, Eigen::Vector2d observation, const Eigen::Matrix3d &K) : p3d(point), p2d(observation), K(K) {

	}

	//jacobians是row-major的, jacobians[0]是2x6矩阵,第一索引为parameters,第二索引是flatten的数组
	//jacobians大小的计算：jacobian=d residuals / parameters^T, 分子是2x1, 分母是6x1, 因此是2x6
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

		Eigen::Map<const Vector6d> camera{ parameters[0] };
		Sophus::SE3d pose = Sophus::SE3d::exp(camera);
		Eigen::Vector3d estimatePoint = pose * p3d;
		Eigen::Vector2d reproj = Point3DToPixel(K, estimatePoint);

		Eigen::Map<Eigen::Vector2d>{residuals} = p2d - reproj;
		if (jacobians == nullptr || jacobians[0] == nullptr) return true;

		//Matrix26d j = Jaccobian_Pixel_To_DeltaSE3(K, estimatePoint);
		//for (int y = 0; y < 2; ++y) {
		//	for (int x = 0; x < 6; ++x) {
		//		jacobians[0][y*6+x] = -j(y, x);
		//	}
		//}
		Eigen::Map< Eigen::Matrix<double, 2, 6, Eigen::RowMajor> >{jacobians[0]} = -Jaccobian_Pixel_To_DeltaSE3(K, estimatePoint);

		return true;
	}
};

Sophus::SE3d CeresPnP(const std::vector< std::pair<Eigen::Vector3d, Eigen::Vector2d> > &samples, const Eigen::Matrix3d &K) {
	ceres::Problem problem;
	Vector6d pose = Sophus::SE3d().log(); //提供符合SE3d的初始值
	for (auto &sample : samples) {
		problem.AddResidualBlock(new PnPCostFunction(sample.first, sample.second, K), new ceres::HuberLoss(0.2), pose.data());
	}
	problem.SetManifold(pose.data(), new SE3Manifold);

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;

	std::cout << "R: " << Sophus::SE3d::exp(pose).matrix() << std::endl;
	return Sophus::SE3d::exp(pose);
}

#pragma endregion

#pragma region G2O
#include "../KittiSLAM/Algorithm/G2OTypes.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

class PnPEdge : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE3Pose> {
public:
	PnPEdge(Eigen::Vector3d p3d, const Eigen::Matrix3d &K) : m_p3d{p3d}, m_K{K} {

	}
	// 通过 BaseUnaryEdge 继承
	virtual void computeError() override {
		auto v = static_cast<VertexSE3Pose*>(_vertices[0]);
		Eigen::Vector2d projectivePoint = Point3DToPixel(m_K, v->estimate() * m_p3d);
		_error = _measurement - projectivePoint;
	}
	virtual bool read(std::istream &is) override {
		return false;
	}
	virtual bool write(std::ostream &os) const override {
		return false;
	}
	virtual void linearizeOplus() override {
		const VertexSE3Pose *pose = static_cast<VertexSE3Pose*>(_vertices[0]);
		const Eigen::Vector3d transformPoint = pose->estimate() * m_p3d;
		_jacobianOplusXi = -Jaccobian_Pixel_To_DeltaSE3(m_K, transformPoint);
	}
private:
	Eigen::Vector3d m_p3d;
	const Eigen::Matrix3d &m_K;
};

Sophus::SE3d G2OPnP(const std::vector< std::pair<Eigen::Vector3d, Eigen::Vector2d> > &samples, const Eigen::Matrix3d &K) {
	Sophus::SE3d pose{};

	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > BlockSolverType;
	typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

	auto solver = new g2o::OptimizationAlgorithmLevenberg(
		std::make_unique<BlockSolverType>(
			std::make_unique<LinearSolverType>()
		)
	);

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(true);

	VertexSE3Pose *v = new VertexSE3Pose();
	v->setId(0);
	v->setEstimate(pose);
	optimizer.addVertex(v);

	for (int i = 0; i < samples.size(); ++i) {
		PnPEdge *edge = new PnPEdge(samples[i].first, K);
		edge->setId(i);
		edge->setVertex(0, v);
		edge->setMeasurement(samples[i].second);
		edge->setInformation(Eigen::Matrix2d::Identity());
		edge->setRobustKernel(new g2o::RobustKernelHuber);
		optimizer.addEdge(edge);
	}

	optimizer.initializeOptimization();
	optimizer.optimize(40);

	pose = v->estimate();
	std::cout << pose.matrix() << std::endl;
	return pose;
}
#pragma endregion


void Pose3dTo2d() {
	TumDataset dataset;
	MatchResult matches = MatchTwoImages(dataset.img1, dataset.img2);

	cv::Mat visualizer;
	cv::drawMatches(dataset.img1, matches.features1.keypoints, dataset.img2, matches.features2.keypoints, matches.matches, visualizer);
	cv::imshow("matches", visualizer);
	cv::waitKey();
	cv::destroyAllWindows();

	std::vector< std::pair<Eigen::Vector3d, Eigen::Vector2d> > samples;
	std::vector<cv::Point2f> pt1; std::vector<cv::Point3f> pt2;
	for (cv::DMatch &match : matches.matches) {
		cv::KeyPoint p2d = matches.features1.keypoints[match.queryIdx];
		unsigned short depth = GetDepth(dataset.depth1, p2d.pt);
		if (depth == 0) continue;

		double dd = depth / 5000.0;
		Eigen::Vector3d p3d_ = PixelToPoint3D(dataset.intrinsic.Eigen(), p2d.pt, dd);
		Eigen::Vector2d p1d_{ matches.features2.keypoints[match.trainIdx].pt.x, matches.features2.keypoints[match.trainIdx].pt.y };
		samples.push_back(std::make_pair(p3d_, p1d_));

		pt1.emplace_back(p1d_[0], p1d_[1]);
		pt2.emplace_back(p3d_[0], p3d_[1], p3d_[2]);
	}

	//{
	//	GuardTimer timer{ "OpenCV" };
	//	cv::Mat r, t, R;
	//	cv::solvePnP(pt2, pt1, dataset.intrinsic.OpenCV(), cv::Mat(), r, t, false);
	//	cv::Rodrigues(r, R);
	//	std::cout << "OpenCV R: " << R << std::endl;
	//	std::cout << "OpenCV t: " << t << std::endl;
	//}

	Sophus::SE3d pose;
	{
		GuardTimer timer{ "Gaussian Newton" };
		pose = GaussianNewtonPnP(samples, dataset.intrinsic.Eigen());
	}
	{
		GuardTimer timer{ "Ceres" };
		pose = CeresPnP(samples, dataset.intrinsic.Eigen());
	}
	{
		GuardTimer timer{ "G2O" };
		pose = G2OPnP(samples, dataset.intrinsic.Eigen());
	}

	PointCloud pointCloud;
	for (int y = 0; y < dataset.img1.rows; ++y) {
		for (int x = 0; x < dataset.img1.cols; ++x) {
			unsigned short depth = GetDepth(dataset.depth1, y, x);
			if (depth == 0) continue;
			double dd = depth / 5000.0;

			Eigen::Vector3d point = PixelToPoint3D(dataset.intrinsic.Eigen(), y, x, dd);
			Eigen::Vector2d point2 = Point3DToPixel(dataset.intrinsic.Eigen(), pose * point);
			if (GetDepth(dataset.depth2, point2) == 0) {
				//pointCloud.emplace_back(MakePointInCloud(point, Eigen::Vector3d{1.0, 0.0, 0.0})); //查看outliner
				continue;
			}

			Eigen::Vector3d color = GetColor(dataset.img1, y, x);
			Eigen::Vector3d color2 = GetColor(dataset.img2, point2);
			pointCloud.emplace_back(MakePointInCloud(point, (color + color2) / 2)); //融合颜色
		}
	}
	WindowMainLoop([&]() {
		DrawPointCloud(pointCloud);
	});
}