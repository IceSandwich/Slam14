/*****************************************************************//**
 * \file    ICP.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "ICP.h"
#include "TumDataset.h"
#include "CameraUtils.h"
#include "../EigenExample/GuardTimer.h"
#include "../PangolinExample/PointCloud.h"
#include <sophus/se3.hpp>
#include <Eigen/Eigen>
#include <iostream>

struct RemoveCentroid {
	std::vector<Eigen::Vector3d> source, target;
	Eigen::Vector3d massOfSource, massOfTarget;
	size_t size;

	RemoveCentroid(const std::vector<Eigen::Vector3d> &target, const std::vector<Eigen::Vector3d> &source) {
		assert(source.size() == target.size());
		size = source.size();

		massOfSource = Eigen::Vector3d::Zero();
		massOfTarget = Eigen::Vector3d::Zero();
		for (int i = 0; i < size; ++i) {
			massOfSource += source[i];
			massOfTarget += target[i];
		}
		massOfSource /= size;
		massOfTarget /= size;

		std::vector<Eigen::Vector3d> q, q_;
		for (int i = 0; i < source.size(); ++i) {
			this->source.emplace_back(source[i] - massOfSource);
			this->target.emplace_back(target[i] - massOfTarget);
		}
	}
};

enum ICPMethod {
	SVD = 0,
	PolarDecomposition,
	NonQuatRegulation,
	MyMethod,
};

template <ICPMethod Method>
Sophus::SE3d TraditionalICP(const std::vector<Eigen::Vector3d> &target, const std::vector<Eigen::Vector3d> &source) {
	assert(source.size() == target.size());

	RemoveCentroid removedCentroid{ target, source };

	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i < removedCentroid.size; ++i) {
		W += removedCentroid.source[i] * removedCentroid.target[i].transpose();
	}

	Eigen::Matrix3d R;

	if constexpr (Method == ICPMethod::SVD) {
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::DecompositionOptions::ComputeFullU | Eigen::DecompositionOptions::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();

		std::cout << "U: " << U << std::endl;
		std::cout << "V: " << V << std::endl;

		R = V * U.transpose(); //U * V.transpose();
	} else if constexpr (Method == ICPMethod::PolarDecomposition) {
		// Ref: https://zalo.github.io/blog/polar-decomposition/#robust-polar-decomposition

		Eigen::Quaterniond q; //extract rotation from W
		for (int iter = 0; iter < 100; ++iter) {
			Eigen::Matrix3d R = q.matrix();
			Eigen::Vector3d omega_factor1 = R.col(0).cross(W.col(0)) + R.col(1).cross(W.col(1)) + R.col(2).cross(W.col(2));
			double omega_factor2 = std::fabs(R.col(0).dot(W.col(0)) + R.col(1).dot(W.col(1)) + R.col(2).dot(W.col(2)) + 1.0e-9);

			Eigen::Vector3d omega = omega_factor1 / omega_factor2;
			double w = omega.norm();
			if (w < 1.0e-9) break;
			q = Eigen::Quaterniond(Eigen::AngleAxisd(w, (1.0 / w) * omega)) * q;
			q.normalize();

			std::cout << "at iter: " << iter << " got cost: " << (removedCentroid.massOfTarget - q * removedCentroid.massOfSource).squaredNorm() << std::endl;
		}

		R = q.matrix();
	} else if constexpr (Method == ICPMethod::NonQuatRegulation) { // 目前有问题
		Eigen::Vector3d inX = W.col(0).normalized(), inY = W.col(1).normalized(), inZ = W.col(2).normalized();
		std::array<double, 3> mB{ 1.0, 1.0, 1.0 }; // inX.norm(), inY.norm(), inZ.norm() };
		for (int iter = 0; iter < 50; ++iter) {
			Eigen::Vector3d unitX = (inY.cross(inZ) + inX).normalized();
			Eigen::Vector3d unitY = (inZ.cross(inX) + inY).normalized();
			Eigen::Vector3d unitZ = (inX.cross(inY) + inZ).normalized();
			inX = unitX * mB[0];
			inY = unitY * mB[1];
			inZ = unitZ * mB[2];
		}
		R.col(0) = inX.normalized();
		R.col(1) = inY.normalized();
		R.col(2) = inZ.normalized();
	} else if constexpr (Method == ICPMethod::MyMethod) { // 目前有问题
		Eigen::Vector3d inX = W.col(0).normalized(), inY = W.col(1).normalized(), inZ = W.col(2).normalized();
		Eigen::Vector3d xyCenter = (inX + inY).normalized();
		Eigen::Vector3d center = (xyCenter + inZ).normalized();

		Eigen::Vector3d vecNearZPenCenter = center.cross(center.cross(xyCenter)).normalized();
		R.col(2) = (vecNearZPenCenter + center).normalized(); // half center and vecNearZPenCenter

		Eigen::Vector3d vecNearXYPlanePenCenter = (center - vecNearZPenCenter).normalized();
		R.col(1) = (center.cross(vecNearXYPlanePenCenter).normalized() + vecNearXYPlanePenCenter).normalized();
		R.col(0) = R.col(1).cross(R.col(2)).normalized();

		Eigen::Matrix3d normalW;
		normalW.col(0) = inX;
		normalW.col(1) = inY;
		normalW.col(2) = inZ;
		std::cout << "W: " << normalW << std::endl;
		std::cout << "R: " << R << std::endl;
	} else {
		assert(false, "Unknown Method!");
	}

	if (R.determinant() < 0) R = -R;
	Eigen::Vector3d t = removedCentroid.massOfTarget - R * removedCentroid.massOfSource; // massOfSource - R * massOfTarget;

	std::cout << "R: " << R << std::endl;
	std::cout << "t: " << t << std::endl;

	return Sophus::SE3d{ R, t };
}

// 优化位姿T
Sophus::SE3d GaussianNewtonICP(const std::vector<Eigen::Vector3d> &target, const std::vector<Eigen::Vector3d> &source) {
	assert(source.size() == target.size());

	double cost = std::numeric_limits<double>::infinity();
	Sophus::SE3d pose;
	for (int iter = 0; iter < 100; ++iter) {
		typedef Eigen::Matrix<double, 6, 6> Matrix6d;
		typedef Eigen::Matrix<double, 6, 1> Vector6d;

		Matrix6d H = Matrix6d::Zero();
		Vector6d b = Vector6d::Zero();
		cost = 0.0;

		for (int i = 0; i < source.size(); ++i) {
			Eigen::Vector3d p = pose * source[i];
			Eigen::Vector3d e = target[i] - p;

			Eigen::Matrix<double, 3, 6> J;
			//J << -1,  0,  0,     0, -p[2],  p[1],
			//	  0, -1,  0,  p[2],     0, -p[0],
			//	  0,  0, -1, -p[1],  p[0],     0;
			J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
			J.block<3, 3>(0, 3) = Sophus::SO3d::hat(p);

			H += J.transpose() * J;
			b += -J.transpose() * e;

			cost += e.squaredNorm();
		}

		Vector6d delta = H.ldlt().solve(b);
		if (delta.norm() < std::numeric_limits<double>::epsilon()) continue;
		pose = Sophus::SE3d::exp(delta) * pose;
		std::cout << "at iter: " << iter << " got cost: " << cost << std::endl;
	}

	std::cout << "Final result: " << pose.matrix() << std::endl;
	std::cout << "Last cost: " << cost << std::endl;
	return pose;
}

// 仅优化旋转R
Sophus::SE3d GaussianNewtonICP2(const std::vector<Eigen::Vector3d> &target, const std::vector<Eigen::Vector3d> &source) {
	assert(source.size() == target.size());

	RemoveCentroid removedCentroid{ target, source };
	Sophus::SO3d sR;

	double error;
	for (int iter = 0; iter < 100; ++iter) {
		Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
		Eigen::Vector3d b = Eigen::Vector3d::Zero();

		error = 0.0;
		for (int i = 0; i < removedCentroid.size; ++i) {
			Eigen::Vector3d p = sR * source[i];
			Eigen::Vector3d e = target[i] - p;

			Eigen::Matrix3d J; // 只对R求导
			J = -Sophus::SO3d::hat(p);
			//J <<  0.0,  p[2], -p[1],
			//	-p[2],   0.0,  p[0],
			//	 p[1], -p[0],   0.0;

			H += J * J.transpose();
			b += -J * e;
			error += e.squaredNorm();
		}

		Eigen::Vector3d delta = H.ldlt().solve(b);
		if (delta.norm() < std::numeric_limits<double>::epsilon()) continue;

		sR = Sophus::SO3d::exp(delta) * sR;
		std::cout << "at iter: " << iter << " error: " << error << std::endl;
	}
	Eigen::Matrix3d R = sR.matrix();
	Eigen::Vector3d t = removedCentroid.massOfTarget - R * removedCentroid.massOfSource;

	std::cout << "R: " << R << std::endl;
	std::cout << "t: " << t << std::endl;

	return Sophus::SE3d{ R, t };
}

void Pose3dTo3d() {
	TumDataset dataset;
	MatchResult matches = MatchTwoImages(dataset.img1, dataset.img2, false, cv::ORB::create(1000));

	std::vector<Eigen::Vector3d> pc1, pc2;
	for (cv::DMatch &match : matches.matches) {
		cv::Point2f pointInImg1 = matches.features1.keypoints[match.queryIdx].pt;
		cv::Point2f pointInImg2 = matches.features2.keypoints[match.trainIdx].pt;

		unsigned short depthInImg1 = GetDepth(dataset.depth1, pointInImg1);
		unsigned short depthInImg2 = GetDepth(dataset.depth2, pointInImg2);

		if (depthInImg1 == 0 || depthInImg2 == 0) continue;

		Eigen::Vector3d point3DInImg1 = PixelToPoint3D(dataset.intrinsic.Eigen(), pointInImg1, depthInImg1 / 5000.0);
		Eigen::Vector3d point3DInImg2 = PixelToPoint3D(dataset.intrinsic.Eigen(), pointInImg2, depthInImg2 / 5000.0);

		pc1.push_back(point3DInImg1);
		pc2.push_back(point3DInImg2);
	}

	Sophus::SE3d pose;
	{
		GuardTimer time{ "SVD" };
		pose = TraditionalICP<ICPMethod::SVD>(pc1, pc2);
	}
	//{
	//	GuardTimer time{ "SVD Polar decomposition" };
	//	pose = TraditionalICP<ICPMethod::PolarDecomposition>(pc1, pc2);
	//}
	//{
	//	GuardTimer time{ "SVD NonQuat Polar decomposition" };
	//	pose = TraditionalICP<ICPMethod::NonQuatRegulation>(pc1, pc2);
	//}
	{
		GuardTimer time{ "My method" };
		pose = TraditionalICP<ICPMethod::MyMethod>(pc1, pc2);
	}
	//{ // 只对R优化，效果不是很好
	//	GuardTimer time{ "GussianNewton2" };
	//	pose = GaussianNewtonICP2(pc1, pc2);
	//}
	//{ // 对T优化，跟SVD效果相同，但是很慢
	//	GuardTimer time{ "GussianNewton" };
	//	pose = GaussianNewtonICP(pc1, pc2);
	//}

	PointCloud vpc1, vpc2;
	for (int i = 0; i < pc1.size(); ++i) {
		vpc1.push_back(MakePointInCloud(pc1[i], Eigen::Vector3d{ 1.0, 0.0, 0.0 }));
		vpc2.push_back(MakePointInCloud(pose * pc2[i], Eigen::Vector3d{ 0.0, 0.0, 1.0 }));
	}
	//WindowMainLoop([&]() {
	//	DrawPointCloud(vpc1);
	//	DrawPointCloud(vpc2);
	//});
}
