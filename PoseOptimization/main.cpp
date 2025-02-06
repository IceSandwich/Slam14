/*****************************************************************//**
 * \file    main.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include <iostream>
#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <g2o/core/eigen_types.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include "../NumericalOptimization/CameraUtils.h"
#include "../PangolinExample/PointCloud.h"
#include "../NumericalOptimization/SE3Manifold.h"

struct G2ODataset {

	struct Edge {
		int from, to;
		Vector6d pose;
		Matrix6d infoL;

		Edge(int from, int to, Vector6d pose, Matrix6d infoL) : from(from), to(to), pose(pose), infoL(infoL) {

		}
	};

	std::vector<Vector6d> vertices;
	std::vector<Edge> edges;

	static G2ODataset ReadDataset(const char *filename) {
		G2ODataset dataset;

		std::ifstream fin(filename, std::ios::in);
		assert(fin.good());

		//std::array<double, 7 + 6 * 6 / 2 + 6 / 2> lineData{};
		std::array<double, 7> lineData;
		for (int counter = 0; !fin.eof(); ++counter) {
			std::string node_type; fin >> node_type;
			if (!node_type.compare("VERTEX_SE3:QUAT")) {
				int node_id; fin >> node_id; assert(node_id == counter);
				for (int i = 0; i < 7; ++i) fin >> lineData[i];

				dataset.vertices.emplace_back(Sophus::SE3d(
					Eigen::Quaterniond(lineData[6], lineData[3], lineData[4], lineData[5]),
					Eigen::Vector3d(lineData[0], lineData[1], lineData[2])
				).log());
			} else if (!node_type.compare("EDGE_SE3:QUAT")) {
				int edge_from, edge_to; fin >> edge_from >> edge_to; assert(edge_from < dataset.vertices.size() && edge_to < dataset.vertices.size());

				for (int i = 0; i < 7; ++i) fin >> lineData[i];
				Vector6d pose = Sophus::SE3d(
					Eigen::Quaterniond(lineData[6], lineData[3], lineData[4], lineData[5]),
					Eigen::Vector3d(lineData[0], lineData[1], lineData[2])
				).log();

				double sigma[6 * 6 / 2 + 6 / 2];
				for (int i = 0; i < sizeof(sigma) / sizeof(sigma[0]); ++i) fin >> sigma[i];

				Matrix6d info = Matrix6d::Zero();
				info << 0, sigma[1], sigma[2], sigma[3], sigma[4], sigma[5],
					0, 0, sigma[7], sigma[8], sigma[9], sigma[10],
					0, 0, 0, sigma[12], sigma[13], sigma[14],
					0, 0, 0, 0, sigma[16], sigma[17],
					0, 0, 0, 0, 0, sigma[19],
					0, 0, 0, 0, 0, 0;

				Matrix6d digno = Matrix6d::Zero();
				digno << sigma[0], 0, 0, 0, 0, 0,
					0, sigma[6], 0, 0, 0, 0,
					0, 0, sigma[11], 0, 0, 0,
					0, 0, 0, sigma[15], 0, 0,
					0, 0, 0, 0, sigma[18], 0,
					0, 0, 0, 0, 0, sigma[20];

				Matrix6d infoL = (info.transpose() + info + digno).llt().matrixL();
				//if (std::abs(infoL(0, 1) - 0.0) < std::numeric_limits<double>::epsilon()) {
				//	infoL.transposeInPlace(); //保证是上三角
				//}

				dataset.edges.emplace_back(edge_from, edge_to, pose, infoL);
			}
		}
		std::cout << "Vertex: " << dataset.vertices.size() << std::endl;
		std::cout << "Edges: " << dataset.edges.size() << std::endl;
		return dataset;
	}
};

#pragma region Ceres

class PoseGraphCostFunction: public ceres::SizedCostFunction<Sophus::SE3d::DoF, Sophus::SE3d::DoF, Sophus::SE3d::DoF> {
protected:
	const Matrix6d& infoL;
	Sophus::SE3d pose12Inv;
public:
	PoseGraphCostFunction(Vector6d transform, const Matrix6d infoL) : pose12Inv{Sophus::SE3d::exp(transform).inverse()}, infoL{ infoL } {

	}

	// jacobians维度：residuals是6x1, parameters[0]是6x1，因此是6x6
	virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
		Sophus::SE3d T1 = Sophus::SE3d::exp(Eigen::Map<const Vector6d>{parameters[0]});
		Sophus::SE3d T2 = Sophus::SE3d::exp(Eigen::Map<const Vector6d>{parameters[1]});
		Sophus::SE3d Te = pose12Inv * T1.inverse() * T2;

		Eigen::Map<Vector6d>{ residuals } = infoL * Te.log();
		if (jacobians) {
			Matrix6d JrInv = Matrix6d::Zero();
			JrInv.block(0, 0, 3, 3) = JrInv.block(3, 3, 3, 3) = Sophus::SO3d::hat(Te.so3().log());
			JrInv.block(0, 3, 3, 3) = Sophus::SO3d::hat(Te.translation());
			JrInv = 0.5 * JrInv + Matrix6d::Identity();
			Matrix6d JInv_AdTjInv = JrInv * T2.inverse().Adj();
			typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Matrix6dRowMajor;
			if (jacobians[0]) {
				Eigen::Map<Matrix6dRowMajor>{jacobians[0]} = infoL * -JInv_AdTjInv;
			}
			if (jacobians[1]) {
				Eigen::Map<Matrix6dRowMajor>{jacobians[1]} = infoL * JInv_AdTjInv;
			}
		}
		return true;
	}
};

void CeresSolver(G2ODataset& dataset, const char* plyfilename, Eigen::Vector3d origin = Eigen::Vector3d::Zero(), Eigen::Vector3d color = Eigen::Vector3d(0, 0, 250)) {
	ceres::Problem problem;
	SE3Manifold *manifold = new SE3Manifold;
	for (int edgeIndex = 0; edgeIndex < dataset.edges.size(); ++edgeIndex) {
		G2ODataset::Edge &edge = dataset.edges[edgeIndex];
		Vector6d &from = dataset.vertices[edge.from];
		Vector6d &to = dataset.vertices[edge.to];

		problem.AddResidualBlock(
			new PoseGraphCostFunction(edge.pose, edge.infoL), nullptr,
			{ from.data(), to.data() }
		);
	}
	for (Vector6d &p : dataset.vertices) {
		problem.SetManifold(p.data(), manifold);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;

	if (plyfilename) {
		PointCloud pc;
		for (int i = 0; i < dataset.vertices.size(); ++i) {
			Eigen::Vector3d point = Sophus::SE3d::exp(dataset.vertices[i]) * origin;
			pc.push_back(MakePointInCloud(point, color));
		}
		SavePointCloudPLY(pc, plyfilename);
	}
}

#pragma endregion

#pragma region G2O

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d> {

};


#pragma endregion


int main() {
	G2ODataset dataset = G2ODataset::ReadDataset("Data/sphere.g2o.txt");

	//PointCloud pc;
	//Eigen::Vector3d origin = Eigen::Vector3d::Zero();
	//Eigen::Vector3d color(0, 0, 250);
	//for (int i = 0; i < dataset.vertices.size(); ++i) {
	//	Eigen::Vector3d point = Sophus::SE3d::exp(dataset.vertices[i]) * origin;
	//	pc.push_back(MakePointInCloud(point, color));
	//}
	//SavePointCloudPLY(pc, "initial_geo.ply");

	CeresSolver(dataset, "final_geo_ceres.ply");
	return 0;
}