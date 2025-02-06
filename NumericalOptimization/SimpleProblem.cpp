/*****************************************************************//**
 * \file    SimpleProblem.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "SimpleProblem.h"
#include "../EigenExample/GuardTimer.h"
#include <array>
#include <iostream>
#include <random>

#pragma region Core

template <typename T, size_t N>
class Residual {
protected:
	typedef Eigen::Matrix<T, N, 1> ParametersVector;
	ParametersVector m_parameters;
public:
	virtual T CalculateError(T x, T y) = 0;
	virtual ParametersVector Jacobin(T x) = 0;
	void Update(const ParametersVector &delta) {
		m_parameters += delta;
	}
	const ParametersVector &GetParameters() const {
		return m_parameters;
	}
};

class SimpleProblem : public Residual<double, 3> {
	// y = exp(ax^2 + bx + c) + w
	double &a = m_parameters[0], &b = m_parameters[1], &c = m_parameters[2];
public:
	SimpleProblem() {
		a = 2.0; b = -1.0; c = 5.0; //初始值
	}

	virtual double CalculateError(double x, double y) override {
		return y - func(x);
	}

	virtual ParametersVector Jacobin(double x) override {
		ParametersVector ret;
		ret[2] = -func(x);
		ret[1] = x * ret[2];
		ret[0] = x * ret[1];
		return ret;
	}

private:
	inline double func(double x) {
		return std::exp(a * x * x + b * x + c);
	}
};

template <size_t Count>
std::array<std::pair<double, double>, Count> GenerateSamples(double mean = 0.0, double sigma = 1.0) {
	std::array<std::pair<double, double>, Count> samples;
	std::normal_distribution<double> gaussian{ mean, sigma };
	for (int i = 0; i < samples.size(); ++i) {
		static std::mt19937 engine;

		double x = static_cast<double>(i) / Count;
		samples[i] = std::make_pair(x, std::exp(1.0 * x * x + 2.0 * x + 1.0) + gaussian(engine));
	}
	return samples;
}

#pragma endregion


#pragma region Gaussian Newton

void GaussianNewton() {
	std::array<std::pair<double, double>, 100> samples = GenerateSamples<100>();

	SimpleProblem problem;
	double cost = std::numeric_limits<double>::infinity();
	for (int iter = 0; iter < 100; ++iter) {

		Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
		Eigen::Vector3d b = Eigen::Vector3d::Zero();

		cost = 0.0;

		for (int i = 0; i < samples.size(); ++i) {
			const auto &[realX, realY] = samples[i];

			Eigen::Vector3d J = problem.Jacobin(realX);
			double error = problem.CalculateError(realX, realY);

			H += J * J.transpose();
			b += -J * error;

			cost += error * error;
		}
		std::cout << "at iter: " << iter << " got cost: " << cost << std::endl;

		Eigen::Vector3d deltaX = H.ldlt().solve(b);
		problem.Update(deltaX);
	}
	std::cout << "Final result: " << problem.GetParameters().transpose() << std::endl;
	std::cout << "Last cost: " << cost << std::endl;
}

#pragma endregion

#pragma region Ceres
#include <ceres/ceres.h>

class CostFunctor {
protected:
	const double x, y;
public:
	CostFunctor(double x, double y) : x{ x }, y{ y } {

	}

	//计算残差e
	template <typename T>
	bool operator()(const T *const parameters, T *residual) const { // 3x1, 1x1
		// y = exp(a * x^2 + b * x + c)
		residual[0] = T(y) - ceres::exp(parameters[0] * T(x) * T(x) + parameters[1] * T(x) + parameters[2]);
		return true;
	}
};

void CeresSolver() {
	std::array<std::pair<double, double>, 100> samples = GenerateSamples<100>();

	std::array<double, 3> parameters{ 2.0, -1.0, 5.0 };
	ceres::Problem problem;
	for (int i = 0; i < samples.size(); ++i) {
		const auto &[realX, realY] = samples[i];

		problem.AddResidualBlock(
			// 1: 输出只有一个维度， 3：优化的变量有3个
			new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(
				new CostFunctor(realX, realY),
				ceres::Ownership::TAKE_OWNERSHIP //表示由ceres释放这个new出来的内存
			),
			nullptr,
			parameters.data()
		);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	std::cout << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << std::endl;
}

#pragma endregion

#pragma region G2OExample
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
	virtual void setToOriginImpl() override {
		_estimate << 0, 0, 0;
	}

	virtual void oplusImpl(const double *update) override {
		_estimate += Eigen::Vector3d(update);
	}

	virtual bool read(std::istream &in) { return false; }
	virtual bool write(std::ostream &out) const { return false; }
};


// 导数de/dx，其中e的维度是1，x的维度是CurveFittingVertex::Dimension=3，雅可比矩阵为1x3
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
	CurveFittingEdge(double x) : g2o::BaseUnaryEdge<1, double, CurveFittingVertex>(), _x(x) {

	}

	virtual void computeError() override {
		const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		_error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
	}

	// 雅可比矩阵
	virtual void linearizeOplus() override {
		const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
		const Eigen::Vector3d abc = v->estimate();
		double y = std::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
		_jacobianOplusXi[0] = -_x * _x * y;
		_jacobianOplusXi[1] = -_x * y;
		_jacobianOplusXi[2] = -y;
	}

	virtual bool read(std::istream &in) override { return false; }
	virtual bool write(std::ostream &out) const override { return false; }
protected:
	double _x;
};

void G2O() {
	std::array<std::pair<double, double>, 100> samples = GenerateSamples<100>();

	Eigen::Vector3d parameters{ 2.0, -1.0, 5.0 };
	const double w_sigma = 1.0;

	typedef g2o::BlockSolver< g2o::BlockSolverTraits<3, 1> > BlockSolverType;
	typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

	auto solver = new g2o::OptimizationAlgorithmLevenberg(
		std::make_unique<BlockSolverType>(
			std::make_unique<LinearSolverType>()
		)
	);

	// 优化器
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(true);

	// 顶点
	auto v = new CurveFittingVertex();
	v->setEstimate(parameters); //初始值
	v->setId(0); // 设置id
	optimizer.addVertex(v);

	// 边
	for (int i = 0; i < samples.size(); ++i) {
		CurveFittingEdge *edge = new CurveFittingEdge(samples[i].first); //设置样本输入值
		edge->setId(i); //设置id
		edge->setVertex(0, v); //设置0号顶点
		edge->setMeasurement(samples[i].second); //设置目标值
		edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); //设置信息矩阵
		optimizer.addEdge(edge);
	}

	// 执行优化
	optimizer.initializeOptimization();
	{
		GuardTimer timer{"G2o Optimization"};
		optimizer.optimize(10);
	}

	parameters = v->estimate();
	std::cout << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << std::endl;
}

#pragma endregion

