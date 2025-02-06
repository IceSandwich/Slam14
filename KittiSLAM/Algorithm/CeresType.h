/*****************************************************************//**
 * \file    CeresType.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include "EigenType.h"
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <sophus/se3.hpp>

namespace CeresType {

class SE3Manifold : public ceres::Manifold {
	// ����GaussianNewton������,����� H * dx = -Je, ����, J�ǲв�����ſɱ�, dx�ǲ���Ҳ�������ĵ�Tangent/delta
public:
	SE3Manifold() = default;
	virtual int AmbientSize() const { // x ��ά��
		return 6;
	}
	virtual int TangentSize() const { // dx ��ά��
		return 6;
	}
	virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
		Sophus::SE3d T = Sophus::SE3d::exp(Eigen::Map<const Vector6d>{x});
		Sophus::SE3d T_delta = Sophus::SE3d::exp(Eigen::Map<const Vector6d>{delta});
		Eigen::Map<Vector6d>{x_plus_delta} = (T_delta * T).log();
		return true;
	}
	virtual bool PlusJacobian(const double *x, double *jacobian) const { // d( Plus(x, dx) ) / d( dx ), 6x6ά��
		ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
		return true;
	}

	// unimplement
	virtual bool Minus(const double *y,
		const double *x,
		double *y_minus_x) const {
		return false;
	}

	// unimplement
	virtual bool MinusJacobian(const double *x, double *jacobian) const {
		return false;
	}
};

}