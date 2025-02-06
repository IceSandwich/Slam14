/*****************************************************************//**
 * \file    ImageUtils.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "ImageUtils.h"

Matrix26d ImageUtils::Jaccobian_Pixel_To_DeltaSE3(const Eigen::Matrix3d &K, Eigen::Vector3d pointAfterTransform) {
	double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
	const double &X = pointAfterTransform[0], &Y = pointAfterTransform[1], &Z = pointAfterTransform[2];
	double invZ = 1.0 / Z, invZ2 = invZ * invZ;
	double XmInvZ = X * invZ, YmInvZ = Y * invZ;

	Matrix26d J;
	J << fx * invZ, 0, -fx * X * invZ2, -fx * X * Y * invZ2, fx + fx * XmInvZ * XmInvZ, -fx * YmInvZ,
		0, fy *invZ, -fy * Y * invZ2, -fy - fy * YmInvZ * YmInvZ, fy *Y *X *invZ2, fy *XmInvZ;
	return J;
}
