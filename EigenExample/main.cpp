/*****************************************************************//**
 * \file    main.cpp
 * \brief   Eigen Example
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "GuardTimer.h"
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>


void base() {
	Eigen::Matrix3d matrix_3x3 = Eigen::Matrix3d::Random();
	Eigen::Vector3d x_3d = Eigen::Vector3d::Zero();
	std::cout << "matrix 3x3: " << matrix_3x3 << std::endl;

	Eigen::Vector3d x_offset = Eigen::Vector3d::Ones();
	x_3d = x_3d + x_offset;
	std::cout << "x_3d^T: " << x_3d.transpose() << std::endl;

	Eigen::Matrix<double, 3, 1> xn_3d;
	xn_3d << 3, 2, 1;
	std::cout << "xn_3d^T: " << xn_3d.transpose() << std::endl;

	// Solve Ax=b using inverse.
	{
		GuardTimer timer{ "inverse solver" };
		Eigen::Matrix3d matrix_3x3_inverse = matrix_3x3.inverse();
		Eigen::Vector3d x = matrix_3x3_inverse * xn_3d;
		std::cout << "^T x^T: " << x.transpose() << std::endl;
	}

	// Solve Ax=b using QR
	{
		GuardTimer timer{ "QR solver" };
		Eigen::Vector3d x = matrix_3x3.colPivHouseholderQr().solve(xn_3d);
		std::cout << "QR x^T: " << x.transpose() << std::endl;
	}

	{
		GuardTimer timer{ "cholesky" };
		Eigen::Vector3d x = matrix_3x3.ldlt().solve(xn_3d);
		std::cout << "cholesky x^T: " << x.transpose() << std::endl;
	}
}

void geometry() {
	Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
	Eigen::AngleAxisd rotation_vector{ M_PI / 4, Eigen::Vector3d{0, 0, 1} };
	std::cout << "rotation matrix: " << rotation_vector.matrix() << std::endl;
	rotation_matrix = rotation_vector.toRotationMatrix();
	std::cout << "rotation matrix: " << rotation_matrix << std::endl;

	Eigen::Vector3d v{ 1, 0, 0 };
	Eigen::Vector3d rotated = rotation_vector * v;
	std::cout << "rotation after vec: " << rotated.transpose() << std::endl;
	rotated = rotation_matrix * v;
	std::cout << "rotation after vec: " << rotated.transpose() << std::endl;

	Eigen::Vector3d euler_angles = rotation_matrix.canonicalEulerAngles(2, 1, 0); // Z, Y, X
	std::cout << "yaw pitch roll: " << euler_angles.transpose() << std::endl;

	Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); // Rigid transforms recorder, 4x4
	T.rotate(rotation_vector);
	T.pretranslate(Eigen::Vector3d{ 1, 3, 4 }); // perform Rx + t matrix.
	std::cout << "transform matrix: " << T.matrix() << std::endl;

	Eigen::Vector3d transform = T * v;
	std::cout << "transform: " << transform.transpose() << std::endl;

	Eigen::Quaterniond q = Eigen::Quaterniond{ rotation_vector };
	std::cout << "quaternion form rotation vector = " << q.coeffs().transpose() << std::endl; // x, y, z, w

	q = Eigen::Quaterniond{ rotation_matrix };
	std::cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << std::endl;

	Eigen::Vector3d v_rotated = q * v;
	std::cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << std::endl;

	std::cout << "should be equal to " << (q * Eigen::Quaterniond{ 0, 1, 0, 0 } *q.inverse()).coeffs().transpose() << std::endl;
}

void practice() {
	Eigen::Quaterniond q1{ 0.35, 0.2, 0.3, 0.1 }, q2{ -0.5, 0.4, -0.1, 0.2 };
	Eigen::Vector3d t1{ 0.3, 0.1, 0.1 }, t2{ -0.1, 0.5, 0.3 };
	q1.normalize(); q2.normalize();

	Eigen::Isometry3d TR1W = Eigen::Isometry3d::Identity();
	TR1W.rotate(q1);
	TR1W.pretranslate(t1);
	Eigen::Isometry3d TWR1 = TR1W.inverse();

	Eigen::Isometry3d TR2W = Eigen::Isometry3d::Identity();
	TR2W.rotate(q2);
	TR2W.pretranslate(t2);

	Eigen::Vector3d pR1{ 0.5, 0, 0.2 };
	Eigen::Vector3d pR2 = TR2W * TWR1 * pR1;
	std::cout << pR2.transpose() << std::endl;
}

void extract3x3andmakeidentify() {
	static constexpr int size = 20;
	Eigen::Matrix<double, size, size> matrix = Eigen::Matrix<double, size, size>::Random();
	std::cout << "matrix: " << matrix << std::endl;

	Eigen::Matrix<double, 3, 3> extract;
	extract << 
		matrix(0, 0), matrix(0, 1), matrix(0, 2),
		matrix(1, 0), matrix(1, 1), matrix(1, 2),
		matrix(2, 0), matrix(2, 1), matrix(2, 2);
	std::cout << "extract: " << extract << std::endl;

	matrix(0, 0) = matrix(0, 1) = matrix(0, 2) =
		matrix(1, 0) = matrix(1, 1) = matrix(1, 2) =
		matrix(2, 0) = matrix(2, 1) = matrix(2, 2) = 1.0;
	std::cout << "matrix: " << matrix << std::endl;

}

void sophus() {
	Eigen::AngleAxis rotateAxis = Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d{ 0, 0, 1 });
	Eigen::Matrix3d R = rotateAxis.toRotationMatrix();
	Eigen::Quaterniond q{ R };
	Sophus::SO3d SO3_R{ R };
	Sophus::SO3d SO3_q{ q };
	std::cout << "SO(3) from matrix: " << SO3_R.matrix() << std::endl;
	std::cout << "SO(3) from quaternion: " << SO3_q.matrix() << std::endl;
	
	Eigen::Vector3d so3 = SO3_R.log(); // \epsilon = \vee( \ln( R ) )
	std::cout << "so3 = " << so3.transpose() << std::endl;
	std::cout << "recover R = " << Sophus::SO3d::exp(so3).matrix() << std::endl;
	Eigen::Matrix3d hatso3 = Sophus::SO3d::hat(so3); // 向量到反对称矩阵 a^
	std::cout << "so3 hat = " << hatso3 << std::endl;
	std::cout << "so3 hat vee = " << Sophus::SO3d::vee(hatso3).transpose() << std::endl; // 反对称矩阵到向量

	Eigen::Vector3d update_so3{ 1e-4, 0, 0 }; // delta
	Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
	std::cout << "SO3 updated = " << SO3_updated.matrix() << std::endl;

	std::cout << "=======================================================" << std::endl;

	Eigen::Vector3d t{ 1, 0, 0 };
	Sophus::SE3d SE3_Rt(R, t);
	Sophus::SE3d SE3_qt(q, t);
	std::cout << "SE3 from R, t = " << SE3_Rt.matrix() << std::endl;
	std::cout << "SE3 from q, t = " << SE3_qt.matrix() << std::endl;

	typedef Eigen::Matrix<double, 6, 1> Vector6d;
	Vector6d se3 = SE3_Rt.log();
	std::cout << "se3 = " << se3.transpose() << std::endl; // 平移在前，旋转在后

	Eigen::Matrix4d hatse3 = Sophus::SE3d::hat(se3);
	std::cout << "se3 hat = " << hatse3 << std::endl;
	std::cout << "se3 hat vee = " << Sophus::SE3d::vee(hatse3).transpose() << std::endl;

	Vector6d update_se3 = Vector6d::Zero();
	update_se3.setZero();
	update_se3(0, 0) = 1e-4;
	Sophus::SE3d SE3d_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
	std::cout << "SE3 updated = " << SE3d_updated.matrix() << std::endl;
}

int main() {
	//base();
	//geometry();
	//practice();
	//extract3x3andmakeidentify();
	sophus();

	getchar();

	return 0;
}