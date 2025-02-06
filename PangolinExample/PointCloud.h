/*****************************************************************//**
 * \file    PointCloud.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include <vector>
#include <GL/glew.h> // The order is important
#include <Eigen/Eigen>
#include <fstream>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef std::vector< Vector6d, Eigen::aligned_allocator<Vector6d> > PointCloud;

// 0~2: xyz, 3~5: color
inline void DrawPointCloud(const PointCloud &pointcloud) {
	glPointSize(2);
	glBegin(GL_POINTS);
	for (auto &p : pointcloud) {
		glColor3f(p[3], p[4], p[5]);
		glVertex3d(p[0], p[1], p[2]);
	}
	glEnd();
}

inline void SavePointCloudPLY(const PointCloud &pointcloud, const char *filename) {
	std::ofstream fin(filename, std::ios::out);
	assert(fin.good());

	fin << "ply\nformat ascii 1.0\nelement vertex " << pointcloud.size() << "\nproperty float32 x\nproperty float32 y\nproperty float32 z\nproperty uint8 red\nproperty uint8 green\nproperty uint8 blue\nend_header" << std::endl;
	for (const Vector6d &v : pointcloud) {
		fin << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << std::clamp<int>(v[3] * 255.0, 0, 255) << ' ' << std::clamp<int>(v[4] * 255.0, 0, 255) << ' ' << std::clamp<int>(v[5] * 255.0, 0, 255) << '\n';
	}
	fin.close();
	std::cout << "Save ply to " << filename << std::endl;
}

// 三维点和颜色组合成一个可视化的点
inline Vector6d MakePointInCloud(Eigen::Vector3d point, Eigen::Vector3d color) {
	Vector6d ret;
	ret << point[0], point[1], point[2], color[0], color[1], color[2];
	return ret;
}