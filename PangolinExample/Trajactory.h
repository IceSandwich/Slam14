/*****************************************************************//**
 * \file    Trajactory.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <string_view>
#include <iostream>
#include <fstream>
#include <GL/glew.h> // The order is important

typedef std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > Trajactory;

template <bool UseTime = true>
Trajactory ReadTrajactory(const std::string_view filename) {
	std::ifstream fin{ filename.data() };
	if (!fin) {
		std::cerr << "Cannot find trajactory file at " << filename << std::endl;
		std::exit(-1);
	}

	Trajactory ret;
	for (double time, tx, ty, tz, qx, qy, qz, qw; !fin.eof();) {
		if constexpr (UseTime) {
			fin >> time;
		}
		fin >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
		ret.emplace_back(Eigen::Quaterniond{ qw, qx, qy, qz });
		ret.back().pretranslate(Eigen::Vector3d{ tx, ty, tz });
	}
	std::cout << "[Info] Read total " << ret.size() << " pose entries from " << filename << std::endl;

	return ret;
}

void DrawTrajectoryAxis(const Trajactory &poses) {
	glLineWidth(2);
	for (int i = 0; i < poses.size(); ++i) {
		static constexpr float AxisLength = 0.1;
		Eigen::Vector3d Ow = poses[i].translation();
		Eigen::Vector3d Xw = poses[i] * (AxisLength * Eigen::Vector3d{ 1, 0, 0 });
		Eigen::Vector3d Yw = poses[i] * (AxisLength * Eigen::Vector3d{ 0, 1, 0 });
		Eigen::Vector3d Zw = poses[i] * (AxisLength * Eigen::Vector3d{ 0, 0, 1 });

		glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3d(Ow[0], Ow[1], Ow[2]);
		glVertex3d(Xw[0], Xw[1], Xw[2]);
		glColor3f(0.0, 1.0, 0.0);
		glVertex3d(Ow[0], Ow[1], Ow[2]);
		glVertex3d(Yw[0], Yw[1], Yw[2]);
		glColor3f(0.0, 0.0, 1.0);
		glVertex3d(Ow[0], Ow[1], Ow[2]);
		glVertex3d(Zw[0], Zw[1], Zw[2]);
		glEnd();
	}
}

void DrawTrajectory(const Trajactory &poses, float line_r = 0, float line_g = 0, float line_b = 0) {
	for (int i = 0; i < poses.size() - 1; ++i) {
		glColor3f(line_r, line_g, line_b);
		glBegin(GL_LINES);
		Eigen::Vector3d p1t = poses[i].translation(), p2t = poses[static_cast<size_t>(i + 1)].translation();
		glVertex3d(p1t[0], p1t[1], p1t[2]);
		glVertex3d(p2t[0], p2t[1], p2t[2]);
		glEnd();
	}
}