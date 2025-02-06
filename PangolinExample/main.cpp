/*****************************************************************//**
 * \file    main.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "PangolinUtils.h"
#include "Trajactory.h"
#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

void trajactoryMain() {
	Trajactory poses = ReadTrajactory("Data/trajectory.txt");
	WindowMainLoop([&poses]() {
		DrawTrajectoryAxis(poses);
		DrawTrajectory(poses);
	});
}

void trajactoryError() {
	Trajactory estimated = ReadTrajactory("Data/estimated.txt");
	Trajactory groundtruth = ReadTrajactory("Data/groundtruth.txt");

	double rmse = 0;
	for (int i = 0; i < estimated.size(); ++i) {
		Eigen::Quaterniond qe{ estimated[i].rotation() }, qg{ groundtruth[i].rotation() };
		Eigen::Vector3d te{ estimated[i].translation() }, tg{ groundtruth[i].translation() };

		Sophus::SE3d p1{ qe, te }, p2{ qg, tg }; //Isometry3D = SE3d
		double error = (p2.inverse() * p1).log().norm();
		rmse += error * error;
	}

	rmse = std::sqrt( rmse / double(estimated.size()) );

	std::cout << "RMSE = " << rmse << std::endl;

	WindowMainLoop([&]() {
		DrawTrajectory(estimated, 0, 0, 1);
		DrawTrajectory(groundtruth, 1, 0, 0);
	});
}

#include <opencv2/opencv.hpp>
#include "PointCloud.h"
void stero() {
	cv::Mat left = cv::imread("Data/left.png", cv::ImreadModes::IMREAD_GRAYSCALE);
	cv::Mat right = cv::imread("Data/right.png", cv::ImreadModes::IMREAD_GRAYSCALE);

	// 内参
	double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
	// 基线
	double b = 0.573;

	auto sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
	cv::Mat disparity_sgbm, disparity;
	sgbm->compute(left, right, disparity_sgbm);
	disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

	PointCloud pointcloud;
	for (int v = 0; v < left.rows; ++v) {
		for (int u = 0; u < left.cols; ++u) {
			float disparityValue = disparity.at<float>(v, u);
			if (disparityValue <= 10.0 || disparityValue >= 96.0) continue;

			double x = (u - cx) / fx;
			double y = (u - cy) / fy;
			double depth = fx * b / disparityValue;
			double color = left.at<uchar>(v, u) / 255.0;

			pointcloud.emplace_back(x * depth, y * depth, depth, color, color, color);
		}
	}

	cv::imshow("left", left);
	cv::imshow("right", right);
	cv::imshow("disparity", disparity / 96.0);
	WindowMainLoop([&]() {
		DrawPointCloud(pointcloud);
	});
	cv::destroyAllWindows();
}

void jointMap() {
	std::vector<cv::Mat> colorImgs, depthImgs;
	for (int i = 0; i < 5; ++i) {
		colorImgs.push_back(cv::imread(std::string{ "Data/color/" } + std::to_string(i+1) + ".png", cv::ImreadModes::IMREAD_COLOR));
		depthImgs.push_back(cv::imread(std::string{ "Data/depth/" } + std::to_string(i+1) + ".pgm", cv::ImreadModes::IMREAD_UNCHANGED));
	}

	Trajactory poses = ReadTrajactory<false>("Data/pose.txt");

	// 计算点云并拼接
	// 相机内参 
	double cx = 325.5, cy = 253.5;
	double fx = 518.0, fy = 519.0;
	double depthScale = 1000.0;
	PointCloud pointcloud;
	pointcloud.reserve(1000000);

	for (int i = 0; i < 5; i++) {
		std::cout << "转换图像中: " << i + 1 << std::endl;
		cv::Mat color = colorImgs[i];
		cv::Mat depth = depthImgs[i];
		Eigen::Quaterniond rotation{ poses[i].rotation() };
		Sophus::SE3d T{ rotation, poses[i].translation()};
		static constexpr int skip = 2;
		for (int v = 0; v < color.rows; v+=skip)
			for (int u = 0; u < color.cols; u+=skip) {
				unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
				if (d == 0) continue; // 为0表示没有测量到
				Eigen::Vector3d point;
				point[2] = double(d) / depthScale;
				point[0] = (u - cx) * point[2] / fx;
				point[1] = (v - cy) * point[2] / fy;
				Eigen::Vector3d pointWorld = T * point;

				Vector6d p;
				p.head<3>() = pointWorld;
				p[5] = color.data[v * color.step + u * color.channels()] / 255.0;   // blue
				p[4] = color.data[v * color.step + u * color.channels() + 1] / 255.0; // green
				p[3] = color.data[v * color.step + u * color.channels() + 2] / 255.0; // red
				pointcloud.push_back(p);
			}
	}

	std::cout << "点云共有" << pointcloud.size() << "个点." << std::endl;
	WindowMainLoop([&]() {
		DrawPointCloud(pointcloud);
	});
}

int main() {
	//trajactoryMain();
	//trajactoryError();
	//stero();
	jointMap();

	return 0;
}
