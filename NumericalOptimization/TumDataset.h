/*****************************************************************//**
 * \file    TumDataset.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include "CameraUtils.h"
#include <opencv2/opencv.hpp>
struct TumDataset {
	cv::Mat img1, img2;
	cv::Mat depth1, depth2;
	Intrinsic intrinsic;
	const int width, height;

	TumDataset() :
		img1{ cv::imread("Data/1.png", cv::ImreadModes::IMREAD_COLOR) },
		img2{ cv::imread("Data/2.png", cv::ImreadModes::IMREAD_COLOR) },
		depth1{ cv::imread("Data/1_depth.png", cv::ImreadModes::IMREAD_UNCHANGED) },
		depth2{ cv::imread("Data/2_depth.png", cv::ImreadModes::IMREAD_UNCHANGED) },
		intrinsic{ 520.9, 521.0, 325.1, 249.7 },
		width{ img1.cols },
		height{ img1.rows } {

	}
};
