/*****************************************************************//**
 * \file    ImageUtils.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once

#include "EigenType.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

namespace ImageUtils {

enum class Interpolation {
	Nearest,
	Bilinear
};

//若返回值为浮点数，则返回0.0~1.0的范围
template <typename RetType = double, Interpolation Inter = Interpolation::Bilinear>
RetType GetGrayColor(const cv::Mat image, Eigen::Vector2d p) {
	if constexpr (Inter == Interpolation::Nearest) {
		const int x = p[0], y = p[1];
		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x >= image.cols - 1) x = image.cols - 2;
		if (y >= image.rows - 1) y = image.rows - 2;

		if (image.channels() == 1) {
			float xx = x - floor(x);
			float yy = y - floor(y);
			int x_a1 = std::min(image.cols - 1, int(x) + 1);
			int y_a1 = std::min(image.rows - 1, int(y) + 1);

			return (1 - xx) * (1 - yy) * image.at<uchar>(y, x)
				+ xx * (1 - yy) * image.at<uchar>(y, x_a1)
				+ (1 - xx) * yy * image.at<uchar>(y_a1, x)
				+ xx * yy * image.at<uchar>(y_a1, x_a1);
		}
		if constexpr (std::is_floating_point<RetType>::value) {
			return image.data[y * image.step + x * image.channels() + 1] / 255.0; // green
		} else {
			return image.data[y * image.step + x * image.channels() + 1]; // green
		}
	} else if constexpr (Inter == Interpolation::Bilinear) {
		uchar *d = &image.data[int(p(1, 0)) * image.step + int(p(0, 0))];
		double xx = p(0, 0) - std::floor(p(0, 0));
		double yy = p(1, 0) - std::floor(p(1, 0));
		if constexpr (std::is_floating_point<RetType>::value) {
			return ((1 - xx) * (1 - yy) * double(d[0]) +
				xx * (1 - yy) * double(d[1]) +
				(1 - xx) * yy * double(d[image.step]) +
				xx * yy * double(d[image.step + 1])) / 255.0;
		} else {
			return static_cast<RetType>((1 - xx) * (1 - yy) * double(d[0]) +
				xx * (1 - yy) * double(d[1]) +
				(1 - xx) * yy * double(d[image.step]) +
				xx * yy * double(d[image.step + 1]));
		}
	}
	return RetType();
}

inline Eigen::Vector2d GetGradient(cv::Mat img, Eigen::Vector2d p) {
	return Eigen::Vector2d{
		0.5 * (GetGrayColor(img, p + Eigen::Vector2d{1, 0}) - GetGrayColor(img, p + Eigen::Vector2d{-1, 0})),
		0.5 * (GetGrayColor(img, p + Eigen::Vector2d{0, 1}) - GetGrayColor(img, p + Eigen::Vector2d{0, -1}))
	};
}

// 获取深度图某个值
inline unsigned short GetDepth(const cv::Mat depthImage, int x, int y) {
	if (y < 0 || y >= depthImage.rows || x < 0 || x >= depthImage.cols) return 0;
	return depthImage.at<unsigned short>(y, x);
}

// 获取深度图某个值
inline unsigned short GetDepth(const cv::Mat depthImage, cv::Point2f pt) {
	return GetDepth(depthImage, static_cast<int>(pt.x), static_cast<int>(pt.y));
}

// 获取深度图某个值
inline unsigned short GetDepth(const cv::Mat depthImage, Eigen::Vector2d pt) {
	return GetDepth(depthImage, static_cast<int>(pt[0]), static_cast<int>(pt[1]));
}

Matrix26d Jaccobian_Pixel_To_DeltaSE3(const Eigen::Matrix3d &K, Eigen::Vector3d pointAfterTransform);

}
