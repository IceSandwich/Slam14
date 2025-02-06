/*****************************************************************//**
 * \file    CameraUtils.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;

// 图像坐标系到相机坐标系
inline Eigen::Vector2d PixelToCam(const Eigen::Matrix3d &K, double y, double x) {
	return Eigen::Vector2d{
		(x - K(0, 2)) / K(0, 0),
		(y - K(1, 2)) / K(1, 1)
	};
}

// 图像坐标系到相机坐标系
inline Eigen::Vector2d PixelToCam(const Eigen::Matrix3d &K, const cv::Point2f pt) {
	return PixelToCam(K, pt.y, pt.x);
}

// 图像坐标系到三维点
inline Eigen::Vector3d PixelToPoint3D(const Eigen::Matrix3d &K, double y, double x, double depth) {
	Eigen::Vector2d cam = PixelToCam(K, y, x); //先转换到相机坐标系
	return Eigen::Vector3d{
		cam[0] * depth,
		cam[1] * depth,
		depth
	};
}

// 图像坐标系到三维点
inline Eigen::Vector3d PixelToPoint3D(const Eigen::Matrix3d &K, const cv::Point2f pt, double depth) {
	return PixelToPoint3D(K, pt.y, pt.x, depth);
}

// 图像坐标系到三维点
inline Eigen::Vector3d PixelToPoint3D(const Eigen::Matrix3d &K, const Eigen::Vector2d pt, double depth) {
	return PixelToPoint3D(K, pt[1], pt[0], depth);
}

// 三维点投影到图像坐标
inline Eigen::Vector2d Point3DToPixel(const Eigen::Matrix3d &K, const Eigen::Vector3d point) {
	return Eigen::Vector2d(
		K(0, 0) * point[0] / point[2] + K(0, 2),
		K(1, 1) * point[1] / point[2] + K(1, 2)
	);
}

// 获取彩色图像某个像素
inline Eigen::Vector3d GetColor(const cv::Mat bgrImage, int y, int x) {
	Eigen::Vector3d ret;
	ret[0] = bgrImage.data[y * bgrImage.step + x * bgrImage.channels() + 2] / 255.0;   // blue
	ret[1] = bgrImage.data[y * bgrImage.step + x * bgrImage.channels() + 1] / 255.0; // green
	ret[2] = bgrImage.data[y * bgrImage.step + x * bgrImage.channels() + 0] / 255.0; // red
	return ret;
}

inline double GetGrayColorWithInterpolation(const cv::Mat image, Eigen::Vector2d p) {
	uchar *d = &image.data[int(p(1, 0)) * image.step + int(p(0, 0))];
	double xx = p(0, 0) - std::floor(p(0, 0));
	double yy = p(1, 0) - std::floor(p(1, 0));
	return ((1 - xx) * (1 - yy) * double(d[0]) +
		xx * (1 - yy) * double(d[1]) +
		(1 - xx) * yy * double(d[image.step]) +
		xx * yy * double(d[image.step + 1])) / 255.0;
}

inline double GetGrayColor(const cv::Mat image, int x, int y) {
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
	return image.data[y * image.step + x * image.channels() + 1] / 255.0; // green
}

// 获取彩色图像某个像素
inline Eigen::Vector3d GetColor(const cv::Mat bgrImage, Eigen::Vector2d pt) {
	return GetColor(bgrImage, pt[1], pt[0]);
}

// 获取深度图某个值
inline unsigned short GetDepth(const cv::Mat depthImage, int y, int x) {
	if (y < 0 || y >= depthImage.rows || x < 0 || x >= depthImage.cols) return 0;
	return depthImage.at<unsigned short>(y, x);
}

// 获取深度图某个值
inline unsigned short GetDepth(const cv::Mat depthImage, cv::Point2f pt) {
	return GetDepth(depthImage, static_cast<int>(pt.y), static_cast<int>(pt.x));
}

// 获取深度图某个值
inline unsigned short GetDepth(const cv::Mat depthImage, Eigen::Vector2d pt) {
	return GetDepth(depthImage, static_cast<int>(pt[1]), static_cast<int>(pt[0]));
}

inline Matrix26d Jaccobian_Pixel_To_DeltaSE3(const Eigen::Matrix3d& K, Eigen::Vector3d pointAfterTransform) {
	double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
	const double &X = pointAfterTransform[0], &Y = pointAfterTransform[1], &Z = pointAfterTransform[2];
	double invZ = 1.0 / Z, invZ2 = invZ * invZ;
	double XmInvZ = X * invZ, YmInvZ = Y * invZ;

	Matrix26d J;
	J << fx * invZ,         0, -fx * X * invZ2,        -fx * X * Y * invZ2, fx + fx * XmInvZ * XmInvZ, -fx * YmInvZ,
				 0, fy * invZ, -fy * Y * invZ2, -fy - fy * YmInvZ * YmInvZ,        fy * Y * X * invZ2,  fy * XmInvZ;
	return J;
}

// 图像梯度
inline Eigen::Vector2d GetGradient(cv::Mat img, int y, int x) {
	return Eigen::Vector2d {
		0.5 * (GetGrayColor(img, x + 1, y) - GetGrayColor(img, x - 1, y)),
		0.5 * (GetGrayColor(img, x, y + 1) - GetGrayColor(img, x, y - 1))
	};
}

class Intrinsic {
public:
	Intrinsic(float fx, float fy, float cx, float cy) {
		m_Eigen << fx, 0, cx, 0, fy, cy, 0, 0, 1;
		m_OpenCV = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
	}
	const Eigen::Matrix3d& Eigen() const {
		return m_Eigen;
	}
	const cv::Mat& OpenCV() const {
		return m_OpenCV;
	}
private:
	Eigen::Matrix3d m_Eigen;
	cv::Mat m_OpenCV;
};


struct MatchResult {
	cv::detail::ImageFeatures features1, features2;
	std::vector<cv::DMatch> matches;
};

inline MatchResult MatchTwoImages(cv::Mat img1, cv::Mat img2, bool useRansac = false, cv::Ptr<cv::FeatureDetector> featuresFinder = cv::ORB::create(), cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING)) {
	MatchResult ret;

	cv::detail::computeImageFeatures(featuresFinder, img1, ret.features1);
	cv::detail::computeImageFeatures(featuresFinder, img2, ret.features2);

	std::vector<cv::DMatch> matches;
	descriptorMatcher->match(ret.features1.descriptors, ret.features2.descriptors, matches);

	auto min_max = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) {
		return m1.distance < m2.distance;
	});
	std::cout << "Min dist: " << min_max.first->distance << ", Max dist: " << min_max.second->distance << std::endl;

	matches.erase(std::remove_if(matches.begin(), matches.end(), [min_dist = min_max.first->distance](cv::DMatch &m) {
		return m.distance > std::max(2.f * min_dist, 30.0f);
	}), matches.end());

	if (useRansac) {
		std::vector<cv::Point2f> srcPoints, dstPoints;
		for (auto match : matches) {
			cv::KeyPoint &srcKp = ret.features1.keypoints[match.queryIdx];
			cv::KeyPoint &dstKp = ret.features2.keypoints[match.trainIdx];
			srcPoints.emplace_back(srcKp.pt.x, srcKp.pt.y);
			dstPoints.emplace_back(dstKp.pt.x, dstKp.pt.y);
		}
		cv::Mat inliner;
		cv::findHomography(srcPoints, dstPoints, inliner, cv::RANSAC);

		std::vector<cv::DMatch>& good_matches = ret.matches;
		for (int i = 0; i < inliner.rows; ++i) {
			if (inliner.at<uint8_t>(i, 0) == 1) {
				good_matches.push_back(matches[i]);
			}
		}
	} else {
		ret.matches.resize(matches.size());
		std::copy(matches.begin(), matches.end(), ret.matches.begin());
	}

	std::cout << "匹配点数量: " << ret.matches.size() << std::endl;

	return ret;
}
