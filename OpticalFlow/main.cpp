/*****************************************************************//**
 * \file    main.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "../NumericalOptimization/CameraUtils.h"
#include "../EigenExample/GuardTimer.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <sophus/se3.hpp>

#pragma region OpticalFlow
class OpticalFlowTracker {
public:
	OpticalFlowTracker(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse = true, bool has_initial = false) :
		img1{img1}, img2{img2}, kp1{kp1}, kp2{kp2}, success{success}, inverse{inverse}, has_initial{has_initial}
	{

	}

	void CalculateOpticalFlow(const cv::Range &range) {
		int half_patch_size = 4;
		int iterations = 10;
		for (int i = range.start; i < range.end; ++i) {
			auto kp = kp1[i];
			double dx = 0, dy = 0;
			if (has_initial) {
				dx = kp2[i].pt.x - kp.pt.x;
				dy = kp2[i].pt.y - kp.pt.y;
			}

			double cost = 0, lastCost = 0;
			bool succ = true;

			Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
			Eigen::Vector2d b = Eigen::Vector2d::Zero();
			Eigen::Vector2d J;
			
			for (int iter = 0; iter < iterations; ++iter) {
				if (inverse == false) {
					H = Eigen::Matrix2d::Zero();
					b = Eigen::Vector2d::Zero();
				} else {
					b = Eigen::Vector2d::Zero();
				}

				cost = 0;
				for (int x = -half_patch_size; x < half_patch_size; ++x) {
					for (int y = -half_patch_size; y < half_patch_size; ++y) {
						double error = GetGrayColor(img1, kp.pt.x + x, kp.pt.y + y) - GetGrayColor(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
						if (inverse == false) {
							J = -GetGradient(img2, kp.pt.y + dy + y, kp.pt.x + dx + x);
							//J = -1.0 * Eigen::Vector2d{ //图像梯度
							//	0.5 * (GetGrayColor(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) - GetGrayColor(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
							//	0.5 * (GetGrayColor(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) - GetGrayColor(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
							//};
						} else if (iter == 0) {
							J = -GetGradient(img1, kp.pt.y + y, kp.pt.x + x);
							//J = -1.0 * Eigen::Vector2d{
							//	0.5 * (GetGrayColor(img1, kp.pt.x + x + 1, kp.pt.y + y) - GetGrayColor(img1, kp.pt.x + x - 1, kp.pt.y + y)),
							//	0.5 * (GetGrayColor(img1, kp.pt.x + x, kp.pt.y + y + 1) - GetGrayColor(img1, kp.pt.x + x, kp.pt.y + y - 1))
							//};
						}

						b += -error * J;
						cost += error * error;
						if (inverse == false || iter == 0) {
							H += J * J.transpose();
						}
					}
				}

				Eigen::Vector2d update = H.ldlt().solve(b);
				if (std::isnan(update[0])) {
					std::cout << "update is nan" << std::endl;
					succ = false;
					break;
				}

				if (iter > 0 && cost > lastCost) {
					break;
				}

				dx += update[0];
				dy += update[1];
				lastCost = cost;
				succ = true;

				if (update.norm() < 1e-2) {
					break;
				}
			}

			success[i] = succ;
			kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
		}
	}
private:
	const cv::Mat &img1, &img2;
	const std::vector<cv::KeyPoint> &kp1;
	std::vector<cv::KeyPoint> &kp2;
	std::vector<bool> &success;
	bool inverse;
	bool has_initial;
};

void OpticalFlowSingleLevel(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse = false, bool has_initial = false) {
	kp2.resize(kp1.size());
	success.resize(kp1.size());
	OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
	cv::parallel_for_(cv::Range(0, kp1.size()), std::bind(&OpticalFlowTracker::CalculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowMultiLevel(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success, bool inverse = false) {
	int pyramids = 4;
	double pyramid_scale = 0.5;
	double scales[] = { 1.0, 0.5, 0.25, 0.125 };

	std::vector<cv::Mat> pyr1, pyr2;
	for (int i = 0; i < pyramids; ++i) {
		if (i == 0) {
			pyr1.push_back(img1);
			pyr2.push_back(img2);
		} else {
			cv::Mat img1_pyr, img2_pyr;
			cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
			cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
			pyr1.push_back(img1_pyr);
			pyr2.push_back(img2_pyr);
		}
	}

	std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
	for (auto &kp : kp1) {
		auto kp_top = kp;
		kp_top.pt *= scales[pyramids - 1];
		kp1_pyr.push_back(kp_top);
		kp2_pyr.push_back(kp_top);
	}

	for (int level = pyramids - 1; level >= 0; --level) {
		success.clear();

		OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);

		if (level > 0) {
			for (auto &kp : kp1_pyr) {
				kp.pt /= pyramid_scale;
			}
			for (auto &kp : kp2_pyr) {
				kp.pt /= pyramid_scale;
			}
		}
	}

	for (auto &kp: kp2_pyr) {
		kp2.push_back(kp);
	}
}

void OpticalFlow() {
	cv::Mat img1 = cv::imread("Data/LK1.png", cv::ImreadModes::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("Data/LK2.png", cv::ImreadModes::IMREAD_GRAYSCALE);

	std::vector<cv::KeyPoint> kp1;
	cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
	detector->detect(img1, kp1);

	std::vector<cv::KeyPoint> kp2_single;
	std::vector<bool> success_single;
	{
		GuardTimer timer{ "Optical Single" };
		OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
	}

	std::vector<cv::KeyPoint> kp2_multi;
	std::vector<bool> success_multi;
	{
		GuardTimer timer{ "Optical Multi" };
		OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
	}

	// use opencv's flow for validation
	std::vector<cv::Point2f> pt1, pt2;
	for (auto &kp : kp1) pt1.push_back(kp.pt);
	std::vector<uchar> status;
	std::vector<float> error;
	{
		GuardTimer timer{ "Opencv opti flow" };
		cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
	}

	cv::Mat img2_single;
	cv::cvtColor(img2, img2_single, cv::ColorConversionCodes::COLOR_GRAY2BGR);
	for (int i = 0; i < kp2_single.size(); i++) {
		if (success_single[i]) {
			cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
			cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
		}
	}

	cv::Mat img2_multi;
	cv::cvtColor(img2, img2_multi, cv::ColorConversionCodes::COLOR_GRAY2BGR);
	for (int i = 0; i < kp2_multi.size(); i++) {
		if (success_multi[i]) {
			cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
			cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
		}
	}

	cv::Mat img2_CV;
	cv::cvtColor(img2, img2_CV, cv::ColorConversionCodes::COLOR_GRAY2BGR);
	for (int i = 0; i < pt2.size(); i++) {
		if (status[i]) {
			cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
			cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
		}
	}

	cv::imshow("tracked single level", img2_single);
	cv::imshow("tracked multi level", img2_multi);
	cv::imshow("tracked by opencv", img2_CV);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

#pragma endregion

class DirectMethodProcessor {
public:
	DirectMethodProcessor(cv::Mat img, const std::vector<cv::Point> &kps, const std::vector<double> &depths, const Eigen::Matrix3d& K, const size_t border = 1) :
		m_img{ img }, m_keypoints{ kps.begin(), kps.end() }, m_depths{ depths.begin(), depths.end() }, m_K{K}, m_halfPatchSize{static_cast<int>(border)}
	{
		assert(m_keypoints.size() == m_depths.size());
	}

	void TrackNextImage(cv::Mat img, Sophus::SE3d& pose, int maxIteration = 10) {
		double error = std::numeric_limits<double>::infinity();
		for (int iter = 0; iter < maxIteration; ++iter) {
			Matrix6d H = Matrix6d::Zero();
			Vector6d b = Vector6d::Zero();

			error = 0;               
			for (int i = 0; i < m_keypoints.size(); ++i) {
				cv::Point &kp = m_keypoints[i];
				if (kp.x < 0 || kp.y < 0 || kp.x >= img.cols || kp.y >= img.rows) continue;
				
				Eigen::Vector3d pReference = PixelToPoint3D(m_K, kp, m_depths[i]);
				Eigen::Vector3d pCurrent = pose * pReference;
				if (pCurrent[2] < 0) continue; // depth invalid

				Eigen::Vector2d reproject = Point3DToPixel(m_K, pCurrent);
				if (reproject[0] < m_halfPatchSize || reproject[0] > img.cols - m_halfPatchSize || reproject[1] < m_halfPatchSize || reproject[1] > img.rows - m_halfPatchSize) continue;

				Matrix26d J_pixel_xi = Jaccobian_Pixel_To_DeltaSE3(m_K, pCurrent);

				for (int dy = -m_halfPatchSize; dy <= m_halfPatchSize; ++dy) {
					for (int dx = -m_halfPatchSize; dx <= m_halfPatchSize; ++dx) {
						// I2(T)-I1
						double e = GetGrayColor(m_img, kp.x + dx, kp.y + dy) - GetGrayColor(img, reproject[0] + dx, reproject[1] + dy);

						//Eigen::Matrix<double, 1, 6> J = GetGradient(img, reproject[1] + dy, reproject[0] + dx).transpose() * J_pixel_xi;
						Vector6d J = -1.0 * (GetGradient(img, reproject[1] + dy, reproject[0] + dx).transpose() * J_pixel_xi).transpose();

						H += J * J.transpose();
						b += -e * J;

						error += e * e;
					}
				}
			}

			Vector6d delta = H.ldlt().solve(b);
			if (std::isnan(delta[0])) {
				std::cerr << "delta is nan!!!" << std::endl;
				break;
			}
			if (delta.norm() < std::numeric_limits<double>::epsilon()) { // coverage
				break;
			}

			pose = Sophus::SE3d::exp(delta) * pose;
			std::cout << "Iter " << iter << ", error: " << error << std::endl;
		}

		std::cout << "Last error: " << error << std::endl;
		std::cout << "Transform: " << pose.matrix() << std::endl;

		// For tracking continually, replace previous state to current state.
		for (int i = 0; i < m_keypoints.size(); ++i) {
			Eigen::Vector3d p3D = pose * PixelToPoint3D(m_K, m_keypoints[i], m_depths[i]);
			Eigen::Vector2d reproject = Point3DToPixel(m_K, p3D);

			m_keypoints[i].x = reproject[0];
			m_keypoints[i].y = reproject[1];
			m_depths[i] = p3D[2];
		}
		m_img = img;
	}
private:
	cv::Mat m_img;
	std::vector<cv::Point> m_keypoints;
	std::vector<double> m_depths;
	const Eigen::Matrix3d m_K;
	const int m_halfPatchSize;
};

class DirectMethodMultiLevelProcessor {
public:
	DirectMethodMultiLevelProcessor(cv::Mat img, const std::vector<cv::Point> &kps, const std::vector<double>& depths, Eigen::Matrix3d K, const size_t border = 1) {
		cv::Mat lastImg = img;
		Eigen::Matrix3d lastK = K;
		std::vector<cv::Point> lastKps{ kps.begin(), kps.end() };
		m_processors[0] = std::make_unique<DirectMethodProcessor>(lastImg, lastKps, depths, lastK, border);

		for (int i = 1; i < pyramids; ++i) {
			cv::Mat resizedImg;
			cv::resize(lastImg, resizedImg, cv::Size(lastImg.cols * pyramid_scale, lastImg.rows * pyramid_scale));
			lastK(0, 0) *= pyramid_scale;
			lastK(1, 1) *= pyramid_scale;
			lastK(0, 2) *= pyramid_scale;
			lastK(1, 2) *= pyramid_scale;

			for (cv::Point &p : lastKps) {
				p = p * pyramid_scale;
			}

			m_processors[i] = std::make_unique<DirectMethodProcessor>(resizedImg, lastKps, depths, lastK, border);

			lastImg = resizedImg;
		}
	}

	void TrackNextImage(cv::Mat img, Sophus::SE3d &pose, int eachInteration = 10) {
		std::array<cv::Mat, pyramids> pyramidImgs;
		pyramidImgs[0] = img;
		for (int i = 1; i < pyramids; ++i) {
			cv::Mat resized;
			cv::resize(pyramidImgs[i-1], resized, cv::Size(pyramidImgs[i - 1].cols * pyramid_scale, pyramidImgs[i - 1].rows * pyramid_scale));
			pyramidImgs[i] = resized;
		}

		for (int i = pyramids - 1; i >= 0; --i) {
			m_processors[i]->TrackNextImage(pyramidImgs[i], pose, eachInteration);
		}
	}
private:
	static constexpr int pyramids = 4;
	const double pyramid_scale = 0.5;
	//const std::array<double, 4>scales{1.0, 0.5, 0.25, 0.125};

	std::array< std::unique_ptr<DirectMethodProcessor>, pyramids> m_processors;

	//std::vector<cv::Point> &m_keypoints;
	//const Eigen::Matrix3d &m_K;
	//const int m_halfPatchSize;
};

cv::Mat DrawTracks(int currentIndex, cv::Mat currentImage, const std::vector<Sophus::SE3d> &poses, std::vector<cv::Point> keypoints, std::vector<double> depths) {
	Eigen::Vector3d currentColor = Eigen::Vector3d{ 0, 0, 255 };
	Eigen::Vector3d deltaColor = (Eigen::Vector3d{ 255, 0, 0 } - currentColor) / poses.size(); //RGB

	cv::Mat ret;
	cv::cvtColor(currentImage, ret, cv::ColorConversionCodes::COLOR_GRAY2BGR);

	Sophus::SE3d currentPose{ Eigen::Matrix4d::Identity() };
	for (int poseIndex = 0; poseIndex <= currentIndex; ++poseIndex) {
		currentPose = poses[poseIndex] * currentPose;
		currentColor += deltaColor;
	}

	for (int i = 0; i < keypoints.size(); ++i) {
		//PixelToPoint3D(K, kp1[i], depths[i]);
	}

	for (int poseIndex = currentIndex; poseIndex < poses.size(); ++poseIndex) {

	}

	return ret;
}

int main() {
	//OpticalFlow();
	//return 0;

	Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
	K(0, 0) = K(1, 1) = 718.856;
	K(0, 2) = 607.1928;
	K(1, 2) = 185.2157;
	K(2, 2) = 1.0;
	const double baseline = 0.573;

	cv::Mat left = cv::imread("Data/left.png", cv::ImreadModes::IMREAD_GRAYSCALE);
	// 视差与深度是反比关系
	cv::Mat disparity_8u = cv::imread("Data/disparity.png", cv::ImreadModes::IMREAD_UNCHANGED);
	//std::vector<cv::KeyPoint> kp1;
	//cv::GFTTDetector::create(500, 0.01, 20)->detect(left, kp1);
	std::vector<cv::Point> kp1;
	std::vector<double> depths;
	// generate pixels in ref and load depth data
	const int border = 15;
	cv::RNG rng;
	for (int i = 0; i < 1000; i++) {
		int x = rng.uniform(border, left.cols - border);  // don't pick pixels close to boarder
		int y = rng.uniform(border, left.rows - border);  // don't pick pixels close to boarder
		kp1.push_back(cv::Point(x, y));

		int disparity = disparity_8u.at<uchar>(y, x);
		double depth = K(0, 0) * baseline / disparity; // you know this is disparity to depth
		depths.push_back(depth);
	}


	typedef DirectMethodMultiLevelProcessor DMProcessor;
	DMProcessor processor{ left, kp1, depths, K };

	std::vector<Sophus::SE3d> poses;
	std::vector<cv::Mat> visualizers;
	cv::cvtColor(left, visualizers.emplace_back(), cv::ColorConversionCodes::COLOR_GRAY2BGR);
	for (int i = 0; i < 5; ++i) {
		char filename[256];
		sprintf_s(filename, "Data/%06d.png", i + 1);

		cv::Mat img = cv::imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);

		Sophus::SE3d pose;
		processor.TrackNextImage(img, pose, 10);
		poses.push_back(pose);
		cv::cvtColor(img, visualizers.emplace_back(), cv::ColorConversionCodes::COLOR_GRAY2BGR);
	}


	for (int poseIndex = 0; poseIndex < poses.size(); ++poseIndex) {

	}


	const size_t NumPoints = kp1.size();
	std::vector< std::vector<cv::Point> > finalPoints;
	finalPoints.reserve(NumPoints);

	for (int i = 0; i < NumPoints; ++i) {
		finalPoints[i].reserve(poses.size() + 1);
		
		finalPoints[i][0] = kp1[0];
	}

	std::vector<double> finalDepths(NumPoints);
	for (int poseIndex = 0; poseIndex < poses.size(); ++poseIndex) {
		for (int i = 0; i < NumPoints; ++i) {
			Eigen::Vector3d p3D = poses[poseIndex] * PixelToPoint3D(K, finalPoints[i][poseIndex], finalDepths[i]);
			Eigen::Vector2d kp = Point3DToPixel(K, p3D);
			cv::Point cvkp{ static_cast<int>(kp[0]), static_cast<int>(kp[1]) };

			finalPoints[i][poseIndex + 1] = cvkp;
			finalDepths[i] = p3D[2];
		}
	}

	for (int i = 0; i < visualizers.size(); ++i) {
		cv::Mat &visualizer = visualizers[i];
		

		for (int j = 0; j < NumPoints; ++j) {
			cv::circle(visualizer, finalPoints[j][i], 2, cv::Scalar(0, 250, 0), 2);
		}
	}

	return 0;
	
	cv::Mat visualizer;
	cv::cvtColor(left, visualizer, cv::ColorConversionCodes::COLOR_GRAY2BGR);
	Eigen::Vector3d currentColor = Eigen::Vector3d{ 0, 0, 255 };
	Eigen::Vector3d deltaColor = (Eigen::Vector3d{ 255, 0, 0 } - currentColor) / poses.size(); //RGB

	//std::vector<cv::Point> kp1_local{ kp1.begin(), kp1.end() };
	//std::vector<double> depth_local{ depths.begin(), depths.end() };
	//std::vector<cv::Point> kp1_last{ kp1.begin(), kp1.end() };

	//Sophus::SE3d accumulatePose{ Eigen::Matrix4d::Identity() };

	for (int poseIndex = 0; poseIndex < poses.size(); ++poseIndex) {
		//accumulatePose = poses[poseIndex] * accumulatePose;

		for (int i = 0; i < kp1.size(); ++i) {
			//if (kp1_last[i].x < 0 || kp1_last[i].y < 0 || kp1_last[i].x >= left.cols || kp1_last[i].y >= left.rows) continue;
			if (kp1[i].x < 0 || kp1[i].y < 0 || kp1[i].x >= left.cols || kp1[i].y >= left.rows) continue;

			//Eigen::Vector3d p3D = poses[poseIndex] * PixelToPoint3D(K, kp1_local[i], depth_local[i]);
			//Eigen::Vector2d kp = Point3DToPixel(K, p3D);

			//Eigen::Vector3d p3DInVeryFirstImageCoord = accumulatePose * PixelToPoint3D(K, kp1[i], depths[i]);
			//Eigen::Vector2d kpInVeryFirstImageCoord = Point3DToPixel(K, p3DInVeryFirstImageCoord);
			//cv::Point cvkp{ static_cast<int>(kpInVeryFirstImageCoord[0]), static_cast<int>(kpInVeryFirstImageCoord[1]) };

			//cv::circle(visualizer, cvkp, 2, cv::Scalar(0, 250, 0), 2);
			//cv::line(visualizer, kp1_last[i], cvkp, cv::Scalar(currentColor[2], currentColor[1], currentColor[0]));

			//kp1_local[i].x = kp[0];
			//kp1_local[i].y = kp[1];
			//depth_local[i] = p3D[2];

			//kp1_last[i] = cvkp;

			//if (kpInVeryFirstImageCoord[0] < 0 || kpInVeryFirstImageCoord[0] >= left.cols || kpInVeryFirstImageCoord[1] < 0 || kpInVeryFirstImageCoord[1] >= left.rows) {
			//	kp1_last[i].x = -1;
			//	kp1_last[i].y = -1;
			//}

			Eigen::Vector3d p3D = poses[poseIndex] * PixelToPoint3D(K, kp1[i], depths[i]);
			Eigen::Vector2d kp = Point3DToPixel(K, p3D);
			cv::Point cvkp{ static_cast<int>(kp[0]), static_cast<int>(kp[1]) };
			cv::line(visualizer, kp1[i], cvkp, cv::Scalar(currentColor[2], currentColor[1], currentColor[0]));

			kp1[i] = cvkp;
			depths[i] = p3D[2];

			if (kp[0] < 0 || kp[0] >= left.cols || kp[1] < 0 || kp[1] >= left.rows) {
				kp1[i].x = -1;
				kp1[i].y = -1;
			}
		}

		currentColor += deltaColor;
	}
	cv::imshow("track", visualizer);
	cv::waitKey();
	cv::destroyAllWindows();

	return 0;
}
