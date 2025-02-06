/*****************************************************************//**
 * \file    Frontend.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once

#include "../Devices/common.h"
#include "../Devices/Camera.h"
#include <sophus/se3.hpp>
#include <algorithm>
#include <numeric>

static std::vector<cv::Point2f> ConvertKeypointToPoint(const std::vector<cv::KeyPoint> &kps) {
	std::vector<cv::Point2f> ret(kps.size());
	std::transform(kps.begin(), kps.end(), ret.begin(), [](const cv::KeyPoint &kp) {
		return cv::Point2f{ kp.pt.x, kp.pt.y };
		});
	return ret;
}

class Frontend {
public:
	Frontend() = default;

	void AddFrame(const std::shared_ptr<Frame> frame);

private:
	enum class FrontendState {
		Init,
		TrackingGood,
		TrackingBad,
		Lost
	};

	std::shared_ptr<Frame> m_currentFrame = nullptr;
	std::shared_ptr<Frame> m_lastFrame = nullptr;
	std::shared_ptr<Camera> m_cameraLeft = nullptr;
	std::shared_ptr<Camera> m_cameraRight = nullptr;

	std::shared_ptr<Map> m_map = nullptr;
	std::shared_ptr<Backend> m_backend = nullptr;
	std::shared_ptr<Viewer> m_viewer = nullptr;

	Sophus::SE3d m_relativePose;

	int m_trackingInliers = 0;
	int m_numFeatures = 200;
	int m_numFeaturesInit = 100;
	int m_numFeaturesTracking = 50;
	int m_numFeaturesTrackingBad = 20;
	int m_numFeaturesNeededForKeyframe = 80;

	cv::Ptr<cv::GFTTDetector> m_featureDetector = cv::GFTTDetector::create();
	FrontendState m_currentState = FrontendState::Init;

	bool steroInit();

	void track();

	void reset() {

	}

	int detectFeatures();
	int findFeaturesInRight();
	bool buildInitMap();

};
