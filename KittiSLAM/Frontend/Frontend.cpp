/*****************************************************************//**
 * \file    Frontend.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "Frontend.h"
#include "../Devices/Frame.h"
#include "../Devices/Feature.h"
#include "../Devices/MapPoint.h"
#include "../Algorithm/EstimatePose.h"
#include <glog/logging.h>

void Frontend::AddFrame(const std::shared_ptr<Frame> frame) {
	m_currentFrame = frame;

	switch (m_currentState) {
	case Frontend::FrontendState::Init:
		steroInit();
		break;
	case Frontend::FrontendState::TrackingGood:
	case Frontend::FrontendState::TrackingBad:
		track();
		break;
	case Frontend::FrontendState::Lost:
		reset();
		break;
	}

	m_lastFrame = m_currentFrame;
}

bool Frontend::steroInit() {
	int num_features_left = detectFeatures();
	int num_coor_features = findFeaturesInRight();
	if (num_coor_features < m_numFeaturesInit) {
		return false;
	}

	if (bool build_map_success = buildInitMap(); !build_map_success) {
		return false;
	}

	m_currentState = FrontendState::TrackingGood;
	if (m_viewer) {
		//TODO
	}
	return true;
}

void Frontend::track() {

	std::vector<cv::Point2f> kps_last = ConvertKeypointToPoint(getLastFrameKeypoints());
	std::vector<cv::Point2f> kps_current(kps_last.size());
	std::vector<uchar> kps_status(kps_last.size());
	cv::Mat kps_error;
	// use LK flow
	cv::calcOpticalFlowPyrLK(getLastFrame().left, getCurrentFrame().left, kps_last, kps_current, kps_status, kps_error, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

	size_t num_track_last = std::reduce(kps_status.begin(), kps_status.end(), size_t(0), std::plus<size_t>());
	std::cout << "[INFO][Track][Frame" << m_lastFrameIdx << "->Frame" << m_lastFrameIdx + 1 << "] Find " << num_track_last << " in the last image." << std::endl;
}

int Frontend::detectFeatures() {
	static const cv::Point2f halfBlockSize{ 10, 10 };

	Frame::FeatureVector &featuresLeft = m_currentFrame->GetFeatures<Frame::Direction::Left>();
	cv::Mat mask(m_currentFrame->GetImage<Frame::Direction::Left>().size(), CV_8UC1, 255);
	for (std::shared_ptr<Feature> &feat : featuresLeft) {
		cv::rectangle(mask, feat->GetPosition().pt - halfBlockSize, feat->GetPosition().pt + halfBlockSize, 0, cv::LineTypes::FILLED);
	}
	std::vector<cv::KeyPoint> keypoints;
	m_featureDetector->detect(m_currentFrame->GetImage<Frame::Direction::Left>(), keypoints, mask);

	std::transform(keypoints.begin(), keypoints.end(), std::back_inserter(featuresLeft), [this](const cv::KeyPoint &kp) -> std::shared_ptr<Feature> {
		return std::make_shared<Feature>(m_currentFrame, kp);
		});

	LOG(INFO) << "Detect " << keypoints.size() << " new features";
	return keypoints.size();
}

int Frontend::findFeaturesInRight() {
	std::vector<cv::Point2f> kpt_left, kpt_right;
	for (std::shared_ptr<Feature> &kp : m_currentFrame->GetFeatures<Frame::Direction::Left>()) {
		cv::Point2f currentPointInLeft = kp->GetPosition().pt;
		kpt_left.push_back(currentPointInLeft);
		if (std::shared_ptr<MapPoint> mp = kp->GetMappoint().lock(); mp) {
			Eigen::Vector2d px = m_cameraRight->Point3DToPixel(m_currentFrame->GetCurrentPose() * mp->GetPosition());
			kpt_right.emplace_back(px[0], px[1]);
		} else {
			kpt_right.emplace_back(currentPointInLeft);
		}
	}

	std::vector<uchar> status;
	cv::Mat error;
	cv::calcOpticalFlowPyrLK(m_currentFrame->GetImage<Frame::Direction::Left>(), m_currentFrame->GetImage<Frame::Direction::Right>(), kpt_left, kpt_right, status, error, cv::Size(11, 11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

	int num_good_pts = 0;
	for (int i = 0; i < status.size(); ++i) {
		if (status[i]) {
			cv::KeyPoint kp{ kpt_right[i], 7 };
			std::shared_ptr<Feature> feat{ new Feature(m_currentFrame, kp, false) };

			m_currentFrame->GetFeatures<Frame::Direction::Right>().push_back(feat);

			++num_good_pts;
		} else {
			m_currentFrame->GetFeatures<Frame::Direction::Right>().push_back(nullptr);
		}
	}
	LOG(INFO) << "Find " << num_good_pts << " in the right image.";

	return num_good_pts;
}

bool Frontend::buildInitMap() {
	std::vector<Sophus::SE3d> poses{ m_cameraLeft->GetPose(), m_cameraRight->GetPose() };
	int cnt_init_landmarks = 0;
	for (int i = 0; i < m_currentFrame->GetFeatures<Frame::Direction::Left>().size(); ++i) {
		std::shared_ptr<Feature> &leftFeature = m_currentFrame->GetFeatures<Frame::Direction::Left>()[i];
		std::shared_ptr<Feature> &rightFeature = m_currentFrame->GetFeatures<Frame::Direction::Right>()[i];
		if (rightFeature == nullptr) continue;
		std::vector<Eigen::Vector3d> points{
			m_cameraLeft->PixelToPoint3D(leftFeature->GetPosition().pt, 1),
			m_cameraRight->PixelToPoint3D(rightFeature->GetPosition().pt, 1)
		};
		Eigen::Vector3d pworld = Eigen::Vector3d::Zero();

		if (EstimatePose::Triangulation(poses, points, pworld) && pworld[2] > 0) {
			auto newMapPoint = std::make_shared<MapPoint>();
			newMapPoint->SetPosition(pworld);
			newMapPoint->AddObservation(leftFeature);
			newMapPoint->AddObservation(rightFeature);
			++cnt_init_landmarks;
			m_map->
		}
	}
}
