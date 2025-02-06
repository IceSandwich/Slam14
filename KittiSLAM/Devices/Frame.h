/*****************************************************************//**
 * \file    Frame.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once

#include "common.h"
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <vector>

class Frame {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
public:
	typedef std::shared_ptr<Frame> Ptr;
	typedef std::vector< std::shared_ptr<Feature> > FeatureVector;
	enum class Direction {
		Left,
		Right
	};

	template <Direction D>
	inline cv::Mat &GetImage() {
		if constexpr (D == Direction::Left) {
			return m_imageLeft;
		} else {
			return m_imageRight;
		}
	}

	template <Direction D>
	inline FeatureVector &GetFeatures() {
		if constexpr (D == Direction::Left) {
			return m_featureLeft;
		} else {
			return m_featureRight;
		}
	}

	inline Sophus::SE3d GetCurrentPose() {
		return m_pose;
	}

private:
	unsigned long m_id = 0;
	unsigned long m_keyframeId = 0;
	bool m_isKeyFrame = false;
	Sophus::SE3d m_pose; //Tcw
	std::mutex m_poseMutex;
	cv::Mat m_imageLeft;
	cv::Mat m_imageRight;

	FeatureVector m_featureLeft;
	FeatureVector m_featureRight;
};
