#pragma once
#include "common.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <memory>

class Feature {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
public:
	typedef std::shared_ptr<Feature> Ptr;

	Feature() = delete;
	Feature(const Feature &) = delete;
	Feature(Feature &&) = delete;
	Feature &operator=(Feature &) = delete;
	Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, bool isOnLeft = false) : m_frame{ frame }, m_position{ kp }, m_isOnLeftImage{ isOnLeft } {

	}

	inline const cv::KeyPoint &GetPosition() const {
		return m_position;
	}
	inline std::weak_ptr<MapPoint> &GetMappoint() {
		return m_mapPoint;
	}

private:
	cv::KeyPoint m_position;
	std::weak_ptr<Frame> m_frame;
	std::weak_ptr<MapPoint> m_mapPoint;

	bool m_isOutliner = false;
	bool m_isOnLeftImage = true;
};
