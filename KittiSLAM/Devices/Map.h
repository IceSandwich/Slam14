#pragma once

#include "common.h"
#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>

class Map {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
public:
	typedef std::shared_ptr<Map> Ptr;
	typedef std::unordered_map< unsigned long, std::shared_ptr<MapPoint> > LandmarksType;
	typedef std::unordered_map< unsigned long, std::shared_ptr<Frame> > KeyframesType;

	void InsertMapPoint(std::shared_ptr<MapPoint> mappoint);

private:
	LandmarksType m_landmarks;
	std::shared_ptr<Frame> m_currentFrame = nullptr;
	int m_numActiveKeyframes = 7;
};
