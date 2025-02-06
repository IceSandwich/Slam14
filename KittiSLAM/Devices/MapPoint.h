#pragma once
#include "common.h"
#include <Eigen/Eigen>
#include <mutex>
#include <list>

class MapPoint {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
private:
	unsigned long m_id;
	bool m_isOutliner = false;
	Eigen::Vector3d m_pos = Eigen::Vector3d::Zero();
	std::mutex m_dataMutex;
	int m_observedTimes = 0;
	std::list< std::weak_ptr<Feature> > m_observations;

	static long m_factoryId;
public:
	typedef std::shared_ptr<MapPoint> Ptr;

	MapPoint() : m_id{ m_factoryId++ } {

	}

	MapPoint(long id, Eigen::Vector3d position) : m_id{ id }, m_pos{ position } {

	}

	inline Eigen::Vector3d &GetPosition() {
		return m_pos;
	}

	inline void SetPosition(const Eigen::Vector3d &pos) {
		std::unique_lock<std::mutex> lck(m_dataMutex);
		m_pos = pos;
	}

	inline void AddObservation(std::shared_ptr<Feature> feature) {
		std::unique_lock<std::mutex> lck(m_dataMutex);
		m_observations.emplace_front(feature);
		++m_observedTimes;
	}

	inline unsigned long GetId() const {
		return m_id;
	}

};
