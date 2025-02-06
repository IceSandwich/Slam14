#include "Map.h"
#include "MapPoint.h"

void Map::InsertMapPoint(std::shared_ptr<MapPoint> mappoint) {
	if (m_landmarks.find(mappoint->GetId()) == m_landmarks.end()) {
		m_landmarks.insert(std::make_pair(mappoint->GetId(), mappoint));
		
	}
}
