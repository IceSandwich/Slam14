/*****************************************************************//**
 * \file    G2OTypes.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include "../Devices/Camera.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <sophus/se3.hpp>

class VertexSE3Pose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	// 通过 BaseVertex 继承
	virtual bool read(std::istream &is) override;

	virtual bool write(std::ostream &os) const override;

	virtual void oplusImpl(const double *v) override;

	virtual void setToOriginImpl() override;

};

class Edge2DTo2DProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE3Pose> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	Edge2DTo2DProjection(const Camera &camera) : m_camera{camera} {

	}


private:
	const Camera &m_camera;

	// 通过 BaseUnaryEdge 继承
	virtual void computeError() override;
	virtual bool read(std::istream &is) override;
	virtual bool write(std::ostream &os) const override;
};

class EdgeProjectionPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE3Pose> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	EdgeProjectionPose(const Eigen::Vector3d &point, const Camera &camera) : m_point{point}, m_camera{camera} {

	}

	// 通过 BaseUnaryEdge 继承
	virtual void computeError() override;
	virtual bool read(std::istream &is) override;
	virtual bool write(std::ostream &os) const override;
	virtual void linearizeOplus() override;

private:
	const Eigen::Vector3d m_point;
	const Camera &m_camera;
};
