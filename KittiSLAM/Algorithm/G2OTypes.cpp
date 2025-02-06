/*****************************************************************//**
 * \file    G2OTypes.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "EstimatePose.h"
#include "G2OTypes.h"
#include "EigenType.h"
#include "ImageUtils.h"

bool VertexSE3Pose::read(std::istream &is) {
    return false;
}

bool VertexSE3Pose::write(std::ostream &os) const {
    return false;
}

void VertexSE3Pose::oplusImpl(const double *v) {
    Vector6d delta;
    delta << v[0], v[1], v[2], v[3], v[4], v[5];
    _estimate = Sophus::SE3d::exp(delta) * _estimate;
}

void VertexSE3Pose::setToOriginImpl() {
    _estimate = Sophus::SE3d();
}

void EdgeProjectionPose::computeError() {
    const VertexSE3Pose* v = static_cast<VertexSE3Pose*>(_vertices[0]);

    const Eigen::Vector2d projectivePoint = m_camera.Point3DToPixel(v->estimate() * m_point);
    _error = _measurement - projectivePoint;
}

bool EdgeProjectionPose::read(std::istream &is) {
    return false;
}

bool EdgeProjectionPose::write(std::ostream &os) const {
    return false;
}

void EdgeProjectionPose::linearizeOplus() {
    const VertexSE3Pose *v = static_cast<VertexSE3Pose *>(_vertices[0]);

    const Eigen::Vector3d projectivePoint = v->estimate() * m_point;

    _jacobianOplusXi = ImageUtils::Jaccobian_Pixel_To_DeltaSE3(m_camera.K(), projectivePoint);
}

void Edge2DTo2DProjection::computeError() {
    _error = _measurement - 
}

bool Edge2DTo2DProjection::read(std::istream &is) {
    return false;
}

bool Edge2DTo2DProjection::write(std::ostream &os) const {
    return false;
}
