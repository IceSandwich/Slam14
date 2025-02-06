/*****************************************************************//**
 * \file    Camera.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once

#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

class Camera {
public:
	Camera() = default;
	Camera(double fx, double fy, double cx, double cy, double baseline, const Sophus::SE3d &pose) {
		Reset(fx, fy, cx, cy, baseline, pose);
	}

	inline void Reset(double fx, double fy, double cx, double cy, double baseline, const Sophus::SE3d &pose) {
		m_fx = fx;
		m_fy = fy;
		m_cx = cx;
		m_cy = cy;
		m_baseline = baseline;
		m_pose = pose;
		m_poseInv = pose.inverse();

		m_K << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1;
	}

	inline const Eigen::Matrix3d &K() const {
		return m_K;
	}

	// 图像坐标系到相机坐标系
	inline Eigen::Vector2d PixelToCam(const Eigen::Vector2d &p) const {
		return Eigen::Vector2d{
			(p[0] - m_cx) / m_fx,
			(p[1] - m_cy) / m_fy
		};
	}

	inline Eigen::Vector2d PixelToCam(const cv::Point2f &p) const {
		return Eigen::Vector2d{
			(p.x - m_cx) / m_fx,
			(p.y - m_cy) / m_fy
		};
	}

	// 图像坐标系到三维点
	inline Eigen::Vector3d PixelToPoint3D(const Eigen::Vector2d &p, double depth) const {
		Eigen::Vector2d cam = PixelToCam(p); //先转换到相机坐标系
		return Eigen::Vector3d{
			cam[0] * depth,
			cam[1] * depth,
			depth
		};
	}

	inline Eigen::Vector3d PixelToPoint3D(const cv::Point2f &p, double depth) const {
		Eigen::Vector2d cam = PixelToCam(p); //先转换到相机坐标系
		return Eigen::Vector3d{
			cam[0] * depth,
			cam[1] * depth,
			depth
		};
	}

	// 三维点投影到图像坐标
	inline Eigen::Vector2d Point3DToPixel(const Eigen::Vector3d point) const {
		return Eigen::Vector2d(
			m_fx * point[0] / point[2] + m_cx,
			m_fy * point[1] / point[2] + m_cy
		);
	}

	inline Sophus::SE3d &GetPose() {
		return m_pose;
	}

private:
	double m_fx = 0, m_fy = 0, m_cx = 0, m_cy = 0, m_baseline = 0;
	Sophus::SE3d m_pose;
	Sophus::SE3d m_poseInv;

	Eigen::Matrix3d m_K;
};
