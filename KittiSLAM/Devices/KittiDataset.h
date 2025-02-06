/*****************************************************************//**
 * \file    KittiDataset.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once

#include "Camera.h"
#include "Frame.h"
#include <sophus/so3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <filesystem>
#include <fstream>

class KittiDataset {
public:
    KittiDataset(const char *dataBaseDir = "Data"): m_baseDir{dataBaseDir} {
		readCameraInfo();
	}

    Frame NextFrame() {
        static char tmp[256];

        std::string baseDir = m_baseDir.string();
        sprintf(tmp, "%s/image_0/%06d.png", baseDir.c_str(), m_currentImageIndex);
        cv::Mat left = cv::imread(tmp, cv::ImreadModes::IMREAD_GRAYSCALE);

        sprintf(tmp, "%s/image_1/%06d.png", baseDir.c_str(), m_currentImageIndex);
        cv::Mat right = cv::imread(tmp, cv::ImreadModes::IMREAD_GRAYSCALE);

        Frame ret;
        if (left.data != nullptr && right.data != nullptr) {
            ret.validate = false;
            return ret;
        }

        cv::resize(left, ret.left, cv::Size(), 0.5, 0.5, cv::InterpolationFlags::INTER_NEAREST);
        cv::resize(right, ret.right, cv::Size(), 0.5, 0.5, cv::InterpolationFlags::INTER_NEAREST);
        ++m_currentImageIndex;
        ret.validate = true;
        return ret;
    }

private:
	void readCameraInfo() {
		std::ifstream fin{ m_baseDir / "calib.txt" };
		if (!fin.good()) {
			throw std::exception("Cannot find calib.txt");
		}

        for (int i = 0; i < 4; ++i) {
            char camera_name[3];
            for (int k = 0; k < 3; ++k) {
                fin >> camera_name[k];
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k) {
                fin >> projection_data[k];
            }
            Eigen::Matrix3d K;
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];
            Eigen::Vector3d t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            K = K * 0.5;
            m_cameras[i].Reset(K(0, 0), K(1, 1), K(0, 2), K(1, 2), t.norm(), Sophus::SE3d(Sophus::SO3d(), t));
            std::cout << "[Info] Camera " << i << " extrinsics: " << t.transpose() << std::endl;
        }
        fin.close();
	}

private:
	std::filesystem::path m_baseDir;
    int m_currentImageIndex = 0;
    std::array<Camera, 4> m_cameras;
};
