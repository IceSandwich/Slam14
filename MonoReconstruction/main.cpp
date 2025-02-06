/*****************************************************************//**
 * \file    main.cpp
 * \brief   Mono Reconstruction
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <sophus/se3.hpp>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "../NumericalOptimization/CameraUtils.h"

Eigen::Matrix3d K = Intrinsic(481.2, -480.2, 319.5, 239.5).Eigen(); // 相机内参

struct RemodeDataset {
	std::vector<Sophus::SE3d> poses; //T_{CW}
	cv::Mat refDepth;
	const int imgWidth, imgHeight;

	RemodeDataset(const char *filename, const int imgWidth = 640, const int imgHeight = 480): imgWidth(imgWidth), imgHeight(imgHeight) {
		std::ifstream fin{ filename, std::ios::in };
		assert(fin.good());

		for (int counter = 0; !fin.eof(); ++counter) {
			std::string imgname; fin >> imgname;
			if (!imgname.compare(GetImgName(counter))) {
				Eigen::Vector3d translation;
				fin >> translation[0] >> translation[1] >> translation[2];

				Eigen::Quaterniond rotation;
				fin >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();

				poses.push_back(Sophus::SE3d{ rotation, translation}.inverse()); // 数据集是T_{WC}，不是T_{CW}，因此求逆
			}
		}
		std::cout << "Read " << poses.size() << " poses." << std::endl;

		refDepth = ReadDepthImage(0);
	}

	static const char *GetImgName(int index) {
		static char ret[256];
		std::sprintf(ret, "scene_%03d.png", index);
		return ret;
	}

	static bool IsValid(int index) { // scene_107不存在
		return index != 107;
	}

	cv::Mat ReadColorImage(int index) {
		static char imagesFilename[256];
		std:sprintf(imagesFilename, "Data/images/scene_%03d.png", index);
		return cv::imread(imagesFilename, cv::ImreadModes::IMREAD_GRAYSCALE);
	}

	cv::Mat ReadDepthImage(int index) {
		static char depthsFilename[256];
		std::sprintf(depthsFilename, "Data/depthmaps/scene_%03d.depth", index);
		cv::Mat ret(imgHeight, imgWidth, CV_64F);

		std::ifstream fin{ depthsFilename, std::ios::in };
		assert(fin.good());
		for (int y = 0; y < imgHeight; ++y) {
			for (int x = 0; x < imgWidth; ++x) {
				double depth = 0; fin >> depth;
				ret.ptr<double>(y)[x] = depth / 100.0;
			}
		}

		return ret;
	}
};

struct ZNCCPatchMatch {
	static double Match(const cv::Mat &img1, Eigen::Vector2d pImg1, const cv::Mat &img2, Eigen::Vector2d pImg2, int patchSize) {
		const int ncc_area = (2 * patchSize + 1) * (2 * patchSize + 1); // NCC窗口面积

		double meanImg1Color = 0.0, meanImg2Color = 0.0;
		std::vector<double> values_ref, values_curr;
		for (int dy = -patchSize; dy <= patchSize; ++dy) {
			for (int dx = -patchSize; dx <= patchSize; ++dx) {
				Eigen::Vector2d delta{ dx, dy };
				double value_ref = static_cast<double>(img1.ptr<uchar>(pImg1[1] + delta[1])[static_cast<int>(pImg1[0] + delta[0])]) / 255.0;
				double value_curr = GetGrayColorWithInterpolation(img2, pImg2 + delta);
				meanImg1Color += value_ref;
				meanImg2Color += value_curr;
				values_ref.push_back(value_ref);
				values_curr.push_back(value_curr);
			}
		}
		meanImg1Color /= ncc_area;
		meanImg2Color /= ncc_area;

		double numerator = 0.0, demoniatorA = 0.0, demoniatorB = 0.0; //分子和分母
		for (int i = 0; i < values_ref.size(); ++i) {
			double &img1Color = values_ref[i];
			double &img2Color = values_curr[i];
			double diff1Color = img1Color - meanImg1Color;
			double diff2Color = img2Color - meanImg2Color;
			numerator += diff1Color * diff2Color;
			demoniatorA += diff1Color * diff1Color;
			demoniatorB += diff2Color * diff2Color;
		}

		return numerator / std::sqrt(demoniatorA * demoniatorB + 1e-10); // 防止分母出现0
	}
};

struct EpipolarSearchResult {
	bool success;
	double score;
	Eigen::Vector2d correspond;
	Eigen::Vector2d direction;
	EpipolarSearchResult(double score, Eigen::Vector2d p, Eigen::Vector2d d) : success{ score >= 0.85 }, score{ score }, correspond{ p }, direction{ d } { }
};

template <typename PatchMatchMethod>
EpipolarSearchResult EpipolarSearch(const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_CR, const Eigen::Vector2d &pt_ref, const double &depth_mu, const double &depth_cov, const int patch_size, const int border) {
	Eigen::Vector3d f_ref = PixelToPoint3D(K, pt_ref, 1).normalized();
	Eigen::Vector3d P_ref = f_ref * depth_mu;
	Eigen::Vector2d p_cur = Point3DToPixel(K, T_CR * P_ref);

	Eigen::Vector2d p_cur_epipolar_direction;
	double d_cur_epipolar_halflength;
	{
		Eigen::Vector3d P_ref_near = f_ref * std::max(depth_mu - 3 * depth_cov, 0.1);
		Eigen::Vector3d P_ref_far = f_ref * (depth_mu + 3 * depth_cov);

		// 限制直线的搜索范围为 [ p_cur - d_cur_epipolar_halflength, p_cur + d_cur_epipolar_halflength ]
		Eigen::Vector2d p_cur_nearPproj = Point3DToPixel(K, T_CR * P_ref_near);
		Eigen::Vector2d p_cur_farPproj = Point3DToPixel(K, T_CR * P_ref_far);
		Eigen::Vector2d p_cur_epipolar = p_cur_farPproj - p_cur_nearPproj;
		p_cur_epipolar_direction = p_cur_epipolar.normalized();
		d_cur_epipolar_halflength = std::min(p_cur_epipolar.norm() * 0.5, 100.0); //限制最远
	}

	// Winner-takes-all
	double best_patch_score = -1.0;
	Eigen::Vector2d best_respondpt;
	for (double t = -d_cur_epipolar_halflength; t <= d_cur_epipolar_halflength; t += 0.7) { // 0.7 = sqrt(2)
		Eigen::Vector2d p_cur_iterPositionInEpipolar = p_cur + p_cur_epipolar_direction * t;
		// 检查是否在图像内
		if (p_cur_iterPositionInEpipolar[0] < border || p_cur_iterPositionInEpipolar[1] < border || p_cur_iterPositionInEpipolar[0] + border >= ref.cols || p_cur_iterPositionInEpipolar[1] + border >= ref.rows)
			continue;

		double patch_score = PatchMatchMethod::Match(ref, pt_ref, curr, p_cur_iterPositionInEpipolar, patch_size);
		if (patch_score > best_patch_score) {
			best_patch_score = patch_score;
			best_respondpt = p_cur_iterPositionInEpipolar;
		}
	}

	return EpipolarSearchResult{ best_patch_score, best_respondpt, p_cur_epipolar_direction };
}

struct TriangulateRecoverDepthResult {
	double depth;
	double uncertainty_2;
	TriangulateRecoverDepthResult(double depth, double uncertainty): depth{depth}, uncertainty_2{uncertainty * uncertainty} { }
};

TriangulateRecoverDepthResult TriangulateRecoverDepth(const Eigen::Vector2d ref, const Eigen::Vector2d curr, const Sophus::SE3d &T_CR, Eigen::Vector2d curr_epipolar_direction) {
	Sophus::SE3d T_RC = T_CR.inverse();
	Eigen::Matrix3d R = T_RC.rotationMatrix();
	Eigen::Vector3d t = T_RC.translation();
	Eigen::Vector3d x1 = PixelToPoint3D(K, ref, 1).normalized(); //f_ref
	Eigen::Vector3d x2 = PixelToPoint3D(K, curr, 1).normalized(); //f_curr

	Eigen::Vector3d Rx = R * x2; // R *x1; //f2
	Eigen::Matrix2d equationLeft;
	Eigen::Vector2d equationRight;
	//equationLeft(0, 0) = x2.dot(Rx);
	//equationLeft(0, 1) = -x2.dot(x2);
	//equationLeft(1, 0) = -Rx.dot(Rx);
	//equationLeft(1, 1) = Rx.dot(x2);
	//equationRight[0] = Rx.dot(x2);
	//equationRight[1] = Rx.dot(t);
	equationRight[0] = t.dot(x1);
	equationRight[1] = t.dot(Rx);
	equationLeft(0, 0) = x1.dot(x1);
	equationLeft(0, 1) = -x1.dot(Rx);
	equationLeft(1, 0) = -equationLeft(0, 1);
	equationLeft(1, 1) = -Rx.dot(Rx);

	double d1, d2;
	if constexpr (false) {
		// Solve equation, Using carmel
		double demoniator = equationLeft.determinant();
		d1 = (equationLeft(0, 1) * equationRight[1] - equationLeft(1, 1) * equationRight[0]) / demoniator;
		d2 = (equationLeft(0, 0) * equationRight[1] - equationLeft(1, 0) * equationRight[0]) / demoniator;
	} else {
		// Solve equation, Using Matrix
		Eigen::Vector2d d = equationLeft.inverse() * equationRight;
		d1 = d[0];
		d2 = d[1];
	}

	Eigen::Vector3d estimatedP1 = x1 * d1; //PixelToPoint3D(K, ref, d1);
	Eigen::Vector3d estimatedP2 = Rx * d2 + t; // PixelToPoint3D(K, curr, d2);
	Eigen::Vector3d estimatedP = (estimatedP1 + estimatedP2) / 2.0; //(estimatedP1 + T_CR.inverse() * estimatedP2) / 2;

	// Calculate uncertainty
	Eigen::Vector3d p = x1 * estimatedP.norm(); // estimatedP;
	Eigen::Vector3d a = p - t;
	double alpha = std::acos(x1.dot(t) / t.norm()); //std::acos(p.dot(t) / (p.norm() * t.norm()));
	double beta = std::acos(a.dot(-t) / (a.norm() * t.norm()));
	//Eigen::Vector3d p2_prime = PixelToPoint3D(K, Point3DToPixel(K, T_CR * estimatedP) + curr_epipolar_direction.normalized(), 1);
	Eigen::Vector3d p2_prime = PixelToPoint3D(K, curr + curr_epipolar_direction.normalized(), 1).normalized();
	double beta_prime = std::acos(p2_prime.dot(-t) / (p2_prime.norm() * t.norm()));
	double gamma = M_PI - beta_prime - alpha;
	double p_prime = std::sin(beta_prime) * (t.norm() / std::sin(gamma));
	//double sigma_obs = p.norm() - p_prime;
	double sigma_obs = estimatedP.norm() - p_prime;

	return TriangulateRecoverDepthResult{ estimatedP[2], sigma_obs };
}

void showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const EpipolarSearchResult& epipolar) {
	cv::Mat ref_show, curr_show;
	cv::cvtColor(ref, ref_show, cv::ColorConversionCodes::COLOR_GRAY2BGR);
	cv::cvtColor(curr, curr_show, cv::ColorConversionCodes::COLOR_GRAY2BGR);

	cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
	if (epipolar.success) {
		cv::circle(curr_show, cv::Point2f(epipolar.correspond(0, 0), epipolar.correspond(1, 0)), 5, cv::Scalar(0, 250, 0), 2);
	} else {
		cv::circle(curr_show, cv::Point2f(epipolar.correspond(0, 0), epipolar.correspond(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
	}

	Eigen::Vector2d t_wh = (Eigen::Vector2d(curr.cols, curr.rows) - epipolar.correspond).array() / epipolar.direction.array();
	Eigen::Vector2d p2 = std::max(t_wh[0], t_wh[1]) * epipolar.direction + epipolar.correspond;
	t_wh = -epipolar.correspond.array() / epipolar.direction.array();
	Eigen::Vector2d p1 = std::min(t_wh[0], t_wh[1]) * epipolar.direction + epipolar.correspond;
	cv::line(curr_show, cv::Point(p1[0], p1[1]), cv::Point(p2[0], p2[1]), cv::Scalar(250, 0, 0), 1);

	cv::imshow("ref", ref_show);
	cv::imshow("curr", curr_show);
	cv::waitKey(1);

	//std::this_thread::sleep_for(std::chrono::seconds(1));
}

enum class Status {
	OK = 0,
	What
};


template <typename T, typename E, E Default>
class Result {
public:
	Result() = delete;
	Result(const Result &) = delete;
	//Result(Result &&) = delete;

	/*explicit*/ Result(T &&retVal) : val{retVal}, err{Default} {

	}

	/*explicit*/ Result(const E &&err) : err{ err } {

	}

	template <typename ...X>
	Result(X&& ...args) : T{args...} {

	}

	operator bool() const {
		return err == Default;
	}

	//void operator=(const E &e) const {
	//	err = e;
	//}

	//void operator=(T &&retVal) const {
	//	val = retVal;
	//}

	T* operator->() {
		return val;
	}

	T val;
	E err;
};

Result<TriangulateRecoverDepthResult, Status, Status::OK > Cal() {
	return {
		1, 1
	};

	return Status::What;
}

void f() {
	auto p = Cal();
	std::cout << p->depth << std::endl;
	if (p) {
		std::cout << "Error occur" << std::endl;
	}
}

#define USE_PARALLEL 1

int main() {
	f();

	RemodeDataset dataset{ "Data/first_200_frames_traj_over_table_input_sequence.txt" };

	const double init_depth = 3.0, init_cov2 = 3.0; //初始深度值, 方差初始值
	cv::Mat depth(dataset.imgHeight, dataset.imgWidth, CV_64F, init_depth); //深度图
	cv::Mat depth_cov2(dataset.imgHeight, dataset.imgWidth, CV_64F, init_cov2); //深度图方差

	cv::Mat refColor = dataset.ReadColorImage(0);
	Sophus::SE3d refPose_WR = dataset.poses[0].inverse();

	cv::imshow("depth_truth", dataset.refDepth * 0.4);
	cv::imshow("depth_estimate", depth * 0.4);
	cv::imshow("depth_error", dataset.refDepth - depth);
	cv::waitKey(1);

	const int border = 20;
	const double min_cov = 0.1;     // 收敛判定：最小方差
	const double max_cov = 10;      // 发散判定：最大方差
	for (int i = 1; i < dataset.poses.size(); ++i) {
		if (!dataset.IsValid(i)) continue;

		std::cout << "*** loop " << i << " ***" << std::endl;

		cv::Mat currColor = dataset.ReadColorImage(i);
		Sophus::SE3d currPose_CW = dataset.poses[i];
		Sophus::SE3d currPose_CR = currPose_CW * refPose_WR;

		const int subViewWidth = dataset.imgWidth - border - border, subViewHeight = dataset.imgHeight - border - border;
#if USE_PARALLEL
		cv::parallel_for_(cv::Range{ 0, subViewWidth * subViewHeight }, [&](const cv::Range &range) {
			for (int r = range.start; r < range.end; ++r) {
				int y = r / subViewWidth + border;
				int x = r % subViewWidth + border;
#else
			for (int y = border; y < dataset.imgHeight - border; ++y) {
			for (int x = border; x < dataset.imgWidth - border; ++x) {
#endif
				Eigen::Vector2d refCoordinate{ x, y };
				double &depthCov2 = depth_cov2.ptr<double>(y)[x];
				double &depthMu = depth.ptr<double>(y)[x];
				if (depthCov2 < min_cov || depthCov2 > max_cov) {
					continue; //已经收敛或者发散
				}

				// 寻找refColor中[x,y]在currColor对应的点
				const int ncc_window_size = 3;    // NCC 取的窗口半宽度
				EpipolarSearchResult search = EpipolarSearch<ZNCCPatchMatch>(refColor, currColor, currPose_CR, refCoordinate, depth.ptr<double>(y)[x], depthCov2, ncc_window_size, border);
				if (!search.success) continue;
				//showEpipolarMatch(refColor, currColor, refCoordinate, search);

				// 三角化，得到估计的深度，并计算估计的不确定性
				TriangulateRecoverDepthResult triangulate = TriangulateRecoverDepth(refCoordinate, search.correspond, currPose_CR, search.direction);

				// 融合状态
				double fuseConv2 = depthCov2 * triangulate.uncertainty_2 / (depthCov2 + triangulate.uncertainty_2);
				double fuseMu = (depthMu * triangulate.uncertainty_2 + triangulate.depth * depthCov2) / (depthCov2 + triangulate.uncertainty_2);
				depthCov2 = fuseConv2;
				depthMu = fuseMu;
#if !USE_PARALLEL
			}
			}
#else
			}
		});
#endif

		// 评估当前深度图
		double ave_depth_error = 0;     // 平均误差
		double ave_depth_error_sq = 0;      // 平方误差
		int cnt_depth_data = (dataset.refDepth.rows - border - border) * (dataset.refDepth.cols - border - border);
		for (int y = border; y < dataset.refDepth.rows - border; y++) {
			for (int x = border; x < dataset.refDepth.cols - border; x++) {
				double error = dataset.refDepth.ptr<double>(y)[x] - depth.ptr<double>(y)[x];
				ave_depth_error += error;
				ave_depth_error_sq += error * error;
			}
		}
		ave_depth_error /= cnt_depth_data;
		ave_depth_error_sq /= cnt_depth_data;
		std::cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << std::endl;

		cv::imshow("depth_truth", dataset.refDepth * 0.4);
		cv::imshow("depth_estimate", depth * 0.4);
		cv::imshow("depth_error", dataset.refDepth - depth);
		cv::waitKey(1);
	}

	cv::imwrite("depth.png", depth);
	std::cout << "Saved to depth.png" << std::endl;

	return 0;
}
