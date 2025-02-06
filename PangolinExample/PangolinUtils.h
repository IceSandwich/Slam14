/*****************************************************************//**
 * \file    PangolinUtils.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include <functional>
#include <string>
#include <GL/glew.h> // The order is important
#include <pangolin/pangolin.h>
#include <Eigen/Eigen>
#include <chrono>
#include <thread>

static constexpr int WindowWidth = 1024, WindowHeight = 768;
void WindowMainLoop(std::function<void()> loop, std::string name = "Viewer", float bg_r = 1.0f, float bg_g = 1.0f, float bg_b = 1.0f, float bg_a = 1.0f) {
	pangolin::CreateWindowAndBind(name, WindowWidth, WindowHeight);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(WindowWidth, WindowHeight, 500, 500, 512, 389, 0.1, 1000),
		pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
	);

	pangolin::Handler3D handler(s_cam);
	pangolin::View &d_cam = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, 0.0, 1.0, -static_cast<float>(WindowWidth) / WindowHeight)
		.SetHandler(&handler);

	while (!pangolin::ShouldQuit()) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(bg_r, bg_g, bg_b, bg_a);

		d_cam.Activate(s_cam);

		loop();

		pangolin::FinishFrame();

		std::this_thread::sleep_for(std::chrono::milliseconds(2));
	}
}