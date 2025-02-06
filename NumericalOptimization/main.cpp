/*****************************************************************//**
 * \file    main.cpp
 * \brief   Numerical Optimization
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "SimpleProblem.h"
#include "PnP.h"
#include "ICP.h"

int main() {
	//GaussianNewton();
	//CeresSolver();
	//G2O();
	Pose3dTo2d();
	//Pose3dTo3d();

	return 0;
}
