/*****************************************************************//**
 * \file    GuardTimer.cpp
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#include "GuardTimer.h"
#include <iostream>

GuardTimer::GuardTimer(const std::string_view name) : m_name{name}, m_checkpoint{Clock::now()} {

}

GuardTimer::~GuardTimer() {
	auto tock = Clock::now();
	std::cout << "D GuardTimer.cpp] " << m_name << " cost: " << std::chrono::duration_cast<TimeDuration>(tock - m_checkpoint).count() << " ms" << std::endl;
}
