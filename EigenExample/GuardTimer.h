/*****************************************************************//**
 * \file    GuardTimer.h
 * \brief   
 * 
 * \author  gh @IceSandwich
 * \date    March 2024
 * \license MIT
 *********************************************************************/

#pragma once
#include <string_view>
#include <chrono>

class GuardTimer {
	using TimeDuration = std::chrono::duration< double, std::ratio<1, 1000> >;
	using Clock = std::chrono::high_resolution_clock;

public:
	GuardTimer(const std::string_view name);
	~GuardTimer();

protected:
	const std::string_view m_name;

	Clock::time_point m_checkpoint;
};

