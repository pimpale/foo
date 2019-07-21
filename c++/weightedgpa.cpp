//
//  main.cpp
//  WeightedGPA
//
//  Created by Christopher Dumas on 9/20/18.
//  Copyright © 2018 Christopher Dumas. All rights reserved.
//

#include <iostream>
#include <regex>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <locale>

std::string trim(std::string& str) {
	str.erase(0, str.find_first_not_of(' '));       //prefixing spaces
	str.erase(str.find_last_not_of(' ')+1);         //surfixing spaces
	return str;
}

int main(int argc, const char * argv[]) {
	std::cout << "Welcome to the College GPA calculator.\n"
		<< "Please enter your GPA and its credits separated by a comma.\n"
		<< "Grades are a letter (A, B, C, D, F) plus an optional +/-. "
		<< "Credit can be no more than 45 and no less than 0.\nType 'exit' to stop.\n";

	std::string input{""};
	std::regex rx{"([ABCDFabcdf][+-]?)[:blank:]*,[:blank:]*([1-9][0-9]*)"};

	std::vector<float> gpas;
	std::vector<int> credits_list;

	while (1) {
		std::cout << "Please enter a GPA and associated credits: ";
		std::getline(std::cin, input);
		if (input == "exit" && gpas.size() > 0 && credits_list.size() > 0) {
			std::cout << "\n\nOutputting summary:\n";
			break;
		} else if (input == "exit") {
			std::cout << "\nYou need to enter at least one GPA/credit pair. Thank you!\n\n";
		}

		std::smatch matches;
		std::regex_search(input, matches, rx);
		if (matches.size() < 2) {
			std::cout << "\nYou entered '" << input << "', which was something we didn't \
				recognize. Please enter a letter grade, then an optional +/-, \
				and then a space, and then a number for the credits.\n\n";
			continue;
		}

		// Calculate GPA from input
		float gpa = -1.0;
		char gradeLetter = matches[0].str()[0];
		std::cout << (gradeLetter == 'A');

		switch (gradeLetter) {
			case 'A':
			case 'a':
				gpa = 4.0;
				break;
			case 'B':
			case 'b':
				gpa = 3.0;
				break;
			case 'C':
			case 'c':
				gpa = 2.0;
				break;
			case 'D':
			case 'd':
				gpa = 1.0;
				break;
			case 'F':
			case 'f':
				gpa = 0.0;
				break;
			default:
				std::cout << "Unrecognized letter for grade: " << gradeLetter << ".\n";
				continue;
				break;
		}
		if (matches.length() > 2) {
			char gradePlusMinus = matches[1].str()[0];
			switch (gradePlusMinus) {
				case '+':
					if (gpa < 4.0)
						gpa += 0.3;
					else
						std::cout << "We're just going to count that A+ as an A.\n";
					break;
				case '-':
					if (gpa > 0.0)
						gpa -= 0.3;
					else
						std::cout << "We're going to count that F- as F\n";
				default:
					std::cout << "Unrecognized letter for grade: " << gradePlusMinus << ".\n";
					continue;
					break;
			}
		}
		if (gpa > 0) gpas.push_back(gpa);

		// Calculate credits from input
		int credits = std::stoi(matches[3].str());
		if (credits > 45 || credits < 0) {
			std::cout << "Credits not within bounds of 0 and 45!\n";
			continue;
		}
		credits_list.push_back(credits);
	}

	float average{0};
	float credits_sum{0};
	std::cout << std::setw(10) << "GPA" << std::setw(10) << "Credits\n";
	std::cout << std::setw(10) << "---" << std::setw(10) << "-------\n";
	for (int i=0; i < gpas.size(); i++) {
		std::cout << std::setw(10) << gpas[i] << std::setw(10) << credits_list[i];
		average += gpas[i]*credits_list[i];
		credits_sum += credits_list[i];
	}
	average /= credits_sum;

	std::cout << "\nYour weighted GPA is: " << average << ".\n";
	return 0;
}
