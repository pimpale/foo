#include <iostream>
#include <vector>

int main() {
    std::vector<double> x(2);

    for(int i = 0; i < x.size(); i++) {
      std::cin >> x[i];
    }

    for(int i = 0; i < x.size(); i++) {
      std::cout << x[i] << std::endl;
    }
}
