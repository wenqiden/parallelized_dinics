#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
   if (argc < 4) {
      std::cout << "Usage: " << argv[0] << " <input file>" << " <source>" << " <sink>" << std::endl;
      return 1;
   }
   else {
    return 0;
   }
}