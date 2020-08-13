#include <iostream>
#include "CSRMatrix.hpp"

int main(int argc, char** argv) {
  using T = float;
  using index_type = int;

  if (argc < 2) {
    fprintf(stderr, "usage: ./print_mat [matrix market file]\n");
    exit(1);
  }

  std::string fname = std::string(argv[1]);

  BCL::CSRMatrix<T, index_type> mat(fname);

  std::cout << "Printing out matrix:" << std::endl;
  for (size_t i = 0; i < mat.shape()[0]; i++) {
    for (index_type j_ptr = mat.rowptr_[i]; j_ptr < mat.rowptr_[i+1]; j_ptr++) {
      index_type j = mat.colind_[j_ptr];
      T value = mat.vals_[j_ptr];
      std::cout << i << " " << j << " " << value << std::endl;
    }
  }

  return 0;
}
