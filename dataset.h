#ifndef DATASET_H
#define DATASET_H
#include <Eigen/Eigen>
#include <fstream>
#include <string>
#include <vector>

Eigen::VectorXd encode(const int n) {
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(10, 1);
  ans(n) = 1;
  return ans;
}

Eigen::MatrixXd encodeMatrix(const std::vector<int> &nums) {
  Eigen::MatrixXd ans(nums.size(), 10);
  for (size_t i = 0; i < nums.size(); ++i) {
    ans.row(i) = encode(nums[i]).transpose();
  }
  return ans;
}

Eigen::MatrixXd encodeData(const std::vector<Eigen::MatrixXd> &vec) {
  int size = vec[0].rows() * vec[0].cols();
  Eigen::MatrixXd ans(vec.size(), size);
  for (size_t i = 0; i < vec.size(); ++i) {
    Eigen::VectorXd a = Eigen::Map<const Eigen::VectorXd>(vec[i].data(), size);

    ans.row(i) = a.transpose();
  }

  return ans;
}

////////////////////////////////////////////////
// code for reading mnist data was taken from
// http://eric-yuan.me/cpp-read-mnist/
////////////////////////////////////////////

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Eigen::MatrixXd read_Mnist_Label(std::string filename) {
  std::vector<int> vec;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char *)&magic_number, sizeof(magic_number));

    magic_number = ReverseInt(magic_number);

    file.read((char *)&number_of_images, sizeof(number_of_images));

    number_of_images = ReverseInt(number_of_images);

    for (int i = 0; i < number_of_images; ++i) {
      unsigned char temp = 0;
      file.read((char *)&temp, sizeof(temp));
      vec.push_back((int)temp);
    }
  }
  return encodeMatrix(vec);
}

Eigen::MatrixXd read_Mnist(std::string filename) {
  std::vector<Eigen::MatrixXd> vec;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);
    for (int i = 0; i < number_of_images; ++i) {
      Eigen::MatrixXd tp = Eigen::MatrixXd::Zero(n_rows, n_cols);
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          tp(r, c) = (int)temp;
        }
      }
      vec.push_back(tp);
    }
  }
  return encodeData(vec);
}

#endif // DATASET_H
