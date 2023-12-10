#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "math/number.h"
#include "math/tensor.h"

namespace sdl {

namespace utils {

// Function to reverse an integer
template <typename T>
T reverse_int(T value) {
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&value);
    std::reverse(bytes, bytes + sizeof(T));
    return value;
}

// Function to read MNIST image file into a sdlm::Tensor<Number<T>> 
template <typename T>
sdlm::Tensor<sdlm::Number<T>> read_mnist_image_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int columns = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);
    file.read((char*)&rows, sizeof(rows));
    rows = reverse_int(rows);
    file.read((char*)&columns, sizeof(columns));
    columns = reverse_int(columns);

    std::cout << "Number of images: " << number_of_images << std::endl;

    sdlm::Tensor<sdlm::Number<T>> data({number_of_images, rows, columns});
    std::cout << "Data shape: " << std::endl;
    data.print_shape();

    int progress = 0;
    int total_progress = number_of_images * rows * columns;
    int update_interval = total_progress / 10; // Update every 10%

    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                data[i * rows * columns + j * columns + k] = (T)temp;

                // Update progress bar every 10%
                progress++;
                if (progress % update_interval == 0) {
                    float percentage = static_cast<float>(progress) / total_progress * 100;
                    std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << percentage << "%";
                    std::cout.flush();
                }
            }
        }
    }

    std::cout << std::endl;

    file.close();

    return data;
}

// Function to read MNIST label file into a Tensor of sdlm::Number<T>
template <typename T>
sdlm::Tensor<sdlm::Number<T>> read_mnist_label_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    sdlm::Tensor<sdlm::Number<T>> data({number_of_labels});

    int progress = 0;
    int total_progress = number_of_labels;
    int update_interval = total_progress / 10; // Update every 10%

    for (int i = 0; i < number_of_labels; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        data[i] = (T)temp;

        // Update progress bar every 10%
        progress++;
        if (progress % update_interval == 0) {
            float percentage = static_cast<float>(progress) / total_progress * 100;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << percentage << "%";
            std::cout.flush();
        }
    }

    std::cout << std::endl;

    file.close();

    return data;
}

template <typename T>
void load_mnist_data(const std::string& image_filename, const std::string& label_filename, sdlm::Tensor<sdlm::Number<T>>& images, sdlm::Tensor<sdlm::Number<T>>& labels) {
    std::cout << "Loading MNIST images from " << image_filename << std::endl;
    images = read_mnist_image_file<T>(image_filename);
    std::cout << "Loading MNIST labels from " << label_filename << std::endl;
    labels = read_mnist_label_file<T>(label_filename);
}

template <typename T>
void write_tga_image(const std::string& filename, sdlm::Tensor<sdlm::Number<T>>& image) {
    // check if image is 2D
    if (image.get_shape().size() != 2) {
        std::cout << "Error: image must be 2D" << std::endl;
        exit(1);
    }

    // save image to file
    auto& shape = image.get_shape();
    int width = shape[0];
    int height = shape[1];

        // Create TGA header
    const int bytesPerPixel = 1; // Grayscale image
    std::vector<uint8_t> tgaHeader = {0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<uint8_t>(width % 256), static_cast<uint8_t>(width / 256), static_cast<uint8_t>(height % 256), static_cast<uint8_t>(height / 256), static_cast<uint8_t>(bytesPerPixel * 8), 0};

    // Open file for writing
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error opening file for writing." << std::endl;
        exit(1);
    }

    // Write the header to the file
    file.write(reinterpret_cast<char*>(&tgaHeader[0]), tgaHeader.size());

    // Write pixel data to the file
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int row = height - y - 1;
            // Assuming the image pixel data is accessed by (row, column)
            T pixelValue = image.get_values()[row * width + x].value();
            uint8_t pixel = static_cast<uint8_t>(pixelValue * 255); // Convert pixel value to 8-bit grayscale
            file.write(reinterpret_cast<char*>(&pixel), sizeof(uint8_t));
        }
    }

    // Close the file
    file.close();

}


} // namespace utils
} // namespace sdl