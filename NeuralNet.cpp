#include "NeuralNet.h"

int main()
{
    NeuralNet n{ { 2, 3, 2 }, 0.5 };

    auto out = n.feed_forward({ 1, 1 });
    std::cout << "Output: " << out[0] << " " << out[1] << '\n';
    std::cout << "Error: " << loss(out[0] - 1) << " " << loss(out[1] - 1) << '\n';
    
    std::vector<double> target = { -1, 1 };
    for (int i = 0; i < 1000; i++)
        n.train({ 1, 1 }, target);
    
    out = n.feed_forward({ 1, 1 });
    std::cout << "Output: " << out[0] << " " << out[1] << '\n';
    std::cout << "Error: " << loss(out[0] - target[0]) << " " << loss(out[1] - target[1]) << '\n';
}