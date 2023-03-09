#include "NeuralNet.h"

int main()
{
    NeuralNet n{ { 2, 7, 100, 250, 42, 2 } };

    auto out = n.feed_forward({ 1, 0 });
    std::cout << out[0] << " " << out[1] << '\n';
}