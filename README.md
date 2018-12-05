# INF573Project2018
This repo was created by Christopher Murray and Guillaume Dufau.
The following software is the product of a Image Analysis project focusing on Very Deep Neural Networks 
for Image Super-Resolution Reconstruction. 

2 main ideas for this paper
    -Teach a network to perform SR from 2 images of nearly the same perspective
    -Leverage the ability of larger filters to capture more contextual information.
        In order to do this, we tried to minimize the effect of the receptive field
        on padding by inpainting the border of the input at each level of the neural
        network.