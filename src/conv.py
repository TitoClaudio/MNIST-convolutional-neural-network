import numpy as np

class Conv3x3:
  
  def __init__(self, num_filters):
    self.num_filters = num_filters

    '''
    Xavier initialization. Divide by 9 to reduce the variance of our initial values.
    filters is a 3d array with dimensions (num_filters, 3, 3)
    '''
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j
  
  def forward(self, input):
    #Caching input for backpropagation
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
    return output

  def backProp(self, d_L_d_out, learn_rate):
    #d_L_d_out is the loss gradient for this layer's outputs.
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # Not returning anything since this is the first layer. 
    # Otherwise, we would return the gradient of the loss with respect to the input.
    return None
