import numpy as np

class MaxPool:

  def __init__(self, pool_size):
    self.pool_size = pool_size
  
  def iterate_regions(self, image):
    h, w, _ = image.shape
    new_h = h // self.pool_size
    new_w = w // self.pool_size
 
    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * self.pool_size):(i * self.pool_size + self.pool_size), (j * self.pool_size):(j * self.pool_size + self.pool_size)]
        yield im_region, i, j

  def forward(self, input):
    #Caching input for backpropagation
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))
    return output

  def backProp(self, d_L_d_out):
    #d_L_d_out is the loss gradient for this layer's outputs.
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input