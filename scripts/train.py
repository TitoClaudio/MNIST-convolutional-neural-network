from torchvision import datasets
import numpy as np

from src.layers import Conv3x3, MaxPool, Softmax
from src.utils import save_model

# Load MNIST dataset using PyTorch
test_dataset = datasets.MNIST(root='./data', train=False, download=True)
train_dataset = datasets.MNIST(root='./data', train=True, download=True)

test_images = test_dataset.data.numpy()[:200]
test_labels = test_dataset.targets.numpy()[:200]
train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_images = (test_images / 255.0) - 0.5  # Normalize to [-0.5, 0.5]
train_images = (train_images / 255.0) - 0.5  # Normalize to [-0.5, 0.5]

conv = Conv3x3(8)
pool = MaxPool(2)
softmax = Softmax(13 * 13 * 8, 10)


def forward(image, label):
  out = conv.forward(image)
  out = pool.forward(out)
  out = softmax.forward(out)

  loss = -np.log(out[label])
  accuracy = 1 if np.argmax(out) == label else 0

  return out, loss, accuracy

def train(image, label, learn_rate = 0.005):
  out, loss, acc = forward(image, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backProp(gradient, learn_rate)
  gradient = pool.backProp(gradient)
  gradient = conv.backProp(gradient, learn_rate)

  return loss, acc
  

print('MNIST CNN initialized! Running test set...')

for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  loss = 0
  num_correct = 0
  for i, (image, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(image, label)
    loss += l
    num_correct += acc

print('--- Testing ---')

loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
test_loss = loss / num_tests
test_accuracy = num_correct / num_tests
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Save the trained model
save_model(
    conv.filters,
    softmax.weights,
    softmax.biases,
    filepath='trained_cnn_model.npz',
    metadata={
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'epochs': 3,
        'learning_rate': 0.005,
        'train_samples': len(train_images),
        'test_samples': len(test_images)
    }
)