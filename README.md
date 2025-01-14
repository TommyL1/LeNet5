# LeNet5
This project involves implementing LeNet5, a CNN developed by Yann LeCun for recognizing handwritten digit images. LeNet5 formalized a deep-CNN architecture, including components such as convolutional layers, pooling layers, and fully connected layers. 


Model 1: 
Data: Run data.py to collect MNIST data samples. The MNIST database consists of 28 × 28
images. You will need to resize them to 32×32 to match the regular database in the original paper.
– We started by defining a class called ”MNISTDataset”. This class is used to prepare and
process the MNIST dataset for training and testing. It takes images and labels stored in a
DataFrame.
– The images are resized from 28x28 to 32x32 pixels since the LeNet-5 model expects 32x32
input size. This is done using the ”resize” method from the ”PIL” library.
– After resizing, the images are normalized so that the pixel values range between 0 and 1,
which helps the model learn better.
– To match the input format of the model, the images are inverted (black background and white
digits) and reshaped to add a channel dimension.
– Finally, the dataset is split into a training set and a test set. Both are loaded into DataLoader
objects, which let us feed the data into the madel in batches.
• Optimization: Use the “Stochastic Diagonal Levenberg-Marquardt” method. Compute the exact
second derivative for hkk in equation (22) without the approximations in Appendix C.
– The optimizer used here is a custom-defined ”ConstantStepOptimizer”. With the clarifications
given from the teacher, we used a fixed learning rate of 0.001 to update the model parameters.
– This keeps things simple and helps us focus on understanding how weights are updated during
back propagation.
• Loss function: Use equation (9). j = 0.1 and i denotes incorrect classes.
– The loss function, implemented in ”compute loss”, measures how well the model predicts the
correct label. It uses a parameter j = 0.1 to control the penalty for incorrect predictions.
– The function calculates the distance between the predicted and true classes, applying some
mathematical transformations to get the average loss across all samples in a batch.
• At every pass (every epoch), track the error rates. You will need the rates for plotting in the
performance evaluation.
– The training loop runs for 20 epochs, In each epoch:
– We pass batches of images through the model to compute the predicted distances.
– The loss is calculated, and gradients are computed using backpropagation.
– The optimizer updates the model’s weights based on the gradients.
– We calculate the average loss for the epoch and track the training and testing error rates.

Model 2: 
(0) adding image pre-processing block,
– Image Pre-Processing Block: In the modified code, the images are not only resized but also
undergo several transformations in the pre-processing block before being fed into the model.
– Test Set Transformation: The test set images are only resized to 32x32, without augmentation,
to maintain consistency during evaluation.
• (1) data augmentation,
– Data Augmentation: It uses the ”torchvision.transforms” module to apply:
∗ Images are rotated randomly by up to 15 degrees, which helps the model become invariant
to small rotations in the data.
∗ The images are translated randomly by 10% of their width and height, allowing the model
to recognize objects despite slight positional changes.
∗ The images are flipped horizontally at random, which allows the model to handle flipped
or mirrored versions of the input.
• (2) adopting the modern blocks such as max pooling, and dropout, ReLU, softmax, or Spatial
Transformation Network,
– ReLU Activation: The ScaledTanh activation is replaced with ReLU, a much more commonly
used activation fuction in deep learning. ReLU is faster to compute and helps with the
vanishing gradient problem by providing a constant gradient for positive inputs.
– Max Pooling: The modified code adopts Max Pooling instead of average pooling. Max pooling
retains the most significant feature from a region, helping the model focus on the strongest
features.
– Dropout: Dropout was introduced with a probability of 50%. Dropout randomly disables
a certain percentage of the neurons during training to prevent overfitting by reducing the
model’s reliance on any perticular neuron.
– Softmax: Softmax is a function that converts raw scores into probabilities, making the model’s
predictions easier to interpret by showing the likelihood of each class, with the class having
the highest probability being the predicted one.
• (3) different training schemes.
– Learning Rate Scheduler: A StepLR scheduler is introduced, which reduces the learning rate
by half after every 5 epochs. This allows the model to fine-tune its parameters and converge
more efficiently.
• After running ”test2 custom test(needs our mnist.py file to run).py”, we got an accuracy of 93.51%.
