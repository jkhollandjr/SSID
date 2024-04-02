# PyTorch SSID Implementation

Here, the datasets representing the selected flows (with first first set of flows in one folder and the second set in another) are preprocessed by the preprocessing.py script. Then, train.py will train the flow embedders and output the models. These models, along with the validation data, can then be used by test.py to create a new dataset representing the distances between windows from many of the possible matched flow pairs. Finally, these values can be used by the neural network defined in nn\_classifier.py to output the model performance. 

This implementation varies from the DeepCoFFEA approach in that it uses multiple feature representations, doesn't filter traces that have empty windows, outputs validation loss performance, uses the Adam optimizer with learning rate scheduling, and adds an additional window. These chages improve performance significantly. 


## 'Live' Defense simulation

To improve performance against traffic obfuscated by delayed or 'dummy' traffic, we also add functionality to simulate the defense on the base traffic while the model is training. This is done by using a variety of torch functions to alter the traffic representations. By simulating the defense in this manner, rather than by simply simulating the defense on the original dataset, the model learns for a larger variety of defense configurations. This improves the model's ability to generalize, especially against randomized defenses, which can alter traffic in many different ways. 

To use the 'live' defense functionality, run preprocessing\_simple.py rather than preprocessing.py. Then, for training and testing, run train\_live.py and test\_batched\_live.py. The final performance can then be found with nn\_classifier.py.  

