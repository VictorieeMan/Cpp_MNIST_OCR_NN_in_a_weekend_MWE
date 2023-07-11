# Working journal
Journal for my_first_cpp_NN_experiment, by @VictorieeMan

Repository link: https://github.com/VictorieeMan/my_first_cpp_NN_experiment

## Week 27
### 1: 2023-07-06
#### 1.1: Opening journal
This journal.md is for documenting some thoughts, milestones and decisions that may come up during the development of this project.

#### 1.2: Starting the project
Following the [C++ Neural Netwoek in a Weekend](https://raw.githubusercontent.com/jeremyong/cpp_nn_in_a_weekend/master/doc/DOC.pdf) by [Jeremy Ong](https://github.com/jeremyong) as a guide. My idea is to start by building the program step by step, and then try on some of the extra challenges stated in the end of the guide. See if I can improve the performance by programming a convolutional neural netowrk layer for instance.

#### 1.3: The Computational Graph and comments
The neural network is a computational graph. I really liek that phrasing, because it's both true and very clear in its meaning. For this project we are building a sequential model, but with the fact in mind that a NN model is a graph, we can easily see how we can expand the model to be more complex.

I have decided to add some extra informational comments within the code wherever I learn something new, or feel it's good to take an extra note of why something is added. A contextual why! These comments I will mark in a new paragraph using this comment notation /* */. My regular progammer comments will still be in the // style.

#### 1.4: Base class finished
Having created the base class code, we are now ready to use this base class to start building our model. The pathway forward is to set up the following interface of nodes in the model:
```mermaid
graph LR
A[Input<br>MNIST] --> B[Hidden<br>ReLU]
B --> C[Output<br>Softmax]
C --> D[Loss<br>Crossentropy]
D --Label query.-> A
```
Computational nodes can be read as layers as well, I think that lingo is comparable.

### 2: 2023-07-07
#### 2.1: Finishing up the MNIST input node
Now the MNIST input node is finished, the declaration of the class is in the MNIST.h and the constructors of it's functions in the MNIST.cpp file. The Node is now specialized for delaing with the MNISt data set, reading it to memory and forwarding it to the next node. It also includes a little ASCII art generator, to give a little visual feedback (if called) of what is happening in the program.

## Week 28
### 3: 2023-07-10
#### 3.1: Added the FFNode
Just finished adding the FFNode.h and \*.cpp, which is the feed forward node. This node is the one that will be used for the hidden and output layers. It is a fully connected layer, and it's forward pass is a matrix multiplication. See the code for more details and comments. THis is where the action is!

### 4: 2023-07-11
#### 4.1: Cross Entropy Loss Node
Added the Cross Entropy Loss Node. This node is the one that will be used for the loss function. *"The CCELossNode is similar to other nodes in that it implements a forward
pass for computing the loss of a given sample, and a reverse pass to compute
gradients of that loss and pass them back to the antecedent node. Distinct from
the previous nodes is that the argument to CCELossNode::reverse is ignored
as the loss node is not expected to have any subsequents."*


#### 4.2: Putting things together
Optimizer done, now starting to putting things together in main.cpp. Important do-do:
	
 - [x] In main.cpp void train(char* argv[]) function, test and fix filepaths.
     - train(argv + 2) was forwarded, and this broke the argv[0] usage within the train funciton when it came to opening the files. Fixed by only forwarding argv. I'm considering to create a function that constructs the filepath and returns it to where it is needed.
