## Mini Project 2
### Authors: Shubham Sonawane, Raj Shinde, Yash Savle, Moumita Paul


## Introduction
We aim to implement the findings and results obtained in the paper ["Flocking in Fixed and Switching Networks"](https://www.seas.upenn.edu/~jadbabai/papers/boids_automatica5.pdf) by H. G. Tanner, A. Jadbabaie and G. J. Pappas, in IEEE Transactions on Automatic Control, vol. 52, no. 5, pp. 863-868, May 2007. doi: 10.1109/TAC.2007.895948.

The authors have presented the performance of their flocking algorithm in this paper. In the results, it can be seen that the agents, which have different orientations initially, gradually align themselves. Also, the velocities of the agents can be seen to converge over a period of time

We have implemented the algorithm for a fixed network of agents. A fixed network means that the network is time invariant. We represent the communication network with the help of a graph. We have reproduced 2 results from the original paper, which are as follows:
1. Demonstration of gradual orientation and position convergence of the agents
2. Demonstration of velocity convergence of the agents

Our algorithm was tested on 10 agents, similar to the results presented in the paper.

The results are as follows:

1. Initial state of the agents

![Initial positions](results/agents_0.png?style=centerme)

2. Agent position and orientation after some time

![Flocking](results/agents_1.png?style=centerme)

3. Velocity convergence

![velocity](results/vel_convergence.png?style=centerme)

The code for the implementation of this algorithm is available at: https://github.com/yashsavle/Flocking-in-Fixed-and-Switching-Networks/blob/flocking_dev/flocking.py
