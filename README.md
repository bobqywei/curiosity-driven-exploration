# Curiosity Driven Exploration
### Department of Computer Science, University of Waterloo
This is the source code for our work, Motivating Exploration in Reinforcement Learning. Our code is written using Pytorch, including the first implementation of the ICM module in Pytorch. Supplementary videos of our agents interacting in different OpenAI Gym environments can be found [here](https://drive.google.com/drive/folders/1afjQFvrJse2phQuTrLDE0FhVifQKpM2K?usp=sharing)!

## Code References
- https://pathak22.github.io/noreward-rl/ </br>
This code includes Tensorflow implementations of A2C and the ICM Module.
- https://github.com/rpatrik96/pytorch-a2c </br>
This code includes Pytorch implementations of A2C.
- https://github.com/rpatrik96/AttA2C </br>
This code includes Pytorch implementations of AttA2C.

## How to Run this Code
- Install all the required packages, and then simply run the main.py file to train an agent. 
- The configs/main.yaml file can be modified to choose what specific configurations you would like to use (GPU/CPU, Environment, etc.). 
- To generate videos of the trained agents, obtain the saved model, and then update the path of the model checkpoint in play.py. Afterwards, run play.py to generate videos of the agent playing the game.

## Authors

* **Bob Wei**
* **Akshay Patel**
* **Samir Alazzam**
