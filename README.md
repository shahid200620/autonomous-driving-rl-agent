\# Autonomous Driving RL Agent



This project demonstrates a reinforcement learning agent trained to drive autonomously inside a custom 2D racing environment. The environment is built using Pygame and wrapped using Gymnasium so that it can interact with reinforcement learning algorithms from Stable-Baselines3.



Instead of relying on pixel-based vision, the agent uses simulated distance sensors created through ray casting. These sensors act similar to LiDAR, allowing the agent to detect walls and navigate the track efficiently using lightweight computations that run comfortably on a standard CPU.



The main goal of this project is to demonstrate how a full reinforcement learning pipeline can be designed from scratch, including environment creation, agent training, evaluation, and visualization of learned behaviour.



---



\## Project Structure





.

├── Dockerfile

├── requirements.txt

├── config.yaml

├── train.py

├── evaluate.py

├── record\_video.py

├── src

│ └── environment.py

├── tracks

│ └── track\_1.txt

├── models

│ └── ppo\_car\_agent.zip

├── results

│ ├── training\_log.json

│ ├── reward\_curve.png

│ └── agent\_demonstration.mp4





---



\## Environment Design



The driving environment is implemented using Pygame and follows the Gymnasium API standard.



Key features of the environment:



\- Custom 2D racetrack defined using wall coordinates

\- Collision detection against track boundaries

\- Ray-casting based distance sensors for perception

\- Discrete action space for vehicle control

\- Continuous observation space representing sensor readings and velocity



The agent receives environmental information through multiple distance rays which simulate LiDAR-style sensors. This allows the agent to understand how close it is to obstacles and learn navigation strategies.



---



\## Action Space



The agent can perform the following actions:



0 – No operation  

1 – Accelerate  

2 – Brake  

3 – Turn Left  

4 – Turn Right  



---



\## Observation Space



The observation vector contains:



\- Distance values from multiple sensor rays

\- Normalized vehicle velocity



These values are normalized to maintain training stability.



---



\## Reward Design



The reward function is designed to encourage safe and continuous driving behaviour.



Positive reward is given for:



\- Surviving longer in the environment

\- Moving without collisions



Negative reward is given for:



\- Collisions with track boundaries



This reward shaping helps the agent learn stable driving behaviour over time.



---



\## Training



The agent is trained using the \*\*Proximal Policy Optimization (PPO)\*\* algorithm from Stable-Baselines3.



Training is configured through `config.yaml`, which defines:



\- PPO hyperparameters

\- Environment parameters

\- Total training timesteps



To start training run:





python train.py





After training completes, the following artifacts are generated:



\- `models/ppo\_car\_agent.zip` → trained model

\- `results/training\_log.json` → training statistics

\- `results/reward\_curve.png` → reward progression plot



---



\## Evaluation



To evaluate the trained agent:





python evaluate.py





The script runs multiple episodes and prints the average performance.



Example output:





Mean Reward: 215.43

Std Reward: 34.12





---



\## Recording Agent Behaviour



To record a demonstration of the trained agent navigating the track:





python record\_video.py





This generates a video file:





results/agent\_demonstration.mp4





The video provides a visual demonstration of the agent’s learned behaviour within the environment.



---



\## Docker Support



The project includes a Dockerfile to allow reproducible execution.



To build the container:





docker build -t car-agent .





To run evaluation inside the container:





docker run car-agent python evaluate.py





This ensures the project can be executed consistently across different environments.



---



\## Key Technologies Used



\- Python

\- Pygame

\- Gymnasium

\- Stable-Baselines3

\- PyTorch

\- Reinforcement Learning (PPO)



---



\## Learning Outcome



This project demonstrates how reinforcement learning agents interact with custom simulation environments and learn behaviours through trial and error. It highlights the importance of environment design, reward shaping, and sensor-based perception in autonomous systems.



The final result is a lightweight yet complete reinforcement learning pipeline capable of training an autonomous driving agent within a custom simulation.

