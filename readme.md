Dynamic VN Allocation Project
This project is about creating a model for Dynamic Virtual Network (VN) Allocation using a method called Cooperative Multi-Agent Deep Reinforcement Learning (Coop-MADRL). The goal is to optimize how network resources are used in real-time, changing allocations as needed when demand goes up or down. The project involves several steps, including creating a network topology, generating network requests, training a model, and evaluating its performance.

Requirements
Before you start, make sure you have the following installed:

Mininet: A tool for simulating network topologies.
Python 3.x: You need Python version 3 or above to run the scripts.
Python Packages: Several Python libraries are needed for this project.
How to Install Mininet
Mininet is used to simulate the network environment. Follow these steps to install it on your system:

On an Ubuntu system, open the terminal and run:
sudo apt update
sudo apt install mininet
Check if Mininet is installed correctly by running:
sudo mn --test pingall
This command will test a basic network setup by pinging all nodes.
How to Install Python Packages
The following Python packages are necessary for the project:

mininet
pickle
random
numpy
matplotlib
gym
torch
To install these, you can use pip by typing this command in the terminal:

pip install mininet numpy matplotlib gym torch
If you prefer using conda, you can run:

conda install mininet numpy matplotlib gym pytorch
Make sure you are in the correct Python environment before installing.

Project Steps
This project requires you to run several Python scripts in a specific order. Here’s a breakdown of each step:

Generate the Substrate Network Topology using sn.py:

This step creates a basic network setup that represents the physical resources available for virtual networks.
Use the command: sudo python sn.py
Note: The sn.py script uses Mininet, which needs administrative access, so you have to use sudo.
Generate Virtual Network Requests (VNR) using vnr.py:

This script creates virtual network requests that represent the demands the virtual networks will place on the physical network.
Use the command: python vnr.py
This step generates requests with details like required CPU, memory, and bandwidth.
Train the Model using model.py:

This step involves training the Coop-MADRL model using the virtual network requests and the previously generated network topology.
Use the command: python model.py
The model learns how to allocate resources efficiently and the trained model will be saved for later evaluation.
Evaluate the Model using eval.py:

This script tests how well the trained model performs when allocating resources based on a set of new virtual network requests.
Use the command: python eval.py
It will provide feedback on how effective the model is at managing network resources.
Setting Up the Project
Here are the steps you need to follow to set up everything:

Install Mininet by following the instructions above.
Install Python Packages using the pip or conda commands provided.
Download the Project by cloning the repository to your computer:
git clone https://github.com/AK1672/VN_Allocation.git
Then navigate to the project folder: cd <repository-directory>
Run the Scripts in the order mentioned in the "Project Steps" section.
How to Run the Project
Here’s a quick summary of the commands you need to run, in the right order:

To create the network setup: sudo python sn.py
To generate virtual network requests: python vnr.py
To train the model: python model.py
To evaluate the model’s performance: python eval.py
Troubleshooting Tips
Here are some common problems you might encounter and how to fix them:

Problems Installing Mininet: If Mininet isn’t installing properly, make sure your system is up to date. Use the commands:

sudo apt-get update
sudo apt-get upgrade
Permission Issues: If you run into permission errors, make sure to use sudo where necessary (like when running sudo python sn.py).

Missing Python Packages: If any of the Python packages are missing, you can install them individually by typing:

pip install <package-name>
Network Interface Issues: If you encounter problems related to network interfaces (especially with Mininet), restart your computer and try again.