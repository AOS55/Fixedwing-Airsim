# Fixedwing-Airsim
Combines JSBSim and Airsim within a python module to simulate a fixedwing aircraft for Reinforcement Learning (RL)

## Installation

- [Install Airsim](https://github.com/microsoft/AirSim)
- [Install JSBSim](https://github.com/JSBSim-Team/jsbsim)
- Clone this repository onto your system
- Move the x8 dir into your JSBSim aircraft dir
- Move the electric800w.xml file into the x8 dir into the engine dir

## Usage

Fixedwing-Airsim has several modules that can be used to control a fixedwing aircraft:
- [main.py](https://github.com/AOS55/Fixedwing-Airsim/blob/b501e63d4172e3ad0149f51b7738db3112cf3ad0/Python-Client/main.py)
 is the main module that uses the aircraft's autopilot top fly a simple flight profile. This would be a good
 place to start with the program.
- [Autopilot.py](https://github.com/AOS55/Fixedwing-Airsim/blob/b501e63d4172e3ad0149f51b7738db3112cf3ad0/Python-Client/autopilot.py)
  contains the basic and path planning autopilots for a default JSBSim C172 and an x8 UAV. The x8 UAV
  is validated against real flight data and based on an [FDM developed in a wind tunnel](https://github.com/krisgry/x8).
- [baisc_ic.xml](https://github.com/AOS55/Fixedwing-Airsim/blob/master/Python-Client/basic_ic.xml) is the default
  starting conditions for the vehicle 1pixel in UE4 equals 1cm for reference. 
- [image_processing.py](https://github.com/AOS55/Fixedwing-Airsim/blob/b501e63d4172e3ad0149f51b7738db3112cf3ad0/Python-Client/image_processing.py) is
 still in development and takes images from simulation for use with computer vision algorithms. 
- [environment.py](https://github.com/AOS55/Fixedwing-Airsim/blob/b501e63d4172e3ad0149f51b7738db3112cf3ad0/Python-Client/environment.py)
 is an abstract class of an [openAI gym environment](https://gym.openai.com/docs/).
- [tasks.py](https://github.com/AOS55/Fixedwing-Airsim/blob/master/Python-Client/tasks.py) is another abstract class
 that defines state, space, action, reward functions for specific RL tasks. 
- [jsbsim_properties.py](https://github.com/AOS55/Fixedwing-Airsim/blob/b501e63d4172e3ad0149f51b7738db3112cf3ad0/Python-Client/jsbsim_properties.py)
 defines properties available from the JSBSim module and controls possible I/O. 
- [jsbsim_simulator.py](https://github.com/AOS55/Fixedwing-Airsim/blob/master/Python-Client/jsbsim_simulator.py
) links JSBSim to AirSim and allows for I/O out of the 2 programmes.

## Attribution

If you find our work useful we'd love to hear from you. If you use this repositorty as part of your research can you please cite the repository in your work:

```
@misc{FW-AirSim,
  author = {Quessy, Alexander},
  title = {Fixedwing-Airsim},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AOS55/Fixedwing-Airsim}}
}
```

Thank you!
