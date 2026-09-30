# Assignment 1: Toy Race Car
## Prof. Jacky Baltes <jacky.baltes@ntnu.edu.tw>
## Prof. Reinhard Gerndt <r.gerndt@ostfalia.de>
## Due Date: Friday, 9th Oct. 2026 23:59 

# Introduction

Develop a global vision system that is able to track and identify a small toy race car.

The goal is to play a treasure hunt game by controlling the
toy cars from a computer (laptop or desktop PC) that
sends signals to the car using a hacked remote controller connected to a Raspberry Pico 2W.

# Group Assignment

This assignment is part of the global virtual classroom (GVC) project at NTNU. Hence the assignment must be done jointly by students from NTNU and Ostfalia University.

The groups for this assignment are available in the Ostfalia Open Moodle system.

# Specifications

The following section describe some of the technical details for this assignment. 

![Sample Image 1](https://i.postimg.cc/Pfwm0cdB/Screenshot-2026-09-28-141425.png) | ![Sample Image 2](https://i.postimg.cc/L6kj81Yx/Screenshot-2026-09-28-141441.png) | ![Sample Image 3](https://i.postimg.cc/xTGL1zJs/Screenshot-2026-09-28-141456.png) |
![Sample Image 4](https://i.postimg.cc/k49m9zkz/Screenshot-2026-09-28-151150.png) | ![Sample Image 5](https://i.postimg.cc/gJDP04W7/Screenshot-2026-09-28-151225.png) | ![Sample Image 6](https://i.postimg.cc/vBzdmXsj/Screenshot-2026-09-28-151243.png) |

## 3D Mapping

The camera will be positioned on a selfie stick or tripod with a side angle view of the playing field.

Use the OpenCV functions `cv:.calibrateCamera()` and `cv.findHomography ()` to map image coordinates to real world coordinates on the playing field. Another approach is to place  Aruco markers or April tags outside of the playing field.

## Toy Cars

The toy cars are commercially available toy cars as shown in this image. The actual cars are available in the Educational Robotics Center (ERC) at the National Taiwan Normal University (NTNU) in Taiwan.

## Floor/Background

The playing field is a flat field of approximate size 
1.5m by 2.5m . The playing field is a mostly untectured tile floor or carpet.

## Computing

The global vision system should run on a standard laptop with a single or multiple external webcams or on mobile phones using the Android or IOS operating system.

Since the car can move up to 20 km/h, you want to use a computationally efficient approach that tracks the car at 60 frames per seond, that is 16.67 ms per frame.

## Output

The global vision server detects and outputs the following information for each detected car:
    * timestamp (relative clock in microseconds)
    * car id
    * the real world position of the center of the car(x,y in millimeters, or -1000.0, -1000.0 if no car in the image), 
    * real world orientation (theta in degrees),
    * real world velocity (dx, dy) of the car in millimeters per second
    * real world angular velocity of the car in degrees per second
    * as well as image coordinates (u,w) of the center of the car in the image

Example:
```
1000023:"Red Racer",123.0,230.0,90.0,1000.0,-300.0,0.6,237,1024\n
1016027:"Green Hornet",124.0,232,....\n
```



The global vision server should transmit this information as a utf-8 encoded string via UDP port 5000 - make sure that the port can be changed via the command line. This way, controllers can run on different computers and control the car.

# Evaluation

You are expected to carefully evaluate the performance of your vision system. 

First, show the receiver operator characteristic (ROC) curve by plotting the true positive rate (TPR) against the flase positive rate (FPR) at various settings of your classifier. The specific setting depends on the type of classifier that you implemented. For example, if you use a histogram based approach, you may set different thresholds between the recorded histogram and the template. If you use a CNN based approach, you can use train or use an exisiting confidence value and use different threshold settings for it.

Secondly, show the mapping error of your system, that is given a non-moving car, what is the average and maximum mapping error, i.e., difference between real position and real orientation versus true position and orientation, of your vision system. 

For the oral exam, you should prepare convincing evidence of your system running and its performance.

# Report

Create a two - five page report describing your approach to solving this assignment. Explain why you chose this particular approach (e.g., CNN, color detection), how you structured your code, what major issues did you face during implementation, and how you evaluated the system.

# Submission

The assignment should be submitted as a group using the 
Open Moodle system.

Include a filled out copy of the [NTNU honesty declaration](https://docs.google.com/document/d/1Z9FyOFt--OL_cqse55b-CifBgPDfFZi2L6-sUQBvZ4I/edit?usp=sharing) in your assignment.

# Marking

Due to the increased capabilities of current state of art AI systems and the increased use of these systems by students, the submission of the assignment will be marked as pass/fail woth 1% of the final grade.

You must demonstrate the capabilties of your system and answer questions about your submission in an oral exam after the submission deadline, which is worth 19% of the final grade.
