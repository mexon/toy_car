# Detection of car using difference between adjacent frames

The main detection algorithm uses motion detection from two adjacent frames.  This does have the limitation that it can only detect the car while it is moving.  The implementation is contained in a class `ToyCarTracker`.

The previous frame is always stored as a member variable, and with each new frame the difference is calculated, as a full-colour comparison.  The colour is then collapsed to grayscale.  Colour comparison was used in case later we wished to exploit the relatively saturated colours of the cars in this step, although this appears to be unnecessary.

The difference signal is then blurred with a uniform filter of size 50x50 pixels, in order to allow nearby motion pixels to reinforce each other.  The result is a smooth ellipsoid in the direction of the car.  However, this includes multiple noise areas.  These are filtered out by finding the maximum smoothed value and removing pixels less than half that maximum.

From this signal, the OpenCV `findContours` function is used to find the contours sorted by size.

# Classification of car identity

This step exploits the fact that the cars are painted in distinctive selections from a highly saturated colour palette.  The goal is to use a histogram classification of the hue of the most relevant pixels.  The hue values should be robust to changes in lighting conditions, allowing for a stable classification.

The previously-identified car region is clipped from the image and used for these steps, to reduce processing time.  The frame is converted to HSV.

Initially only the saturation signal was used to search for classification pixels.  However, this was found to include very dark regions which, due to noise, happened to have highly saturated regions.  For example, a pixel with 0, 0, 1 brightness levels for red, green and blue respectively would be interpreted as fully saturated blue, even though the single value is most likely noise.

To resolve this, a mixed signal consisting of the high 4 bits of the saturation channel multiplied by the high 4 bits of the lightness channel was used.  Pixels exceeding half of the maximum value of this signal are considered to be signal pixels.

A histogram is computed using 8 buckets.  This was intended to separate yellow from red pixels, these being the most distinctive colours between the two cars.  In practice, we observed that yellow and red shade into each other quite often.  A better approach might be to imitate the human vision system and have three buckets that partially overlap.

The computed histogram is compared as a vector against the average of the histograms of earlier training runs.  The available training histogram with the smallest vector distance to the observed histogram is used as the classification.

This classification also changes the structure of the code: there is now a `CarData` class to hold separate motion data for the two cars in the field.  Classification allows separating this data, and supporting detection of multiple cars on the same playing field.

# Evaluation

No objective evaluation was conducted of the performance of the system.

Evaluation of performance requires an objective ground truth against which to compare the output of the implementation.  In this case there is no objective information about the position of the toy cars in millimeters.  A typical test setup would solve this problem by introducing additional sensors or controlled conditions that would allow completely accurate ground truth measurement.  In this case, however, we only have sensor (camera) data that is the same as the real test environment.

Another common approach to dealing with the ground truth problem is to perform competitor analysis: comparing the results of our implementation against results from competing implementations.  In this case we are fortunate that there are several independent implementations.  However, since all these implementations are under development simultaneously, it is not practical to make such a comparison before submission.  It would be a useful exercise to run exactly such a comparison after all implementations have been submitted.

Nevertheless, a common problem with competitor analysis comes when your own implementation outperforms all competitors, at least in a subset of cases.  The metric cannot evaluate the additional performance in such cases, they are invisible.  If the metric is used to direct development efforts, it can lead to the implementation actually becoming worse.

Furthermore, in order to accurately assess performance, tests should be run on a representative selection of inputs.  Typically performance evaluation results are part of a feedback loop that guide further development.  Effectively the implementation loop becomes an optimisation problem for the performance metric.  If the metric is assessed against data that does not reflect the real operating environment, the developed system will converge to a suboptimal design.  During our development work, we already encountered such effects: some implementations were tested against low-resolution video files, but failed when tested against high-resolution video files.

In this case we do not have test data that accurately reflects the real test environment.  Therefore any performance evaluation results would be suspect.

One option for evaluating positioning accuracy would be to exploit the fact that the test environment is flat, and so the toy car should achieve the same top speed in all directions.  By driving the car in straight lines in various directions during a calibration run, we can assess whether the velocity calculated by the system is constant.  The metric in this case is the variance in the measured velocity.  However, the currently gathered test data is not suitable for this purpose, since it consists of the cars driving along random trajectories.

# Position and velocity calculation using detected car positions
The position, velocity, and angular velocity of the car are calculated in the CarData class. After the car is detected, the OpenCV minAreaRect() function is used to obtain a rotated bounding rectangle around the detected contour. The center of this rectangle is used as the estimated position of the car.
The detected position is converted into real-world coordinates before calculating velocity. The program stores the previous position and timestamp, then compares them with the current values to calculate the displacement over time.

# Transformation of screen position into real-world position using known coordinates
The program uses a homography transformation to convert positions from image pixel coordinates into real-world coordinates. This is necessary because the camera measures positions in pixels, while the physical position of the car needs to be expressed in millimeters.
The transformation is calculated using four corresponding points between the image and the real-world playing field. The field dimensions are defined as 2500 mm × 1500 mm.
The OpenCV findHomography() function is used to calculate the transformation matrix H_matrix. The image_to_real() function then applies this matrix to convert image coordinates (u, v) into real-world coordinates (X, Y).

# Detection of orientation
The orientation of the car is estimated using a combination of the OpenCV minAreaRect() function and HSV color detection.
Initially, minAreaRect() is used to determine the main axis of the detected car. However, this method cannot distinguish between the front and rear of the car. It only provides an axis with a 180-degree ambiguity.
To address this limitation, HSV color detection was added to identify the blue marker located at the front of the car. The detect_blue_front() function converts the image into HSV color space and applies a predefined color threshold to detect blue pixels within the car's bounding region.
However, the orientation detection is still sometimes unstable. Changes in lighting conditions, inaccuracies in the detected contour, and variations in the blue marker position can affect the estimated heading. Further improvements are needed to make the orientation detection more reliable.
