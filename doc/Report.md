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
