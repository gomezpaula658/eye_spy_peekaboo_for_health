# EYE SPY
CNN for retina health recognition.

## Our Front End
Our clean interface will welcome health professionals and maintain an elegant, minimal design.

### Stage 1 (MVP)
Upload an image of a retina alongside we'll return a binary classification of Healthy/Unhealthy.

### Stage 2
If the eye is unhealthy, we'll also provide diseases that could factor into the eye's state.

### Stage 3
Identify areas of the eye that are unhealthy.

## The Data

### MVP
- RMFiD Training Labels: A CSV linking retina images to their health data; Disease Risk Binary Classification and 45 Disease Types.
- 32000 retina images.

### Data Needed For Project Expansion
- General health data linked to diseases detailed in RMFiD Training Labels.


## Targets
Defined targets for all our output.

### MVP
Disease Risk: Risk/Not At Risk; 1 or 0 - Binary Classification (M)
45 Targets for Potential Disease Classification
Identify Unhealthy Parts of an Eye

### Models
Computer Vision
CNN
