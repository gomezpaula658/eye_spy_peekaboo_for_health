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
Data ranging from what's required for our MVP to wishlists for data pertaining to additional stages.

[Dataset available here.](https://www.mdpi.com/2306-5729/6/2/14)

### MVP
- RMFiD Training Labels: A CSV linking retina images to their health data; Disease Risk Binary Classification and 45 Disease Types.
- 3200 retina images.

### Data Needed For Project Expansion
- General health data linked to diseases detailed in RMFiD Training Labels.

## Targets
Defined targets for all our output.

### MVP
Disease Risk: At Risk/Not At Risk; 1 or 0 - Binary Classification

### Additional Stages
- 45 Targets for Potential Disease Classification if eye health is At Risk.
- Identify Unhealthy Parts of an At Risk Eye; multiple undefined targets.

## Libraries

- [Pycaret](https://pycaret.org/); for data analysis and potential deployment.
- [Numpy]();
- [Pandas]();
- [TensorFlow]();
- [Keras]();


### Models
CNN
