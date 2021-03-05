# seer
Synthetic Energy & Environment Replicator

Seer is a conditional GAN that creates synthetic building performance profiles. The model requires three sets of inputs for training, i.e. "performance", "operation", and "weather". Synthetic projections of building performance profiles are conditioned based on climate and operation constraints. The "performance" input should be a tensor of size [x,24,y], where x is the number of samples and y is the number of building perfromance features. The "operation" input should be a one-hot-encoded array of size [x,z]: where z corresponds to the length of the Boolean operation constraints. The "weather" input should have a size of [x,w], where w is the length of the weather constraints.
The data for training seer is obtained from "https://github.com/intelligent-environments-lab/CityLearn".

### Discriminator:
![Picture7](https://user-images.githubusercontent.com/27851066/110112203-b8520b00-7db1-11eb-8203-35613200e0fe.png)


### Generator:
![Picture61](https://user-images.githubusercontent.com/27851066/110112225-c011af80-7db1-11eb-9430-1d790408ef45.png)

# Requirements
Seer has been tested on the following Python libraries:
* Numpy 1.18.5
* Scipy 1.4.1
* TensorFlow 2.3.0

# Contact
Fazel Khayatian, Urban Energy Systems Laboratory, Empa
https://www.empa.ch/web/khfa
