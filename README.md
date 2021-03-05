# seer
Synthetic Energy & Environment Replicator

seer is a conditional GAN that creates synthetic building performance profiles. The synthetic projections are conditioned based on climate and operation constraints. To train the model, reshape the building performance data into a tensor of size [x,24,y], where x is the number of samples and y is the number of building perfromance features. Building operation should be a one-hot-encoded array of size [x,z]: where z is the length of boolean operation constraints. Weather should have a size of [x,w], where w is the length of the weather constraints.
The data for training seer is obtained from "https://github.com/intelligent-environments-lab/CityLearn".
