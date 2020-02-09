# Modifications

- *model.py*: add member self.initializer for initializing kernel weights in convolutional layers.
- *main.py*: add codes to get validation loss & accuracy every 10 iterations; save loss & accuracy data in `.npy` files; terminate the training if the validation accuracy doesn't reach new high for a consecutive 5 epochs.
- add *plot.py*: plot loss & accurcy based on the `.npy` files generated in training.
