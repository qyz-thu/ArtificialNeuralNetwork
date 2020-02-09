# Extra Modifications
  
- *main.py*: 
	- add codes to record the loss and accuracy during training in form of numpy arrays;
	- add extra command line arguments:
		- *model_type*: argument determining the type of model used with the type of int. 0 for basic RNN, 1 for LSTM and 2 for GRU. Default is 2.
		- *stop_time*: argument determining early stop time with the type of int. Default is 10.
		- *attention*: argument determining whether to use self-atteniton technique with the type of bool. Default is true.
- *plot.py*: python script to plot figures using numpy arrays created during training. Takes following argument:
	- *model_type*: argument determining the type of model to be ploted with the type of int. 0 for basic RNN, 1 for LSTM and 2 for GRU. 
	- *layers*: argument determining the layers of model to be ploted with the type of int.