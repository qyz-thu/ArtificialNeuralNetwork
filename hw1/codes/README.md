#README
<font size=4>
&#8195;Directory *2017013622/codes* should contain following python scripts:  

- *layers.py*: 
	- implement forward and backward function.
	- Add argument *new_method* in constructor of class *Linear*. In updating weights in class Linear, if the argument is true, the following codes

&#8195;
			
		self.diff_W = mm * self.diff_W - lr * (self.grad_W + wd * self.W)
        self.W = self.W + self.diff_W

        self.diff_b = mm * self.diff_b - lr * (self.grad_b + wd * self.b)
        self.b = self.b + self.diff_b


&#8195;&#8195;&#8195;will be replaced by  


		self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b


- *loss.py*:
	- implement forward and backward function.
- *load_data.py*: no change.
- *network.py*: no change.
- *utils.py*: add function to support boolean command-line argument.
	- *Type*: takes one argument(boolean of string). Return the intended value of argument. 
- *solve_net.py*: 
	- add return value for function *test_net*: returns the mean accuracy and lost in the test.
- *plot.py*: a newly added script to plot accuracy and loss. Accept following command-line arguments:
	- *--plot_one_layer*: when true, the script will plot the accuracy and loss during training of models with **one** hidden layer based on .npy files created after training. Default is False.
	- *--plot_two_layer*: when true, the script will plot the accuracy and loss during training of models with **two** hidden layers based on .npy files created after training. Default is False.  
  
&#8195;&#8195;Following example shos how to plot the figures of one hidden layer models:
  
		$ python plot.py --plot_one_layer true   

- *run_mlp.py*: rearrange codes. Accept following command-line arguments:
	- *--train_one_layer*: when true, the script will train the models with **one** hidden layer. Default is False.
	- *--train_two_layer*: when true, the script will train the models with **two** hidden layers. Default is False.
	- *--modified_gd*: when true, some model will adopt modified gradient descenting method. See *layers.py* and report for details. Default is False.
	- *--stop_time*: a hyperparameter for new stop criteria. See report for details.

&#8195;&#8195;Following example shows how to run the two  hidden layer models with modified gradient descenting and a stop time of 5:
  
		$ python run_mlp.py --train_two_layer true --modified_gd true --stop_time 5 