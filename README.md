**Repository for MICCAI submission**

**"Decoupling Inherent Risk and Early Cancer Signs in Image-based Breast Cancer Risk Models”.** 

![alt text](https://github.com/yueliukth/decoupling_breast_cancer/master/main_plot.png?raw=true)

Implemented in [TensorFlow](https://www.tensorflow.org/).

Main scripts

	preprocessing.py		Mammogram preprocessing from dicom to 16-bit png
	tfbuilder.py 			Store input data in TFRecord format
	main_run.py 			Execute the main scripts 

Scripts to support the main execution script ‘main_run.py’

	exp.py 				Define model hyperparameters 
	train_network.py		Define training process
	eval_network.py		        Define testing process
	dataset_functions.py	        Define dataset helping functions
	train_functions.py		Define training helping functions
	model_functions.py		Define ResNet model
