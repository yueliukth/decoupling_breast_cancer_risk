Implementation of paper ‘Decoupling Inherent Risk and Early Cancer Signs in Image-based Breast Cancer Risk Models” in Tensorflow.

	preprocessing.py		Mammogram preprocessing from dicom to 16-bit png
	tfbuilder.py 			Store input data in TFRecord format
	main_run.py 			Execute the main script 

Scripts to support the main execution script ‘main_run.py’
	exp.py 				Define model hyperparameters 
	train_network.py		Define training process
	eval_network.py			Define testing process
	dataset_functions.py		Define dataset helping functions
	train_functions.py		Define training helping functions
	model_functions.py		Define ResNet model

