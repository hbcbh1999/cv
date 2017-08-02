from architecture.model_sanyu import Model
from architecture.trainer_sanyu import Trainer
from helper.rmrc import rmrcData
from helper.rmrc import Rmrc
import numpy as np
import torch

if __name__ == "__main__":
    	torch.cuda.set_device(0)
    	print('Chosen GPU device: ' + str(torch.cuda.current_device()))

    
    	rmrc_data_path = "/usr/prakt/s243/DL4CV_ML/dlcv_proj"
	#adapte to notebook
	file_path_v2 = "/usr/prakt/s243/DL4CV_ML/dlcv_proj/nyu_depth_v2_labeled"
	dataset = Rmrc(data_path = rmrc_data_path)
	data_v2 = dataset.read_rmrc_data(file_path_v2)
	file_path_v1 = "/usr/prakt/s243/DL4CV_ML/dlcv_proj/nyu_depth_data_labeled"
	data_v1 = dataset.read_rmrc_data(file_path_v1)    
	images = data_v2['images']
	depths = data_v2['depths']

	#images = np.append(images, data_v1['images'], axis = 0)
	#depths = np.append(depths, data_v1['depths'], axis = 0)
	depths = np.expand_dims(depths, axis=1)
	print images.shape
	print depths.shape

	#num_training = 3632
	num_training = 1348
	num_validation = 100
	#num_test = 100
	mask = range(num_training)
	X_train = images[mask]
	y_train = depths[mask]
	mask = range(num_training, num_training + num_validation)
	X_val = images[mask]
	y_val = depths[mask]

	train_data = rmrcData(X_train, y_train)
	val_data = rmrcData(X_val, y_val)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=1)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size = 10, shuffle=False, num_workers=1)

	train_loss_history = []
	train_error_history = []
	val_error_history = []

	model = Model()
	args_adam = {"lr": 1e-3,
             "betas": (0.9, 0.999),
             "eps": 1e-8,
             "weight_decay": 0.0005}

	solver = Trainer(model, args_adam = args_adam)
	solver.train(train_loader, val_loader, train_loss_history, train_error_history, val_error_history, 
             num_epochs = (200, 100),nth= 500, lr_decay = 0.9851)




