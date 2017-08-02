import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Trainer(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    
    def __init__(self, model, optim = torch.optim.Adam, loss_func = torch.nn.MSELoss(), args_adam = {}):
        self.model = model
        args_update = self.default_adam_args.copy()
        args_update.update(args_adam)
        self.optim_sc2 = optim
        self.optim_sc3 = optim
        self.loss_func = loss_func
        self.optim_args = args_update
        weight_decay = 0.0001
        #weight_decay = 0.0005
        self.optim_list_sc2 = [{'params':model.scale_1_section_1.parameters(),'lr': 1e-4, 'weight_decay': weight_decay},
                           {'params':model.scale_1_section_2.parameters(),'lr': 1e-4, 'weight_decay': weight_decay},
                           {'params':model.scale_1_section_3.parameters(),'lr': 1e-4, 'weight_decay': weight_decay},
                           {'params':model.scale_1_fc.parameters(),'lr': 1e-1, 'weight_decay': weight_decay},
                           {'params':model.scale_1_skip_1_1_output.parameters(),'lr': 1e-2, 'weight_decay': weight_decay},
                           {'params':model.scale_1_skip_1_2_output.parameters(),'lr': 1e-2, 'weight_decay': weight_decay},
                           {'params':model.scale_2_section_1.parameters(),'lr': 1e-3, 'weight_decay': weight_decay},
                           {'params':model.scale_2_section_2.parameters(),'lr': 1e-2, 'weight_decay': weight_decay},
                           {'params':model.scale_2_section_3.parameters(),'lr': 1e-2, 'weight_decay': weight_decay},
                           {'params':model.scale_2_section_4.parameters(),'lr': 1e-2, 'weight_decay': weight_decay},
                           {'params':model.scale_2_section_5.parameters(),'lr': 1e-2, 'weight_decay': weight_decay}]
        
        self.optim_list_sc3=[{'params':model.scale_3_section_1.parameters(),'lr': 1e-3, 'weight_decay': 0.001},
                             {'params':model.scale_3_section_2.parameters(),'lr': 1e-2, 'weight_decay': 0.001},
                             {'params':model.scale_3_section_3.parameters(),'lr': 1e-2, 'weight_decay': 0.001},
                             {'params':model.scale_3_section_4.parameters(),'lr': 1e-2, 'weight_decay': 0.001}]
        


    def train(self, train_loader, val_loader, train_loss_history, train_rel_history, val_rel_history, 
        num_epochs = 10, nth = 1, lr_decay = 0.1):
    
        model = self.model
        optim = self.optim_sc2(self.optim_list_sc2, **self.optim_args)
               
        optim.zero_grad()
        total_it_epo = len(train_loader)
        total_it = total_it_epo * num_epochs[0]
        
        n = 55 * 75
        counter = 0
        print("training scale2 started...")
        
        for epoch in range(num_epochs[0]):
            for _, (xbatch, ybatch) in enumerate(train_loader):
                x = Variable(xbatch.cuda(0))
                y = Variable(ybatch.cuda(0))
                x_out = model.forward(x)
                d = torch.abs(x_out - y)
                regu = torch.pow(torch.sum(torch.sum(d,dim = 2), dim = 3),2)/(n*n)

                #loss = torch.sum(torch.sum(torch.pow(d, 2),dim = 2), dim = 3)/n + 0.5*regu              
                #loss = torch.sum(loss, dim = 0)/10
                #print loss
                loss = self.loss_func(x_out, y) + 0.5 * torch.sum(regu)/10
                train_loss_history.append(loss.type(torch.FloatTensor).data.numpy())
                optim.zero_grad()
                loss.backward()
                optim.step()
                counter += 1
                if (counter % nth == 0):
                     print '[Iteration'+ str(counter)+'/'+str(total_it) +']'+ 'TRAIN loss:' + str(train_loss_history[-1])
                    
                     #Compute training relative error per epoch       
               	     y_pred = model.forward(x)
                     y_pred = y_pred.type(torch.FloatTensor).data.numpy() 
                     y_gt = y.type(torch.FloatTensor).data.numpy()
                     train_rel_history.append(np.mean(np.absolute(y_pred - y_gt)/y_gt))
                
                if (counter % nth == 0):
                    print '[Epoch' + str(epoch) + '/' + str(num_epochs[0]) + '] Train  error:'+ str(train_rel_history[-1])
            
            #Compute validation relative error per epoch
            val = []
            for _, (xbatch, ybatch) in enumerate(val_loader):
                x_val = Variable(xbatch.cuda(0))
                y_pred = model.forward(x_val)
                y_pred = y_pred.type(torch.FloatTensor).data.numpy() 
                y_gt = ybatch.numpy()
                val.append(np.mean(np.absolute(y_pred - y_gt)/y_gt))
            val_rel_history.append(np.mean(val))
                
            print '[Epoch' + str(epoch) + '/' + str(num_epochs[0]) + '] VAL  error:'+ str(val_rel_history[-1])
            
            #adapt learning rates
            for param_group in optim.param_groups:
                param_group['lr'] *= lr_decay
            
        print("scale2 training complete..")
        model.save("/usr/prakt/s243/DL4CV_ML/dlcv_proj/dlcv_proj/RGB_Depth_sc2.model")
            
        
        print("training scale3 started...")
 
        
        optim = []
        optim = torch.optim.Adam(self.optim_list_sc3)
        optim.zero_grad() 
        
        for epoch in range(num_epochs[1]):
            for _, (xbatch, ybatch) in enumerate(train_loader):
                x = Variable(xbatch.cuda(0))
                y = Variable(ybatch.cuda(0))
                y = F.upsample_bilinear(y, size=(112,151))
                x_out = model.forward(x, scale3 = True)
                loss = self.loss_func(x_out, y) #dimension of y must be changed
                train_loss_history.append(loss.type(torch.FloatTensor).data.numpy())
                optim.zero_grad()
                loss.backward()
                optim.step()
                counter += 1
                if (counter % nth == 0):
                    print '[Iteration'+ str(counter)+'/'+str(total_it) +']'+ 'TRAIN loss:' + str(train_loss_history[-1])
                    
                #Compute training relative error per epoch       
                y_pred = model.forward(x, scale3 = True)
                y_pred = y_pred.type(torch.FloatTensor).data.numpy() 
                y_gt = y.type(torch.FloatTensor).data.numpy()
                train_rel_history.append(np.mean(np.absolute(y_pred - y_gt)/y_gt))
                if (counter % nth == 0):
                    print '[Epoch' + str(epoch) + '/' + str(num_epochs[1]) + '] Train  error:'+ str(train_rel_history[-1])
            
            #Compute validation relative error per epoch
            val = []
            for _, (xbatch, ybatch) in enumerate(val_loader):
                x_val = Variable(xbatch.cuda(0))
                y_gt = Variable(ybatch.cuda(0))
                y_pred = model.forward(x_val, scale3 = True)
                y_pred = y_pred.type(torch.FloatTensor).data.numpy()
                y_gt = F.upsample_bilinear(y_gt, size=(112,151))
                y_gt = y_gt.type(torch.FloatTensor).data.numpy() 
                val.append(np.mean(np.absolute(y_pred - y_gt)/y_gt))
            val_rel_history.append(np.mean(val))
            print '[Epoch' + str(epoch) + '/' + str(num_epochs[1]) + '] VAL  error:'+ str(val_rel_history[-1])
            
            #adapt learning rates
            for param_group in optim.param_groups:
                param_group['lr'] *= lr_decay   
                
        print("scale2 training complete..")
        model.save("/usr/prakt/s243/DL4CV_ML/dlcv_proj/dlcv_proj/RGB_Depth_sc3.model")
            
