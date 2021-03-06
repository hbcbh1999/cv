import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import numpy as np


class Model(nn.Module):
    def __init__(self, pretrained_net = 'vgg16', D = 1):

        def scale_1():
            if (pretrained_net == 'vgg16'):
                vgg16 = models.vgg16(pretrained = True)


                layers = nn.Sequential()
                
                #set up skip layers
                skip_1_1_output = nn.Sequential()
                skip_1_2_output = nn.Sequential()
                skip_1_1_output.add_module(name = 'sc_1_skip_layer_1_op', module = torch.nn.Conv2d(256,64,5,padding=2))
                skip_1_1_output.add_module(name = 'sc_1_skip_layer_1_relu', module = torch.nn.ReLU())
                skip_1_1_output.add_module(name = 'sc_1_skip_layer_1_upsample', module = torch.nn.UpsamplingBilinear2d(size = (55,75)))
                skip_1_2_output.add_module(name = 'sc_1_skip_layer_2_op', module=torch.nn.Conv2d(512, 64, 5, padding=2))
                skip_1_2_output.add_module(name = 'sc_1_skip_layer_2_relu', module = torch.nn.ReLU())
                skip_1_2_output.add_module(name = 'sc_1_skip_layer_2_upsample', module = torch.nn.UpsamplingBilinear2d(size = (55,75)))
                self.scale_1_skip_1_1_output = skip_1_1_output.cuda(0)
                self.scale_1_skip_1_2_output = skip_1_2_output.cuda(0)
                
                count = 0

                # Extract all vgg_layers to get the skip layers
                for name, module in vgg16.features.named_children():
                    layers.add_module(name = name, module = module)
                    if (isinstance(module, nn.MaxPool2d)):
                        count += 1
                        if (count == 3):
                            self.scale_1_section_1 = layers.cuda(0)
                            layers = nn.Sequential()
                        elif (count == 4):
                            self.scale_1_section_2 = layers.cuda(0)
                            layers = nn.Sequential()

                self.scale_1_section_3 = layers.cuda()

                #tune the first block of vgg to fit dimensions
                self.scale_1_section_1[0].padding = (0,0)
                self.scale_1_section_1[2].padding = (0,0)
                
                #set the fully connected layer
                scale_1_fc = nn.Sequential()              
                scale_1_fc.add_module(name = 'sc_1_fc', module = nn.Linear(7 * 9 * 512, 4125 * D))
                scale_1_fc.add_module(name = 'sc_1_drop', module = nn.Dropout(0.5))
                self.scale_1_fc = scale_1_fc.cuda(0)
                
            else:
                print('Initialization failed, supply the correct parameter {''vgg16''}')
                return

        def scale_2():
           
            
            #scale2 sect.1 : 2.1
            scale_2_section_1 = nn.Sequential()
            scale_2_section_1.add_module(name = '2_1_conv_1',module=torch.nn.Conv2d(3, 96, 9, stride = 1, padding = 0))       
            scale_2_section_1.add_module(name = '2_1_norm_1',module=torch.nn.BatchNorm2d(96))
            scale_2_section_1.add_module(name = '2_1_relu_1',module=torch.nn.ReLU())
            scale_2_section_1.add_module(name = '2_1_pool_1',module=torch.nn.MaxPool2d(2, stride=2, padding=0))
            scale_2_section_1.add_module(name = '2_1_conv_2',module=torch.nn.Conv2d(96, 64, 9, stride = 1, padding = (3,4)))
            scale_2_section_1.add_module(name = '2_1_norm_2',module=torch.nn.BatchNorm2d(64))
            scale_2_section_1.add_module(name = '2_1_relu_2',module=torch.nn.ReLU())    
            scale_2_section_1.add_module(name = '2_1_pool_2',module=torch.nn.MaxPool2d(2, stride=2, padding=0))
            self.scale_2_section_1 = scale_2_section_1.cuda(0)

            #scale2 sect.2 : 2.2 + 2.3
            scale_2_section_2 = nn.Sequential()
            scale_2_section_2.add_module(name = '2_2_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2)) 
            scale_2_section_2.add_module(name = '2_2_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_2.add_module(name = '2_2_relu_1',module=torch.nn.ReLU())
            scale_2_section_2.add_module(name = '2_3_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_2_section_2.add_module(name = '2_3_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_2.add_module(name = '2_3_relu_1',module=torch.nn.ReLU()) 
            self.scale_2_section_2 = scale_2_section_2.cuda(0)

            #scale2 sect.3 : 2.4 + 2.5
            scale_2_section_3 = nn.Sequential()
            scale_2_section_3.add_module(name = '2_4_conv_1',module=torch.nn.Conv2d(64+64, 64, 5, stride = 1, padding = 2))
            scale_2_section_3.add_module(name = '2_4_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_3.add_module(name = '2_4_relu_1',module=torch.nn.ReLU())            
            scale_2_section_3.add_module(name = '2_5_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_2_section_3.add_module(name = '2_5_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_3.add_module(name = '2_5_relu_1',module=torch.nn.ReLU())            
            scale_2_section_3.add_module(name = '2_5_conv_2',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_2_section_3.add_module(name = '2_5_norm_2',module=torch.nn.BatchNorm2d(64))
            scale_2_section_3.add_module(name = '2_5_relu_2',module=torch.nn.ReLU())
            self.scale_2_section_3 = scale_2_section_3.cuda(0)

            #scale2 sect.4 : 2.6 + 2.7 + 2.8
            scale_2_section_4 = nn.Sequential()
            scale_2_section_4.add_module(name = '2_6_conv_1',module=torch.nn.Conv2d(64+D, 64, 5, stride = 1, padding = 2))
            scale_2_section_4.add_module(name = '2_6_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_4.add_module(name = '2_6_relu_1',module=torch.nn.ReLU())            
            scale_2_section_4.add_module(name = '2_7_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_2_section_4.add_module(name = '2_7_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_4.add_module(name = '2_7_relu_1',module=torch.nn.ReLU())            
            scale_2_section_4.add_module(name = '2_8_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_2_section_4.add_module(name = '2_8_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_2_section_4.add_module(name = '2_8_relu_1',module=torch.nn.ReLU())
            self.scale_2_section_4 = scale_2_section_4.cuda(0)
            
            #scale2 sect.5 : 2.9
            scale_2_section_5 = nn.Sequential()
            scale_2_section_5.add_module(name = '2_9_conv_1',module=torch.nn.Conv2d(64, D, 5, stride = 1, padding = 2))
            scale_2_section_5.add_module(name = '2_9_norm_1',module=torch.nn.BatchNorm2d(D))
            scale_2_section_5.add_module(name = '2_9_relu_1',module=torch.nn.ReLU())
            #scale_2_section_5.add_module(name = '2_9_relu_1',module=torch.nn.LeakyReLU())
            self.scale_2_section_5 = scale_2_section_5.cuda(0)

            pass

        def scale_3():
            self.scale3_upsample = nn.Sequential()
            self.scale3_upsample = nn.UpsamplingBilinear2d(size = (112,151))
            #scale3 sect. 1 : 3.1
            scale_3_section_1 = nn.Sequential()
            scale_3_section_1.add_module(name = '3_1_conv_1',module=torch.nn.Conv2d(3, 96, 9, stride = 1, padding = 0)) 
            scale_3_section_1.add_module(name = '3_1_norm_1',module=torch.nn.BatchNorm2d(96))
            scale_3_section_1.add_module(name = '3_1_relu_1',module=torch.nn.ReLU())
            scale_3_section_1.add_module(name = '3_1_pool_1',module=torch.nn.MaxPool2d(2, stride=2, padding=0))            
            #the result is (112,151) different from the paper
            self.scale_3_section_1 = scale_3_section_1.cuda(0)
            
            #scale3 sect. 2 : 3.2 + 3.3
            scale_3_section_2 = nn.Sequential()
            scale_3_section_2.add_module(name = '3_2_conv_1',module=torch.nn.Conv2d(96, 64, 5, stride = 1, padding = 2))
            scale_3_section_2.add_module(name = '3_2_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_3_section_2.add_module(name = '3_2_relu_1',module=torch.nn.ReLU())
            scale_3_section_2.add_module(name = '3_3_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_3_section_2.add_module(name = '3_3_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_3_section_2.add_module(name = '3_3_relu_1',module=torch.nn.ReLU())  
            scale_3_section_2.add_module(name = '3_3_conv_2',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_3_section_2.add_module(name = '3_3_norm_2',module=torch.nn.BatchNorm2d(64))
            scale_3_section_2.add_module(name = '3_3_relu_2',module=torch.nn.ReLU())   
            
            self.scale_3_section_2 = scale_3_section_2.cuda(0)

            #scale3 sect. 3 : 3.4 ~ 3.5
            scale_3_section_3 = nn.Sequential() 
            scale_3_section_3.add_module(name = '3_4_conv_1',module=torch.nn.Conv2d(64+D, 64, 5, stride = 1, padding = 2))
            scale_3_section_3.add_module(name = '3_4_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_3_section_3.add_module(name = '3_4_relu_1',module=torch.nn.ReLU())    
            scale_3_section_3.add_module(name = '3_5_conv_1',module=torch.nn.Conv2d(64, 64, 5, stride = 1, padding = 2))
            scale_3_section_3.add_module(name = '3_5_norm_1',module=torch.nn.BatchNorm2d(64))
            scale_3_section_3.add_module(name = '3_5_relu_1',module=torch.nn.ReLU())
            self.scale_3_section_3 = scale_3_section_3.cuda(0)
            
            #scale3 sect. 4 : 3.4 ~ 3.5
            scale_3_section_4 = nn.Sequential()
            scale_3_section_4.add_module(name = '3_6_conv_1',module=torch.nn.Conv2d(64, D, 5, stride = 1, padding = 2))
            scale_3_section_4.add_module(name = '3_6_relu_1',module=torch.nn.ReLU())
            self.scale_3_section_4 = scale_3_section_4.cuda(0)

            pass

        super(Model, self).__init__()
        self.D = D

        scale_1()
        scale_2()
        scale_3()
        
    def forward(self, x, scale3 = False):
        # scale_1
    
        sc_1_main_op = self.scale_1_section_1(x)
        sc_1_skip_1_1 = self.scale_1_skip_1_1_output(sc_1_main_op)
        sc_1_main_op = self.scale_1_section_2(sc_1_main_op)
        sc_1_skip_1_2 = self.scale_1_skip_1_2_output(sc_1_main_op)
        sc_1_main_op = self.scale_1_section_3(sc_1_main_op)
        sc_1_main_op=sc_1_main_op.view(sc_1_main_op.size(0),-1)
        sc_1_main_op = self.scale_1_fc(sc_1_main_op)
        sc_1_main_op = sc_1_main_op.view(-1,1,55,75)
        
        #print "====================scale2======================"
        # forward pass for scale 2, 
        # reuse x, x_skip_1_1, x_skip_1_2, xout
        sc_2_main_op = self.scale_2_section_1(x)
        sc_2_main_op += sc_1_skip_1_1
        sc_2_main_op = self.scale_2_section_2(sc_2_main_op)
        sc_2_main_op =torch.cat((sc_2_main_op,sc_1_skip_1_2),dim = 1)
        sc_1_skip_1_2 = []
        sc_2_main_op = self.scale_2_section_3(sc_2_main_op)
        sc_2_main_op = torch.cat((sc_2_main_op, sc_1_main_op), dim = 1)
        sc_1_main_op = []
        sc_2_main_op = self.scale_2_section_4(sc_2_main_op)
        sc_2_main_op = self.scale_2_section_5(sc_2_main_op)
        
        if(scale3):
            #print "====================scale3======================"
            # forward pass for scale 3, 
            # reuse x, x_scale2
            sc_3_main_op = self.scale_3_section_1(x)
            sc_3_main_op = self.scale_3_section_2(sc_3_main_op)
            sc_2_main_op = self.scale3_upsample(sc_2_main_op)
            sc_3_main_op = torch.cat((sc_3_main_op, sc_2_main_op), dim = 1)
            sc_3_main_op = self.scale_3_section_3(sc_3_main_op)
            sc_3_main_op = self.scale_3_section_4(sc_3_main_op)
        else:
            sc_3_main_op = sc_2_main_op
            
        return sc_3_main_op

    
