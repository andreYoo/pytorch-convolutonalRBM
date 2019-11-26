import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
import pdb

class ConvRBM():#Convolusional RBM : The RBM for real value units

    def __init__(self, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,num_visible=0, num_hidden=0,batch_size=64,  use_cuda=True):
        #self.num_visible = num_visible
        #self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.k = k #Step for CD
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda


        self.conv1_weights = nn.Parameter(torch.randn(32,1,4,4))
        self.conv1_visible_bias = nn.Parameter(torch.randn(1))
        self.conv1_hidden_bias = nn.Parameter(torch.randn(32))
        self.maxpool1 = nn.MaxPool2d(3,stride=2,return_indices=True)


        self.conv1_weights_momentum = torch.zeros(32,1,4,4)
        self.conv1_visible_bias_momentum = torch.zeros(1)
        self.conv1_hidden_bias_momentum = torch.zeros(32)
        self.upsample1 = nn.MaxUnpool2d(2, stride=1)


        if self.use_cuda:
            self.conv1_weights = self.conv1_weights.cuda()
            self.conv1_visible_bias = self.conv1_visible_bias.cuda()
            self.conv1_hidden_bias = self.conv1_hidden_bias.cuda()


            self.conv1_weights_momentum = self.conv1_weights_momentum.cuda()
            self.conv1_visible_bias_momentum = self.conv1_visible_bias_momentum.cuda()
            self.conv1_hidden_bias_momentum = self.conv1_hidden_bias_momentum.cuda()



    def sample_hidden(self, visible_probabilities):
        out1 = F.conv2d(visible_probabilities,weight=self.conv1_weights,bias=self.conv1_hidden_bias)
        out1 = F.leaky_relu(out1)
        self.out1,self.out_idc = self.maxpool1(out1)


        hidden_probabilities = self.out1
        return hidden_probabilities #output

    def sample_visible(self, hidden_probabilities):
        de_output1 = self.upsample1(hidden_probabilities,self.out_idc)
        de_output1 = F.conv_transpose2d(de_output1,weight=self.conv1_weights,bias=self.conv1_visible_bias,stride=2,padding=0)
        self.de_output1 = F.leaky_relu(de_output1)

        visible_probabilities = self.de_output1

        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)  #up
        pose_scale = positive_hidden_probabilities.size()


        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(pose_scale)).float() #Noise sampleing?
        positive_associations = torch.matmul(input_data.view(self.batch_size,784).t(), positive_hidden_activations.view(self.batch_size,pose_scale[1]*pose_scale[2]*pose_scale[3])) #Back_down?

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities= self.sample_hidden(visible_probabilities)
            neg_scale = hidden_probabilities.size()

            hidden_activations = (hidden_probabilities >= self._random_probabilities(neg_scale)).float()

        negative_visible_probabilities = visible_probabilities

        neg_vis_scale = negative_visible_probabilities.size()



        negative_hidden_probabilities = hidden_probabilities

        neg_hid_scale = negative_hidden_probabilities.size()


        negative_associations1 = torch.matmul(negative_visible_probabilities.view(self.batch_size,neg_vis_scale[1]*neg_vis_scale[2]*neg_vis_scale[3]).t(), negative_hidden_probabilities.view(self.batch_size,neg_hid_scale[1]*neg_hid_scale[2]*neg_hid_scale[3]))


        # Update parameters
        self.conv1_weights_momentum *= self.momentum_coefficient
        self.conv1_weights_momentum += (1./(144.*49.))*torch.sum((positive_associations.view(144,49,32,1,4,4) - negative_associations1.view(144,49,32,1,4,4)),dim=(0,1))

        self.conv1_visible_bias_momentum *= self.momentum_coefficient
        vis_bm_tmp = torch.sum(input_data - negative_visible_probabilities, dim=(0,1,2,3))
        self.conv1_visible_bias_momentum += (1.0/(64.*28.*28.))*vis_bm_tmp

        self.conv1_hidden_bias_momentum *= self.momentum_coefficient
        hidden_bm_tmp = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=(0, 2, 3))
        self.conv1_hidden_bias_momentum += (1.0/(64.*12.*12.))*hidden_bm_tmp

        batch_size = input_data.size(0)

        self.conv1_weights += self.conv1_weights_momentum * self.learning_rate / batch_size
        self.conv1_visible_bias += self.conv1_visible_bias_momentum * self.learning_rate / batch_size
        self.conv1_hidden_bias += self.conv1_hidden_bias_momentum * self.learning_rate / batch_size
        self.conv1_weights -= self.conv1_weights * self.weight_decay  # L2 weight decay



        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities


