import torch
import torch.nn as nn
import torch.nn.functional as F


class network(nn.Module):

    def __init__(self):
        super(network, self).__init__()

        # self.zero_pad = nn.ZeroPad2d((3))
        self.relu = nn.ReLU()

        self.batch_norm = nn.BatchNorm2d(32)

        self.max_pool = nn.MaxPool2d((4,4))
        # self.max_pool_1 = nn.MaxPool2d((3,3))
        
        self.conv_9 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(9,9), stride=(1,1), padding=(4,4))
        self.conv_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv_7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7,7), stride=(1,1), padding=(3,3))

        self.fc0 = nn.Linear(in_features=288, out_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

        self.dropout_0 = nn.Dropout(0.3)
        # self.dropout_1 = nn.Dropout(0.2)

        self.apply(self._init_weights)
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        # print('Weights initialized successfully!')


    def forward(self, x):

        # identity = x
        # print(x.shape)
        # x = self.zero_pad(x)
        # print('ok pool', x.shape)
        x_init = x
        # print(x_init.shape[2:])
        x = self.conv_9(x)
        assert x_init.shape[2:] == x.shape[2:] # for same convolution
        # # print('ok conv0', x.shape)

        x = self.batch_norm(x)
        # # print('ok bn', x.shape)
        x = self.relu(x)
        # # print('ok relu', x.shape)

        x = self.max_pool(x)
        # # print('ok max pool 0', x.shape)

        # # x = self.zero_pad(x)
        x_init_2 = x
        x = self.conv_7(x)
        assert x_init_2.shape[2:] == x.shape[2:]
        # # print('ok conv0', x.shape)

        x = self.batch_norm(x)
        # # print('ok bn', x.shape)
        x = self.relu(x)
        # # print('ok relu', x.shape)

        x = self.max_pool(x)

        x_init_3 = x
        x = self.conv_5(x)
        assert x_init_3.shape[2:] == x.shape[2:]
        # # print('ok con1', x.shape)
        x = self.batch_norm(x)
        # # print('ok bn1', x.shape)
        x = self.relu(x)
        # # print('ok relu', x.shape)
        x = self.max_pool(x)
        # # print(x.shape)
        # # print('ok maxpool', x.shape)

        x = torch.flatten(x, start_dim=1)
        # print('flatten')
        x = self.fc0(x)
        x = self.relu(x)

        # print('fc0')
        x = self.dropout_0(x)
        # print('dropout')
        # x = self.relu(x)
        # print('relu')

        x = self.fc1(x)
        # print('fc1')
        x = self.relu(x)
        # print('relu')
        x = self.fc2(x)
        # print('fc2')


        return x

    