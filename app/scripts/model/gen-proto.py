import sys, os
#sys.path.insert(0, 'path/to/caffe/python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import numpy as np

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
        param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def full_conv(bottom, name, lr):
    return L.Convolution(bottom, name=name, kernel_size=1,num_output=1,# weight_filler=dict(type='xavier'),
        param=[dict(lr_mult=0.01*lr, decay_mult=1), dict(lr_mult=0.02*lr, decay_mult=0)])

def fcn(split):
    n = caffe.NetSpec()
    n.data = L.Input(name = 'data', input_param=dict(shape=dict(dim=[1,3,500,500])))
    if split=='train':
        n.label = L.Input(name='label', input_param=dict(shape=dict(dim=[1,1,500,500])))
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)
    
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, mult=[100,1,200,0])
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, mult=[100,1,200,0])
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, mult=[100,1,200,0])
    
    # DSN1
    n.score_dsn1=full_conv(n.conv1_2, 'score-dsn1', lr=1)
    n.upscore_dsn1 = crop(n.score_dsn1, n.data)
    if split=='train':
        n.loss1 = L.SigmoidCrossentropyLoss(n.upscore_dsn1, n.label)
    if split=='test':
        n.sigmoid_dsn1 = L.Sigmoid(n.upscore_dsn1)
    # n.sigmoid_dsn1 = L.Sigmoid(n.upscore_dsn1)
    
    # DSN2
    n.score_dsn2=full_conv(n.conv2_2, 'score-dsn2', lr=1)
    n.score_dsn2_up = L.Deconvolution(n.score_dsn2, name='upsample_2', 
        convolution_param=dict(num_output=1, kernel_size=4, stride=2),
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.upscore_dsn2 = crop(n.score_dsn2_up, n.data)
    if split=='train':
        n.loss2 = L.SigmoidCrossentropyLoss(n.upscore_dsn2, n.label)
    if split=='test':
        n.sigmoid_dsn2 = L.Sigmoid(n.upscore_dsn2)
    # n.sigmoid_dsn2 = L.Sigmoid(n.upscore_dsn2)
    
    # DSN3
    n.score_dsn3=full_conv(n.conv3_3, 'score-dsn3', lr=1)
    n.score_dsn3_up = L.Deconvolution(n.score_dsn3, name='upsample_4', 
        convolution_param=dict(num_output=1, kernel_size=8, stride=4),
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.upscore_dsn3 = crop(n.score_dsn3_up, n.data)
    if split=='train':
        n.loss3 = L.SigmoidCrossentropyLoss(n.upscore_dsn3, n.label)
    if split=='test':
        n.sigmoid_dsn3 = L.Sigmoid(n.upscore_dsn3)
    # n.sigmoid_dsn3 = L.Sigmoid(n.upscore_dsn3)
    
    # DSN4
    n.score_dsn4=full_conv(n.conv4_3, 'score-dsn4', lr=1)
    n.score_dsn4_up = L.Deconvolution(n.score_dsn4, name='upsample_8', 
        convolution_param=dict(num_output=1, kernel_size=16, stride=8),
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.upscore_dsn4 = crop(n.score_dsn4_up, n.data)
    if split=='train':
        n.loss4 = L.SigmoidCrossentropyLoss(n.upscore_dsn4, n.label)
    if split=='test':
        n.sigmoid_dsn4 = L.Sigmoid(n.upscore_dsn4)
    # n.sigmoid_dsn4 = L.Sigmoid(n.upscore_dsn4)

    # DSN5
    n.score_dsn5=full_conv(n.conv5_3, 'score-dsn5', lr=1)
    n.score_dsn5_up = L.Deconvolution(n.score_dsn5, name='upsample_16', 
        convolution_param=dict(num_output=1, kernel_size=32, stride=16),
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.upscore_dsn5 = crop(n.score_dsn5_up, n.data)
    if split=='train':
        n.loss5 = L.SigmoidCrossentropyLoss(n.upscore_dsn5, n.label)
    if split=='test':
        n.sigmoid_dsn5 = L.Sigmoid(n.upscore_dsn5)
    # n.sigmoid_dsn5 = L.Sigmoid(n.upscore_dsn5)

    # concat and fuse
    n.concat_upscore = L.Concat(n.upscore_dsn1,
                        n.upscore_dsn2,
                        n.upscore_dsn3,
                        n.upscore_dsn4,
                        n.upscore_dsn5,
                        name='concat', concat_param=dict({'concat_dim':1}))
    n.upscore_fuse = L.Convolution(n.concat_upscore, name='new-score-weighting', 
                   num_output=1, kernel_size=1,
                   param=[dict(lr_mult=0.001, decay_mult=1), dict(lr_mult=0.002, decay_mult=0)],
                   weight_filler=dict(type='constant', value=0.2))
    if split=='test':
        n.sigmoid_fuse = L.Sigmoid(n.upscore_fuse)
    if split=='train':
        n.loss_fuse = L.SigmoidCrossentropyLoss(n.upscore_fuse, n.label)

    return n.to_proto()

def make_net():
    with open('hed_train.pt', 'w') as f:
        f.writelines(os.linesep+'force_backward: true'+os.linesep)
        f.write(str(fcn('train')))

    with open('hed_test.pt', 'w') as f:
        f.write(str(fcn('test')))
def make_solver():
    sp = {}
    sp['net'] = '"train.pt"'
    sp['base_lr'] = '0.001'
    sp['lr_policy'] = '"step"'
    sp['momentum'] = '0.9'
    sp['weight_decay'] = '0.0002'
    sp['iter_size'] = '10'
    sp['stepsize'] = '1000'
    sp['display'] = '20'
    sp['snapshot'] = '100000'
    sp['snapshot_prefix'] = '"net"'
    sp['gamma'] = '0.1'
    sp['max_iter'] = '100000'
    sp['solver_mode'] = 'CPU'
    f = open('solver.pt', 'w')
    for k, v in sorted(sp.items()):
        if not(type(v) is str):
            raise TypeError('All solver parameters must be strings')
        f.write('%s: %s\n'%(k, v))
    f.close()
if __name__ == '__main__':
    make_net()
    make_solver()