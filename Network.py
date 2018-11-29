import numpy as np
import h5py

class relu:
    ##this layer used as a non-linear transformer
    ##output has the same shape as input
    def __init__(self, name, k=1.0):  ## output = k*input when input>=0;else output=0 ; default k = 1.0
        self.name = name
        self.type = 'relu'
        self.input = np.zeros([1, 1, 1, 1])
        self.input_delta = np.zeros([1, 1, 1, 1])
        self.forward_output = np.zeros([1, 1, 1, 1])
        self.k = k
        self.reconstruct_input = np.zeros([1, 1, 1, 1])
        self.layer_id = 0
        self.pre_connect_layer_id = []
        self.post_connect_layer_id = []
        self.first_forward = True
        self.BP_flag = True
        self.first_bp = True

    def set_layer_id(self, id):
        self.layer_id = id
        if (id > 1):
            self.pre_connect_layer_id.append((id - 1))
        self.post_connect_layer_id.append((id + 1))

    def add_pre_connect_layer(self, id):
        self.pre_connect_layer_id.append(id)

    def add_post_connect_layer(self, id):
        self.post_connect_layer_id.append(id)

    def forward(self, input):  ##
        if len(input.shape) == 4:  ##input coming from conv or pooling layers
            # print "%s input is blob"%self.name
            pass
        elif len(input.shape) == 2:  ##input coming from fc layer
            # print "%s input is matrix"%self.name
            pass
        self.input = input.copy()
        #temp = 0.5 * self.k * np.add(input, np.abs(input))
        self.input[self.input<0] = 0
        self.forward_output = self.input
        if self.first_forward == True:
            # print "Layer %s" % (self.name), "input shape", input.shape, "->output shape", self.forward_output.shape
            self.first_forward = False
        return self.forward_output

    def backward(self, output_delta):
        if self.BP_flag == True:
            if self.first_bp == True:
                # print self.name,'need BP'
                self.first_bp = False
            assert output_delta.shape == self.input.shape
            temp = self.forward_output.copy()
            temp[temp > 0] = 1
            self.input_delta = np.multiply(output_delta, temp)
            return self.input_delta

    def update(self, lr):
        pass

    def switch(self, feature_map):
        assert feature_map.shape == self.input.shape
        temp = self.forward_output.copy()
        temp[temp > 0] = 1
        self.reconstruct_input = np.multiply(feature_map, temp)
        return self.reconstruct_input

class softmax:
    ##this layer has format:[num of sample,output dimension],output dimension means the total number of output neurons
    # Type of loss layer can be squared loss for regression or softmax loss(log loss) for classification
    def __init__(self, name):
        self.name = name
        self.type = 'softmax'
        self.forward_output = np.zeros([1, 1])
        self.input_delta = np.zeros([1, 1])
        self.label = 0
        self.layer_id = 0
        self.loss = 0
        self.first_forward = True
        self.first_bp = True
        self.BP_flag = True

    def set_layer_id(self, id):
        self.layer_id = id

    def get_loss(self):
        return self.loss

    def forward(self, input, label):
        num_sample, w = input.shape
        temp_result_matrix = np.exp(input)
        denominator_col = np.reshape(np.sum(temp_result_matrix, 1), [input.shape[0], 1])
        denominator_matrix = np.repeat(denominator_col, input.shape[1], 1)
        assert denominator_matrix.shape == temp_result_matrix.shape
        #denominator_matrix = np.add(denominator_matrix, 1e-15)
        ##prediction range [0,1]
        self.predict = np.divide(temp_result_matrix, denominator_matrix)
        ##compute loss
        self.label = label
        cost_matrix = -np.multiply(np.log(self.predict), self.label)
        total_loss = np.sum(cost_matrix)
        self.loss = np.divide(total_loss, num_sample)
        if self.first_forward == True:
            # print "Layer %s" % (self.name), "input shape", input.shape, "->output shape", self.predict.shape
            self.first_forward = False
        return self.loss, self.predict
        ##only the target cost taken into account , which label index should be 1

    def backward(self):
        ##backward
        ##error term should be output - label  http://www.cnblogs.com/tornadomeet/p/3468450.html
        if self.first_bp == True:
            # print self.name,'need BP'
            self.first_bp = False
        num_sample, w = self.label.shape
        self.input_delta = np.subtract(self.predict, self.label)
        self.input_delta = np.divide(self.input_delta, num_sample)
        return self.input_delta

    def update(self, lr):
        pass

class fc:
    ##fully connected layer
    ##this layer can have blob or matrix as input,
    ##but the input blob should be reshape to matrix for matrix operations
    ##that is, self.original_input should be reshape to matrix as needed

    def __init__(self, name, EWC_flag=False, EWC_lam=0):
        self.name = name
        self.EWC_flag = EWC_flag
        self.EWC_lam = EWC_lam
        self.type = 'fc'
        self.input_shape = [1, 1, 1, 1]  ##this could be blob or matrix
        self.input = np.zeros([1, 1])  # default is matrix
        self.weight_matrix = np.zeros([1, 1])
        self.forward_output = np.zeros([1, 1])
        self.bias = np.zeros([1, 1])
        self.output_dim = 0
        self.weight_gradient_matrix = np.zeros([1, 1])
        self.input_delta = np.zeros(
            [1, 1])  # this could be blob or matrix,which should have the same shape as original input
        self.bias_gradient = np.zeros([1, 1])
        self.init_weight_flag = False
        self.init_bias_flag = False
        self.layer_id = 0
        self.pre_connect_layer_id = []
        self.post_connect_layer_id = []
        self.first_forward = True
        self.mean = 0
        self.std = 0.05
        self.L2 = False
        self.lambda_ = 0
        self.BP_flag = True
        self.first_bp  = True

    def init_fisher_params(self, old_weight_grad=None, old_weight=None, old_bias_grad=None, old_bias=None):
        self.old_weight_grad = old_weight_grad
        self.old_weight = old_weight
        self.old_bias_grad = old_bias_grad
        self.old_bias = old_bias

    def set_layer_id(self, id):
        self.layer_id = id
        if (id > 1):
            self.pre_connect_layer_id.append((id - 1))
        self.post_connect_layer_id.append((id + 1))

    def add_pre_connect_layer(self, id):
        self.pre_connect_layer_id.append(id)

    def add_post_connect_layer(self, id):
        self.post_connect_layer_id.append(id)

    def set_weight_matrix(self, weight):  ##set weight matrix from hdf5 file
        self.weight_matrix = weight.value
        # print "set %s layer weight matrix shape is" % self.name, weight.shape
        self.init_weight_flag = True

    def set_bias_matrix(self, bias):
        self.bias = bias.value
        # print "set %s layer bias matrix shape is" % self.name, bias.shape
        self.init_bias_flag = True

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    def set_init_params(self, mean, std):
        self.mean = mean
        self.std = std

    def set_L2_regularization(self, lambda_):
        self.L2 = True
        self.lambda_ = lambda_

    def get_L2_cost(self):
        temp = np.power(self.weight_matrix, 2)
        sum_temp = np.sum(temp)
        return sum_temp

    def forward(self, input, first=False):
        self.input_shape = input.shape
        if self.init_weight_flag == False:  ##we need to initialize the fc weight matrix in the first time
            if len(self.input_shape) == 4:
                # print "FC input is blob [num,channels,h,w]"
                num, channels, h, w = self.input_shape
                self.weight_matrix = np.random.normal(self.mean, self.std, [channels * h * w, self.output_dim])
            elif len(self.input_shape) == 2:
                # print "FC input is matrix [num,size]"
                self.weight_matrix = np.random.normal(self.mean, self.std, [input.shape[1], self.output_dim])
            else:
                # dft 181122
                # raise TypeError, ('The input format of FC should be blob or matrix!')
                print('The input format of FC should be blob or matrix!')
                return 1
            # print "%s initialization done!"%self.name
            self.init_weight_flag = True
        ##check if input is blob or matrix
        if len(self.input_shape) == 4:
            ##change blob format[num,channels,h,w] into kernel-matrix [num,channels*h*w]
            self.input = np.reshape(input, [input.shape[0], input.shape[1] * input.shape[2] * input.shape[3]]);
            ##add bias:
            self.forward_output = np.dot(self.input, self.weight_matrix)
            # assert self.forward_output.shape == (self.input.shape[0],self.output_dim)
        elif len(self.input_shape) == 2:
            self.input = input.copy()
            self.forward_output = np.dot(self.input, self.weight_matrix)
        else:
            # dft 181122
            # raise TypeError, ('The input format of FC should be blob or matrix!')
            print('The input format of FC should be blob or matrix!')
            return 1
        if self.first_forward == True:
            # print "Layer %s" % (self.name), "input shape", input.shape, "->output shape", self.forward_output.shape, \
            #     "|||weight matrix shape ", self.weight_matrix.shape
            self.first_forward = False
        if self.init_bias_flag == False:
            self.bias = np.zeros([1,self.output_dim])
            self.init_bias_flag = True
        self.forward_output = np.add(self.forward_output, self.bias)
        return self.forward_output

    def backward(self, output_delta):
        if self.BP_flag == True:
            if self.first_bp == True:
                self.first_bp = False

            assert output_delta.shape == self.forward_output.shape
            self.bias_gradient = np.sum(output_delta,0)
            self.bias_gradient = np.reshape(self.bias_gradient,self.bias.shape)
            self.weight_gradient_matrix = np.dot(np.transpose(self.input), output_delta)
            if self.L2 == True:
                L2_weight_gradient = self.lambda_ * self.weight_matrix
                self.weight_gradient_matrix = np.add(self.weight_gradient_matrix, L2_weight_gradient)
            assert self.weight_gradient_matrix.shape == self.weight_matrix.shape
            input_delta_matrix = np.dot(output_delta, np.transpose(self.weight_matrix))
            assert input_delta_matrix.shape == self.input.shape
            if len(self.input_shape) == 2:
                self.input_delta = input_delta_matrix
            elif len(self.input_shape) == 4:
                self.input_delta = np.reshape(input_delta_matrix, self.input_shape)
            else:
                # dft 181122
                # raise TypeError, ('The input format of FC should be blob or matrix!')
                print('The input format of FC should be blob or matrix!')
                return 1

            return self.input_delta

    def update(self, lr):
        if self.BP_flag ==True:
            if self.EWC_flag == True:
                weight_gradient = np.add(self.EWC_lam * self.old_weight_grad * (self.weight_matrix - self.old_weight),
                                         self.weight_gradient_matrix)
                L2_weight_gradient = np.power(weight_gradient, 2)
                L2_weight_gradient_sum_sqrt = np.sqrt(np.sum(L2_weight_gradient))
                threshold = 1
                if L2_weight_gradient_sum_sqrt > threshold:
                    self.weight_gradient_matrix = threshold / L2_weight_gradient_sum_sqrt * weight_gradient

                bias_gradient =  self.EWC_lam * self.old_bias_grad * (self.bias - self.old_bias)
                L2_bias_gradient = np.power(bias_gradient, 2)
                L2_bias_gradient_sum_sqrt = np.sqrt(np.sum(L2_bias_gradient))
                bias_threshold = 1
                if L2_bias_gradient_sum_sqrt > bias_threshold:
                    self.bias_gradient = threshold / L2_bias_gradient_sum_sqrt * bias_gradient
            # elif L2_flag == True:
            self.weight_matrix = np.subtract(self.weight_matrix, lr * self.weight_gradient_matrix)
            self.bias = np.subtract(self.bias, lr * self.bias_gradient)
        else:
            pass

class EWCNetwork:
    def __init__(self,name,old_net=None):
        self.name = name
        self.layer_list = []
        self.old_net = old_net

    def init_fisher(self, data, label):
        # must be used after set up all layers done
        if self.old_net:
            # pass parameters in old nets to new net
            for old_layer in self.old_net.layer_list:
                for layer in self.layer_list:
                    if (old_layer.name == layer.name) and (layer.type == 'fc'):
                        old_weight = old_layer.weight_matrix
                        old_bias = old_layer.bias
                        old_weight_grad = np.zeros(old_weight.shape)
                        old_bias_grad = np.zeros(old_bias.shape)
                        layer.init_fisher_params(old_weight_grad=old_weight_grad, old_weight=old_weight, old_bias_grad=old_bias_grad, old_bias=old_bias)
            # compute fisher
            print('initial fisher done!')
            num_samples = label.shape[0]
            for i in range(num_samples):
                sample_input = np.array([data[i]]).copy()
                sample_label = np.array([label[i]]).copy()
                self.old_net.forward(sample_input, sample_label)
                self.old_net.backward()
                for old_layer in self.old_net.layer_list:
                    for layer in self.layer_list:
                        if (old_layer.name == layer.name) and (layer.type == 'fc'):
                            layer.old_weight_grad += old_layer.weight_gradient_matrix ** 2
                            layer.old_bias_grad += old_layer.bias_gradient ** 2

            for old_layer in self.old_net.layer_list:
                for layer in self.layer_list:
                    if (old_layer.name == layer.name) and (layer.type == 'fc'):
                        layer.old_weight_grad /= num_samples
                        layer.old_bias_grad /= num_samples

    def set_layer_in_order(self,type,name,fc_output_num=1, EWC_flag=True, EWC_lam=0):
        this_layer_id = len(self.layer_list)
        if type == 'fc':
            this_layer = fc(name, EWC_flag, EWC_lam)
            this_layer.set_output_dim(fc_output_num)
        elif type == 'relu':
            this_layer = relu(name)
        elif type == 'softmax':
            this_layer = softmax(name)
        else:raise NotImplementedError
        this_layer.set_layer_id(this_layer_id)
        self.layer_list.append(this_layer)
        return this_layer

    def forward(self,input,label,phase='train'):
        for layer in self.layer_list:
            if layer.layer_id == 0:
                temp_output = layer.forward(input)
            else:
                if layer.type == 'softmax':
                    this_loss,predict = layer.forward(temp_output,label)
                    if phase == 'train':
                        # comment by dft
                        continue
                        # print 'Softmax loss is:',this_loss
                else:
                    temp_output = layer.forward(temp_output)
        if phase == 'train':
            return this_loss
        else:
            return predict

    def backward(self):
        for layer in self.layer_list[::-1]:
            if layer.type == 'softmax':
                temp_output = layer.backward()
            else:
                temp_output = layer.backward(temp_output)

    def update(self,lr):
        ###lr:learning rate
        for layer in self.layer_list:
            if layer.name == 'output_layer' and layer.EWC_flag == True:
                layer.update(lr)
                continue
            layer.update(lr)

    def save_network(self,file_name):
        h5_file = h5py.File('%s'%file_name)
        for layer in self.layer_list:
            if layer.type == 'fc':
                h5_file.create_dataset(name='%s_weight'%layer.name,data=layer.weight_matrix)
                h5_file.create_dataset(name='%s_bias'%layer.name,data=layer.bias)
            else:pass

    def init_from_pretrained_model(self, model_path):
        model_file = h5py.File(name=model_path,mode='r')
        for layer in self.layer_list:
            if layer.type == 'fc':
                layer_name = layer.name
                if model_file.get(name="%s_weight" % layer.name) == None:
                    pass
                else:
                    weight = model_file.get('%s_weight'%layer_name)
                    bias = model_file.get('%s_bias'%layer_name)
                    layer.set_weight_matrix(weight)
                    layer.set_bias_matrix(bias)