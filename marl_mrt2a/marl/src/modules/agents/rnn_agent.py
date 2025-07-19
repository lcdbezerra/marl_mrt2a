import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        return

        ########################################
        self.layers = []
        if type(args.hidden_dim)==tuple:
            self.layers.append(nn.Linear(input_shape, args.hidden_dim[0]))
            for i in range(len(args.hidden_dim)-1):
                if self.args.use_rnn:
                    self.layers.append(nn.GRUCell(args.hidden_dim[i], args.hidden_dim[i+1]))
                else:
                    self.layers.append(nn.Linear(args.hidden_dim[i], args.hidden_dim[i+1]))
            self.layers.append(nn.Linear(args.hidden_dim[-1], args.n_actions))
        else:
            self.layers.append(nn.Linear(input_shape, args.hidden_dim))
            if self.args.use_rnn:
                self.layers.append(nn.GRUCell(args.hidden_dim, args.hidden_dim))
            else:
                self.layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
            self.layers.append(nn.Linear(input_shape, args.hidden_dim))
        # if self.args.use_rnn:
        #     self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        # else:
        #     self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

        ########################################
        raise NotImplementedError("Rnn agent init_hidden")
        if type(self.args.hidden_dim)==tuple:
            return [self.layers[i].weight.data.fill_(0.) for i in range(1,len(self.layers)-1)]
        else:
            return self.layers[1].weight.data.fill_(0.)
    
    # def rnn(self, x, h):
    #     pass

    def forward(self, inputs, hidden_state):

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

        ########################################
        x = F.relu(self.layers[0](inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

