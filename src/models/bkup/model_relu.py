import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.2,
        norm_layers=(),
        latent_in=(),
        activation='relu',
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        dims = [3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.use_tanh = use_tanh

        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            if weight_norm and layer in self.norm_layers:  # except the last layer
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))


        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        self.activation = activation

    # input: N x 3
    def forward(self, input, interval=False):
        #print(len(input))
        if len(input) > 1:
            x = input[:, -3:]
        else:
            x = input[-3:]

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)

            x = lin(x)

            if layer < self.num_layers - 2:  # except last layer
                if self.activation == 'relu':
                    x = self.relu(x)
                elif self.activation == 'tanh':
                    x = self.th(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_tanh:
            x = self.th(x)

        return x
