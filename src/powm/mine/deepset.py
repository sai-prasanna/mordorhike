import torch.nn as nn


class DeepSet(nn.Module):
    """Deep set (see arXiv:1703.06114). The implementation follows the one from
    <https://github.com/manzilzaheer/DeepSets>.

    Arguments:
    - input_size: int
        The input size
    - representation_size: int
        The vector size for the set representation
    """

    def __init__(self, input_size, representation_size, hidden_sizes=(32, 64)):
        super().__init__()

        layers = []
        in_size = input_size
        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size
        layers.append(nn.Linear(in_size, representation_size))
        self.phi = nn.Sequential(*layers)

        self.rho = nn.Linear(representation_size, representation_size)
    def forward(self, input):
        """
        Computes the forward pass in the encoder and outputs the set
        representation.

        Arguments:
        - inputs: a batch of inputs of size [B, N, D] where B is the batch
            size, N, the number of elements in the set, and D the
            dimension.

        Returns:
        - representation: tensor
            The intermediate representation of the set.
        """
        features = self.phi(input)
        return self.rho(features.sum(dim=1))
