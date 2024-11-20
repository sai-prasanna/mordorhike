import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from .deepset import DeepSet


def split(latents, beliefs, valid_size=0.2):
    """
    Splits the joint dataset of latents and beliefs using the first samples for
    the validation set.

    Arguments:
    - latents: tensor
        Samples of latent states such that `latents[i, :]` is jointly drawn
        with `beliefs[i, :]`.
    - beliefs: tuple of tensors
        Samples of beliefs such that `beliefs[i, :]` is jointly drawn with
        `latents[i, :]`.
    - valid_size: float
        Proportion of samples to use in the validation set.
    """
    num_samples = latents.size(0)

    split = int(num_samples * valid_size)

    latents_valid = latents[:split, :]
    latents_train = latents[split:, :]

    beliefs_valid = tuple(belief_part[:split, :] for belief_part in beliefs)
    beliefs_train = tuple(belief_part[split:, :] for belief_part in beliefs)

    return latents_train, beliefs_train, latents_valid, beliefs_valid


class LogMeanExpWithGradDenom(torch.autograd.Function):
    """Autograd implementation of a log of mean exponentials function that
    takes an external estimation for its gradient denominator, instead of
    estimating it from the input directly.

    Thank to `gtegner` for its GitHub repository `mine-pytorch` that
    helped me understand how to reimplement the gradient while using a
    "detached" EMA estimation at the denominator.
    """

    @staticmethod
    def forward(ctx, T_pom, grad_denom):
        """Computes the forward pass (log of mean exponentials) while
        memorizing input data for the backward pass. Uses an external
        estimation of the denominator for the backward pass.

        Arguments:
        - T_pom: tensor
            Input vector for which to compute the log of mean
            exponential.
        - grad_denom: tensor
            The denominator to use for the gradient estimation.

        Returns:
            log_mean_exp: the log of mean exponential of  the input `T_pom`.
        """
        ctx.save_for_backward(T_pom, grad_denom)

        n = torch.tensor(T_pom.size(0))
        return T_pom.logsumexp(dim=(0, 1)) - torch.log(n)

    @staticmethod
    def backward(ctx, grad_output):
        """Computes the backward pass (log of mean exponentials) using the
        saved data.

        Arguments:
        - grad_output: tensor
            The gradient of the quantity of interest with respect
            to the output of the forward pass.

        Returns:
        - grad: tensor
            the gradient of the quantity of interest with respect to the
            first input of the forward pass (`T_pom`)
        - None
            The gradient of the quantity of interest with respect to the
            second input of the forward pass (`grad_denom`): not provided.
        """
        T_pom, grad_denom = ctx.saved_tensors

        grad = T_pom.exp().detach() / T_pom.size(0) * grad_output / grad_denom
        return grad, None


class LogMeanExpWithEMAGradDenom:
    """Bias-corrected differentiable log of mean exponentials function, using
    the exponentially moving average of the gradient denominator as the
    gradient denominator.

    Arguments:
        alpha: exponentially moving average rate
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.T_pom_exp_ma = None

    def __call__(self, T_pom, use_ema=True):
        """Computes the forward pass of the log of mean exponentials function.

        Arguments:
        - T_pom: tensor
            The function input
        - use_ema: bool
            Whether to use the exponentially moving average for the
            denominator of the gradient in the backward pass.

        Returns:
        - logmeanexp: tensor
            The function output, bias-corrected differentiable
        """
        T_pom_exp = T_pom.exp().mean().detach()

        if self.T_pom_exp_ma is None:
            self.T_pom_exp_ma = T_pom_exp
        else:
            self.T_pom_exp_ma = (
                self.alpha * T_pom_exp + (1.0 - self.alpha) * self.T_pom_exp_ma
            )

        if use_ema:
            return LogMeanExpWithGradDenom.apply(T_pom, self.T_pom_exp_ma)

        n = torch.tensor(T_pom.size(0))
        return T_pom.logsumexp(dim=(0, 1)) - torch.log(n)


class MutualInformationNeuralEstimator(nn.Module):
    """Mutual Information Neural Estimator (see arXiv:1801.04062).

    Arguments:
    - hs_size: int
        Dimension of the latent states
    - belief_sizes: List[int]
        Dimension of the beliefs (or state particles)
    - latent_size: int
        Number of neurons in latent layers
    - num_layers: int
        Number of latent layers
    - alpha: float
        Exponentially moving average rate for the bias-corrected
        gradient denominator
    - latent_deepset_rep_size: int
        If None, latent is processed as a vector, if
        integer, the latent is processed as a set of particles using a deep
        set whose set representation size is `latent_deepset_rep_size`.
    - belief_deepset_rep_sizes: List[int]
        If None, belief is processed as a vector, if
        integer, the belief is processed as a set of particles using a deep
        set whose set representation size is `belief_deepset_rep_sizes`.
    """

    def __init__(
        self,
        hs_size,
        belief_sizes,
        latent_size,
        num_layers,
        alpha,
        latent_deepset_rep_size,
        belief_deepset_rep_sizes,
        latent_hidden_sizes=(32, 64),
        belief_hidden_sizes=(32, 64),
        belief_part=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.belief_part = belief_part
        if self.belief_part is not None:
            belief_deepset_rep_sizes = (
                belief_deepset_rep_sizes[self.belief_part],
            )
            belief_sizes = (belief_sizes[self.belief_part],)

        if latent_deepset_rep_size is not None:
            self.latent_encoder = DeepSet(hs_size, latent_deepset_rep_size, hidden_sizes=latent_hidden_sizes)
            input_size = latent_deepset_rep_size
        else:
            input_size = hs_size

        self.belief_encoders = []
        for representation_size, belief_size in zip(
            belief_deepset_rep_sizes,
            belief_sizes,
        ):
            if representation_size is None:
                self.belief_encoders.append(None)
                input_size += belief_size
            else:
                encoder = DeepSet(belief_size, representation_size, hidden_sizes=belief_hidden_sizes)
                self.belief_encoders.append(encoder)
                input_size += representation_size

        self.belief_encoders = nn.ModuleList(self.belief_encoders)

        latent_layers = []

        for _ in range(num_layers - 1):
            latent_layers.append(nn.Linear(latent_size, latent_size))
            latent_layers.append(nn.ReLU())

        self.sequential = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            *latent_layers,
            nn.Linear(latent_size, 1),
        )

        self.logmeanexp = LogMeanExpWithEMAGradDenom(alpha)

        self.device = device
        self.to(device)

    def forward(
        self, latents, beliefs, latents_marginal, beliefs_marginal, use_ema=True
    ):
        """Returns the estimated lower bound of the mutual information from
        samples from the joint distribution and from the product of marginal
        distributions.

        Arguments:
        - latents: tensor
            Samples of latent states such that `latents[i, :]` is
            jointly drawn with `beliefs[i, :]`.
        - beliefs: tuple of tensors
            Samples of beliefs such that `beliefs[i, :]` are jointly
            drawn with `latents[i, :]`.
        - latents_marginal: tensor
            Samples of latent states
        - beliefs_marginal: tuple of tensors
            Samples of beliefs
        - use_ema: bool
            Whether to use the exponentially moving average bias-
            correction in the estimation of the gradient.

        Returns:
        - T_joint: tensor
            Average statistic for samples from the joint distribution
        - log_mean_exp_t_pom: tensor
            Average log mean exp of the statistic for samples from the product
            of marginal distributions.
        """
        encoded, encoded_marginal = [], []
        for belief_part, belief_part_marginal, encoder in zip(
            beliefs, beliefs_marginal, self.belief_encoders
        ):

            if encoder is not None:
                belief_part = encoder(belief_part)
                belief_part_marginal = encoder(belief_part_marginal)

            encoded.append(belief_part)
            encoded_marginal.append(belief_part_marginal)

        if self.latent_encoder is not None:
            latents = self.latent_encoder(latents)
            latents_marginal = self.latent_encoder(latents_marginal)

        joint = torch.cat([latents] + encoded, dim=1)
        pom = torch.cat([latents_marginal] + encoded_marginal, dim=1)

        T_joint = self.sequential.forward(joint)
        T_pom = self.sequential.forward(pom)

        return T_joint.mean(), self.logmeanexp(T_pom, use_ema=use_ema)

    def optimize(
        self,
        latents,
        beliefs,
        num_epochs,
        logger,
        learning_rate,
        batch_size,
        lambd,
        latents_valid=None,
        beliefs_valid=None,
        valid_size=None,
    ):
        """Optimize (maximize) the lower bound of the mutual information by
        gradient ascent over samples of the joint distribution. The mutual
        information is between the latent states `latents` and the belief
        representations `beliefs`.

        Arguments:
        - latents: tensor
            The samples of the latent states such that `latents[i, :]`
            are jointly drawn with `beliefs[i, :]`.
        - beliefs: tuple of tensors
            The samples of the beliefs such that `beliefs[k][i, :]` are
            jointly drawn with `latents[i, :]`.
        - num_epochs: int
            The number of epochs to perform on the joint samples.
        - logger: function
            The function to call for logging the optimization statistics.
        - learning_rate: float
            The learning rate used in the Adam optimizer.
        - batch_size: int
            The number of samples in each minibatch.
        - lambd: float
            The regularization of MINE (ReMINE).
        - valid_size: float
            Proportion of samples to use in the validation set. The final
            weights are those with the highest validation estimate. If None,
            the final weights are the last ones.
        """
        if self.belief_part is not None:
            beliefs = (beliefs[self.belief_part],)

        best_valid_estimate = -float("inf")
        best_train_mi = -float("inf")
        best_epoch = 0
        best_weights = deepcopy(self.state_dict())

        print("Optimizing...")

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if valid_size is not None:
            latents, beliefs, latents_valid, beliefs_valid = split(
                latents, beliefs, valid_size=valid_size
            )

        num_samples = latents.size(0)

        for epoch in range(num_epochs):

            total_value = 0.0

            perm_joint = torch.randperm(num_samples)
            perm_pom_1 = torch.randperm(num_samples)
            perm_pom_2 = torch.randperm(num_samples)

            for i in range(0, num_samples, batch_size):

                indices_joint = perm_joint[i : i + batch_size]
                indices_pom_1 = perm_pom_1[i : i + batch_size]
                indices_pom_2 = perm_pom_2[i : i + batch_size]

                latents_batch = latents[indices_joint, :]
                beliefs_batch = tuple(
                    belief_part[indices_joint, ...] for belief_part in beliefs
                )
                latents_marginal_batch = latents[indices_pom_1, :]
                beliefs_marginal_batch = tuple(
                    belief_part[indices_pom_2, ...] for belief_part in beliefs
                )

                beliefs_batch = tuple(
                    belief_part.to(self.device) for belief_part in beliefs_batch
                )
                beliefs_marginal_batch = tuple(
                    belief_part.to(self.device)
                    for belief_part in beliefs_marginal_batch
                )

                T_joint, T_pom_logmeanexp = self.forward(
                    latents_batch.to(self.device),
                    beliefs_batch,
                    latents_marginal_batch.to(self.device),
                    beliefs_marginal_batch,
                )

                v = T_joint - T_pom_logmeanexp - lambd * T_pom_logmeanexp**2

                optimizer.zero_grad()
                (-v).backward()
                optimizer.step()

                total_value += v.item()

            optimizer.step()

            mean_value = total_value / int(num_samples / batch_size)

            if valid_size is not None:
                valid_estimate = self.estimate(latents_valid, beliefs_valid)
            else:
                valid_estimate = mean_value

            # Update statistics for convergence detection
            if valid_estimate > best_valid_estimate:
                best_valid_estimate = valid_estimate
                best_train_mi = mean_value
                best_epoch = epoch
                best_weights = deepcopy(self.state_dict())

            print("\033[F\033[K", end="")
            print(
                f"Epoch {epoch:03d}: mean value: {mean_value:.6f}, "
                f"current valid estimate: {valid_estimate:.6f}, "
                f"best valid mi: {best_train_mi:.6f} at epoch {best_epoch}"
            )
            logger(
                {
                    "mine_optim/epoch": epoch,
                    "mine_optim/mean_value": mean_value,
                    "mine_optim/valid_estimate": valid_estimate,
                    "mine_optim/best_train_mi": best_train_mi,
                    "mine_optim/best_valid_mi": best_valid_estimate,
                    "mine_optim/best_epoch": best_epoch,
                }
            )

        if valid_size is not None:
            self.load_state_dict(best_weights)

    def estimate(self, latents, beliefs):
        """
        Compute the mutual information estimation using optimized network
        weights.

        Arguments:
        - latents: tensor
            The samples of the latent states such that `latents[i, :]`
            are jointly drawn with `beliefs[i, :]`.
        - beliefs: tuple of tensors
            The samples of the beliefs such that `beliefs[k][i, :]` are
            jointly drawn with `latents[i, :]`.

        Returns:
        - estimate: float
            The estimated mutual information based on the lower bound
            maximization.
        """
        if self.belief_part is not None:
            beliefs = (beliefs[self.belief_part],)

        perm_pom_1 = torch.randperm(latents.size(0))
        perm_pom_2 = torch.randperm(latents.size(0))

        latents_marginal = latents[perm_pom_1, :]
        beliefs_marginal = tuple(
            belief_part[perm_pom_2, ...] for belief_part in beliefs
        )

        self.cpu()
        with torch.no_grad():
            T_joint, T_pom_logmeanexp = self.forward(
                latents, beliefs, latents_marginal, beliefs_marginal, use_ema=False
            )
        self.to(self.device)

        return (T_joint - T_pom_logmeanexp).item()

    def save(self, run_id, episode=None):
        """
        Saves the weights of a trained estimator on disk. Does not save the
        optimizer's momenta.

        Arguments:
        - run_id: str
            Unique identifier for the training run
        - episode: int
            The episode at which the agent was saved
        """
        os.makedirs("weights", exist_ok=True)
        path = f"weights/{run_id}-{episode}-{{}}.pth"
        torch.save(self.state_dict(), path.format("T"))

    def load(self, run_id, episode=None):
        """Loads the weights of an estimator saved on disk. Does not load the
        optimizer's momenta.

        Arguments:
        - run_id: str
            Unique identifier for the training run
        - episode: int
            The episode at which the agent was saved
        """
        path = f"weights/{run_id}-{episode}-{{}}.pth"
        self.load_state_dict(torch.load(path.format("T")))
