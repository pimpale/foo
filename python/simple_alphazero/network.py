import env

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BOARD_CONV_FILTERS = 128

ACTOR_LR = 1e-4  # Lower lr stabilises training greatly
CRITIC_LR = 1e-3  # Lower lr stabilises training greatly
GAMMA = 0.60  # Discount factor for advantage estimation and reward discounting

PPO_EPS = 0.1  # PPO clipping parameter
PPO_GRAD_DESCENT_STEPS = 10  # Number of gradient descent steps to take on the surrogate loss

# (Channel, Width, Height)
def reshape_board(o: env.Observation) -> np.ndarray:
    return np.stack([o.board == 1, o.board == 2])


# output in (Batch, Channel, Width, Height)
def obs_batch_to_tensor(
    o_batch: list[env.Observation], device: torch.device
) -> torch.Tensor:
    # Convert state batch into correct format
    return torch.from_numpy(np.stack([reshape_board(o) for o in o_batch])).to(device)


# output in (Batch, Channel, Width, Height)
def obs_to_tensor(o: env.Observation, device: torch.device) -> torch.Tensor:
    # we need to add a batch axis and then convert into a tensor
    return torch.from_numpy(np.stack([reshape_board(o)])).to(device)


def deviceof(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


class Actor(nn.Module):
    def __init__(self, width: int, height: int):
        super(Actor, self).__init__()

        self.board_width = width
        self.board_height = height

        self.conv1 = nn.Conv2d(2, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(
            BOARD_CONV_FILTERS, BOARD_CONV_FILTERS, kernel_size=3, padding=0
        )
        self.fc1 = nn.Linear((width - 4) * (height - 4) * BOARD_CONV_FILTERS, 512)
        self.fc2 = nn.Linear(512, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Channels, Width, Height)
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # flatten everything except for batch
        x = torch.flatten(x, 1)
        # apply fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output in (Batch, Width)
        output = F.softmax(x, dim=1)
        return output


class Critic(nn.Module):
    def __init__(self, width: int, height: int):
        super(Critic, self).__init__()

        self.board_width = width
        self.board_height = height

        self.conv1 = nn.Conv2d(2, BOARD_CONV_FILTERS, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(
            BOARD_CONV_FILTERS, BOARD_CONV_FILTERS, kernel_size=3, padding=0
        )
        self.fc1 = nn.Linear((width - 4) * (height - 4) * BOARD_CONV_FILTERS, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Width, Height)
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # flatten everything except for batch
        x = torch.flatten(x, 1)
        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # delete extra dimension
        # output in (Batch,)
        output = x.view((x.shape[0]))
        return output



def compute_ppo_loss(
    # Old policy network's probability of choosing an action
    # in (Batch, Action)
    pi_thetak_given_st: torch.Tensor,
    # Current policy network's probability of choosing an action
    # in (Batch, Action)
    pi_theta_given_st: torch.Tensor,
    # One hot encoding of which action was chosen
    # in (Batch, Action)
    a_t: torch.Tensor,
    # Advantage of the chosen action
    A_pi_theta_given_st_at: torch.Tensor,
) -> torch.Tensor:
    # in (Batch,)
    pi_theta_given_st_at = torch.sum(pi_theta_given_st * a_t, 1)
    pi_thetak_given_st_at = torch.sum(pi_thetak_given_st * a_t, 1)

    # the likelihood ratio (used to penalize divergence from the old policy)
    likelihood_ratio = pi_theta_given_st_at / pi_thetak_given_st_at

    # in (Batch,)
    ppo_loss_per_example = -torch.minimum(
        likelihood_ratio * A_pi_theta_given_st_at,
        torch.clip(likelihood_ratio, 1 - PPO_EPS, 1 + PPO_EPS) * A_pi_theta_given_st_at,
    )

    # in (Batch,)
    entropy_per_example = -torch.sum(torch.log(pi_theta_given_st) * pi_theta_given_st, 1)

    # we reward entropy, since excessive certainty indicate the model is 'overfitting'
    loss_per_example = ppo_loss_per_example - 0.1 * entropy_per_example

    # we take the average loss over all examples
    return loss_per_example.mean()


def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[env.Observation],
    action_batch: list[env.Action],
    advantage_batch: list[env.Advantage],
    value_batch: list[env.Value],
) -> tuple[list[float], list[float]]:
    # assert that the models are on the same device
    assert next(critic.parameters()).device == next(actor.parameters()).device
    # assert that the batch_lengths are the same
    assert len(observation_batch) == len(action_batch)
    assert len(observation_batch) == len(advantage_batch)
    assert len(observation_batch) == len(value_batch)

    # get device
    device = next(critic.parameters()).device

    # convert data to tensors on correct device

    # in (Batch, Width, Height)
    observation_batch_tensor = obs_batch_to_tensor(observation_batch, device)

    # in (Batch,)
    true_value_batch_tensor = torch.tensor(
        value_batch, dtype=torch.float32, device=device
    )

    # in (Batch, Action)
    chosen_action_tensor = F.one_hot(
        torch.tensor(action_batch).to(device).long(), num_classes=actor.board_width
    )

    # in (Batch,)
    advantage_batch_tensor = torch.tensor(advantage_batch).to(device)

    # train critic
    critic_optimizer.zero_grad()
    pred_value_batch_tensor = critic.forward(observation_batch_tensor)
    critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor)
    critic_loss.backward()
    critic_optimizer.step()

    # train actor

    # Recall that in the PPO algorithm, we need to set theta to the *optimal* theta with respect to the surrogate loss function,
    # as opposed to the standard policy gradient, where we just update theta.

    # Here's what that means:
    # We have a policy network with parameters theta_k
    # We want to find the optimal theta, theta*, that maximizes the surrogate loss function L_CLIP(theta, theta_k)
    # Note: In practice, we just use gradient descent for PPO_GRAD_DESCENT_STEPS steps to approximate theta*, since we can't analytically solve for theta*
    # We then update: theta_k <- theta*

    # The amount theta* can diverge from theta_k is limited by L_CLIP(theta, theta_k).
    # This is because we want to avoid the new policy diverging too far from the old policy, since that can lead to instability.

    # the old_policy_action_probs are the the predictions made by the pre-train-step network that we want to not diverge too far away from
    # in (Batch, Action)
    old_policy_action_probs = actor.forward(observation_batch_tensor).detach()

    actor_losses = []
    for _ in range(PPO_GRAD_DESCENT_STEPS):
        actor_optimizer.zero_grad()
        current_policy_action_probs = actor.forward(observation_batch_tensor)
        actor_loss = compute_ppo_loss(
            old_policy_action_probs,
            current_policy_action_probs,
            chosen_action_tensor,
            advantage_batch_tensor,
        )
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses += [float(actor_loss)]


    # return the respective losses
    return (actor_losses, [float(critic_loss)]*PPO_GRAD_DESCENT_STEPS)

def compute_advantage(
    critic: Critic,
    trajectory_observations: list[env.Observation],
    trajectory_rewards: list[env.Reward],
) -> list[env.Advantage]:
    """
    Computes advantage using GAE.

    See here for derivation: https://arxiv.org/abs/1506.02438
    """

    trajectory_len = len(trajectory_rewards)

    assert len(trajectory_observations) == trajectory_len
    assert len(trajectory_rewards) == trajectory_len

    trajectory_returns = np.zeros(trajectory_len)

    # calculate the value of each state
    obs_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
    obs_values = critic.forward(obs_tensor).detach().cpu().numpy()

    trajectory_returns[-1] = trajectory_rewards[-1]

    # Use GAMMA to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        trajectory_returns[t] = (
            trajectory_rewards[t] + GAMMA * trajectory_returns[t + 1]
        )

    trajectory_advantages = trajectory_returns - obs_values

    return list(trajectory_advantages)


def compute_value(
    trajectory_rewards: list[env.Reward],
) -> list[env.Value]:
    """
    Computes the gamma discounted reward-to-go for each state in the trajectory.

    Note: In this particular case, we only get rewards at the end of the game,
    but this function assumes that rewards may be given at each step.
    """

    trajectory_len = len(trajectory_rewards)

    v_batch = np.zeros(trajectory_len)

    v_batch[-1] = trajectory_rewards[-1]

    # Use GAMMA to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + GAMMA * v_batch[t + 1]

    return list(v_batch)
