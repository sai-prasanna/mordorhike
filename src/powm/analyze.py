import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dreamerv3 import embodied
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from powm.algorithms.train_dreamer import make_logger
from powm.envs.mordor import MordorHike
from powm.utils import set_seed


class BeliefPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

def preprocess_episodes(episodes, env):
    """Preprocess episodes into training data"""
    X, X_rand, Y = [], [], []
    for episode in episodes:
        episode["discrete_belief"] = []
        for step in range(len(episode["belief"])):
            # Sum over angles when discretizing
            discretized_belief = env.discretize_belief(episode["belief"][step])
            episode["discrete_belief"].append(discretized_belief)
            latent = episode["latent"][step]
            control_latent = episode["control_latent"][step]
            X_rand.append(control_latent.reshape(-1))
            X.append(latent.reshape(-1))
            Y.append(discretized_belief.reshape(-1))
        episode["discrete_belief"] = np.array(episode["discrete_belief"])
        # shuffle Y within the episode to get Y_control
    return np.array(X), np.array(X_rand), np.array(Y)

def train_belief_predictor(train_X, train_Y, val_X, val_Y, device):
    """Train belief predictor and return best model"""
    model = BeliefPredictor(train_X.shape[1], train_Y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    dataset = TensorDataset(torch.FloatTensor(train_X).to(device), 
                           torch.FloatTensor(train_Y).to(device))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    best_model = None
    best_val_loss = float('inf')
    
    for epoch in range(300):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(val_X).to(device))
            val_loss = criterion(val_pred, torch.FloatTensor(val_Y).to(device))
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model)
    torch.cuda.empty_cache()
    
    return model.eval()

def visualize_trajectory_step(env, episode, step_idx):
    """
    Visualize a single step showing true and predicted belief distributions side by side
    """
    # Get true and predicted belief grids
    true_grid = episode['discrete_belief'][step_idx].sum(axis=-1)
    pred_grid = episode['predicted_belief'][step_idx].sum(axis=-1)

    # Normalize grids to 0-255 range
    true_grid = (((true_grid - true_grid.min()) / (true_grid.max() - true_grid.min())) * 255).astype(np.uint8)
    pred_grid = (((pred_grid - pred_grid.min()) / (pred_grid.max() - pred_grid.min())) * 255).astype(np.uint8)
    # Make y coordinates negative 
    true_grid = np.flip(true_grid, axis=1).T
    pred_grid = np.flip(pred_grid, axis=1).T
    
    render_size = (128, 128)
    true_grid = cv2.resize(true_grid, render_size)
    pred_grid = cv2.resize(pred_grid, render_size)
    
    # Create two grayscale backgrounds
    env.render_size = render_size
    true_img = env._create_background()
    true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2GRAY)
    true_img = cv2.cvtColor(true_img, cv2.COLOR_GRAY2BGR)
    pred_img = true_img.copy()
    h, w = true_img.shape[:2]
    
    # Convert belief grids to RGB heatmaps (not BGR)
    true_heatmap = cv2.applyColorMap(255 - true_grid, cv2.COLORMAP_JET)  # Invert input to colormap
    true_heatmap = cv2.cvtColor(true_heatmap, cv2.COLOR_BGR2RGB)
    pred_heatmap = cv2.applyColorMap(255 - pred_grid, cv2.COLORMAP_JET)  # Invert input to colormap
    pred_heatmap = cv2.cvtColor(pred_heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend heatmaps with background
    alpha = 0.33
    true_vis = cv2.addWeighted(true_img, 1-alpha, true_heatmap, alpha, 0)
    pred_vis = cv2.addWeighted(pred_img, 1-alpha, pred_heatmap, alpha, 0)
    
    # Draw path and current position
    path = episode['state'][:step_idx+1, :2]
    current_state = episode['state'][step_idx]
    path_pixels = env._world_to_pixel(path)
    pos_pixel = env._world_to_pixel(current_state[:2])
    
    # Draw path
    for img in [true_vis, pred_vis]:
        cv2.polylines(img, [path_pixels.astype(np.int32)], False, (0, 0, 255), 2)
        cv2.circle(img, tuple(pos_pixel.astype(np.int32)), 5, (0, 0, 255), -1)
    
    # Create colorbar space
    colorbar_height = 20
    
    # Create colorbar in RGB
    colorbar = np.linspace(255, 0, w*2).astype(np.uint8)  # Invert colorbar range
    colorbar = cv2.applyColorMap(colorbar.reshape(1, -1), cv2.COLORMAP_JET)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)
    colorbar = cv2.resize(colorbar, (w*2, colorbar_height))
    
    # Add min/max labels to colorbar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colorbar, "0.0", (5, 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(colorbar, "1.0", (w*2-30, 15), font, 0.5, (255, 255, 255), 1)
    
    # Stack everything together
    combined = np.vstack([
        np.hstack([true_vis, pred_vis]),
        colorbar
    ])
    return combined
def calculate_episodic_kldivs(episodes, pred, Y, device, criterion):
    """Calculate KL divergence for each episode in the dataset.
    
    Args:
        episodes: List of episodes
        pred: Model predictions (log probabilities)
        Y: Ground truth labels
        device: torch device
        criterion: Loss function (KLDivLoss)
    
    Returns:
        episode_kldivs: List of KL divergences for each episode
        processed_episodes: Episodes with predicted_belief and discrete_belief added
    """
    belief_shape = episodes[0]["discrete_belief"][0].shape
    episode_kldivs = []
    idx = 0
    for episode in episodes:
        end_idx = idx + len(episode["belief"])
        episode_pred = pred[idx:end_idx]
        episode_Y = torch.FloatTensor(Y[idx:end_idx]).to(device)
        episode["predicted_belief"] = torch.exp(pred[idx:end_idx]).reshape(-1, *belief_shape).detach().cpu().numpy()
        episode["discrete_belief"] = Y[idx:end_idx].reshape(-1, *belief_shape)
        episode_kldiv = criterion(episode_pred, episode_Y).cpu().item()
        episode_kldivs.append(episode_kldiv)
        idx = end_idx
    return episode_kldivs, episodes

def compute_prediction_metrics(episodes):
    num_pred_steps = episodes[0]['obs_hat'].shape[1]
    step_mses = [[] for _ in range(num_pred_steps)]

    for episode in episodes:        
        for step in range(num_pred_steps):
            # Get predictions and true observations
            preds = episode['obs_hat'][:, step, :]
            trues = episode['obs'][step+1:len(preds)+step+1][:, np.newaxis, :]
            
            # Calculate MSE for this episode at this prediction step
            mse = np.mean((preds - trues) ** 2)
            step_mses[step].append(mse)
    step_mses = np.array(step_mses)
    # Calculate statistics across episodes
    metrics = {}
    for step in range(num_pred_steps):
        metrics[f'obs_pred_{step+1}_step_mean'] = np.mean(step_mses[step])
        metrics[f'obs_pred_{step+1}_step_std'] = np.std(step_mses[step])
    return metrics, step_mses

def compute_overall_metrics(episodes, discount_factor=0.99):
    """Compute metrics for episodes"""
    scores = [episode['reward'].sum() for episode in episodes]
    returns = [np.sum(episode['reward'] * (discount_factor ** np.arange(len(episode['reward'])))) for episode in episodes]
    successes = [1. if episode['success'] else 0. for episode in episodes]
    episode_lengths = [len(episode['state']) for episode in episodes]
            
    metrics = {
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'success_mean': np.mean(successes),
        'success_std': np.std(successes),
        'length_mean': np.mean(episode_lengths),
        'length_std': np.std(episode_lengths),
    }
    if 'waypoints' in episodes[0]:
        waypoint_scores = [episode['reward'][episode['waypoints_step'][-1]:].sum() for episode in episodes]
        metrics['after_waypoints_score_mean'] = np.mean(waypoint_scores)
        metrics['after_waypoints_score_std'] = np.std(waypoint_scores)
    if 'obs_hat' in episodes[0]:
        pred_metrics, step_mses = compute_prediction_metrics(episodes)
        metrics.update(pred_metrics)
        avg_mses = np.mean(step_mses, axis=0) 
        score_pred_error_corr = pearsonr(scores, avg_mses)[0]
        metrics['score_pred_error_corr'] = score_pred_error_corr
    return metrics


def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        metric_dir="eval",
        single_probe=False,
    ).parse_known(argv)
    
    assert parsed.logdir, "Logdir is required"
    logdir = Path(parsed.logdir)
    
    # Load config
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)
    set_seed(config.seed)
    
    # the difficulty doesnot matter for the purposes of this script
    # TODO: use config to create the environment
    env = MordorHike.medium(render_mode="human", estimate_belief=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create logger
    logger = make_logger(config, metric_dir=parsed.metric_dir)
    
    # Process the rollouts
    episode_paths = sorted(
        logdir.glob("episodes_*.npz"), 
        key=lambda x: int(x.stem.split("_")[-1])
    )
    in_distribution_scores = []
    in_distribution_kldivs = []
    waypoint_scores = []
    waypoint_kldivs = []
    noisy_scores = []
    noisy_kldivs = []
    for episode_path in episode_paths:
        ckpt_number = int(episode_path.stem.split("_")[-1])
        logger.step.load(ckpt_number)  # Set logger step to checkpoint number
        # Load episodes
        rollouts = np.load(
            logdir / f"episodes_{ckpt_number}.npz", 
            allow_pickle=True
        )
        control_probe = None
        probe = None
        for key in ["episodes","noisy_episodes", "waypoint_episodes"]:
            
            episodes = rollouts[key]
            n_episodes = len(episodes)
            train_idx = int(n_episodes / 3)
            val_idx = 2 * train_idx            
            train_episodes = list(episodes[:train_idx])
            val_episodes =  list(episodes[train_idx:val_idx])
            test_episodes = list(episodes[val_idx:])

            # Preprocess data
            train_X, train_X_control, train_Y = preprocess_episodes(train_episodes, env)
            val_X, val_X_control, val_Y = preprocess_episodes(val_episodes, env)
            test_X, test_X_control, test_Y = preprocess_episodes(test_episodes, env)
            if probe is None or not parsed.single_probe:
                probe = train_belief_predictor(
                    train_X, train_Y, 
                    val_X, val_Y,
                    device
                )
                control_probe = train_belief_predictor(
                    train_X_control, train_Y,
                    val_X_control, val_Y,
                    device
                )
            # Calculate episode KL divergences for all sets
            test_pred = probe(torch.FloatTensor(test_X).to(device))
            val_pred = probe(torch.FloatTensor(val_X).to(device))
            train_pred = probe(torch.FloatTensor(train_X).to(device))
            criterion = nn.KLDivLoss(reduction='batchmean')
            # Calculate episode KL divergences of the control probe
            control_test_pred = control_probe(torch.FloatTensor(test_X_control).to(device))
            control_test_episode_kldivs, _ = calculate_episodic_kldivs(
                test_episodes, control_test_pred, test_Y, device, criterion)
            test_episode_kldivs, test_episodes = calculate_episodic_kldivs(
                test_episodes, test_pred, test_Y, device, criterion)
            val_episode_kldivs, _ = calculate_episodic_kldivs(
                val_episodes, val_pred, val_Y, device, criterion)
            train_episode_kldivs, _ = calculate_episodic_kldivs(
                train_episodes, train_pred, train_Y, device, criterion)
            overall_metrics = compute_overall_metrics(episodes)
            diff_episode_kldivs = np.array(control_test_episode_kldivs) - np.array(test_episode_kldivs)
            test_scores = [episode['reward'].sum() for episode in test_episodes]
            if key == "episodes":
                in_distribution_scores.extend(test_scores)
                in_distribution_kldivs.extend(test_episode_kldivs)
            elif key == "noisy_episodes":
                noisy_scores.extend(test_scores)
                noisy_kldivs.extend(test_episode_kldivs)
            elif key == "waypoint_episodes":
                waypoint_scores.extend(test_scores)
                waypoint_kldivs.extend(test_episode_kldivs)
            
            score_kldiv_corr = pearsonr(test_scores, test_episode_kldivs)[0]
            score_kldiv_corr_spearman = spearmanr(test_scores, test_episode_kldivs)[0]
            overall_metrics.update({
                'episodic_kldiv': np.mean(test_episode_kldivs),
                'train_episodic_kldiv': np.mean(train_episode_kldivs),
                "val_episodic_kldiv": np.mean(val_episode_kldivs),
                'score_episodic_kldiv_corr_pearson': score_kldiv_corr,
                'score_episodic_kldiv_corr_spearman': score_kldiv_corr_spearman,
                'control_episodic_kldiv': np.mean(control_test_episode_kldivs),
                'episodic_kldiv_diff': np.mean(diff_episode_kldivs),
            })
            logger.add(overall_metrics, prefix=key)
            visualize_episodes = np.random.choice(test_episodes, 10, replace=False)
            for i, episode in enumerate(visualize_episodes):
                frames = []
                for step_idx in range(len(episode['state'])):
                    frame = visualize_trajectory_step(env, episode, step_idx=step_idx)
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames = np.array(frames)
                logger.add({
                    f"eval_video/{i}": frames
                }, prefix=key)
            logger.write()

    score_kldiv_corr = pearsonr(in_distribution_scores, in_distribution_kldivs)[0]
    score_kldiv_corr_spearman = spearmanr(in_distribution_scores, in_distribution_kldivs)[0]
    waypoint_score_kldiv_corr = pearsonr(waypoint_scores, waypoint_kldivs)[0]
    waypoint_score_kldiv_corr_spearman = spearmanr(waypoint_scores, waypoint_kldivs)[0]
    noisy_score_kldiv_corr = pearsonr(noisy_scores, noisy_kldivs)[0]
    noisy_score_kldiv_corr_spearman = spearmanr(noisy_scores, noisy_kldivs)[0]
    
    overall_scores = in_distribution_scores + waypoint_scores + noisy_scores
    overall_kldivs = in_distribution_kldivs + waypoint_kldivs + noisy_kldivs
    overall_score_kldiv_corr = pearsonr(overall_scores, overall_kldivs)[0]
    overall_score_kldiv_corr_spearman = spearmanr(overall_scores, overall_kldivs)[0]
    logger.add({
        'in_dist_score_kldiv_corr_pearson': score_kldiv_corr,
        'in_dist_score_kldiv_corr_spearman': score_kldiv_corr_spearman,
        'waypoint_score_kldiv_corr_pearson': waypoint_score_kldiv_corr,
        'waypoint_score_kldiv_corr_spearman': waypoint_score_kldiv_corr_spearman,
        'noisy_score_kldiv_corr_pearson': noisy_score_kldiv_corr,
        'noisy_score_kldiv_corr_spearman': noisy_score_kldiv_corr_spearman,
        'overall_score_kldiv_corr_pearson': overall_score_kldiv_corr,
        'overall_score_kldiv_corr_spearman': overall_score_kldiv_corr_spearman,
    }, prefix="overall")
    logger.close()


if __name__ == "__main__":
    main()