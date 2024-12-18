import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import cv2
from pathlib import Path
from dreamerv3 import embodied
from powm.envs.mordor import MordorHike
from powm.algorithms.train_dreamer import make_logger

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
    X, Y = [], []
    for episode in episodes:
        episode["discrete_belief"] = []
        for step in range(len(episode["belief"])):
            discretized_belief = env.discretize_belief(episode["belief"][step])
            episode["discrete_belief"].append(discretized_belief)
            latent = episode["latent"][step]
            X.append(latent.reshape(-1))
            Y.append(discretized_belief.reshape(-1))
        episode["discrete_belief"] = np.array(episode["discrete_belief"])
    return np.array(X), np.array(Y)

def train_belief_predictor(train_X, train_Y, val_X, val_Y, belief_shape, device):
    """Train belief predictor and return best model"""
    model = BeliefPredictor(train_X.shape[1], train_Y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    dataset = TensorDataset(torch.FloatTensor(train_X).to(device), 
                           torch.FloatTensor(train_Y).to(device))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    best_model = None
    best_val_loss = float('inf')
    
    for epoch in range(100):
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
            best_model = model.state_dict().copy()
            
    model.load_state_dict(best_model)
    return model

def visualize_trajectory_step(env, episode, step_idx):
    """
    Visualize a single step showing true and predicted belief distributions side by side
    """
    # Get true and predicted belief grids
    true_grid = episode['discrete_belief'][step_idx].sum(-1)
    pred_grid = episode['predicted_belief'][step_idx].sum(-1)

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
    
    # Create label space and colorbar
    label_height = 40
    colorbar_height = 20
    bottom_space = np.zeros((label_height + colorbar_height, w*2, 3), dtype=np.uint8)
    
    # Add labels below images
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bottom_space, "True", (w//2 - 50, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(bottom_space, "Predicted", (w + w//2 - 70, 25), font, 0.7, (255, 255, 255), 2)
    
    # Create colorbar in RGB
    colorbar = np.linspace(255, 0, w*2).astype(np.uint8)  # Invert colorbar range
    colorbar = cv2.applyColorMap(colorbar.reshape(1, -1), cv2.COLORMAP_JET)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)
    colorbar = cv2.resize(colorbar, (w*2, colorbar_height))
    
    # Add min/max labels to colorbar
    cv2.putText(colorbar, "0.0", (5, 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(colorbar, "1.0", (w*2-30, 15), font, 0.5, (255, 255, 255), 1)
    
    # Place colorbar in bottom space
    bottom_space[label_height:, :] = colorbar
    
    # Stack everything together
    combined = np.vstack([
        np.hstack([true_vis, pred_vis]),
        bottom_space
    ])
    return combined

def compute_prediction_metrics(episodes):
    num_pred_steps = episodes[0]['obs_hat'].shape[2]
    num_particles = episodes[0]['obs_hat'].shape[1]
    assert num_particles == 1
    step_mses = [[] for _ in range(num_pred_steps)]

    for episode in episodes:
        num_timesteps = episode['obs'].shape[0]
        
        # For each prediction step (1 to max_pred_steps)
        for step in range(num_pred_steps):
            # Get predictions and true observations
            preds = episode['obs_hat'][:num_timesteps-step, :, step, :]  # index 0 for first (only) particle
            trues = episode['obs'][step:step+len(preds)][:, np.newaxis, :]
            
            # Calculate MSE for this episode at this prediction step
            mse = np.mean((preds - trues) ** 2)
            step_mses[step].append(mse)

    # Calculate statistics across episodes
    metrics = {}
    for step in range(num_pred_steps):
        metrics[f'obs_pred_{step+1}_step_mean'] = np.mean(step_mses[step])
        metrics[f'obs_pred_{step+1}_step_std'] = np.std(step_mses[step])
    return metrics

def compute_overall_metrics(episodes, discount_factor=0.99):
    """Compute metrics for episodes"""
    scores = [episode['reward'].sum() for episode in episodes]
    returns = [np.sum(episode['reward'] * (discount_factor ** np.arange(len(episode['reward'])))) for episode in episodes]
    successes = [1. if episode['success'] else 0. for episode in episodes]
    
    
    
    metrics = {
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'success_mean': np.mean(successes),
        'success_std': np.std(successes),
    }
    if 'obs_hat' in episodes[0]:
        metrics.update(compute_prediction_metrics(episodes))
    return metrics



def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        metric_dir="eval",
    ).parse_known(argv)
    
    assert parsed.logdir, "Logdir is required"
    logdir = Path(parsed.logdir)
    
    # Load config
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)
    
    # Setup environment and device
    env = MordorHike.medium(render_mode="human", estimate_belief=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create logger
    logger = make_logger(config, metric_dir=parsed.metric_dir)
    
    # Process the rollouts
    episode_paths = sorted(
        logdir.glob("episodes_*.npz"), 
        key=lambda x: int(x.stem.split("_")[-1])
    )
    
    for episode_path in episode_paths:
        ckpt_number = int(episode_path.stem.split("_")[-1])
        logger.step.load(ckpt_number)  # Set logger step to checkpoint number
        
        # Load episodes
        episodes = np.load(
            logdir / f"episodes_{ckpt_number}.npz", 
            allow_pickle=True
        )['episodes']

        # Split episodes
        n_episodes = len(episodes)
        train_idx = int(0.8 * n_episodes)
        val_idx = int(0.9 * n_episodes)

        train_episodes = episodes[:train_idx]
        val_episodes = episodes[train_idx:val_idx]
        test_episodes = episodes[val_idx:]

        # Preprocess data
        train_X, train_Y = preprocess_episodes(train_episodes, env)
        val_X, val_Y = preprocess_episodes(val_episodes, env)
        test_X, test_Y = preprocess_episodes(test_episodes, env)
        belief_shape = test_episodes[0]["discrete_belief"][0].shape
        # Train predictor
        model = train_belief_predictor(
            train_X, train_Y, 
            val_X, val_Y,
            belief_shape,
            device
        )

        # Evaluate and log metrics
        overall_metrics = compute_overall_metrics(episodes)
        model.eval()
        criterion = nn.KLDivLoss(reduction='batchmean')

        # Compute KL divergence only on test set
        with torch.no_grad():
            test_pred = model(torch.FloatTensor(test_X).to(device))
            val_pred = model(torch.FloatTensor(val_X).to(device))
            train_pred = model(torch.FloatTensor(train_X).to(device))
            test_kldiv = criterion(test_pred, torch.FloatTensor(test_Y).to(device)).cpu().item()
            val_kldiv = criterion(val_pred, torch.FloatTensor(val_Y).to(device)).cpu().item()
            train_kldiv = criterion(train_pred, torch.FloatTensor(train_Y).to(device)).cpu().item()
        # Calculate episode KL divergences for test episodes only
        belief_shape = test_episodes[0]["discrete_belief"][0].shape
        test_episode_kldivs = []
        idx = 0
        for episode in test_episodes:
            end_idx = idx + len(episode["belief"])
            episode_pred = test_pred[idx:end_idx]
            episode_Y = torch.FloatTensor(test_Y[idx:end_idx]).to(device)
            episode["predicted_belief"] = torch.exp(test_pred[idx:end_idx]).reshape(-1, *belief_shape).detach().cpu().numpy()
            episode["discrete_belief"] = test_Y[idx:end_idx].reshape(-1, *belief_shape)
            episode_kldiv = criterion(episode_pred, episode_Y).cpu().item()
            test_episode_kldivs.append(episode_kldiv)
            idx = end_idx
        # Calculate test scores for correlation
        test_scores = [episode['reward'].sum() for episode in test_episodes]

        # Calculate correlation between episode KL divergence and scores (test set only)
        score_kldiv_correlation = pearsonr(test_scores, test_episode_kldivs)[0]

        overall_metrics.update({
            'test_stepwise_kldiv': test_kldiv,
            'val_stepwise_kldiv': val_kldiv,
            'train_stepwise_kldiv': train_kldiv,
            'test_episodic_kldiv_mean': np.mean(test_episode_kldivs),
            'test_episodic_kldiv_std': np.std(test_episode_kldivs),
            'test_score_episodic_kldiv_corr': score_kldiv_correlation,
        })
        logger.add(overall_metrics)
        
        video_dir = logdir / parsed.metric_dir / f"videos_{ckpt_number}"
        video_dir.mkdir(parents=True, exist_ok=True)
        for i, episode in enumerate(test_episodes):
            frames = []
            for i in range(len(episode['state'])):
                frame = visualize_trajectory_step(env, episode, step_idx=i)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames = np.array(frames)
            logger.add({
                f"eval_video/{i}": frames
            })
        logger.write()
    logger.close()


if __name__ == "__main__":
    main()