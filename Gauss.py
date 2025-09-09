
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class GMM_MLP(nn.Module):
    def __init__(self, obs_dim=17, action_dim=6, hidden_dim=256, num_gaussians=5):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        
        # Same as MSE
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.mixture_weights = nn.Linear(hidden_dim, num_gaussians)
        self.means = nn.Linear(hidden_dim, num_gaussians * action_dim)
        self.log_stds = nn.Linear(hidden_dim, num_gaussians * action_dim)
        
    def forward(self, obs):
        features = self.net(obs)
        
        mixture_logits = self.mixture_weights(features)
        mixture_weights = F.softmax(mixture_logits, dim=-1)
        
        means = self.means(features)
        means = means.view(-1, self.num_gaussians, self.action_dim)

        log_stds = self.log_stds(features)
        log_stds = log_stds.view(-1, self.num_gaussians, self.action_dim)
        log_stds = torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = torch.exp(log_stds)
        
        return mixture_weights, means, stds
    
    def log_prob(self, obs, actions):
        """Compute log probability of actions under the GMM"""
        mixture_weights, means, stds = self(obs)  # Get GMM parameters
        
        # actions: (batch, action_dim) -> (batch, 1, action_dim) for broadcasting
        actions = actions.unsqueeze(1)
        
        diff = (actions - means) / stds  # (batch, num_gaussians, action_dim)
        log_probs_gaussian = -0.5 * (diff ** 2 + 2 * torch.log(stds) + np.log(2 * np.pi))
        log_probs_gaussian = log_probs_gaussian.sum(dim=-1)  # Sum over action dimensions
        
        log_mixture_weights = torch.log(mixture_weights + 1e-8)  # (batch, num_gaussians)
        log_probs_weighted = log_probs_gaussian + log_mixture_weights
       
        log_prob = torch.logsumexp(log_probs_weighted, dim=-1)  # (batch,)
        
        return log_prob
    
    def get_mean_action(self, obs):
        """Get mean action (weighted average of all Gaussian means)"""
        mixture_weights, means, stds = self(obs)
        
        # Σ π_k * μ_k
        mean_action = (mixture_weights.unsqueeze(-1) * means).sum(dim=1)  # (batch, action_dim)
        
        return mean_action

class GMM_Trainer:
    def __init__(self, network, train_loader, val_loader, lr=1e-4, log_dir="runs/gmm"):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        self.writer = SummaryWriter(log_dir)
        
        self.train_losses = []
        self.val_losses = []
        self.epoch_rewards = []
        
    def train_step(self, obs_batch, action_batch):
        self.optimizer.zero_grad()
        
        log_probs = self.network.log_prob(obs_batch, action_batch)
        loss = -log_probs.mean()  # Negative log likelihood
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self):
        self.network.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for obs_batch, action_batch in self.val_loader:
                log_probs = self.network.log_prob(obs_batch, action_batch)
                loss = -log_probs.mean()
                total_loss += loss.item()
                num_batches += 1
        
        self.network.train()
        avg_loss = total_loss / num_batches
        
        return avg_loss

def evaluate_reward_env(network, num_episodes=10):
    """Evaluate reward in environment using mean actions"""
    try:
        import gymnasium as gym
        env = gym.make('HalfCheetah-v4')
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    action = network.get_mean_action(obs_tensor).squeeze(0).numpy()
                    action = np.clip(action, -1, 1)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            episode_rewards.append(episode_reward)
        
        env.close()
        return np.array(episode_rewards)
        
    except Exception as e:
        print(f"Could not evaluate reward: {e}")
        return None

def main():
    print("=== GMM IMPLEMENTATION ===")
    print("Loading data...")
    
    data = torch.load('halfcheetah_v4_data.pt', weights_only=False)
    observations = data['observations']
    actions = data['actions']
    target_reward = data['mean_reward']
    
    print(f"Total samples: {len(observations)}")
    print(f"Target reward: {target_reward:.2f}")
    
    # Gaussian ablation (what nabil said)
    gaussian_counts = [1, 3, 5, 8, 10]
    results = {}
    
    for num_gaussians in gaussian_counts:
        print(f"\n--- Testing {num_gaussians} Gaussians ---")

        dataset = TensorDataset(observations, actions)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # train GMM
        network = GMM_MLP(obs_dim=17, action_dim=6, num_gaussians=num_gaussians)
        trainer = GMM_Trainer(network, train_loader, val_loader, lr=1e-4, 
                             log_dir=f"runs/gmm_ablation_{num_gaussians}")
        
        for epoch in range(30):
            for obs_batch, action_batch in train_loader:
                trainer.train_step(obs_batch, action_batch)
        
        rewards = evaluate_reward_env(network, num_episodes=5)
        if rewards is not None:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            results[num_gaussians] = {'mean': mean_reward, 'std': std_reward}
            print(f"  {num_gaussians} Gaussians: {mean_reward:.1f} ± {std_reward:.1f}")
        
        trainer.writer.close()
    
    # plot ablation results
    if results:
        gaussians = list(results.keys())
        means = [results[g]['mean'] for g in gaussians]
        stds = [results[g]['std'] for g in gaussians]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(gaussians, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
        plt.fill_between(gaussians, np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds), alpha=0.3)
        plt.xlabel('Number of Gaussians')
        plt.ylabel('Mean Reward')
        plt.title('GMM: Gaussian Ablation Study')
        plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
        plt.axhline(y=8500, color='orange', linestyle='--', alpha=0.7, label='Goal: 8500')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('gmm_gaussian_ablation.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
