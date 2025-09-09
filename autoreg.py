# autoreg.py
# Discretization approach for imitation learning
# Bins continuous actions and predicts with cross-entropy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

class ActionDiscretizer:
    def __init__(self, action_dim=6, num_bins=256, action_range=(-1, 1)):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.action_min, self.action_max = action_range
        
        # Create bin centers
        self.bin_edges = torch.linspace(self.action_min, self.action_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
    def discretize(self, actions):
        """Convert continuous actions to discrete bin indices"""
        actions_clamped = torch.clamp(actions, self.action_min, self.action_max)
        bin_indices = torch.bucketize(actions_clamped, self.bin_edges[1:-1])
        return torch.clamp(bin_indices, 0, self.num_bins - 1).long()
    
    def undiscretize(self, bin_indices):
        """Convert bin indices back to continuous actions"""
        return self.bin_centers[bin_indices]

class DiscreteMLP(nn.Module):
    def __init__(self, obs_dim=17, action_dim=6, hidden_dim=256, num_bins=256):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        # Simple MLP architecture
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * num_bins)
        )
        
    def forward(self, obs):
        # Predict logits for all action dimensions
        logits = self.net(obs)
        return logits.view(-1, self.action_dim, self.num_bins)

class DiscreteTrainer:
    def __init__(self, network, discretizer, train_loader, val_loader, lr=1e-4):
        self.network = network
        self.discretizer = discretizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        # Track training progress
        self.train_losses = []
        self.val_losses = []
        self.epoch_rewards = []
        
    def train_step(self, obs_batch, action_batch):
        self.optimizer.zero_grad()
        
        discrete_actions = self.discretizer.discretize(action_batch)
        logits = self.network(obs_batch)
        
        # Cross-entropy loss for each action dimension
        total_loss = 0
        for i in range(self.network.action_dim):
            loss = nn.CrossEntropyLoss()(logits[:, i, :], discrete_actions[:, i])
            total_loss += loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(self):
        self.network.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for obs_batch, action_batch in self.val_loader:
                discrete_actions = self.discretizer.discretize(action_batch)
                logits = self.network(obs_batch)
                
                batch_loss = 0
                for i in range(self.network.action_dim):
                    loss = nn.CrossEntropyLoss()(logits[:, i, :], discrete_actions[:, i])
                    batch_loss += loss
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        self.network.train()
        return total_loss / num_batches

def evaluate_reward_env(model, discretizer, num_episodes=10):
    """Evaluate reward in environment using discrete actions"""
    try:
        import gymnasium as gym
        env = gym.make('HalfCheetah-v4')
        episode_rewards = []
        
        model.eval()
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    # Get action logits and use argmax for deterministic policy
                    logits = model(obs_tensor)
                    discrete_actions = torch.argmax(logits, dim=2).squeeze(0)
                    continuous_actions = discretizer.undiscretize(discrete_actions).numpy()
                    
                    # Check for NaN/inf values
                    if not np.isfinite(continuous_actions).all():
                        print(f"Non-finite action at episode={episode}, step={step_count}")
                        env.close()
                        return None
                    
                    continuous_actions = np.clip(continuous_actions, -1, 1)
                
                obs, reward, terminated, truncated, _ = env.step(continuous_actions)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            if not np.isfinite(episode_reward):
                print(f"Non-finite reward detected")
                env.close()
                return None
            
            episode_rewards.append(episode_reward)
        
        env.close()
        return np.array(episode_rewards)
        
    except Exception as e:
        print(f"Could not evaluate reward: {e}")
        return None

def main():
    print("Loading data...")
    
    # Load training data
    data = torch.load('halfcheetah_v4_data.pt', weights_only=False)
    observations = data['observations']
    actions = data['actions']
    target_reward = data['mean_reward']
    
    print(f"Total samples: {len(observations)}")
    print(f"Target reward: {target_reward:.2f}")
    
    # Train/validation split
    dataset = TensorDataset(observations, actions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Initialize model components
    discretizer = ActionDiscretizer(action_dim=6, num_bins=256)
    network = DiscreteMLP(obs_dim=17, action_dim=6, hidden_dim=256, num_bins=256)
    trainer = DiscreteTrainer(network, discretizer, train_loader, val_loader, lr=1e-4)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print("Starting training...")
    
    # Training loop with early stopping
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        epoch_train_loss = 0
        num_train_batches = 0
        
        for obs_batch, action_batch in trainer.train_loader:
            loss = trainer.train_step(obs_batch, action_batch)
            epoch_train_loss += loss
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches
        
        # Validation phase
        avg_val_loss = trainer.validate()
        
        # Store metrics
        trainer.train_losses.append(avg_train_loss)
        trainer.val_losses.append(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'epoch_rewards': trainer.epoch_rewards
            }, 'best_discrete_model.pt')
        else:
            patience_counter += 1
        
        # Evaluate reward periodically
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Evaluating reward...")
            rewards = evaluate_reward_env(network, discretizer, num_episodes=5)
            if rewards is not None:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                trainer.epoch_rewards.append((epoch, mean_reward, std_reward))
                print(f"  Reward: {mean_reward:.1f} ± {std_reward:.1f}")
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load('best_discrete_model.pt', weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training progress
    if trainer.epoch_rewards:
        epochs, rewards, stds = zip(*trainer.epoch_rewards)
        plt.figure(figsize=(10, 6))
        plt.errorbar(epochs, rewards, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
        plt.fill_between(epochs, np.array(rewards) - np.array(stds), np.array(rewards) + np.array(stds), alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Reward')
        plt.title('Discretization Training: Reward Progress During Training')
        plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
        plt.axhline(y=8500, color='orange', linestyle='--', alpha=0.7, label='Goal: 8500')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('discretization_reward_over_training.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # final eval
    print("\n=== final results ===")
    final_rewards = evaluate_reward_env(network, discretizer, num_episodes=20)
    
    if final_rewards is not None:
        final_mean = np.mean(final_rewards)
        final_std = np.std(final_rewards)
        
        print(f"average reward: {final_mean:.2f} ± {final_std:.2f}")
        print(f"target: {target_reward:.2f}")
        print(f"achievement: {final_mean/target_reward*100:.1f}%")
        print(f"goal (8500+): {'✓' if final_mean >= 8500 else '✗'}")
        
    else:
        print("evaluation failed!")
    
    print(f"\nbins used: {discretizer.num_bins}")

if __name__ == "__main__":
    main()
