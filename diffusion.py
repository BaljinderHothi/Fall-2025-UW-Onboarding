# diffusion model for imitation learning
# consulted claude for ddpm math details

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import matplotlib.pyplot as plt

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class DiffusionScheduler:
    def __init__(self, num_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # linear schedule seems to work well
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise, noise

class DiffusionMLP(nn.Module):
    def __init__(self, obs_dim=17, action_dim=6, hidden_dim=256, num_timesteps=100):
        super().__init__()
        
        # time embedding helps the model know what step we're at
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )
        
        input_dim = obs_dim + action_dim + hidden_dim//4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, noisy_actions, timesteps):
        t_embed = self.time_embed(timesteps.float().unsqueeze(-1))
        x = torch.cat([obs, noisy_actions, t_embed], dim=-1)
        return self.net(x)

class DiffusionTrainer:
    def __init__(self, network, scheduler, train_loader, val_loader, lr=1e-4, log_dir="runs/diffusion"):
        self.network = network
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        self.writer = SummaryWriter(log_dir)
        
        self.train_losses = []
        self.val_losses = []
        self.train_probs = []
        self.val_probs = []
        self.epoch_rewards = []
        
    def train_step(self, obs_batch, action_batch):
        self.optimizer.zero_grad()
        
        batch_size = obs_batch.shape[0]
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,))
        
        noisy_actions, true_noise = self.scheduler.add_noise(action_batch, t)
        pred_noise = self.network(obs_batch, noisy_actions, t)
        
        loss = nn.MSELoss()(pred_noise, true_noise)
        
        # track probability metric
        neg_log_prob = loss.item()
        prob_a_given_s = np.exp(-neg_log_prob)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), prob_a_given_s
    
    def validate(self):
        self.network.eval()
        total_loss = 0
        total_prob = 0
        num_batches = 0
        
        with torch.no_grad():
            for obs_batch, action_batch in self.val_loader:
                batch_size = obs_batch.shape[0]
                t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,))
                
                noisy_actions, true_noise = self.scheduler.add_noise(action_batch, t)
                pred_noise = self.network(obs_batch, noisy_actions, t)
                
                loss = nn.MSELoss()(pred_noise, true_noise)
                neg_log_prob = loss.item()
                prob_a_given_s = np.exp(-neg_log_prob)
                
                total_loss += loss.item()
                total_prob += prob_a_given_s
                num_batches += 1
        
        self.network.train()
        avg_loss = total_loss / num_batches
        avg_prob = total_prob / num_batches
        
        return avg_loss, avg_prob
    
    def compute_proper_log_prob(self, obs_batch, action_batch, num_steps=50):
        # approximate log probability using learned denoising
        self.network.eval()
        batch_size = obs_batch.shape[0]
        
        with torch.no_grad():
            total_log_prob = 0
            
            for step in range(num_steps):
                t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,))
                noisy_actions, true_noise = self.scheduler.add_noise(action_batch, t)
                pred_noise = self.network(obs_batch, noisy_actions, t)
                
                mse = nn.MSELoss(reduction='none')(pred_noise, true_noise)
                log_prob = -0.5 * mse.sum(dim=1)
                total_log_prob += log_prob.mean()
            
            avg_log_prob = total_log_prob / num_steps
            neg_log_prob = -avg_log_prob.item()
            prob_a_given_s = np.exp(avg_log_prob.item())
            
        self.network.train()
        return neg_log_prob, prob_a_given_s

def sample_diffusion_action(network, scheduler, obs, num_steps=20):
    # ddpm sampling with stability checks
    network.eval()
    batch_size = obs.shape[0]
    
    actions = torch.randn(batch_size, 6)
    
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if num_steps > scheduler.num_timesteps:
        print(f"Warning: num_steps ({num_steps}) > scheduler timesteps ({scheduler.num_timesteps})")
    
    with torch.no_grad():
        if num_steps == 1:
            timesteps = [scheduler.num_timesteps // 2]
        else:
            timesteps = torch.linspace(scheduler.num_timesteps-1, 0, num_steps).long().tolist()
        
        for i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), int(t_idx), dtype=torch.long)
            
            pred_noise = network(obs, actions, t)
            
            if not torch.isfinite(pred_noise).all():
                print(f"Non-finite predicted noise at timestep {t_idx}")
                return None
            
            # denoise step
            if t_idx < len(scheduler.alphas_cumprod):
                alpha_t = scheduler.alphas_cumprod[t_idx]
                if alpha_t > 1e-8:
                    actions = (actions - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                else:
                    actions = actions - pred_noise
            
            if not torch.isfinite(actions).all():
                print(f"Non-finite actions after timestep {t_idx}")
                return None
            
            # add noise except final step
            if i < len(timesteps) - 1:
                noise_scale = 0.01 * (len(timesteps) - i) / len(timesteps)
                noise = torch.randn_like(actions) * noise_scale
                actions = actions + noise
    
    return actions

def evaluate_reward_env(network, scheduler, num_steps=20, num_episodes=10):
    # evaluate in halfcheetah
    try:
        import gymnasium as gym
        env = gym.make('HalfCheetah-v4')
        episode_rewards = []
        
        if num_steps >= 100:
            print(f"Warning: {num_steps} timesteps may cause numerical instability")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                try:
                    action = sample_diffusion_action(network, scheduler, obs_tensor, num_steps=num_steps)
                    if action is None:
                        print(f"Action sampling returned None at timesteps={num_steps}")
                        env.close()
                        return None
                    
                    action = action.squeeze(0).numpy()
                    
                    if not np.isfinite(action).all():
                        print(f"Non-finite action detected at timesteps={num_steps}, episode={episode}, step={step_count}")
                        env.close()
                        return None
                    
                    action = np.clip(action, -1, 1)
                    
                except Exception as e:
                    print(f"Action sampling failed at timesteps={num_steps}: {e}")
                    env.close()
                    return None
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            if not np.isfinite(episode_reward):
                print(f"Non-finite reward detected at timesteps={num_steps}")
                env.close()
                return None
            
            episode_rewards.append(episode_reward)
        
        env.close()
        return np.array(episode_rewards)
        
    except Exception as e:
        print(f"Could not evaluate reward with {num_steps} timesteps: {e}")
        return None

def inference_timestep_sweep(network, scheduler, timestep_counts=[1, 10, 50, 100], num_episodes=10):
    # test different numbers of denoising steps
    print("\n=== INFERENCE TIMESTEP SWEEP ===")
    
    results = {}
    valid_timesteps = []
    valid_means = []
    valid_stds = []
    
    for num_steps in timestep_counts:
        print(f"Evaluating with {num_steps} inference timesteps...")
        rewards = evaluate_reward_env(network, scheduler, num_steps=num_steps, num_episodes=num_episodes)
        
        if rewards is not None and len(rewards) > 0:
            finite_rewards = rewards[np.isfinite(rewards)]
            if len(finite_rewards) == 0:
                print(f"  {num_steps} steps: All rewards are non-finite (NaN/inf)")
                results[num_steps] = {'mean': np.nan, 'std': np.nan, 'rewards': rewards, 'valid': False}
            else:
                mean_reward = np.mean(finite_rewards)
                std_reward = np.std(finite_rewards)
                results[num_steps] = {'mean': mean_reward, 'std': std_reward, 'rewards': finite_rewards, 'valid': True}
                print(f"  {num_steps} steps: {mean_reward:.2f} ± {std_reward:.2f} ({len(finite_rewards)}/{len(rewards)} valid episodes)")
                
                valid_timesteps.append(num_steps)
                valid_means.append(mean_reward)
                valid_stds.append(std_reward)
        else:
            print(f"  {num_steps} steps: Evaluation failed (returning NaN)")
            results[num_steps] = {'mean': np.nan, 'std': np.nan, 'rewards': [], 'valid': False}
    
    # plot results
    if len(valid_timesteps) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.errorbar(valid_timesteps, valid_means, yerr=valid_stds, 
                    marker='o', markersize=8, capsize=5, capthick=2, linewidth=2, 
                    color='blue', label='Valid Results')
        plt.fill_between(valid_timesteps, 
                        np.array(valid_means) - np.array(valid_stds), 
                        np.array(valid_means) + np.array(valid_stds), 
                        alpha=0.3, color='blue')
        
        invalid_timesteps = [t for t in timestep_counts if not results[t]['valid']]
        if invalid_timesteps:
            plt.scatter(invalid_timesteps, [7000] * len(invalid_timesteps), 
                       marker='x', s=200, color='red', label='Failed/NaN Results')
            for t in invalid_timesteps:
                plt.annotate('NaN', (t, 7000), xytext=(t, 6500), 
                           ha='center', va='center', color='red', fontweight='bold')
        
        plt.xlabel('Inference Timesteps', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title('Diffusion Sampling: Reward vs Inference Timesteps', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.xscale('log')
        plt.xticks(timestep_counts, [str(t) for t in timestep_counts])
        
        target_reward = 9567.78
        plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
        plt.axhline(y=8500, color='orange', linestyle='--', alpha=0.7, label='Goal: 8500')
        
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'inference_timestep_sweep_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n=== SUMMARY ===")
        for num_steps in timestep_counts:
            result = results[num_steps]
            if result['valid']:
                print(f"{num_steps:3d} timesteps: {result['mean']:7.1f} ± {result['std']:5.1f}")
            else:
                print(f"{num_steps:3d} timesteps: {'NaN/Failed':>13s}")
        
        if valid_means:
            best_idx = np.argmax(valid_means)
            best_timesteps = valid_timesteps[best_idx]
            best_reward = valid_means[best_idx]
            print(f"\nOptimal setting: {best_timesteps} timesteps -> {best_reward:.1f} reward")
    
    else:
        print("No valid results to plot!")
    
    return results

def main():
    print("Loading data...")
    
    data = torch.load('halfcheetah_v4_data.pt', weights_only=False)
    observations = data['observations']
    actions = data['actions']
    target_reward = data['mean_reward']
    
    print(f"Total samples: {len(observations)}")
    print(f"Target reward: {target_reward:.2f}")
    
    dataset = TensorDataset(observations, actions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    scheduler = DiffusionScheduler(num_timesteps=100)
    network = DiffusionMLP(obs_dim=17, action_dim=6)
    
    # quick lr search
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
    best_lr = 1e-4
    best_early_reward = 0
    
    print("Testing learning rates...")
    for lr in learning_rates:
        print(f"Testing LR: {lr}")
        test_network = DiffusionMLP(obs_dim=17, action_dim=6)
        test_trainer = DiffusionTrainer(test_network, scheduler, train_loader, val_loader, lr=lr, 
                                     log_dir=f"runs/lr_test_{lr}")
        
        for epoch in range(5):
            for obs_batch, action_batch in train_loader:
                test_trainer.train_step(obs_batch, action_batch)
        
        rewards = evaluate_reward_env(test_network, scheduler, num_steps=10, num_episodes=3)
        if rewards is not None:
            mean_reward = np.mean(rewards)
            print(f"  LR {lr}: {mean_reward:.1f}")
            if mean_reward > best_early_reward:
                best_early_reward = mean_reward
                best_lr = lr
        
        test_trainer.writer.close()
    
    print(f"Using best LR: {best_lr}")
    
    trainer = DiffusionTrainer(network, scheduler, train_loader, val_loader, lr=best_lr)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print("Starting training...")
    
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    reward_check_freq = 5
    
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_train_prob = 0
        num_train_batches = 0
        
        for batch_idx, (obs_batch, action_batch) in enumerate(train_loader):
            loss, prob = trainer.train_step(obs_batch, action_batch)
            epoch_train_loss += loss
            epoch_train_prob += prob
            num_train_batches += 1
            
            global_step = epoch * len(train_loader) + batch_idx
            trainer.writer.add_scalar('Batch/Train_Loss', loss, global_step)
            trainer.writer.add_scalar('Batch/Train_Prob', prob, global_step)
        
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_prob = epoch_train_prob / num_train_batches
        
        avg_val_loss, avg_val_prob = trainer.validate()
        
        trainer.train_losses.append(avg_train_loss)
        trainer.val_losses.append(avg_val_loss)
        trainer.train_probs.append(avg_train_prob)
        trainer.val_probs.append(avg_val_prob)
        
        trainer.writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
        trainer.writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
        trainer.writer.add_scalar('Epoch/Train_Prob_As_Given_S', avg_train_prob, epoch)
        trainer.writer.add_scalar('Epoch/Val_Prob_As_Given_S', avg_val_prob, epoch)
        trainer.writer.add_scalar('Epoch/Loss_Gap', avg_train_loss - avg_val_loss, epoch)
        
        # check reward periodically
        if epoch % reward_check_freq == 0:
            print(f"Epoch {epoch}: Evaluating reward...")
            rewards = evaluate_reward_env(network, scheduler, num_steps=20, num_episodes=5)
            if rewards is not None:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                trainer.epoch_rewards.append((epoch, mean_reward, std_reward))
                
                trainer.writer.add_scalar('Epoch/Mean_Reward', mean_reward, epoch)
                trainer.writer.add_scalar('Epoch/Reward_Std', std_reward, epoch)
                
                print(f"  Reward: {mean_reward:.1f} ± {std_reward:.1f}")
        
        if epoch % 10 == 0:
            val_obs, val_actions = next(iter(val_loader))
            neg_log_prob, proper_prob = trainer.compute_proper_log_prob(val_obs[:100], val_actions[:100])
            
            trainer.writer.add_scalar('Epoch/Neg_Log_Prob_Proper', neg_log_prob, epoch)
            trainer.writer.add_scalar('Epoch/Prob_As_Given_S_Proper', proper_prob, epoch)
            
            sampled_actions = sample_diffusion_action(network, scheduler, val_obs[:10])
            if sampled_actions is not None:
                action_mse = nn.MSELoss()(sampled_actions, val_actions[:10]).item()
                trainer.writer.add_scalar('Epoch/Action_MSE', action_mse, epoch)
                
                trainer.writer.add_histogram('Actions/True_Actions', val_actions[:100], epoch)
                trainer.writer.add_histogram('Actions/Sampled_Actions', sampled_actions, epoch)
        
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
            }, 'best_diffusion_model.pt')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Train P(a*|s): {avg_train_prob:.6f} | Val P(a*|s): {avg_val_prob:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load('best_diffusion_model.pt', weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    
    if trainer.epoch_rewards:
        epochs, rewards, stds = zip(*trainer.epoch_rewards)
        plt.figure(figsize=(10, 6))
        plt.errorbar(epochs, rewards, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
        plt.fill_between(epochs, np.array(rewards) - np.array(stds), np.array(rewards) + np.array(stds), alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Reward')
        plt.title('Reward Progress During Training')
        plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
        plt.axhline(y=8500, color='orange', linestyle='--', alpha=0.7, label='Goal: 8500')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward_over_training.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    sweep_results = inference_timestep_sweep(network, scheduler, timestep_counts=[1, 10, 50, 100], num_episodes=10)
    
    print("\n=== FINAL EVALUATION ===")
    valid_results = {k: v for k, v in sweep_results.items() if v['valid']}
    if valid_results:
        best_timesteps = max(valid_results.keys(), key=lambda k: valid_results[k]['mean'])
        print(f"Best inference timesteps: {best_timesteps}")
        
        final_rewards = evaluate_reward_env(network, scheduler, num_steps=best_timesteps, num_episodes=20)
        if final_rewards is not None:
            final_mean = np.mean(final_rewards)
            final_std = np.std(final_rewards)
            
            print(f"FINAL RESULTS (with {best_timesteps} timesteps):")
            print(f"Average Reward: {final_mean:.2f} ± {final_std:.2f}")
            print(f"Target Reward: {target_reward:.2f}")
            print(f"Achievement: {final_mean/target_reward*100:.1f}%")
            print(f"Goal Achievement (8500+): {'✓' if final_mean >= 8500 else '✗'}")
            
            trainer.writer.add_scalar('Final/Average_Reward', final_mean, 0)
            trainer.writer.add_scalar('Final/Best_Timesteps', best_timesteps, 0)
            trainer.writer.add_scalar('Final/Achievement_Percent', final_mean/target_reward*100, 0)
    else:
        print("No valid timestep configurations found!")
    
    print(f"\nTo view results in TensorBoard, run:")
    print(f"tensorboard --logdir=runs/diffusion")
    
    trainer.writer.close()

if __name__ == "__main__":
    main()
