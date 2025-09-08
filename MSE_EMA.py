# mse with ema for imitation learning
# simple baseline approach

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

class MSE_MLP(nn.Module):
    def __init__(self, obs_dim=17, action_dim=6, hidden_dim=256):
        super().__init__()
        
        # direct obs -> action mapping
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs):
        return self.net(obs)

class EMAModel:
    """exponential moving average for smoother training"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.ema_model = copy.deepcopy(model)
        
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, current_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * current_param.data
    
    def eval_mode(self):
        self.ema_model.eval()
        return self.ema_model
    
    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

class MSE_Trainer:
    def __init__(self, network, train_loader, val_loader, lr=1e-4, log_dir="runs/mse", ema_decay=0.999):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        self.ema = EMAModel(network, decay=ema_decay)
        
        self.writer = SummaryWriter(log_dir)
        
        self.train_losses = []
        self.val_losses = []
        self.epoch_rewards = []
        
    def train_step(self, obs_batch, action_batch):
        self.optimizer.zero_grad()
        
        pred_actions = self.network(obs_batch)
        loss = nn.MSELoss()(pred_actions, action_batch)
        
        loss.backward()
        self.optimizer.step()
        
        # update ema after each step
        self.ema.update(self.network)
        
        return loss.item()
    
    def validate(self):
        self.network.eval()
        total_loss = 0
        total_loss_ema = 0
        num_batches = 0
        
        with torch.no_grad():
            for obs_batch, action_batch in self.val_loader:
                pred_actions = self.network(obs_batch)
                loss = nn.MSELoss()(pred_actions, action_batch)
                total_loss += loss.item()
                
                # ema validation
                pred_actions_ema = self.ema(obs_batch)
                loss_ema = nn.MSELoss()(pred_actions_ema, action_batch)
                total_loss_ema += loss_ema.item()
                
                num_batches += 1
        
        self.network.train()
        avg_loss = total_loss / num_batches
        avg_loss_ema = total_loss_ema / num_batches
        
        return avg_loss, avg_loss_ema

def evaluate_reward_env(model, num_episodes=10, use_ema=True):
    # evaluate in halfcheetah
    try:
        import gymnasium as gym
        env = gym.make('HalfCheetah-v4')
        episode_rewards = []
        
        eval_model = model
        if hasattr(model, 'ema') and use_ema:
            eval_model = model.ema.eval_mode()
        elif hasattr(model, 'eval'):
            eval_model = model
            eval_model.eval()
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    action = eval_model(obs_tensor).squeeze(0).numpy()
                    
                    if not np.isfinite(action).all():
                        print(f"Non-finite action detected at episode={episode}, step={step_count}")
                        env.close()
                        return None
                    
                    action = np.clip(action, -1, 1)
                
                obs, reward, terminated, truncated, _ = env.step(action)
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
    
    network = MSE_MLP(obs_dim=17, action_dim=6)
    
    # quick lr search
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
    best_lr = 1e-4
    best_early_reward = 0
    
    print("Testing learning rates...")
    for lr in learning_rates:
        print(f"Testing LR: {lr}")
        test_network = MSE_MLP(obs_dim=17, action_dim=6)
        test_trainer = MSE_Trainer(test_network, train_loader, val_loader, lr=lr, 
                                 log_dir=f"runs/mse_lr_test_{lr}")
        
        for epoch in range(5):
            for obs_batch, action_batch in train_loader:
                test_trainer.train_step(obs_batch, action_batch)
        
        rewards = evaluate_reward_env(test_trainer, num_episodes=3, use_ema=True)
        if rewards is not None:
            mean_reward = np.mean(rewards)
            print(f"  LR {lr}: {mean_reward:.1f}")
            if mean_reward > best_early_reward:
                best_early_reward = mean_reward
                best_lr = lr
        
        test_trainer.writer.close()
    
    print(f"Using best LR: {best_lr}")
    
    trainer = MSE_Trainer(network, train_loader, val_loader, lr=best_lr, ema_decay=0.999)
    
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print("Starting training with EMA...")
    
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    reward_check_freq = 5
    
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        num_train_batches = 0
        
        for batch_idx, (obs_batch, action_batch) in enumerate(train_loader):
            loss = trainer.train_step(obs_batch, action_batch)
            epoch_train_loss += loss
            num_train_batches += 1
            
            global_step = epoch * len(train_loader) + batch_idx
            trainer.writer.add_scalar('Batch/Train_Loss', loss, global_step)
        
        avg_train_loss = epoch_train_loss / num_train_batches
        
        avg_val_loss, avg_val_loss_ema = trainer.validate()
        
        trainer.train_losses.append(avg_train_loss)
        trainer.val_losses.append(avg_val_loss)
        
        trainer.writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
        trainer.writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
        trainer.writer.add_scalar('Epoch/Val_Loss_EMA', avg_val_loss_ema, epoch)
        trainer.writer.add_scalar('Epoch/Loss_Gap', avg_train_loss - avg_val_loss, epoch)
        trainer.writer.add_scalar('Epoch/Loss_Gap_EMA', avg_train_loss - avg_val_loss_ema, epoch)
        
        # check reward periodically
        if epoch % reward_check_freq == 0:
            print(f"Epoch {epoch}: Evaluating reward...")
            
            rewards_regular = evaluate_reward_env(trainer.network, num_episodes=5, use_ema=False)
            rewards_ema = evaluate_reward_env(trainer, num_episodes=5, use_ema=True)
            
            if rewards_regular is not None and rewards_ema is not None:
                mean_reward_regular = np.mean(rewards_regular)
                std_reward_regular = np.std(rewards_regular)
                mean_reward_ema = np.mean(rewards_ema)
                std_reward_ema = np.std(rewards_ema)
                
                trainer.epoch_rewards.append((epoch, mean_reward_ema, std_reward_ema))
                
                trainer.writer.add_scalar('Epoch/Mean_Reward_Regular', mean_reward_regular, epoch)
                trainer.writer.add_scalar('Epoch/Reward_Std_Regular', std_reward_regular, epoch)
                trainer.writer.add_scalar('Epoch/Mean_Reward_EMA', mean_reward_ema, epoch)
                trainer.writer.add_scalar('Epoch/Reward_Std_EMA', std_reward_ema, epoch)
                
                print(f"  Regular: {mean_reward_regular:.1f} ± {std_reward_regular:.1f}")
                print(f"  EMA:     {mean_reward_ema:.1f} ± {std_reward_ema:.1f}")
        
        if epoch % 10 == 0:
            val_obs, val_actions = next(iter(val_loader))
            with torch.no_grad():
                pred_actions = network(val_obs[:100])
                pred_actions_ema = trainer.ema(val_obs[:100])
                
                action_mse = nn.MSELoss()(pred_actions, val_actions[:100]).item()
                action_mse_ema = nn.MSELoss()(pred_actions_ema, val_actions[:100]).item()
                
                trainer.writer.add_scalar('Epoch/Action_MSE', action_mse, epoch)
                trainer.writer.add_scalar('Epoch/Action_MSE_EMA', action_mse_ema, epoch)
                
                trainer.writer.add_histogram('Actions/True_Actions', val_actions[:100], epoch)
                trainer.writer.add_histogram('Actions/Pred_Actions', pred_actions, epoch)
                trainer.writer.add_histogram('Actions/Pred_Actions_EMA', pred_actions_ema, epoch)
        
        # early stopping based on ema validation
        if avg_val_loss_ema < best_val_loss:
            best_val_loss = avg_val_loss_ema
            patience_counter = 0
            torch.save({
                'model_state_dict': network.state_dict(),
                'ema_state_dict': trainer.ema.ema_model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_loss_ema': avg_val_loss_ema,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'epoch_rewards': trainer.epoch_rewards
            }, 'best_mse_ema_model.pt')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val EMA: {avg_val_loss_ema:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("\nLoading best EMA model for final evaluation...")
    checkpoint = torch.load('best_mse_ema_model.pt', weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    trainer.ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])
    
    # plot reward progress during training
    if trainer.epoch_rewards:
        epochs, rewards, stds = zip(*trainer.epoch_rewards)
        plt.figure(figsize=(10, 6))
        plt.errorbar(epochs, rewards, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
        plt.fill_between(epochs, np.array(rewards) - np.array(stds), np.array(rewards) + np.array(stds), alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Reward')
        plt.title('MSE+EMA: Reward Over Training')
        plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
        plt.axhline(y=8500, color='orange', linestyle='--', alpha=0.7, label='Goal: 8500')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('mse_ema_reward_over_training.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # final comparison between regular and ema models
    print("\n=== FINAL EVALUATION ===")
    
    final_rewards_regular = evaluate_reward_env(network, num_episodes=20, use_ema=False)
    final_rewards_ema = evaluate_reward_env(trainer, num_episodes=20, use_ema=True)
    
    if final_rewards_regular is not None and final_rewards_ema is not None:
        final_mean_regular = np.mean(final_rewards_regular)
        final_std_regular = np.std(final_rewards_regular)
        final_mean_ema = np.mean(final_rewards_ema)
        final_std_ema = np.std(final_rewards_ema)
        
        print(f"Regular MSE: {final_mean_regular:.2f} ± {final_std_regular:.2f}")
        print(f"EMA MSE:     {final_mean_ema:.2f} ± {final_std_ema:.2f}")
        print(f"Target:      {target_reward:.2f}")
        print(f"Regular achievement: {final_mean_regular/target_reward*100:.1f}%")
        print(f"EMA achievement:     {final_mean_ema/target_reward*100:.1f}%")
        print(f"Goal check (8500+): Regular {'✓' if final_mean_regular >= 8500 else '✗'}, EMA {'✓' if final_mean_ema >= 8500 else '✗'}")
        
        # variance reduction calculation
        variance_reduction = ((final_std_regular - final_std_ema) / final_std_regular * 100)
        print(f"Variance reduction from EMA: {variance_reduction:.1f}%")
        
        # save final metrics for tensorboard
        trainer.writer.add_scalar('Final/Average_Reward_Regular', final_mean_regular, 0)
        trainer.writer.add_scalar('Final/Average_Reward_EMA', final_mean_ema, 0)
        trainer.writer.add_scalar('Final/Std_Regular', final_std_regular, 0)
        trainer.writer.add_scalar('Final/Std_EMA', final_std_ema, 0)
        trainer.writer.add_scalar('Final/Variance_Reduction_Percent', variance_reduction, 0)
        
        # create comparison plot
        plt.figure(figsize=(10, 6))
        methods = ['Regular MSE', 'MSE + EMA']
        means = [final_mean_regular, final_mean_ema]
        stds = [final_std_regular, final_std_ema]
        colors = ['lightcoral', 'lightblue']
        
        bars = plt.bar(methods, means, yerr=stds, capsize=10, alpha=0.7, color=colors)
        plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
        plt.axhline(y=8500, color='orange', linestyle='--', alpha=0.7, label='Goal: 8500')
        plt.ylabel('Mean Reward')
        plt.title('Regular vs EMA Model Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # add variance info annotation
        plt.text(1, final_mean_ema + final_std_ema + 200, 
                f'Variance reduced by {variance_reduction:.1f}%', 
                ha='center', va='bottom', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        plt.tight_layout()
        plt.savefig('regular_vs_ema_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # test different ema decay values
        print("\n=== EMA DECAY ABLATION ===")
        ema_decays = [0.99, 0.999, 0.9999]
        decay_results = {}
        
        for decay in ema_decays:
            print(f"Testing EMA decay: {decay}")
            test_net = MSE_MLP(obs_dim=17, action_dim=6, hidden_dim=256)
            test_trainer = MSE_Trainer(test_net, train_loader, val_loader, lr=best_lr,
                                     log_dir=f"runs/mse_decay_{decay}", ema_decay=decay)
            
            # train for a bit to test the decay setting
            for epoch in range(10):
                for obs_batch, action_batch in train_loader:
                    test_trainer.train_step(obs_batch, action_batch)
            
            # quick evaluation
            rewards = evaluate_reward_env(test_trainer, num_episodes=5, use_ema=True)
            if rewards is not None:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                decay_results[decay] = {'mean': mean_reward, 'std': std_reward}
                print(f"  Decay {decay}: {mean_reward:.1f} ± {std_reward:.1f}")
            
            test_trainer.writer.close()
        
        # plot decay ablation results
        if decay_results:
            decays = list(decay_results.keys())
            means = [decay_results[d]['mean'] for d in decays]
            stds = [decay_results[d]['std'] for d in decays]
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(decays, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
            plt.xlabel('EMA Decay Value')
            plt.ylabel('Mean Reward')
            plt.title('EMA Decay Parameter Study (10 epochs)')
            plt.axhline(y=target_reward, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_reward:.0f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('ema_decay_ablation.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    else:
        print("Final evaluation failed!")
    
    print(f"\nView results: tensorboard --logdir=runs/mse")
    
    trainer.writer.close()

if __name__ == "__main__":
    main()
