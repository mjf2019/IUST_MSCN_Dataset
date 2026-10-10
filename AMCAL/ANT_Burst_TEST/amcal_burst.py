"""Burst AMCAL: existing oversampled splits and original query gate.
No training or evaluation runs on import.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
from collections import deque
import os
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import joblib
from sklearn.model_selection import GroupShuffleSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define 1D-CNN model
class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.flatten = nn.Flatten()
        output_length = input_length // 8
        self.dense_layers = nn.Sequential(
            nn.Linear(256 * output_length, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x

# Define ContextAwareNetwork
class ContextAwareNetwork(nn.Module):
    def __init__(self, input_size, num_layers=4, dim_feedforward=1024):
        super(ContextAwareNetwork, self).__init__()
        self.input_size = input_size
        possible_heads = [8, 4, 2, 1]
        num_heads = next(h for h in possible_heads if input_size % h == 0)
        print(f"Selected num_heads={num_heads} for input_size={input_size}")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.3,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Flatten(),
            nn.Linear(128 * input_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.ReLU(),
            nn.LayerNorm(input_size)
        )
    
    def forward(self, x):
        x_flat = x.squeeze(1)
        x_transformed = self.transformer(x_flat.unsqueeze(1)).squeeze(1)
        weights = torch.sigmoid(x_transformed)
        combined = torch.stack([x_flat, weights], dim=1)
        encoded = self.encoder(combined)
        return encoded.unsqueeze(1)

# Define DQNSelector
class DQNSelector(nn.Module):
    def __init__(self, input_size):
        super(DQNSelector, self).__init__()
        hidden_size1 = min(512, max(128, input_size * 2))
        hidden_size2 = min(256, hidden_size1 // 2)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# Replay Buffer for DQN
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = state.squeeze().cpu()
        next_state = next_state.squeeze().cpu()
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        states = torch.stack([torch.FloatTensor(s) for s in state])
        next_states = torch.stack([torch.FloatTensor(s) for s in next_state])
        return (
            states,
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            next_states,
            torch.tensor(done, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNSelectorAgent:
    def __init__(self, input_size, device, lr=1e-4, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_capacity=50000, batch_size=256):
        self.device = device
        self.policy_net = DQNSelector(input_size).to(device)
        self.target_net = DQNSelector(input_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
    
    def select_action(self, state, explore=True):
        """Select action for given state. Always returns a tensor."""
        self.steps_done += 1
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        if explore and random.random() < self.epsilon:
            return torch.randint(0, 2, (state.size(0),), device=self.device)
        else:
            with torch.no_grad():
                was_training = self.policy_net.training
                self.policy_net.eval()
                q_values = self.policy_net(state)
                self.policy_net.train(was_training)
                return q_values.argmax(dim=1)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            was_training = self.policy_net.training
            self.policy_net.eval()
            next_actions = self.policy_net(next_states).argmax(1)
            self.policy_net.train(was_training)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Load and preprocess data - GENERALIZED version
def load_data(dataset_path='Dataset', csv_filename=None):
    train_df, val_df, test_df = read_burst_splits(dataset_path)
    X_train, y_train = train_df.iloc[:, :-1].to_numpy(), train_df.iloc[:, -1].to_numpy()
    X_val, y_val = val_df.iloc[:, :-1].to_numpy(), val_df.iloc[:, -1].to_numpy()
    X_test, y_test = test_df.iloc[:, :-1].to_numpy(), test_df.iloc[:, -1].to_numpy()

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    
    print(f"Detected {num_classes} classes in the dataset")
    print(f"Classes: {label_encoder.classes_}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    save_preprocessing(train_df, val_df, test_df, dataset_path, scaler,
                       label_encoder, np.bincount(y_train, minlength=num_classes))

    if len(train_df) < 128:
        raise ValueError("Training partition needs at least 128 rows for the current batch size.")
    if not all(np.isfinite(array).all() for array in (X_train, X_val, X_test)):
        raise ValueError("Features contain NaN/infinity after scaling.")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Handle class imbalance
    class_counts = np.bincount(y_train)
    if len(class_counts) < num_classes:
        class_counts = np.append(class_counts, [0] * (num_classes - len(class_counts)))
    
    weights = 1.0 / class_counts[y_train]
    train_sampler = WeightedRandomSampler(weights, len(weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, drop_last=True) 
    full_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)  
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)  
    
    print("Class distribution in training data:", np.bincount(y_train))
    print("Class distribution in validation data:", np.bincount(y_val))
    print("Class distribution in test data:", np.bincount(y_test))
    print(f"Total training samples: {X_train.shape[0]}, validation samples: {X_val.shape[0]}, test samples: {X_test.shape[0]}")
    print(f"Input length: {X_train.shape[1]}")
    print(f"Full dataset size for adversarial training: {len(train_dataset)}")
    
    return train_loader, val_loader, test_loader, full_loader, num_classes, X_train.shape[1], class_counts

# Compute combined loss
def compute_combined_loss(outputs, labels, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    per_sample_loss = criterion(outputs, labels)
    mean_loss = per_sample_loss.mean()
    return mean_loss, per_sample_loss

# Train initial CNN model
def train_initial_cnn(cnn_model, train_loader, val_loader, class_counts, epochs=500, patience=50):
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0005, weight_decay=1e-4)  
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        cnn_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        cnn_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        print(f"Initial CNN Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(cnn_model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                if best_model_state is not None:
                    cnn_model.load_state_dict(best_model_state)
                break
        
        scheduler.step()
    
    if best_model_state is not None:
        cnn_model.load_state_dict(best_model_state)
    return cnn_model

# Evaluate model
def evaluate_model(model, test_loader, class_weights, context_model=None, dqn_agent=None, threshold=None):
    model.eval()
    if context_model is not None:
        context_model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    total_samples = 0
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if context_model and dqn_agent and threshold is not None:
                cnn_outputs = model(inputs)
                probs = softmax(cnn_outputs)
                
                encoded_inputs = context_model(inputs)
                outputs = model(encoded_inputs)
                
                _, predicted = torch.max(outputs, 1)
                final_loss = criterion(outputs, labels)
                running_loss += final_loss.item() * batch_size
            else:
                outputs = model(inputs)
                final_loss = criterion(outputs, labels)
                running_loss += final_loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    test_loss = running_loss / total_samples
    
    return {
        'Test Loss': test_loss,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def compute_reward(action, cnn_pred, weighted_pred, label):
    if action == 1:
        if weighted_pred != label and cnn_pred != label:
            reward = 3.0
        elif weighted_pred == label and cnn_pred != label:
            reward = 2.0
        elif weighted_pred != label and cnn_pred == label:
            reward = -2.0
        else:
            reward = 0.0
    else:
        reward = 0.0  # Eq. (19): unqueried action needs no label.
    return reward

def train_adversarial_system(cnn_model, dqn_agent, context_model, full_loader, val_loader, class_counts, input_length, num_classes, fine_tune_epochs=500, patience=50, pretrain_epochs=20):
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    cnn_criterion = nn.CrossEntropyLoss(weight=class_weights)
    context_optimizer = optim.Adam(context_model.parameters(), lr=5e-4, weight_decay=1e-4)
    context_scheduler = lr_scheduler.CosineAnnealingLR(context_optimizer, T_max=100)
    
    best_val_accuracy = float('-inf')
    epochs_no_improve = 0
    softmax = nn.Softmax(dim=1)
    best_context_state = None
    best_dqn_state = None
    
    cnn_model.eval()
    for parameter in cnn_model.parameters():
        parameter.requires_grad_(False)

    # Pre-train ContextAwareNetwork
    print("Pre-training ContextAwareNetwork...")
    for pretrain_epoch in range(pretrain_epochs):
        context_model.train()
        total_context_loss = 0.0
        num_batches = 0
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            context_optimizer.zero_grad()
            encoded_inputs = context_model(inputs)
            cnn_outputs_weighted = cnn_model(encoded_inputs)
            context_loss = cnn_criterion(cnn_outputs_weighted, labels)
            context_loss.backward()
            context_optimizer.step()
            total_context_loss += context_loss.item()
            num_batches += 1
        context_scheduler.step()
        print(f"Pretrain Epoch {pretrain_epoch+1}/{pretrain_epochs}, Loss: {total_context_loss/num_batches:.4f}")
    
    print("Starting adversarial training on full dataset...")
    # Eq. (18): offline target synchronization every 15 epochs.
    
    for epoch in range(fine_tune_epochs):
        cnn_model.eval()
        context_model.train()
        dqn_agent.policy_net.train()
        
        running_context_loss = 0.0
        running_dqn_loss = 0.0
        dqn_loss_count = 0
        total_samples = 0
        context_updated_samples = 0
        rewards = []
        actions = []
        context_updated = False
        
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Eqs. (13)-(14): uncertainty from the modulated input.
            with torch.no_grad():
                cnn_outputs = cnn_model(inputs).detach()
                cnn_pred = cnn_outputs.argmax(dim=1)
                previous_mode = context_model.training
                context_model.eval()
                weighted_before = cnn_model(context_model(inputs))
                uncertainty = 1.0 - weighted_before.softmax(dim=1).max(dim=1, keepdim=True).values
                context_model.train(previous_mode)
            state = torch.cat([inputs.squeeze(1), uncertainty], dim=1).detach()
            actions_batch = dqn_agent.select_action(state)
            actions.extend(actions_batch.cpu().tolist())

            # Find indices where action=1
            action_mask = (actions_batch == 1).cpu()
            action_indices = action_mask.nonzero(as_tuple=True)[0]
            
            # Update ContextAwareNetwork only for samples with action=1
            if len(action_indices) > 0:
                context_updated = True
                context_updated_samples += len(action_indices)
                
                selected_inputs = inputs[action_indices]
                selected_labels = labels[action_indices]
                
                # Handle batch size 1
                if selected_inputs.size(0) == 1:
                    selected_inputs = torch.cat([selected_inputs, selected_inputs], dim=0)
                    selected_labels = torch.cat([selected_labels, selected_labels], dim=0)
                
                context_optimizer.zero_grad()
                encoded_inputs = context_model(selected_inputs)
                cnn_outputs_weighted = cnn_model(encoded_inputs)
                context_loss = cnn_criterion(cnn_outputs_weighted, selected_labels)
                orig_cnn_loss = cnn_criterion(cnn_model(selected_inputs), selected_labels)
                epsilon = 1e-8
                total_context_loss = context_objective(context_loss, orig_cnn_loss)
                total_context_loss.backward()
                context_optimizer.step()
                running_context_loss += total_context_loss.item() * len(action_indices)
            
            # Algorithm 1: recompute modulated prediction/state after context update.
            with torch.no_grad():
                previous_mode = context_model.training
                context_model.eval()
                weighted_after = cnn_model(context_model(inputs))
                weighted_pred = weighted_after.argmax(dim=1)
                uncertainty_after = 1.0-weighted_after.softmax(dim=1).max(dim=1, keepdim=True).values
                next_state_new = torch.cat([inputs.squeeze(1), uncertainty_after], dim=1).detach()
                context_model.train(previous_mode)
            for i in range(batch_size):
                selected = int(actions_batch[i].item())
                # Short-circuit: action 0 has reward 0 without accessing its label.
                reward = (compute_reward(1, int(cnn_pred[i].item()),
                          int(weighted_pred[i].item()), int(labels[i].item()))
                          if selected else 0.0)
                rewards.append(reward)
                next_state = next_state_new[i:i+1] if selected else state[i:i+1]
                agent_state = state[i:i+1]
                # Continuing local transition defined in the paper; no terminal flag.
                dqn_agent.replay_buffer.push(agent_state, selected, reward, next_state, 0.0)

            # Optimize DQN
            dqn_loss = dqn_agent.optimize()
            if dqn_loss is not None:
                running_dqn_loss += dqn_loss
                dqn_loss_count += 1
        
        if context_updated:
            context_scheduler.step()
        
        dqn_agent.update_epsilon()
        
        if (epoch + 1) % 15 == 0:
            dqn_agent.update_target_network()
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        action_dist = np.bincount(actions, minlength=2) if actions else [0, 0]
        
        print(f"Adversarial Epoch {epoch+1}/{fine_tune_epochs}:")
        print(f"  Context Loss: {running_context_loss / max(context_updated_samples, 1):.4f}")
        print(f"  DQN Loss: {running_dqn_loss / max(dqn_loss_count, 1):.4f}")
        print(f"  Context Updated Samples: {context_updated_samples}")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Action Distribution (skip/apply): {action_dist}")
        
        # Validation
        val_results = evaluate_model(cnn_model, val_loader, class_weights, context_model, dqn_agent, threshold=0.5)
        print(f"Validation Results at Epoch {epoch+1}:")
        print(f"  Val Loss: {val_results['Test Loss']:.4f}")
        print(f"  Val Accuracy: {val_results['Accuracy']:.4f}")
        print(f"  Val Precision: {val_results['Precision']:.4f}")
        print(f"  Val Recall: {val_results['Recall']:.4f}")
        print(f"  Val F1 Score: {val_results['F1 Score']:.4f}")
        
        if val_results['Accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['Accuracy']
            epochs_no_improve = 0
            best_context_state = deepcopy(context_model.state_dict())
            best_dqn_state = deepcopy(dqn_agent.policy_net.state_dict())
            print(f"✓ Updated best adversarial model states (Accuracy: {best_val_accuracy:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    # Load best models
    if best_context_state is not None:
        context_model.load_state_dict(best_context_state)
        print("Loaded best context model")
    
    if best_dqn_state is not None:
        dqn_agent.policy_net.load_state_dict(best_dqn_state)
        print("Loaded best DQN model")
    
    # Save best models
    os.makedirs('Models', exist_ok=True)
    
    torch.save({
        'preprocessing_sha256': file_sha256('Models/protocol_preprocessing.joblib'),
        'protocol_version': 'burst-v1',
        'context_model_state_dict': context_model.state_dict(),
        'input_length': input_length,
        'num_classes': num_classes
    }, 'Models/adversarial_best_context.pth')
    print("Saved best adversarial Context model")
    
    torch.save({
        'preprocessing_sha256': file_sha256('Models/protocol_preprocessing.joblib'),
        'protocol_version': 'burst-v1',
        'dqn_policy_net_state_dict': dqn_agent.policy_net.state_dict(),
        'input_length': input_length,
        'num_classes': num_classes
    }, 'Models/adversarial_best_dqn.pth')
    print("Saved best selector weights (online replay starts empty)")
    
    return cnn_model, dqn_agent, context_model

# Main execution
def train_main(dataset_path, base_epoch=200, base_patience=50,
               adver_epoch=200, adver_patience=20, pretrain_epochs=20):
    
    print("=" * 60)
    print("Starting Adversarial Training System for Network Traffic Classification")
    print("=" * 60)
    
    # Load data
    train_loader, val_loader, test_loader, full_loader, num_classes, input_length, class_counts = load_data(dataset_path=dataset_path)
    if train_loader is None or full_loader is None:
        raise RuntimeError("Failed to load training data; review the preceding error.")
    
    print(f"\nDataset Statistics:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Input feature length: {input_length}")
    print(f"  - Class distribution (train): {class_counts}")
    
    # Train and evaluate initial CNN
    print("\n" + "=" * 60)
    print("Training Initial CNN Model")
    print("=" * 60)
    
    cnn_model = CNN1D(input_length, num_classes).to(device)
    cnn_model = train_initial_cnn(cnn_model, train_loader, val_loader, class_counts, base_epoch, base_patience)
    
    os.makedirs('Models', exist_ok=True)
    
    torch.save({
        'preprocessing_sha256': file_sha256('Models/protocol_preprocessing.joblib'),
        'protocol_version': 'burst-v1',
        'model_state_dict': cnn_model.state_dict(),
        'input_length': input_length,
        'num_classes': num_classes
    }, 'Models/initial_cnn_model.pth')
    print("Saved initial CNN model")
    
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    initial_results = evaluate_model(cnn_model, test_loader, class_weights)
    print("\n" + "=" * 60)
    print("Initial CNN Test Results:")
    print("=" * 60)
    print(f"  Test Loss: {initial_results['Test Loss']:.4f}")
    print(f"  Accuracy: {initial_results['Accuracy']:.4f}")
    print(f"  Precision: {initial_results['Precision']:.4f}")
    print(f"  Recall: {initial_results['Recall']:.4f}")
    print(f"  F1 Score: {initial_results['F1 Score']:.4f}")
    
    # Initialize DQN and Context models
    print("\n" + "=" * 60)
    print("Initializing Adversarial Components")
    print("=" * 60)
    
    dqn_agent = DQNSelectorAgent(input_length + 1, device)
    context_model = ContextAwareNetwork(input_length).to(device)
    
    print(f"DQN Agent initialized with input size: {input_length + 1}")
    print(f"Context Model initialized with input size: {input_length}")
    
    # Train adversarial system
    print("\n" + "=" * 60)
    print("Starting Adversarial Training")
    print("=" * 60)
    
    cnn_model, dqn_agent, context_model = train_adversarial_system(
        cnn_model, dqn_agent, context_model, full_loader, val_loader, class_counts, 
        input_length, num_classes, adver_epoch, adver_patience, pretrain_epochs
    )
    
    # Evaluate adversarial system
    print("\n" + "=" * 60)
    print("Final Evaluation - Adversarial System")
    print("=" * 60)
    
    adversarial_results = evaluate_model(cnn_model, test_loader, class_weights, context_model, dqn_agent, threshold=0.5)
    print("Adversarial System Test Results:")
    print(f"  Test Loss: {adversarial_results['Test Loss']:.4f}")
    print(f"  Accuracy: {adversarial_results['Accuracy']:.4f}")
    print(f"  Precision: {adversarial_results['Precision']:.4f}")
    print(f"  Recall: {adversarial_results['Recall']:.4f}")
    print(f"  F1 Score: {adversarial_results['F1 Score']:.4f}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("Performance Improvement Summary")
    print("=" * 60)
    print(f"Accuracy Improvement: {(adversarial_results['Accuracy'] - initial_results['Accuracy']) * 100:+.2f}%")
    print(f"F1 Score Improvement: {(adversarial_results['F1 Score'] - initial_results['F1 Score']) * 100:+.2f}%")
    print(f"Precision Improvement: {(adversarial_results['Precision'] - initial_results['Precision']) * 100:+.2f}%")
    print(f"Recall Improvement: {(adversarial_results['Recall'] - initial_results['Recall']) * 100:+.2f}%")
    
    summary = dict(base=initial_results, frozen_context=adversarial_results,
                   seed=RUN_SEED, evaluation="existing oversampled test CSV; no adaptation",
                   note="Pilot results do not constitute revised manuscript results.")
    Path("Results").mkdir(exist_ok=True)
    Path("Results/training_clean_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining completed successfully!")


# Protocol v2 implements the article's same-input local DDQN transitions.
# This alignment alone does not establish meta-learning or min-max guarantees.
RUN_SEED = 42

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_burst_splits(data_dir):
    frames = [pd.read_csv(Path(data_dir)/name) for name in
        ("oversampled_train_dataset.csv","oversampled_validation_dataset.csv","oversampled_test_dataset.csv")]
    for frame in frames:
        if frame.empty or list(frame.columns) != list(frames[0].columns):
            raise ValueError("Empty split or inconsistent schema.")
        if not np.isfinite(frame.iloc[:,:-1].to_numpy(dtype=float)).all():
            raise ValueError("Features contain NaN/infinity.")
    return frames

def save_preprocessing(train, val, test, data_dir, scaler, encoder, counts):
    names=("oversampled_train_dataset.csv","oversampled_validation_dataset.csv","oversampled_test_dataset.csv")
    hashes={name:file_sha256(Path(data_dir)/name) for name in names}
    artifact=dict(protocol_version="burst-v1",seed=RUN_SEED,
        feature_names=list(train.columns[:-1]),label_name=train.columns[-1],
        scaler=scaler,label_encoder=encoder,class_counts=counts,split_sha256=hashes)
    Path("Models").mkdir(exist_ok=True)
    joblib.dump(artifact,"Models/protocol_preprocessing.joblib")
    report=dict(protocol_version="burst-v1",seed=RUN_SEED,split_sha256=hashes,
        split_sizes={name:len(frame) for name,frame in zip(("train","validation","test"),(train,val,test))},
        split_rule="existing CSV splits; no new oversampling or split",
        preprocessing="fit scaler and encoder on oversampled train only",
        architecture="original Burst CNN and single-token Context",
        training_sampler="inverse-frequency WeightedRandomSampler; no new rows",
        limitation="Existing split independence is not certified.")
    Path("Models/protocol_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

def context_objective(context_loss, original_loss):
    return 0.8 * context_loss + 0.2 / (1.0 + original_loss.detach() / (context_loss + 1e-8))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_evaluation_frame(data_dir, attack_file, artifact):
    for name, expected_hash in artifact["split_sha256"].items():
        if file_sha256(Path(data_dir)/name) != expected_hash:
            raise ValueError("A training split changed: "+name)
    path=Path(attack_file) if attack_file else Path(data_dir)/"oversampled_test_dataset.csv"
    frame=pd.read_csv(path)
    if list(frame.columns) != artifact["feature_names"]+[artifact["label_name"]]:
        raise ValueError("Schema/order differs from training.")
    return frame,np.arange(len(frame)),path,"file-local row IDs; source overlap unverified"

def evaluate_stream(args):
    output = Path(args.output).resolve()
    models = output / "Models"
    artifact_path = models / "protocol_preprocessing.joblib"
    artifact = joblib.load(artifact_path)
    if artifact.get('protocol_version') != 'burst-v1':
        raise ValueError('Preprocessing is not Burst protocol v1; retrain in a fresh directory.')
    frame, source_ids, attack_path, alignment = load_evaluation_frame(
        args.data_dir, args.attack_file, artifact)
    if args.max_samples is not None:
        frame, source_ids = frame.iloc[:args.max_samples], source_ids[:args.max_samples]
    if frame.empty:
        raise ValueError("Empty evaluation stream.")
    X = artifact["scaler"].transform(frame.iloc[:, :-1].to_numpy())
    y = artifact["label_encoder"].transform(frame.iloc[:, -1].to_numpy())
    if not np.isfinite(X).all():
        raise ValueError("Non-finite features after scaling.")
    unique_source_count = len(set(source_ids.tolist()))
    budget = int(np.floor(args.budget_fraction * unique_source_count))
    input_length, num_classes = X.shape[1], len(artifact["label_encoder"].classes_)
    digest = file_sha256(artifact_path)
    def checkpoint(name):
        saved = torch.load(models / name, map_location=device, weights_only=True)
        if saved.get("protocol_version") != 'burst-v1':
            raise ValueError("Checkpoint is not Burst protocol v1; retrain in a fresh output directory.")
        if saved.get("preprocessing_sha256") != digest:
            raise ValueError("Checkpoint/preprocessing mismatch; train with this protocol runner.")
        if saved["input_length"] != input_length or saved["num_classes"] != num_classes:
            raise ValueError("Checkpoint shape/class mismatch.")
        return saved
    cnn = CNN1D(input_length, num_classes).to(device)
    cnn.load_state_dict(checkpoint("initial_cnn_model.pth")["model_state_dict"])
    cnn.eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    context = ContextAwareNetwork(input_length).to(device)
    context.load_state_dict(checkpoint("adversarial_best_context.pth")["context_model_state_dict"])
    context.eval()
    agent = DQNSelectorAgent(input_length + 1, device,
                             batch_size=args.online_selector_batch_size)
    agent.policy_net.load_state_dict(checkpoint("adversarial_best_dqn.pth")["dqn_policy_net_state_dict"])
    agent.update_target_network()
    agent.policy_net.eval()
    # Online replay starts empty; skipped samples create label-free zero-reward transitions.
    optimizer = optim.Adam(context.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = torch.tensor(1.0 / artifact["class_counts"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    frozen_snapshot = deepcopy(cnn.state_dict())
    queried_ids, trace, predicted, base_predicted = set(), [], [], []
    selector_updates, context_updates = 0, 0
    first_query_step = last_query_step = first_selector_update_step = None
    for step, (features, truth, source_id) in enumerate(zip(X, y, source_ids), start=1):
        source_id = int(source_id)
        inputs = torch.tensor(features, dtype=torch.float32, device=device).reshape(1, 1, -1)
        # No label enters prediction, policy state, or the query decision.
        with torch.no_grad():
            base_logits = cnn(inputs)
            base_prob = base_logits.softmax(dim=1)
            base_prediction = int(base_logits.argmax(1).item())
            adapted_logits = cnn(context(inputs))
            adapted_prob = adapted_logits.softmax(dim=1)
            state = torch.cat([inputs.squeeze(1), 1.0-adapted_prob.max(1, keepdim=True).values], dim=1)
            prediction = int(adapted_logits.argmax(1).item())
            action = int(agent.select_action(state, explore=False).item())
            confidence_gap = float(abs(adapted_prob.max().item()-base_prob.max().item()))
        # Capture the scored prediction before acquiring a label or changing weights.
        predicted.append(prediction)
        base_predicted.append(base_prediction)
        adaptation_allowed = budget > 0 and len(queried_ids) < budget
        gate_passed = prediction != base_prediction and confidence_gap >= args.threshold
        query = (action == 1 and gate_passed and adaptation_allowed and source_id not in queried_ids)
        reward, loss_value, selector_loss = None, None, None
        if query:
            queried_ids.add(source_id)
            if first_query_step is None:
                first_query_step = step
            last_query_step = step
            labels = torch.tensor([int(truth)], dtype=torch.long, device=device)
            optimizer.zero_grad()
            # Keep inference-mode batch statistics and dropout during online adaptation.
            context_loss = criterion(cnn(context(inputs)), labels)
            original_loss = criterion(base_logits, labels)
            loss = context_objective(context_loss, original_loss)
            loss.backward()
            optimizer.step()
            context_updates += 1
            loss_value = float(loss.item())
            # Reward/state may use post-update outputs of this queried sample.
            # Those outputs never replace the pre-update scored prediction.
            with torch.no_grad():
                after_logits = cnn(context(inputs))
                after_prob = after_logits.softmax(dim=1)
                after_prediction = int(after_logits.argmax(1).item())
                next_state = torch.cat([inputs.squeeze(1),
                    1.0-after_prob.max(1, keepdim=True).values], dim=1)
            reward = compute_reward(1, base_prediction, after_prediction, int(truth))
            effective_action = 1
        else:
            # No label access for an unqueried sample (including exhausted budget).
            effective_action, reward, next_state = 0, 0.0, state
        # A budget-blocked action 1 is not an observed policy action 0.
        # Freeze both learners at zero budget and after budget exhaustion.
        transition_stored = adaptation_allowed and (query or action == 0)
        if transition_stored:
            agent.replay_buffer.push(state.detach(), effective_action, reward,
                                     next_state.detach(), 0.0)
            agent.policy_net.train()
            selector_loss = agent.optimize()
            agent.policy_net.eval()
            if selector_loss is not None:
                selector_updates += 1
                if first_selector_update_step is None:
                    first_selector_update_step = step
                if selector_updates % 15 == 0:
                    agent.update_target_network()
        # The evaluator may score all truths; the learner sees only queried labels.
        trace.append(dict(step=step, source_row=source_id, prediction_before_update=prediction,
            base_prediction=base_prediction, true_label=int(truth), correct=int(prediction == truth),
            action=action, effective_action=effective_action,
            confidence_gap=confidence_gap, gate_passed=gate_passed, queried=query,
            transition_stored=transition_stored, budget_blocked=not adaptation_allowed,
            replay_size=len(agent.replay_buffer),
            unique_labels_used=len(queried_ids), context_updates=context_updates,
            selector_updates=selector_updates, reward=reward,
            context_loss=loss_value, selector_loss=selector_loss))
    for name, value in cnn.state_dict().items():
        if not torch.equal(value, frozen_snapshot[name]):
            raise AssertionError("Frozen classifier weights or running statistics changed: " + name)
    metrics = dict(
        accuracy=float(accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, labels=np.arange(num_classes), average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, predicted, average="weighted", zero_division=0)),
        base_accuracy=float(accuracy_score(y, base_predicted)),
        base_macro_f1=float(f1_score(y, base_predicted, labels=np.arange(num_classes), average="macro", zero_division=0)),
        evaluated_rows=len(y), unique_stream_rows=unique_source_count,
        label_budget=budget, unique_labels_queried=len(queried_ids),
        actual_label_fraction=len(queried_ids)/unique_source_count,
        context_updates=context_updates, selector_updates=selector_updates,
        online_selector_batch_size=agent.batch_size, replay_size=len(agent.replay_buffer),
        first_query_step=first_query_step, last_query_step=last_query_step,
        first_selector_update_step=first_selector_update_step,
        classifier_unchanged=True, protocol="predict-score-query-update",
        selector_freeze_rule="freeze at zero budget or after budget exhaustion",
        alignment=alignment,
        evaluation_scope="existing Burst evaluation files", source_identity_verified=False,
        seed=args.seed, budget_fraction=args.budget_fraction, threshold=args.threshold,
        query_rule='action 1, CNN/Context disagreement, confidence gap >= threshold, budget remaining',
        online_lr=args.lr, attack_sha256=file_sha256(attack_path),
        preprocessing_sha256=digest, device=str(device),
        protocol_version='burst-v1', selector_target="DDQN gamma=0.95; same-input local transitions",
        uncertainty="1 - max modulated softmax", target_sync_steps=15,
        train_seed=artifact["seed"], torch_version=str(torch.__version__),
        numpy_version=np.__version__, pandas_version=pd.__version__)
    name = "clean" if args.attack_file is None else Path(args.attack_file).stem
    tag = f"{name}_budget{args.budget_fraction:g}_seed{args.seed}_n{len(y)}_lr{args.lr:g}_sb{args.online_selector_batch_size}_th{args.threshold:g}_burst_v1"
    results_dir = output / "Results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame(trace).to_csv(results_dir / (tag+"_trace.csv"), index=False)
    (results_dir / (tag+"_metrics.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def audit_dataset(args):
    frames=read_burst_splits(args.data_dir)
    report=dict(split_rule="existing CSV files; unchanged",
        splits={name:dict(rows=len(frame),features=frame.shape[1]-1,
            classes={str(k):int(v) for k,v in frame.iloc[:,-1].value_counts().items()})
            for name,frame in zip(("train","validation","test"),frames)},
        levels=[p.name for p in sorted((Path(args.data_dir)/"AdvBurst_FTSC-IAT").glob("*.csv"))])
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

def cli():
    global RUN_SEED, device
    parser=argparse.ArgumentParser(description="Burst AMCAL: existing oversampled data, original query gate, pre-update scoring.")
    parser.add_argument("command",choices=("audit","train","evaluate"))
    parser.add_argument("--data-dir",default=str(Path(__file__).resolve().parent/"Dataset"))
    parser.add_argument("--output",default=str(Path(__file__).resolve().parent/"protocol_runs"/"burst_original_seed42"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--base-epochs",type=int,default=200)
    parser.add_argument("--context-epochs",type=int,default=200)
    parser.add_argument("--pretrain-epochs",type=int,default=20)
    parser.add_argument("--base-patience",type=int,default=50)
    parser.add_argument("--context-patience",type=int,default=20)
    parser.add_argument("--attack-file")
    parser.add_argument("--per",type=int,choices=(0,1,3,5,7,10,12,15,17,20))
    parser.add_argument("--threshold",type=float,default=0.0005)
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--budget-fraction",type=float,default=0.20)
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--online-selector-batch-size",type=int,default=32,
                        help="Online replay minibatch size; offline training remains at 256.")
    args=parser.parse_args()
    if not 0 <= args.budget_fraction <= 1:
        parser.error("Budget fraction must be between 0 and 1.")
    if args.online_selector_batch_size < 1:
        parser.error("--online-selector-batch-size must be positive.")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive.")
    if args.base_epochs < 1 or args.context_epochs < 1 or args.pretrain_epochs < 0:
        parser.error("Training epochs must be positive; pretraining may be zero.")
    if args.base_patience < 1 or args.context_patience < 1 or args.lr <= 0:
        parser.error("Patience and learning rate must be positive.")
    RUN_SEED=args.seed
    set_seed(args.seed)
    if args.device != "auto":
        device=torch.device(args.device)
    args.data_dir=str(Path(args.data_dir).resolve())
    args.output=str(Path(args.output).resolve())
    if args.attack_file:
        args.attack_file=str(Path(args.attack_file).resolve())
    if args.per is not None:
        if args.attack_file or args.command != "evaluate":
            parser.error("--per is only for evaluate and cannot be combined with --attack-file.")
        args.attack_file=str(Path(args.data_dir)/"AdvBurst_FTSC-IAT"/
            f"AdvBurstTrainedBurst_size_{args.per}_Step_2000.csv")
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and nonnegative.")
    if args.command == "audit":
        audit_dataset(args)
    elif args.command == "train":
        destination=Path(args.output)
        if (destination/"Models").exists():
            parser.error("Models already exist in --output; choose a fresh output directory.")
        destination.mkdir(parents=True,exist_ok=True)
        previous=Path.cwd()
        try:
            os.chdir(destination)
            train_main(args.data_dir,args.base_epochs,args.base_patience,
                       args.context_epochs,args.context_patience,args.pretrain_epochs)
        finally:
            os.chdir(previous)
    else:
        evaluate_stream(args)

if __name__ == "__main__":
    cli()
