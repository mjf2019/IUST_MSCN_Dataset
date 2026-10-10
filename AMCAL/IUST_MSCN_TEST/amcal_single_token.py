# Architecture-control runner: original single-token Context, corrected protocol v2.
# Not an exact reproduction of the legacy notebook's evaluation.
"""AMCAL IUST protocol revision. No experiments are executed on import.

Architectures originate from AMCAL.ipynb; use the CLI below for revision runs.
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
        
        # محاسبه خودکار تعداد heads بر اساس input_size
        possible_heads = [8, 4, 2, 1]
        num_heads = 1
        for h in possible_heads:
            if input_size % h == 0:
                num_heads = h
                break
        
        if input_size % num_heads != 0:
            num_heads = 1
            print(f"Warning: Using num_heads=1 for input_size={input_size}")
        
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
def load_data(dataset_path='Dataset', csv_filename='IUST_MSCN_original.csv'):
    try:
        original_df = pd.read_csv(f'{dataset_path}/{csv_filename}')
        
        # تقسیم داده به train (60%)، validation (20%)، test (20%)
        train_ids, val_ids, test_ids = split_source_rows(original_df, RUN_SEED)
        train_df, val_df, test_df = (original_df.iloc[ids] for ids in (train_ids, val_ids, test_ids))
        
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_val = val_df.iloc[:, :-1].values
        y_val = val_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found: {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None, None, None, None
    
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
    
    save_preprocessing(original_df, dataset_path, csv_filename, scaler, label_encoder,
                       class_counts=np.bincount(y_train, minlength=num_classes),
                       train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    if len(train_ids) < 64:
        raise ValueError("Training partition needs at least 64 rows for the current batch size.")
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
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, drop_last=True) 
    full_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)  
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  
    
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
        'protocol_version': 2,
        'context_model_state_dict': context_model.state_dict(),
        'input_length': input_length,
        'num_classes': num_classes
    }, 'Models/adversarial_best_context.pth')
    print("Saved best adversarial Context model")
    
    torch.save({
        'preprocessing_sha256': file_sha256('Models/protocol_preprocessing.joblib'),
        'protocol_version': 2,
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
        'protocol_version': 2,
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
                   seed=RUN_SEED, evaluation="held-out clean; no adaptation",
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

def feature_groups(frame):
    # Feature-identical rows stay in one partition, even with differing labels.
    return pd.util.hash_pandas_object(frame.iloc[:, :-1], index=False).to_numpy()

def split_source_rows(frame, seed):
    groups = feature_groups(frame)
    ids = np.arange(len(frame))
    train, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.4,
                                        random_state=seed).split(ids, groups=groups))
    val_local, test_local = next(GroupShuffleSplit(n_splits=1, test_size=0.5,
                                random_state=seed).split(rest, groups=groups[rest]))
    val, test = rest[val_local], rest[test_local]
    all_classes = set(frame.iloc[:, -1])
    if set(frame.iloc[train, -1]) != all_classes:
        raise ValueError("Training partition lacks classes. Review grouped splitting before training.")
    for left, right in ((train, val), (train, test), (val, test)):
        if set(groups[left]) & set(groups[right]):
            raise AssertionError("Feature-identical samples cross partitions.")
    return train, val, test

def save_preprocessing(frame, dataset_path, filename, scaler, encoder,
                       class_counts, train_ids, val_ids, test_ids):
    source = Path(dataset_path) / filename
    artifact = dict(
        protocol_version=2, seed=RUN_SEED, source_sha256=file_sha256(source),
        source_rows=len(frame), feature_names=list(frame.columns[:-1]),
        label_name=frame.columns[-1], scaler=scaler, label_encoder=encoder,
        class_counts=class_counts, train_ids=train_ids, val_ids=val_ids,
        test_ids=test_ids, feature_groups=feature_groups(frame),
        source_labels=frame.iloc[:, -1].to_numpy())
    Path("Models").mkdir(exist_ok=True)
    joblib.dump(artifact, "Models/protocol_preprocessing.joblib")
    report = dict(protocol_version=2, seed=RUN_SEED,
        source_sha256=artifact["source_sha256"], source_rows=len(frame),
        split_sizes={k:len(v) for k,v in
                     (("train",train_ids),("validation",val_ids),("test",test_ids))},
        split_rule="group feature-identical rows before random splitting",
        selector_target="DDQN gamma=0.95; same-input local transitions; continuing task",
        uncertainty="1 - max softmax of modulated input",
        target_sync="offline: 15 epochs; online: 15 successful selector optimizer steps",
        preprocessing="fit on training only", offline_labels="fully supervised train partition",
        loss="0.8*context_ce + 0.2/(1 + original_ce/(context_ce+1e-8)); no clamp",
        limitation="Exact-feature grouping does not establish session/time independence.")
    Path("Models/protocol_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

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

def load_evaluation_frame(data_dir, attack_file, artifact, row_map=None,
                          assume_row_aligned=False, attack_as_stream=False):
    source_path = Path(data_dir) / "IUST_MSCN_original.csv"
    if file_sha256(source_path) != artifact["source_sha256"]:
        raise ValueError("Original CSV differs from the training source.")
    path = source_path if attack_file is None else Path(attack_file)
    frame = pd.read_csv(path)
    expected = artifact["feature_names"] + [artifact["label_name"]]
    if list(frame.columns) != expected:
        raise ValueError("Feature/label schema or column order differs from training.")
    if attack_as_stream:
        if attack_file is None or row_map is not None or assume_row_aligned:
            raise ValueError("--attack-as-stream requires an attack file and no row mapping options.")
        # Local stream IDs are not original-source identities. Do not claim held-out provenance.
        return frame, np.arange(len(frame)), path, "attack-file local row IDs; source overlap unverified"
    if attack_file is None:
        source_ids = np.arange(len(frame))
        alignment = "original CSV row IDs"
    elif row_map is not None:
        mapping = pd.read_csv(row_map)
        if "source_row" not in mapping or len(mapping) != len(frame):
            raise ValueError("Row map needs one source_row (zero-based original row) per attack row.")
        raw_ids = pd.to_numeric(mapping["source_row"], errors="raise").to_numpy()
        if not np.isfinite(raw_ids).all() or not np.equal(raw_ids, np.floor(raw_ids)).all():
            raise ValueError("source_row IDs must be finite integers.")
        source_ids = raw_ids.astype(np.int64)
        alignment = "explicit source_row mapping"
    elif assume_row_aligned:
        if len(frame) != artifact["source_rows"]:
            raise ValueError("Aligned attack CSV must have the same row count as the original.")
        source_ids = np.arange(len(frame))
        alignment = "user-asserted preserved original row order (not provenance proof)"
    else:
        raise ValueError("Attack evaluation requires --row-map or --assume-row-aligned; do not guess provenance.")
    if (source_ids < 0).any() or (source_ids >= artifact["source_rows"]).any():
        raise ValueError("source_row outside original CSV bounds.")
    if not np.array_equal(frame.iloc[:, -1].to_numpy(), artifact["source_labels"][source_ids]):
        raise ValueError("Attack labels disagree with the asserted source rows.")
    mask = np.isin(source_ids, artifact["test_ids"])
    frame, source_ids = frame.loc[mask].copy(), source_ids[mask]
    if frame.empty:
        raise ValueError("No attack rows correspond to the held-out test partition.")
    return frame, source_ids, path, alignment


def evaluate_stream(args):
    output = Path(args.output).resolve()
    models = output / "Models"
    artifact_path = models / "protocol_preprocessing.joblib"
    artifact = joblib.load(artifact_path)
    if artifact.get('protocol_version') != 2:
        raise ValueError('Preprocessing is not protocol v2; retrain in a fresh directory.')
    frame, source_ids, attack_path, alignment = load_evaluation_frame(
        args.data_dir, args.attack_file, artifact, args.row_map, args.assume_row_aligned,
        args.attack_as_stream)
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
        if saved.get("protocol_version") != 2:
            raise ValueError("Checkpoint is not protocol v2; retrain in a fresh output directory.")
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
        query = (action == 1 and adaptation_allowed and source_id not in queried_ids)
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
            confidence_gap=confidence_gap, queried=query,
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
        evaluated_rows=len(y), unique_source_rows=unique_source_count,
        label_budget=budget, unique_labels_queried=len(queried_ids),
        actual_label_fraction=len(queried_ids)/unique_source_count,
        context_updates=context_updates, selector_updates=selector_updates,
        online_selector_batch_size=agent.batch_size, replay_size=len(agent.replay_buffer),
        first_query_step=first_query_step, last_query_step=last_query_step,
        first_selector_update_step=first_selector_update_step,
        classifier_unchanged=True, protocol="predict-score-query-update",
        selector_freeze_rule="freeze at zero budget or after budget exhaustion",
        alignment=alignment,
        evaluation_scope="diagnostic attack stream" if args.attack_as_stream else "mapped held-out test",
        source_identity_verified=not args.attack_as_stream and not args.assume_row_aligned, seed=args.seed, budget_fraction=args.budget_fraction,
        query_rule='selector action 1 and remaining unique-label budget',
        online_lr=args.lr, attack_sha256=file_sha256(attack_path),
        preprocessing_sha256=digest, device=str(device),
        protocol_version=2, selector_target="DDQN gamma=0.95; same-input local transitions",
        uncertainty="1 - max modulated softmax", target_sync_steps=15,
        train_seed=artifact["seed"], torch_version=str(torch.__version__),
        numpy_version=np.__version__, pandas_version=pd.__version__)
    name = "clean" if args.attack_file is None else Path(args.attack_file).stem
    name = name + ("_unmapped_stream" if args.attack_as_stream else "")
    tag = f"{name}_budget{args.budget_fraction:g}_seed{args.seed}_n{len(y)}_lr{args.lr:g}_sb{args.online_selector_batch_size}_v2"
    results_dir = output / "Results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame(trace).to_csv(results_dir / (tag+"_trace.csv"), index=False)
    (results_dir / (tag+"_metrics.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def legacy_split_candidates(frame, seed):
    """Label-order hypotheses only; these cannot certify attack provenance."""
    ids = np.arange(len(frame))
    y = frame.iloc[:, -1].to_numpy()
    groups = feature_groups(frame)
    candidates = []
    for stratified in (False, True):
        labels = y if stratified else None
        train, test = train_test_split(ids, test_size=0.2, random_state=seed, stratify=labels)
        candidates.append(("80_20_" + ("stratified" if stratified else "unstratified"), train, test))
        train, rest = train_test_split(ids, test_size=0.4, random_state=seed, stratify=labels)
        val, test = train_test_split(rest, test_size=0.5, random_state=seed,
                                    stratify=y[rest] if stratified else None)
        candidates.append(("60_20_20_" + ("stratified" if stratified else "unstratified"), train, test))
    return [dict(name=name, train_rows=len(train), test_rows=len(test),
                 duplicate_feature_groups_crossing_train_test=len(set(groups[train]) & set(groups[test])),
                 source_ids=test) for name, train, test in candidates]

def audit_dataset(args):
    source = Path(args.data_dir) / "IUST_MSCN_original.csv"
    frame = pd.read_csv(source)
    values = frame.iloc[:, :-1].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Original features contain NaN/infinity.")
    train, val, test = split_source_rows(frame, args.seed)
    groups = feature_groups(frame)
    report = dict(source=str(source), sha256=file_sha256(source),
        rows=len(frame), features=values.shape[1], classes=sorted(map(str,frame.iloc[:, -1].unique())),
        duplicate_feature_rows=int(len(frame)-len(np.unique(groups))),
        seed=args.seed, split_sizes={"train":len(train),"validation":len(val),"test":len(test)},
        duplicate_feature_overlap=0,
        split_classes={name:{str(k):int(v) for k,v in frame.iloc[ids,-1].value_counts().items()}
                       for name,ids in (("train",train),("validation",val),("test",test))},
        note="Exact duplicates are grouped; session/time provenance remains to be checked.")
    candidates = legacy_split_candidates(frame, args.seed)
    report["legacy_split_candidates"] = [
        {key:value for key,value in item.items() if key != "source_ids"}
        for item in candidates]
    report["legacy_candidate_warning"] = (
        "Matching label sequences are hypotheses, not confirmed source-row maps. "
        "Do not use a candidate for evaluation without attack-generation provenance.")
    attack_info=[]
    for path in sorted(Path(args.data_dir).glob("*/*.csv")):
        sample=pd.read_csv(path)
        matches = [item["name"] for item in candidates
                   if len(sample) == len(item["source_ids"]) and
                   np.array_equal(sample.iloc[:, -1].to_numpy(),
                                  frame.iloc[item["source_ids"], -1].to_numpy())]
        attack_info.append(dict(file=str(path), rows=len(sample),
            schema_matches=list(sample.columns)==list(frame.columns),
            rowwise_labels_match=(len(sample)==len(frame) and
                np.array_equal(sample.iloc[:,-1].to_numpy(),frame.iloc[:,-1].to_numpy())),
            candidate_label_sequence_matches=matches))
    report["attack_files"]=attack_info
    Path(args.output).mkdir(parents=True, exist_ok=True)
    target=Path(args.output)/"audit.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

def cli():
    global RUN_SEED, device
    parser=argparse.ArgumentParser(description="IUST AMCAL revision protocol; original architectures, corrected evaluation.")
    parser.add_argument("command",choices=("audit","train","evaluate"))
    parser.add_argument("--data-dir",default=str(Path(__file__).resolve().parent/"Dataset"))
    parser.add_argument("--output",default=str(Path(__file__).resolve().parent/"protocol_runs"/"seed42_single_token_v2"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--base-epochs",type=int,default=200)
    parser.add_argument("--context-epochs",type=int,default=200)
    parser.add_argument("--pretrain-epochs",type=int,default=20)
    parser.add_argument("--base-patience",type=int,default=50)
    parser.add_argument("--context-patience",type=int,default=20)
    parser.add_argument("--attack-file")
    parser.add_argument("--attack-as-stream", action="store_true",
                        help="Diagnostic evaluation of the entire existing attack CSV; source overlap unverified.")
    parser.add_argument("--row-map")
    parser.add_argument("--assume-row-aligned",action="store_true")
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--budget-fraction",type=float,default=0.20)
    parser.add_argument("--lr",type=float,default=0.0001)
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
    if args.row_map:
        args.row_map=str(Path(args.row_map).resolve())
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
