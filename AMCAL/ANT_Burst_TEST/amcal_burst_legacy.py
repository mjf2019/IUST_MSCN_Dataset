"""Diagnostic reproduction of Burst AMCAL.ipynb cells 2, 7 and 9.
Preserves legacy numerical logic, including post-update scoring and random online actions.
Not the corrected evaluation protocol.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
BURST_DATA_DIR = Path(__file__).resolve().parent / "Dataset"
LEGACY_PER_LEVELS = None
LEGACY_DIAGNOSTIC_SEED = None
LEGACY_SELECTOR_MODE = 'original'
LEGACY_QUERY_GATE = 'no-threshold'
SELECTOR_REWARD_MODE = 'original'
SELECTOR_TRANSITION_MODE = 'terminal'
SELECTOR_GAMMA = 0.2
CONTEXT_LOSS_MODE = 'legacy'
TRAINING_STATE_VERSION = 2
RUN_TRAINING_STATE_VERSION = TRAINING_STATE_VERSION
ONLINE_EPSILON_START = 0.0
ONLINE_EPSILON_END = 0.0
ONLINE_EPSILON_DECAY = 1.0
TRAINING_SELECTOR_MODE = 'original'
AGREEMENT_PENALTY = 0.2
TEST_LABEL_BUDGET = 214
SELECTION_COST = 0.0


def replay_done(terminal=False):
    """Local continuing transitions; preserve terminal-only legacy by default."""
    return float(terminal) if SELECTOR_TRANSITION_MODE == "local" else 1.0
LEGACY_RESULTS_DIR = 'Results'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
from collections import deque
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@contextmanager
def evaluation_mode(*models):
    """Evaluate without changing BatchNorm buffers; restore all previous modes."""
    previous_modes = {
        module: module.training
        for model in models if model is not None
        for module in model.modules()
    }
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield
    finally:
        for module, training in previous_modes.items():
            module.training = training


# Shared Context objective: explicitly retain the old reproduction as a control.
def context_objective(context_loss, base_loss, online=False):
    epsilon = 1e-8
    if CONTEXT_LOSS_MODE == "aligned" or not online:
        loss_ratio = base_loss / (context_loss + epsilon)
    else:
        loss_ratio = context_loss / (base_loss + epsilon)
    loss = 0.8 * context_loss + 0.2 / (1.0 + loss_ratio)
    if CONTEXT_LOSS_MODE == "legacy" and not online:
        loss = torch.clamp(loss, max=1.0)
    return loss


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
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
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
    def __init__(self, input_size, device, lr=1e-4, gamma=None, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_capacity=50000, batch_size=256):
        self.device = device
        self.policy_net = DQNSelector(input_size).to(device)
        self.target_net = DQNSelector(input_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = SELECTOR_GAMMA if gamma is None else gamma
        self.bootstrap_optimizer_steps = 0
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
    
    def select_action(self, state):
        """Select action for given state. Always returns a tensor."""
        self.steps_done += 1
        
        # Ensure state has batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        if random.random() < self.epsilon:
            # Random action - return tensor
            return torch.randint(0, 2, (state.size(0),), device=self.device)
        else:
            # Greedy action - return tensor
            with torch.no_grad():
                q_values = self.policy_net(state)
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
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            if self.gamma > 0 and (dones < 1).any().item():
                self.bootstrap_optimizer_steps += 1
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Load and preprocess data
def load_training_data(dataset_path=str(BURST_DATA_DIR)):
    try:
        train_df = pd.read_csv(f'{dataset_path}/oversampled_train_dataset.csv')
        val_df = pd.read_csv(f'{dataset_path}/oversampled_validation_dataset.csv')
        test_df = pd.read_csv(f'{dataset_path}/oversampled_test_dataset.csv')
        
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_val = val_df.iloc[:, :-1].values
        y_val = val_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found: {e}")
        return None, None, None, None, None, None, None
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    
    if num_classes != 6:
        print(f"Error: Expected 6 classes, found {num_classes} classes.")
        return None, None, None, None, None, None, None
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts[y_train]
    train_sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler) 
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
    best_epoch = None
    
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
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        scheduler.step()
    
    # Restore the selected snapshot even when training reaches its epoch limit.
    if best_model_state is not None:
        cnn_model.load_state_dict(best_model_state)
        cnn_model.best_validation_epoch = best_epoch
        cnn_model.best_validation_loss = best_val_loss
        print(f"Loaded best CNN snapshot (epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
    return cnn_model

# Evaluate model
def evaluate_model(model, test_loader, class_weights, context_model=None, dqn_agent=None, threshold=None):
    y_true = []
    y_pred = []
    running_loss = 0.0
    total_samples = 0
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    with evaluation_mode(model, context_model), torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if context_model and dqn_agent and threshold is not None:
                cnn_outputs = model(inputs)
                probs = softmax(cnn_outputs)
                one_hot = torch.zeros_like(probs).to(device)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                
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
        reward = 0.0 if SELECTOR_REWARD_MODE == "paper" else (1.5 if cnn_pred == label else -1.0)
    return reward - SELECTION_COST if action == 1 else reward

def train_adversarial_system(cnn_model, dqn_agent, context_model, full_loader, val_loader, class_counts, input_length, num_classes, fine_tune_epochs=500, patience=50):
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    cnn_criterion = nn.CrossEntropyLoss(weight=class_weights)
    context_optimizer = optim.Adam(context_model.parameters(), lr=5e-4, weight_decay=1e-4)
    context_scheduler = lr_scheduler.CosineAnnealingLR(context_optimizer, T_max=100)
    
    best_val_accuracy = -float('inf')
    epochs_no_improve = 0
    softmax = nn.Softmax(dim=1)
    best_context_state = None
    best_dqn_state = None
    best_target_state = None
    best_replay = None
    best_epoch = None
    best_selector_counters = None
    
    # Pre-train ContextAwareNetwork
    print("Pre-training ContextAwareNetwork...")
    for _ in range(20):
        context_model.train()
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            context_optimizer.zero_grad()
            encoded_inputs = context_model(inputs)
            cnn_outputs_weighted = cnn_model(encoded_inputs)
            context_loss = cnn_criterion(cnn_outputs_weighted, labels)
            context_loss.backward()
            context_optimizer.step()
        context_scheduler.step()
    
    print("Starting adversarial training on full dataset...")
    target_update_counter = 0
    TARGET_UPDATE_FREQ = 5
    
    for epoch in range(fine_tune_epochs):
        cnn_model.eval()
        context_model.train()
        dqn_agent.policy_net.train()
        
        running_context_loss = 0.0
        running_dqn_loss = 0.0
        dqn_loss_count = 0
        total_samples = 0
        context_updated_samples = 0
        rejected_agreement_samples = 0
        rewards = []
        actions = []
        context_updated = False
        
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Compute CNN outputs and state for DQN
            with torch.no_grad():
                cnn_outputs = cnn_model(inputs).detach()
                probs = softmax(cnn_outputs)
                _, cnn_pred = torch.max(cnn_outputs, 1)
                max_prob, _ = torch.max(probs, dim=1, keepdim=True)
                uncertainty = 1 - max_prob
            
            features = inputs.squeeze(1).detach()
            state = torch.cat([features, uncertainty], dim=1).detach()
            
            # Select actions for all samples in batch - returns tensor
            actions_batch = dqn_agent.select_action(state)
            actions.extend(actions_batch.cpu().tolist())
            
            # Optional training-only gate, observed BEFORE this batch's update.
            # Eval mode avoids dropout noise and extra BatchNorm updates.
            rejected_agreement = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if TRAINING_SELECTOR_MODE == 'disagreement':
                with evaluation_mode(cnn_model, context_model), torch.no_grad():
                    pre_context_pred = cnn_model(context_model(inputs)).argmax(dim=1)
                rejected_agreement = (actions_batch == 1) & (cnn_pred == pre_context_pred)
                rejected_agreement_samples += int(rejected_agreement.sum().item())
            action_mask = ((actions_batch == 1) & ~rejected_agreement).cpu()
            action_indices = action_mask.nonzero(as_tuple=True)[0]
            
            # Update ContextAwareNetwork only for samples with action=1
            if len(action_indices) > 0:
                context_updated = True
                context_updated_samples += len(action_indices)
                
                selected_inputs = inputs[action_indices]
                selected_labels = labels[action_indices]
                
                # Handle batch size 1
                if selected_inputs.size(0) == 1:
                    selected_inputs = selected_inputs.repeat(2, 1, 1)
                    selected_labels = selected_labels.repeat(2)
                
                context_optimizer.zero_grad()
                encoded_inputs = context_model(selected_inputs)
                cnn_outputs_weighted = cnn_model(encoded_inputs)
                context_loss = cnn_criterion(cnn_outputs_weighted, selected_labels)
                orig_cnn_loss = cnn_criterion(cnn_model(selected_inputs), selected_labels)
                total_context_loss = context_objective(context_loss, orig_cnn_loss)
                total_context_loss.backward()
                context_optimizer.step()
                running_context_loss += total_context_loss.item() * len(action_indices)
            
            # Compute rewards and next states
            with torch.no_grad():
                encoded_inputs = context_model(inputs).detach()
                cnn_outputs_weighted = cnn_model(encoded_inputs).detach()
                _, weighted_pred = torch.max(cnn_outputs_weighted, 1)
            
            # Compute rewards for all samples
            for i in range(batch_size):
                r = compute_reward(actions_batch[i].item(), cnn_pred[i].item(), weighted_pred[i].item(), labels[i].item())
                if rejected_agreement[i].item():
                    # Keep action=1 in replay so Q learns that this request is wasteful.
                    r = -AGREEMENT_PENALTY
                rewards.append(r)
            
            # Compute next states based on action
            with torch.no_grad():
                if len(action_indices) > 0:
                    cnn_outputs_new = cnn_model(context_model(inputs)).detach()
                    probs_new = softmax(cnn_outputs_new)
                    max_prob_new, _ = torch.max(probs_new, dim=1, keepdim=True)
                    uncertainty_new = 1 - max_prob_new
                    features_new = inputs.squeeze(1).detach()
                    next_state_new = torch.cat([features_new, uncertainty_new], dim=1).detach()
                else:
                    next_state_new = state
            
            # Push to replay buffer for ALL actions
            for i in range(batch_size):
                if actions_batch[i] == 1 and not rejected_agreement[i].item():
                    next_state = next_state_new[i:i+1]  # state جدید بعد از context update
                else:
                    next_state = state[i:i+1]  # state فعلی (بدون تغییر)
                
                dqn_agent.replay_buffer.push(
                    state[i:i+1],
                    actions_batch[i].item(),
                    rewards[total_samples - batch_size + i],
                    next_state,
                    replay_done()
                )
                target_update_counter += 1
            
            # Update target network every N steps
            if target_update_counter >= TARGET_UPDATE_FREQ:
                dqn_agent.update_target_network()
                target_update_counter = 0
            
            # Optimize DQN
            dqn_loss = dqn_agent.optimize()
            if dqn_loss is not None:
                running_dqn_loss += dqn_loss
                dqn_loss_count += 1
        
        if context_updated:
            context_scheduler.step()
        
        dqn_agent.update_epsilon()
        
        #if epoch % 5 == 0:
        #    dqn_agent.update_target_network()
        
        print(f"Adversarial Epoch {epoch+1}/{fine_tune_epochs}:")
        print(f"  Context Loss: {running_context_loss / max(context_updated_samples, 1):.4f}")
        print(f"  DQN Loss: {running_dqn_loss / max(dqn_loss_count, 1):.4f}")
        print(f"  Context Updated Samples: {context_updated_samples}")
        if TRAINING_SELECTOR_MODE == 'disagreement':
            print(f"  Agreement requests rejected/penalized: {rejected_agreement_samples}")
        print(f"  Average Reward: {np.mean(rewards) if rewards else 0.0:.4f}")
        print(f"  Action Distribution: {np.bincount(actions, minlength=2) if actions else [0, 0]}")
        
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
            # state_dict tensors otherwise alias the live model and keep changing.
            best_context_state = deepcopy(context_model.state_dict())
            best_dqn_state = deepcopy(dqn_agent.policy_net.state_dict())
            best_target_state = deepcopy(dqn_agent.target_net.state_dict())
            best_replay = deepcopy(list(dqn_agent.replay_buffer.buffer))
            best_epoch = epoch + 1
            best_selector_counters = dict(
                epsilon=dqn_agent.epsilon, steps_done=dqn_agent.steps_done,
                bootstrap_optimizer_steps=dqn_agent.bootstrap_optimizer_steps)
            print(f"Updated best adversarial snapshot (epoch {best_epoch})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    # Restore the SAME selected epoch for the in-memory final evaluation and files.
    if best_context_state is not None:
        context_model.load_state_dict(best_context_state)
        dqn_agent.policy_net.load_state_dict(best_dqn_state)
        dqn_agent.target_net.load_state_dict(best_target_state)
        dqn_agent.replay_buffer.buffer = deque(
            best_replay, maxlen=dqn_agent.replay_buffer.buffer.maxlen)
        for name, value in best_selector_counters.items():
            setattr(dqn_agent, name, value)
        print(f"Loaded best Context/selector snapshot (epoch {best_epoch}, Val Accuracy: {best_val_accuracy:.4f})")

    # Save best models
    if best_context_state is not None:
        torch.save({
            'context_model_state_dict': best_context_state,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'input_length': input_length,
            'num_classes': num_classes
        }, 'Models/adversarial_best_context.pth')
        print("Saved best adversarial Context model")
    
    if best_dqn_state is not None:
        torch.save({
            'dqn_policy_net_state_dict': best_dqn_state,
            'dqn_target_net_state_dict': best_target_state,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'selector_training_counters': best_selector_counters,
            'input_length': input_length,
            'num_classes': num_classes,
            'replay_buffer': best_replay
        }, 'Models/adversarial_best_dqn.pth')
        print("Saved best adversarial DQN model with replay buffer")
    
    return cnn_model, dqn_agent, context_model


# Main execution
def train_main():
    base_epoch = 200
    base_patience = 30
    adver_epoch = 300
    adver_patience = 30
    
    # Load data
    train_loader, val_loader, test_loader, full_loader, num_classes, input_length, class_counts = load_training_data(dataset_path=str(BURST_DATA_DIR))
    if train_loader is None or full_loader is None:
        print("Failed to load data. Exiting.")
        return
    
    # Train and evaluate initial CNN
    cnn_model = CNN1D(input_length, num_classes).to(device)
    cnn_model = train_initial_cnn(cnn_model, train_loader, val_loader, class_counts, base_epoch, base_patience)
    
    if not os.path.exists('Models'):
        os.makedirs('Models')
    
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'best_epoch': cnn_model.best_validation_epoch,
        'best_val_loss': cnn_model.best_validation_loss,
        'input_length': input_length,
        'num_classes': num_classes
    }, 'Models/initial_cnn_model.pth')
    print("Saved initial CNN model")
    
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    initial_results = evaluate_model(cnn_model, test_loader, class_weights)
    print("\nInitial CNN Test Results:")
    print(f"  Test Loss: {initial_results['Test Loss']:.4f}")
    print(f"  Accuracy: {initial_results['Accuracy']:.4f}")
    print(f"  Precision: {initial_results['Precision']:.4f}")
    print(f"  Recall: {initial_results['Recall']:.4f}")
    print(f"  F1 Score: {initial_results['F1 Score']:.4f}")
    
    # Initialize DQN and Context models
    dqn_agent = DQNSelectorAgent(input_length + 1, device)
    context_model = ContextAwareNetwork(input_length).to(device)
    
    # Train adversarial system using full_loader (full dataset)
    cnn_model, dqn_agent, context_model = train_adversarial_system(
        cnn_model, dqn_agent, context_model, full_loader, val_loader, class_counts, input_length, num_classes,
        adver_epoch, adver_patience
    )
    
    # Evaluate adversarial system
    adversarial_results = evaluate_model(cnn_model, test_loader, class_weights, context_model, dqn_agent, threshold=0.5)
    print("\nAdversarial System Test Results:")
    print(f"  Test Loss: {adversarial_results['Test Loss']:.4f}")
    print(f"  Accuracy: {adversarial_results['Accuracy']:.4f}")
    print(f"  Precision: {adversarial_results['Precision']:.4f}")
    print(f"  Recall: {adversarial_results['Recall']:.4f}")
    print(f"  F1 Score: {adversarial_results['F1 Score']:.4f}")



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import glob
import hashlib
import uuid

# Check for GPU (CUDA) availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Class Definitions (Assuming these are defined in your previous code blocks) ---
# Make sure CNN1D, ContextAwareNetwork, DQNSelector, ReplayBuffer, DQNSelectorAgent are defined here
# If they are in another file, import them. For this solution, I will assume they are available.
# If not, please paste the class definitions from the previous turn.

# For the sake of this complete runnable script, I will re-define the necessary classes briefly if they are missing,
# but usually, you should have them defined. I will focus on the `test_system` and `main` logic.

def compute_reward(action, cnn_pred, weighted_pred, label):
    # Ensure inputs are integers for comparison
    action = int(action)
    cnn_pred = int(cnn_pred)
    weighted_pred = int(weighted_pred)
    label = int(label)
    
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
        reward = 0.0 if SELECTOR_REWARD_MODE == "paper" else (1.5 if cnn_pred == label else -1.0)
    return reward - SELECTION_COST if action == 1 else reward

def load_metadata(base_path='AdvBurst_FTSC-IAT', train_perturb_level=0):
    X_all, y_all = [], []
    dataset_indices = []
    if not os.path.exists(base_path):
        print(f"Error: Data directory '{base_path}' not found.")
        return None, None, None, None, None, None, None, None, None, None, None
    try:
        df = pd.read_csv(f'{base_path}/AdvBurstTrainedBurst_size_{train_perturb_level}_Step_2000.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_all.append(X)
        y_all.append(y)
        dataset_indices.append(list(range(len(y))))
        print(f"Loaded data for perturbation level {train_perturb_level} with {X.shape[0]} samples.")
    except FileNotFoundError:
        print(f"Error: File for perturbation level {train_perturb_level} not found.")
        return None, None, None, None, None, None, None, None, None, None, None
    if not X_all or not y_all:
        print("Error: No valid data loaded for training.")
        return None, None, None, None, None, None, None, None, None, None, None
    # Create full dataset before splitting
    X = np.vstack(X_all)
    y = np.hstack(y_all)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    if num_classes != 6:
        print(f"Error: Expected 6 classes, found {num_classes} classes.")
        return None, None, None, None, None, None, None, None, None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    class_counts = np.bincount(y)
    print(f"Input length: {X.shape[1]}")

    return num_classes, label_encoder, scaler, X.shape[1], class_counts

def test_system(cnn_model, dqn_agent, context_model, base_path, perturb_level, label_encoder, scaler, device, threshold, class_count, budget=300, fine_tune_lr=0.0001, max_test_samples=100):
    class_weights = torch.FloatTensor(class_count).to(device)
    cnn_criterion = nn.CrossEntropyLoss(weight=class_weights)
    cumulative_accuracy_df = pd.DataFrame(columns=['Sample Index', f'Perturbation {perturb_level}'])
    results = []
    softmax = nn.Softmax(dim=1)
    
    # Ensure models are in eval mode
    cnn_model.eval()
    context_model.eval()
    
    # Load test data
    test_file = f'{base_path}/AdvBurstTrainedBurst_size_{perturb_level}_Step_2000.csv'
    try:
        df_test = pd.read_csv(test_file)
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values
        max_samples = min(len(df_test), max_test_samples)
        X_test = X_test[:max_samples]
        y_test = y_test[:max_samples]
        cumulative_accuracy_df = pd.DataFrame({'Sample Index': range(1, max_samples + 1)})
    except FileNotFoundError:
        print(f"Error: File {test_file} not found.")
        return [], cumulative_accuracy_df
    
    try:
        y_test = label_encoder.transform(y_test)
    except ValueError:
        print(f"Error: Unknown labels in {test_file}. Skipping.")
        return [], cumulative_accuracy_df
    
    # Observer only: compare the frozen CNN under a train-fitted scaler.
    # This never replaces the scaler or input used by legacy learning.
    train_frame = pd.read_csv(BURST_DATA_DIR / "oversampled_train_dataset.csv")
    train_scaler = StandardScaler().fit(train_frame.iloc[:, :-1].to_numpy())
    train_scaled_inputs = torch.FloatTensor(train_scaler.transform(X_test)).unsqueeze(1).to(device)
    X_test = scaler.transform(X_test)
    X_test = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Optimizer for context_model
    context_optimizer = optim.Adam(context_model.parameters(), lr=fine_tune_lr, weight_decay=1e-4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    sample_accuracies = []
    updates_used = 0
    dqn_updates_used = 0
    dqn_loss_count = 0
    running_dqn_loss = 0.0
    sample_idx = 0
    rewards = []
    actions = []
    target_update_counter = 0
    TARGET_UPDATE_FREQ = 10
    if LEGACY_SELECTOR_MODE == "learned":
        dqn_agent.epsilon = ONLINE_EPSILON_START
        dqn_agent.epsilon_end = ONLINE_EPSILON_END
        dqn_agent.epsilon_decay = ONLINE_EPSILON_DECAY
        print(f"Online RL epsilon: {dqn_agent.epsilon:g} -> "
              f"{dqn_agent.epsilon_end:g}; decay={dqn_agent.epsilon_decay:g} per optimizer step")
    diagnostic_trace, before_predictions, base_predictions, train_base_predictions = [], [], [], []
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        sample_idx += 1
        
        updates_before = updates_used
        observed_reward, observed_loss = None, None
        # Compute CNN outputs and state
        with torch.no_grad():
            cnn_outputs = cnn_model(inputs)
            probs = softmax(cnn_outputs)
            _, cnn_pred = torch.max(probs, 1)
            max_prob, _ = torch.max(probs, dim=1, keepdim=True)
            uncertainty = 1 - max_prob
            features = inputs.squeeze(1)
            state = torch.cat([features, uncertainty], dim=1)
        
        # Select action
        # select_action returns a tensor or int. We need int for appending to list.
        action_epsilon = float(dqn_agent.epsilon)
        action_source = "original"
        if LEGACY_SELECTOR_MODE == "learned":
            # Original epsilon-greedy choice, now explicit and observable.
            # A zero starting epsilon preserves the previous greedy path/RNG.
            previous_mode = dqn_agent.policy_net.training
            dqn_agent.policy_net.eval()
            try:
                with torch.no_grad():
                    if dqn_agent.epsilon > 0 and random.random() < dqn_agent.epsilon:
                        action_tensor = torch.randint(0, 2, (state.size(0),), device=device)
                        action_source = "exploration"
                    else:
                        action_tensor = dqn_agent.policy_net(state).argmax(dim=1)
                        action_source = "policy"
            finally:
                dqn_agent.policy_net.train(previous_mode)
            dqn_agent.steps_done += 1
        else:
            action_tensor = dqn_agent.select_action(state)
        action = action_tensor.item() if isinstance(action_tensor, torch.Tensor) else int(action_tensor)
        actions.append(action)
        
        # Compute weighted outputs BEFORE any context update
        with torch.no_grad():
            encoded_inputs_before = context_model(inputs)
            cnn_outputs_weighted_before = cnn_model(encoded_inputs_before)
            weighted_probs_before = softmax(cnn_outputs_weighted_before)
            _, weighted_pred_before = torch.max(cnn_outputs_weighted_before, 1)
        
        # Observe the learned policy without dropout; do not alter legacy actions/RNG.
        previous_policy_mode = dqn_agent.policy_net.training
        dqn_agent.policy_net.eval()
        with torch.no_grad():
            diagnostic_q = dqn_agent.policy_net(state)
            train_scaled_prediction = cnn_model(train_scaled_inputs[sample_idx-1:sample_idx]).argmax(1).item()
        dqn_agent.policy_net.train(previous_policy_mode)
        before_predictions.append(int(weighted_pred_before.item()))
        base_predictions.append(int(cnn_pred.item()))
        train_base_predictions.append(int(train_scaled_prediction))

        # Update ContextAwareNetwork only when action=1
        if action == 1:
            _, weighted_pred = torch.max(cnn_outputs_weighted_before, 1)
            max_weighted_prob, _ = torch.max(weighted_probs_before, dim=1, keepdim=True)
            max_prob_diff = torch.max(torch.abs(max_weighted_prob - max_prob)).item()
            
            disagreement_passes = (LEGACY_QUERY_GATE == "none" or weighted_pred != cnn_pred)
            if disagreement_passes and (threshold is None or max_prob_diff >= threshold) and updates_used < budget:
                
                updates_used += 1
                context_optimizer.zero_grad()
                encoded_inputs = context_model(inputs)
                outputs = cnn_model(encoded_inputs)
                context_loss = cnn_criterion(outputs, labels)
                orig_cnn_loss = cnn_criterion(cnn_outputs.detach(), labels)
                total_context_loss = context_objective(context_loss, orig_cnn_loss, online=True)
                total_context_loss.backward()
                context_optimizer.step()
                
                # Compute reward for action=1
                reward = compute_reward(action, cnn_pred.item(), weighted_pred.item(), labels.item())
                rewards.append(reward)
                observed_reward = float(reward)
                observed_loss = float(total_context_loss.item())
                
                # Compute next state with updated context
                with torch.no_grad():
                    encoded_inputs_new = context_model(inputs)
                    cnn_outputs_new = cnn_model(encoded_inputs_new)
                    probs_new = softmax(cnn_outputs_new)
                    max_prob_new, _ = torch.max(probs_new, dim=1, keepdim=True)
                    uncertainty_new = 1 - max_prob_new
                    next_state = torch.cat([features, uncertainty_new], dim=1)
                
                # Push action=1 to replay buffer
                dqn_agent.replay_buffer.push(
                    state, action, reward, next_state,
                    replay_done(updates_used >= budget or sample_idx >= max_samples))
                target_update_counter += 1
                
                # Hindsight Experience Replay: اگر reward منفی بود، action=0 رو هم اضافه کن
                if reward < 0:
                    reward_action0 = compute_reward(0, cnn_pred.item(), weighted_pred.item(), labels.item())
                    # Next state for action=0 is the current state (no context update)
                    dqn_agent.replay_buffer.push(
                        state, 0, reward_action0, state,
                        replay_done(sample_idx >= max_samples))
                    target_update_counter += 1
                
                # Optimize DQN
                dqn_loss = dqn_agent.optimize()
                if dqn_loss is not None:
                    running_dqn_loss += dqn_loss
                    dqn_loss_count += 1
                    dqn_updates_used += 1
                    if LEGACY_SELECTOR_MODE == "learned":
                        dqn_agent.update_epsilon()
                
                # Update target network every N steps
                if target_update_counter >= TARGET_UPDATE_FREQ:
                    dqn_agent.update_target_network()
                    target_update_counter = 0
                
                # Final output with updated context
                with torch.no_grad():
                    encoded_inputs_final = context_model(inputs)
                    outputs = cnn_model(encoded_inputs_final)
            else:
                # No context update, use current context
                with torch.no_grad():
                    encoded_inputs_final = context_model(inputs)
                    outputs = cnn_model(encoded_inputs_final)
        else:
            # Action = 0: no context update, no reward, no replay buffer
            with torch.no_grad():
                encoded_inputs_final = context_model(inputs)
                outputs = cnn_model(encoded_inputs_final)
        
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_accuracy = correct / total
            sample_accuracies.append(current_accuracy)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            diagnostic_trace.append(dict(step=sample_idx, true_label=int(labels.item()),
                base_prediction=int(cnn_pred.item()),
                base_prediction_train_scaler=int(train_scaled_prediction),
                prediction_before_update=int(weighted_pred_before.item()),
                prediction_after_update=int(predicted.item()),
                action=int(action), greedy_action=int(diagnostic_q.argmax(1).item()),
                q_skip=float(diagnostic_q[0,0].item()), q_apply=float(diagnostic_q[0,1].item()),
                epsilon=float(dqn_agent.epsilon),
                action_epsilon=action_epsilon, action_source=action_source,
                labels_before=updates_before,
                confidence_gap=float(abs(weighted_probs_before.max().item()-max_prob.item())),
                updated=updates_used>updates_before, labels_used=updates_used,
                context_loss=observed_loss, reward=observed_reward))
            cumulative_accuracy_df.loc[cumulative_accuracy_df['Sample Index'] == sample_idx, f'Perturbation {perturb_level}'] = current_accuracy
    
    test_accuracy = correct / total
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    from sklearn.metrics import accuracy_score, classification_report
    os.makedirs(LEGACY_RESULTS_DIR, exist_ok=True)
    import json
    diagnostics = dict(per=perturb_level, seed=LEGACY_DIAGNOSTIC_SEED,
        base_accuracy_legacy_scaler=float(accuracy_score(y_true,base_predictions)),
        base_accuracy_train_scaler=float(accuracy_score(y_true,train_base_predictions)),
        amcal_accuracy_before_update=float(accuracy_score(y_true,before_predictions)),
        amcal_accuracy_after_update=float(test_accuracy),
        before_weighted_f1=float(f1_score(y_true,before_predictions,average="weighted",zero_division=0)),
        after_weighted_f1=float(f1), labels_used=updates_used,
        epsilon=float(dqn_agent.epsilon), selector_mode=LEGACY_SELECTOR_MODE,
        epsilon_start=ONLINE_EPSILON_START if LEGACY_SELECTOR_MODE=="learned" else 1.0,
        epsilon_end=ONLINE_EPSILON_END if LEGACY_SELECTOR_MODE=="learned" else 1.0,
        epsilon_decay=ONLINE_EPSILON_DECAY if LEGACY_SELECTOR_MODE=="learned" else 1.0,
        epsilon_decay_rule="after successful online selector optimization",
        label_budget=budget, actual_label_fraction=updates_used / total,
        first_query_step=next((row["step"] for row in diagnostic_trace if row["updated"]), None),
        last_query_step=next((row["step"] for row in reversed(diagnostic_trace) if row["updated"]), None),
        policy_decisions_with_budget=sum(row["action_source"]=="policy" and row["labels_before"]<budget
                                        for row in diagnostic_trace),
        exploration_decisions_with_budget=sum(row["action_source"]=="exploration" and row["labels_before"]<budget
                                             for row in diagnostic_trace),
        reward_mode=SELECTOR_REWARD_MODE,
        transition_mode=SELECTOR_TRANSITION_MODE, gamma=float(dqn_agent.gamma),
        query_gate=LEGACY_QUERY_GATE, confidence_gap_threshold=threshold,
        context_loss_mode=CONTEXT_LOSS_MODE,
        training_selector_mode=TRAINING_SELECTOR_MODE,
        agreement_penalty=AGREEMENT_PENALTY if TRAINING_SELECTOR_MODE=='disagreement' else None,
        selection_cost=SELECTION_COST,
        training_state_version=RUN_TRAINING_STATE_VERSION,
        bootstrap_optimizer_steps=dqn_agent.bootstrap_optimizer_steps,
        replay_terminal_rows=sum(float(e[4]) == 1.0 for e in dqn_agent.replay_buffer.buffer),
        replay_nonterminal_rows=sum(float(e[4]) == 0.0 for e in dqn_agent.replay_buffer.buffer),
        learned_policy_used_for_actions=LEGACY_SELECTOR_MODE=="learned",
        classes_before=classification_report(y_true,before_predictions,output_dict=True,zero_division=0),
        classes_after=classification_report(y_true,y_pred,output_dict=True,zero_division=0))
    pd.DataFrame(diagnostic_trace).to_csv(f"{LEGACY_RESULTS_DIR}/legacy_PER_{perturb_level}_trace.csv",index=False)
    Path(f"{LEGACY_RESULTS_DIR}/legacy_PER_{perturb_level}_diagnostics.json").write_text(
        json.dumps(diagnostics,indent=2),encoding="utf-8")
    print("Diagnostic frozen CNN / pre-update AMCAL / post-update AMCAL:",
          diagnostics["base_accuracy_legacy_scaler"],
          diagnostics["amcal_accuracy_before_update"],
          diagnostics["amcal_accuracy_after_update"])

    results.append({
        'Perturbation Level': perturb_level,
        'Test Accuracy': test_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Context Updates Used': updates_used,
        'DQN Updates Used': dqn_updates_used,
        'DQN Loss': running_dqn_loss / max(dqn_loss_count, 1)
    })
    
    print(f"\nResults for Perturbation Level {perturb_level}:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    print(f"  Context Updates Used: {updates_used}/{budget}")
    print(f"  DQN Updates Used: {dqn_updates_used}/{budget}")
    print(f"  Average DQN Loss: {running_dqn_loss / max(dqn_loss_count, 1):.4f}")
    print(f"  Average Reward: {np.mean(rewards) if rewards else 0.0:.4f}")
    if LEGACY_SELECTOR_MODE == "learned":
        print(f"  Final Epsilon: {dqn_agent.epsilon:.4f}")
        print("  Policy / Exploration Decisions While Budget Remained:",
              diagnostics["policy_decisions_with_budget"], "/",
              diagnostics["exploration_decisions_with_budget"])
    # Convert actions list to numpy array for bincount to avoid CUDA errors
    actions_np = np.array(actions)
    print(f"  Action Distribution: {np.bincount(actions_np, minlength=2) if len(actions_np) > 0 else [0, 0]}")
    
    return results, cumulative_accuracy_df

def plot_accuracies(combined_accuracy_df, test_perturb_levels):
    plt.figure(figsize=(12, 8))
    for perturb in test_perturb_levels:
        if f'Perturbation {perturb}' in combined_accuracy_df.columns:
            plt.plot(combined_accuracy_df['Sample Index'], combined_accuracy_df[f'Perturbation {perturb}'], 
                     label=f'Perturbation {perturb}', marker='o', markersize=4, linewidth=1.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Cumulative Accuracy per Sample Across Perturbation Levels')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    os.makedirs(LEGACY_RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{LEGACY_RESULTS_DIR}/cumulative_accuracy_per_sample_all_levels.png', dpi=300)
    plt.show()
    print(f"Plot saved to '{LEGACY_RESULTS_DIR}/cumulative_accuracy_per_sample_all_levels.png'")



def evaluate_main():
    fine_tune_lr = 0.25
    test_perturb_levels = LEGACY_PER_LEVELS or [0, 1, 3, 5, 7, 10, 12, 15, 17, 20]
    cnn_perturb_level = 0
    base_path = str(BURST_DATA_DIR / 'AdvBurst_FTSC-IAT')
    context_budget_test = TEST_LABEL_BUDGET
    cnn_model_path = 'Models/initial_cnn_model.pth'
    context_model_path = 'Models/adversarial_best_context.pth'
    dqn_model_path = 'Models/adversarial_best_dqn.pth'
    max_test_samples = 1073
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Expected data directory: {os.path.abspath(base_path)}")
    print("Available files in data directory:")
    for file in glob.glob(f"{base_path}/*.csv"):
        print(file)
        
    all_results = []
    all_accuracy_dfs = []
    max_sample_index = 0
    accuracy_by_perturb = {}
    
    for perturb_level in test_perturb_levels:
        print(f"\n{'='*50}")
        print(f"Starting processing for perturbation level {perturb_level}")
        print(f"{'='*50}")
        
        try:
            num_classes, label_encoder, scaler, input_length, class_counts = load_metadata(
                base_path, cnn_perturb_level
            )
        except Exception as e:
            print(f"Error loading data metadata: {e}")
            continue
            
        if num_classes is None:
            print("Failed to load data metadata. Skipping.")
            continue

        try:
            # Load CNN model
            cnn_checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=False)
            cnn_input_length = cnn_checkpoint['input_length']
            cnn_num_classes = cnn_checkpoint['num_classes']
            
            if cnn_input_length != input_length or cnn_num_classes != num_classes:
                print(f"Error: Mismatch in CNN model parameters (input_length: {cnn_input_length} vs {input_length}, num_classes: {cnn_num_classes} vs {num_classes})")
                continue
                
            cnn_model = CNN1D(input_length=input_length, num_classes=num_classes).to(device)
            cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
            cnn_model.eval() # Set to eval mode
            print(f"Successfully loaded CNN model from {cnn_model_path}")
            
            # Load ContextAwareNetwork model
            context_checkpoint = torch.load(context_model_path, map_location=device, weights_only=False)
            context_input_length = context_checkpoint['input_length']
            context_num_classes = context_checkpoint['num_classes']
            
            if context_input_length != input_length or context_num_classes != num_classes:
                print(f"Error: Mismatch in Context model parameters (input_length: {context_input_length} vs {input_length}, num_classes: {context_num_classes} vs {num_classes})")
                continue
                
            context_model = ContextAwareNetwork(input_size=input_length, num_layers=4).to(device)
            context_model.load_state_dict(context_checkpoint['context_model_state_dict'])
            context_model.eval() # Set to eval mode
            print(f"Successfully loaded Context model from {context_model_path}")
            
            # Load DQNSelector model
            dqn_checkpoint = torch.load(dqn_model_path, map_location=device, weights_only=False)
            dqn_input_length = dqn_checkpoint['input_length']
            dqn_num_classes = dqn_checkpoint['num_classes']
            
            if dqn_input_length != input_length or dqn_num_classes != num_classes:
                print(f"Error: Mismatch in DQN model parameters (input_length: {dqn_input_length} vs {input_length}, num_classes: {dqn_num_classes} vs {num_classes})")
                continue
                
            dqn_agent = DQNSelectorAgent(
                input_size=input_length + 1,
                device=device,
                epsilon_decay=0.5,
                buffer_capacity=5000,
                batch_size=128
            )
            dqn_agent.policy_net.load_state_dict(dqn_checkpoint['dqn_policy_net_state_dict'])
            dqn_agent.target_net.load_state_dict(dqn_checkpoint['dqn_policy_net_state_dict'])
            dqn_agent.target_net.eval()
            
            # Load replay buffer if exists
            if 'replay_buffer' in dqn_checkpoint:
                dqn_agent.replay_buffer = ReplayBuffer(500)
                for experience in dqn_checkpoint['replay_buffer']:
                    state, action, reward, next_state, done = experience
                    
                    if not isinstance(state, torch.Tensor):
                        state = torch.FloatTensor(state).to(device)
                    if not isinstance(next_state, torch.Tensor):
                        next_state = torch.FloatTensor(next_state).to(device)
                        
                    if state.dim() == 0:
                        state = state.unsqueeze(0)
                    if next_state.dim() == 0:
                        next_state = next_state.unsqueeze(0)
                        
                    dqn_agent.replay_buffer.push(state, action, reward, next_state, done)
                print(f"Loaded replay buffer with {len(dqn_agent.replay_buffer)} experiences")
            else:
                print("Warning: No replay buffer found in checkpoint. Starting with empty buffer.")
                dqn_agent.replay_buffer = ReplayBuffer(200)
                
            print(f"Successfully loaded DQN model from {dqn_model_path}")
            
        except FileNotFoundError as e:
            print(f"Error: Model file not found: {e}")
            continue
        except KeyError as e:
            print(f"Error: Missing key {e} in checkpoint.")
            continue
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        threshold = None if LEGACY_QUERY_GATE in ("no-threshold", "none", "disagreement") else 0.005
        if LEGACY_QUERY_GATE == "none":
            print("Test - All prediction/confidence filters disabled; selector action and budget only.")
        elif threshold is None:
            print("Test - Disagreement condition only; confidence threshold disabled, budget retained.")
        else:
            print(f"Test - Using DQN threshold: {threshold:.4f}")
        
        results, cumulative_accuracy_df = test_system(
            cnn_model, dqn_agent, context_model, base_path, perturb_level,
            label_encoder, scaler, device, threshold, class_counts, budget=context_budget_test, 
            fine_tune_lr=fine_tune_lr, max_test_samples=max_test_samples
        )
        
        all_results.extend(results)
        if not cumulative_accuracy_df.empty:
            all_accuracy_dfs.append(cumulative_accuracy_df)
            max_sample_index = max(max_sample_index, cumulative_accuracy_df['Sample Index'].max())
            accuracy_by_perturb[perturb_level] = cumulative_accuracy_df[f'Perturbation {perturb_level}'].values.tolist()
            
    if accuracy_by_perturb:
        max_samples = max(len(acc) for acc in accuracy_by_perturb.values())
        # Build all columns together instead of repeatedly inserting into a DataFrame.
        combined_columns = {'Size': list(accuracy_by_perturb.keys())}
        for i in range(max_samples):
            combined_columns[f'{i+1}'] = [
                acc_list[i] if i < len(acc_list) else np.nan
                for acc_list in accuracy_by_perturb.values()
            ]
        combined_accuracy_df = pd.DataFrame(combined_columns)
        
        os.makedirs(LEGACY_RESULTS_DIR, exist_ok=True)
        combined_accuracy_df.to_csv(f'{LEGACY_RESULTS_DIR}/Micfoal_Conf_cumulative_accuracy_all_perturbations_transposed.csv', index=False)
        print(f"Combined cumulative accuracy saved to '{LEGACY_RESULTS_DIR}/Micfoal_Conf_cumulative_accuracy_all_perturbations_transposed.csv'")
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        os.makedirs(LEGACY_RESULTS_DIR, exist_ok=True)
        results_df.to_csv(f'{LEGACY_RESULTS_DIR}/AMCAL_20_cumulative_accuracy.csv', index=False)
        print("\nCombined Results Table:")
        print(results_df.to_string(index=False))
        print(f"\nCombined results saved to '{LEGACY_RESULTS_DIR}/AMCAL_20_cumulative_accuracy.csv'.")
        
    if accuracy_by_perturb:
        plt.figure(figsize=(12, 8))
        for perturb_level in test_perturb_levels:
            if perturb_level in accuracy_by_perturb:
                plt.plot(range(1, len(accuracy_by_perturb[perturb_level]) + 1), 
                         accuracy_by_perturb[perturb_level], 
                         label=f'Perturbation {perturb_level}', marker='o', markersize=4, linewidth=1.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Cumulative Accuracy')
        plt.title('Cumulative Accuracy per Sample Across Perturbation Levels')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        os.makedirs(LEGACY_RESULTS_DIR, exist_ok=True)
        plt.savefig(f'{LEGACY_RESULTS_DIR}/AMCAL_20_cumulative_accuracy.png', dpi=300)
        plt.close()
        print(f"Plot saved to '{LEGACY_RESULTS_DIR}/AMCAL_20_cumulative_accuracy.png'")



def cli():
    global TRAINING_SELECTOR_MODE, AGREEMENT_PENALTY, TEST_LABEL_BUDGET, SELECTION_COST
    global BURST_DATA_DIR, LEGACY_PER_LEVELS, LEGACY_DIAGNOSTIC_SEED, LEGACY_SELECTOR_MODE, LEGACY_RESULTS_DIR, SELECTOR_REWARD_MODE, SELECTOR_TRANSITION_MODE, SELECTOR_GAMMA, LEGACY_QUERY_GATE, CONTEXT_LOSS_MODE, RUN_TRAINING_STATE_VERSION, ONLINE_EPSILON_START, ONLINE_EPSILON_END, ONLINE_EPSILON_DECAY
    import argparse
    import json
    parser=argparse.ArgumentParser(description="Original Burst notebook reproduction, 20% configuration.")
    parser.add_argument("command",choices=("train","evaluate"))
    parser.add_argument("--data-dir",default=str(BURST_DATA_DIR))
    parser.add_argument("--output",help="Output/model folder; original and paper reward modes have separate defaults.")
    parser.add_argument("--per",type=int,choices=(0,1,3,5,7,10,12,15,17,20))
    parser.add_argument("--seed",type=int,help="Optional fixed random seed; legacy default stays unseeded.")
    parser.add_argument("--selector-mode", choices=("original","random","learned"), default="original",
                        help="original: notebook selector with online DQN; random: no-RL random updater; "
                             "learned: Q actions with optional epsilon-greedy exploration and online DQN.")
    parser.add_argument("--reward-mode", choices=("original","paper"), default="original",
                        help="paper sets only action-0 reward to zero; other legacy logic is unchanged.")
    parser.add_argument("--transition-mode", choices=("terminal","local"), default="terminal",
                        help="local enables same-input DDQN bootstrap; terminal preserves legacy immediate targets.")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Discount factor; legacy default 0.2, paper value 0.95.")
    parser.add_argument("--query-gate", choices=("none","disagreement","no-threshold","legacy"), default="no-threshold",
                        help="none removes both filters; disagreement retains ONLY unequal predictions in all modes; "
                             "legacy retains both. no-threshold preserves its historical mode-dependent behavior.")
    parser.add_argument("--epsilon-start", type=float, default=0.0,
                        help="Learned online selector only; 0 preserves greedy inference.")
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument("--epsilon-decay", type=float, default=1.0,
                        help="Learned online selector only; multiply epsilon after each successful optimizer step.")
    parser.add_argument("--context-loss-mode", choices=("legacy","aligned"), default="legacy",
                        help="aligned uses base/Context CE ratio in train and online, without the training loss cap.")
    parser.add_argument("--training-selector-mode", choices=("original","disagreement"), default="original",
                        help="disagreement: penalize action-1 requests on equal pre-update predictions; do not update Context.")
    parser.add_argument("--agreement-penalty", type=float, default=0.2,
                        help="Positive penalty magnitude for rejected training requests; default 0.2.")
    parser.add_argument("--budget", type=int, default=214,
                        help="Maximum labels/Context updates per test level, from 0 to 1073. Gates may use fewer.")
    parser.add_argument("--selection-cost", type=float, default=0.0,
                        help="Subtract this cost from every action-1 reward in training and online replay; skip stays zero in paper mode.")
    args=parser.parse_args()
    if not np.isfinite(args.selection_cost) or args.selection_cost < 0:
        parser.error("selection-cost must be finite and nonnegative.")
    if args.selection_cost > 0 and (args.reward_mode != "paper" or args.training_selector_mode != "original"):
        parser.error("selection-cost experiments require --reward-mode paper and --training-selector-mode original.")
    if args.selection_cost > 0 and args.selector_mode == "random":
        parser.error("random has no reward computation; omit --selection-cost for the no-RL control.")
    SELECTION_COST = args.selection_cost
    if not np.isfinite(args.agreement_penalty) or args.agreement_penalty <= 0:
        parser.error("agreement-penalty must be finite and positive.")
    if not 0 <= args.budget <= 1073:
        parser.error("budget must be between 0 and 1073.")
    TRAINING_SELECTOR_MODE = args.training_selector_mode
    AGREEMENT_PENALTY = args.agreement_penalty
    TEST_LABEL_BUDGET = args.budget
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        parser.error("epsilon requires 0 <= end <= start <= 1.")
    if not 0.0 < args.epsilon_decay <= 1.0:
        parser.error("epsilon-decay must be in (0, 1].")
    scheduled_epsilon = (args.epsilon_start != 0.0 or args.epsilon_end != 0.0
                         or args.epsilon_decay != 1.0)
    if scheduled_epsilon and (args.command != "evaluate" or args.selector_mode != "learned"):
        parser.error("epsilon settings apply only to evaluate --selector-mode learned.")
    ONLINE_EPSILON_START = args.epsilon_start
    ONLINE_EPSILON_END = args.epsilon_end
    ONLINE_EPSILON_DECAY = args.epsilon_decay
    if args.selector_mode == "random":
        if args.command != "evaluate":
            parser.error("random is an online-only ablation; use existing CNN/Context checkpoints.")
        # Dispatch before ANY legacy DQN checkpoint validation or construction.
        # Reuse this already-loaded module; avoid importing/printing it twice.
        import sys
        if __name__ == "__main__":
            sys.modules.setdefault("amcal_burst_legacy", sys.modules[__name__])
        from amcal_burst_random import evaluate as evaluate_random
        evaluate_random(argparse.Namespace(
            command="evaluate", data_dir=Path(args.data_dir),
            output=Path(args.output) if args.output else
                   Path(__file__).resolve().parent / "protocol_runs" / "notebook_original",
            results_output=None, per=args.per,
            seed=args.seed if args.seed is not None else 42,
            budget=args.budget, max_samples=1073, lr=0.25,
            context_loss_mode=args.context_loss_mode,
            query_gate=args.query_gate if args.query_gate in ("legacy","disagreement") else "none"))
        return
    if not 0.0 <= args.gamma < 1.0:
        parser.error("--gamma must be in [0, 1).")
    SELECTOR_REWARD_MODE=args.reward_mode
    SELECTOR_TRANSITION_MODE=args.transition_mode
    SELECTOR_GAMMA=args.gamma
    CONTEXT_LOSS_MODE=args.context_loss_mode
    if args.output is None:
        folder="notebook_original" if args.reward_mode=="original" else "notebook_skip_zero"
        if args.transition_mode != "terminal" or args.gamma != 0.2:
            folder += f"_{args.transition_mode}_gamma{args.gamma:g}"
        if args.context_loss_mode == "aligned":
            folder += "_context_aligned"
        if args.selection_cost > 0:
            folder += f"_selection_cost{args.selection_cost:g}"
        folder += "_validated_best"
        if args.training_selector_mode == "disagreement":
            folder += f"_train_disagreement_penalty{args.agreement_penalty:g}"
        args.output=str(Path(__file__).resolve().parent/"protocol_runs"/folder)
    if args.command=="train" and args.selector_mode!="original":
        parser.error("--selector-mode is only for evaluate.")
    LEGACY_SELECTOR_MODE=args.selector_mode
    LEGACY_QUERY_GATE=args.query_gate
    seed_tag=str(args.seed) if args.seed is not None else "unseeded"
    gate_suffix = ("_no_filters" if args.query_gate=="none" else
                   "_disagreement_only" if args.query_gate=="disagreement" else
                   "_no_threshold" if args.query_gate=="no-threshold" else "")
    if args.selector_mode=="learned" and scheduled_epsilon:
        gate_suffix += f"_eps{args.epsilon_start:g}to{args.epsilon_end:g}_decay{args.epsilon_decay:g}"
    if args.budget != 214:
        gate_suffix += f"_budget{args.budget}"
    LEGACY_RESULTS_DIR=str(Path("Results") / f"{args.selector_mode}_seed{seed_tag}{gate_suffix}")
    if args.per is not None and args.command!="evaluate":
        parser.error("--per is only for evaluate.")
    LEGACY_PER_LEVELS = [args.per] if args.per is not None else None
    LEGACY_DIAGNOSTIC_SEED = args.seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    BURST_DATA_DIR=Path(args.data_dir).resolve()
    output=Path(args.output).resolve()
    if args.command=="train" and (output/"Models").exists():
        parser.error("Models already exist; choose a fresh --output.")
    if args.command=="evaluate" and not (output/"Models").exists():
        parser.error("Train this reproduction first, or use --output with its Models folder.")
    RUN_TRAINING_STATE_VERSION = TRAINING_STATE_VERSION
    if args.command=="evaluate":
        saved_versions = set()
        for checkpoint_name in ("initial_cnn_model.pth","adversarial_best_context.pth","adversarial_best_dqn.pth"):
            checkpoint=torch.load(output/"Models"/checkpoint_name,map_location="cpu",weights_only=False)
            saved_versions.add(checkpoint.get("training_state_version", 1))
            saved_mode=checkpoint.get("selector_reward_mode","original")
            if saved_mode!=args.reward_mode:
                parser.error("Checkpoint reward mode mismatch: use the matching --reward-mode and --output.")
            if checkpoint.get("selector_transition_mode","terminal") != args.transition_mode:
                parser.error("Checkpoint transition mode mismatch: train in a fresh --output with the requested mode.")
            if checkpoint.get("selector_gamma",0.2) != args.gamma:
                parser.error("Checkpoint gamma mismatch: use the gamma from training.")
            if checkpoint.get("context_loss_mode","legacy") != args.context_loss_mode:
                parser.error("Checkpoint Context loss mismatch: train with the requested --context-loss-mode in a fresh --output.")
            if checkpoint.get("selection_cost", 0.0) != args.selection_cost:
                parser.error("Selection cost mismatch: use the training cost and matching Models folder.")
            if checkpoint.get("training_selector_mode", "original") != args.training_selector_mode:
                parser.error("Training selector mode mismatch: use the mode from training.")
            if args.training_selector_mode == "disagreement" and checkpoint.get("agreement_penalty") != args.agreement_penalty:
                parser.error("Agreement penalty mismatch: use the penalty from training.")
        if len(saved_versions) != 1:
            parser.error("Checkpoint training-state versions differ; use one complete Models folder.")
        RUN_TRAINING_STATE_VERSION = saved_versions.pop()
        if RUN_TRAINING_STATE_VERSION < TRAINING_STATE_VERSION:
            print("Using older checkpoints: validation/snapshot corrections require retraining.")
    output.mkdir(parents=True,exist_ok=True)
    import sys
    manifest_name="reproduction_"+args.command
    if args.command=="evaluate":
        manifest_name += f"_{args.selector_mode}_seed{seed_tag}{gate_suffix}"
    (output/(manifest_name+".json")).write_text(json.dumps(dict(
        mode="legacy notebook diagnostic",source_cells=[2,7,9],random_seed=args.seed if args.seed is not None else "not fixed in original",
        scoring="post-update",selector_mode=args.selector_mode,reward_mode=args.reward_mode,
        transition_mode=args.transition_mode,gamma=args.gamma,
        query_gate=args.query_gate,confidence_gap_threshold=None if args.query_gate in ("none","no-threshold","disagreement") else 0.005,
        context_loss_mode=args.context_loss_mode,
        training_selector_mode=args.training_selector_mode,
        agreement_penalty=args.agreement_penalty if args.training_selector_mode=="disagreement" else None,
        test_label_budget=args.budget,
        selection_cost=args.selection_cost,
        training_state_version=RUN_TRAINING_STATE_VERSION,
        next_state_scope="same input after Context update; not next traffic row",
        online_epsilon=args.epsilon_start if args.selector_mode=="learned" else 1.0,
        online_epsilon_end=args.epsilon_end if args.selector_mode=="learned" else 1.0,
        online_epsilon_decay=args.epsilon_decay if args.selector_mode=="learned" else 1.0,
        epsilon_decay_rule="after successful online selector optimization",
        torch_version=str(torch.__version__),
        data_dir=str(BURST_DATA_DIR)),indent=2),encoding="utf-8")
    previous=Path.cwd()
    try:
        os.chdir(output)
        if args.command=="train":
            print("Selector reward / transitions / gamma:", SELECTOR_REWARD_MODE, SELECTOR_TRANSITION_MODE, SELECTOR_GAMMA)
            print("Context loss mode:", CONTEXT_LOSS_MODE)
            print("Training selector mode / agreement penalty:", TRAINING_SELECTOR_MODE, AGREEMENT_PENALTY)
            print("Action-1 reward cost:", SELECTION_COST)
            train_main()
            # Tag artifacts without altering the saved weights or replay experiences.
            for checkpoint_name in ("initial_cnn_model.pth","adversarial_best_context.pth","adversarial_best_dqn.pth"):
                checkpoint_path=Path("Models")/checkpoint_name
                checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
                checkpoint["selector_reward_mode"]=SELECTOR_REWARD_MODE
                checkpoint["selector_transition_mode"]=SELECTOR_TRANSITION_MODE
                checkpoint["selector_gamma"]=SELECTOR_GAMMA
                checkpoint["context_loss_mode"]=CONTEXT_LOSS_MODE
                checkpoint["training_state_version"]=TRAINING_STATE_VERSION
                checkpoint["selection_cost"]=SELECTION_COST
                checkpoint["training_selector_mode"]=TRAINING_SELECTOR_MODE
                checkpoint["agreement_penalty"]=AGREEMENT_PENALTY if TRAINING_SELECTOR_MODE=="disagreement" else None
                torch.save(checkpoint,checkpoint_path)
        else:
            evaluate_main()
    finally:
        os.chdir(previous)

if __name__=="__main__":
    cli()

