"""AMCAL shared two-action protocol for IUST_MSCN and SDNCampus.

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
        
        # One token per feature. Identity embeddings preserve column semantics.
        token_dim = 64
        num_heads = 4
        self.feature_projection = nn.Linear(1, token_dim)
        self.feature_embedding = nn.Parameter(torch.empty(1, input_size, token_dim))
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        self.weight_projection = nn.Linear(token_dim, 1)
        print(f"Feature-token attention: tokens={input_size}, "
              f"d_model={token_dim}, num_heads={num_heads}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
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
        # [B, F, 1] -> [B, F, D]: attention now spans F feature tokens.
        tokens = self.feature_projection(x_flat.unsqueeze(-1)) + self.feature_embedding
        attended = self.transformer(tokens)
        weights = torch.sigmoid(self.weight_projection(attended).squeeze(-1))
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
            nn.Linear(hidden_size2, 3)
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

def mask_query_action(q_values, states):
    """State[-2] is remaining budget fraction; exhausted states permit routes 0/1."""
    masked = q_values.clone()
    masked[states[:, -2] <= 0, 2] = float("-inf")
    return masked


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
        self.route_examples = []
    
    def select_action(self, state, explore=True):
        """Select action for given state. Always returns a tensor."""
        self.steps_done += 1
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        if explore and random.random() < self.epsilon:
            limits = torch.where(state[:, -2] > 0, 3, 2)
            return (torch.rand(state.size(0), device=self.device) * limits).long()
        else:
            with torch.no_grad():
                was_training = self.policy_net.training
                self.policy_net.eval()
                q_values = mask_query_action(self.policy_net(state), state)
                self.policy_net.train(was_training)
                return q_values.argmax(dim=1)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def observe_route(self, state, frozen_prediction, context_prediction, truth):
        if ROUTE_AUX_WEIGHT <= 0:
            return
        frozen_ok, context_ok = frozen_prediction == truth, context_prediction == truth
        if frozen_ok == context_ok:
            return
        target = 0 if frozen_ok else 1
        self.route_examples.append((state.detach().reshape(-1).cpu().clone(), target))
        if len(self.route_examples) > 50000:
            del self.route_examples[:len(self.route_examples)-50000]

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
            next_actions = mask_query_action(self.policy_net(next_states), next_states).argmax(1)
            self.policy_net.train(was_training)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        if ROUTE_AUX_WEIGHT > 0 and self.route_examples:
            examples = random.sample(self.route_examples, min(32,len(self.route_examples)))
            route_states = torch.stack([x for x,y in examples]).to(self.device)
            route_targets = torch.tensor([y for x,y in examples],dtype=torch.long,device=self.device)
            # Classify only routes 0/1; query utility remains governed by DDQN.
            aux_loss = nn.functional.cross_entropy(self.policy_net(route_states)[:,:2],route_targets)
            loss = loss + ROUTE_AUX_WEIGHT * aux_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Load and preprocess data - GENERALIZED version
SOURCE_FILENAME = 'IUST_MSCN_original.csv'
SELECTION_COST = 0.0001
GAIN_WEIGHT = 1.0
RETENTION_WEIGHT = 1.0
REFERENCE_BATCH_SIZE = 32
REFERENCE_HISTORY_SIZE = 256
TRAIN_BUDGET_FRACTION = 0.2
ONLINE_TRAIN_LR = 1e-5
TRAIN_PERTURBATIONS = (0.0,)

def load_data(dataset_path='Dataset', csv_filename=None):
    csv_filename = csv_filename or SOURCE_FILENAME
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

def transfer_utility(before, after):
    # Same independent examples before/after; current paid sample is excluded.
    if before is None:
        gain, harm = 0.0, 0.0
    else:
        delta = (before-after)/(before+after+1e-8)
        gain = float(np.maximum(delta, 0).mean())
        harm = float(np.maximum(-delta, 0).mean())
    return GAIN_WEIGHT*gain - RETENTION_WEIGHT*harm - SELECTION_COST, gain, harm


class PaidHistory:
    """Reservoir of previously queried labels only, with an isolated RNG."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.rows = []
        self.seen = 0

    def sample(self, current):
        # Also exclude feature-identical past samples, not just the current row.
        eligible = [i for i, row in enumerate(self.rows) if not torch.equal(row[0], current.detach().cpu())]
        if not eligible:
            return None
        ids = self.rng.choice(eligible, min(REFERENCE_BATCH_SIZE, len(eligible)), replace=False)
        return (torch.cat([self.rows[i][0] for i in ids]).to(device),
                torch.cat([self.rows[i][1] for i in ids]).to(device))

    def add(self, x, y):
        self.seen += 1
        row = (x.detach().cpu().clone(), y.detach().cpu().clone())
        if len(self.rows) < REFERENCE_HISTORY_SIZE:
            self.rows.append(row)
        else:
            idx = int(self.rng.integers(self.seen))
            if idx < REFERENCE_HISTORY_SIZE:
                self.rows[idx] = row


ROUTE_AUX_WEIGHT = 0.0
ROUTING_REWARD_MODE = "absolute"

def route_reward_value(action, frozen_prediction, context_prediction, truth):
    frozen_ok = int(frozen_prediction == truth)
    context_ok = int(context_prediction == truth)
    if ROUTING_REWARD_MODE == "relative":
        advantage = context_ok - frozen_ok
        return float(-advantage if action == 0 else advantage)
    selected_ok = frozen_ok if action == 0 else context_ok
    return 1.0 if selected_ok else -1.0

def route_reward_version():
    return "three_action_relative_route_transfer_v1" if ROUTING_REWARD_MODE == "relative" else "three_action_route_transfer_v1"

CONTEXT_REPLAY_SIZE = 0
CONTEXT_CURRENT_WEIGHT = 0.5

class ContextReplay:
    """Only previously paid labels; isolated from the reward reference sampler."""
    def __init__(self, seed):
        self.rows = []
        self.rng = np.random.default_rng(seed)
    def objective(self, cnn, context, x, y, current_loss):
        if CONTEXT_REPLAY_SIZE and self.rows:
            eligible = [r for r in self.rows if not torch.equal(r[0], x.detach().cpu())]
            if eligible:
                ids = self.rng.choice(len(eligible), min(CONTEXT_REPLAY_SIZE,len(eligible)), replace=False)
                rx = torch.cat([eligible[i][0] for i in ids]).to(x.device)
                ry = torch.cat([eligible[i][1] for i in ids]).to(y.device)
                replay_ce = nn.functional.cross_entropy(cnn(context(rx)), ry)
                with torch.no_grad():
                    frozen_ce = nn.functional.cross_entropy(cnn(rx), ry)
                current_loss = CONTEXT_CURRENT_WEIGHT*current_loss + (1-CONTEXT_CURRENT_WEIGHT)*context_objective(replay_ce,frozen_ce)
        return current_loss
    def add(self,x,y):
        if CONTEXT_REPLAY_SIZE:
            self.rows.append((x.detach().cpu().clone(),y.detach().cpu().clone()))
            self.rows = self.rows[-REFERENCE_HISTORY_SIZE:]

def probe_losses(cnn, context, batch):
    if batch is None:
        return None
    with torch.no_grad():
        return nn.functional.cross_entropy(cnn(context(batch[0])), batch[1], reduction='none').cpu().numpy()


class TrainingProbe:
    def __init__(self, features, labels, seed):
        self.x, self.y = features.detach().cpu(), labels.detach().cpu()
        self.rng = np.random.default_rng(seed)

    def sample(self, current):
        eligible = torch.any(self.x != current.detach().cpu(), dim=(1, 2)).nonzero().flatten().numpy()
        if not len(eligible):
            return None
        ids = self.rng.choice(eligible, min(REFERENCE_BATCH_SIZE, len(eligible)), replace=False)
        return self.x[ids].to(device), self.y[ids].to(device)


ROUTE_HISTORY_ENABLED = False

class RouteHistory:
    def __init__(self):
        self.values = [0.0,0.0,0.0,0.0]
        self.count = 0
    def update(self, base_logits, context_logits, labels):
        if not ROUTE_HISTORY_ENABLED:
            return
        with torch.no_grad():
            values = [
                float((base_logits.argmax(1) != labels).float().mean().item()),
                float((context_logits.argmax(1) != labels).float().mean().item()),
                float(nn.functional.cross_entropy(base_logits,labels).item()),
                float(nn.functional.cross_entropy(context_logits,labels).item())]
        # Bounded CE values; cold start is explicit zeros until first paid label.
        values[2:] = [v/(1.0+v) for v in values[2:]]
        self.values = values if self.count == 0 else [0.9*old+0.1*new for old,new in zip(self.values,values)]
        self.count += 1

def budget_state(inputs, probabilities, base_probabilities, remaining, cap, position, count, route_history=None):
    extra = inputs.new_tensor([[remaining / max(cap, 1), position / max(count, 1)]])
    parts = [inputs.squeeze(1), base_probabilities, probabilities]
    if ROUTE_HISTORY_ENABLED:
        values = route_history.values if route_history is not None else [0.,0.,0.,0.]
        parts.append(inputs.new_tensor([values]).expand(inputs.size(0),-1))
    return torch.cat(parts + [extra], dim=1)


def budget_validation(cnn, initial_context, agent, loader, input_length):
    route_history = RouteHistory()
    # Isolated greedy online validation; no selector updates or changes to training RNG.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        context = ContextAwareNetwork(input_length).to(device)
        context.load_state_dict(initial_context)
        context.eval()
        optimizer = optim.Adam(context.parameters(), lr=ONLINE_TRAIN_LR, weight_decay=1e-4)
        features, truths = loader.dataset.tensors
        order = np.random.default_rng(RUN_SEED + 40001).permutation(len(truths))
        cap, used, predictions = int(TRAIN_BUDGET_FRACTION * len(order)), 0, []
        context_replay = ContextReplay(RUN_SEED + 810033)
        validation_loss = 0.0
        was_training = agent.policy_net.training
        agent.policy_net.eval()
        for pos, idx in enumerate(order):
            x, y = features[idx:idx+1], truths[idx:idx+1]
            with torch.no_grad():
                base, logits = cnn(x), cnn(context(x))
                state = budget_state(x, logits.softmax(1), base.softmax(1), cap-used, cap, pos, len(order), route_history=route_history)
                action = int(mask_query_action(agent.policy_net(state), state).argmax(1).item())
                routed_logits = base if action == 0 else logits
                predictions.append(int(routed_logits.argmax(1).item()))
                validation_loss += float(nn.functional.cross_entropy(routed_logits, y).item())
            if action == 2 and used < cap:
                route_history.update(base,logits,y)
                used += 1
                optimizer.zero_grad()
                loss = context_objective(nn.functional.cross_entropy(cnn(context(x)), y), nn.functional.cross_entropy(base, y))
                loss = context_replay.objective(cnn,context,x,y,loss)
                loss.backward()
                optimizer.step()
                context_replay.add(x,y)
        agent.policy_net.train(was_training)
        y = truths[order].cpu().numpy()
        return {'Accuracy': float(accuracy_score(y, predictions)), 'Test Loss': validation_loss / len(order),
                'Precision': float(precision_score(y, predictions, average='weighted', zero_division=0)),
                'Recall': float(accuracy_score(y, predictions)),
                'F1 Score': float(f1_score(y, predictions, average='weighted', zero_division=0))}


def build_training_domains(cnn, features, labels):
    """FGSM in saved StandardScaler units; called only on train or validation tensors."""
    levels = TRAIN_PERTURBATIONS
    domains = {0.0: features.detach()}
    if any(level > 0 for level in levels):
        signs = []
        for start in range(0, len(labels), 64):
            with torch.enable_grad():
                x = features[start:start+64].detach().clone().requires_grad_(True)
                loss = nn.functional.cross_entropy(cnn(x), labels[start:start+64])
                gradient, = torch.autograd.grad(loss, x)
            signs.append(gradient.sign().detach())
        direction = torch.cat(signs)
        for level in levels:
            if level > 0:
                domains[level] = (features.detach() + level*direction).detach()
    return {level: domains[level] for level in levels}


def train_adversarial_system(cnn_model, dqn_agent, context_model, full_loader, val_loader, class_counts, input_length, num_classes, fine_tune_epochs=500, patience=50, pretrain_epochs=20, episode_loaders=None):
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
    
    episode_full, episode_val = episode_loaders if episode_loaders is not None else (full_loader, val_loader)
    clean_train_x, train_y = episode_full.dataset.tensors
    train_domains = build_training_domains(cnn_model, clean_train_x, train_y)
    validation_x, validation_y = episode_val.dataset.tensors
    validation_domains = build_training_domains(cnn_model, validation_x, validation_y)
    print("Training domain IDs:", TRAIN_PERTURBATIONS, "; see domain manifest for construction; no target/test files used.")
    source_train_y = train_y
    episode_initial_context = deepcopy(context_model.state_dict())
    print("Starting budget-aware episodic training...")
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
        
        # Each epoch is an episode: each training row appears once, shuffled.
        context_model.load_state_dict(episode_initial_context)
        context_model.eval()
        context_optimizer = optim.Adam(context_model.parameters(), lr=ONLINE_TRAIN_LR, weight_decay=1e-4)
        episode_level = TRAIN_PERTURBATIONS[epoch % len(TRAIN_PERTURBATIONS)]
        domain = train_domains[episode_level]
        train_x, train_y = domain if isinstance(domain, tuple) else (domain, source_train_y)
        print(f"  Training perturbation: {episode_level:g}")
        order = np.random.default_rng(RUN_SEED + epoch).permutation(len(train_y))
        cap, used = int(np.floor(TRAIN_BUDGET_FRACTION * len(order))), 0
        route_history = RouteHistory()
        context_replay = ContextReplay(RUN_SEED + epoch + 810033)
        training_probe = TrainingProbe(train_x, train_y, RUN_SEED + epoch + 910033)
        selected_rewards, selected_gains, selected_harms = [], [], []
        allowed_actions = []
        last_query = None
        for position, idx in enumerate(order):
            inputs, labels = train_x[idx:idx+1], train_y[idx:idx+1]
            total_samples += 1
            with torch.no_grad():
                base = cnn_model(inputs)
                before = cnn_model(context_model(inputs))
                state = budget_state(inputs, before.softmax(1), base.softmax(1), cap-used, cap, position, len(order), route_history=route_history)
            proposed = int(dqn_agent.select_action(state).item())
            actions.append(proposed)
            allowed = used < cap
            selected = proposed == 2 and allowed
            routed_prediction = int((base if proposed == 0 else before).argmax(1).item())
            route_reward = route_reward_value(proposed, int(base.argmax(1).item()), int(before.argmax(1).item()), int(labels.item()))
            reward = route_reward
            if allowed:
                allowed_actions.append(proposed)
            if selected:
                with torch.no_grad():
                    if position+1 < len(order):
                        nx = train_x[order[position+1]:order[position+1]+1]
                        skip_next = budget_state(nx, cnn_model(context_model(nx)).softmax(1), cnn_model(nx).softmax(1), cap-used, cap, position+1, len(order), route_history=route_history)
                    else:
                        skip_next = torch.zeros_like(state)
                dqn_agent.observe_route(state,int(base.argmax(1).item()),int(before.argmax(1).item()),int(labels.item()))
                route_history.update(base,before,labels)
                reference_batch = training_probe.sample(inputs)
                reference_before = probe_losses(cnn_model, context_model, reference_batch)
                used += 1
                context_optimizer.zero_grad()
                context_loss = cnn_criterion(cnn_model(context_model(inputs)), labels)
                loss = context_objective(context_loss, cnn_criterion(base, labels))
                loss = context_replay.objective(cnn_model,context_model,inputs,labels,loss)
                loss.backward()
                context_optimizer.step()
                context_replay.add(inputs,labels)
                context_updated_samples += 1
                running_context_loss += float(loss.item())
                with torch.no_grad():
                    after = cnn_model(context_model(inputs))
                reference_after = probe_losses(cnn_model, context_model, reference_batch)
                utility, gain, harm = transfer_utility(reference_before, reference_after)
                reward = route_reward + utility
                selected_rewards.append(reward)
                selected_gains.append(gain)
                selected_harms.append(harm)
                last_query = position+1
            else:
                after = before
            with torch.no_grad():
                if position+1 < len(order) and used < cap:
                    next_idx = order[position+1]
                    next_inputs = train_x[next_idx:next_idx+1]
                    next_prob = cnn_model(context_model(next_inputs)).softmax(1)
                    next_state = budget_state(next_inputs, next_prob, cnn_model(next_inputs).softmax(1), cap-used, cap, position+1, len(order), route_history=route_history)
                else:
                    next_state = torch.zeros_like(state)
            if allowed:
                terminal = float(position+1 == len(order) or used >= cap)
                dqn_agent.replay_buffer.push(state.detach(), proposed, reward, next_state.detach(), terminal)
                if selected:
                    for skip_action, skip_logits in ((0, base), (1, before)):
                        skip_reward = route_reward_value(skip_action, int(base.argmax(1).item()), int(before.argmax(1).item()), int(labels.item()))
                        dqn_agent.replay_buffer.push(state.detach(), skip_action, skip_reward, skip_next.detach(), float(position+1 == len(order)))
                dqn_loss = dqn_agent.optimize()
                if dqn_loss is not None:
                    running_dqn_loss += dqn_loss
                    dqn_loss_count += 1
                    dqn_agent.update_epsilon()
            rewards.append(reward)
        assert used <= cap
        print(f"  Training episode labels: {used}/{cap}; online LR: {ONLINE_TRAIN_LR:g}")
        # Episodic adaptation uses the fixed deployment learning rate.
        
        dqn_agent.update_epsilon()
        
        if (epoch + 1) % 15 == 0:
            dqn_agent.update_target_network()
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        action_dist = np.bincount(actions, minlength=3) if actions else [0, 0]
        
        print(f"Adversarial Epoch {epoch+1}/{fine_tune_epochs}:")
        print(f"  Context Loss: {running_context_loss / max(context_updated_samples, 1):.4f}")
        print(f"  DQN Loss: {running_dqn_loss / max(dqn_loss_count, 1):.4f}")
        print(f"  Context Updated Samples: {context_updated_samples}")
        print(f"  Mean paid reward: {np.mean(selected_rewards) if selected_rewards else 0:.7f}")
        print(f"  Mean probe gain/harm/cost: {np.mean(selected_gains) if selected_gains else 0:.7f} / {np.mean(selected_harms) if selected_harms else 0:.7f} / {SELECTION_COST:.7f}")
        print(f"  Positive paid rewards: {100*np.mean(np.asarray(selected_rewards)>0) if selected_rewards else 0:.2f}%")
        print(f"  Actions while budget available (direct/context/query): {np.bincount(allowed_actions, minlength=3)}; last query: {last_query}/{len(order)}")
        print(f"  Proposed actions over entire stream, including after cap: {action_dist}")
        
        # Validation
        domain_results = []
        for validation_level in TRAIN_PERTURBATIONS:
            domain = validation_domains[validation_level]
            domain_x, domain_y = domain if isinstance(domain, tuple) else (domain, validation_y)
            domain_loader = DataLoader(TensorDataset(domain_x, domain_y), batch_size=64)
            result = budget_validation(cnn_model, episode_initial_context, dqn_agent, domain_loader, input_length)
            print(f"  Validation perturbation {validation_level:g}: online accuracy {result['Accuracy']:.4f}")
            domain_results.append(result)
        val_results = {key: float(np.mean([result[key] for result in domain_results])) for key in domain_results[0]}
        print(f"Validation Results at Epoch {epoch+1}:")
        print(f"  Val Loss: {val_results['Test Loss']:.4f}")
        print(f"  Val Accuracy: {val_results['Accuracy']:.4f}")
        print(f"  Val Precision: {val_results['Precision']:.4f}")
        print(f"  Val Recall: {val_results['Recall']:.4f}")
        print(f"  Val F1 Score: {val_results['F1 Score']:.4f}")
        
        if val_results['Accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['Accuracy']
            epochs_no_improve = 0
            best_context_state = deepcopy(episode_initial_context)
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
        'protocol_version': 3,
        'context_model_state_dict': context_model.state_dict(),
        'input_length': input_length,
        'num_classes': num_classes
    }, 'Models/adversarial_best_context.pth')
    print("Saved best adversarial Context model")
    
    torch.save({
        'preprocessing_sha256': file_sha256('Models/protocol_preprocessing.joblib'),
        'protocol_version': 3,
        'training_perturbations': list(TRAIN_PERTURBATIONS),
        'training_shift_protocol': 'one shuffled FGSM domain per epoch; reset Context and Adam; validation mean across domains; epsilon in training StandardScaler units',
        'budget_state_version': 4,
        'route_history': ROUTE_HISTORY_ENABLED, 'route_history_ema': 0.1,
        'route_aux_weight': ROUTE_AUX_WEIGHT,
        'reward_version': route_reward_version(), 'routing_reward_mode': ROUTING_REWARD_MODE,
        'gain_weight': GAIN_WEIGHT, 'retention_weight': RETENTION_WEIGHT,
        'reference_batch_size': REFERENCE_BATCH_SIZE, 'reference_history_size': REFERENCE_HISTORY_SIZE,
        'context_replay_size': CONTEXT_REPLAY_SIZE, 'context_current_weight': CONTEXT_CURRENT_WEIGHT,
        'selection_cost': SELECTION_COST,
        'training_budget_fraction': TRAIN_BUDGET_FRACTION,
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
        'protocol_version': 3,
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
    
    dqn_agent = DQNSelectorAgent(input_length + 2*num_classes + 2 + (4 if ROUTE_HISTORY_ENABLED else 0), device)
    context_model = ContextAwareNetwork(input_length).to(device)
    
    print(f"DQN Agent initialized with input size: {input_length + 2*num_classes + 2 + (4 if ROUTE_HISTORY_ENABLED else 0)}")
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


# Utility variant uses next-sample transitions and paid-history retention reward.
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
        protocol_version=3, seed=RUN_SEED, source_sha256=file_sha256(source),
        source_rows=len(frame), feature_names=list(frame.columns[:-1]),
        label_name=frame.columns[-1], scaler=scaler, label_encoder=encoder,
        class_counts=class_counts, train_ids=train_ids, val_ids=val_ids,
        test_ids=test_ids, feature_groups=feature_groups(frame),
        source_labels=frame.iloc[:, -1].to_numpy())
    Path("Models").mkdir(exist_ok=True)
    joblib.dump(artifact, "Models/protocol_preprocessing.joblib")
    report = dict(protocol_version=3, seed=RUN_SEED,
        training_perturbations=list(TRAIN_PERTURBATIONS),
        source_sha256=artifact["source_sha256"], source_rows=len(frame),
        split_sizes={k:len(v) for k,v in
                     (("train",train_ids),("validation",val_ids),("test",test_ids))},
        split_rule="group feature-identical rows before random splitting",
        selector_target="DDQN gamma=0.95; next sample; terminal at budget exhaustion or stream end",
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
                          assume_row_aligned=False):
    source_path = Path(data_dir) / SOURCE_FILENAME
    if file_sha256(source_path) != artifact["source_sha256"]:
        raise ValueError("Original CSV differs from the training source.")
    path = source_path if attack_file is None else Path(attack_file)
    frame = pd.read_csv(path)
    expected = artifact["feature_names"] + [artifact["label_name"]]
    if list(frame.columns) != expected:
        raise ValueError("Feature/label schema or column order differs from training.")
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


def print_stream_summary(metrics, name, mode, lr):
    level = name.split("_size_")[-1].split("_Step_")[0] if "_size_" in name else name
    print("\n" + "="*68)
    print(f"Level: {level} | Mode: {mode} | LR: {lr:g} | Samples: {metrics['evaluated_rows']}")
    print("-"*68)
    print(f"Online accuracy: {100*metrics['accuracy']:.2f}% | Macro-F1: {100*metrics['macro_f1']:.2f}%")
    print(f"Frozen CNN: {100*metrics['base_accuracy']:.2f}% | Initial Context: {100*metrics['no_adaptation_context_accuracy']:.2f}%")
    if "routing_oracle" in metrics:
        oracle = metrics["routing_oracle"]
        print(f"Routing oracle: {100*oracle['accuracy']:.2f}% | Gap: {oracle['routing_gap_pp']:.2f} pp")
        print(f"Context-only online: {100*oracle['context_only_online_accuracy']:.2f}%")
    print(f"Labels used: {metrics['unique_labels_queried']}/{metrics['label_budget']} | Actual: {100*metrics['actual_label_fraction']:.2f}%")
    if metrics.get("router_online_lr",0) > 0:
        print("Independent router updates:",metrics["router_online_updates"],"| Preference examples:",metrics["router_online_examples"])
    if metrics.get("query_schedule") == "uniform":
        print("Feedback: fixed label-blind schedule (diagnostic)")
    for key,label in (("source_validation","Clean reference"),("same_level","Current-level Context")):
        row = metrics.get("forgetting",{}).get(key)
        if row:
            print(f"{label}: {100*row['initial_accuracy']:.2f}% -> {100*row['final_accuracy']:.2f}% | Drop: {row['drop_pp']:+.2f} pp | Broken: {row['correct_to_wrong']}")
    if metrics.get("continual"):
        print(f"Continual domain: {metrics['continual_domain_index']} | Total reference drop: {metrics.get('continual_reference_drop_pp',float('nan')):+.2f} pp")
    print("Positive Drop = net forgetting; online scores are before feedback; final scores are retrospective.")
    print("="*68, flush=True)


def evaluate_stream(args):
    route_history = RouteHistory()
    output = Path(args.output).resolve()
    models = output / "Models"
    artifact_path = models / "protocol_preprocessing.joblib"
    artifact = joblib.load(artifact_path)
    if artifact.get('protocol_version') != 3:
        raise ValueError('Preprocessing is not protocol v3; retrain in a fresh directory.')
    frame, source_ids, attack_path, alignment = load_evaluation_frame(
        args.data_dir, args.attack_file, artifact, args.row_map, args.assume_row_aligned)
    if args.stream_order == "shuffle":
        order = np.random.default_rng(args.seed).permutation(len(frame))
        frame, source_ids = frame.iloc[order].copy(), source_ids[order]
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
        if saved.get("protocol_version") != 3:
            raise ValueError("Checkpoint is not protocol v3; retrain in a fresh output directory.")
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
    agent = DQNSelectorAgent(input_length + 2*num_classes + 2 + (4 if ROUTE_HISTORY_ENABLED else 0), device,
                             batch_size=args.online_selector_batch_size)
    independent_router = None
    online_router = None
    if args.independent_router:
        if args.query_schedule != "uniform" or args.continual_state:
            raise ValueError("Independent routing control requires uniform queries and independent levels.")
        from amcal_independent_router import RouteHead
        router_checkpoint = torch.load(models/"independent_router.pth",map_location="cpu",weights_only=True)
        if router_checkpoint["context_lr"] != args.lr or router_checkpoint["budget_fraction"] != args.budget_fraction:
            print(f'Router development LR/budget: {router_checkpoint["context_lr"]:g}/{router_checkpoint["budget_fraction"]:g}; evaluation: {args.lr:g}/{args.budget_fraction:g}')
        for name, expected in router_checkpoint["model_hashes"].items():
            if file_sha256(models/name) != expected:
                raise ValueError("Independent router CNN/Context/preprocessing mismatch: "+name)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            independent_router = RouteHead(input_length+2*num_classes).to(device)
        independent_router.load_state_dict(router_checkpoint["state_dict"])
        independent_router.eval()
        if args.router_online_lr > 0 and args.selector_mode == "learned":
            from amcal_independent_router import OnlineRouteLearner
            online_router = OnlineRouteLearner(independent_router,args.router_online_lr,args.seed+720019)
    selector_checkpoint = checkpoint("adversarial_best_dqn.pth")
    if selector_checkpoint.get("budget_state_version") != 4 or selector_checkpoint.get("selection_cost") != SELECTION_COST:
        raise ValueError("Budget-aware checkpoint and matching selection cost required; retrain in a fresh output.")
    if selector_checkpoint.get("route_history",False) != ROUTE_HISTORY_ENABLED:
        raise ValueError("Route-history state differs from training; use matching checkpoint.")
    if selector_checkpoint.get("route_aux_weight",0.0) != ROUTE_AUX_WEIGHT:
        raise ValueError("Route auxiliary weight differs from training.")
    if selector_checkpoint.get("routing_reward_mode","absolute") != ROUTING_REWARD_MODE:
        raise ValueError("Routing reward differs from training; train a new matching checkpoint.")
    expected_reward = dict(gain_weight=GAIN_WEIGHT, retention_weight=RETENTION_WEIGHT,
                           reference_batch_size=REFERENCE_BATCH_SIZE, reference_history_size=REFERENCE_HISTORY_SIZE)
    if any(selector_checkpoint.get(k) != v for k, v in expected_reward.items()):
        raise ValueError("Reward/reference configuration differs from training checkpoint.")
    if selector_checkpoint.get("context_replay_size",0) != CONTEXT_REPLAY_SIZE or selector_checkpoint.get("context_current_weight",0.5) != CONTEXT_CURRENT_WEIGHT:
        raise ValueError("Context replay differs from checkpoint; use the matching new training run.")
    agent.policy_net.load_state_dict(selector_checkpoint["dqn_policy_net_state_dict"])
    agent.update_target_network()
    agent.policy_net.eval()
    # Online replay starts empty; skipped samples create label-free zero-reward transitions.
    optimizer = optim.Adam(context.parameters(), lr=args.lr, weight_decay=1e-4)
    session_path = Path(args.continual_state) if args.continual_state else None
    session_identity = dict(preprocessing_sha256=digest, cnn_sha256=file_sha256(models/"initial_cnn_model.pth"),
        context_sha256=file_sha256(models/"adversarial_best_context.pth"), selector_sha256=file_sha256(models/"adversarial_best_dqn.pth"),
        seed=args.seed, selector_mode=args.selector_mode, lr=args.lr, budget_fraction=args.budget_fraction,
        query_schedule=args.query_schedule, eval_epsilon=args.eval_epsilon, reward=ROUTING_REWARD_MODE, replay=CONTEXT_REPLAY_SIZE,
        route_history=ROUTE_HISTORY_ENABLED, aux=ROUTE_AUX_WEIGHT)
    continuation = None
    if session_path is not None and session_path.exists():
        # Keep replay and paid-history tensors on CPU; parameter state loaders handle device transfer.
        continuation = torch.load(session_path,map_location="cpu",weights_only=False)
        if continuation["identity"] != session_identity:
            raise ValueError("Continual session configuration/model mismatch.")
        if file_sha256(attack_path) in continuation["domains"]:
            raise ValueError("This domain was already evaluated in this session; start a new session to repeat.")
        context.load_state_dict(continuation["context"])
        optimizer.load_state_dict(continuation["context_optimizer"])
        agent.policy_net.load_state_dict(continuation["policy"])
        agent.target_net.load_state_dict(continuation["target"])
        agent.optimizer.load_state_dict(continuation["selector_optimizer"])
        agent.replay_buffer = continuation["replay_buffer"]
        agent.route_examples = continuation["route_examples"]
        agent.epsilon = continuation["epsilon"]
        agent.steps_done = continuation["steps_done"]
        route_history = continuation["route_history"]
        print("Continuing prior domain: Context, DDQN, optimizers and replay retained.",flush=True)
    weights = torch.tensor(1.0 / artifact["class_counts"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    def observer(features, truths):
        predictions = []
        with torch.no_grad():
            for start in range(0, len(features), 256):
                batch = torch.tensor(features[start:start+256], dtype=torch.float32, device=device).unsqueeze(1)
                predictions.extend(cnn(context(batch)).argmax(1).cpu().tolist())
        return np.asarray(predictions, dtype=np.int64)

    initial_predictions = observer(X, y)
    forgetting_initial = None
    if args.forgetting_diagnostics:
        source = pd.read_csv(Path(args.data_dir) / SOURCE_FILENAME)
        if artifact.get("reference_csv"):
            if file_sha256(artifact["reference_csv"]) != artifact["reference_sha256"]:
                raise ValueError("Reference CSV changed since training.")
            reference = pd.read_csv(artifact["reference_csv"])
        else:
            reference = source.iloc[artifact["val_ids"]]
        reference_x = artifact["scaler"].transform(reference.iloc[:, :-1].to_numpy())
        reference_y = artifact["label_encoder"].transform(reference.iloc[:, -1].to_numpy())
        forgetting_initial = observer(reference_x, reference_y)

    frozen_snapshot = deepcopy(cnn.state_dict())
    queried_ids, trace, predicted, base_predicted = set(), [], [], []
    selector_updates, context_updates = 0, 0
    random_selector = np.random.default_rng(args.seed + 719003)
    exploration_rng = np.random.default_rng(args.seed + 619003)
    exploration_decisions = 0
    context_replay = ContextReplay(args.seed + 810033)
    paid_history = PaidHistory(args.seed + 910033)
    if continuation is not None:
        context_replay = continuation["context_replay"]
        paid_history = continuation["paid_history"]
        random_selector.bit_generator.state = continuation["random_selector_rng"]
        exploration_rng.bit_generator.state = continuation["exploration_rng"]
        random.setstate(continuation["python_rng"])
        torch.set_rng_state(continuation["torch_rng"].cpu())
        if torch.cuda.is_available() and continuation.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in continuation["cuda_rng"]])
    first_query_step = last_query_step = first_selector_update_step = None
    scheduled_queries = set()
    if args.query_schedule == "uniform":
        # Independent RNG and features/labels are not consulted.
        schedule_rng = np.random.default_rng(args.seed + 819003)
        scheduled_queries = set(schedule_rng.choice(len(y),budget,replace=False).tolist())
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
            state = budget_state(inputs, adapted_prob, base_prob, budget-len(queried_ids), budget, step-1, len(y), route_history=route_history)
            available_actions = 3 if len(queried_ids) < budget else 2
            explored = False
            if args.selector_mode == "random":
                action = int(random_selector.integers(0, available_actions))
            elif exploration_rng.random() < args.eval_epsilon:
                action = int(exploration_rng.integers(0, available_actions))
                explored = True
                exploration_decisions += 1
            else:
                action = int(agent.select_action(state, explore=False).item())
            if args.query_schedule == "uniform":
                if step-1 in scheduled_queries:
                    action = 2
                elif action == 2:
                    if args.selector_mode == "random":
                        action = int(random_selector.integers(0,2))
                    elif explored:
                        action = int(exploration_rng.integers(0,2))
                    else:
                        action = int(agent.policy_net(state)[:,:2].argmax(1).item())
            if independent_router is not None and args.selector_mode == "learned" and action != 2 and not explored:
                from amcal_independent_router import route_features
                action = int(independent_router(route_features(inputs,base_prob,adapted_prob)).argmax(1).item())
            prediction = base_prediction if action == 0 else int(adapted_logits.argmax(1).item())
            confidence_gap = float(abs(adapted_prob.max().item()-base_prob.max().item()))
        # Capture the scored prediction before acquiring a label or changing weights.
        predicted.append(prediction)
        base_predicted.append(base_prediction)
        adaptation_allowed = budget > 0 and len(queried_ids) < budget
        query = (action == 2 and adaptation_allowed and source_id not in queried_ids)
        reward, loss_value, selector_loss = None, None, None
        gain, harm = None, None
        if query:
            with torch.no_grad():
                if step < len(y):
                    nx = torch.tensor(X[step], dtype=torch.float32, device=device).reshape(1, 1, -1)
                    skip_next = budget_state(nx, cnn(context(nx)).softmax(1), cnn(nx).softmax(1), budget-len(queried_ids), budget, step, len(y), route_history=route_history)
                else:
                    skip_next = torch.zeros_like(state)
            queried_ids.add(source_id)
            if first_query_step is None:
                first_query_step = step
            last_query_step = step
            labels = torch.tensor([int(truth)], dtype=torch.long, device=device)
            if online_router is not None:
                online_router.observe(inputs,base_prob,adapted_prob,labels)
            if args.selector_mode == "learned":
                agent.observe_route(state,base_prediction,int(adapted_logits.argmax(1).item()),int(truth))
            route_history.update(base_logits,adapted_logits,labels)
            reference_batch = paid_history.sample(inputs)
            reference_before = probe_losses(cnn, context, reference_batch)
            optimizer.zero_grad()
            # Keep inference-mode batch statistics and dropout during online adaptation.
            context_loss = criterion(cnn(context(inputs)), labels)
            original_loss = criterion(base_logits, labels)
            loss = context_objective(context_loss, original_loss)
            loss = context_replay.objective(cnn,context,inputs,labels,loss)
            loss.backward()
            optimizer.step()
            context_replay.add(inputs,labels)
            context_updates += 1
            loss_value = float(loss.item())
            # Reward/state may use post-update outputs of this queried sample.
            # Those outputs never replace the pre-update scored prediction.
            with torch.no_grad():
                after_logits = cnn(context(inputs))
                after_prob = after_logits.softmax(dim=1)
                after_prediction = int(after_logits.argmax(1).item())
            reference_after = probe_losses(cnn, context, reference_batch)
            utility, gain, harm = transfer_utility(reference_before, reference_after)
            reward = route_reward_value(action, base_prediction, int(adapted_logits.argmax(1).item()), int(truth)) + utility
            paid_history.add(inputs, labels)
            effective_action = 2
        else:
            # No label access for an unqueried sample (including exhausted budget).
            effective_action, reward = (1 if action == 2 else action), None
        # Next transition observes the next feature vector, never its label.
        with torch.no_grad():
            if step < len(y) and len(queried_ids) < budget:
                next_inputs = torch.tensor(X[step], dtype=torch.float32, device=device).reshape(1, 1, -1)
                next_prob = cnn(context(next_inputs)).softmax(1)
                next_state = budget_state(next_inputs, next_prob, cnn(next_inputs).softmax(1), budget-len(queried_ids), budget, step, len(y), route_history=route_history)
            else:
                next_state = torch.zeros_like(state)
        # A budget-blocked action 1 is not an observed policy action 0.
        # Freeze both learners at zero budget and after budget exhaustion.
        transition_stored = args.selector_mode == "learned" and query
        if transition_stored:
            agent.replay_buffer.push(state.detach(), effective_action, reward,
                                     next_state.detach(), float(step == len(y) or len(queried_ids) >= budget))
            for skip_action, skip_prediction in ((0, base_prediction), (1, int(adapted_logits.argmax(1).item()))):
                skip_reward = route_reward_value(skip_action, base_prediction, int(adapted_logits.argmax(1).item()), int(truth))
                agent.replay_buffer.push(state.detach(), skip_action, skip_reward, skip_next.detach(), float(step == len(y)))
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
            base_prediction=base_prediction, context_prediction=int(adapted_logits.argmax(1).item()),
            true_label=int(truth), correct=int(prediction == truth),
            action=action, effective_action=effective_action, exploration=explored,
            confidence_gap=confidence_gap, queried=query,
            transition_stored=transition_stored, budget_blocked=not adaptation_allowed,
            replay_size=len(agent.replay_buffer),
            unique_labels_used=len(queried_ids), context_updates=context_updates,
            selector_updates=selector_updates, reward=reward,
            context_loss=loss_value, selector_loss=selector_loss, normalized_gain=gain, normalized_reference_harm=harm))
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
        alignment=alignment, seed=args.seed, budget_fraction=args.budget_fraction,
        query_rule='action 2 and remaining unique-label budget; query action masked after exhaustion; choose route 0/1',
        route_history=ROUTE_HISTORY_ENABLED, route_history_ema=0.1, route_aux_weight=ROUTE_AUX_WEIGHT, reward_version=route_reward_version(), routing_reward_mode=ROUTING_REWARD_MODE, gain_weight=GAIN_WEIGHT, retention_weight=RETENTION_WEIGHT,
        reward_reference="online: previous queried samples only, excluding feature-identical current; train: independent training batch excluding feature-identical current; per-example positive gain and damage; no reference gradient",
        context_replay_size=CONTEXT_REPLAY_SIZE, context_current_weight=CONTEXT_CURRENT_WEIGHT,
        reference_batch_size=REFERENCE_BATCH_SIZE, reference_history_size=REFERENCE_HISTORY_SIZE,
        eval_epsilon=args.eval_epsilon, exploration_decisions=exploration_decisions, action_mask="query disabled at exhausted budget in behavior and DDQN next-action maximization",
        selection_cost=SELECTION_COST, state_protocol=("features; frozen probabilities; context probabilities; optional four previous-paid pre-feedback route EMA statistics; remaining-budget fraction; elapsed-stream fraction"),
        selector_mode=args.selector_mode, random_query_probability=1/3 if args.selector_mode == 'random' else None,
        online_lr=args.lr, attack_sha256=file_sha256(attack_path),
        preprocessing_sha256=digest, device=str(device),
        protocol_version=3, selector_target="DDQN gamma=0.95; next sample transitions; terminal at budget exhaustion or stream end",
        uncertainty="1 - max modulated softmax", target_sync_steps=15,
        train_seed=artifact["seed"], torch_version=str(torch.__version__),
        numpy_version=np.__version__, pandas_version=pd.__version__)
    paid_rows = [row for row in trace if row["queried"]]
    metrics["paid_reward_statistics"] = dict(
        mean_reward=float(np.mean([r["reward"] for r in paid_rows])) if paid_rows else None,
        mean_gain=float(np.mean([r["normalized_gain"] for r in paid_rows])) if paid_rows else None,
        mean_harm=float(np.mean([r["normalized_reference_harm"] for r in paid_rows])) if paid_rows else None,
        positive_fraction=float(np.mean([r["reward"] > 0 for r in paid_rows])) if paid_rows else None,
        cost=SELECTION_COST,
        actions_while_budget_available=np.bincount([r["action"] for r in trace if not r["budget_blocked"]], minlength=3).tolist())
    metrics["action_counts"] = np.bincount([r["action"] for r in trace], minlength=3).tolist()
    metrics["routing_reward"] = ("pre-feedback relative correctness: chosen route minus alternative route; ties zero; query adds transfer gain minus damage minus cost" if ROUTING_REWARD_MODE == "relative" else "pre-feedback routed correctness +1/-1; query adds transfer gain minus damage minus cost")
    metrics["online_reward_access"] = "queried labels only; skip-route counterfactuals use the same paid label; no unqueried correctness reward"
    metrics["training_perturbations"] = selector_checkpoint.get("training_perturbations", [0.0])
    metrics["no_adaptation_context_accuracy"] = float(accuracy_score(y, initial_predictions))
    metrics["stream_order"] = args.stream_order
    metrics["source_file"] = SOURCE_FILENAME
    if args.forgetting_diagnostics:
        def comparison(truths, before, after):
            initial = float(accuracy_score(truths, before))
            final = float(accuracy_score(truths, after))
            return dict(initial_accuracy=initial, final_accuracy=final,
                        drop_pp=100*(initial-final),
                        correct_to_wrong=int(np.sum((before == truths) & (after != truths))),
                        wrong_to_correct=int(np.sum((before != truths) & (after == truths))))
        metrics["forgetting"] = dict(
            same_level=comparison(y, initial_predictions, observer(X, y)),
            source_validation=comparison(reference_y, forgetting_initial, observer(reference_x, reference_y)),
            protocol="observer only; reference is source validation partition; no gradient or reward use; final target score retrospective")

    if args.routing_oracle:
        # Observer only, after the stream: never feed oracle labels/predictions to RL.
        base_ok = np.asarray([r["base_prediction"] == r["true_label"] for r in trace])
        context_ok = np.asarray([r["context_prediction"] == r["true_label"] for r in trace])
        routed_ok = np.asarray([bool(r["correct"]) for r in trace])
        oracle_ok = base_ok | context_ok
        metrics["routing_oracle"] = dict(
            accuracy=float(oracle_ok.mean()),
            routing_gap_pp=float(100*(oracle_ok.mean()-routed_ok.mean())),
            context_only_online_accuracy=float(context_ok.mean()),
            both_correct=int((base_ok & context_ok).sum()),
            only_frozen_correct=int((base_ok & ~context_ok).sum()),
            only_context_correct=int((~base_ok & context_ok).sum()),
            both_wrong=int((~oracle_ok).sum()),
            missed_correct_route=int((oracle_ok & ~routed_ok).sum()),
            protocol="true-label observer; exact per-sample accuracy ceiling over pre-feedback frozen/Context routes along unchanged policy update trajectory; not an oracle query policy or global adaptation optimum")
        for row, ok in zip(trace, oracle_ok):
            row["oracle_route_correct"] = bool(ok)

    metrics["router_online_lr"] = args.router_online_lr
    metrics["router_online_updates"] = online_router.updates if online_router is not None else 0
    metrics["router_online_examples"] = online_router.examples if online_router is not None else 0
    metrics["router_online_protocol"] = "paid pre-feedback route preferences only; replay 256, minibatch 32, reset per level; one update per paid label after first preference" if args.router_online_lr > 0 else "disabled"
    metrics["independent_router"] = bool(args.independent_router and args.selector_mode == "learned")
    if args.independent_router:
        metrics["router_development_context_lr"] = router_checkpoint["context_lr"]
        metrics["router_development_budget_fraction"] = router_checkpoint["budget_fraction"]
        metrics["router_exploration_protocol"] = "epsilon random route 0/1 on non-query rows; fixed uniform query schedule"
    metrics["query_schedule"] = args.query_schedule
    if args.query_schedule == "uniform":
        if len(queried_ids) != budget:
            raise RuntimeError("Scheduled query budget was not consumed exactly.")
        metrics["query_rule"] = "diagnostic fixed uniform label-blind query positions; action 2 forced at scheduled positions; policy/random controls only routes elsewhere"
        metrics["scheduled_query_count"] = len(scheduled_queries)
    metrics["continual"] = session_path is not None
    if session_path is not None:
        prior_domains = continuation["domains"] if continuation is not None else []
        baseline_ref = continuation.get("initial_reference") if continuation is not None else forgetting_initial
        metrics["continual_session"] = session_path.parent.name
        metrics["continual_domain_index"] = len(prior_domains) + 1
        metrics["continual_budget_protocol"] = "20% cap separately per domain; labels charged again per domain; optimizer/replay retained"
        if baseline_ref is not None and args.forgetting_diagnostics:
            metrics["continual_reference_initial_accuracy"] = float((np.asarray(baseline_ref)==reference_y).mean())
            metrics["continual_reference_final_accuracy"] = float((observer(reference_x,reference_y)==reference_y).mean())
            metrics["continual_reference_drop_pp"] = 100*(metrics["continual_reference_initial_accuracy"]-metrics["continual_reference_final_accuracy"])
        payload = dict(identity=session_identity, domains=prior_domains+[file_sha256(attack_path)],
            context=context.state_dict(),context_optimizer=optimizer.state_dict(),
            policy=agent.policy_net.state_dict(),target=agent.target_net.state_dict(),selector_optimizer=agent.optimizer.state_dict(),
            replay_buffer=agent.replay_buffer,route_examples=agent.route_examples,epsilon=agent.epsilon,steps_done=agent.steps_done,
            route_history=route_history,context_replay=context_replay,paid_history=paid_history,
            random_selector_rng=random_selector.bit_generator.state,exploration_rng=exploration_rng.bit_generator.state,
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,initial_reference=baseline_ref)
        session_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=session_path.with_suffix(".tmp")
        torch.save(payload,temporary)
        temporary.replace(session_path)
        print("Saved continual stream state:",session_path,flush=True)

    name = "clean" if args.attack_file is None else Path(args.attack_file).stem
    tag = f"oracle{int(args.routing_oracle)}_masked_eps{args.eval_epsilon:g}_three_route_transfer_a{GAIN_WEIGHT:g}_b{RETENTION_WEIGHT:g}_cost{SELECTION_COST:g}_{name}_{args.selector_mode}_{args.stream_order}_forgetting{int(args.forgetting_diagnostics)}_budget{args.budget_fraction:g}_seed{args.seed}_n{len(y)}_lr{args.lr:g}_sb{args.online_selector_batch_size}_v3"
    if args.independent_router:
        tag += "_independentrouter"
    if args.router_online_lr > 0:
        tag += f"_routeronline{args.router_online_lr:g}"
    if args.query_schedule != "policy":
        tag += "_query"+args.query_schedule
    if session_path is not None:
        tag += "_continual_"+session_path.parent.name
    results_dir = output / "Results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame(trace).to_csv(results_dir / (tag+"_trace.csv"), index=False)
    (results_dir / (tag+"_metrics.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_stream_summary(metrics, name, args.selector_mode, args.lr)



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
    source = Path(args.data_dir) / SOURCE_FILENAME
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
    global ROUTE_HISTORY_ENABLED, ROUTE_AUX_WEIGHT, ROUTING_REWARD_MODE, CONTEXT_REPLAY_SIZE, CONTEXT_CURRENT_WEIGHT
    global RUN_SEED, device, SOURCE_FILENAME, SELECTION_COST, TRAIN_BUDGET_FRACTION, ONLINE_TRAIN_LR
    global GAIN_WEIGHT, RETENTION_WEIGHT, REFERENCE_BATCH_SIZE, REFERENCE_HISTORY_SIZE, TRAIN_PERTURBATIONS
    parser=argparse.ArgumentParser(description="Three-action shared-Context AMCAL; train-only scaling, global label budget, pre-feedback scoring.")
    parser.add_argument("command",choices=("audit","train","evaluate"))
    parser.add_argument("--routing-oracle", action="store_true", help="Observer-only true-label route ceiling; keeps all policy decisions and updates unchanged.")
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--train-perturbations", type=float, nargs="+", default=[0.0], help="Training/validation FGSM domains; never target test files.")
    parser.add_argument("--selection-cost", type=float, default=0.0001)
    parser.add_argument("--gain-weight", type=float, default=1.0)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument("--reference-history-size", type=int, default=256)
    parser.add_argument("--train-online-lr", type=float, default=1e-5)
    parser.add_argument("--source-file", default="IUST_MSCN_original.csv")
    parser.add_argument("--selector-mode", choices=("learned", "random"), default="learned")
    parser.add_argument("--stream-order", choices=("input", "shuffle"), default="shuffle")
    parser.add_argument("--forgetting-diagnostics", action="store_true")
    parser.add_argument("--data-dir",default=str(Path(__file__).resolve().parent/"Dataset"))
    parser.add_argument("--output",default=str(Path(__file__).resolve().parent/"protocol_runs"/"seed42_v3"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--base-epochs",type=int,default=200)
    parser.add_argument("--context-epochs",type=int,default=200)
    parser.add_argument("--pretrain-epochs",type=int,default=20)
    parser.add_argument("--base-patience",type=int,default=50)
    parser.add_argument("--context-patience",type=int,default=20)
    parser.add_argument("--attack-file")
    parser.add_argument("--row-map")
    parser.add_argument("--assume-row-aligned",action="store_true")
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--budget-fraction",type=float,default=0.20)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--online-selector-batch-size",type=int,default=32,
                        help="Online replay minibatch size; offline training remains at 256.")
    parser.add_argument("--context-replay-size",type=int,default=0)
    parser.add_argument("--context-current-weight",type=float,default=0.5)
    parser.add_argument("--routing-reward-mode",choices=("absolute","relative"),default="absolute")
    parser.add_argument("--route-aux-weight",type=float,default=0.0)
    parser.add_argument("--route-history",action="store_true")
    parser.add_argument("--independent-router",action="store_true",help="Uniform-feedback control using Models/independent_router.pth for routes 0/1.")
    parser.add_argument("--continual-state",type=str,help="Local state file to retain Context/DDQN/replay between domain evaluations.")
    parser.add_argument("--router-online-lr",type=float,default=0.0,help="Independent router online Adam LR; zero keeps it frozen.")
    parser.add_argument("--query-schedule",choices=("policy","uniform"),default="policy",help="Diagnostic uniform fixed-budget feedback; keeps learned routing but overrides query selection.")
    args=parser.parse_args()
    if not np.isfinite(args.router_online_lr) or args.router_online_lr < 0:
        parser.error("Router online LR must be finite and nonnegative.")
    if args.router_online_lr > 0 and (not args.independent_router or args.command != "evaluate"):
        parser.error("Online router learning requires evaluate --independent-router.")
    if args.continual_state:
        if args.command != "evaluate":parser.error("--continual-state is evaluation only")
        args.continual_state=str(Path(args.continual_state).resolve())
    ROUTE_HISTORY_ENABLED = args.route_history
    if not np.isfinite(args.route_aux_weight) or args.route_aux_weight < 0:
        parser.error("Auxiliary weight must be finite and nonnegative.")
    ROUTE_AUX_WEIGHT = args.route_aux_weight
    ROUTING_REWARD_MODE = args.routing_reward_mode
    if args.context_replay_size < 0 or not 0 < args.context_current_weight <= 1:
        parser.error("Invalid context replay configuration.")
    CONTEXT_REPLAY_SIZE, CONTEXT_CURRENT_WEIGHT = args.context_replay_size,args.context_current_weight
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
    if not 0 <= args.eval_epsilon <= 1:
        parser.error("Evaluation epsilon must be between zero and one.")
    if args.selection_cost < 0 or not np.isfinite(args.selection_cost):
        parser.error("Selection cost must be finite and nonnegative.")
    if args.train_online_lr <= 0 or not np.isfinite(args.train_online_lr):
        parser.error("Training online LR must be finite and positive.")
    if any(not np.isfinite(v) or v < 0 for v in (args.gain_weight, args.retention_weight)):
        parser.error("Reward weights must be finite and nonnegative.")
    if args.reference_batch_size < 1 or args.reference_history_size < 1:
        parser.error("Reference sizes must be positive.")
    GAIN_WEIGHT, RETENTION_WEIGHT = args.gain_weight, args.retention_weight
    REFERENCE_BATCH_SIZE, REFERENCE_HISTORY_SIZE = args.reference_batch_size, args.reference_history_size
    if any(not np.isfinite(level) or level < 0 for level in args.train_perturbations):
        parser.error("Training perturbations must be finite and nonnegative.")
    TRAIN_PERTURBATIONS = tuple(sorted(set(args.train_perturbations)))
    SELECTION_COST=args.selection_cost
    TRAIN_BUDGET_FRACTION=args.budget_fraction
    ONLINE_TRAIN_LR=args.train_online_lr
    SOURCE_FILENAME=args.source_file
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




