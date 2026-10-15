"""Diagnostic reproduction of Burst AMCAL.ipynb cells 2, 7 and 9.
Preserves legacy numerical logic, including post-update scoring and random online actions.
Not the corrected evaluation protocol.
"""
from amcal_burst_stream import stream_permutation, permutation_sha256, ClassInterleavedSampler, split_class_budget, fractional_class_budget, class_budget_available, class_budget_report
from amcal_burst_holdout import split_holdout_indices, evaluate_heldout
from amcal_burst_context_memory import make_context_memory, with_context_rehearsal, context_update_probe, final_context_memory_report
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
BURST_DATA_DIR = Path(__file__).resolve().parent / "Dataset"
LEGACY_PER_LEVELS = None
LEGACY_DIAGNOSTIC_SEED = None
LEGACY_SELECTOR_MODE = 'original'
LEGACY_QUERY_GATE = 'none'
SELECTOR_REWARD_MODE = 'original'
SELECTOR_TRANSITION_MODE = 'terminal'
SELECTOR_GAMMA = 0.2
CONTEXT_LOSS_MODE = 'legacy'
CONTEXT_ARCHITECTURE = 'feature_tokens_v1'
SELECTOR_STATE_VERSION = 'context_probability_vector_v1'
TRAINING_STATE_VERSION = 3
RUN_TRAINING_STATE_VERSION = TRAINING_STATE_VERSION
ONLINE_EPSILON_START = 0.0
ONLINE_EPSILON_END = 0.0
ONLINE_EPSILON_DECAY = 1.0
TRAINING_SELECTOR_MODE = 'original'
AGREEMENT_PENALTY = 0.2
TEST_LABEL_BUDGET = 214
SELECTION_COST = 0.0
STREAM_ORDER = 'class-interleaved'
STREAM_SEED = 42
BUDGET_MODE = 'total'
CLASS_BUDGET_FRACTION = None
QUOTA_POLICY = 'optional'
CONTEXT_MEMORY_MODE = 'off'
MEMORY_PER_CLASS = 32
MEMORY_BATCH_SIZE = 16
MEMORY_WEIGHT = 1.0
HOLDOUT_FRACTION = 0.0
ONLINE_CONTEXT_LR = 0.25
TRAIN_EPSILON_START = 1.0
TRAIN_EPSILON_END = 0.01
TRAIN_EPSILON_DECAY = 0.995


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
        self.architecture = CONTEXT_ARCHITECTURE
        if self.architecture == "residual_affine_v1":
            token_dim = 64
            self.feature_projection = nn.Linear(1, token_dim)
            self.feature_identity = nn.Parameter(torch.zeros(1, input_size, token_dim))
            nn.init.normal_(self.feature_identity, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=8, dim_feedforward=dim_feedforward,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
            # Normalize hidden tokens only, never the adapted feature vector.
            self.modulation_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 2))
            nn.init.zeros_(self.modulation_head[-1].weight)
            nn.init.zeros_(self.modulation_head[-1].bias)
            print("Context residual affine: identity initialization, feature tokens, no BatchNorm/dropout")
            return
        if self.architecture == "feature_tokens_v1":
            token_dim = 64
            self.feature_projection = nn.Linear(1, token_dim)
            self.feature_identity = nn.Parameter(torch.zeros(1, input_size, token_dim))
            nn.init.normal_(self.feature_identity, std=0.02)
            self.weight_projection = nn.Linear(token_dim, 1)
            num_heads = 8
        else:
            token_dim = input_size
            num_heads = next(h for h in [8, 4, 2, 1] if input_size % h == 0)
        print(f"Selected num_heads={num_heads} for input_size={input_size}")
        
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
        if self.architecture == "residual_affine_v1":
            tokens = self.feature_projection(x_flat.unsqueeze(-1)) + self.feature_identity
            modulation = self.modulation_head(self.transformer(tokens))
            delta_scale = torch.tanh(modulation[..., 0])
            delta_shift = torch.tanh(modulation[..., 1])
            return (x_flat * (1.0 + delta_scale) + delta_shift).unsqueeze(1)
        if self.architecture == "feature_tokens_v1":
            tokens = self.feature_projection(x_flat.unsqueeze(-1)) + self.feature_identity
            transformed = self.transformer(tokens)  # [batch, features, token_dim]
            weights = torch.sigmoid(self.weight_projection(transformed).squeeze(-1))
        else:
            x_transformed = self.transformer(x_flat.unsqueeze(1)).squeeze(1)
            weights = torch.sigmoid(x_transformed)
        combined = torch.stack([x_flat, weights], dim=1)
        encoded = self.encoder(combined)
        return encoded.unsqueeze(1)

def selector_state(inputs, cnn_model, context_model, after_update=False):
    """Raw x plus 1 - softmax(CNN(Context(x))); labels are never required."""
    with evaluation_mode(cnn_model, context_model), torch.no_grad():
        if SELECTOR_STATE_VERSION == "context_probability_vector_v1":
            probabilities = torch.softmax(cnn_model(context_model(inputs)), dim=1)
            uncertainty = 1.0 - probabilities
        elif SELECTOR_STATE_VERSION == "legacy_scalar_v2":
            source = context_model(inputs) if after_update else inputs
            probabilities = torch.softmax(cnn_model(source), dim=1)
            uncertainty = 1.0 - probabilities.max(dim=1, keepdim=True).values
        else:
            raise ValueError(f"Unsupported selector state: {SELECTOR_STATE_VERSION}")
        return torch.cat([inputs.squeeze(1), uncertainty], dim=1).detach()


def selector_input_dimension(input_length, num_classes):
    return input_length + (num_classes if SELECTOR_STATE_VERSION ==
                           "context_probability_vector_v1" else 1)


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
        if STREAM_ORDER == "class-interleaved":
            # Training order is generated by the FIFO sampler each epoch.
            val_order = stream_permutation(y_val, seed=STREAM_SEED)
            test_order = stream_permutation(y_test, seed=STREAM_SEED)
            X_val, y_val = X_val[val_order], y_val[val_order]
            X_test, y_test = X_test[test_order], y_test[test_order]

        
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
    
    if STREAM_ORDER == "class-interleaved":
        # CSVs are already oversampled; do not resample or duplicate them again.
        train_sampler = ClassInterleavedSampler(y_train, STREAM_SEED)
        full_sampler = ClassInterleavedSampler(y_train, STREAM_SEED)
    else:
        full_sampler = train_sampler
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    full_loader = DataLoader(train_dataset, batch_size=128, sampler=full_sampler)
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
    context_memory = make_context_memory(CONTEXT_MEMORY_MODE, MEMORY_PER_CLASS, LEGACY_DIAGNOSTIC_SEED if LEGACY_DIAGNOSTIC_SEED is not None else 42)
    
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
            state = selector_state(inputs, cnn_model, context_model)
            
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
                total_context_loss, memory_info = with_context_rehearsal(
                    total_context_loss, context_memory, context_model, cnn_model,
                    MEMORY_BATCH_SIZE, MEMORY_WEIGHT)
                total_context_loss.backward()
                context_optimizer.step()
                if context_memory is not None:
                    # Store original selected rows, not the singleton duplicate for BN.
                    context_memory.add_batch(inputs[action_indices], labels[action_indices])
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
                    next_state_new = selector_state(inputs, cnn_model, context_model, after_update=True)
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
        if context_memory is not None:
            print("  Context memory:", context_memory.stats())
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
    dqn_agent = DQNSelectorAgent(
        selector_input_dimension(input_length, num_classes), device,
        epsilon_start=TRAIN_EPSILON_START,
        epsilon_end=TRAIN_EPSILON_END,
        epsilon_decay=TRAIN_EPSILON_DECAY,
    )
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
    import json
    if HOLDOUT_FRACTION > 0:
        # Reserved target rows cannot influence preprocessing or class weights.
        training_reference = pd.read_csv(BURST_DATA_DIR / "oversampled_train_dataset.csv")
        scaler = StandardScaler().fit(training_reference.iloc[:, :-1].to_numpy())
        class_count = np.bincount(label_encoder.transform(training_reference.iloc[:, -1].to_numpy()),
                                 minlength=len(label_encoder.classes_))
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
        adaptation_indices, heldout_indices = split_holdout_indices(y_test, HOLDOUT_FRACTION)
        heldout_X, heldout_y = X_test[heldout_indices].copy(), y_test[heldout_indices].copy()
        if HOLDOUT_FRACTION > 0 and set(heldout_y) != set(label_encoder.classes_):
            raise ValueError("Heldout set must contain all checkpoint classes.")
        X_test, y_test = X_test[adaptation_indices], y_test[adaptation_indices]
        max_samples = len(adaptation_indices)
        # Interleave only the adaptation part; reserved rows never enter learners.
        source_indices = stream_permutation(y_test, STREAM_ORDER, STREAM_SEED)
        X_test = X_test[source_indices]
        y_test = y_test[source_indices]
        source_indices = [adaptation_indices[i] for i in source_indices]
        print(f"Adaptation rows: {max_samples}; heldout rows: {len(heldout_indices)}")
        print(f"Stream order: {STREAM_ORDER}; seed={STREAM_SEED}; "
              f"permutation={permutation_sha256(source_indices)}")
        cumulative_accuracy_df = pd.DataFrame({'Sample Index': range(1, max_samples + 1)})
    except FileNotFoundError:
        print(f"Error: File {test_file} not found.")
        return [], cumulative_accuracy_df
    
    try:
        y_test = label_encoder.transform(y_test)
    except ValueError:
        print(f"Error: Unknown labels in {test_file}. Skipping.")
        return [], cumulative_accuracy_df
    
    # Offline experiment block IDs; not exposed to RL or prediction features.
    stream_class_ids = y_test.copy()
    class_quotas = split_class_budget(budget, len(label_encoder.classes_))
    if CLASS_BUDGET_FRACTION is not None:
        class_quotas = fractional_class_budget(stream_class_ids, len(label_encoder.classes_), CLASS_BUDGET_FRACTION)
        budget = sum(class_quotas.values())
        print("Fractional per-class label caps:", class_quotas, "; total:", budget)
    class_labels_used = {c: 0 for c in class_quotas}
    class_remaining = np.bincount(stream_class_ids, minlength=len(class_quotas))
    class_targets = {c: min(class_quotas[c], int(class_remaining[c])) for c in class_quotas}
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
    print(f"Online Context Adam learning rate: {fine_tune_lr:g}")
    context_optimizer = optim.Adam(context_model.parameters(), lr=fine_tune_lr, weight_decay=1e-4)
    context_memory = make_context_memory(CONTEXT_MEMORY_MODE, MEMORY_PER_CLASS, LEGACY_DIAGNOSTIC_SEED if LEGACY_DIAGNOSTIC_SEED is not None else 42)
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
        block_id = int(stream_class_ids[sample_idx-1])
        class_used_before = class_labels_used[block_id]
        observed_reward, observed_loss = None, None
        memory_info = dict(memory_loss=0.0, replay_samples=0)
        update_probe = None
        memory_size_before = len(context_memory) if context_memory is not None else 0
        # Compute CNN outputs and state
        with torch.no_grad():
            cnn_outputs = cnn_model(inputs)
            probs = softmax(cnn_outputs)
            _, cnn_pred = torch.max(probs, 1)
            max_prob, _ = torch.max(probs, dim=1, keepdim=True)
            uncertainty = 1 - max_prob
            features = inputs.squeeze(1)
            state = selector_state(inputs, cnn_model, context_model)
        
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
        requested_action = action
        forced_query = (QUOTA_POLICY == "complete" and
                        class_labels_used[block_id] < class_targets[block_id] and
                        int(class_remaining[block_id]) <= class_targets[block_id] - class_labels_used[block_id] and
                        action == 0)
        if forced_query:
            action = 1
            action_source = "quota-completion"
        class_remaining[block_id] -= 1
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
            if disagreement_passes and (threshold is None or max_prob_diff >= threshold) and updates_used < budget and class_budget_available(BUDGET_MODE, block_id, class_labels_used, class_quotas):
                
                updates_used += 1
                class_labels_used[block_id] += 1
                probe_before = context_update_probe(context_model, cnn_model, inputs, labels, context_memory)
                context_optimizer.zero_grad()
                encoded_inputs = context_model(inputs)
                outputs = cnn_model(encoded_inputs)
                context_loss = cnn_criterion(outputs, labels)
                orig_cnn_loss = cnn_criterion(cnn_outputs.detach(), labels)
                total_context_loss = context_objective(context_loss, orig_cnn_loss, online=True)
                total_context_loss, memory_info = with_context_rehearsal(
                    total_context_loss, context_memory, context_model, cnn_model,
                    MEMORY_BATCH_SIZE, MEMORY_WEIGHT)
                total_context_loss.backward()
                context_optimizer.step()
                probe_after = context_update_probe(context_model, cnn_model, inputs, labels, context_memory)
                update_probe = dict(before=probe_before, after=probe_after)
                
                if context_memory is not None:
                    context_memory.add_batch(inputs, labels, [source_indices[sample_idx-1]+1])
                # Compute reward for action=1
                reward = compute_reward(action, cnn_pred.item(), weighted_pred.item(), labels.item())
                rewards.append(reward)
                observed_reward = float(reward)
                observed_loss = float(total_context_loss.item())
                
                # Compute next state with updated context
                with torch.no_grad():
                    next_state = selector_state(inputs, cnn_model, context_model, after_update=True)
                
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
            diagnostic_trace.append(dict(update_probe=json.dumps(update_probe), step=sample_idx, source_row=source_indices[sample_idx-1]+1,
                true_label=int(labels.item()),
                base_prediction=int(cnn_pred.item()),
                base_prediction_train_scaler=int(train_scaled_prediction),
                prediction_before_update=int(weighted_pred_before.item()),
                prediction_after_update=int(predicted.item()),
                action=int(action), requested_action=int(requested_action), forced_query=bool(forced_query),
                greedy_action=int(diagnostic_q.argmax(1).item()),
                q_skip=float(diagnostic_q[0,0].item()), q_apply=float(diagnostic_q[0,1].item()),
                epsilon=float(dqn_agent.epsilon),
                action_epsilon=action_epsilon, action_source=action_source,
                labels_before=updates_before,
                class_budget=class_quotas[block_id] if BUDGET_MODE=='per-class' else None,
                class_labels_before=class_used_before, class_labels_used=class_labels_used[block_id],
                confidence_gap=float(abs(weighted_probs_before.max().item()-max_prob.item())),
                updated=updates_used>updates_before, labels_used=updates_used,
                context_loss=observed_loss, reward=observed_reward,
                memory_size_before=memory_size_before,
                memory_size_after=len(context_memory) if context_memory is not None else 0,
                **memory_info))
            cumulative_accuracy_df.loc[cumulative_accuracy_df['Sample Index'] == sample_idx, f'Perturbation {perturb_level}'] = current_accuracy
    
    test_accuracy = correct / total
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    from sklearn.metrics import accuracy_score, classification_report
    os.makedirs(LEGACY_RESULTS_DIR, exist_ok=True)
    import json
    if BUDGET_MODE == "per-class" and any(class_labels_used[c] > class_quotas[c] for c in class_quotas):
        raise RuntimeError("Per-class label budget exceeded.")
    if QUOTA_POLICY == "complete" and class_labels_used != class_targets:
        raise RuntimeError("Complete per-class quota was not consumed.")
    if context_memory is not None:
        if context_memory.total_added != updates_used:
            raise RuntimeError("Context memory contains unbudgeted examples.")
        memory_rows = {item[2] for bucket in context_memory.buckets.values() for item in bucket}
        if memory_rows.intersection(i+1 for i in heldout_indices):
            raise RuntimeError("Heldout row entered Context memory.")
    class_summary, class_curve = class_budget_report(diagnostic_trace, class_quotas)
    heldout_result, heldout_predictions = None, []
    if heldout_indices:
        heldout_result, heldout_predictions = evaluate_heldout(
            cnn_model, context_model, scaler.transform(heldout_X),
            label_encoder.transform(heldout_y), device, heldout_indices, len(label_encoder.classes_))
        print(f"Final unseen holdout accuracy: {heldout_result['accuracy']:.4f}")
        print("Holdout per-class recall:", {c: heldout_result['classes'][str(c)]['recall'] for c in class_quotas})
    if not 0 <= updates_used <= budget:
        raise RuntimeError("Online label budget exceeded.")
    final_memory_report = final_context_memory_report(context_model, cnn_model, context_memory)
    diagnostics = dict(final_context_memory=final_memory_report, per=perturb_level, seed=LEGACY_DIAGNOSTIC_SEED,
        stream_order=STREAM_ORDER, stream_seed=STREAM_SEED,
        within_class_order_preserved=True,
        permutation_sha256=permutation_sha256(source_indices),
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
        budget_mode=BUDGET_MODE, class_budget_fraction=CLASS_BUDGET_FRACTION, quota_policy=QUOTA_POLICY,
        context_memory_mode=CONTEXT_MEMORY_MODE, memory_per_class=MEMORY_PER_CLASS,
        memory_batch_size=MEMORY_BATCH_SIZE, memory_weight=MEMORY_WEIGHT,
        per_class_results=class_summary,
        context_memory_stats=context_memory.stats() if context_memory is not None else None,
        forced_queries=sum(row['forced_query'] and row['updated'] for row in diagnostic_trace),
        holdout_fraction=HOLDOUT_FRACTION, final_holdout=heldout_result,
        adaptation_source_rows=[i+1 for i in source_indices], heldout_source_rows=[i+1 for i in heldout_indices],
        online_context_lr=float(fine_tune_lr),
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
        context_architecture=CONTEXT_ARCHITECTURE,
        selector_state_version=SELECTOR_STATE_VERSION,
        bootstrap_optimizer_steps=dqn_agent.bootstrap_optimizer_steps,
        replay_terminal_rows=sum(float(e[4]) == 1.0 for e in dqn_agent.replay_buffer.buffer),
        replay_nonterminal_rows=sum(float(e[4]) == 0.0 for e in dqn_agent.replay_buffer.buffer),
        learned_policy_used_for_actions=LEGACY_SELECTOR_MODE=="learned",
        classes_before=classification_report(y_true,before_predictions,output_dict=True,zero_division=0),
        classes_after=classification_report(y_true,y_pred,output_dict=True,zero_division=0))
    if BUDGET_MODE == "per-class":
        pd.DataFrame(class_summary).to_csv(f"{LEGACY_RESULTS_DIR}/legacy_PER_{perturb_level}_class_results.csv", index=False)
        pd.DataFrame(class_curve).to_csv(f"{LEGACY_RESULTS_DIR}/legacy_PER_{perturb_level}_class_curve.csv", index=False)
        print("Per-class pre-update accuracy and label usage:")
        print(pd.DataFrame(class_summary).to_string(index=False))
    if heldout_result is not None:
        pd.DataFrame(heldout_predictions).to_csv(f"{LEGACY_RESULTS_DIR}/legacy_PER_{perturb_level}_heldout_predictions.csv", index=False)
        Path(f"{LEGACY_RESULTS_DIR}/legacy_PER_{perturb_level}_heldout_results.json").write_text(json.dumps(heldout_result, indent=2), encoding="utf-8")
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
    if QUOTA_POLICY == "complete":
        print("  Forced quota-completion queries:", diagnostics["forced_queries"])
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
    
    if heldout_result is not None:
        results[-1]['Heldout Accuracy'] = heldout_result['accuracy']
        results[-1]['Heldout Macro F1'] = heldout_result['macro_f1']
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
    fine_tune_lr = ONLINE_CONTEXT_LR
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
                input_size=selector_input_dimension(input_length, num_classes),
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
    global TRAIN_EPSILON_START, TRAIN_EPSILON_END, TRAIN_EPSILON_DECAY
    global CONTEXT_ARCHITECTURE, SELECTOR_STATE_VERSION
    global CONTEXT_MEMORY_MODE, MEMORY_PER_CLASS, MEMORY_BATCH_SIZE, MEMORY_WEIGHT
    global STREAM_ORDER, STREAM_SEED, ONLINE_CONTEXT_LR, BUDGET_MODE, HOLDOUT_FRACTION, QUOTA_POLICY, CLASS_BUDGET_FRACTION
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
    parser.add_argument("--query-gate", choices=("none","disagreement","no-threshold","legacy"), default="none",
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
    parser.add_argument("--train-epsilon-start", type=float, default=1.0,
                        help="Offline RL random-action probability at initialization (default 1).")
    parser.add_argument("--train-epsilon-end", type=float, default=0.01,
                        help="Offline RL minimum random-action probability (default 0.01).")
    parser.add_argument("--train-epsilon-decay", type=float, default=0.995,
                        help="Multiply offline epsilon once per adversarial epoch (default 0.995); 1 keeps it fixed.")
    parser.add_argument("--stream-order", choices=("input", "class-interleaved"),
                        default="class-interleaved",
                        help="Interleave class slots in oversampled training splits and attack evaluation; preserve within-class order.")
    parser.add_argument("--stream-seed", type=int, default=42,
                        help="Independent reproducible stream order seed; default 42.")
    parser.add_argument("--online-context-lr", type=float, default=0.25,
                        help="Evaluation-only Adam learning rate for Context updates; default 0.25 preserves previous runs.")
    parser.add_argument("--context-architecture", choices=("feature_tokens_v1", "residual_affine_v1"),
                        default=None,
                        help="Training Context architecture; default feature_tokens_v1. Evaluation auto-detects checkpoint architecture.")
    parser.add_argument("--budget-mode", choices=("total", "per-class"), default="total",
                        help="Per-class divides the total cap over offline class blocks, no model resets or quota transfer.")
    parser.add_argument("--class-budget-fraction", type=float, default=None,
                        help="Per-class cap floor(fraction * class samples); replaces --budget.")
    parser.add_argument("--holdout-fraction", type=float, default=0.0,
                        help="Evaluation only: reserve the final fraction of each class for final frozen inference.")
    parser.add_argument("--quota-policy", choices=("optional", "complete"), default="optional",
                        help="Complete: fill each class quota with forced tail queries; requires per-class budget and no gate.")
    parser.add_argument("--context-memory", choices=("off", "per-class"), default="off",
                        help="Context-only rehearsal using previously labeled samples; DQN replay is separate.")
    parser.add_argument("--memory-per-class", type=int, default=32)
    parser.add_argument("--memory-batch-size", type=int, default=16)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--context-guard", choices=("off", "per-class"), default="off",
                        help="Retention guard for the random online Context diagnostic.")
    parser.add_argument("--guard-tolerance", type=float, default=0.02)
    parser.add_argument("--guard-retries", type=int, default=4)
    parser.add_argument("--guard-lr-factor", type=float, default=0.5)
    args=parser.parse_args()
    if args.memory_per_class < 1 or args.memory_batch_size < 1:
        parser.error("memory capacities and batch size must be positive.")
    if not np.isfinite(args.memory_weight) or args.memory_weight < 0:
        parser.error("memory-weight must be finite and nonnegative.")
    if args.context_guard != "off" and (args.command != "evaluate" or args.selector_mode != "random"):
        parser.error("This first retention-guard experiment requires evaluate --selector-mode random.")
    CONTEXT_MEMORY_MODE = args.context_memory
    MEMORY_PER_CLASS, MEMORY_BATCH_SIZE, MEMORY_WEIGHT = args.memory_per_class, args.memory_batch_size, args.memory_weight
    QUOTA_POLICY = args.quota_policy
    if QUOTA_POLICY == "complete" and (args.command != "evaluate" or args.budget_mode != "per-class" or args.query_gate != "none"):
        parser.error("quota-policy complete requires evaluate --budget-mode per-class --query-gate none.")
    if not np.isfinite(args.holdout_fraction) or not 0 <= args.holdout_fraction < 1:
        parser.error("holdout-fraction must be finite and in [0,1).")
    if args.command != "evaluate" and args.holdout_fraction != 0:
        parser.error("holdout-fraction is evaluation only.")
    HOLDOUT_FRACTION = args.holdout_fraction
    CLASS_BUDGET_FRACTION = args.class_budget_fraction
    if CLASS_BUDGET_FRACTION is not None:
        if args.command != "evaluate" or args.budget_mode != "per-class" or not np.isfinite(CLASS_BUDGET_FRACTION) or not 0 <= CLASS_BUDGET_FRACTION <= 1:
            parser.error("class-budget-fraction requires evaluate, per-class budget, and a finite fraction in [0,1].")
    BUDGET_MODE = args.budget_mode
    if BUDGET_MODE == "per-class":
        if args.command != "evaluate":
            parser.error("per-class budget is evaluation only.")
        args.stream_order = "input"
        print("Per-class budget: original class blocks, one checkpoint per perturbation level.")
    if args.command == "train":
        CONTEXT_ARCHITECTURE = args.context_architecture or "feature_tokens_v1"
    if not np.isfinite(args.online_context_lr) or args.online_context_lr <= 0:
        parser.error("online-context-lr must be finite and positive.")
    if args.command != "evaluate" and args.online_context_lr != 0.25:
        parser.error("online-context-lr applies only to evaluate; training Context LR is unchanged.")
    ONLINE_CONTEXT_LR = args.online_context_lr
    STREAM_ORDER, STREAM_SEED = args.stream_order, args.stream_seed
    if not 0.0 <= args.train_epsilon_end <= args.train_epsilon_start <= 1.0:
        parser.error("train epsilon requires 0 <= end <= start <= 1.")
    if not 0.0 < args.train_epsilon_decay <= 1.0:
        parser.error("train-epsilon-decay must be in (0, 1].")
    training_schedule = (args.train_epsilon_start, args.train_epsilon_end, args.train_epsilon_decay)
    if args.command != "train" and training_schedule != (1.0, 0.01, 0.995):
        parser.error("train-epsilon settings apply only to train; use --epsilon-* for online evaluation.")
    TRAIN_EPSILON_START, TRAIN_EPSILON_END, TRAIN_EPSILON_DECAY = training_schedule
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
            stream_order=args.stream_order, stream_seed=args.stream_seed,
            seed=args.seed if args.seed is not None else 42,
            budget=args.budget, budget_mode=args.budget_mode, quota_policy=args.quota_policy,
            class_budget_fraction=args.class_budget_fraction,
            context_guard=args.context_guard, guard_tolerance=args.guard_tolerance,
            guard_retries=args.guard_retries, guard_lr_factor=args.guard_lr_factor,
            context_memory=args.context_memory, memory_per_class=args.memory_per_class,
            memory_batch_size=args.memory_batch_size, memory_weight=args.memory_weight, holdout_fraction=args.holdout_fraction, max_samples=1073, lr=ONLINE_CONTEXT_LR,
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
        if training_schedule != (1.0, 0.01, 0.995):
            folder += f"_train_eps{TRAIN_EPSILON_START:g}to{TRAIN_EPSILON_END:g}_decay{TRAIN_EPSILON_DECAY:g}"
        if args.command == "train" and STREAM_ORDER != "input":
            folder += f"_class_interleaved_seed{STREAM_SEED}"
        if CONTEXT_MEMORY_MODE != "off":
            folder += f"_context_memory{MEMORY_PER_CLASS}_batch{MEMORY_BATCH_SIZE}_weight{MEMORY_WEIGHT:g}"
        folder += f"_{CONTEXT_ARCHITECTURE}_context_probabilities_validated_best"
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
    if args.command == "evaluate" and STREAM_ORDER != "input":
        gate_suffix += f"_class_interleaved_seed{STREAM_SEED}"
    if args.command == "evaluate" and ONLINE_CONTEXT_LR != 0.25:
        gate_suffix += f"_context_lr{ONLINE_CONTEXT_LR:g}"
    if BUDGET_MODE == "per-class":
        gate_suffix += "_budget_per_class"
        if CLASS_BUDGET_FRACTION is not None:
            gate_suffix += f"_fraction{CLASS_BUDGET_FRACTION:g}"
    if HOLDOUT_FRACTION > 0:
        gate_suffix += f"_holdout{HOLDOUT_FRACTION:g}"
    if QUOTA_POLICY == "complete":
        gate_suffix += "_quota_complete"
    if CONTEXT_MEMORY_MODE != "off":
        gate_suffix += f"_context_memory{MEMORY_PER_CLASS}_batch{MEMORY_BATCH_SIZE}_weight{MEMORY_WEIGHT:g}"
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
        saved_architectures, saved_states = set(), set()
        for checkpoint_name in ("initial_cnn_model.pth","adversarial_best_context.pth","adversarial_best_dqn.pth"):
            checkpoint=torch.load(output/"Models"/checkpoint_name,map_location="cpu",weights_only=False)
            saved_versions.add(checkpoint.get("training_state_version", 1))
            saved_architectures.add(checkpoint.get("context_architecture", "legacy_single_token"))
            saved_states.add(checkpoint.get("selector_state_version", "legacy_scalar_v2"))
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
        if len(saved_architectures) != 1 or len(saved_states) != 1:
            parser.error("Mixed Context architectures or selector states in Models folder.")
        CONTEXT_ARCHITECTURE = saved_architectures.pop()
        SELECTOR_STATE_VERSION = saved_states.pop()
        if CONTEXT_ARCHITECTURE not in ("legacy_single_token", "feature_tokens_v1", "residual_affine_v1"):
            parser.error("Unsupported Context architecture.")
        if args.context_architecture is not None and args.context_architecture != CONTEXT_ARCHITECTURE:
            parser.error("Requested Context architecture does not match the saved checkpoint.")
        if SELECTOR_STATE_VERSION not in ("legacy_scalar_v2", "context_probability_vector_v1"):
            parser.error("Unsupported selector state version.")
        RUN_TRAINING_STATE_VERSION = saved_versions.pop()
        if RUN_TRAINING_STATE_VERSION < TRAINING_STATE_VERSION:
            print("Using legacy Context/scalar-state checkpoints; feature-token corrections require fresh training.")
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
        budget_mode=BUDGET_MODE, quota_policy=QUOTA_POLICY,
        context_memory_mode=CONTEXT_MEMORY_MODE, memory_per_class=MEMORY_PER_CLASS,
        memory_batch_size=MEMORY_BATCH_SIZE, memory_weight=MEMORY_WEIGHT,
        selection_cost=args.selection_cost,
        online_context_lr=ONLINE_CONTEXT_LR if args.command=="evaluate" else None,
        training_state_version=RUN_TRAINING_STATE_VERSION,
        context_architecture=CONTEXT_ARCHITECTURE,
        selector_state_version=SELECTOR_STATE_VERSION,
        next_state_scope="same input after Context update; not next traffic row",
        training_epsilon_schedule=dict(start=TRAIN_EPSILON_START, end=TRAIN_EPSILON_END,
                                       decay=TRAIN_EPSILON_DECAY, rule="once per adversarial epoch") if args.command=="train" else None,
        online_epsilon=args.epsilon_start if args.selector_mode=="learned" else 1.0,
        online_epsilon_end=args.epsilon_end if args.selector_mode=="learned" else 1.0,
        online_epsilon_decay=args.epsilon_decay if args.selector_mode=="learned" else 1.0,
        epsilon_decay_rule="after successful online selector optimization",
        torch_version=str(torch.__version__),
        stream_order=STREAM_ORDER,
        stream_seed=STREAM_SEED,
        data_dir=str(BURST_DATA_DIR)),indent=2),encoding="utf-8")
    previous=Path.cwd()
    try:
        os.chdir(output)
        if args.command=="train":
            print("Selector reward / transitions / gamma:", SELECTOR_REWARD_MODE, SELECTOR_TRANSITION_MODE, SELECTOR_GAMMA)
            print("Context loss mode:", CONTEXT_LOSS_MODE)
            print("Oversampled split ordering:", STREAM_ORDER, "seed:", STREAM_SEED)
            print("Context architecture / selector state:", CONTEXT_ARCHITECTURE, SELECTOR_STATE_VERSION)
            print("Training selector mode / agreement penalty:", TRAINING_SELECTOR_MODE, AGREEMENT_PENALTY)
            print("Action-1 reward cost:", SELECTION_COST)
            print(f"Offline RL epsilon: {TRAIN_EPSILON_START:g} -> {TRAIN_EPSILON_END:g}; "
                  f"decay={TRAIN_EPSILON_DECAY:g} per adversarial epoch")
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
                checkpoint["context_architecture"]=CONTEXT_ARCHITECTURE
                checkpoint["selector_state_version"]=SELECTOR_STATE_VERSION
                checkpoint["training_stream_order"]=STREAM_ORDER
                checkpoint["training_stream_seed"]=STREAM_SEED
                checkpoint["training_context_memory"] = dict(mode=CONTEXT_MEMORY_MODE,
                    per_class=MEMORY_PER_CLASS, batch_size=MEMORY_BATCH_SIZE, weight=MEMORY_WEIGHT)
                checkpoint["selection_cost"]=SELECTION_COST
                checkpoint["training_epsilon_schedule"] = dict(
                    start=TRAIN_EPSILON_START, end=TRAIN_EPSILON_END,
                    decay=TRAIN_EPSILON_DECAY, rule="once per adversarial epoch")
                checkpoint["training_selector_mode"]=TRAINING_SELECTOR_MODE
                checkpoint["agreement_penalty"]=AGREEMENT_PENALTY if TRAINING_SELECTOR_MODE=="disagreement" else None
                torch.save(checkpoint,checkpoint_path)
        else:
            evaluate_main()
    finally:
        os.chdir(previous)

if __name__=="__main__":
    cli()

