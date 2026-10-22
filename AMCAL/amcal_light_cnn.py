"""Explicitly tagged small CNN; original checkpoint architectures remain unchanged."""
import torch
from torch import nn

ARCHITECTURE = 'light_ps_cnn_v1'


class LightCNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv1d(1,16,5,padding=2), nn.GroupNorm(4,16),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(.15))
        self.conv_block2 = nn.Sequential(nn.Conv1d(16,32,3,padding=1), nn.GroupNorm(4,32),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(.2))
        self.conv_block3 = nn.Sequential(nn.Conv1d(32,64,3,padding=1), nn.GroupNorm(8,64),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.flatten = nn.Flatten()
        # Keep the selector's 128-wide embedding interface.
        self.dense_layers = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Dropout(.35),
                                         nn.Linear(128,num_classes))

    def forward(self, x):
        x = self.flatten(self.conv_block3(self.conv_block2(self.conv_block1(x))))
        return self.dense_layers(x)


def classifier_for_checkpoint(saved, width, classes, original_factory):
    architecture = saved.get('architecture', 'legacy')
    if architecture == ARCHITECTURE: return LightCNN1D(width, classes)
    if architecture == 'legacy': return original_factory(width, classes)
    raise ValueError('Unknown frozen CNN architecture: '+str(architecture))
