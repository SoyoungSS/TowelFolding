"""Shared teaching model, imported without starting a training run."""
import torch
from torch import nn
from decoder_demo import TeachingPolicy

class VisualTeachingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_projection=nn.Linear(512,128)
        self.state_projection=nn.Linear(16,128)
        self.camera_embedding=nn.Embedding(3,128)
        self.row_embedding=nn.Embedding(7,128)
        self.column_embedding=nn.Embedding(7,128)
        self.policy=TeachingPolicy()

    def encode_observations(self,features,states):
        r=torch.arange(7).repeat_interleave(7)
        c=torch.arange(7).repeat(7)
        pos=self.row_embedding(r)+self.column_embedding(c)
        visual=self.visual_projection(features)+pos[None,None]+self.camera_embedding(torch.arange(3))[None,:,None]
        tokens=torch.cat((self.state_projection(states)[:,None],visual.flatten(1,2)),dim=1)
        return tokens

    def forward(self,features,states):
        return self.policy(self.encode_observations(features,states))[0]

