import torch as t
from torch import nn

class BasicMoDule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):

        if name is None:
            prefix = 'checkpoints\\' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
