import torch
from torch import nn

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        # Move p and q to CPU and ensure they are in float64 for high precision calculation
        p, q = p.cpu().double(), q.cpu().double()
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))