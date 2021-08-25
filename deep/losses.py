from .packages import *

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        super().__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=p)

    def forward(self, x, y):
        d = self.distance(*x)
        losses = (1 - y)*d**2 + y*torch.clamp(self.margin - d, 0)**2
        return 0.5*torch.mean(losses)

    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        super().__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=p)

    def forward(self, x, y=None):
        anchor, pos, neg = x
        d_pos = self.distance(anchor, pos)
        d_neg = self.distance(anchor, neg)
        losses = torch.clamp(d_pos**2 - d_neg**2 + self.margin, 0)
        return torch.mean(losses)
    
    
class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=1.0, p=2):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.distance = nn.PairwiseDistance(p=p)
    
    def forward(self, x, y=None):
        anchor, pos, neg1, neg2 = x
        d_pos = self.distance(anchor, pos)
        d_neg = self.distance(anchor, neg1)
        d_nn = self.distance(neg1, neg2)
        losses = (torch.clamp(d_pos**2 - d_neg**2 + self.margin1, 0)
                 + torch.clamp(d_pos**2 - d_nn**2 + self.margin2, 0))
        return torch.mean(losses)
                  
    
class RegressionEmbeddingLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.distance = nn.PairwiseDistance(p=p)

    def forward(self, x, y):
        d = self.distance(*x)
        losses = torch.clamp(y - d, 0)**2
        return 0.5*torch.mean(losses)
    
    
class SSLoss(nn.Module):
    def __init__(self, embed_loss, pred_loss, embed_loss_weight=0.5):
        super().__init__()
        self.embed_loss = embed_loss
        self.pred_loss = pred_loss
        self.embed_loss_weight = embed_loss_weight
    
    def forward(self, x, y):
        embeds, xs = x
        if len(y) == 2:
            similarities, ys = y
        else:
            similarities = None
            ys = y
        embed_loss = self.embed_loss(embeds, similarities)
        pred_loss = 0
        for xi, yi in zip(xs, ys):
            pred_loss += self.pred_loss(xi, yi)
        pred_loss /= len(xs)
        return self.embed_loss_weight*embed_loss + (1-self.embed_loss_weight)*pred_loss