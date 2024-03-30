import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cosine_sim(anchor, positive)
        neg_sim = self.cosine_sim(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()

class OnlineCosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1, semihard=True):
        super(OnlineCosineTripletLoss, self).__init__()
        self.margin = margin
        self.semihard = semihard

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
        """
        # Check that i, j, and k are distinct
        indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        labels = labels.unsqueeze(0)
        label_equal = labels == labels.transpose(0, 1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = i_equal_j & (~i_equal_k)

        # Combine the two masks
        mask = distinct_indices & valid_labels

        return mask

    def forward(self, embeddings, labels):
        # Normalize each vector (element) to have unit norm
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)  # Compute L2 norms
        embeddings = embeddings / norms  # Divide by norms to normalize
        
        # Compute pairwise cosine similarity
        all_sim = torch.mm(embeddings, embeddings.t())

        mask = self._get_triplet_mask(labels).float()

        anc_pos_sim = all_sim.unsqueeze(2)
        anc_neg_sim = all_sim.unsqueeze(1)

        loss = F.relu((-1*anc_pos_sim) - (-1*anc_neg_sim) + self.margin) * mask

        # calculate average loss (disregarding invalid & easy triplets)
        if self.semihard:
            loss = torch.sum(loss) / (torch.gt(loss, 1e-16).float().sum() + 1e-16)
        else:
            loss = loss.sum() / torch.sum(mask)

        return loss


class OnlineHardCosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(OnlineHardCosineTripletLoss, self).__init__()
        self.margin = margin

    def _get_anc_pos_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have the same label.

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]

        Returns:
            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0)).to(labels.device).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Combine the two masks
        mask = indices_not_equal & labels_equal

        return mask

    def _get_anc_neg_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]

        Returns:
            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
        """
        return labels.unsqueeze(0) != labels.unsqueeze(1)

    def forward(self, embeddings, labels):
        #all_sim = self.cosine_sim(embeddings.unsqueeze(-1), embeddings.unsqueeze(-1).T)
        # Normalize each vector (element) to have unit norm
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)  # Compute L2 norms
        embeddings = embeddings / norms  # Divide by norms to normalize
        
        # Compute pairwise cosine similarity
        all_sim = torch.mm(embeddings, embeddings.t())

        # find hardest positive pairs (when positive has low sim)
        # mask of all valid positives
        mask_anc_pos = self._get_anc_pos_triplet_mask(labels)
        # prevent invalid pos by increasing sim to above 1
        anc_pos_sim = all_sim# + (~mask_anc_pos).float()
        anc_pos_sim[~mask_anc_pos] = 1.
        # select minimum sim positives
        hardest_pos_sim = anc_pos_sim.min(dim=1, keepdim=True)[0]

        # find hardest negative triplets (when negative has high sim)
        # mask of all valid negatives
        mask_anc_neg = self._get_anc_neg_triplet_mask(labels).float()
        # set invalid negatives to 0
        anc_neg_sim = all_sim * mask_anc_neg
        # select maximum sim negatives
        hardest_neg_sim = anc_neg_sim.max(dim=1, keepdim=True)[0]

        loss = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)

        # calculate average loss (disregarding invalid & easy triplets)
        loss = torch.sum(loss) / (torch.gt(loss, 1e-16).float().sum() + 1e-16)

        return loss
