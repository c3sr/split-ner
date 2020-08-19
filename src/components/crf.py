import torch
from torch import nn as nn

from src.utils.general import log_sum_exp


class CRF(nn.Module):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    def __init__(self, out_tags, device, post_padding=True, pad_tag="<PAD>"):
        super(CRF, self).__init__()
        self.tag_to_index = {k: i for i, k in enumerate(out_tags)}
        self.device = device
        self.pad_tag = pad_tag
        self.post_padding = post_padding
        self.tagset_size = len(self.tag_to_index)
        self.transitions = nn.Parameter(data=torch.randn(self.tagset_size, self.tagset_size, device=self.device))

        self.transitions.data[self.tag_to_index[CRF.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_index[CRF.STOP_TAG]] = -10000
        self.transitions.data[:, self.tag_to_index[self.pad_tag]] = -10000
        self.transitions.data[self.tag_to_index[self.pad_tag], :] = -10000

        self.transitions.data[self.tag_to_index[self.pad_tag], self.tag_to_index[CRF.STOP_TAG]] = 0
        self.transitions.data[self.tag_to_index[self.pad_tag], self.tag_to_index[self.pad_tag]] = 0

    def score_sentence(self, feats, masks, tags):
        """
        :param feats: [B, S, T]
        :param tags: [B, S]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        score = torch.zeros(batch_size, dtype=torch.float32, device=self.device)  # [B]
        start_tag_tensor = torch.full(size=(batch_size, 1), fill_value=self.tag_to_index[CRF.START_TAG],
                                      dtype=torch.long, device=self.device)  # [B, 1]
        tags = torch.cat([start_tag_tensor, tags], 1)  # [B, S+1]

        for i in range(seq_len):
            curr_mask = masks[:, i]
            curr_emission = torch.zeros(batch_size, dtype=torch.float32, device=self.device)  # [B]
            curr_transition = torch.zeros(batch_size, dtype=torch.float32, device=self.device)  # [B]
            for j in range(batch_size):
                curr_emission[j] = feats[j, i, tags[j, i + 1]].unsqueeze(0)
                curr_transition[j] = self.transitions[tags[j, i + 1], tags[j, i]].unsqueeze(0)
            score += (curr_emission + curr_transition) * curr_mask
        last_tags = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        score += self.transitions[self.tag_to_index[CRF.STOP_TAG], last_tags]
        return score

    def veterbi_decode(self, feats, masks):
        """
        :param feats: [B, S, T]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        backpointers = torch.LongTensor().to(self.device)
        init_vvars = torch.full(size=(batch_size, self.tagset_size), fill_value=-10000., device=self.device)
        init_vvars[:, self.tag_to_index[CRF.START_TAG]] = 0.

        for s in range(seq_len):
            curr_masks = masks[:, s].unsqueeze(1)
            curr_vvars = init_vvars.unsqueeze(1) + self.transitions  # [B, 1, T] -> [B, T, T]
            curr_vvars, curr_bptrs = curr_vvars.max(2)
            curr_vvars += feats[:, s]
            backpointers = torch.cat((backpointers, curr_bptrs.unsqueeze(1)), 1)
            init_vvars = curr_vvars * curr_masks + init_vvars * (1 - curr_masks)
        init_vvars += self.transitions[self.tag_to_index[CRF.STOP_TAG]]
        best_score, best_tag_id = init_vvars.max(1)

        backpointers = backpointers.tolist()
        best_path = [[p] for p in best_tag_id.tolist()]

        for b in range(batch_size):
            n = masks[b].sum().long().item()
            t = best_tag_id[b]
            for curr_bptrs in reversed(backpointers[b][:n]):
                t = curr_bptrs[t]
                best_path[b].append(t)
            start = best_path[b].pop()
            assert start == self.tag_to_index[CRF.START_TAG]
            best_path[b].reverse()
            if self.post_padding:
                best_path[b] += [self.tag_to_index[self.pad_tag]] * (seq_len - len(best_path[b]))
            else:
                best_path[b] = [self.tag_to_index[self.pad_tag]] * (seq_len - len(best_path[b])) + best_path[b]

        return best_score, best_path

    def forward_algo(self, feats, masks):
        """
        :param feats: [B, S, T]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        score = torch.full(size=(batch_size, self.tagset_size), fill_value=-10000., device=self.device)
        score[:, self.tag_to_index[CRF.START_TAG]] = 0.

        for s in range(seq_len):
            curr_mask = masks[:, s].unsqueeze(-1).expand_as(score)  # [B, T]
            curr_score = score.unsqueeze(1).expand(-1, *self.transitions.size())  # [B, T, T]
            curr_emission = feats[:, s].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transitions.unsqueeze(0).expand_as(curr_score)
            curr_score = log_sum_exp(curr_score + curr_emission + curr_transition)
            score = curr_score * curr_mask + score * (1 - curr_mask)

        score = log_sum_exp(score + self.transitions[self.tag_to_index[CRF.STOP_TAG]])
        return score
