from typing import Dict, Tuple, Union

import torch
import transformers as tr


class EmbeddingTrainer(tr.Trainer):
    """Trainer for an embedding model.

    This trainer is to be used in conjunction with LcsExtractionDataset in order
    to incentivize generating similar embeddings for matching examples. An
    example's embedding is computed by taking the weighted average of its token
    embeddings, with the outputted model probabilities used as weights. Note
    that the embeddings are normalized to unit norm. We then add a penalty to
    the loss function to penalize dissimilarity between matching examples, while
    rewarding it for non-matching ones. Batch size must always be an even number.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Overrides the compute_loss method.
        outputs = model(**inputs, output_hidden_states=True)

        # Compute embeddings.
        weights = torch.nn.functional.softmax(outputs['logits'], dim=-1)[:, :,
                                                                         1]
        weights = weights * inputs['attention_mask']
        embeddings = outputs['hidden_states'][-1]
        embeddings = (weights.unsqueeze(-1) * embeddings).sum(axis=1)
        embeddings = embeddings / weights.sum(axis=1, keepdim=True)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Split embeddings into a/b subsets.
        embeddings_a = embeddings[::2]
        embeddings_b = embeddings[1::2]
        embeddings_b_shifted = torch.vstack([embeddings_b[1:], embeddings_b[0]])

        # Compute loss.
        loss_bce = outputs.loss
        loss_same = (embeddings_a - embeddings_b).norm(dim=1).mean()
        loss_shifted = (embeddings_a - embeddings_b_shifted).norm(dim=1).mean()
        loss = loss_bce + loss_same - loss_shifted
        if return_outputs:
            return loss, {'loss': loss, 'logits': outputs.logits}

        return loss
