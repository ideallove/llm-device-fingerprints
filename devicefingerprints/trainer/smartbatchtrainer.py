from typing import Any, Dict, Union

import torch
import transformers as tr


class SmartBatchTrainer(tr.Trainer):
    """Trainer with a dynamic batch size.

    This trainer is to be used in conjunction with SmartBatchDataset to speed up
    training by sorting examples in each batch according to their length. Note
    that batch generation and data collation are delegated to SmartBatchDataset,
    and therefore the `args.per_device_train_batch_size` and `data_collator`
    arguments provided to the constructor are ignored.
    """

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        # Overrides the get_train_dataloader method.
        if self.train_dataset is None:
            raise ValueError('self.train_dataset is None')
        if self.args.world_size > 1:
            raise ValueError('Distributed training is not supported')

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=None,
            collate_fn=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory)

    def training_step(
            self, model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # Overrides the training_step method.
        model.train()
        inputs = self._prepare_inputs(inputs)

        if tr.utils.is_sagemaker_mp_enabled():
            raise ValueError('SageMaker is not supported')

        # SmartBatchDataset may return empty minibatches if the associated batch
        # is exhaused before reaching gradient_accumulation_steps.
        if not inputs['input_ids'].numel():
            return torch.tensor(0.0)

        # The total number of active tokens in the batch is used to weight the
        # loss from each minibatch.
        if 'num_active_tokens' in inputs:
            num_active_tokens = inputs.pop('num_active_tokens')
        elif self.args.n_gpu > 1 or self.args.gradient_accumulation_steps > 1:
            # Number of active tokens is not provided, estimate it from the
            # current minibatch.
            num_active_tokens = (self.args.gradient_accumulation_steps *
                                 (inputs['labels'] != -100).sum())

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            chunks = torch.nn.parallel.scatter(
                torch.ones(inputs['input_ids'].shape[0]), model.device_ids)
            start = 0
            for i, chunk in enumerate(chunks):
                end = start + chunk.shape[0]
                loss[i] = loss[i] * (
                    (inputs['labels'][start:end] != -100).sum() /
                    num_active_tokens)
                start = end

            loss = loss.sum()
        elif self.args.gradient_accumulation_steps > 1:
            loss = loss * (inputs['labels'] != -100).sum() / num_active_tokens

        if self.args.gradient_accumulation_steps > 1 and self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in
            # its `backward`.
            loss = loss * self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()
