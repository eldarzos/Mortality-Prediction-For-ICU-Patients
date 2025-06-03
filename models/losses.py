"""Module containing all binary classification losses."""

import abc
import torch.nn.functional as F


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        How many steps between each loss record.
    """

    def __init__(self, record_loss_every=1):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every

    @abc.abstractmethod
    def __call__(self, y_pred, y_true, is_train, storer):
        """Calculates loss for a batch of data."""
        pass # Added pass for abstract method

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 0:
            storer = storer
        else:
            storer = None

        return storer


class BCE(BaseLoss):
    def __init__(self):
        """Compute the binary cross entropy loss using logits."""
        super().__init__()

    def __call__(self, y_pred, y_true, is_train, storer):
        """Binary cross entropy loss function expecting logits.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model output logits.
        y_true : torch.Tensor
            Ground truth labels (0 or 1).
        is_train : bool
            Whether model is training.
        storer: collections.defaultdict or None
        """
        storer = self._pre_call(is_train, storer)

        # Use binary_cross_entropy_with_logits which expects raw logits
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        if storer is not None:
            # Ensure storer is actually a dict-like object before appending
            if hasattr(storer, 'setdefault'):
                key = 'train_loss' if is_train else 'valid_loss'
                storer[key].append(loss.item())
            # else: log a warning maybe?

        return loss
