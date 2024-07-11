import torch
import numpy as np

def custom_pair_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training.

    Creates batches of pairs of genuine and spoofing speech for differential-based detection.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """
    
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths_gt = torch.tensor([item[1].size(1) for item in batch])
    lengths_test = torch.tensor([item[2].size(1) for item in batch])

    # Find the maximum length
    max_length_gt = int(torch.max(lengths_gt))
    max_length_test = int(torch.max(lengths_test))

    # Pad the tensors to have the maximum length
    file_names = []
    padded_gts = torch.zeros(batch_size, max_length_gt)
    padded_tests = torch.zeros(batch_size, max_length_test)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        file_names.append(item[0])
        waveform_gt = item[1]
        waveform_test = item[2]
        padded_waveform_gt = torch.nn.functional.pad(
            waveform_gt, (0, max_length_gt - waveform_gt.size(1))
        ).squeeze(0)
        padded_waveform_test = torch.nn.functional.pad(
            waveform_test, (0, max_length_test - waveform_test.size(1))
        ).squeeze(0)
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[3])
        except:
            label = np.nan

        padded_gts[i] = padded_waveform_gt
        padded_tests[i] = padded_waveform_test
        labels[i] = label

    return file_names, padded_gts, padded_tests, labels

def custom_single_batch_create(batch: list):
    """
    Custom collate_fn for the dataloader to create batches for batch training.

    Creates batches of single recordings for "normal" detection.
    Shorter waveforms are padded with zeros to match the length of the longest waveform in the batch.
    """
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths = torch.tensor([item[1].size(1) for item in batch])

    # Find the maximum length
    max_length = int(torch.max(lengths))

    # Pad the tensors to have the maximum length
    file_names = []
    padded_waveforms = torch.zeros(batch_size, max_length)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        file_names.append(item[0])
        waveform = item[1]
        padded_waveform = torch.nn.functional.pad(
            waveform, (0, max_length - waveform.size(1))
        ).squeeze(0)
        try:  # If the label is not available (or is None), set it to np.nan
            label = torch.tensor(item[2])
        except:
            label = np.nan

        padded_waveforms[i] = padded_waveform
        labels[i] = label

    return file_names, padded_waveforms, labels
