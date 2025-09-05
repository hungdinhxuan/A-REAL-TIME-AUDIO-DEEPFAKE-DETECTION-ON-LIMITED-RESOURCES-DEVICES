import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple, Union

def pad(x: np.ndarray, padding_type: str = 'zero', max_len=64000, random_start=False) -> np.ndarray:
    '''
    pad audio signal to max_len
    x: audio signal
    padding_type: 'zero' or 'repeat' when len(X) < max_len, default 'zero'
        zero: pad with zeros
        repeat: repeat the signal until it reaches max_len
    max_len: max length of the audio, default 64000
    random_start: if True, randomly choose the start point of the audio
    '''
    # Ensure that max_len should be integer
    max_len = int(max_len)
    x_len = x.shape[0]
    padded_x = None
    if max_len == 0:
        # no padding
        print("Warning: max_len is 0, no padding will be applied")
        padded_x = x
    elif max_len > 0:
        if x_len >= max_len:
            if random_start:
                start = np.random.randint(0, x_len - max_len+1)
                padded_x = x[start:start + max_len]
            else:
                padded_x = x[:max_len]
        else:
            if random_start:
                # keep at least half of the signal
                start = np.random.randint(0, int((x_len+1)/2))
                x_new = x[start:]
            else:
                x_new = x

            if padding_type == "repeat":
                num_repeats = int(max_len / len(x_new)) + 1
                padded_x = np.tile(x_new, (1, num_repeats))[:, :max_len][0]

            elif padding_type == "zero":
                padded_x = np.zeros(max_len)
                padded_x[:len(x_new)] = x_new

    else:
        raise ValueError("max_len must be >= 0")

    return padded_x

def multi_view_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='repeat', random_start=False, view_padding_configs: Dict[str, Dict[str, bool]] = None):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    multi_view_collate_fn(batch, views=[1, 2], sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    # Set default configurations if none provided
    if view_padding_configs is None:
        view_padding_configs = {
            str(i): {'padding_type': 'repeat', 'random_start': False}
            for i in range(1, 5)
        }

    # Extract views from config and convert to integers
    views = [int(view) for view in view_padding_configs]

    view_batches = {view: [] for view in views}
    # Warning: padding_type and random_start are not used in this function
    # print("Warning: padding_type and random_start are not used in this function. Please use view_padding_configs instead")

    # Process each sample in the batch
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=view_padding_configs[str(view)]['padding_type'],
                         max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):

                x_view = torch.from_numpy(x_view)
            view_batches[view].append((x_view, label))

    # Convert lists to tensors
    for view in views:
        sequences, labels = zip(*view_batches[view])
        padded_sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        view_batches[view] = (padded_sequences, labels)

    return view_batches


def multi_view_collate_fn_for_scl(batch, views=[1, 2, 3, 4], sample_rate=16000,
                                  padding_type='repeat', random_start=False,
                                  view_padding_configs: Dict[str, Dict[str, bool]] = None):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels
    '''
    # Set default configurations if none provided
    if view_padding_configs is None:
        view_padding_configs = {
            str(i): {'padding_type': 'repeat', 'random_start': False}
            for i in range(1, 5)
        }

    # Extract views from config and convert to integers
    views = [int(view) for view in view_padding_configs]

    view_batches = {view: [] for view in views}

    # Process each sample in the batch
    for x_input, label in batch: # Batch is 1
        """
        x_input: list of numpy arrays
        label: list of integers with 0 for spoof, 1 bonafide
        """
        for index, (x) in enumerate(x_input):
            # Ensure x is a Tensor
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)

            # Pad each sample for each view
            for view in views:
                view_length = view * sample_rate

                x_view = pad(x.numpy(), padding_type=view_padding_configs[str(view)]['padding_type'],
                             max_len=view_length, random_start=view_padding_configs[str(view)]['random_start'])

                # Convert padded output to Tensor
                if not torch.is_tensor(x_view):
                    x_view = torch.tensor(x_view, dtype=torch.float32)

                # Ensure label is a tensor (fixing the warning)
                if not isinstance(label[index], torch.Tensor):
                    label_tensor = torch.tensor(label[index], dtype=torch.long)
                else:
                    label_tensor = label[index].clone().detach()

                view_batches[view].append((x_view, label_tensor))

    # Convert lists to tensors
    for view in views:
        sequences, labels = zip(*view_batches[view])
        # Ensuring all sequences are tensors
        padded_sequences = torch.stack(sequences)
        labels = torch.stack(labels)  # Convert labels to tensor properly
        view_batches[view] = (padded_sequences, labels)


    return view_batches


def variable_multi_view_collate_fn(batch, top_k=4, min_duration=16000, max_duration=64000, sample_rate=16000, padding_type='zero', random_start=True):
    '''
    Collate function to pad each sample in a batch to multiple views with variable duration
    :param batch: list of tuples (x, label)
    :param top_k: number of views to pad each sample to
    :param min_duration: minimum duration of the audio
    :param max_duration: maximum duration of the audio
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param random_start: whether to randomly start the sample
    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    variable_multi_view_collate_fn(batch, top_k=2, min_duration=16000, max_duration=32000, sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    # Duration of each view should be picked from a range of min_duration to max_duration by a uniform distribution
    # Duration in seconds for each view
    durations = np.random.uniform(
        min_duration, max_duration, top_k).astype(int)
    # Ensure unique durations to avoid key collisions
    views = np.unique(durations)
    view_batches = {view: [] for view in views}
    # Process each sample in the batch
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view
            x_view = pad(x, padding_type=padding_type,
                         max_len=view_length, random_start=random_start)
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):
                x_view = torch.from_numpy(x_view)
            view_batches[view].append((x_view, label))

    # Convert lists to tensors
    for view in views:
        sequences, labels = zip(*view_batches[view])
        padded_sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        view_batches[view] = (padded_sequences, labels)

    return view_batches


def mdt_collate_fn(batch, views=[1, 2, 3, 4], sample_rate=16000, padding_type='zero', num_per_sample=4):
    '''
    Collate function to pad each sample in a batch to multiple views
    :param batch: list of tuples (x, label)
    :param views: list of views to pad each sample to
    :param sample_rate: sample rate of the audio
    :param padding_type: padding type to use
    :param max_duration_per_view: maximum duration of the audio per view. 

    :return: dictionary with keys as views and values as tuples of padded sequences and labels

    Example:
    batch = [([1, 2, 3], 0), ([1, 2, 3, 4], 1)]
    multi_view_collate_fn(batch, views=[1, 2], sample_rate=16000)
    Output:
    {
        1: (tensor([[1, 2, 3], [1, 2, 3, 4]]), tensor([0, 1])),
        2: (tensor([[1, 2, 3, 0], [1, 2, 3, 4]]), tensor([0, 1]))
    }
    '''
    view_batches = {view: [] for view in views}

    # Process each sample in the batch
    for x, label in batch:
        # Pad each sample for each view
        for view in views:
            view_length = view * sample_rate
            x_view = pad(x, padding_type=padding_type,  max_len=view_length)
            # Check if x_view is Tensor or numpy array and convert to Tensor if necessary
            if not torch.is_tensor(x_view):
                x_view = torch.from_numpy(x_view)
            view_batches[view].append((x_view, label))

    # Convert lists to tensors
    for view in views:
        sequences, labels = zip(*view_batches[view])
        padded_sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        view_batches[view] = (padded_sequences, labels)

    return view_batches


class ChunkingCollator(object):
    def __init__(self, **params):
        self.enable_chunking = params.get('enable_chunking', False)
        print("🐍 File: components/collate_fn.py | Line: 146 | __init__ ~ self.enable_chunking",
              self.enable_chunking)
        self.chunk_size = params.get('chunk_size', 64600)
        print("🐍 File: components/collate_fn.py | Line: 149 | __init__ ~ self.chunk_size", self.chunk_size)
        self.overlap_size = params.get(
            'overlap_size', 0)  # Default overlap size is 0
        print("🐍 File: components/collate_fn.py | Line: 153 | __init__ ~ self.overlap_size", self.overlap_size)

    def __call__(self, batch):
        if self.enable_chunking:
            return self.chunking(batch)
        return batch

    def chunking(self, batch):
        chunk_size = self.chunk_size
        overlap_size = self.overlap_size
        step_size = chunk_size - overlap_size

        split_data = []

        for x_inp, utt_id in batch:
            # Calculate number of chunks with overlap
            num_chunks = (len(x_inp) - overlap_size) // step_size

            # handle case where the utterance is smaller than overlap_size
            if num_chunks <= 0:
                padded_chunk = pad(
                    x=x_inp, padding_type='repeat', max_len=chunk_size)
                padded_chunk = Tensor(padded_chunk)
                chunk_id = f"{utt_id}___0"
                split_data.append((padded_chunk, chunk_id))
                continue

            for i in range(num_chunks):
                start = i * step_size
                end = start + chunk_size
                chunk = x_inp[start:end]
                chunk_id = f"{utt_id}___{i+1}"
                split_data.append((chunk, chunk_id))

            # Handle the case where the utterance is smaller than chunk_size
            if num_chunks * step_size + overlap_size < len(x_inp):
                start = num_chunks * step_size
                chunk = x_inp[start:]
                padded_chunk = pad(
                    x=chunk, padding_type='repeat', max_len=chunk_size)
                padded_chunk = Tensor(padded_chunk)
                chunk_id = f"{utt_id}___{num_chunks+1}"
                split_data.append((padded_chunk, chunk_id))

        # Convert to tensors (if they are not already tensors)
        x_inp_list, utt_id_list = zip(*split_data)

        x_inp_tensor = torch.stack(x_inp_list) if isinstance(
            x_inp_list[0], torch.Tensor) else torch.tensor(x_inp_list)
        return x_inp_tensor, utt_id_list
