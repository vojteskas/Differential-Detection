import torch
import torch.nn as nn

import fairseq  # type: ignore

# TODO: Try updating to PyTorch 2 and use built-in Wav2Vec2 XLSR model from torchaudio


class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()

        cp_path = "./xlsr2_300m.pt"
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it is not there
        if (
            next(self.model.parameters()).device != input_data.device
            or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # Batch support - process one utterance at a time
        emb_list = []
        for seq in input_tmp:
            seq = seq.unsqueeze(0)  # add batch dimension (batch of size 1)
            emb_seq = self.model(seq, mask=False, features_only=True)["x"]  # extract embedding
            emb_list.append(emb_seq)  # append to list

        emb = torch.cat(emb_list, dim=0)  # concatenate all embeddings into one batch

        return emb
