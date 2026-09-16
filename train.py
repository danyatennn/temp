# > A slow and inefficient implementation of a slightly modified Trompt model
# > From the ICLM 2023 paper https://arxiv.org/abs/2305.18446

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
import os
import urllib.request
from tqdm import tqdm


class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3)
        # Without non-linearities! This is important to make significant speed-ups possible.
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)

    def forward(self, x: torch.Tensor, prev_cell_out: torch.Tensor) -> torch.Tensor:
        x_emb = x.T.unsqueeze(-1) * self.feature_emb_weight.unsqueeze(
            1
        ) + self.feature_emb_bias.unsqueeze(1)
        x_emb = F.relu(x_emb)
        x_emb = self.ln_emb(x_emb)

        d = prev_cell_out.shape[-1]
        W = self.dense_imp.weight

        prompt_part = F.linear(
            self.ln_prompt(self.emb_prompt),
            W[:, :d],
            self.dense_imp.bias,
        )

        prev_part = F.linear(
            prev_cell_out,
            W[:, d:],
            None,
        )

        x_prompt = prev_part + prompt_part + self.emb_prompt
        x_column = self.ln_col(self.emb_column)
        mask = torch.softmax(torch.matmul(x_prompt, x_column.T), dim=-1)

        C, B, D = x_emb.shape

        x_weighted = (mask @ x_emb.flatten(1)).view(-1, B, D).permute(1, 0, 2)

        x_out = x_weighted * (
            1.0 + self.dense_expand.weight
        ) + self.dense_expand.bias.unsqueeze(-1)
        return x_out


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw = torch.softmax(self.dense0(x).squeeze(-1), dim=-1)
        xnew = (pw.unsqueeze(-1) * x).sum(dim=-2)
        return self.dense_out(self.ln(F.relu(self.dense1(xnew))))


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList(
            [TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)]
        )
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x):
        outputs = []

        for cell in self.tcells:
            outputs.append(cell(x, self.prompt))

        outputs = torch.stack(outputs, dim=1)

        return self.tdown(outputs).squeeze(-1)


def load_from_url(url, cache_dir="."):
    filename = os.path.join(cache_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        with tqdm(unit="B", unit_scale=True, desc=filename) as pbar:
            urllib.request.urlretrieve(
                url, filename, reporthook=lambda _, b, t: pbar.update(b)
            )
    return torch.load(filename, map_location=torch.device("cpu"), weights_only=True)


TRAIN_DATA = (
    "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
)
VAL_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

if __name__ == "__main__":
    torch.manual_seed(0)

    train_dataset = torch.utils.data.TensorDataset(
        *map(torch.nan_to_num, load_from_url(TRAIN_DATA))
    )
    val_dataset = torch.utils.data.TensorDataset(
        *map(torch.nan_to_num, load_from_url(VAL_DATA))
    )

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (
        train_dataset.tensors[0],
        (train_dataset.tensors[1] - Y_mean) / Y_std,
    )

    model = Trompt(
        n_columns=train_dataset.tensors[0].shape[1],
        n_prompts=128,
        d_model=128,
        n_cycles=6,
    )
    device = torch.device("cuda:0")
    model.to(device)

    train_dl = torch.utils.data.DataLoader(
        train_dataset, num_workers=0, batch_size=1024, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=1024)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-5, fused=True
    )
    scaler = torch.amp.GradScaler("cuda")
    EPOCHS = 5
    MAX_BATCHES = 100

    for e in range(1, EPOCHS + 1):
        model.train()

        measured_samples = 0
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for i, (x, y) in enumerate(tqdm(train_dl, total=MAX_BATCHES)):
            if i >= MAX_BATCHES:
                break

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)

                loss = F.mse_loss(pred, y[:, None].expand_as(pred))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            measured_samples += x.shape[0]

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        print(f"{measured_samples / elapsed:.2f} samples/sec")

        model.eval()
        mae = 0

        with torch.inference_mode():
            for batch in val_dl:
                x, y = batch
                pred = model(x.to(device))
                mae += (
                    (pred.mean(dim=-1) * Y_std + Y_mean - y.to(device))
                    .abs()
                    .sum()
                    .item()
                )
            mae = mae / len(val_dataset)

            print(f">>> Epoch {e:>02}")
            print(f"Validation MAE = {mae:.5f}")
            print(">>>\n")
