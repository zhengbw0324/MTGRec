import numpy as np
from sklearn.cluster import KMeans
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        hidden_sizes (list): List of integers representing the sizes of hidden layers.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Attributes:
        mlp (nn.Sequential): Sequential container for the MLP layers.

    """

    def __init__(self, hidden_sizes: list, dropout: float = 0.0):
        super(MLP, self).__init__()
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            activation_func = nn.ReLU()
            if idx != len(hidden_sizes) - 2:
                mlp_modules.append(activation_func)
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.mlp(x)


class RQLayer(nn.Module):


    def __init__(self, n_codebooks, codebook_size, latent_size, vq_type="ema", beta=0.25, decay=0.99, eps=1e-5):
        super(RQLayer, self).__init__()
        self.n_codebooks = n_codebooks
        self.latent_size = latent_size
        # Check if codebook_size is an int and convert it to a list of the same size for each level
        if isinstance(codebook_size, int):
            self.codebook_sizes = [codebook_size] * n_codebooks
        elif isinstance(codebook_size, list):
            if len(codebook_size) == n_codebooks:
                self.codebook_sizes = codebook_size
            else:
                raise ValueError("codebook_size must be an int or a list of int with length equal to n_codebooks")
        self.decay = decay
        self.eps = eps
        if vq_type == "ema":
            self.quantization_layers = nn.ModuleList([
                EMAVQLayer(latent_size, codebook_size, beta, decay, eps)
                for codebook_size in self.codebook_sizes
            ])
        else:
            self.quantization_layers = nn.ModuleList([
                VQLayer(latent_size, codebook_size, beta)
                for codebook_size in self.codebook_sizes
            ])


    def forward(self, x: torch.Tensor):

        batch_size, _ = x.shape
        quantized_x = torch.zeros(batch_size, self.latent_size, device=x.device)
        sum_quant_loss = 0.0
        num_unused_codes = 0.0
        output = torch.empty(batch_size, self.n_codebooks, dtype=torch.long, device=x.device)
        for quantization_layer, level in zip(self.quantization_layers, range(self.n_codebooks)):
            quant, quant_loss, unused_codes, output[:, level] = quantization_layer(x)
            x = x - quant
            quantized_x += quant
            sum_quant_loss += quant_loss
            num_unused_codes += unused_codes

        mean_quant_loss = sum_quant_loss / self.n_codebooks


        return quantized_x, mean_quant_loss, num_unused_codes, output

    def generate_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:

        for quantization_layer in self.quantization_layers:
            x = quantization_layer.generate_codebook(x, device)
        return x


class VQLayer(nn.Module):

    def __init__(self, latent_size, codebook_size, beta=0.25):
        super(VQLayer, self).__init__()
        self.dim = latent_size
        self.n_embed = codebook_size
        self.beta=beta

        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self):
        return self.embed.weight

    def _copy_init_embed(self, init_embed):
        self.embed.weight.data.copy_(init_embed)

    def forward(self, x: torch.Tensor):

        latent = x.view(-1, self.dim)
        code_embs = self.get_code_embs()
        dist = (
                latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ code_embs.t()
                + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)

        embed_onehot = F.one_hot(embed_ind, self.n_embed)
        embed_onehot_sum = embed_onehot.sum(0)
        if distributed.is_initialized():
            distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
        unused_codes = (embed_onehot_sum == 0).sum().item()


        x_q = F.embedding(embed_ind, code_embs).view(x.shape)

        quant_loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()

        embed_ind = embed_ind.view(*x.shape[:-1])


        return x_q, quant_loss, unused_codes, embed_ind

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:

        code_embs = self.get_code_embs()
        return F.embedding(embed_id, code_embs)

    def encode_to_id(self, x: torch.Tensor) -> torch.Tensor:

        latent = x.view(-1, self.dim)
        code_embs = self.get_code_embs()
        dist = (
                latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ code_embs.t()
                + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])

        return embed_ind


    def generate_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:

        kmeans = KMeans(n_clusters=self.n_embed, n_init='auto').fit(x.detach().cpu().numpy())

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).view(self.n_embed, self.dim)
        # cluster_size = torch.tensor(np.bincount(kmeans.labels_), dtype=torch.float, device=device)

        if distributed.is_initialized():
            distributed.broadcast(centers, 0)
            # distributed.broadcast(cluster_size, 0)


        self._copy_init_embed(centers.clone())

        code_embs = self.get_code_embs()
        dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ code_embs.t()
                + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        x_q = self.embed_code(embed_ind)

        return x - x_q



class EMAVQLayer(VQLayer):

    def __init__(self, latent_size, codebook_size, beta=0.25, decay=0.99, eps=1e-5):
        super(EMAVQLayer, self).__init__(latent_size, codebook_size, beta)

        self.decay = decay
        self.eps = eps

        embed = torch.zeros(self.n_embed, self.dim)
        self.embed = nn.Parameter(embed, requires_grad=False)
        nn.init.xavier_normal_(self.embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(self.n_embed))


    def _copy_init_embed(self, init_embed):
        self.embed.data.copy_(init_embed)
        self.embed_avg.data.copy_(init_embed)
        self.cluster_size.data.copy_(torch.ones(self.n_embed, device=init_embed.device))

    def get_code_embs(self):
        return self.embed



    def forward(self, x: torch.Tensor):

        latent = x.view(-1, self.dim)
        code_embs = self.get_code_embs()
        dist = (
                latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ code_embs.t()
                + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)

        x_q = F.embedding(embed_ind, code_embs).view(x.shape)

        if self.training:

            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = embed_onehot.t() @ latent
            if distributed.is_initialized():
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)

            unused_codes = (embed_onehot_sum == 0).sum().item()

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )

            n = self.cluster_size.sum()
            norm_w = (
                n * (self.cluster_size + self.eps) / (n + self.n_embed * self.eps)
            )
            embed_normalized = self.embed_avg / norm_w.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        else:
            embed_onehot = F.one_hot(embed_ind, self.n_embed)
            embed_onehot_sum = embed_onehot.sum(0)
            if distributed.is_initialized():
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
            unused_codes = (embed_onehot_sum == 0).sum().item()


        quant_loss = self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()

        embed_ind = embed_ind.view(*x.shape[:-1])

        return x_q, quant_loss, unused_codes, embed_ind



class RQVAEModel(nn.Module):


    def __init__(
        self,
        hidden_sizes: list,
        n_codebooks: int,
        codebook_size: Union[int, list],
        dropout: float,
        vq_type: str,
        beta: float,
    ):
        super(RQVAEModel, self).__init__()
        self.encoder = MLP(hidden_sizes, dropout=dropout)
        # n_codebooks, codebook_size, latent_size, vq_type="ema", beta=0.25, decay=0.99, eps=1e-5
        self.quantization_layer = RQLayer(n_codebooks, codebook_size, hidden_sizes[-1], vq_type, beta)
        self.decoder = MLP(hidden_sizes[::-1], dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple:

        encoded = self.encoder(x)
        quantized_x, quant_loss, num_unused_codes, _ = self.quantization_layer(encoded)
        decoded = self.decoder(quantized_x)
        return decoded, quant_loss, num_unused_codes

    def encode(self, x: torch.Tensor) -> np.ndarray:

        encoded = self.encoder(x)
        _, _, _, output = self.quantization_layer(encoded)
        return output.detach().cpu().numpy()

    def generate_codebook(self, x: torch.Tensor, device: torch.device):
        x = x.to(device)
        encoded = self.encoder(x)
        self.quantization_layer.generate_codebook(encoded, device)
