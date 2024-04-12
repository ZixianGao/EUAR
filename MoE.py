import torch
import torch.nn.functional as F
from torch import Tensor, nn


def load_balanced_loss(router_probs, expert_mask):
    num_experts = expert_mask.size(-1)

    density = torch.mean(expert_mask, dim=0)
    density_proxy = torch.mean(router_probs, dim=0)
    loss = torch.mean(density_proxy * density) * (num_experts ** 2)

    return loss

class Gate(nn.Module):
    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=True):
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        loss_gate_scores = gate_scores
        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(3, dim=-1) #3/4

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            # load = gate_scores.sum(0)  # Sum over all examples
            # importance = gate_scores.sum(1)  # Sum over all experts

            # # Aux loss is mean suqared difference between load and importance
            # loss = ((load - importance) ** 2).mean()
            loss = load_balanced_loss(loss_gate_scores,mask)

            return gate_scores, loss

        return gate_scores, None


class FeedForward(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim,output_dim)
        self.logvar = nn.Linear(input_dim,output_dim)

    def reparameterise(self,mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def KL_loss(self,mu,logvar):
        return (-(1+logvar-mu.pow(2)-logvar.exp())/2).sum(dim=1).mean()

    def forward(self,x):
        mu=self.mu(x)
        logvar=self.logvar(x)
        z = self.reparameterise(mu,logvar)
        kl_loss = self.KL_loss(mu,logvar)
        return mu , kl_loss ,torch.exp(logvar)

class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim)
                for _ in range(num_experts)
            ]
        )

        self.gate = Gate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )
        expert_outputs = []
        loss_kl = []
        Uncertainty =[]
        for expert_output, kl_loss,sigma in [expert(x) for expert in self.experts]:
            expert_outputs.append(expert_output)
            loss_kl.append(kl_loss)
            Uncertainty.append(sigma)
        
        loss_KL=0
        for i in range(self.num_experts):
            loss_KL+=loss_kl[i]
        
        loss = loss +(loss_KL)/self.num_experts
        Uncertainty = torch.stack(Uncertainty).sum(2).permute(1,0)
        loss_u=(Uncertainty * gate_scores).sum(1).mean()
        loss = loss + loss_u


        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss


class MoE_block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.1,
        num_experts: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout



        self.ffn = MoE(
            dim, dim * mult, dim, num_experts, *args, **kwargs
        )
        
        self.add_norm = nn.LayerNorm(dim)


    def forward(self, x: Tensor):
        # resi = x
        # # x, _, _ = self.attn(x)
        # x = x + resi
        # x = self.add_norm(x)
        # add_normed = x
        
        #### MoE #####
        x, loss = self.ffn(x)
        # x = x + add_normed
        # x = self.add_norm(x)
        return x , loss

if __name__ == "__main__":
    x = torch.randint(0, 100, (1, 10)).cuda()
    model = MoE_block(
        num_tokens=10, dim=128, heads=8, dim_head=64
    ).float().cuda()

    Embedding_layer=nn.Embedding(100,128).cuda()
    x=Embedding_layer(x).squeeze()
    out = model(x)

    print(out.shape)