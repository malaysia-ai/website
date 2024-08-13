import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

def main():
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}'
    dist.init_process_group(backend='nccl')

    Q_block = torch.randn(50, 128).cuda(device=device).to(torch.bfloat16)
    K = torch.randn(50, 128).cuda(device=device).to(torch.bfloat16)
    V = torch.randn(50, 128).cuda(device=device).to(torch.bfloat16)

    block_attentions = []
    block_maxes = []

    for i in range(world_size):
        if i == local_rank:
            dist.broadcast(K, src=i)
            dist.broadcast(V, src=i)

            K_block = K
            V_block = V
        else:
            K_block = torch.empty_like(K)
            V_block = torch.empty_like(V)

            dist.broadcast(K_block, src=i)
            dist.broadcast(V_block, src=i)
        
        scores = torch.matmul(Q_block, K_block.T)
        block_max = scores.max(dim=-1, keepdim=True)[0]
        block_maxes.append(block_max)
        block_attention = torch.matmul(F.softmax(scores - block_max, dim=-1), V_block)
        block_attentions.append(block_attention)
    
    global_max = torch.max(torch.cat(block_maxes, dim=-1), dim=-1, keepdim=True)[0]

    scaled_attentions = [
        torch.exp(block_max - global_max) * block_attention
        for block_max, block_attention in zip(block_maxes, block_attentions)
    ]

    output = sum(scaled_attentions)
    print(local_rank, len(block_maxes), output.shape)

if __name__ == "__main__":
    main()