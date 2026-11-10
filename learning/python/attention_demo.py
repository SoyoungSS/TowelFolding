"""Inspect a single untrained attention head with explicit Q/K/V arithmetic."""
import json
import math
import torch
from torch import nn
import matplotlib.pyplot as plt

def demonstrate(sequence, index, out):
    out.mkdir(exist_ok=True)
    torch.manual_seed(7)
    width=sequence.shape[-1];head_dim=32
    q_proj=nn.Linear(width,head_dim)
    k_proj=nn.Linear(width,head_dim)
    v_proj=nn.Linear(width,head_dim)
    output_proj=nn.Linear(head_dim,width)
    with torch.no_grad():
        q=q_proj(sequence);k=k_proj(sequence);v=v_proj(sequence)
        scores=q@k.transpose(-1,-2)/math.sqrt(head_dim)
        weights=scores.softmax(dim=-1)
        mixed=weights@v
        output=output_proj(mixed)
    assert q.shape==k.shape==v.shape==(1,148,32)
    assert weights.shape==(1,148,148) and output.shape==sequence.shape
    assert torch.allclose(weights.sum(-1),torch.ones(1,148),atol=1e-6)
    manual=(weights[0,0,:,None]*v[0]).sum(0)
    assert torch.allclose(manual,mixed[0,0],atol=1e-6)
    # A hand-selected arithmetic example, separate from the image tokens.
    toy_scores=torch.tensor([0.,1.,2.])
    toy_values=torch.tensor([10.,20.,40.])
    toy_weights=toy_scores.softmax(0)
    toy_result=(toy_weights*toy_values).sum().item()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    im=axes[0].imshow(weights[0].numpy(),aspect='auto',cmap='viridis')
    axes[0].set(xlabel='Source token (key / value)',ylabel='Receiving token (query)',title='Untrained attention: 148 x 148')
    fig.colorbar(im,ax=axes[0],label='Weight')
    axes[1].plot(weights[0,0].numpy())
    for edge in (.5,49.5,98.5):axes[1].axvline(edge,color='gray',linestyle=':',alpha=.6)
    axes[1].set(xlabel='Source token index',ylabel='Weight',title='How state token 0 reads all source tokens')
    fig.suptitle('Random Q/K/V projections: these weights do not explain a learned robot policy')
    fig.savefig(out/'attention.png',dpi=150);plt.close(fig)
    top=weights[0,0].topk(5)
    report={'q_shape':list(q.shape),'k_shape':list(k.shape),'v_shape':list(v.shape),'weights_shape':list(weights.shape),'output_shape':list(output.shape),'row_sum_max_error':float((weights.sum(-1)-1).abs().max()),'state_query_top_sources':[{'source':index[i],'weight':float(w)} for i,w in zip(top.indices.tolist(),top.values.tolist())],'toy_scores':toy_scores.tolist(),'toy_values':toy_values.tolist(),'toy_weights':toy_weights.tolist(),'toy_output':toy_result,'trained':False,'limitations':['single attention head; not a full Transformer block','Q K V and input projections untrained','same shape does not mean same contents','weights are not proof of causal importance or grasp locations']}
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('\n===== 第七课：单头自注意力 =====')
    print('输入：',tuple(sequence.shape))
    print('Q、K、V：',tuple(q.shape),'；注意力权重：',tuple(weights.shape))
    print('权重每行之和≈1；行0加权求和与矩阵计算一致，检查通过。')
    print('加权信息：',tuple(mixed.shape),'→ 输出投影：',tuple(output.shape))
    print('手算例：分数[0,1,2] → 权重', [round(x,4) for x in toy_weights.tolist()])
    print(f'值[10,20,40]加权求和 → {toy_result:.4f}，不是只选择最大值40。')
    print('图表：',out/'attention.png')
    print('此处新增参数全部未训练；权重大不代表理解毛巾或识别了抓取点。')
