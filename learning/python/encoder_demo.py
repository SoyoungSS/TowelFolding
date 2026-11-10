"""One explicit post-norm encoder block, checked against PyTorch's implementation."""
import json
import torch
from torch import nn
import matplotlib.pyplot as plt

class TeachingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.norm1=nn.LayerNorm(128)
        self.expand=nn.Linear(128,256)
        self.activation=nn.ReLU()
        self.contract=nn.Linear(256,128)
        self.norm2=nn.LayerNorm(128)

    def forward(self,x):
        attention,_=self.attention(x,x,x,need_weights=False)
        residual1=x+attention
        norm1=self.norm1(residual1)
        expanded=self.activation(self.expand(norm1))
        feedforward=self.contract(expanded)
        residual2=norm1+feedforward
        output=self.norm2(residual2)
        stages={'input':x,'attention':attention,'residual1':residual1,'norm1':norm1,
                'ffn_hidden':expanded,'ffn_output':feedforward,'residual2':residual2,'output':output}
        return output,stages

def demonstrate_encoder(sequence,out):
    torch.manual_seed(8)
    block=TeachingEncoder()
    block.eval()
    x=sequence.detach().clone().requires_grad_(True)
    output,stages=block(x)
    assert output.shape==x.shape==(1,148,128)
    assert stages['ffn_hidden'].shape==(1,148,256)
    # Same weights, same post-norm order, no dropout: compare with standard layer.
    reference=nn.TransformerEncoderLayer(128,4,dim_feedforward=256,dropout=0.,activation='relu',batch_first=True,norm_first=False)
    reference.self_attn.load_state_dict(block.attention.state_dict())
    reference.linear1.load_state_dict(block.expand.state_dict())
    reference.linear2.load_state_dict(block.contract.state_dict())
    reference.norm1.load_state_dict(block.norm1.state_dict())
    reference.norm2.load_state_dict(block.norm2.state_dict())
    reference.eval()
    with torch.no_grad():expected=reference(x.detach())
    assert torch.allclose(output,expected,atol=2e-6,rtol=1e-5)
    # Dummy readout checks gradient flow only; not a robot action target.
    readout=nn.Linear(128,1)
    parameters_before=[p.detach().clone() for p in block.parameters()]
    loss=readout(output).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradient_norms={name:float(p.grad.norm()) for name,p in block.named_parameters()}
    assert all(torch.isfinite(p.grad).all() for p in block.parameters())
    assert all(torch.equal(old,p) for old,p in zip(parameters_before,block.parameters()))
    out.mkdir(exist_ok=True)
    selected=['input','residual1','norm1','residual2','output']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name in ('input','attention','output'):
        axes[0].plot(stages[name][0,0].detach().numpy(),label=name,alpha=.8)
    axes[0].set(xlabel='Feature dimension',ylabel='Value',title='State token 0: same width, different contents')
    axes[0].legend()
    means=[float(stages[k][0,0].detach().mean()) for k in selected]
    stds=[float(stages[k][0,0].detach().std(unbiased=False)) for k in selected]
    axes[1].plot(means,'o-',label='Mean across 128 features')
    axes[1].plot(stds,'o-',label='Std across 128 features')
    axes[1].set_xticks(range(len(selected)),selected,rotation=20)
    axes[1].set(title='LayerNorm at initialization: mean near 0, std near 1')
    axes[1].legend();fig.suptitle('Untrained encoder block; dummy backward pass, no optimizer update')
    fig.savefig(out/'encoder.png',dpi=150);plt.close(fig)
    report={'input_shape':list(x.shape),'stage_shapes':{k:list(v.shape) for k,v in stages.items()},'attention_heads':4,'head_dimension':32,'ffn_hidden_dimension':256,'parameter_count':sum(p.numel() for p in block.parameters()),'reference_max_error':float((output-expected).detach().abs().max()),'dummy_loss':float(loss.detach()),'input_gradient_norm':float(x.grad.norm()),'parameter_gradient_norms':gradient_norms,'optimizer_updates':0,'parameters_unchanged':True,'limitations':['post-norm teaching configuration; not complete ACT','dropout set to zero for a reproducible comparison','dummy readout loss is not an action learning objective','input detached: no gradient check of the earlier visual backbone']}
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('\n===== 第八课：完整编码器块 =====')
    for name,value in stages.items():print(f'{name:12s}: {tuple(value.shape)}')
    print('4个注意力头，每头32维；FFN：128 → 256 → 128。')
    print('编码器块参数量：',report['parameter_count'])
    print('与相同权重的PyTorch标准编码器层比较：通过，最大差异',report['reference_max_error'])
    print('教学损失反向传播通过，输入梯度范数：',report['input_gradient_norm'])
    print('未调用optimizer.step，检查确认编码器参数没有改变。')
    print('图表：',out/'encoder.png')
    print('此处没有动作训练；LayerNorm初始缩放为1、偏置为0，训练后不能保证输出均值0、标准差1。')
