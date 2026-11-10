"""Lesson 9: learned action queries and cross-attention, without optimization."""
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
from encoder_demo import TeachingEncoder

class TeachingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.norm1=nn.LayerNorm(128)
        self.norm2=nn.LayerNorm(128)
        self.norm3=nn.LayerNorm(128)
        self.ffn=nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Linear(256,128))

    def forward(self,queries,memory):
        self_result,self_weights=self.self_attention(queries,queries,queries,need_weights=True,average_attn_weights=False)
        x=self.norm1(queries+self_result)
        cross_result,cross_weights=self.cross_attention(x,memory,memory,need_weights=True,average_attn_weights=False)
        x=self.norm2(x+cross_result)
        x=self.norm3(x+self.ffn(x))
        return x,self_weights,cross_weights

class TeachingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=TeachingEncoder()
        self.action_queries=nn.Embedding(30,128)
        self.decoder=TeachingDecoder()
        self.action_head=nn.Linear(128,16)

    def forward(self,observations):
        # Expert targets are deliberately absent from the forward signature.
        memory,_=self.encoder(observations)
        queries=self.action_queries.weight.unsqueeze(0).expand(observations.shape[0],-1,-1)
        decoded,self_weights,cross_weights=self.decoder(queries,memory)
        return self.action_head(decoded),memory,queries,decoded,self_weights,cross_weights

def demonstrate_decoder(sequence,actions,frame,fps,out):
    torch.manual_seed(9)
    model=TeachingPolicy();model.eval()
    x=sequence.detach().clone().requires_grad_(True)
    prediction,memory,queries,decoded,self_weights,cross_weights=model(x)
    assert prediction.shape==(1,30,16)
    assert self_weights.shape==(1,4,30,30)
    assert cross_weights.shape==(1,4,30,148)
    assert torch.allclose(cross_weights.sum(-1),torch.ones(1,4,30),atol=1e-6)
    assert not torch.equal(queries[0,0],queries[0,1])
    valid=min(30,len(actions)-frame)
    assert valid>0
    # Build answers only after the forward pass, never as a model input.
    target=torch.tensor(actions[frame:frame+valid],dtype=torch.float32)[None,:,:]
    loss=(prediction[:,:valid]-target).square().mean()
    before=[p.detach().clone() for p in model.parameters()]
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for module in (model.encoder,model.action_queries,model.decoder,model.action_head):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in module.parameters())
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    with torch.no_grad():
        repeated=model(x.detach())[0]
        assert torch.allclose(prediction,repeated,atol=1e-6)
    avg=cross_weights[0].detach().mean(0)
    out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    im=axes[0].imshow(avg.numpy(),aspect='auto',cmap='viridis')
    axes[0].set(xlabel='Observation source token (148)',ylabel='Action slot (30)',title='Cross-attention averaged over 4 heads')
    fig.colorbar(im,ax=axes[0],label='Weight')
    t=torch.arange(valid).numpy()/fps
    axes[1].plot(t,target[0,:,0].numpy(),label='Expert target',linewidth=2)
    axes[1].plot(t,prediction[0,:valid,0].detach().numpy(),label='Untrained prediction',linestyle='--')
    axes[1].set(xlabel='Time from observation (s)',ylabel='Left joint 1 target (rad)',title='Output shape is correct; action is not learned')
    axes[1].legend();fig.suptitle('Teaching encoder-decoder; no CVAE, no optimizer update, no robot execution')
    fig.savefig(out/'decoder.png',dpi=150);plt.close(fig)
    report={'frame':frame,'observations':list(x.shape),'memory':list(memory.shape),'queries':list(queries.shape),'self_attention_per_head':list(self_weights.shape),'cross_attention_per_head':list(cross_weights.shape),'decoded':list(decoded.shape),'prediction':list(prediction.shape),'valid_target_steps':valid,'raw_action_mse':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),'parameters_unchanged':True,'gradient_reaches_encoder_queries_decoder_head':True,'optimizer_updates':0,'limitations':['untrained teaching architecture; no CVAE or complete ACT','raw action MSE mixes radians and gripper values; shape/gradient demonstration only','input tokens detached from visual backbone','untrained attention weights are not grasp explanations','action queries are random learned vectors, not expert actions']}
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('\n===== 第九课：动作查询、解码器与动作输出 =====')
    for name,v in [('观测token',x),('编码器输出',memory),('动作查询',queries),('逐头自注意力',self_weights),('逐头交叉注意力',cross_weights),('解码特征',decoded),('动作预测',prediction)]:
        print(f'{name}: {tuple(v.shape)}')
    print('平均各头后的交叉注意力：(30,148)，每个动作位置读取148个观测来源。')
    print('输出层128→16参数量：',sum(p.numel() for p in model.action_head.parameters()))
    print('有效标签步数：',valid,'；未归一化动作MSE（仅演示）：',report['raw_action_mse'])
    print('反向传播到编码器、查询、解码器和输出层：通过；参数未更新。')
    print('专家答案不作为本教学模型forward输入。输出不是可执行的合格动作。')
    print('图表：',out/'decoder.png')
