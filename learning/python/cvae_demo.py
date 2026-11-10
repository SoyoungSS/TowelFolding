"""Lesson 12: posterior sampling, KL regularization and answer-free inference."""
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
from decoder_demo import TeachingPolicy

class TeachingCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_projection=nn.Linear(16,128)
        self.summary_token=nn.Parameter(torch.randn(1,1,128)*.02)
        self.posterior_position=nn.Embedding(32,128)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True)
        self.posterior_encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.distribution=nn.Linear(128,64) # 32 means and 32 log variances.
        self.latent_projection=nn.Linear(32,128)
        self.policy=TeachingPolicy()
        self.posterior_calls=0

    def posterior(self,observations,actions,padding):
        self.posterior_calls+=1
        b=observations.shape[0]
        tokens=torch.cat((self.summary_token.expand(b,-1,-1),observations[:,:1],self.action_projection(actions)),dim=1)
        tokens=tokens+self.posterior_position(torch.arange(32))[None]
        mask=torch.cat((torch.zeros(b,2,dtype=torch.bool),padding),dim=1)
        summary=self.posterior_encoder(tokens,src_key_padding_mask=mask)[:,0]
        return self.distribution(summary).chunk(2,dim=-1)

    def decode(self,observations,z):
        latent_token=self.latent_projection(z)[:,None]
        return self.policy(torch.cat((latent_token,observations),dim=1))[0]

    def predict(self,observations):
        # No future action argument or posterior encoder call is possible here.
        return self.decode(observations,torch.zeros(observations.shape[0],32))

def standard_normal_kl(mu,logvar):
    return .5*(mu.square()+logvar.exp()-1-logvar).sum(-1).mean()

def demonstrate_cvae(sequence,actions,frame,fps,out):
    torch.manual_seed(12)
    model=TeachingCVAE();model.eval()
    obs=sequence.detach().clone()
    raw=torch.tensor(actions,dtype=torch.float32)
    valid=min(30,len(raw)-frame);assert valid>0
    ids=torch.arange(frame,frame+30).clamp(max=len(raw)-1)
    target=raw[ids][None]
    padding=(torch.arange(30)>=valid)[None]
    # Only a local teaching scale; not statistics for a deployable model.
    mean=target[:,:valid].mean((0,1))
    std=target[:,:valid].std((0,1),unbiased=False).clamp_min(.05)
    normalized=(target-mean)/std
    mu,logvar=model.posterior(obs,normalized,padding)
    sigma=(logvar*.5).exp()
    epsilon=torch.randn_like(mu)
    z=mu+sigma*epsilon
    prediction=model.decode(obs,z)
    reconstruction=(prediction-normalized).abs()[~padding].mean()
    kl=standard_normal_kl(mu,logvar)
    beta=10.
    total=reconstruction+beta*kl
    before=[v.detach().clone() for v in model.parameters()]
    total.backward()
    assert torch.isfinite(total)
    for module in (model.posterior_encoder,model.distribution,model.latent_projection,model.policy):
        assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in module.parameters())
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    assert standard_normal_kl(torch.zeros(1,32),torch.zeros(1,32)).item()==0
    calls_before=model.posterior_calls
    with torch.no_grad():
        zero_prediction=model.predict(obs)
        repeat=model.predict(obs)
        sampled_prediction=model.decode(obs,mu+sigma*torch.randn_like(mu))
        # Padded action tokens must not affect summary-token outputs.
        altered=normalized.clone();altered[padding]=999.
        mu_check,logvar_check=model.posterior(obs,altered,padding)
    assert model.posterior_calls==calls_before+1 # Only the explicit padding check above.
    assert torch.allclose(mu,mu_check,atol=1e-6) and torch.allclose(logvar,logvar_check,atol=1e-6)
    assert torch.allclose(zero_prediction,repeat,atol=1e-6)
    assert zero_prediction.shape==(1,30,16)
    out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    axes[0].plot(mu.detach()[0].numpy(),label='mu')
    axes[0].plot(sigma.detach()[0].numpy(),label='sigma')
    axes[0].plot(z.detach()[0].numpy(),label='sampled z',alpha=.7)
    axes[0].set(xlabel='Latent dimension',ylabel='Value',title='Posterior: 32 means and 32 standard deviations');axes[0].legend()
    t=torch.arange(valid).numpy()/fps
    axes[1].plot(t,target[0,:valid,0],label='Expert target',linewidth=2)
    for name,pred in [('posterior sample 1',prediction.detach()),('posterior sample 2',sampled_prediction),('inference z=0',zero_prediction)]:
        restored=pred*std+mean
        axes[1].plot(t,restored[0,:valid,0],label=name,linestyle='--')
    axes[1].set(xlabel='Time from observation (s)',ylabel='Left joint 1 target (rad)',title='Untrained model: different latent inputs, not learned styles');axes[1].legend()
    fig.savefig(out/'cvae.png',dpi=150);plt.close(fig)
    report={'frame':frame,'valid_steps':valid,'latent_dim':32,'mu_shape':list(mu.shape),'logvar_shape':list(logvar.shape),'prediction_shape':list(prediction.shape),
            'l1_reconstruction':float(reconstruction.detach()),'kl':float(kl.detach()),'beta':beta,'total_loss':float(total.detach()),
            'gradient_check':'passed','parameters_updated':False,'answer_free_inference':'passed','repeat_zero_latent_prediction':'passed','padding_invariance':'passed',
            'limitations':['randomly initialized teaching CVAE; not trained standard ACT','single chunk local normalization for illustration only','latent token prepended to teaching observations; simplified architecture','input visual features detached; no backbone gradient check','z=0 is the prior mean, not generally posterior mean or mean action']}
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print('\n===== 第十二课：CVAE训练分支与无答案推理 =====')
    print('专家动作(1,30,16) → 变分编码器 → mu、logvar各',tuple(mu.shape))
    print('sigma=exp(0.5*logvar)；z=mu+sigma*epsilon，epsilon来自标准正态分布。')
    print('有效步数：',valid,'；训练预测：',tuple(prediction.shape))
    print(f'L1动作重建={report["l1_reconstruction"]:.6f}；KL={report["kl"]:.6f}；beta={beta}；总损失={report["total_loss"]:.6f}')
    print('反向传播通过，未更新参数；标准正态分布与自身的KL为0。')
    print('predict只接收观测、不调用变分编码器；z=0重复预测一致。')
    print('图表：',out/'cvae.png')
    print('未训练曲线只展示计算，不代表已学会不同折叠风格或部署能力。')
