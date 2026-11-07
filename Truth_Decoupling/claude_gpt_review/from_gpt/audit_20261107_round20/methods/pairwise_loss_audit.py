"""Minimal differentiable audit of a reference-adjusted paired content loss."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    z=torch.tensor([.3,.1,.2],dtype=torch.float64,requires_grad=True)  # good, bad, IDK
    ref=torch.tensor([.2,.1,.3],dtype=torch.float64)
    lp=z.log_softmax(0);rp=ref.log_softmax(0)
    ce=-lp[0]
    margin=(lp[0]-rp[0])-(lp[1]-rp[1])
    pair=F.softplus(-margin)
    ce_grad=torch.autograd.grad(ce,z,retain_graph=True)[0]
    pair_grad=torch.autograd.grad(pair,z,retain_graph=True)[0]
    assert ce_grad[2]>0 and abs(float(pair_grad[2]))<1e-15
    assert torch.allclose(margin,(z[0]-z[1])-(ref[0]-ref[1]),atol=1e-15,rtol=0)
    changed=z.detach().clone();changed[2]+=3
    clp=changed.log_softmax(0)
    changed_pair=F.softplus(-((clp[0]-rp[0])-(clp[1]-rp[1])))
    assert torch.allclose(changed_pair,pair,atol=1e-15,rtol=0)
    # A two-token positive and one-token negative share the first distribution.
    tail=torch.log(torch.tensor(.6,dtype=torch.float64));tail_ref=torch.log(torch.tensor(.5,dtype=torch.float64))
    shared=((lp[0]+tail-rp[0]-tail_ref)-(lp[1]-rp[1]))/1.5
    separate=(lp[0]+tail-rp[0]-tail_ref)/2-(lp[1]-rp[1])
    shared_grad=torch.autograd.grad(F.softplus(-shared),z,retain_graph=True)[0]
    separate_grad=torch.autograd.grad(F.softplus(-separate),z)[0]
    assert abs(float(shared_grad[2]))<1e-15 and abs(float(separate_grad[2]))>1e-4
    # After the first differing token, the two prefixes are different: their
    # normalization terms must remain and can exert nonzero local IDK gradients.
    prefix=torch.tensor([.3,.1,.2],dtype=torch.float64,requires_grad=True)
    good_tail=torch.tensor([.2,.1,.7],dtype=torch.float64,requires_grad=True)
    bad_tail=torch.tensor([.1,.4,.5],dtype=torch.float64,requires_grad=True)
    m=(prefix.log_softmax(0)[0]+good_tail.log_softmax(0)[0]
       -prefix.log_softmax(0)[1]-bad_tail.log_softmax(0)[1])/2
    gradients=torch.autograd.grad(F.softplus(-m),(prefix,good_tail,bad_tail))
    assert abs(float(gradients[0][2]))<1e-15 and gradients[1][2]>0 and gradients[2][2]<0
    # Shared model parameters couple output coordinates: zero direct gradient
    # at z_IDK does NOT imply that an update preserves its probability.
    theta=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    logits=torch.stack((theta,theta*0,theta*2))
    before=float(logits.softmax(0)[2].detach());loss=F.softplus(-(logits[0]-logits[1]-.2))
    g=torch.autograd.grad(loss,theta)[0];new_theta=theta.detach()-.1*g
    after=float(torch.stack((new_theta,new_theta*0,new_theta*2)).softmax(0)[2])
    assert after>before
    result={'classes':['good','bad','IDK'],'logits':z.detach().tolist(),'reference_logits':ref.tolist(),
            'positive_CE_gradient':ce_grad.tolist(),'paired_gradient':pair_grad.tolist(),
            'IDK_logit_plus_three_preserves_pair_loss':True,
            'unequal_lengths':{'good_tokens':2,'bad_tokens':1,'common_length_factor':1.5,
                              'shared_factor_gradient':shared_grad.tolist(),'separate_mean_gradient':separate_grad.tolist()},
            'diverged_prefix_IDK_gradients':[float(x[2]) for x in gradients],
            'coupled_parameter_counterexample':{'IDK_probability_before':before,'IDK_probability_after':after},
            'scope':'direct gradients at an identical prefix only; no actual-answer-rate invariance theorem',
            'all_assertions_passed':True}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
