"""Table-derived analytic checks and synthetic counterexamples; no model/data/judge access."""
import json, math
from pathlib import Path

N = 529  # hypothetical sample size, not access to held-out questions
cdf = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2

def marginal(plus, minus, sample_n=192):
    mu = (plus - minus) / sample_n
    variance = (plus + minus) / sample_n - mu * mu
    se = math.sqrt(variance / N)
    return mu, se

mu,se = marginal(18,11)
main_probability = 1-cdf((max(.03,1.96*se)-mu)/se)
mu_a,se_a = marginal(5,6)
lo,hi=-.03+1.96*se_a,.03-1.96*se_a
retention_probability=max(0.,cdf((hi-mu_a)/se_a)-cdf((lo-mu_a)/se_a))
# These are approximations, not reproduction of full-arm joint simulation.
assert .56 < main_probability < .61
assert .56 < retention_probability < .63
# Frozen B's zero-event guard versus power script's normal interval.
zero_bound=1-.05**(1/N)
assert zero_bound>0
# Different baseline labels can add spurious apparent randomized-axis superiority.
main_right, main_base_right, random_right, axis_base_right=0,0,0,1
reported_contrast=(main_right-main_base_right)-(random_right-axis_base_right)
true_output_contrast=main_right-random_right
assert reported_contrast==1 and true_output_contrast==0
# Candidate decision, taken from the submitted four-gain tables.
gains={11:[17,13,7,2],13:[17,2,6,3]}
choice=sorted(gains,key=lambda l:(-min(gains[l]),l))[0]
assert choice==11
mc={}
for name,p in [('all_Pro',.005),('all_AllenAI',.14),('F_Pro',.014)]:
    sd=math.sqrt(p*(1-p)/2000)
    mc[name]={'reported_estimate':p,'monte_carlo_standard_error':sd,'approx_95_monte_carlo_interval':[max(0,p-1.96*sd),min(1,p+1.96*sd)]}
output={'scope':'CPU arithmetic from submitted aggregate tables and synthetic values; not raw-label or full-simulation replication; no test questions read',
        'hypothetical_n':N,'M_L11_Pro_main_normal_power_approx':main_probability,
        'M_L11_Pro_retention_normal_power_approx':retention_probability,
        'zero_change_interval':{'power_script':[0,0],'frozen_zero_event_bound':[-zero_bound,zero_bound]},
        'cross_directory_baseline_drift_example':{'reported_delta_difference':reported_contrast,'actual_right_count_difference':true_output_contrast},
        'M_candidate':choice,'four_gains':gains,'monte_carlo_uncertainty':mc}
Path(__file__).with_suffix('.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(output,ensure_ascii=False,indent=2))
