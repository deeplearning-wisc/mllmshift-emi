import os
import os.path as osp
import argparse
import math
import json
from PIL import Image
import pickle
import pdb
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPProcessor
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from datasets import load_dataset


SVSHIFT = ["","_defocus_blur_1","_defocus_blur_3","_defocus_blur_5","_frost_1","_frost_3","_frost_5"]
STSHIFT = ["","_sr_4","_sr_7","_KeyboardAug_1","_KeyboardAug_3"]
SYNTHETIC_SHIFT_V_LIST = [f"llava_bench_coco{v}" for v in SVSHIFT]
SYNTHETIC_SHIFT_T_LIST = [f"llava_bench_coco{t}" for t in STSHIFT]
SYNTHETIC_SHIFT_J_LIST = ["llava_bench_coco_English"]+[f"llava_bench_coco{v}{t}" for v in SVSHIFT[1:] for t in STSHIFT[1:]]
SYNTHETIC_SHIFT_ALL_LIST = [f"llava_bench_coco{v}{t}" for v in SVSHIFT for t in STSHIFT]

NVSHIFT = ["llava_bench_coco","llava_bench_in_the_wild_easy","llava_bench_in_the_wild_normal","llava_bench_in_the_wild_hard"]
NTSHIFT = ["English","German","Chinese","Korean","Greek","Arabic","Hindi"]
NATURAL_SHIFT_V_LIST = [f"{v}_English" for v in NVSHIFT]
NATURAL_SHIFT_T_LIST = [f"llava_bench_coco_{t}" for t in NTSHIFT]
NATURAL_SHIFT_J_LIST = ["llava_bench_coco_English"]+[f"{v}_{t}" for v in NVSHIFT[1:] for t in NTSHIFT[1:]]
NATURAL_SHIFT_ALL_LIST = [f"{v}_{t}" for v in NVSHIFT for t in NTSHIFT]


#! Neural MI Estimator
'''
Source: "https://github.com/Linear95/CLUB", 
'''
class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size//2, y_dim),
                                    nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
   

#! JSD Estimator
'''
Source: "https://github.com/uk-cliplab/representationJSD/blob/main/Neural_estimation/jsd_estimators.py", 
'''
def vonNeumannEntropy(K, lowRank = False, rank = None):
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)
    if lowRank:
        ek_lr = torch.zeros_like(ek)
        ek_lr[-rank:] = ek[-rank:]
        remainder = ek.sum() - ek_lr.sum()
        ek_lr[:(n-rank)] = remainder/(n-rank)
        mk = torch.gt(ek_lr, 0.0)
        mek = ek_lr[mk]
    else:
        mk = torch.gt(ek, 0.0)
        mek = ek[mk]

    mek = mek/mek.sum()   
    H = -1*torch.sum(mek*torch.log(mek))
    return H

def deep_JSD(X,Y,model):
    phiX = model(X)
    phiY = model(Y)
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

def JSD_cov(covX,covY):
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD


#! EMI Estimator
class EMI(nn.Module):
    def __init__(self,
                 feature_dim=768,
                 mi_est_dim=500,
                 mi_ckpt_path=None,      # "estimator_ckpt/CLUB_synthetic.pt"
                 v_embedder_name=None,   # "openai/clip-vit-base-patch32"
                 t_embedder_name=None,   # "xlm-roberta-base" 
                 ):
        super(EMI, self).__init__()
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        self.mi_est = CLUB(feature_dim, feature_dim, mi_est_dim).to(self.device)
        print(f"\nInitialize MI estimator working on {feature_dim}-dim inputs...\n"
              f"Your embeddings of X, Y should be {feature_dim}-dim\n")

        if mi_ckpt_path is not None:
            print(f"Load (pre-)trained MI estimator from {mi_ckpt_path}...")
            self.mi_est.load_state_dict(torch.load(mi_ckpt_path, map_location=self.device))
        else:
            print("MI estimator needs to be trained first!!!")
        
        if (v_embedder_name is not None) and (t_embedder_name is not None):
            print(f"\nMI estimator will work on raw image and text directly with\n"
                  f"{v_embedder_name} and {t_embedder_name} as encoder models.\n")
            self.v_embedder = CLIPModel.from_pretrained(v_embedder_name).to(self.device) # "openai/clip-vit-base-patch32"
            self.v_processor = CLIPProcessor.from_pretrained(v_embedder_name)
            self.t_embedder = XLMRobertaModel.from_pretrained(t_embedder_name).to(self.device) # "xlm-roberta-base" 
            self.t_processor = XLMRobertaTokenizer.from_pretrained(t_embedder_name)
        else:
            print(f"MI estimator will work on pre-extracted embeddings")
            self.v_embedder = None
            self.v_processor = None
            self.t_embedder = None
            self.t_processor = None
        
    def forward(self,x_v,x_t,y_hat,y, return_emb=True):
        '''
        expected format: 
        (1) list of RGB images + list of raw text (default)
        (2) 2D tensor of pre-extraced img/txt embeddings
        '''
        if (self.v_embedder is not None) and (self.t_embedder is not None):
            v_inputs = self.v_processor(images=x_v, return_tensors="pt", padding=True)
            v_inputs = {k: v.to(self.device) for k, v in v_inputs.items()}
            t_inputs = self.t_processor(x_t, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t_inputs = {k: v.to(self.device) for k, v in t_inputs.items()}
            t_outputs = self.t_processor(y_hat, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t_outputs = {k: v.to(self.device) for k, v in t_outputs.items()}
            t_outputs_ref = self.t_processor(y, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t_outputs_ref = {k: v.to(self.device) for k, v in t_outputs_ref.items()}
            
            attn_mask_i = t_inputs['attention_mask'].unsqueeze(-1)
            attn_mask_yh = t_outputs['attention_mask'].unsqueeze(-1)
            attn_mask_y = t_outputs_ref['attention_mask'].unsqueeze(-1)
            

            #pdb.set_trace()
            with torch.inference_mode():
                z_v = self.v_embedder.vision_model(pixel_values=v_inputs['pixel_values'], output_hidden_states=True).hidden_states[-1][:,1:,:].mean(dim=1).float()
                z_t = self.t_embedder(**t_inputs, output_hidden_states=True).hidden_states[0] * attn_mask_i
                z_yhat = self.t_embedder(**t_outputs, output_hidden_states=True).hidden_states[0] * attn_mask_yh
                z_y = self.t_embedder(**t_outputs_ref, output_hidden_states=True).hidden_states[0] * attn_mask_y
            
                z_t = z_t[:,1:,:].sum(dim=1) / attn_mask_i[:,1:,:].sum(dim=1)#.unsqueeze(-1)
                z_yhat = z_yhat[:,1:,:].sum(dim=1) / attn_mask_yh[:,1:,:].sum(dim=1)#.unsqueeze(-1)
                z_y = z_y[:,1:,:].sum(dim=1) / attn_mask_y[:,1:,:].sum(dim=1)#.unsqueeze(-1)


                z_v = F.normalize(z_v, p=2, dim=-1)
                z_t = F.normalize(z_t, p=2, dim=-1)
                z_yhat = F.normalize(z_yhat, p=2, dim=-1)
                z_y = F.normalize(z_y, p=2, dim=-1)

                z = (z_v+z_t)*0.5
                
                model_mi = self.mi_est(z, z_yhat).item()
                ref_mi = self.mi_est(z, z_y).item()
                emi = model_mi - ref_mi

            if return_emb:
                return emi, model_mi, ref_mi, z_v, z_t, z_yhat, z_y
        else:
            x = (x_v+x_t)/2
            x.to(self.device); y_hat.to(self.device); y.to(self.device)
            emi = self.mi_est(x, y_hat) - self.mi_est(x, y)

        return emi, model_mi, ref_mi

def EMIDupperbound(px_v, px_t, py_hat, py, qx_v, qx_t, qy_hat, qy, entropy_scaler=None):
    covPXv = torch.matmul(torch.t(px_v),px_v)
    covQXv = torch.matmul(torch.t(qx_v),qx_v)
    jsd_v = JSD_cov(covPXv,covQXv).item()

    covPXt = torch.matmul(torch.t(px_t),px_t)
    covQXt = torch.matmul(torch.t(qx_t),qx_t)
    jsd_t = JSD_cov(covPXt,covQXt).item()

    covPYH = torch.matmul(torch.t(py_hat),py_hat)
    covPY = torch.matmul(torch.t(py),py)
    jsd_py = JSD_cov(covPYH,covPY).item()

    covQYH = torch.matmul(torch.t(qy_hat),qy_hat)
    covQY = torch.matmul(torch.t(qy),qy)
    jsd_qy = JSD_cov(covQYH,covQY).item()
    
    if entropy_scaler is None:
        # scale-adjusted upper bound
        emid_ub = jsd_v**(1/2) + jsd_t**(1/2) + jsd_py**(1/4) + jsd_qy**(1/4)
        return emid_ub, jsd_v, jsd_t, jsd_py, jsd_qy
    else:
        # true upper bound estimate
        emid_ub = entropy_scaler*(jsd_v**(1/2) + jsd_t**(1/2)) + 4*(jsd_py**(1/4) + jsd_qy**(1/4))
        return emid_ub, jsd_v, jsd_t, jsd_py, jsd_qy


def get_data_local(rootpath, ds_name, model, reference_model):
    query_fn = f'{rootpath}/{ds_name}.jsonl'
    image_fn = f'{rootpath}/{ds_name}/images'
    resp_model_fn = f'{rootpath}/{ds_name}_{model}.jsonl'
    resp_base_fn = f'{rootpath}/{ds_name}_{reference_model}.jsonl'
    
    queries = [json.loads(q) for q in open(osp.expanduser(query_fn), "r")]
    v_queries = [Image.open(osp.join(image_fn,q['image'])).convert('RGB') for q in queries]
    t_queries = [q['text'] for q in queries]
    resp_model = [json.loads(q)['text'] for q in open(osp.expanduser(resp_model_fn), "r")]
    resp_base = [json.loads(q)['text'] for q in open(osp.expanduser(resp_base_fn), "r")]
    return v_queries, t_queries, resp_model, resp_base

def get_data_hf(rootpath,hf_ds,ds_name,model):
    subset = hf_ds[ds_name]
    ds_name_global = ds_name.replace('_easy','')
    ds_name_global = ds_name_global.replace('_hard','')
    ds_name_global = ds_name_global.replace('_normal','')
    resp_model_fn = f'{rootpath}/{ds_name_global}_{model}.jsonl'
    resp_model = [json.loads(q)['text'] for q in open(osp.expanduser(resp_model_fn), "r")]
    return subset['image'], subset['question'], resp_model, subset['reference_answer']






if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    # specify these for the single dataset / single pair evaluation
    parser.add_argument("--ds_id", type=str, default="changdae/llavabench-shift-synthetic-v1")
    parser.add_argument("--src_ds", type=str, default="")
    parser.add_argument("--tar_ds", type=str, default="")
    
    # conduct evaluation across all combinations of shifts by default
    parser.add_argument("--shift_type", type=str, default="SYNTHETIC", choices=["SYNTHETIC","NATURAL"])
    parser.add_argument("--shift_modality", type=str, default="ALL", choices=["ALL","V","T","J"])
    
    # specify the embedding models (or use pre-extracted ones) and MI estimator configs
    parser.add_argument("--v_embedder_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--t_embedder_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--data_rootpath", type=str, default="data")
    parser.add_argument("--emb_rootpath", type=str, default="embeddings")
    parser.add_argument("--res_rootpath", type=str, default="results")
    parser.add_argument("--mi_ckpt_path", type=str, default="estimator_ckpt/CLUB_global.pt")
    parser.add_argument("--mi_est_dim", type=int, default=500)
    parser.add_argument("--feature_dim", type=int, default=768)

    parser.add_argument("--model_name", type=str, default="llava-v1.5-13b")
    parser.add_argument("--ref_model_name", type=str, default="gpt4")

    args = parser.parse_args()
    os.makedirs(args.res_rootpath,exist_ok=True)
    cfg_dict = vars(args) 
    res_dict = cfg_dict

    args.ds_id = f"changdae/llavabench-shift-{args.shift_type.lower()}-v1"
    if args.ds_id:
        dataset = load_dataset(args.ds_id)

    #* specify your (E)MI estimator
    emi_estimator = EMI(feature_dim=args.feature_dim,
                        mi_est_dim=args.mi_est_dim,
                        mi_ckpt_path=args.mi_ckpt_path,
                        v_embedder_name=args.v_embedder_name,
                        t_embedder_name=args.t_embedder_name,)
        
    #* emi estimation on a single dataset
    if len(args.src_ds)>0:
        res_fn = f'single-{args.src_ds}-{args.model_name}-{args.mi_ckpt_path.split("/")[-1][:-3]}'
        print(f"\nStart inference on {res_fn} !!! \n")

        src_ds = args.src_ds
        v_queries, t_queries, resp_model, resp_base = get_data_hf(args.data_rootpath, dataset, src_ds, args.model_name)

        src_emi, model_mi, ref_mi, p_zv, p_zt, p_zyh, p_zy = emi_estimator(v_queries, t_queries, resp_model, resp_base, True)

        res_dict['SRC_EMI'] = src_emi
        

    #* emid estimation on pairs of two datasets
    # inference on a single pair
    if args.src_ds and args.tar_ds:
        res_fn = f'singlepair-{args.src_ds}-{args.tar_ds}-{args.model_name}-{args.mi_ckpt_path.split("/")[-1][:-3]}'
        print(f"\nStart inference on {res_fn} !!! \n")

        tar_ds = args.tar_ds
        v_queries, t_queries, resp_model, resp_base = get_data_hf(args.data_rootpath, dataset, src_ds, args.model_name)
        tar_emi, model_mi, ref_mi, q_zv, q_zt, q_zyh, q_zy = emi_estimator(v_queries, t_queries, resp_model, resp_base, True)
        emid = src_emi - tar_emi
        emid_ub = EMIDupperbound(p_zv, p_zt, p_zyh, p_zy, q_zv, q_zt, q_zyh, q_zy, None)

        res_dict['TAR_EMI'] = tar_emi
        res_dict['EMID'] = emid
        res_dict['EMID_UB'] = emid_ub[0]
        print(f"SRC_EMI: {src_emi:.3f}, TAR_EMI: {tar_emi:.3f}, EMID: {emid:.3f}, EMID_UB: {emid_ub[0]:.3f}")
    
    # inference on the entire combinations of dataset pairs
    if len(args.src_ds)==0 and len(args.tar_ds)==0:
        res_fn = f'pairs-{args.shift_type}-{args.shift_modality}-{args.model_name}-{args.mi_ckpt_path.split("/")[-1][:-3]}'
        print(f"\nStart inference on {res_fn} !!! \n")
        if args.shift_type == 'NATURAL':
            with open(f"{args.data_rootpath}/lbwild_split_idx_dict.pkl", "rb") as file:
                idx = pickle.load(file)
                eidx, hidx = idx['easy_ood_fst_cd_new'], idx['hard_ood_fst_cd_new']

        emi_dict, emid_dict, emidub_dict = {}, {}, {}
        scenario_list = globals()[f"{args.shift_type}_SHIFT_{args.shift_modality}_LIST"]
        
        src_ds = scenario_list[0]
        v_queries, t_queries, resp_model, resp_base = get_data_hf(args.data_rootpath, dataset, src_ds, args.model_name)
        src_emi, model_mi, ref_mi, p_zv, p_zt, p_zyh, p_zy = emi_estimator(v_queries, t_queries, resp_model, resp_base, True)
        emi_dict[src_ds] = src_emi
        print(f"{src_ds:<{45}} -- S_EMI: {src_emi:.3f}")
        # print(f"source model mi {model_mi:.3f}")
        # print(f"source ref mi {ref_mi:.3f}")
        for tar_ds in scenario_list[1:]:

            v_queries, t_queries, resp_model, resp_base = get_data_hf(args.data_rootpath, dataset, tar_ds, args.model_name)
            
            if 'easy' in tar_ds:
                resp_model_ = [resp_model[i] for i in eidx]
            elif 'hard' in tar_ds:
                resp_model_ = [resp_model[i] for i in hidx]
            else:
                resp_model_ = resp_model

            tar_emi, model_mi, ref_mi, q_zv, q_zt, q_zyh, q_zy = emi_estimator(v_queries, t_queries, resp_model_, resp_base, True)
            emid = src_emi - tar_emi
            emid_ub = EMIDupperbound(p_zv, p_zt, p_zyh, p_zy, q_zv, q_zt, q_zyh, q_zy, None)

            emi_dict[tar_ds] = tar_emi
            emid_dict[tar_ds] = emid
            emidub_dict[tar_ds] = emid_ub[0]
            print(f"{tar_ds:<{45}} -- T_EMI: {tar_emi:.3f}, EMID: {emid:.3f}, EMID_UB: {emid_ub[0]:.3f}")
            # print(f"target model mi {model_mi:.3f}")
            # print(f"target ref mi {ref_mi:.3f}")

        res_dict['EMI']=emi_dict
        res_dict['EMID']=emid_dict
        res_dict['EMID_UB']=emidub_dict

        #pdb.set_trace()
        pr, ppval = stats.pearsonr(np.array(list(emid_dict.values())), np.array(list(emidub_dict.values())))
        res_dict['PearsonR']=pr
        res_dict['PearsonPval']=ppval
        print(f"EMID <-> UB Correlation Coef.: {pr:.3f} (pval: {ppval:.3f})")

    res_json = json.dumps(res_dict,indent=4)
    with open(osp.join(args.res_rootpath, res_fn+'.json'), "w") as f:
        f.write(res_json)