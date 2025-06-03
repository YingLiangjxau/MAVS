
import random



import paddle
import pandas as pd

from paddle.nn import functional as F
import numpy as np

from tqdm import tqdm


from sklearn.cluster import KMeans,SpectralClustering


import os

import csv


import warnings

import utils
import load_data
from model import MAVS, loss_funcation

warnings.filterwarnings("ignore")
DATASET_PATH = r"C:\Users\lkx57\Desktop\omics data"
seed = 123456
FLAGS_eager_delete_tensor_gb=0.0

if __name__ == '__main__':

    for i in range(10):
        cancer_type = 'aml'
        conf = dict()
        conf['dataset'] = cancer_type
        exp = pd.read_csv(os.path.join(DATASET_PATH, cancer_type, "exp3"), sep=",")
        methy = pd.read_csv(os.path.join(DATASET_PATH, cancer_type, "methy3"), sep=",")
        mirna = pd.read_csv(os.path.join(DATASET_PATH, cancer_type , "mirna3"), sep=",")
        survival = pd.read_csv(os.path.join(DATASET_PATH, cancer_type ,"survival"), sep="\t")
        survival = survival.dropna(axis=0)
        # Preprocessing method
        exp_df = paddle.to_tensor(exp.values.T, dtype=paddle.float32)
        methy_df = paddle.to_tensor(methy.values.T, dtype=paddle.float32)
        mirna_df = paddle.to_tensor(mirna.values.T, dtype=paddle.float32)
        full_data = [utils.p_normalize(exp_df), utils.p_normalize(methy_df), utils.p_normalize(mirna_df)]



        # params
        conf = dict()
        conf['dataset'] = cancer_type
        conf['view_num'] = 3
        conf['batch_size'] = 128
        conf['encoder_dim'] = [1024]
        conf['feature_dim'] = 512
        conf['peculiar_dim'] = 128
        conf['common_dim'] = 128
        conf['mu_logvar_dim'] = 10
        conf['cluster_var_dim'] = 3*conf['common_dim']
        conf['up_and_down_dim'] = 512
        conf['use_cuda'] = True
        conf['stop'] = 1e-6
        eval_epoch = 500
        lmda_list = dict()
        lmda_list['rec_lmda'] = 0.9
        lmda_list['KLD_lmda'] = 0.3
        lmda_list['I_loss_lmda'] = 0.1
        conf['kl_loss_lmda'] = 10
        conf['update_interval'] = 50
        conf['lr'] = 1e-4
        conf['min_lr'] = 1e-6
        conf['pre_epochs'] = 1000
        conf['idec_epochs'] = 500
        if conf['dataset'] == "aml":
           conf['cluster_num'] = 5
        if conf['dataset'] == "brca":
            conf['cluster_num'] = 5
        if conf['dataset'] == "skcm":
            conf['cluster_num'] = 5
        if conf['dataset'] == "lihc":
            conf['cluster_num'] = 5
        if conf['dataset'] == "coad":
            conf['cluster_num'] = 4
        if conf['dataset'] == "kirc":
            conf['cluster_num'] = 4
        if conf['dataset'] == "gbm":
            conf['cluster_num'] = 2
        if conf['dataset'] == "ov":
            conf['cluster_num'] = 3
        if conf['dataset'] == "lusc":
            conf['cluster_num'] = 3
        if conf['dataset'] == "sarc":
            conf['cluster_num'] = 5
        seed = 123456
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # ========================Result File====================
        folder = "result/{}_result".format(conf['dataset'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        result = open("{}/{}_{}.csv".format(folder, conf['dataset'], conf['cluster_num']), 'w+')
        writer = csv.writer(result)
        writer.writerow(['p', 'logp', 'log10p', 'epoch', 'step'])
        # =======================Initialize the model and loss function====================
        in_dim = [exp_df.shape[1], methy_df.shape[1], mirna_df.shape[1]]
        model = MAVS(in_dim=in_dim, encoder_dim=conf['encoder_dim'], feature_dim=conf['feature_dim'],
                      common_dim=conf['common_dim'],
                      mu_logvar_dim=conf['mu_logvar_dim'], cluster_var_dim=conf['cluster_var_dim'],
                      up_and_down_dim=conf['up_and_down_dim'], cluster_num=conf['cluster_num'],
                      peculiar_dim=conf['peculiar_dim'], use_cuda=conf['use_cuda'], view_num=conf['view_num'])
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.001)
        opt = paddle.optimizer.AdamW(learning_rate=conf['lr'], parameters=model.parameters(), grad_clip=clip)
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(conf['lr'], T_max=conf['pre_epochs'], eta_min=conf['min_lr'])
        loss = loss_funcation()
        # =======================pre-training VAE====================
        print("pre-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))
        pbar = tqdm(range(conf['pre_epochs']), ncols=120)
        max_log = 0.0
        max_label = []
        for epoch in pbar:
            sample_num = exp_df.shape[0]
            randidx = paddle.randperm(sample_num)
            for i in range(round(sample_num / conf['batch_size'])):
                idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
                data_batch = [utils.p_normalize(exp_df[idx]), utils.p_normalize(methy_df[idx]), utils.p_normalize(mirna_df[idx])]
                out_list, latent_dist = model(data_batch)

                l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                                    latent_dist=latent_dist,
                                    lmda_list=lmda_list, batch_size=conf['batch_size'])
                l.backward()
                opt.step()
                opt.clear_grad()
            # Evaluation model
            if (epoch + 1) % eval_epoch == 0:
                with paddle.no_grad():
                    model.eval()
                    out_list, latent_dist = model(full_data)
                    spectral = SpectralClustering(n_clusters=conf['cluster_num'], random_state=seed,
                                                  affinity='nearest_neighbors', assign_labels='kmeans')
                    spectral.fit(latent_dist['cluster_var'].cpu().numpy())
                    cluster_centers = utils.compute_cluster_centers(latent_dist['cluster_var'].detach().cpu().numpy(),
                                                                    spectral.labels_, conf['cluster_num'])

                    pred = spectral.labels_
                    survival["label"] = np.array(pred)
                    df = survival
                    res = utils.log_rank(df)
                    writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "pre"])
                    result.flush()
                    model.train()

                if (res['log10p'] > max_log):
                    max_log = res['log10p']
                    max_label = pred
                    paddle.save(model.state_dict(), "{}/{}_max_log.pdparams".format(folder, conf['dataset']))

            scheduler.step()
            pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                             rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                             KLD="{:3.4f}".format(loss_dict['KLD'].item()),
                             I_loss="{:3.4f}".format(loss_dict['I_loss'].item()))
        # =======================training MAVS=====================
        out_list, latent_dist = model(full_data)
        print("MAVS-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))


        spectral = SpectralClustering(n_clusters=conf['cluster_num'], random_state=seed, affinity='nearest_neighbors',
                                      assign_labels='kmeans').fit(
            latent_dist['cluster_var'].cpu().numpy())
        cluster_centers = utils.compute_cluster_centers(latent_dist['cluster_var'].detach().cpu().numpy(),
                                                        spectral.labels_, conf['cluster_num'])

        tensor = paddle.to_tensor(cluster_centers, dtype=paddle.float32)
        tensor_gpu = tensor.cuda()
        model.cluster_layer.data = tensor_gpu
        y_pred_last = spectral.labels_
        max_label_log = 0.0
        max_label_pred = y_pred_last

        pbar = tqdm(range(conf['idec_epochs']), ncols=120)
        for epoch in pbar:
            if epoch % conf['update_interval'] == 0:
                _, latent_dist = model(full_data)
                tmp_q = latent_dist['q']
                y_pred = tmp_q.cpu().numpy().argmax(1)
                weight = tmp_q ** 2 / tmp_q.sum(0)
                p = (weight.t() / weight.sum(1)).t()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                df = survival
                df["label"] = np.array(y_pred)
                res = utils.log_rank(df)
                writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "IDEC"])
                result.flush()
                if res['log10p'] > max_label_log:
                    max_label_log = res['log10p']
                    max_label_pred = y_pred
                    paddle.save(model.state_dict(), "{}/{}_max_label_log.pdparams".format(folder, conf['dataset']))

                if epoch > 0 and delta_label < conf['stop']:
                    print('delta_label {:.4f}'.format(delta_label), '< tol',
                          conf['stop'])
                    print('Reached tolerance threshold. Stopping training.')
                    break

            sample_num = exp_df.shape[0]
            randidx = paddle.randperm(sample_num)
            for i in range(round(sample_num / conf['batch_size'])):
                idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
                data_batch = [utils.p_normalize(exp_df[idx]), utils.p_normalize(methy_df[idx]), utils.p_normalize(mirna_df[idx])]
                out_list, latent_dist = model(data_batch)
                kl_loss = F.kl_div(latent_dist['q'].log(), p[idx])
                l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                                    latent_dist=latent_dist,
                                    lmda_list=lmda_list, batch_size=conf['batch_size'])
                l = conf['kl_loss_lmda'] * kl_loss
                l.backward()
                opt.step()
                opt.clear_grad()

            scheduler.step()
            pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                             rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                             KLD="{:3.4f}".format(loss_dict['KLD'].item()),
                             I_loss="{:3.4f}".format(loss_dict['I_loss'].item()),
                             KL_loss="{:3.4f}".format(kl_loss.item()))
        survival["label"] = np.array(max_label)
        clinical_max_data = utils.get_clinical(DATASET_PATH + "/clinical", survival, conf["dataset"])
        cnt_NI = utils.clinical_enrichment(clinical_max_data['label'],clinical_max_data)
        survival["label"] = np.array(max_label_pred)
        clinical_data = utils.get_clinical(DATASET_PATH + "/clinical", survival, conf["dataset"])
        cnt = utils.clinical_enrichment(clinical_data['label'],clinical_data)
        print("{}:    MAVS-NI:  {}/{:.1f}   MAVS-ALL:   {}/{:.1f}".format(conf['dataset'],cnt_NI,max_log,cnt,max_label_log))
        utils.silhouette_plot(np.array(latent_dist["cluster_var"]),cancer=cancer_type)
        utils.lifeline_analysis(df, title_g=cancer_type)


