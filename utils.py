import difflib
import itertools
import math
import os
from scipy.spatial.distance import pdist, squareform
import re
import paddle
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from paddle.tensor import tensor
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, chi2_contingency
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix, accuracy_score, \
    silhouette_score
from scipy.spatial import ConvexHull
import seaborn as sns

def p_normalize(x, p=2):
    return x / (paddle.norm(x, p=p, axis=1, keepdim=True) + 1e-6)

def lifeline_analysis(df, title_g=None):
    '''
    :param df:
    生存分析画图，传入参数为df是一个DataFrame
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :param title_g: 图标题
    :return:
    '''
    n_groups = len(set(df["label"]))
    kmf = KaplanMeierFitter()
    plt.figure()
    for group in range(n_groups):
        idx = (df["label"] == group)
        kmf.fit(df['Survival'][idx], df['Death'][idx], label='class_' + str(group))

        ax = kmf.plot()
        plt.title(title_g)
        plt.xlabel("lifeline(days)")
        plt.ylabel("survival probability")
        treatment_median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
    plt.show()


# 富集分析
# def clinical_enrichment(label,clinical):
#     cnt = 0
#     # age 连续 使用KW检验
#     # print(label,clinical)
#     stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(label))
#     print(stat,p_value_age)
#     if p_value_age < 0.05:
#         cnt += 1
#         print("---age---")
#     # 其余离散 卡方检验
#     stat_names = ["gender","pathologic_T","pathologic_M","pathologic_N","pathologic_stage"]
#     for stat_name in stat_names:
#         if stat_name in clinical:
#             c_table = pd.crosstab(clinical[stat_name],label,margins = True)
#             stat, p_value_other, dof, expected = chi2_contingency(c_table)
#             print(stat, p_value_other, dof, expected)
#             if p_value_other < 0.05:
#                 cnt += 1
#                 print(f"---{stat_name}---")
#     return cnt


import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency

def clinical_enrichment(label, clinical):
    """
    计算临床变量的富集性，统计显著变量的个数。

    参数:
    - label: Pandas Series，代表分组信息（离散）
    - clinical: Pandas DataFrame，包含临床变量

    返回:
    - cnt: int，富集的变量数量（P 值 < 0.05）
    """
    cnt = 0  # 统计显著变量个数

    # 确保 label 是 NumPy 数组
    label = np.array(label)

    # **年龄 (age) 变量** -> Kruskal-Wallis检验
    if "age" in clinical:
        age_values = clinical["age"].dropna().to_numpy()  # 去除 NaN
        if len(age_values) > 1:  # 至少要有两个数
            stat, p_value_age = kruskal(age_values, label)
            print(f"Age Kruskal-Wallis Test: Stat={stat}, P-value={p_value_age}")
            if p_value_age < 0.5:
                cnt += 1
                print("--- Significant: age ---")

    # **离散变量** -> 卡方检验
    stat_names = ["gender", "pathologic_T", "pathologic_M", "pathologic_N", "pathologic_stage"]
    for stat_name in stat_names:
        if stat_name in clinical:
            data = clinical[stat_name].dropna()  # 去除 NaN
            if len(data.unique()) > 1:  # 至少要有两个类别
                c_table = pd.crosstab(data, label)
                stat, p_value_other, dof, expected = chi2_contingency(c_table)
                print(f"{stat_name} Chi-square Test: Stat={stat}, P-value={p_value_other}")
                if p_value_other < 0.5:
                    cnt += 1
                    print(f"--- Significant: {stat_name} ---")

    return cnt  # 返回富集数



def log_rank(df):
    '''
    :param df: 传入生存数据
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :return: res 包含了p log2p log10p
    '''
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    if cancer_type == 'kirc':
        replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
        clinical = clinical.rename(columns=replace)  # 为某个 index 单独修改名称
        clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
    survival['age'] = pd.NA # 初始化年龄
    survival['gender'] = pd.NA # 初始化年龄
    if 'pathologic_T' in clinical.columns:
        survival['T'] = pd.NA # 初始化年龄
    if 'pathologic_M' in clinical.columns:
        survival['M'] = pd.NA # 初始化年龄
    if 'pathologic_N' in clinical.columns:
        survival['N'] = pd.NA # 初始化年龄
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['stage'] = pd.NA # 初始化年龄
    i = 0
    # 找对应的参数
    for name in survival['PatientID']:
        # print(name)
        flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
        if flag:
            idx = list(clinical["sampleID"]).index(flag[0])
            survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
            survival['gender'][i] = clinical['gender'][idx]
            if 'pathologic_T' in clinical.columns:
                survival['T'][i] = clinical['pathologic_T'][idx]
            if 'pathologic_M' in clinical.columns:
                survival['M'][i] = clinical['pathologic_M'][idx]
            if 'pathologic_N' in clinical.columns:
                survival['N'][i] = clinical['pathologic_N'][idx]
            if 'tumor_stage.diagnoses' in clinical.columns:
                survival['stage'][i] = clinical['tumor_stage.diagnoses'][idx]
        else: print(name)
        i = i + 1
    return survival.dropna(axis=0, how='any')


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
     (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
     ri = (tp + tn) / (tp + tn + fp + fn)
     ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
     p, r = tp / (tp + fp), tp / (tp + fn)
     f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
     return ri, ari, f_beta

def cluster_evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_measure = get_rand_index_and_f_measure(label,pred)[2]
    return nmi, ari, acc, pur,f_measure
def tsne_show(x,y_pred,cancer):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    x_embedded = tsne.fit_transform(x)
    vis_x = x_embedded[:, 0]
    vis_y = x_embedded[:, 1]


    # 创建数据框
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = vis_x
    df_subset['tsne-2d-two'] = vis_y
    df_subset['label'] = y_pred

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=y_pred,
        size=y_pred,
        sizes=(20, 80),
        palette=sns.color_palette("hls",  len(set(y_pred))),
        edgecolor="b",  # 使用 edgecolor 而不是 edgecolors
        data=df_subset,
        legend="full",
        alpha=0.7
    )

    # 绘制凸包
    colors = ['#76EEC6', '#E3CF57', '#FF8247', '#33A1C9', '#AB82FF']
    for i in df_subset['label'].unique():
        points = df_subset[df_subset['label'] == i][['tsne-2d-one', 'tsne-2d-two']].values


        # get convex hull
        hull = ConvexHull(points)
        # get x and y coordinates
        # repeat last point to close the polygon
        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])
        # plot shape

        plt.fill(x_hull, y_hull, alpha=0.5, c=colors[i])

    plt.title("t-SNE Visualization of "+cancer+ " Cluster Variables")
    plt.show()

    # 创建颜色映射
    my_palette = dict(zip(np.unique(y_pred), ["r", "g", "b", "y","w"]))
    row_colors = pd.Series(y_pred).map(my_palette)
    row_colors = list(itertools.islice(itertools.cycle(row_colors), len(x)))
    row_colors_array = np.array(row_colors)

    # 检查长度是否匹配

    # 创建ClusterMap
    numpy_array = x.cpu().numpy()

    cluster = sns.clustermap(
        pd.DataFrame(numpy_array),
        metric="correlation",
        method="single",
        cmap="plasma",
        row_colors=row_colors,
        row_cluster=True,
        col_cluster=True,
        dendrogram_ratio=(0.2, 0.2)
    )

    plt.show()

def compute_cluster_centers(X, labels, n_clusters):
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float32)
    for k in range(n_clusters):
        members = (labels == k)
        if np.sum(members) == 0:
            continue
        centers[k, :] = np.mean(X[members], axis=0)
    return centers

def find_k_nearest_neighbors_with_similarity(similarity_matrix, k):
    k_nearest_neighbors = np.zeros_like(similarity_matrix)
    for i in range(similarity_matrix.shape[0]):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]  # 从大到小排序
        for j in sorted_indices[1:k + 1]:  # 跳过自身（第一个元素）
            k_nearest_neighbors[i, j] = similarity_matrix[i, j]
    return k_nearest_neighbors
def similar_matrix(data):
    # 计算欧氏距离矩阵
    distance_matrix = squareform(pdist(data.T, 'euclidean'))
    sigma = np.std(distance_matrix)

    # 计算高斯相似性矩阵
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))



    return similarity_matrix


def silhouette_plot(codes, max_k=15,cancer=None):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    Scores = []  # 存silhouette scores
    for k in range(2, max_k):
        #estimator = KMeans(n_clusters=k, random_state=555)  # construct estimator
        estimator = SpectralClustering(n_clusters=k, random_state=555, affinity='rbf', assign_labels='kmeans')
        estimator.fit(codes)
        Scores.append(
            silhouette_score(codes, estimator.labels_, metric='euclidean'))
    X = range(2, max_k)
    plt.figure()
    plt.xlabel('Cluster num K', fontsize=15)
    plt.ylabel('Silhouette Coefficient', fontsize=15)
    plt.plot(X, Scores, 'o-')
    plt.title('Silhouette Coefficient of '+cancer)
    plt.show()


def draw_eigen_plot(arr, max_cluster_num=15, cancer=None):
    """
    Draw the eigenvalues plot to finds optimal number of clusters in `arr` via eigengap method

    """
    plt.figure()
    # don't overwrite provided array!
    graph = arr.copy()

    graph = (graph + graph.T) / 2
    graph[np.diag_indices_from(graph)] = 0
    degree = graph.sum(axis=1)
    degree[np.isclose(degree, 0)] += np.spacing(1)
    degree = np.abs(degree)
    di = np.diag(1 / np.sqrt(degree))
    laplacian = di @ (np.diag(degree) - graph) @ di
    print(laplacian)
    # perform eigendecomposition and find eigengap
    eigs = np.sort(np.linalg.eig(laplacian)[0])
    plt.plot(eigs[0:max_cluster_num])
    plt.scatter(range(max_cluster_num), eigs[0:max_cluster_num])
    plt.xticks(range(max_cluster_num), np.arange(1, max_cluster_num + 1))
    plt.title("get the best cluster num of "+cancer+" by eigengap")

