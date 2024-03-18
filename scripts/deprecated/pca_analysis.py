from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import os.path as osp
from glob import glob
import matplotlib.pyplot as plt


def pca_folder(folder):
    for name in ["val", "test", "train"]:
        data_path = glob(osp.join(folder, "loss_*_{}.pt".format(name)))[0]
        data = torch.load(data_path)
        embeddings = data["EMBEDDING"]
        diff_e = data['DIFF_E'].abs() * 23.061 # into kcal/mol

        map_small = diff_e < 1.
        desc_small = "mol_lvl_detail < 1kcal/mol"
        map_median = (diff_e > 1.) & (diff_e < 10.)
        desc_median = "1kcal/mol < mol_lvl_detail < 10kcal/mol"
        map_large = diff_e > 10.
        desc_large = "10.kcal/mol < mol_lvl_detail"

        pca = PCA(n_components=2)
        tsne = TSNE()
        embeddings_pca = pca.fit_transform(embeddings)
        embeddings_tsne = tsne.fit_transform(embeddings)
        for embeddings_2d, embeddings_name in zip([embeddings_pca, embeddings_tsne], ["pca", "tsne"]):
            plt.figure(figsize=(15, 10))
            for this_map, this_desc in zip([map_small, map_median, map_large], [desc_small, desc_median, desc_large]):
                plt.scatter(embeddings_2d[:, 0][this_map], embeddings_2d[:, 1][this_map], label=this_desc)
            plt.legend()
            plt.savefig(osp.join(folder, "{}_{}.png".format(name, embeddings_name)))


def pca_folders(name):
    from glob import glob
    folders = glob(name)
    for folder in folders:
        pca_folder(folder)


if __name__ == '__main__':
    pca_folder("../../PhysDimeTestTmp/exp191_test_2021-03-25_163238")
