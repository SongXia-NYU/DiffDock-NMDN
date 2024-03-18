from utils.atom_contrib_analyzer import AtomContribAnalyzer, AtomContribAnalyzerEns
import os.path as osp

FREESOLV_SDFS = "/home/carrot_of_rivia/Documents/BIG_BOIS/freesolv_sol/freesolv_sdfs"
OPENCHEM_LOGP_SDFS = "/home/carrot_of_rivia/Documents/BIG_BOIS/openchem_logP/openchem_logP_mmff_sdfs"

MODEL_FOLDER = "/home/carrot_of_rivia/Documents/PycharmProjects/raw_data/frag20-sol-finals"


def analyze_atom_contrib():
    trained_model = osp.join(MODEL_FOLDER, "exp_ultimate_freeSolv_13_RDrun_2022-05-20_100307__201005",
                                           "exp_ultimate_freeSolv_13_active_ALL_2022-05-20_100309")
    sdfs = FREESOLV_SDFS
    if "_active_ALL_" in osp.basename(trained_model):
        config_folder = osp.join(MODEL_FOLDER, "exp_ultimate_freeSolv_1_run_2022-04-22_185345")
        analyzer = AtomContribAnalyzerEns(trained_model, sdfs, config_folder=config_folder)
    else:
        analyzer = AtomContribAnalyzer(trained_model, sdfs)
    analyzer.run()


if __name__ == '__main__':
    analyze_atom_contrib()
