class Tag:
    """
    Put tags all together for easier management.
    """
    def __init__(self):
        self.concat_props = set(["RAW_PRED", "LABEL", "atom_embedding", "ATOM_MOL_BATCH", "ATOM_Z", "PROP_PRED_MDN",
                                                 "PROP_PRED", "PROP_TGT", "UNCERTAINTY", "Z_PRED", "sample_id"])
        self.avg_props = set(["accuracy", "z_loss", "mdn_loss", "mdn_hist"])

    @property
    def requires_atomic_prop(self):
        return ["names_atomic"]

    @property
    def step_per_step(self):
        return ["StepLR"]

    @property
    def step_per_epoch(self):
        return ["ReduceLROnPlateau"]

    @property
    def loss_metrics(self):
        return ["mae", "rmse", "mse", "ce", "bce", "evidential"]

    # in validation step: concat result
    def val_concat(self, key: str):
        return key.startswith("DIFF") or key.startswith("MDN_") or key in self.concat_props

    # in validation step: calculate average
    def val_avg(self, key: str):
        return key.startswith(("MAE", "MSE", "CE")) or key in self.avg_props


tags = Tag()
