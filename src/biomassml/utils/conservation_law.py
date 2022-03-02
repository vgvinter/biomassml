import torch
from loguru import logger


def conservation_loss(y, names, label_scaler=None):
    """
    Compute the conservation loss.

    volSYNG (%) = volH2 (%) + volCO (%)
    volCOMB (%) = volH2 (%) + volCO (%) + volCH4 (%)
    H2_CO = volH2 (%) / volCO (%)
    HHV (MJ/m3) = f(volH2 (%); volCO (%); volCH4 (%))
    Edensity (MJ/kg biom) = HHV (MJ/m3)*GAS (m3/kg biom)
    CGE (%) = f(GAS (m3/kg biom); HHV (MJ/m3); HHVbiom (MJ/kg))
    [HHVbiom (MJ/kg) is an input variable]
    volCO2 (%) + volCO (%) + volCH4 (%) + volH2 (%) = 100
    """

    volSYNG_loss = 0
    volCOMB_loss = 0
    mass_conservation_loss = 0
    h2_co_loss = 0

    if label_scaler is not None:
        this_y = label_scaler.inverse_transform(y.clone())
    else:
        this_y = y

    column_idx_map = dict(zip(names, range(len(names))))
    if ("volSYNG (%)" in names) and ("volH2 (%)" in names) and ("volCO (%)" in names):
        volSYNG_loss = torch.abs(
            this_y[:, column_idx_map["volSYNG (%)"]]
            - this_y[:, column_idx_map["volH2 (%)"]]
            - this_y[:, column_idx_map["volCO (%)"]]
        ).mean()

    if (
        ("volCOMB (%)" in names)
        and ("volH2 (%)" in names)
        and ("volCO (%)" in names)
        and ("volCH4 (%)" in names)
    ):
        volCOMB_loss = torch.abs(
            this_y[:, column_idx_map["volCOMB (%)"]]
            - this_y[:, column_idx_map["volH2 (%)"]]
            - this_y[:, column_idx_map["volCO (%)"]]
            - this_y[:, column_idx_map["volCH4 (%)"]]
        ).mean()

    if (
        ("volCO2 (%)" in names)
        and ("volCO (%)" in names)
        and ("volCH4 (%)" in names)
        and ("volH2 (%)" in names)
    ):
        mass_conservation_loss = torch.abs(
            100
            - this_y[:, column_idx_map["volCO2 (%)"]]
            - this_y[:, column_idx_map["volCO (%)"]]
            - this_y[:, column_idx_map["volCH4 (%)"]]
            - this_y[:, column_idx_map["volH2 (%)"]]
        ).mean()

        if (("volH2 (%)" in names)
        and ("volCO (%)" in names)  and ("H2_CO" in names) ):
            h2_co_loss = torch.abs(
                this_y[:, column_idx_map["H2_CO"]]
                - this_y[:, column_idx_map["volH2 (%)"]]/this_y[:, column_idx_map["volCO (%)"]]
            ).mean()

    total_conservation_loss = volSYNG_loss + volCOMB_loss + mass_conservation_loss + h2_co_loss

    loss_dict = {
        "mass_conservation_loss": mass_conservation_loss,
        "volSYNG_loss": volSYNG_loss,
        "volCOMB_loss": volCOMB_loss,
        "total_conservation_loss": total_conservation_loss,
        "h2_co_loss": h2_co_loss,
    }
    return loss_dict
