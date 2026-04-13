import os
import csv
import itertools
import torch
from train import train_st, train_dt


def make_cfg_id(cfg):
    return (f"H{cfg['hidden_size']}_l1{cfg['lambda_l1']}_l2{cfg['lambda_l2']}"
            f"_drop{cfg['dropout_p']}_e{cfg['e_prop']}"
            f"_ei{int(cfg.get('ei', True))}_act{cfg.get('activation', 'sigmoid')}"
            f"_lsel{cfg['lambda_sel']}_tau{cfg['tau_sel']}")


def run_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("gridsearch", exist_ok=True)

    # Choose a smaller seed set for testing first; expand later
    seeds = [42]
    # seeds = [
    #     0, 42, 1337, 271828, 314159, 1618033, 1414213, 1732051, 2236067, 57721566,
    #     1813382119, 827308000, 1627694679, 1911784258, 903170603, 86939547,
    #     556019486, 2073320062, 1097954098, 1043521779
    #     ]

    # ---- network size and regularization ----
    hidden_sizes = [100]                   # [50, 100, 200]                           

    lambda_l1s = [1e-4]                    # [0.0, 1e-6, 1e-5, 1e-4]
    lambda_l2s = [1e-4]                    # [0.0, 1e-5, 1e-4]
    dropout_ps = [0.05]                    # [0.0, 0.05]

    # separation loss params
    lambda_sels = [0.0]                    # [0.0, 1.0] ; weight of separation loss in total loss function
    tau_sels = [0.0]                       # [0.0, 1.0] ； 0 toward orthogonal representations, 1 toward more correlated representations

    # fixed RNN params
    base = dict(
        dt=50,
        e_prop=0.8,
        tau_r=100,
        sigma_rec=0.05,
        ei=True,                           # use E-I RNN architecture (if False, will use standard RNN)
        activation="sigmoid",              # "sigmoid" | "tanh" | "relu" | "identity"
        lr=1e-2,
        st_steps=1000,
        dt_steps=100,
        eval_trials=500,
        plot_sort=1,
        save_dir=os.path.join("gridsearch", "test"),
    )

    # results CSV
    os.makedirs(base["save_dir"], exist_ok=True)
    out_csv = os.path.join(base["save_dir"], "results.csv")
    fieldnames = [
        "seed", "cfg_id",
        "hidden_size", "lambda_l1", "lambda_l2", "dropout_p", "e_prop",
        "ei", "activation",
        "lambda_sel", "tau_sel",
        "st_ckpt", "dt_ckpt",
        "st_loss_last", "st_loss_mean_last100", "cs_DTe",
        "dt_loss_last", "dt_loss_mean_last20", "cs_DTl",
    ]
    write_header = not os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        # ---- Sweep loop ----
        for (H, l1, l2, drop, lsel, tau) in itertools.product(
            hidden_sizes, lambda_l1s, lambda_l2s, dropout_ps, lambda_sels, tau_sels
        ):
            cfg = dict(base)
            cfg.update(dict(
                hidden_size=H,
                lambda_l1=l1,
                lambda_l2=l2,
                dropout_p=drop,
                lambda_sel=lsel,
                tau_sel=tau,
            ))
            cfg_id = make_cfg_id(cfg)

            for seed in seeds:
                run_name = f"{cfg_id}_seed{seed}"
                cfg["run_name"] = run_name

                st_ckpt, st_m = train_st(cfg, seed, device)
                dt_ckpt, dt_m = train_dt(cfg, st_ckpt, device)

                row = dict(
                    seed=seed, cfg_id=cfg_id,
                    hidden_size=H, lambda_l1=l1, lambda_l2=l2,
                    dropout_p=drop, e_prop=cfg["e_prop"],
                    ei=cfg.get("ei", True),
                    activation=cfg.get("activation", "sigmoid"),
                    lambda_sel=lsel, tau_sel=tau,
                    st_ckpt=st_ckpt, dt_ckpt=dt_ckpt,
                    **st_m, **dt_m
                )
                w.writerow(row)
                f.flush()

    print("Done. Results:", out_csv)


if __name__ == "__main__":
    run_sweep()
