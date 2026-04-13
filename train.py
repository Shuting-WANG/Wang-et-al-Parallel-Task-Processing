import os
import numpy as np
import random
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from task import plot_activity, plot_behav, dataset, perf_trials
from model import Net


def set_seed(seed, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def weighted_mse(pred, target):
    err2 = (pred - target).pow(2).sum()
    pos = (target > 0.5).sum().clamp_min(1.0)
    return err2 / pos


def separation_loss(activity, cd_mask, e_prop=0.8, tau=0, eps=1e-6):
    """
    activity: (Time, Batch, n_neuron)
    cd_mask : (Time, Batch, 2):
              channel 0: 1=baseline, 2=motor
              channel 1: 3=Go, 4=No-Go
    Returns: sep_loss, cd_lever, cd_gng
    """
    _, _, n_neuron = activity.shape
    n_exc = int(round(e_prop * n_neuron))
    act_e = activity[:,:,:n_exc]  # (T,B,E)

    act_lever = act_e[cd_mask[:,:, 0] == 2].mean(dim=0)  # motor activity
    act_base = act_e[cd_mask[:,:, 0] == 1].mean(dim=0)  # baseline
    act_go = act_e[cd_mask[:,:, 1] == 3].mean(dim=0)  # activity during Go
    act_nogo = act_e[cd_mask[:,:, 1] == 4].mean(dim=0)  # activity during No-Go

    cd_lever = act_lever - act_base
    cd_gng   = act_go - act_nogo

    cs = F.cosine_similarity(cd_lever, cd_gng, dim=0, eps=eps)
    sep_loss = (tau - cs.abs()).abs()  # with tau=0 this is just |cos|
    return sep_loss, cd_lever, cd_gng


def load_net_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    net = Net(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        output_size=ckpt["output_size"],
        dropout_p=cfg["dropout_p"],
        dt=ckpt["dt"],
        e_prop=cfg["e_prop"],
        tau_r=cfg["tau_r"],
        sigma_rec=cfg["sigma_rec"],
        ei=cfg.get("ei", True),
        activation=cfg.get("activation", "sigmoid"),
    ).to(device)

    net.load_state_dict(ckpt["model_state"])
    return net, ckpt


def train_st(cfg, seed, device):
    """Train ST model, evaluate, save ckpt. Returns ckpt_path + summary metrics."""
    
    dt = cfg["dt"]
    tmp = dataset(dt)
    input_size = tmp["inputs"].shape[-1]
    output_size = tmp["targets"].shape[-1]

    set_seed(seed)

    net = Net(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        output_size=output_size,
        dropout_p=cfg["dropout_p"],
        dt=dt,
        e_prop=cfg["e_prop"],
        tau_r=cfg["tau_r"],
        sigma_rec=cfg["sigma_rec"],
        ei=cfg.get("ei", True),
        activation=cfg.get("activation", "sigmoid"),
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["lambda_l2"])

    st_running_loss = []
    net.train()
    for step in range(cfg["st_steps"]):
        outs = dataset(dt)
        x = torch.from_numpy(outs["inputs"]).float().to(device)
        y = torch.from_numpy(outs["targets"]).float().to(device)
        pred, rnn_activity = net(x)

        loss = weighted_mse(pred, y) + cfg["lambda_l1"] * rnn_activity.abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        st_running_loss.append(float(loss.item()))

    # test dual-task performance after ST training (DT-early stage)
    run_name = cfg["run_name"]
    coord_units, cs_DTe = evaluate_model(
        net, dataset_fn=dataset, dt=dt, num_trial=cfg["eval_trials"], e_prop = cfg["e_prop"], device=device,
        sort_task=cfg["plot_sort"], save_dir=cfg["save_dir"], run_name=f"{run_name}_pre",
        extra_mat={
            "st_running_loss": np.asarray(st_running_loss, dtype=np.float32),
            "seed": seed,
            "dt": dt,
            "ei": int(cfg.get("ei", True)),
            "activation": cfg.get("activation", "sigmoid"),
        }
    )

    ckpt_path = os.path.join(cfg["save_dir"], f"{run_name}_st.pt")
    torch.save({
        "model_state": net.state_dict(),
        "seed": seed,
        "dt": dt,
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": cfg["hidden_size"],
        "coord_units": np.asarray(coord_units, dtype=np.int64),
        "cfg": cfg,
    }, ckpt_path)

    metrics = {
        "st_loss_last": st_running_loss[-1],
        "st_loss_mean_last100": float(np.mean(st_running_loss[-100:])) if len(st_running_loss) >= 100 else float(np.mean(st_running_loss)),
        "cs_DTe": float(cs_DTe),
    }
    return ckpt_path, metrics


def train_dt(cfg, st_ckpt_path, device):
    """Load ST model, run your DT training (freeze/unfreeze), evaluate, save DT ckpt."""
    net, ckpt = load_net_from_ckpt(st_ckpt_path, device)
    dt = ckpt["dt"]
    hidden_size = ckpt["hidden_size"]
    coord_units = ckpt.get("coord_units", np.array([], dtype=np.int64))

    set_seed(int(ckpt["seed"]))

    # only update recurrent weights; freeze input and output layers
    for p in net.rnn.input2h.parameters():
        p.requires_grad = False
    for p in net.rnn.h2h.parameters():
        p.requires_grad = True
    for p in net.fc.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=cfg["lr"], weight_decay=cfg["lambda_l2"])

    # gradient masks
    coord_units_t = torch.as_tensor(coord_units, device=device)
    mask_coord = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    if coord_units_t.numel() > 0:
        mask_coord[coord_units_t] = 1.0
    mask_others = 1.0 - mask_coord
    mask_coord_w = mask_coord.view(-1, 1)
    mask_others_w = mask_others.view(-1, 1)

    # DT loop
    st_motorSuc, st_cogSuc, st_hit, st_cr = np.zeros(cfg["dt_steps"]), np.zeros(cfg["dt_steps"]), np.zeros(cfg["dt_steps"]), np.zeros(cfg["dt_steps"])
    dt_motorSuc, dt_cogSuc, dt_hit, dt_cr = np.zeros(cfg["dt_steps"]), np.zeros(cfg["dt_steps"]), np.zeros(cfg["dt_steps"]), np.zeros(cfg["dt_steps"])
    dt_running_loss = []
    net.train()
    for i in range(cfg["dt_steps"]):
        # ensure sufficient trial types
        while True:
            outs = dataset(dt, DT=True)
            trial_type = torch.from_numpy(outs["meta"]["trial_type"]).to(device)
            if trial_type.unique().numel() >= 5:
                break

        x = torch.from_numpy(outs["inputs"]).float().to(device)
        y = torch.from_numpy(outs["targets"]).float().to(device)
        CD_mask = torch.from_numpy(outs["masks"]["CD"]).to(device)
        dual_mask = torch.from_numpy(outs["masks"]["dual"]).to(device)
        trial_onset = torch.from_numpy(outs["meta"]["trial_onset"]).to(device)

        pred, rnn_activity = net(x)

        # task metrics per step
        st_motorSuc[i], st_cogSuc[i], st_hit[i], st_cr[i], dt_motorSuc[i], dt_cogSuc[i], dt_hit[i], dt_cr[i] = perf_trials(
            pred, trial_onset, trial_type, dt)

        # loss
        motor_thr = 0.3
        period_ignored = (dual_mask > 0.5) & (pred > motor_thr)
        loss_task = weighted_mse(pred[~period_ignored], y[~period_ignored])

        loss_sep, _, _ = separation_loss(rnn_activity, CD_mask, tau=cfg["tau_sel"])

        loss1 = loss_task + cfg["lambda_l1"] * rnn_activity.abs().mean()
        loss2 = loss1 + cfg["lambda_sel"] * loss_sep

        optimizer.zero_grad()
        h2h_weight = net.rnn.h2h.weight
        g1 = torch.autograd.grad(loss1, h2h_weight, retain_graph=True)[0]
        g2 = torch.autograd.grad(loss2, h2h_weight, retain_graph=False)[0]
        h2h_weight.grad = g1 * mask_coord_w + g2 * mask_others_w
        optimizer.step()

        dt_running_loss.append(float(loss1.item()))

    # test dual-task performance after DT training (DT-late stage)
    run_name = cfg["run_name"]
    _, cs_DTl = evaluate_model(
        net, dataset_fn=dataset, dt=dt, num_trial=cfg["eval_trials"], e_prop = cfg["e_prop"], device=device,
        sort_task=cfg["plot_sort"], save_dir=cfg["save_dir"], run_name=f"{run_name}_post",
        extra_mat={
            "dt_running_loss": np.asarray(dt_running_loss, dtype=np.float32),
            "ei": int(cfg.get("ei", True)),
            "activation": cfg.get("activation", "sigmoid"),
            "st_motorSuc": st_motorSuc,
            "st_cogSuc": st_cogSuc,
            "st_hit": st_hit,
            "st_cr": st_cr,
            "dt_motorSuc": dt_motorSuc,
            "dt_cogSuc": dt_cogSuc,
            "dt_hit": dt_hit,
            "dt_cr": dt_cr,}
    )

    dt_ckpt_path = os.path.join(cfg["save_dir"], f"{run_name}_dt.pt")
    torch.save({
        "model_state": net.state_dict(),
        "seed": ckpt["seed"],
        "dt": dt,
        "input_size": ckpt["input_size"],
        "output_size": ckpt["output_size"],
        "hidden_size": hidden_size,
        "coord_units": coord_units,
        "cfg": cfg,
    }, dt_ckpt_path)

    metrics = {
        "dt_loss_last": dt_running_loss[-1],
        "dt_loss_mean_last20": float(np.mean(dt_running_loss[-20:])) if len(dt_running_loss) >= 20 else float(np.mean(dt_running_loss)),
        "cs_DTl": float(cs_DTl),
    }
    return dt_ckpt_path, metrics


def evaluate_model(net, dataset_fn, dt, num_trial, e_prop, device,
                   sort_task = 0, align_window = (-1, 6),
                   save_dir: str | None = None, run_name: str | None = None, extra_mat: dict | None = None,
                   base_win = (0,1000), lever_win = (1000,6000), gng_win = (3000,4200)):
    """
    Evaluate a trained RNN model on a dual-task dataset and optionally generate visualizations and save outputs to .mat file.
    Args:
        net: The neural network model to evaluate (should be in eval mode).
        dataset_fn: Function that generates dataset batches. Should return a dict with keys:
            "inputs" (numpy array), "meta" (dict with "trial_onset" and "trial_type").
        dt: Time step.
        num_trial (int): Number of trials to evaluate.
        e_prop: Proportion of excitatory neurons in the RNN.
        device: 'cpu' or 'cuda'.
        sort_task (int, optional): Task type to sort trials for visualization. Defaults to 0 (self sorting).
        align_window (tuple, optional): Defaults to (-1, 6)s aligned to trial onset.
        save_dir (str | None, optional): Directory to save plots and .mat files. 
        run_name (str | None, optional): Name prefix for saved files. 
        extra_mat (dict | None, optional): Additional data to include in saved .mat file. 
        base_win (tuple, optional): Time window (in ms) for baseline activity calculation.
        lever_win (tuple, optional): Time window (in ms) for lever task activity calculation.
        gng_win (tuple, optional): Time window (in ms) for Go/No-Go task activity calculation.
    Returns:
        - Saves PNG plots to save_dir.
        - Saves .mat file with results.
        - Returns coord_units (numpy array of coordination unit indices) and task_cs (float cosine similarity between task CDs).
    """
    net.eval()
    pred_list, activity_list, trial_onsets_list, trial_types_list = [], [], [], []
    with torch.no_grad():
        for _ in range(num_trial):
            outs = dataset_fn(dt, batch_size=1, DT=True)

            x = torch.from_numpy(outs["inputs"]).float().to(device)
            pred, rnn_activity = net(x)  # pred: (T,B,2), activity: (T,B,N)

            # store per-trial (batch_size=1)
            pred_list.append(pred[:, 0, :].detach().cpu().numpy())
            activity_list.append(rnn_activity[:, 0, :].detach().cpu().numpy())

            trial_onsets_list.append((outs["meta"]["trial_onset"]).reshape(-1))
            trial_types_list.append((outs["meta"]["trial_type"]).reshape(-1))

    trial_onsets = np.concatenate(trial_onsets_list, axis=0)
    trial_types  = np.concatenate(trial_types_list, axis=0)

    # align activity and behavior to task onsets and make plots
    aligned_activity = None
    aligned_behav = None
    fig_a = fig_b = None

    activity_dict = {i: a for i, a in enumerate(activity_list)}
    behav_dict = {i: p for i, p in enumerate(pred_list)}

    fig_a, _, aligned_activity = plot_activity(list(align_window), dt, activity_dict, trial_onsets, trial_types, sort_task=sort_task)
    fig_b, _, aligned_behav = plot_behav(list(align_window), dt, behav_dict, trial_onsets, trial_types)

    if save_dir and run_name:
        os.makedirs(save_dir, exist_ok=True)
        fig_a.savefig(os.path.join(save_dir, f"{run_name}_activity.png"),dpi=300, bbox_inches="tight")
        plt.close(fig_a)
        fig_b.savefig(os.path.join(save_dir, f"{run_name}_behav.png"),dpi=300, bbox_inches="tight")
        plt.close(fig_b)

    # find coordination units that are oppositely modulated by the two tasks
    coord_units = np.array([])
    idx_GO = np.where(trial_types == 4)[0]
    idx_NG = np.where(trial_types == 5)[0]

    b0, b1 = int(base_win[0]/dt), int(base_win[1]/dt)
    l0, l1 = int(lever_win[0]/dt), int(lever_win[1]/dt)
    g0, g1 = int(gng_win[0]/dt), int(gng_win[1]/dt)

    exc_units = int(round(e_prop * aligned_activity.shape[2]))
    
    act_base = aligned_activity[idx_NG, b0:b1, :exc_units].mean(axis=(0, 1))
    act_LEVER = aligned_activity[idx_NG, l0:l1, :exc_units].mean(axis=(0, 1))
    act_GO = aligned_activity[idx_GO, g0:g1, :exc_units].mean(axis=(0, 1))
    act_NG = aligned_activity[idx_NG, g0:g1, :exc_units].mean(axis=(0, 1))

    cd_lever = act_LEVER - act_base
    cd_gng = act_GO - act_NG
    coord_units = np.nonzero((cd_lever * cd_gng) < 0)[0]
    cd_lever_exc = cd_lever[:exc_units]
    cd_gng_exc = cd_gng[:exc_units]
    task_cs = np.dot(cd_lever_exc, cd_gng_exc) / (np.linalg.norm(cd_lever_exc) * np.linalg.norm(cd_gng_exc) + 1e-6)
    
    # .mat saving
    if save_dir and run_name:
        os.makedirs(save_dir, exist_ok=True)
        mat_data = {
            "trial_types": trial_types,
            "trial_onsets": trial_onsets,
            "aligned_behavior": aligned_behav,
            "aligned_activity": aligned_activity,
            "coord_units": coord_units,
            "task_cs": task_cs,
        }
        if extra_mat:
            mat_data.update(extra_mat)

        sio.savemat(os.path.join(save_dir, f"{run_name}.mat"), mat_data)

    return coord_units, task_cs
    