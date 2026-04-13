import numpy as np
import matplotlib.pyplot as plt
import torch

def dataset(dt:float, batch_size:int=16, DT:bool=False):
    """
    Generate a synthetic dataset for a multi-task design involving lever manipulation and Go/No-Go decisions,
    for training recurrent neural networks.
    Parameters
    ----------
    dt : Time step in milliseconds for discretizing the trial.
    batch_size: Number of trials to generate per batch. 
    DT : If True, generates dual-task trials (DT). If False, generates single-task trials (ST).
    Returns
    -------
    dict
        A dictionary containing:
        - "inputs" : ndarray of shape (Time, Batch, N_input)
            stimulus and context information.
        - "targets" : ndarray of shape (Time, Batch, N_output)
            Target outputs for lever position and lick rate.
        - "masks" : dict
            - "CD" : ndarray of shape (Time, Batch, 2)
                Coding direction masks indicating task phases (CD motor: baseline vs lever-hold in DT-NG trial; CD cog: cue & response in DT-Go vs DT-NG trial).
            - "dual" : ndarray of shape (Time, Batch, 2)
                Phases when the two tasks were processed simultaneously.
        - "meta" : dict
            - "trial_onset" : list of int; Time step indices where each trial begins.
            - "trial_type" : list of int; 1: ST-Lever, 2: ST-Go, 3: ST-No-Go, 4: DT-Go, 5: DT-No-Go
            - "dt"
    """

    # Task timing (ms)
    T, T_cue, T_lick, T_lever, T_btw, T_delay = 8000, 1000, 200, 4500, 1500, 500

    motor_tuning, cog_tuning, ctx_tuning = create_input_tuning()
    N_input = motor_tuning.shape[0]
    T_steps = round(T / dt)

    # Allocate lists
    all_inputs, all_targets, all_CD_masks, all_dual_masks = [], [], [], []
    all_trial_onset, all_trial_type = [], []

    for _ in range(batch_size):
        # randomly select a time point as start; leave 1000 ms at the beginning and end of the trial
        t_step_start = int(np.random.uniform(1000 / dt, (T - T_lever - T_delay - 1000) / dt)) 
        
        # initiate output arrays
        noise_std = 0.2
        single_input = np.random.normal(0, noise_std, size=(T_steps, N_input)).astype(np.float32) # stimulus input + context
        single_target = np.zeros((T_steps, 2), dtype=np.float32) # output: lever and GNG
        CD_mask = np.zeros((T_steps, 2), dtype=np.int8) # phase mask for calculating coding direction: 1=baseline, 2=motor, 3=Go, 4=No-Go
        dual_mask = np.zeros((T_steps, 2), dtype=np.float32) # dual-task period for evaluating motor performance
       
        # Randomly pick a task: 1: ST-Lever, 2: ST-Go, 3: ST-No-Go, 4: DT-Go, 5: DT-No-Go
        if DT:
            trial_type = np.random.choice([1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]) # 25% ST-Lever, 25% ST-GNG, 50% DT
        else:
            trial_type = np.random.choice([1, 2, 3])

        LEVER = trial_type in (1, 4, 5)
        GO    = trial_type in (2, 4)
        NOGO  = trial_type in (3, 5)

        if LEVER:
            single_input += ctx_tuning[:,0]
            single_input[t_step_start : t_step_start + round(T_lever / dt), :] += motor_tuning[:,0]    # motor stimulus input
            single_target[t_step_start + round(T_delay / dt) : t_step_start + round((T_delay + T_lever) / dt), 0] = 1    # motor output: lever position
        
        if GO:
            single_input += ctx_tuning[:,1]
            single_input[t_step_start + round((T_delay+T_btw) / dt):t_step_start + round((T_delay+T_btw+T_cue) / dt), :] += cog_tuning[:,0]    # Go stimulus input
            single_target[t_step_start + round((T_delay+T_btw+T_cue) / dt):t_step_start + round((T_delay+T_btw+T_cue+T_lick) / dt),1] = 1      # GNG output: lick rate

        if NOGO:
            single_input += ctx_tuning[:,1]
            single_input[t_step_start + round((T_delay+T_btw) / dt):t_step_start + round((T_delay+T_btw+T_cue) / dt), :] += cog_tuning[:,1]    # No-Go stimulus input

        if trial_type in (4, 5): 
            single_input += ctx_tuning[:,2]
            dual_mask[t_step_start + round((T_delay + T_btw) / dt) : t_step_start + round((T_delay + T_lever) / dt), 0] = 1  # from initiation of GNG task until completion of lever task
            if trial_type == 4:
                CD_mask[t_step_start + round((T_delay+T_btw) / dt):t_step_start + round((T_delay+T_btw+T_cue+T_lick) / dt),1] = 3  # Go cue presentation + response window
            if trial_type == 5:
                CD_mask[t_step_start - round(1000/dt) : t_step_start, 0] = 1                           # in DT-No-Go, baseline [-1,0]s
                CD_mask[t_step_start : t_step_start + round((T_delay + T_lever) / dt), 0] = 2          # in DT-No-Go, motor task window
                CD_mask[t_step_start + round((T_delay+T_btw) / dt):t_step_start + round((T_delay+T_btw+T_cue+T_lick) / dt),1] = 4  # No-Go cue presentation + response window        

        # relu
        single_input = np.maximum(single_input, 0)
    
        all_inputs.append(single_input)
        all_targets.append(single_target)
        all_CD_masks.append(CD_mask)
        all_dual_masks.append(dual_mask)
        all_trial_onset.append(t_step_start)
        all_trial_type.append(trial_type)

    out = {
        "inputs": np.stack(all_inputs, axis=1),        # (Time, Batch, N_input) 
        "targets": np.stack(all_targets, axis=1),      # (Time, Batch, N_output) 
        "masks": {
            "CD": np.stack(all_CD_masks, axis=1),      # (Time, Batch,2)
            "dual": np.stack(all_dual_masks, axis=1),  # (Time, Batch,2) 
        },
        "meta": {
            "trial_onset": np.asarray(all_trial_onset, dtype=np.int64),   # (Batch,) 
            "trial_type": np.asarray(all_trial_type, dtype=np.int64),     # (Batch,) 
            "dt": float(dt),
        }
    }
    return out


def create_input_tuning(num_motor=1, num_motor_tuned=24, num_cog=2, num_cog_tuned=12, num_ctx=3, num_ctx_tuned=6, kappa=2):
    """
    Create activity for motor-, cognitive-, and context-input neurons.
    Parameters
    ----------
    num_motor : Number of motor conditions. Default is 1 (Lever).
    num_cog : Number of cognitive stimulus conditions. Default is 2 (Go and No-Go).
    num_ctx : Number of task contexts. Default is 3 (motor task, cognitive task, dual-task).
    num_motor_tuned, num_cog_tuned, num_ctx_tuned:
        Number of motor-tuned, cognitive-tuned, and context-tuned input neurons. Default is 24, 12, and 6, respectively.
    kappa : Concentration parameter controlling the width of tuning curves.
    Returns
    -------
    motor_tuning : ndarray
        Shape (n_input, num_motor). Tuning responses for motor stimuli.
    cog_tuning : ndarray
        Shape (n_input, num_cog). Tuning responses for cognitive stimuli.
    ctx_tuning : ndarray
        Shape (n_input, num_ctx). Tuning responses for task contexts.
    """

    n_input = num_motor_tuned + num_cog_tuned + num_ctx_tuned  # total number of input-tuned neurons
    motor_tuning = np.zeros((n_input, num_motor))
    cog_tuning = np.zeros((n_input, num_cog))
    ctx_tuning = np.zeros((n_input, num_ctx))

    motor_stim = np.float32(np.arange(0,360,360/num_motor))
    motor_pref = np.float32(np.arange(0,360,360/num_motor_tuned)) 
    for n in range(num_motor_tuned):
        d = np.cos((motor_stim[0] - motor_pref[n])/180*np.pi)
        motor_tuning[n,0] = 4*np.exp(kappa*d)/np.exp(kappa)

    cog_stim = np.float32(np.arange(0,360,360/num_cog))
    cog_pref = np.float32(np.arange(0,360,360/num_cog_tuned))
    for n in range(num_cog_tuned):
        for i in range(num_cog):
            d = np.cos((cog_stim[i] - cog_pref[n])/180*np.pi)
            cog_tuning[num_motor_tuned+n,i] = 4*np.exp(kappa*d)/np.exp(kappa)

    for n in range(num_ctx_tuned):
        for i in range(num_ctx):
            if n%num_ctx == i:
                ctx_tuning[num_cog_tuned+num_motor_tuned+n,i] = 4

    return motor_tuning, cog_tuning, ctx_tuning


def align_data (win, data_dict, trial_onsets):
    """
    Align each trial's output trace to task onset.
    """
    num_trial = len(trial_onsets)
    aligned = [data_dict[i][trial_onsets[i]+win[0]:trial_onsets[i]+win[1],:] for i in range(num_trial)]
    aligned = np.array(aligned)
    return aligned


def plot_behav (win, dt, performance_dict, trial_onsets, trial_types, alpha=0.1):
    """
    Visualizes behavioral performance for different trial types and returns aligned performance data.

    Parameters:
        win: Time window (in seconds) to align and plot performance.
        dt: Time step size in milliseconds.
        trial_onsets: onset times (indices) for each trial.
        trial_types: trial type identifiers for each trial.
        performance_dict (dict): Dictionary mapping trial indices to performance arrays.
        
    Returns:
        aligned: Aligned performance data for each trial.
    """
    win = np.array(win)*1000/dt
    win = win.astype(int)
    
    aligned_beh = align_data(win, performance_dict, trial_onsets)

    t = np.arange(win[0], win[1]) * dt / 1000
    events = {"lever": [0, 5], "gng": [2, 3]}
    trial_labels = {1: "ST-Lever", 2: "ST-Go", 3: "ST-No-Go", 4: "DT-Go", 5: "DT-No-Go"}

    fig, ax = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)

    trial_order = [1, 2, 3, 4, 5]
    for col, tt in enumerate(trial_order):
        idx = np.where(trial_types == tt)[0]
        ax[0, col].set_title(trial_labels[tt])

        traces = aligned_beh[idx]
        ax[0, col].plot(t, traces[:, :, 0].T, alpha=alpha, color='#a2142f') # channel 0 (lever)
        ax[1, col].plot(t, traces[:, :, 1].T, alpha=alpha, color='#008281') # channel 1 (GNG)

        # event markers
        if tt in (1, 4, 5):  # lever involved
            for x in events.get("lever", []):
                ax[0, col].axvline(x, linestyle="--", color="gray")
                ax[1, col].axvline(x, linestyle="--", color="gray")
        if tt in (2, 3, 4, 5):  # GNG involved
            for x in events.get("gng", []):
                ax[0, col].axvline(x, linestyle="--", color="gray")
                ax[1, col].axvline(x, linestyle="--", color="gray")

    ax[0, 0].set_ylabel("Lever predictions")
    ax[1, 0].set_ylabel("GNG predictions")
    fig.tight_layout()
    return fig, ax, aligned_beh


def sort_activity(activity,sort_idx=None):
    if sort_idx is None:
        peak_time = np.argmax(activity, axis=0)
        sort_idx = np.argsort(peak_time)
    activity = activity[:,sort_idx]
    return activity, sort_idx


def plot_activity(win, dt, activity_dict, trial_onsets, trial_types, sort_task=None, normalize=True, neuron_type = 'exc', e_prop=0.8, vmin=0, vmax=0.8):
    """
    Visualize neural activity across different task conditions with trial-averaged heatmaps.
    Parameters
    ----------
    win : Time window for analysis [start, end] in seconds.
    dt : Time step/sampling interval in milliseconds.
    activity_dict : Dictionary mapping trial indices to neural activity arrays of shape (time, n_neurons).
    trial_onsets : Onset times for each trial.
    trial_types : Trial type identifier for each trial (1-5 corresponding to ST-Lever, ST-Go, ST-No-Go, DT-Go, DT-No-Go).
    sort_task : Trial type to use for sorting neurons by activity peak. If None (default), no sorting is performed. 
                If '0', sort within each condition; if 1-5, sort based on that trial type.
    normalize : If True (default), normalize activity per neuron across all conditions using max value.
    neuron_type : Neural population to plot: 'all' (default), 'exc' (excitatory), or 'inh' (inhibitory).
    e_prop : Proportion of excitatory neurons (default 0.8). Used to partition neurons when neuron_type='exc'/'inh'.
    vmin, vmax : Maximum and Minimum value for heatmap color scale.
    
    Returns
    -------
    aligned_activity : ndarray; Trial-aligned neural activity array.
    """

    win = np.array(win)*1000/dt
    win = win.astype(int)
    aligned_activity = align_data(win, activity_dict, trial_onsets)
    
    n_neurons = activity_dict[0].shape[1]
    if neuron_type == 'all':
        neuron_id = np.arange(n_neurons)
    elif neuron_type == 'exc':
        neuron_id = np.arange(int(n_neurons*e_prop))
    elif neuron_type == 'inh':
        neuron_id = np.arange(int(n_neurons*e_prop),n_neurons)

    trial_labels = {1: "ST-Lever", 2: "ST-Go", 3: "ST-No-Go", 4: "DT-Go", 5: "DT-No-Go"}
    trialAvg = {}
    for tt in trial_labels.keys():
        idx = np.where(trial_types == tt)[0]
        trialAvg[tt] = np.mean(aligned_activity[idx][:,:,neuron_id], axis=0) # time x neurons
    
    # ---- normalize per neuron across all conditions ----
    if normalize:
        stacked = np.concatenate([trialAvg[k] for k in [1,2,3,4,5]], axis=0) # (time*conditions) x neurons
        denom = np.max(stacked, axis=0) + 1e-6 
        for k in trialAvg:
            trialAvg[k] = trialAvg[k] / denom

    # ---- sorting ----
    if sort_task is not None:
        if sort_task == 0:
            sort_idx = None
        else:
            sort_idx = sort_activity(trialAvg[sort_task])[1]
        for k in trialAvg:
            trialAvg[k], _ = sort_activity(trialAvg[k], sort_idx)

    # ---- plotting ----
    t = np.arange(win[0], win[1]) * dt / 1000
    n_plot = neuron_id.size
    events = {"lever": [0, 5], "gng": [2, 3]}
    
    fig, ax = plt.subplots(1, 5, figsize=(10, 2.5), sharex=True, sharey=True)
    order = [1,2,3,4,5]
    for j, tt in enumerate(order):
        ax[j].imshow(trialAvg[tt].T,extent=[t[0], t[-1], 0, n_plot], aspect="auto", vmin=vmin, vmax=vmax, origin="lower")
        ax[j].set_title(trial_labels[tt])

        # event markers
        if tt in (1,4,5):  # lever involved
            for x in events.get("lever", []):
                ax[j].axvline(x, linestyle="--", color="w")
        if tt in (2,3,4,5):  # gng involved
            for x in events.get("gng", []):
                ax[j].axvline(x, linestyle="--", color="w")

    ax[0].set_ylabel("Trial-averaged activity")
    fig.tight_layout()

    return fig, ax, aligned_activity


def perf_trials(outputs, trial_onsets, trial_types, dt, motor_thr=0.3, go_thr=0.7, nogo_thr=0.3):
    """
    Evaluate task performance during specific task phases during dual-task training.
    Parameters
    ----------
    outputs : torch.Tensor; Neural network outputs with shape (Time, Batch, N_output)
    trial_onsets : Start times for each trial
    trial_types : Trial type labels for each trial. 1: ST-Lever, 2: ST-Go, 3: ST-No-Go, 4: DT-Go, 5: DT-No-Go
    dt : Time step in milliseconds
    motor_thr : optional; threshold for evaluating whether the Lever is held stably (default: 0.3)
    go_thr : optional; threshold for evaluating whether any licking event happens (default: 0.7)
    nogo_thr : optional; threshold for evaluating whether licking behavior was successfully withheld (default: 0.3)
    
    Returns
    -------
    Performance percentages (0-100 or NaN) for:
    1. Single-task motor success rate
    2. Single-task cognitive (go+no-go) accuracy
    3. Single-task go accuracy
    4. Single-task no-go accuracy
    5. Dual-task motor success rate
    6. Dual-task cognitive (go+no-go) accuracy
    7. Dual-task go accuracy
    8. Dual-task no-go accuracy
    """

    T_cue, T_lick, T_lever, T_delay, T_btw, T_tol = 1000, 200, 4500, 500, 1500, 500
    s = lambda ms: int(round(ms / dt)) # convert timing to time steps
    s_cue, s_lick, s_lever, s_delay, s_btw, s_tol = map(s, [T_cue, T_lick, T_lever, T_delay, T_btw, T_tol])

    T, B, _ = outputs.shape

    def evaluate_win(b, start, end, ch):
        if start < 0 or end > T or end <= start:
            return None
        return outputs[start:end, b, ch]

    st_motorSuc = st_hit = st_cr = 0
    dt_motorSuc = dt_hit = dt_cr = 0
    
    for b in range(B):
        trial_onset = int(trial_onsets[b])
        trial_type = int(trial_types[b])

        motor = evaluate_win(b, trial_onset+s_delay+s_btw, trial_onset+s_delay+s_lever-s_tol, 0)
        cog   = evaluate_win(b, trial_onset+s_delay+s_btw+s_cue, trial_onset+s_delay+s_btw+s_cue+s_lick, 1)

        if trial_type == 1 and motor is not None and torch.all(motor > motor_thr):
            st_motorSuc += 1
        elif trial_type == 2 and cog is not None and torch.any(cog > go_thr):
            st_hit += 1
        elif trial_type == 3 and cog is not None and torch.all(cog < nogo_thr):
            st_cr += 1
        elif trial_type in (4, 5):
            if motor is not None and torch.all(motor > motor_thr):
                dt_motorSuc += 1
            if trial_type == 4 and cog is not None and torch.any(cog > go_thr):
                dt_hit += 1
            if trial_type == 5 and cog is not None and torch.all(cog < nogo_thr):
                dt_cr += 1

    # Calculate number of trials for each condition
    st_motor = int((trial_types == 1).sum())
    
    st_go = int((trial_types == 2).sum())
    st_nogo = int((trial_types == 3).sum())
    st_cog   = st_go + st_nogo

    dt_go = int((trial_types == 4).sum())
    dt_nogo = int((trial_types == 5).sum())
    dt_total = dt_go + dt_nogo

    def pct(num, denom):
        return float("nan") if denom == 0 else 100.0 * num / denom

    return (pct(st_motorSuc, st_motor),
            pct(st_hit+st_cr, st_cog),
            pct(st_hit, st_go),
            pct(st_cr, st_nogo),
            pct(dt_motorSuc, dt_total),
            pct(dt_hit+dt_cr, dt_total),
            pct(dt_hit, dt_go),
            pct(dt_cr, dt_nogo))