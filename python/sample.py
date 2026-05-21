# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math
from typing import Literal

# Third-party deps
import torch
from tqdm import tqdm
import numpy as np

# Local deps
import models
from utils import utils
from utils.thruster_data import ThrusterDataset
from noise import NoiseSampler

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, nargs="?")
parser.add_argument("config", type=str, nargs="?")
parser.add_argument("-o", "--out-dir", type=str)
parser.add_argument("-n", "--num-samples", type=int)
parser.add_argument("-b", "--batch-size", type=int)
parser.add_argument("-s", "--num-steps", type=int)
parser.add_argument("--test-dir", type=Path)

DEVICE = torch.device("cpu")

ODEMethod = Literal["euler", "heun", "midpoint", "dpmpp_2s", "dpmpp_2m"]

def build_observation(dataset, observations, param_vec=None, default_stddev=1.0):
    _, data_params, data_tensor = dataset[0]

    data_tensor = torch.tensor(data_tensor, device=DEVICE)
    (num_channels, resolution) = data_tensor.shape

    obs_matrix_loc = torch.zeros(num_channels, resolution, device=DEVICE)
    obs_matrix_dat = torch.zeros(num_channels, resolution, device=DEVICE)
    obs_matrix_var = torch.zeros(num_channels, resolution, device=DEVICE)

    default_stddev = observations.get("stddev", observations.get("std_dev", 1.0))

    grid = dataset.grid

    m = num_channels * resolution
    n = 0

    obs_fields = observations["fields"]

    for obs_field in obs_fields:
        # Get tensor row index
        row_index = dataset.fields()[obs_field]

        # Get stddev (TODO: read from file to allow more values, but this needs to wait for improved stddev handling)
        obs_dict = obs_fields[obs_field]
        stddev = obs_dict.get("stddev", obs_dict.get("std_dev", default_stddev))

        # Get observation from file
        x_inds, x_data, y_data = utils.get_observation_locs(
            obs_fields, obs_field, grid, normalizer=dataset.norm, form="normalized"
        )

        x_inds = np.unique(np.array(x_inds)).tolist()

        if (len(x_data) == resolution) and np.all(x_inds == np.arange(resolution)):
            # If x_data == grid, then we're observing an entire row
            print(obs_field + ":\tobserving entire row.")
            obs_matrix_loc[row_index, :] = 1.0
            obs_matrix_dat[row_index, :] = data_tensor[row_index, :]
            obs_matrix_var[row_index, :] = stddev**2
            n += resolution
        else:
            # Partial/sparse observation of the row
            # If y not provided, we use the underlying data matrix from the dataset
            # Otherwise we use the y found in the file
            # TODO: stddevs that vary point-to-point
            obs_matrix_loc[row_index, x_inds] = 1.0
            obs_matrix_var[row_index, x_inds] = stddev**2
            n += len(x_inds)

            if y_data is None:
                print(obs_field + ":\tusing data from ref sim at selected axial locs.")
                obs_matrix_dat[row_index, x_inds] = data_tensor[row_index, x_inds]
            else:
                print(obs_field + ":\tusing data from file.")
                obs_matrix_dat[row_index, x_inds] = torch.tensor(y_data, dtype=torch.float32, device=DEVICE)

    # Dimensions
    # m = num_channels * resolution
    # n = num_observations
    # A (linear observation operator) = (n, m)
    # y (observed data) = (n,)
    obs_matrix_loc = obs_matrix_loc.reshape(-1)
    obs_A = torch.zeros(n, m, device=DEVICE)

    j = 0
    for i in range(m):
        if obs_matrix_loc[i] == 1.0:
            obs_A[j, i] = 1.0
            j += 1

    obs_y = obs_A @ obs_matrix_dat.reshape(-1)
    obs_var = obs_A @ obs_matrix_var.reshape(-1)

    # If no param vec specified here, we use the one from the reference dataset
    if param_vec is None:
        param_vec = torch.tensor(data_params, device=DEVICE)

    # Read scalar parameters if present
    if (params := observations.get("params", None)) is not None:
        for p, i in dataset.params().items():
            if p in params:
                param_vec[:, i] = dataset.norm.normalize(params[p], p)

    return obs_A, obs_y, obs_var, param_vec


def edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent, num_refinement_steps=0):
    inv_rho = 1 / exponent
    i = torch.arange(0, num_steps)
    f1 = noise_max**inv_rho
    f2 = (noise_min**inv_rho - noise_max**inv_rho) / (num_steps - 2)
    timesteps = (f1 + i * f2) ** exponent
    timesteps[-1] = 0

    if num_refinement_steps > 0:
        timesteps = torch.concat((timesteps, torch.zeros(num_refinement_steps)))

    return timesteps

# =====================================================
# Conditioning on observations and PDEs
# =====================================================
def guidance_score(x_t, x_0, observation, proc_var, retain_graph=False):
    (batch_size, _, _) = x_0.shape

    obs_vec = observation["data"]
    var = observation["var"]
    H = observation["operator"]

    # =====================================================
    # Diffusion posterior sampling (get observation loss)
    # =====================================================
    x_vec = x_0.reshape(batch_size, -1)
    measurement = torch.matmul(H, x_vec.T).T
    total_var = var + proc_var
    obs_loss = torch.sum((measurement - obs_vec[None, ...]) ** 2 / total_var)
    score = -torch.autograd.grad(obs_loss, x_t, retain_graph=retain_graph)[0]

    return score

def reverse_step(
    denoiser,
    x_t,
    t_prev,
    t,
    observation,
    step_scale=1.0,
    method: ODEMethod="midpoint",
    model_args=dict(),
    prev_denoised=None,    # multistep cache: D_theta from the previous step
    t_prev_prev=None,      # multistep cache: t_prev used one step ago
):
    (b, _, _) = x_t.shape

    ones = torch.ones((b, 1, 1), device=x_t.device)

    dt = t - t_prev
    t_mid = 0.5 * (t + t_prev)

    use_const_guidance=True

    if use_const_guidance:
        proc_var = lambda t: t**2 / (t**2 + 1)
        t_max_guidance = 100
    else:
        min_var = torch.min(observation["var"])
        proc_var_scale = torch.sqrt(min_var) * 15
        proc_var = lambda t: proc_var_scale * t**2 / (t**2 + 1)
        t_max_guidance = 10

    x_t = x_t.detach()
    x_t.requires_grad = True
    denoiser.zero_grad()

    # =====================================================
    # DPM-Solver++(2S) branch — early-return implementation
    # =====================================================
    # Reference: Lu et al. 2022b, "DPM-Solver++: Fast Solver for Guided
    # Sampling of Diffusion Probabilistic Models" (arXiv:2211.01095).
    #
    # Reformulates the EDM probability-flow ODE in lambda = -log(sigma) space,
    # where it becomes a linear ODE solvable in closed form for piecewise-
    # constant or piecewise-linear data-prediction model D_theta. The single-
    # step second-order variant (2S) uses 2 NFE per step (matching midpoint)
    # but stays bounded as sigma->0, making it stable for high-strength
    # guided sampling.
    #
    # For EDM noise schedule (alpha=1, sigma=t) with r=1/2:
    #   beta  = sqrt(t/t_prev)  = exp(-h/2)            contraction in predictor
    #   gamma = t/t_prev        = exp(-h)              contraction in corrector
    #   t_s   = sqrt(t_prev*t)                         geometric midpoint of sigma
    #   u     = beta*x_t + (1-beta)*D_theta(x_t, t_prev)        predictor
    #   x_n   = gamma*x_t + (1-gamma)*D_theta(u, t_s)           corrector
    if method == "dpmpp_2s":
        # (2S) is derived for the unscaled probability-flow ODE (step_scale=1).
        # Other values would require a custom rederivation.
        assert step_scale == 1.0, (
            f"dpmpp_2s requires step_scale=1.0, got {step_scale}"
        )

        # First NFE: data prediction at the current state and time.
        x_denoised = denoiser(x_t, t_prev * ones, **model_args)

        # Static thresholding (Lu et al. 2022b, Sec. 4.2 / Imagen-style):
        # Clip the data prediction x_theta to the training-data range to
        # prevent the late-step contracting update from inheriting model
        # excursions outside physical scale. c = 2.0 = 4 * data_std, which
        # preserves ~99.99% of legitimate values under the trained N(0, 0.5)
        # marginal. Without this, the corrector x_pred = gamma*x_t + (1-gamma)*D
        # converges to whatever D the model predicts -- including out-of-range
        # values -- as gamma -> 0 at the final step.
        x_denoised = torch.clamp(x_denoised, min=-2.0, max=2.0)

        # Predictor: interpolate from x_t toward D_1 using lambda-space coeffs.
        beta = (t / t_prev) ** 0.5
        u = beta * x_t + (1 - beta) * x_denoised

        # Geometric midpoint of sigma is the midpoint in lambda. When t==0 (the
        # very last sampling step) this collapses to 0, which is a singularity
        # for the denoiser's log(sigma) preconditioning. Clamp only the *eval*
        # noise level; keep beta/gamma at their true values so the asymptotic
        # behavior x_n -> D_theta(u, ~0) is preserved.
        t_s = (t_prev * t) ** 0.5
        if isinstance(t_s, torch.Tensor):
            t_s_eval = torch.clamp(t_s, min=1e-3)
        else:
            t_s_eval = max(t_s, 1e-3)

        # Second NFE: data prediction at the predicted state and midpoint time.
        x_denoised = denoiser(u, t_s_eval * ones, **model_args)

        # Static thresholding -- same rationale as the D_1 clamp above. This
        # value flows into BOTH the corrector formula and the const_guidance
        # gradient below, so clamping here also bounds how far the guidance
        # correction can push x_pred per step. torch.clamp has well-defined
        # gradients (identity inside range, zero outside), so the autograd
        # chain for guidance_score() remains valid.
        x_denoised = torch.clamp(x_denoised, min=-2.0, max=2.0)

        # Corrector: interpolate from x_t toward D_2 using lambda-space coeffs.
        gamma = t / t_prev
        x_pred = gamma * x_t + (1 - gamma) * x_denoised

        # const_guidance correction -- identical handling to the midpoint path
        # below. x_denoised here is D_2; the gradient chain x_t -> D_1 -> u
        # -> D_2 is preserved by autograd, so guidance_score() works the same
        # way it does for midpoint.
        if use_const_guidance and observation["var"] is not None and t_mid < t_max_guidance:
            obs_score = guidance_score(x_t, x_denoised, observation, proc_var(t_mid))
            x_pred = x_pred + obs_score

        # (2S) doesn't need cross-step state; return None for the cache slot.
        return x_pred.detach(), None

    # =====================================================
    # DPM-Solver++(2M) branch -- multistep, early-return implementation
    # =====================================================
    # Reference: Lu et al. 2022b, Algorithm 2 (arXiv:2211.01095). Uses 1 NFE
    # per step PLUS a cached denoiser output from the previous step to reach
    # second-order accuracy at half the per-step compute of (2S) or midpoint.
    #
    # For EDM noise schedule (alpha=1, sigma=t), with h_i = log(t_{i-1}/t_i)
    # and r_i = h_{i-1}/h_i:
    #   D_extrap = (1 + 1/(2 r_i)) * D_curr - (1/(2 r_i)) * D_prev_cached
    #   x_new    = gamma * x_t + (1 - gamma) * D_extrap   (same corrector shape as 2S)
    #
    # Requires deterministic sampling (S_churn=0) for math correctness; with
    # stochastic injection, the cache from the previous step no longer lies on
    # the trajectory we are now on. Marks's default is S_churn=0 so this is
    # the expected operating regime.
    #
    # First-step fallback: no cache yet -> use first-order DPM-Solver++(1),
    # i.e. D_extrap = D_curr. Same fallback for the last step (t close to 0)
    # because h_curr -> infinity and the extrapolation coefficients blow up.
    if method == "dpmpp_2m":
        assert step_scale == 1.0, (
            f"dpmpp_2m requires step_scale=1.0, got {step_scale}"
        )

        # First (and only) NFE
        x_denoised = denoiser(x_t, t_prev * ones, **model_args)

        # Static thresholding (same as 2S branch above for consistency).
        x_denoised = torch.clamp(x_denoised, min=-2.0, max=2.0)

        gamma = t / t_prev

        # Decide whether we have enough history AND well-defined math for the
        # second-order extrapolation. Otherwise fall back to first-order.
        t_curr_f = float(t) if torch.is_tensor(t) else float(t)
        can_use_2m = (
            prev_denoised is not None
            and t_prev_prev is not None
            and t_curr_f > 1e-3
        )

        if can_use_2m:
            import math
            t_prev_f = float(t_prev) if torch.is_tensor(t_prev) else float(t_prev)
            t_prev_prev_f = (
                float(t_prev_prev) if torch.is_tensor(t_prev_prev)
                else float(t_prev_prev)
            )
            h_curr = math.log(t_prev_f / t_curr_f)
            h_prev = math.log(t_prev_prev_f / t_prev_f)
            r = h_prev / h_curr
            coef = 1.0 / (2.0 * r)
            # prev_denoised is already detached (set by reverse() from a prior
            # .detach()-ed return), so the autograd graph from x_t flows only
            # through the (1 + coef) * x_denoised term -- correct gradient
            # behavior for guidance_score() below.
            D_used = (1.0 + coef) * x_denoised - coef * prev_denoised
        else:
            # First step (no cache) or last step (t -> 0): first-order update.
            D_used = x_denoised

        # Corrector update (same shape as 2S, but D_used is an extrapolation
        # rather than a fresh denoiser eval at the midpoint).
        x_pred = gamma * x_t + (1 - gamma) * D_used

        # const_guidance correction -- identical handling to (2S) and midpoint.
        if use_const_guidance and observation["var"] is not None and t_mid < t_max_guidance:
            obs_score = guidance_score(x_t, D_used, observation, proc_var(t_mid))
            x_pred = x_pred + obs_score

        # Return current denoiser output for the next step's cache.
        return x_pred.detach(), x_denoised.detach()

    # Compute initial step to get predicted sample location
    x_denoised = denoiser(x_t, t_prev * ones, **model_args)
    deriv_1 = -step_scale * (x_denoised - x_t) / t_prev

    if not use_const_guidance and (observation["var"] is not None) and t_prev < t_max_guidance:
        obs_score = guidance_score(x_t, x_denoised, observation, proc_var(t_prev))
        deriv_1 += -t_prev * obs_score

    if method == "midpoint":
        step_1 = 0.5 * dt * deriv_1
    else:
        step_1 = dt * deriv_1

    x_pred = x_t + step_1

    if method == "midpoint" or (method == "heun" and t > 0):
        # Compute corrector step
        if not use_const_guidance:
            x_pred = x_pred.detach()
            x_pred.requires_grad = True
            denoiser.zero_grad()

        t2 = t_mid if method == "midpoint" else t
        x_denoised = denoiser(x_pred, t2 * ones, **model_args)
        deriv_2 = -step_scale * (x_denoised - x_t) / t2

        # Guidance loss
        if not use_const_guidance and (observation["var"] is not None) and t2 < t_max_guidance:
            obs_score = guidance_score(x_pred, x_denoised, observation, proc_var(t2))
            deriv_2 += -t2 * obs_score
        
        if method == "midpoint":
            step_2 = dt * deriv_2
        else:
            step_2 = dt * 0.5 * (deriv_1 + deriv_2)

        x_pred = x_t + step_2

    # Guidance loss
    if use_const_guidance and observation["var"] is not None and t_mid < t_max_guidance:
        obs_score = guidance_score(x_t, x_denoised, observation, proc_var(t_mid))
        x_pred += obs_score

    # midpoint/heun/euler don't need cross-step state; return None for the cache slot.
    return x_pred.detach(), None

def reverse(
    denoiser,
    x,
    timesteps,
    dataset,
    observation,
    showprogress=False,
    pde_args=dict(),
    model_args=dict(),
    S_churn=0.0,
    **kwargs,
):
    """
    Perform iterative denoising to generate a 1D image from Gaussian noise using a provided denoising model

    Args:
        denoiser: a Denoiser model
        x: a tensor containing standard Gaussian noise. The dimension of this tensor should be (b, c, w)
            where b is the batch size, c is the number of channels, and w is the width
        im_masks: An optimal tuple of (data, mask) to apply during generation. These should have the same shape as x.
            When provided, the model will infill areas where the mask is set to 0.
        showprogress: whether we should print a tqdm progress bar.
    Returns:
    """
    (b, c, w) = x.shape
    num_steps = len(timesteps)

    output = torch.zeros((num_steps, b, c, w))
    output[0, ...] = x

    # Multistep state: cached denoiser output and t_prev from the previous
    # iteration. Used only by method='dpmpp_2m'; other methods read None
    # and ignore them.
    prev_denoised = None
    prev_t_prev = None

    for step_idx, t in enumerate(pbar := tqdm(timesteps, disable=(not showprogress))):
        if step_idx == 0:
            continue

        # increase noise level somewhat
        t_prev = timesteps[step_idx - 1]
        gamma = np.minimum(S_churn / len(timesteps), np.sqrt(2) - 1)

        if t_prev == 0:
            t_new = 0.002
            noise_std = 0.002
        elif t_prev < 0.05:
            t_new = t_prev
            noise_std = 0.0
        else:
            t_new = (1 + gamma) * t_prev
            noise_std = (t_new**2 - t_prev**2).sqrt()

        noise = 1.003 * torch.randn_like(x) * noise_std

        pbar.set_description(f"Noise level: {t_new:.4f}, Gamma: {gamma:.4f}")

        x, current_denoised = reverse_step(
            denoiser,
            x + noise,
            t_new,
            t,
            observation,
            model_args=model_args,
            prev_denoised=prev_denoised,
            t_prev_prev=prev_t_prev,
            **kwargs,
        )

        # Update multistep state for the next iteration. For non-multistep
        # methods, current_denoised is None, so prev_denoised stays None and
        # any later (2M) call would correctly fall back to first-order on
        # the very first step.
        prev_denoised = current_denoised
        prev_t_prev = t_new  # the noise level the denoiser was actually called at

        # Check for NaN or Inf
        if not torch.all(torch.isfinite(x)):
            print("NaN/Inf detected during sampling. Exiting")
            exit(1)

        output[step_idx, ...] = x

    output[-1, ...] = x
    return output


def sample(model, noise_sampler, num_samples, args):
    # Load sampling arguments
    num_steps = args.get("num_steps", 256)
    noise_min = args.get("noise_min", 0.002)
    noise_max = args.get("noise_max", 80)
    exponent = args.get("step_exponent", 7)
    step_scale = args.get("step_scale", 1.0)
    method = args.get("method", "midpoint")
    S_churn = args.get("S_churn", 0)
    refinement_steps = args.get("refinement_steps", 0)

    # Determine if we're doing condional or unconditional sampling
    # If there is an `observation` field, then we're conditioning on a partial observation of that simulation
    # If not, we're sampling unconditionally
    # If we sample unconditonally, we need to get some scalar parameters to condition on
    # These are drawn from the same distributions as the training set
    if (uncond_dir := args.get("unconditional_data_dir", None)) is not None:
        unconditional_dataset = ThrusterDataset(uncond_dir)
        param_vec = unconditional_dataset.sample_params(num_samples=num_samples, device=DEVICE)
    else:
        unconditional_dataset = None
        param_vec = None

    print(args)
    if "observation" in args:
        obs_args = utils.read_observation(args["observation"])
        obs_file = Path(obs_args["base_sim"])

        # Load data for conditioning
        dataset = ThrusterDataset(obs_file)

        if (obs_params:= obs_args.get("params", None)) is not None:
            if set(obs_params) != set(dataset.params()) and param_vec is None:
                # We didn't completely specify the parameter vector and have nothing to fall back on
                raise RuntimeError("Incomplete parameter specification without data directory. Exiting.")

        elif "params" not in obs_args:
            # Use the parameter vector from the ref simulation
            param_vec = None

        obs_operator, obs_data, obs_var, param_vec = build_observation(dataset, obs_args, param_vec)
        obs = dict(operator=obs_operator, data=obs_data, var=obs_var)
    else:
        if param_vec is None or unconditional_dataset is None:
            raise RuntimeError("No observation specified and no data directory given. Exiting")

        dataset = unconditional_dataset
        obs = dict(operator=None, var=None, data=None)

    # Sample initial noise
    xt = noise_sampler.sample(num_samples) * noise_max

    # Load timesteps
    steps = edm_sampling_timesteps(num_steps, noise_min, noise_max, exponent, num_refinement_steps=refinement_steps)

    print(f"{param_vec.shape=}")

    output = reverse(
        model,
        xt,
        steps,
        dataset,
        showprogress=True,
        observation=obs,
        step_scale=step_scale,
        method=method,
        S_churn=S_churn,
        model_args=dict(condition_vector=param_vec),
        pde_args=args.get("pde_guidance", None),
    )

    final = output[-1, ...]

    # Save generated samples
    out_dir = Path(args["out_dir"])
    data_dir = out_dir / "data"

    if args.get("replace_samples", False) and data_dir.exists():
        shutil.rmtree(data_dir)

    # Make folder and write metadata
    os.makedirs(out_dir, exist_ok=True)
    dataset.write_metadata(out_dir)

    # Write final sample data to independent output dirs
    os.makedirs(data_dir, exist_ok=True)
    params_cpu = param_vec.cpu().numpy()
    for i in range(num_samples):
        file = data_dir / f"{uuid.uuid4()}.npz"
        tens = final[i, :].cpu().numpy()
        np.savez(file, data=tens, params=params_cpu)
        
    # Write samples at all iterations to a single tensor
    np.savez(out_dir / "data_allsteps.npz", steps=steps, data=output.cpu().numpy(), params=params_cpu)


if __name__ == "__main__":
    args = parser.parse_args()

    DEVICE = utils.get_device()

    # Load sampling configuration
    with open(args.config, "rb") as fp:
        sampling_config = tomllib.load(fp)

    # Read command line args and replace TOML args if needed
    if args.out_dir is not None:
        sampling_config["out_dir"] = args.out_dir

    if args.num_steps is not None:
        sampling_config["num_steps"] = args.num_steps

    if args.num_samples is not None:
        sampling_config["num_samples"] = args.num_samples

    if args.batch_size is not None:
        sampling_config["batch_size"] = args.batch_size

    # Load model and config from checkpoint
    model_dict = torch.load(args.model, weights_only=False, map_location=DEVICE)
    model_config = model_dict["model_config"]
    model = models.from_config(model_config, device=DEVICE)

    # Determine which weights to load
    model_type = sampling_config.get("model_type", "ema")
    assert model_type in ["ema", "best", "last"]
    model_type = "model" if model_type == "last" else model_type
    model.load_state_dict(model_dict[model_type])

    # Switch model to evalution mode and sample
    model.eval()

    num_samples = sampling_config.get("num_samples", 64)
    batch_size = sampling_config.get("batch_size", num_samples)

    full_batches = math.floor(num_samples / batch_size)
    remainder = num_samples - full_batches * batch_size
    batches = [batch_size for i in range(math.floor(num_samples / batch_size))]
    if remainder > 0:
        batches.append(remainder)

    # Create noise sampler and initial noise samples
    if "train_config" in model_dict and "noise_sampler" in model_dict["train_config"]:
        noise_sampler_args = model_dict["train_config"]["noise_sampler"]
    else:
        noise_sampler_args = dict(type="gaussian", scale=1.0)

    channels = model.img_channels
    resolution = model.img_resolution

    noise_sampler = NoiseSampler.from_config(model.img_channels, model.img_resolution, DEVICE, **noise_sampler_args)

    # Sample in batches
    for i, batch_num_samples in enumerate(batches):
        sample(model, noise_sampler, batch_num_samples, sampling_config)

        # Make sure we don't remove old samples
        sampling_config["replace_samples"] = False
