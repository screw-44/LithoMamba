# Release code review

Reviewed on 2026-10-07 against the published DATE 2026 paper, *LithoMamba: High-fidelity lithography simulation with State Space Models* (DOI: [10.23919/DATE69613.2026.11539705](https://doi.org/10.23919/DATE69613.2026.11539705)). The pre-maintenance main commit was `621709c72b6e7bf2a51282d94015e60177876b47`.

## Scope

The review covers the released training/inference entry points, paired data loader, adversarial loss, checkpoint helpers, environment, and the core model's correspondence to the paper. It does not reproduce the paper's metrics.

## Maintenance fixes

| Finding in the original release | Change |
|:--|:--|
| `train_mamba.py` imported the absent `model.pixpix_hd_new_loss` module; `test.py` also imported absent baseline modules. | Entry points import only the selected Mamba model, after argument parsing. Both `--help` commands work without CUDA extensions or data. |
| `test.py` selected Pix2PixHD instead of Mamba and ignored checkpoint directory/epoch arguments. | Inference uses `MambaGAN`, respects the checkpoint arguments, and runs under `torch.inference_mode()`. |
| `GANLoss` indexed a tensor as `input[-1]`, so only the last batch sample contributed to adversarial optimization. | Tensor predictions use the full batch. The loss is scoped to the released local discriminator's LSGAN objective. |
| Dataset length rounded down to a whole batch, silently losing samples and producing zero samples when the dataset was smaller than a batch. | Keep all pairs, including the final partial batch. The loader also applies `max_dataset_size` to the actual data. |
| Independent sorted lists could silently pair unrelated layout and SEM images. | Validate matching relative stems, duplicate keys and original dimensions before applying paired transforms. Existing data with different naming schemes must be renamed into matching pairs. |
| Crop/flip transforms repeatedly called `torch.seed()` and `torch.manual_seed()`, resetting the global RNG and tying horizontal/vertical flips together. | Sample shared spatial parameters for each pair, with independent flip decisions and no RNG reseeding. Inference preprocessing is deterministic. |
| Device parsing overrode explicit choices; scripts unconditionally selected CUDA while non-CUDA parsing selected MPS. | Respect device selection, default to GPU 0, and issue an explicit CUDA requirement for full Mamba execution. CPU remains suitable for helper regression tests. |
| A missing VMamba initialization checkpoint was always loaded, including inference. | Initialization is optional via `--pretrained_path`, used only for fresh training. Trained inference and warm starts load the requested generator checkpoint. |
| Only the generator forward pass used autocast; the following discriminator could receive half-precision data with full-precision convolution weights. The AMP scaler was recreated every epoch. | Autocast covers discriminator/generator loss computations as well as synthesis. One scaler persists across epochs; `--no-fp16` works. |
| Saving moved the live model to CPU and then blindly back to CUDA/MPS; loading lacked portable device mapping. | Save detached CPU state tensors without moving the live network; load with CPU mapping and `weights_only=True`. |
| Training ran at module import and hard-coded its epoch budget. A standalone debug block stepped the wrong optimizer for the discriminator. | Add a `main` guard and `--epochs`; remove the outdated standalone debug loop. |
| The environment was an Apple-specific Conda export with a personal absolute prefix and omitted required Mamba dependencies. | Supply a minimal reference environment and explicit CUDA extension installation instructions. Installation on CUDA has not been validated here. |

These fixes affect future optimization and data processing, so new runs should not be presented as identical to historical experiments. Existing model parameter names and the selective-scan network architecture are retained.

## Cleanup of unused code

The dependency chain from `train_mamba.py` and `test.py` was traced through the model, loader, options and checkpoint helpers. The cleaned release contains 17 Python files. Removed code comprised:

- 37 unreferenced Python files: 2D/3D segmentation networks and factories, alternate discriminators, Fourier/DOINN/transformer baseline definitions, the CLIP defect example, and one-off scripts with hard-coded dataset paths.
- Unused UNet, patch/multi-scale/vanilla discriminator classes, unused cosine/KL losses, image-conversion utilities, unused imports and commented-out alternatives.
- Inactive scan variants, including a branch referencing an undefined CUDA function, plus standalone debug and FLOP-analysis helpers.
- 46 command-line flags inherited from other models that the release entry points never read. Retained flags keep their previous defaults. Previously ignored flags such as `--netG`, `--num_D` and `--use_sigmoid` now produce argument errors; remove them from old commands and consult `--help`.
- SciPy and fvcore from the dependency environment; neither is needed by the retained pipeline.

The active VSSM computation and initialization methods are unchanged. Legacy initialization calls that consume the random number stream are retained to preserve seeded behavior. The generator/discriminator checkpoint parameter names and tensor shapes remain compatible with the preceding release.

## Paper/code differences requiring the final experiment configuration

| Topic | Paper | Released model / retained behavior |
|:--|:--|:--|
| Generator reconstruction loss | `L_adv + 0.1 L1` (Section III-B) | Historical defaults are `lambda_l1=1` plus an additional edge-region L1 term with weight 1. Both weights are now configurable. README's example uses `--lambda_l1 0.1 --lambda_edge 0`. |
| Local discriminator output | Sigmoid follows the convolutional stack (Section II-B). | `NLayerDiscriminator` is instantiated without sigmoid, and `GANLoss` uses MSE/LSGAN. Activation/loss behavior is not changed automatically. |
| Generator bottleneck depth | Figure 1 labels four bottleneck VSS blocks. | `VSSM` defaults to `depths=[2,2,9,2]`; the wrapper passes no alternate depths. Confirm the trained experiment configuration before changing network shapes or checkpoint compatibility. |
| Four-direction scan fusion | Figure 1 and Section II-A describe averaging four scan outputs. | `forward_corev0` sums four aligned outputs before LayerNorm. This is not written identically to the paper; the following normalization may absorb most of the scale change. |
| Legacy edge threshold | No extra edge term in the stated composite objective. | The historical `240/255` threshold is applied to tensors normalized to `[-1,1]`. It is not equivalent to a raw grayscale intensity threshold of 240/255. |
| Training hardware | RTX 4090 (Section III-B). | The old README said RTX 3090. Documentation now cites the published paper. |
| Pretraining and schedule | REFICS experiments are described as trained from scratch; the full schedule is not specified. | Initialization is optional. The retained 500-epoch default comes from the original script and is not a verified final-paper schedule. |

A confirmed final-paper configuration is needed to settle these differences. The published metrics are reference results, not a guarantee for the current default settings.

## Remaining release gaps

- **Data:** neither the private 14 nm dataset nor prepared REFICS pairs are bundled. Training and the current paired inference loader require appropriately matched data.
- **Checkpoints:** no trained generator or VMamba initialization weights are included. Ask the author about availability.
- **Evaluation:** the complete UISS segmentation and IoU/PA/F1/FID/PSNR/SSIM evaluation pipeline is absent. The inference script saves images only.
- **Baselines:** comparison models and their training/evaluation pipelines are not bundled in this release.
- **Warm starts:** saved checkpoints contain network weights and an epoch marker, not optimizer/scaler/RNG state. Continuing training is not an exact resumption.
- **Dependencies:** selective-scan kernels require a compatible NVIDIA CUDA installation. Unused SciPy and fvcore dependencies have been removed from the reference environment.
- **Repository hygiene:** tracked `.idea` settings, `.DS_Store` metadata, Python bytecode caches and the maintenance `tests/` directory have been removed. `.gitignore` excludes these local artifacts from future commits.
- **Attribution:** upstream Mamba/VMamba/Mamba-UNet work is acknowledged. Check upstream notices before redistributing imported components independently.

## Validation performed

- Full-batch GAN loss value and gradients.
- Filename pairing validation, shared paired augmentation, reproducible RNG behavior and deterministic inference transforms.
- Small datasets, the final partial batch and dataset limits.
- CPU checkpoint save/load round trip without changing the live model's device.
- Both command-line `--help` entry points without importing CUDA scan kernels.
- Python syntax compilation, Git whitespace checks and README relative-path checks.

Regression checks ran during maintenance on macOS CPU with Python 3.12, PyTorch 2.5.0 and torchvision 0.20.0; their test files are not included in the public repository. **No end-to-end CUDA model forward/backward pass, GPU extension installation, dataset training or paper-metric reproduction was performed.**

The subsequent cleanup was checked against commit `72f1c3e3c0c29633ebd7a05501c1817199b6c60d`: all remaining Python files compile and local import targets resolve; both `--help` commands and README examples parse; retained option defaults match. A reduced four-stage model using a CPU reference selective-scan recurrence produced identical initialized state tensors, forward outputs, generator/discriminator losses, gradients and one-step Adam updates before and after cleanup. Both legacy VMamba initialization formats and checkpoint save/load also passed. This checks cleanup equivalence on the reference path; it does not validate the CUDA kernels or full-size training.
