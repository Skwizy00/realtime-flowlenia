# Real-time Flow Lenia

GPU-accelerated real-time [Flow Lenia](https://arxiv.org/abs/2212.07906) simulation with liquid shader effects.

![Demo](demo.gif)

## Features

- **GPU acceleration** -- PyTorch MPS (Apple Silicon) / CUDA / CPU fallback
- **Liquid shader rendering** -- fluid distortion, thin-film interference, glow bloom
- **Time-varying parameter modulation** -- multi-harmonic breathing keeps patterns alive
- **Multiple render modes** -- monochrome, warm amber, full liquid shader, direct RGB
- **Interactive controls** -- pause, reset, pattern switching, mouse perturbation
- **Real-time performance** -- 512x512 at 30+ FPS on MPS, higher on CUDA
- **CoreML optimization** -- mixed-precision (fp16 growth + fp32 coords), 512 sim with 1024 display upscale, optional ANE rendering offload
- **Auto-evolving parameters** -- smooth interpolation between random parameter sets with configurable interval and duration

## Requirements

- Python 3.9+
- PyTorch 2.0+
- OpenCV
- NumPy

## Installation

```bash
git clone https://github.com/ochyai/realtime-flowlenia.git
cd realtime-flowlenia
pip install -r requirements.txt
```

## Usage

**GPU version (recommended):**

```bash
python realtime_flowlenia_gpu.py
```

**CoreML-optimized version** (macOS — best performance, auto-evolving patterns):

```bash
pip install coremltools  # optional, for ANE acceleration
python realtime_flowlenia_coreml.py
```

Simulates at 512x512 and upscales to 1024 display. Parameters auto-evolve every 20 seconds with smooth transitions.

**CPU fallback version** (uses the engine with reintegration tracking):

```bash
python realtime_flowlenia.py
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `r` | Reset with new random parameters |
| `q` / `ESC` | Quit |
| `s` | Save screenshot (PNG) |
| `f` | Toggle fullscreen |
| `c` | Cycle colormap / render mode |
| `+` / `-` | Increase / decrease sim steps per frame |
| `m` | Toggle time-varying modulation |
| `g` | Toggle glow effect |
| `t` | Toggle thin-film interference |
| `d` | Toggle fluid distortion |
| `e` | Toggle auto-evolution (CoreML version) |
| `b` | Run benchmark (CoreML version) |
| `1`-`5` | Switch initial pattern |
| Mouse click | Add perturbation |

## Architecture

- `realtime_flowlenia_gpu.py` -- Self-contained GPU implementation with `grid_sample` advection and liquid shader effects (main entry point)
- `realtime_flowlenia_coreml.py` -- CoreML-optimized version: 512 sim + 1024 display upscale, surgical mixed-precision (fp16 growth + fp32 coords), auto-evolving parameters, optional ANE rendering, `torch.compile` fusion
- `realtime_flowlenia_engine.py` -- Engine with proper reintegration tracking transport
- `realtime_flowlenia.py` -- CPU-friendly viewer using the engine
- `record_demo.py` -- Headless demo GIF recorder

## Citation

This implementation is based on Flow Lenia:

```bibtex
@article{plantec2023flowlenia,
  title={Flow Lenia: Mass conservation for the study of virtual creatures in continuous cellular automata},
  author={Plantec, Erwan and Hamon, Gautier and Etcheverry, Mayalen and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment and Chan, Bert Wang-Chak},
  journal={arXiv preprint arXiv:2212.07906},
  year={2023}
}
```

## License

MIT
