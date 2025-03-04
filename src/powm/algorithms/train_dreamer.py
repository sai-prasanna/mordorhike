import warnings
from functools import partial as bind

import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied
from dreamerv3.embodied.core.logger import AsyncOutput, _encode_gif
from dreamerv3.embodied.run.train import train
from dreamerv3.main import make_agent, make_env, make_replay

import powm.envs
from powm.utils import set_seed

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def make_logger(config, metric_dir=""):
    config_logdir = embodied.Path(config.logdir)
    logdir = config_logdir
    if metric_dir:
        logdir = logdir / metric_dir
        logdir.mkdir()

    # Always include terminal output
    loggers = [embodied.logger.TerminalOutput(config.filter)]
    
    # Only add other loggers if write_logs is True
    if config.write_logs:
        loggers.extend([
            embodied.logger.JSONLOutput(logdir, f"metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            VideoOutput(logdir),
        ])
        # Only add wandb if project is specified and write_logs is True
        if config.wandb.project:
            loggers.append(
                embodied.logger.WandBOutput(
                    wandb_init_kwargs=dict(
                        project=config.wandb.project,
                        group=config.wandb.group or config_logdir.parent.name,
                        name=(
                            config.wandb.name
                            or (
                                config_logdir.name + f"_{metric_dir}"
                                if metric_dir
                                else config_logdir.name
                            )
                        ),
                        config=dict(config),
                        resume=True,
                        dir=str(logdir),
                    )
                ),
            )
    return embodied.Logger(
        embodied.Counter(),
        loggers,
    )


class VideoOutput(AsyncOutput):

    def __init__(self, logdir, fps=20, parallel=True):
        super().__init__(self._write, parallel)
        self._logdir = logdir / "videos"
        self._logdir.mkdir()
        self._fps = fps

    @embodied.timer.section("video_write")
    def _write(self, summaries):
        for step, name, value in summaries:
            try:
                if isinstance(value, np.ndarray) and len(value.shape) == 4:
                    gif_bytes = _encode_gif(value, self._fps)
                    with open(
                        self._logdir / f"{step:06d}_{'_'.join(name.split('/'))}.gif",
                        "wb",
                    ) as f:
                        f.write(gif_bytes)
            except Exception as e:
                print("Error writing summary:", name, e)
                raise

    def _encode_video(self, frames, fps):
        from subprocess import PIPE, Popen

        h, w, c = frames[0].shape
        pxfmt = {1: "gray", 3: "rgb24"}[c]
        cmd = " ".join(
            [
                "ffmpeg -y -f rawvideo -vcodec rawvideo",
                f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
                "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
                f"-r {fps:.02f} -f mp4 -",
            ]
        )
        proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        for image in frames:
            proc.stdin.write(image.tobytes())
        out, err = proc.communicate()
        if proc.returncode:
            raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
        del proc
        return out


def main(argv=None):
    configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "configs/dreamer.yaml").read()
    )
    parsed, other = embodied.Flags(configs=["defaults", "mordorhike"]).parse_known(argv)
    config = embodied.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = embodied.Flags(config).parse(other)
    config = config.update(
        logdir=config.logdir.format(timestamp=embodied.timestamp()),
        replay_length=config.replay_length or config.batch_length,
        replay_length_eval=config.replay_length_eval or config.batch_length_eval,
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_length=config.replay_length,
        replay_length_eval=config.replay_length_eval,
        replay_context=config.replay_context,
    )
    logdir = embodied.Path(args.logdir)
    logdir.mkdir()
    if (logdir / "config.yaml").exists():
        config = embodied.Config.load(logdir / "config.yaml")
        print("Loaded config from", logdir / "config.yaml")
    else:
        config.save(logdir / "config.yaml")
        print("Saved config to", logdir / "config.yaml")

    set_seed(config.seed)
    train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config, estimate_belief=False),
        bind(make_logger, config),
        args,
    )


if __name__ == "__main__":
    main()
