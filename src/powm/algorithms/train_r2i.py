import collections
import re
import warnings

import numpy as np
import ruamel.yaml as yaml
from recall2imagine.agent import Agent, embodied
from recall2imagine.embodied.core.logger import AsyncOutput, _encode_gif
from recall2imagine.embodied.run.train import train
from recall2imagine.train import make_envs, make_replay

from powm.utils import set_seed

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")



class WandBOutput:

    def __init__(self, pattern=r".*", wandb_init_kwargs: dict = {}):
        self._pattern = re.compile(pattern)
        import wandb

        wandb.init(**wandb_init_kwargs)
        self._wandb = wandb

    def __call__(self, summaries):
        bystep = collections.defaultdict(dict)
        wandb = self._wandb
        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue
            if isinstance(value, str):
                bystep[step][name] = value
            elif len(value.shape) == 0:
                bystep[step][name] = float(value)
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)
            elif len(value.shape) in (2, 3):
                value = value[..., None] if len(value.shape) == 2 else value
                assert value.shape[3] in [1, 3, 4], value.shape
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 4:
                assert value.shape[3] in [1, 3, 4], value.shape
                value = np.transpose(value, [0, 3, 1, 2])
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                bystep[step][name] = wandb.Video(value)

        for step, metrics in bystep.items():
            self._wandb.log(metrics, step=step)


def make_logger(logdir, step, config, metric_dir=None):
    config_logdir = embodied.Path(config.logdir)
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    logdir = config_logdir
    if metric_dir:
        logdir = logdir / metric_dir
        logdir.mkdirs()

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
                WandBOutput(
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
    return embodied.Logger(step, loggers, multiplier)


class VideoOutput(AsyncOutput):

    def __init__(self, logdir, fps=20, parallel=True):
        super().__init__(self._write, parallel)
        self._logdir = logdir / "videos"
        self._logdir.mkdirs()
        self._fps = fps

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
        (embodied.Path(__file__).parent / "configs/r2i.yaml").read()
    )
    parsed, other = embodied.Flags(configs=["defaults", "mordorhike"]).parse_known(argv)
    config = embodied.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        replay_dir=config.replay_dir,
        checkpoint_dir=config.checkpoint_dir,
        batch_steps=config.batch_size * config.batch_length,
    )
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    if args.replay_dir == "none":
        replay_dir = logdir / "replay"
    else:
        replay_dir = embodied.Path(args.replay_dir)
    replay_dir.mkdirs()
    set_seed(config.seed)
    step = embodied.Counter()
    replay = make_replay(config, replay_dir)
    env = make_envs(config)
    logger = make_logger(logdir, step, config)
    agent = Agent(env.obs_space, env.act_space, step, config)
    replay.set_agent(agent)
    train(agent, env, replay, logger, args, config)
    env.close()


if __name__ == "__main__":
    # main(
    #     argv="--logdir test --configs mordorhike --run.steps 1e5 --run.save_every 1e5 --wandb.project ''".split()
    # )
    main()
