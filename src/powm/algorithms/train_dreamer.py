import re
import warnings
from collections import defaultdict
from functools import partial as bind

import dreamerv3
import gymnasium as gym
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import embodied
from dreamerv3.embodied.core.logger import AsyncOutput, _encode_gif

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


class RenderWrapper(gym.ObservationWrapper):
    def __init__(self, env, render_key="image", obs_key="obs"):
        super().__init__(env)
        self.render_key = render_key
        self.obs_key = obs_key

        # Get the render shape dynamically
        render_shape = self.env.render().shape

        # Update observation space
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space.spaces[render_key] = gym.spaces.Box(
                0, 255, render_shape, dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    self.obs_key: self.observation_space,
                    render_key: gym.spaces.Box(0, 255, render_shape, dtype=np.uint8),
                }
            )

    def observation(self, obs):
        rendered_img = self.env.render()
        if isinstance(obs, dict):
            obs[self.render_key] = rendered_img
        else:
            obs = {self.obs_key: obs, self.render_key: rendered_img}
        return obs


def train(make_agent, make_replay, make_env, make_logger, args):

    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    logdir.mkdir()
    print("Logdir", logdir)
    step = logger.step
    usage = embodied.Usage(**args.usage)
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    policy_fps = embodied.FPS()
    train_fps = embodied.FPS()

    batch_steps = args.batch_size * (args.batch_length - args.replay_context)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Every(args.log_every)
    should_eval = embodied.when.Every(args.eval_every)
    should_save = embodied.when.Every(args.save_every)

    @embodied.timer.section("log_step")
    def log_step(tran, worker):

        episode = episodes[worker]
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 1, agg="sum")
        episode.add("rewards", tran["reward"], agg="stack")

        if tran["is_first"]:
            episode.reset()

        if worker < args.log_video_streams:
            for key in args.log_keys_video:
                if key in tran:
                    episode.add(f"policy_{key}", tran[key], agg="stack")
        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episode.add(key, value, agg="sum")
            if re.match(args.log_keys_avg, key):
                episode.add(key, value, agg="avg")
            if re.match(args.log_keys_max, key):
                episode.add(key, value, agg="max")

        if tran["is_last"]:
            result = episode.result()
            logger.add(
                {
                    "score": result.pop("score"),
                    "length": result.pop("length"),
                },
                prefix="episode",
            )
            rew = result.pop("rewards")
            if len(rew) > 1:
                result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.num_envs)]
    driver = embodied.Driver(fns, args.driver_parallel)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(log_step)

    dataset_train = iter(
        agent.dataset(bind(replay.dataset, args.batch_size, args.batch_length))
    )
    dataset_report = iter(
        agent.dataset(bind(replay.dataset, args.batch_size, args.batch_length_eval))
    )
    carry = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def train_step(tran, worker):
        if len(replay) < args.batch_size or step < args.train_fill:
            return
        for _ in range(should_train(step)):
            with embodied.timer.section("dataset_next"):
                batch = next(dataset_train)
            outs, carry[0], mets = agent.train(batch, carry[0])
            train_fps.step(batch_steps)
            if "replay" in outs:
                replay.update(outs["replay"])
            agg.add(mets, prefix="train")

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "last.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

    print("Start training loop")
    policy = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    driver.reset(agent.init_policy)
    while step < args.steps:

        driver(policy, steps=10)

        if should_eval(step) and len(replay):
            mets, _ = agent.report(next(dataset_report), carry_report)
            logger.add(mets, prefix="report")

        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix="epstats")
            logger.add(embodied.timer.stats(), prefix="timer")
            logger.add(replay.stats(), prefix="replay")
            logger.add(usage.stats(), prefix="usage")
            logger.add({"fps/policy": policy_fps.result()})
            logger.add({"fps/train": train_fps.result()})
            logger.write()

        if should_save(step):
            checkpoint.save()
            timestamped_path = logdir / f"checkpoint_{step.value:08d}.ckpt"
            checkpoint.save(timestamped_path)

    checkpoint._worker.shutdown()
    logger.close()
    driver.close()


def make_env(config, env_id=0, **kwargs):

    from dreamerv3.embodied.envs.from_gymnasium import FromGymnasium

    from powm.envs.mordor import MordorHike

    suite, task = config.task.split("_", 1)
    if suite == "mordorhike":
        task2cls = {
            "medium": MordorHike.medium,
            "hard": MordorHike.hard,
            "easy": MordorHike.easy,
        }
        env = task2cls[task](render_mode="rgb_array", **kwargs)
    else:
        raise ValueError(f"Unknown suite: {suite}")

    seed = hash((config.seed, env_id)) % (2**32 - 1)
    env.reset(seed=seed)
    env = RenderWrapper(env, render_key="log_image", obs_key="vector")
    env = FromGymnasium(env)
    env = dreamerv3.wrap_env(env, config)
    return env


def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / "replay",
        online=config.replay.online,
    )


def make_agent(config):
    env = make_env(config, estimate_belief=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent


def make_logger(config, metric_dir=""):
    config_logdir = embodied.Path(config.logdir)
    logdir = config_logdir
    if metric_dir:
        logdir = logdir / metric_dir
        logdir.mkdir()

    loggers = [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, f"metrics.jsonl"),
        embodied.logger.TensorBoardOutput(logdir),
        VideoOutput(logdir),
    ]
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
                    dir=logdir,
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
        (embodied.Path(__file__).parent / "dreamer_configs.yaml").read()
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

    train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config),
        bind(make_logger, config),
        args,
    )


if __name__ == "__main__":
    # main(argv=["--logdir", f"~/logdir/{embodied.timestamp()}-example"])
    main()
