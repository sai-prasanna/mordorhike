import collections
import re
import warnings
from collections import defaultdict
from functools import partial as bind

import gymnasium as gym
import jax
import numpy as np
import recall2imagine
import ruamel.yaml as yaml
from recall2imagine import embodied
from recall2imagine.embodied.core.logger import AsyncOutput, _encode_gif
from recall2imagine.train import make_env as make_env_r2i
from recall2imagine.train import make_replay

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


def train(agent, env, replay, logger, args, config):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_report = embodied.when.Every(args.eval_every)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Every(args.log_every)
    should_save = embodied.when.Every(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    should_profile = args.profile_path != "none"
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        discount = 1 - 1 / config.horizon
        returns = np.cumsum(ep["reward"] * discount ** np.arange(len(ep["reward"])))
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "return": returns[-1],
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="epstats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    replay.maybe_restore()

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space)
    while len(replay) < max(args.batch_steps * config.envs.amount, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    if config.replay == "lfs":
        dataset = agent.dataset(replay, shared_memory=True)
    elif config.replay == "uniform":
        dataset = agent.dataset(replay.dataset, shared_memory=False)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            if should_profile:
                jax.profiler.start_trace(f"{args.profile_path}/{step.value}")
                print(f"profiling step {step}")
            outs, state[0], mets = agent.train(batch[0], state[0])
            if should_profile:
                jax.profiler.stop_trace()
            metrics.add(mets, prefix="train")
            if "priority" in outs:
                replay.prioritize(outs["key"], outs["priority"])
            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            # TODO: do this rarely
            if should_report(step):
                logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "last.ckpt", parallel=False)
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print("Start training loop.")
    policy = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
            timestamped_path = logdir / f"checkpoint_{step.value:08d}.ckpt"
            checkpoint.save(timestamped_path)
    logger.write()


def make_env(config, env_id=0, **kwargs):

    from recall2imagine.embodied.envs.from_gymnasium import FromGymnasium

    from powm.envs.mordor import MordorHike

    suite, task = config.task.split("_", 1)
    if suite == "mordorhike":
        task2cls = {
            "medium": MordorHike.medium,
            "hard": MordorHike.hard,
            "easy": MordorHike.easy,
        }
        env = task2cls[task](render_mode="rgb_array", **kwargs)
        seed = hash((config.seed, env_id)) % (2**32 - 1)
        env.reset(seed=seed)
        env = RenderWrapper(env, render_key="log_image", obs_key="vector")
        env = FromGymnasium(env)
        env = recall2imagine.wrap_env(env, config)
        return env
    else:
        return make_env_r2i(config, **kwargs)


def make_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda i=index: make_env(config, i, **overrides)
        if config.envs.parallel != "none":
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(embodied.wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != "none"))


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
        logdir.mkdir()

    loggers = [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, f"metrics.jsonl"),
        embodied.logger.TensorBoardOutput(logdir),
        VideoOutput(logdir),
    ]
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
                    dir=logdir,
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
        (embodied.Path(__file__).parent / "r2i_configs.yaml").read()
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

    step = embodied.Counter()
    replay = make_replay(config, replay_dir)
    env = make_envs(config)
    logger = make_logger(logdir, step, config)
    agent = recall2imagine.Agent(env.obs_space, env.act_space, step, config)
    replay.set_agent(agent)
    train(agent, env, replay, logger, args, config)
    env.close()


if __name__ == "__main__":
    # main(
    #     argv="--logdir test --configs mordorhike --run.steps 1e5 --run.save_every 1e5 --wandb.project ''".split()
    # )
    main()
