import warnings
from functools import partial as bind

import dreamerv3
from dreamerv3 import embodied

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def main():

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(
        {
            **dreamerv3.Agent.configs["size12m"],
            "dyn.rssm.deter": 32,
            "dyn.rssm.hidden": 16,
            "dyn.rssm.stoch": 32,
            "dyn.rssm.classes": 32,
            "dyn.rssm.blocks": 4,
            ".*\.layers": 2,
            ".*\.units": 32,
            "logdir": f"~/logdir/{embodied.timestamp()}-example",
            "enc.spaces": "vector",
            "dec.spaces": "vector",
            "run.train_ratio": 32,
            "run.steps": 500000,
            "task": "mordor_hike",
            "contdisc": False,
            "horizon": 100,
        }
    )
    config = embodied.Flags(config).parse()

    print("Logdir:", config.logdir)
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / "config.yaml")

    def make_agent(config):
        env = make_env(config)
        agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
        env.close()
        return agent

    def make_logger(config):
        logdir = embodied.Path(config.logdir)
        return embodied.Logger(
            embodied.Counter(),
            [
                embodied.logger.TerminalOutput(config.filter),
                embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
                embodied.logger.TensorBoardOutput(logdir),
                # embodied.logger.WandBOutput(wandb_init_kwargs={
                #     'project': 'dreamerv3-compat',
                #     'name': logdir.name,
                #     'config': dict(config),
                # }),
            ],
        )

    def make_replay(config):
        return embodied.replay.Replay(
            length=config.batch_length,
            capacity=config.replay.size,
            directory=embodied.Path(config.logdir) / "replay",
            online=config.replay.online,
        )

    def make_env(config, env_id=0):
        from dreamerv3.embodied.envs.from_gymnasium import FromGymnasium

        from powm.envs.mordor import MordorHike

        env = MordorHike.medium(render_mode="rgb_array")
        env = FromGymnasium(env, obs_key="vector")
        env = dreamerv3.wrap_env(env, config)
        return env

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_context=config.replay_context,
    )

    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config),
        bind(make_logger, config),
        args,
    )


if __name__ == "__main__":
    main()
