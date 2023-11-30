from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.offline.estimators import (
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel

config = (
    DQNConfig()
    .environment(env="CartPole-v1")
    .framework("torch")
    .offline_data(input_="/tmp/cartpole-out")
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=10,
        evaluation_num_workers=1,
        evaluation_duration_unit="episodes",
        evaluation_config={"input": "/tmp/cartpole-eval"},
        off_policy_estimation_methods={
            "is": {"type": ImportanceSampling},
            "wis": {"type": WeightedImportanceSampling},
            "dm_fqe": {
                "type": DirectMethod,
                "q_model_config": {"type": FQETorchModel, "polyak_coef": 0.05},
            },
            "dr_fqe": {
                "type": DoublyRobust,
                "q_model_config": {"type": FQETorchModel, "polyak_coef": 0.05},
            },
        },
    )
)

algo = config.build()
for _ in range(100):
    algo.train()