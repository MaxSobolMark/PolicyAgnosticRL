from .continuous.bc import BCAgent
from .continuous.calql import CalQLAgent
from .continuous.cql import ContinuousCQLAgent
from .continuous.ddpm_bc import DDPMBCAgent
from .continuous.iql import IQLAgent
from .continuous.parl_iql import PARLIQLAgent
from .continuous.parl_calql import PARLCalQLAgent
from .continuous.sac import SACAgent
from .continuous.diffusion_q_learning import DiffusionQLearningAgent
from .continuous.auto_regressive_transformer import AutoRegressiveTransformerAgent

agents = {
    "ddpm_bc": DDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "parl_iql": PARLIQLAgent,
    "parl_calql": PARLCalQLAgent,
    "cql": ContinuousCQLAgent,
    "calql": CalQLAgent,
    "diffusion_q_learning": DiffusionQLearningAgent,
    "sac": SACAgent,
    "auto_regressive_transformer": AutoRegressiveTransformerAgent,
}
