import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}

class BaseSelector:
    def __init__(self, args):
        self.args = args

    def _target_updates(self, agent_inputs, target_updates, picked_actions, env_info=None):
        prev_actions = self._build_prev_actions(env_info, agent_inputs)
        # prev_actions = self._build_prev_actions(env_info, agent_inputs).to(picked_actions) # for all possible comms
        # n_comms = prev_actions.shape[-1]
        # # picked_actions = picked_actions.unsqueeze(-1).repeat(1,1,n_comms)

        # temp = th.clone(target_updates.squeeze(-1)).to(picked_actions)
        # for prev in th.split(prev_actions, 1, dim=-1):
        #     # separate prev actions for each possible comms
        #     temp = th.logical_and(temp, picked_actions != prev)
        temp = picked_actions != prev_actions.to(picked_actions)
        target_updates[..., 0] = th.logical_and(temp, target_updates[...,0].to(picked_actions))
        target_updates[..., 1] = th.logical_and(temp, target_updates[...,1].to(picked_actions))
        return 1*target_updates
    
    def _build_prev_actions(self, env_info, agent_inputs):
        env_info = env_info.get("act_info", None)
        if env_info is None: return -1*th.ones_like(agent_inputs)
        return th.Tensor(env_info)


class MultinomialActionSelector(BaseSelector):
    def __init__(self, args):
        super().__init__(args)
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    # def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
    def select_action(self, agent_inputs, target_updates, avail_actions, t_env, test_mode=False, env_info=None):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions, self._target_updates(agent_inputs, target_updates, picked_actions, env_info)

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector(BaseSelector):
    def __init__(self, args):
        super().__init__(args)
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, target_updates, avail_actions, t_env, test_mode=False, env_info=None):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions, self._target_updates(agent_inputs, target_updates, picked_actions, env_info)

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector(BaseSelector):
    def select_action(self, agent_inputs, target_updates, avail_actions, t_env, test_mode=False, env_info=None):
        # probs = agent_inputs*avail_actions
        # m = Categorical(probs)
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions, self._target_updates(agent_inputs, target_updates, picked_actions, env_info)

REGISTRY["soft_policies"] = SoftPoliciesSelector


if __name__=="__main__":
    s = SoftPoliciesSelector([])