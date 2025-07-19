from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np

from itertools import product

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        self.filter_avail_by_objects = args.filter_avail_by_objects
        self.action_grid = args.action_grid

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, env_info=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # Constrain targets to objects
        if (self.filter_avail_by_objects is True) or (self.filter_avail_by_objects=="test" and test_mode):
            avail_actions = self._filter_avail(avail_actions, ep_batch["obs"][:, t_ep])
        elif not self.action_grid:
            avail_actions = self._filter_adjacent(avail_actions)
        agent_outputs, target_updates, env_info = self.forward(ep_batch, t_ep, test_mode=test_mode, env_info=env_info)
        for k,v in env_info.items():
            if v is not None:
                try:
                    env_info[k] = [v[i] for i in bs]
                except:
                    pass
        chosen_actions, target_updates = self.action_selector.select_action(agent_outputs[bs], target_updates[bs], avail_actions[bs], t_env, test_mode=test_mode, env_info=env_info)
        return chosen_actions, target_updates

    def forward(self, ep_batch, t, test_mode=False, env_info=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        # Constrain targets to objects
        if self.filter_avail_by_objects:
            avail_actions = self._filter_avail(avail_actions, ep_batch["obs"][:, t])
        agent_outs, self.hidden_states, target_updates, env_info = self.agent(agent_inputs, self.hidden_states, env_info)
        # agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, env_info)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                try:
                    agent_outs[reshaped_avail_actions == 0] = -1e10
                except:
                    raise ValueError()
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
            target_updates.view(ep_batch.batch_size, self.n_agents, -1), \
            env_info

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        assert not self.args.obs_last_action, "obs_last_action not supported for non-flat observations"
        assert not self.args.obs_agent_id, "obs_agent_id not supported for non-flat observations"

        bs = batch.batch_size
        if isinstance(self.input_shape, tuple):
            inputs = batch["obs"][:, t].reshape(bs*self.n_agents, *self.input_shape)
        else:
            inputs = batch["obs"][:, t].reshape(bs*self.n_agents, self.input_shape)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _filter_avail(self, avail_actions, obs):
        if not self.filter_avail_by_objects:
            return avail_actions
        
        obj_channels = [i for i in range(1,4)] # assuming max_obj_lvl=3 and one_hot_obj_lvl=True
        obj = obs[:,:,obj_channels]
        obj_flag = obj[:,:,0]
        for i in range(1,obj.shape[2]): # loop across all obj channels
            obj_flag = th.logical_or(obj_flag, obj[:,:,i])
        obj_flag_sum = th.sum( obj_flag, dim=tuple(range(2,obj_flag.dim())) )
        for i,j in product(range(obj.shape[0]), range(obj.shape[1])):
            # Only apply constraint if at least one object is in view
            if obj_flag_sum[i,j]>0:
                avail_actions[i,j] = th.logical_and(avail_actions[i,j], obj_flag[i,j].flatten())
        return avail_actions
    
    def _filter_adjacent(self, avail_actions):
        if self.action_grid:
            return avail_actions
        
        adjacent = [(0,1),(1,0),(0,-1),(-1,0),(0,0),(1,1),(-1,-1),(1,-1),(-1,1)]
        adjacent = [self.agent._dif_to_flat(x) for x in adjacent]
        not_adjacent = [i for i in range(avail_actions.shape[-1]) if i not in adjacent]

        avail_actions[:,:,not_adjacent] = 0
        avail_actions[:,:,adjacent] = 1

        return avail_actions
