import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.dreamer import DreamerAgent, stop_gradient
from collections import OrderedDict
from dm_env import specs
import utils
import torchrl.init as init
import torchrl.base as base
import agent.dreamer_utils as common
from agent.skill_utils import *
import torchrl.emb as emb
import random


def null_activation(x):
    return x


class SoftModuleVAE(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, input_shape,
                 em_hidden_shapes,
                 hidden_shapes,

                 num_layers, num_modules,

                 module_hidden,

                 gating_hidden, num_gating_layers, temperature,

                 # gated_hidden
                 device,
                 add_bn=True,
                 pre_softmax=False,
                 cond_ob=True,
                 module_hidden_init_func=init.basic_init,
                 last_init_func=init.uniform_init,
                 activation_func=F.relu,
                 **kwargs):

        super().__init__()

        self.latent_dim = int(output_shape / 2)
        self.temperature = temperature

        self.base = emb.SimpleCNN()
        self.em_base = base.MLPBase(
            last_activation_func=null_activation,
            input_shape=em_input_shape,
            activation_func=activation_func,
            hidden_shapes=em_hidden_shapes,
            **kwargs)
        self.de_base = base.MLPBase(
            last_activation_func=null_activation,
            input_shape=self.latent_dim,
            activation_func=activation_func,
            hidden_shapes=em_hidden_shapes,
            **kwargs)

        self.activation_func = activation_func  # F.relu

        module_input_shape = self.base.output_shape
        self.encoder_layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        self.encoder_weights = []

        self.device = device

        for i in range(num_layers):
            layer_module = []
            for j in range(num_modules):
                fc = nn.Linear(module_input_shape, module_hidden).to(self.device)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    ).to(self.device)
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i, j), module)

            module_input_shape = module_hidden
            self.encoder_layer_modules.append(layer_module)

        self.encoder_last = nn.Linear(module_input_shape, output_shape).to(self.device)
        last_init_func(self.encoder_last)

        self.decoder_layer_modules = []
        module_input_shape = self.de_base.output_shape

        for i in range(num_layers):
            layer_module = []
            for j in range(num_modules):
                fc = nn.Linear(module_input_shape, module_hidden).to(self.device)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    ).to(self.device)
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i, j), module)

            module_input_shape = module_hidden
            self.decoder_layer_modules.append(layer_module)

        self.decoder_last = emb.DeconvNet()

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated"
        gating_input_shape = self.em_base.output_shape

        self.encoder_gating_fcs = []

        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden).to(self.device)
            module_hidden_init_func(gating_fc)
            self.encoder_gating_fcs.append(gating_fc)
            self.__setattr__("encoder_gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.encoder_gating_weight_fcs = []
        self.encoder_gating_weight_cond_fcs = []

        self.encoder_gating_weight_fc_0 = nn.Linear(gating_input_shape,
                                                    num_modules * num_modules).to(self.device)
        last_init_func(self.encoder_gating_weight_fc_0)

        for layer_idx in range(num_layers - 2):
            encoder_gating_weight_cond_fc = nn.Linear((layer_idx + 1) * \
                                                      num_modules * num_modules,
                                                      gating_input_shape).to(self.device)
            module_hidden_init_func(encoder_gating_weight_cond_fc)
            self.__setattr__("encoder_gating_weight_cond_fc_{}".format(layer_idx + 1),
                             encoder_gating_weight_cond_fc)
            self.encoder_gating_weight_cond_fcs.append(encoder_gating_weight_cond_fc)

            encoder_gating_weight_fc = nn.Linear(gating_input_shape,
                                                 num_modules * num_modules).to(self.device)
            last_init_func(encoder_gating_weight_fc)
            self.__setattr__("encoder_gating_weight_fc_{}".format(layer_idx + 1),
                             encoder_gating_weight_fc)
            self.encoder_gating_weight_fcs.append(encoder_gating_weight_fc)

        self.encoder_gating_weight_cond_last = nn.Linear((num_layers - 1) * \
                                                         num_modules * num_modules,
                                                         gating_input_shape).to(self.device)
        module_hidden_init_func(self.encoder_gating_weight_cond_last)

        self.encoder_gating_weight_last = nn.Linear(gating_input_shape, num_modules).to(self.device)
        last_init_func(self.encoder_gating_weight_last)

        assert self.em_base.output_shape == self.de_base.output_shape, \
            "embedding should has the same dimension with de_base output for gated"
        gating_input_shape = self.em_base.output_shape

        self.decoder_gating_fcs = []

        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden).to(self.device)
            module_hidden_init_func(gating_fc)
            self.decoder_gating_fcs.append(gating_fc)
            self.__setattr__("decoder_gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.decoder_gating_weight_fcs = []
        self.decoder_gating_weight_cond_fcs = []

        self.decoder_gating_weight_fc_0 = nn.Linear(gating_input_shape,
                                                    num_modules * num_modules).to(self.device)
        last_init_func(self.decoder_gating_weight_fc_0)

        for layer_idx in range(num_layers - 2):
            decoder_gating_weight_cond_fc = nn.Linear((layer_idx + 1) * \
                                                      num_modules * num_modules,
                                                      gating_input_shape).to(self.device)
            module_hidden_init_func(decoder_gating_weight_cond_fc)
            self.__setattr__("decoder_gating_weight_cond_fc_{}".format(layer_idx + 1),
                             decoder_gating_weight_cond_fc)
            self.decoder_gating_weight_cond_fcs.append(decoder_gating_weight_cond_fc)

            decoder_gating_weight_fc = nn.Linear(gating_input_shape,
                                                 num_modules * num_modules).to(self.device)
            last_init_func(decoder_gating_weight_fc)
            self.__setattr__("decoder_gating_weight_fc_{}".format(layer_idx + 1),
                             decoder_gating_weight_fc)
            self.decoder_gating_weight_fcs.append(decoder_gating_weight_fc)

        self.decoder_gating_weight_cond_last = nn.Linear((num_layers - 1) * \
                                                         num_modules * num_modules,
                                                         gating_input_shape).to(self.device)
        module_hidden_init_func(self.decoder_gating_weight_cond_last)

        self.decoder_gating_weight_last = nn.Linear(gating_input_shape, num_modules).to(self.device)
        last_init_func(self.decoder_gating_weight_last)

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

        self.encoder = self.encoder_func
        self.decoder = self.decoder_func
        self.mse_loss = nn.MSELoss(reduction='none')

    def encoder_func(self, out, embedding):

        if self.cond_ob:
            embedding = (embedding * out).to(self.device)

        out = self.activation_func(out).to(self.device)

        if len(self.encoder_gating_fcs) > 0:
            embedding = self.activation_func(embedding).to(self.device)
            for fc in self.encoder_gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding).to(self.device)
            embedding = self.encoder_gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.encoder_gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                 self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight / self.temperature, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.encoder_gating_weight_fcs,
                                                           self.encoder_gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight / self.temperature, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)

        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.encoder_gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.encoder_gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight / self.temperature, dim=-1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) for layer_module in self.encoder_layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim=-2)

        self.encoder_weights = weights

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.encoder_layer_modules[i + 1]):
                module_input = (module_outputs * \
                                weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                                              layer_module(module_input)
                                          ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim=-2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.encoder_last(out)

        #
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        return mu, log_var

    def decoder_func(self, out, embedding):
        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.decoder_gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.decoder_gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.decoder_gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.decoder_gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                 self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight / self.temperature, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.decoder_gating_weight_fcs,
                                                           self.decoder_gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight / self.temperature, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)

        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.decoder_gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.decoder_gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight / self.temperature, dim=-1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                          for layer_module in self.decoder_layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim=-2)

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.decoder_layer_modules[i + 1]):
                module_input = (module_outputs * \
                                weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                                              layer_module(module_input)
                                          ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim=-2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.decoder_last(out)
        return out

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        Z = mu + epsilon * torch.sqrt(log_var.exp())
        return Z

    def forward(self, obs, skill):
        out = self.base(obs).to(self.device)

        embedding = self.em_base(skill).to(self.device)
        mu, log_var = self.encoder(out, embedding)

        z = self.reparameterization(mu, log_var).to(self.device)
        em_z = self.de_base(z).to(self.device)
        mu_prime_given_Y = self.decoder(em_z, embedding)

        res_loss = self.mse_loss(obs, mu_prime_given_Y).sum(dim=(1, 2, 3))
        kl_loss = -0.5 * torch.sum(1 + log_var - torch.pow(mu, 2) - log_var.exp(), dim=1)
        return res_loss, kl_loss


class DeviationAgent(DreamerAgent):
    def __init__(self, update_skill_every_step, skill_dim, weight, explor_rate, latent_dim, domain, num_init_frames,
                 temperature,
                 **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.num_init_frames = num_init_frames
        self.weight = weight
        self.explor_rate = explor_rate[domain]
        self.latent_dim = latent_dim
        self.temperature = temperature
        # create actor and critic
        super().__init__(**kwargs)
        in_dim = self.wm.inp_size

        self.reward_free = True
        self.solved_meta = None

        self._task_behavior = SkillActorCritic(self.cfg, self.act_spec, self.tfstep, self.skill_dim,
                                               discrete_skills=True).to(self.device)

        # create net
        self.deviation = SoftModuleVAE(self.latent_dim * 2, self.skill_dim, (3, 64, 64), [1024],
                                       [1024], 2, 4, 1024, 1024, 1, self.temperature, self.device).to(
            self.device)
        self.deviation_opt = common.Optimizer('deviation', self.deviation.parameters(), **self.cfg.model_opt,
                                              use_amp=self._use_amp)
        self.deviation.train()
        self.requires_grad_(requires_grad=False)

    def finetune_mode(self):
        self.is_ft = True
        self.reward_free = False
        self._task_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8},
                                                        device=self.device)
        self.cfg.actor_ent = 1e-4
        self.cfg.sf_actor_ent = 1e-4

    def act(self, obs, meta, step, eval_mode, state):
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        meta = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in meta.items()}

        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(latent)

        skill = meta['skill']
        inp = torch.cat([feat, skill], dim=-1)
        if eval_mode:
            actor = self._task_behavior.actor(inp)
            action = actor.mean
        else:
            actor = self._task_behavior.actor(inp)
            action = actor.sample()
        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def get_meta_specs(self):
        return specs.Array((self.skill_dim,), np.float32, 'skill'),

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    @torch.no_grad()
    def regress_meta(self, replay_iter, step):
        """
        Skill version:
            compute E_s[p(z, r| s)]/E_s[p(z|s)] = p(z,r) / p(z) = p(r|z) for each z
            choose the highest :D
        """
        if self.solved_meta is not None:
            return self.solved_meta
        data = next(replay_iter)
        data = self.wm.preprocess(data)
        embed = self.wm.encoder(data)
        post, prior = self.wm.rssm.observe(
            embed, data['action'], data['is_first'])
        feat = self.wm.rssm.get_feat(post)

        reward = data['reward']
        mc_returns = reward.sum(dim=1)  # B X 1

        skill_values = []
        for index in range(self.skill_dim):
            meta = dict()
            skill = torch.zeros(list(feat.shape[:-1]) + [self.skill_dim], device=self.device)
            skill[:, :, index] = 1.0
            meta['skill'] = skill

            inp = torch.cat([feat, skill], dim=-1)
            actor = self._task_behavior.actor(inp)
            a_log_probs = actor.log_prob(data['action']).sum(dim=1)

            skill_values.append((mc_returns * a_log_probs).mean())

        skill_values = torch.stack(skill_values, dim=0)
        skill_selected = torch.argmax(skill_values)

        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[skill_selected] = 1.0
        print("skill selected: ", skill_selected)
        self.solved_meta = {'skill': skill}
        self._task_behavior.solved_meta = self.solved_meta

        return self.solved_meta

    def update_deviation(self, skill, next_obs, step):

        metrics = dict()
        loss, res_loss, kl_loss = self.compute_deviation_loss(next_obs, skill)
        metrics.update(self.deviation_opt(loss, self.deviation.parameters()))

        metrics['deviation_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, skill, obs):
        skill_hat = torch.argmax(skill, dim=1)
        obs_copy = obs.repeat_interleave(self.skill_dim, dim=0).to(self.device)

        eye = torch.eye(self.skill_dim).to(self.device)
        eye_copy = eye.repeat(obs.shape[0], 1).to(self.device)

        size = 4000
        res_list = []
        kl_list = []
        for i in range(0, len(obs_copy), size):
            obs_input = obs_copy[i:i + size]
            eye_input = eye_copy[i:i + size]
            res_out, kl_out = self.deviation(obs_input, eye_input)
            res_list.append(res_out)
            kl_list.append(kl_out)
        res_loss = torch.cat(res_list, dim=0)
        kl_loss = torch.cat(kl_list, dim=0)

        density = torch.exp(-res_loss)
        density[density == 0.] += 1e-44

        density_copy = density.reshape(-1, self.skill_dim)
        kl_loss_copy = kl_loss.reshape(-1, self.skill_dim)
        density_org = density_copy[torch.arange(density_copy.shape[0]), skill_hat]

        density_zi = (self.weight * density_org).to(self.device)
        density_cov = (torch.sum(density_copy, dim=1) + (self.weight - 1) * density_org).to(self.device)

        kl_cov = torch.mean(kl_loss_copy, dim=1).to(self.device)

        intr_reward = torch.log((density_zi * self.skill_dim) / density_cov) + self.explor_rate * kl_cov

        return intr_reward

    def compute_deviation_loss(self, obs, skill, beta=0.1):
        res_loss, kl_loss = self.deviation(obs, skill)
        loss = (res_loss + beta * kl_loss).mean()
        return loss, res_loss, kl_loss

    def update(self, data, step):
        metrics = {}
        B, T, _ = data['action'].shape
        obs_shape = data['observation'].shape[2:]

        if self.reward_free:
            temp_data = self.wm.preprocess(data)
            obs = temp_data['observation'].reshape(B * T, *obs_shape)  # (2500,3,64,64)
            skill = data['skill'].reshape(B * T, -1)  # (2500,16)
            with common.RequiresGrad(self.deviation):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(
                        self.update_deviation(skill, obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, obs).reshape(B, T, 1)

            data['reward'] = intr_reward
            metrics['intr_reward'] = intr_reward.mean().item()

        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k, v in start.items()}
        reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean  # .mode()
        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics

    @torch.no_grad()
    def estimate_value(self, start, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.wm.rssm.get_feat(start)
        start['action'] = torch.zeros_like(actions[0], device=self.device)
        seq = {k: [v] for k, v in start.items()}
        for t in range(horizon):
            action = actions[t]
            state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.wm.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        reward = self.wm.heads['reward'](seq['feat']).mean
        if self.cfg.mpc_opt.use_value:
            B, T, _ = seq['feat'].shape
            seq['skill'] = torch.from_numpy(self.solved_meta['skill']).repeat(B, T, 1).to(self.device)
            value = self._task_behavior._target_critic(get_feat_ac(seq)).mean
        else:
            value = torch.zeros_like(reward, device=self.device)
        disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)

        lambda_ret = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda,
            axis=0)

        # First step is lost because the reward is from the start state
        return lambda_ret[1]

    @torch.no_grad()
    def plan(self, obs, meta, step, eval_mode, state, t0=True):
        """
        Plan next action using Dyna-MPC inference.
        """
        if self.solved_meta is None:
            return self.act(obs, meta, step, eval_mode, state)

        # Get Dreamer's state and features
        obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        post, prior = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(post)

        # Sample policy trajectories
        num_pi_trajs = int(self.cfg.mpc_opt.mixture_coef * self.cfg.mpc_opt.num_samples)
        if num_pi_trajs > 0:
            start = {k: v.repeat(num_pi_trajs, *list([1] * len(v.shape))) for k, v in post.items()}
            img_skill = torch.from_numpy(self.solved_meta['skill']).repeat(num_pi_trajs, 1).to(self.device)
            seq = self.wm.imagine(self._task_behavior.actor, start, None, self.cfg.mpc_opt.horizon, task_cond=img_skill)
            pi_actions = seq['action'][1:]

        # Initialize state and parameters
        start = {k: v.repeat(self.cfg.mpc_opt.num_samples + num_pi_trajs, *list([1] * len(v.shape))) for k, v in
                 post.items()}
        mean = torch.zeros(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        std = 2 * torch.ones(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.mpc_opt.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(self.cfg.mpc_opt.horizon, self.cfg.mpc_opt.num_samples, self.act_dim,
                                              device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(start, actions, self.cfg.mpc_opt.horizon)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.mpc_opt.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.mpc_opt.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                    score.sum(0) + 1e-9))
            _std = _std.clamp_(self.cfg.mpc_opt.min_std, 2)
            mean, std = self.cfg.mpc_opt.momentum * mean + (1 - self.cfg.mpc_opt.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.act_dim, device=std.device)
        new_state = (post, a.unsqueeze(0))
        return a.cpu().numpy(), new_state
