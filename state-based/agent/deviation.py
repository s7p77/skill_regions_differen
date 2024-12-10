import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.ddpg import DDPGAgent
from collections import OrderedDict
from dm_env import specs
import utils
import torchrl.init as init
import torchrl.base as base
import random


def null_activation(x):
    return x


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


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

        self.base = base.MLPBase(
            last_activation_func=null_activation,
            input_shape=input_shape,
            activation_func=activation_func,
            hidden_shapes=hidden_shapes,
            **kwargs)
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
        # module_input_shape = input_shape
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
        # module_input_shape = self.latent_dim

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

        self.decoder_last = nn.Linear(module_input_shape, input_shape).to(self.device)
        last_init_func(self.decoder_last)

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated"
        gating_input_shape = self.em_base.output_shape

        # gating_input_shape = em_input_shape

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

        # TODO can try other loss functions
        res_loss = self.mse_loss(obs, mu_prime_given_Y).sum(dim=1)
        kl_loss = -0.5 * torch.sum(1 + log_var - torch.pow(mu, 2) - log_var.exp(), dim=1)
        return res_loss, kl_loss

class DeviationAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, update_encoder, skill, weight, explor_rate, latent_dim,
                 temperature, num_init_frames, ensemble_size,scale,domain,
                 **kwargs):
        self.skill_dim = skill_dim
        self.ensemble_size = ensemble_size
        self.update_skill_every_step = update_skill_every_step
        self.update_encoder = update_encoder
        self.weight = weight
        self.explor_rate = explor_rate[domain]
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.num_init_frames = num_init_frames
        self.scale = scale
        # specify skill in fine-tuning stage if needed
        self.skill = int(skill) if skill >= 0 else np.random.choice(self.skill_dim)

        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.batch_size = kwargs['batch_size']

        # create actor and critic
        super().__init__(**kwargs)

        # create net
        self.deviation = SoftModuleVAE(self.latent_dim * 2, self.skill_dim, self.obs_dim - self.skill_dim, [1024],
                                       [1024], 2, 4, 1024, 1024, 1, self.temperature, kwargs['device']).to(
            kwargs['device'])

        # optimizers
        self.deviation_opt = torch.optim.Adam(self.deviation.parameters(), lr=self.lr)
        self.deviation.train()

    def get_meta_specs(self):
        return specs.Array((self.skill_dim,), np.float32, 'skill'),

    def init_meta(self):
        skill = np.zeros(self.skill_dim).astype(np.float32)
        if not self.reward_free:
            skill[self.skill] = 1.0
        else:
            skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_deviation(self, next_obs, skill):
        metrics = dict()

        loss, res_loss, kl_loss = self.compute_deviation_loss(next_obs, skill)

        self.deviation_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        flag_nan = torch.isnan(loss)
        if torch.any(flag_nan).item():
            print('loss_nan')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.deviation.parameters(), max_norm=0.5)

        self.deviation_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['deviation_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, skill, obs, metrics):
        skill_hat = torch.argmax(skill, dim=1)
        obs_copy = obs.repeat_interleave(self.skill_dim, dim=0).to(self.device)
        eye = torch.eye(self.skill_dim).to(self.device)
        eye_copy = eye.repeat(obs.shape[0], 1).to(self.device)
        res_loss, kl_loss = self.deviation(obs_copy, eye_copy)

        density = torch.exp(-res_loss)
        density[density == 0.] += 1e-44
        density_copy = density.reshape(-1, self.skill_dim)
        kl_loss_copy = kl_loss.reshape(-1, self.skill_dim)
        density_org = density_copy[torch.arange(density_copy.shape[0]), skill_hat]

        density_zi = (self.weight * density_org).to(self.device)
        density_cov = (torch.sum(density_copy, dim=1) + (self.weight - 1) * density_org).to(self.device)

        kl_cov = torch.mean(kl_loss_copy, dim=1).to(self.device)
        intr_reward = torch.log(density_zi * self.skill_dim / density_cov) + self.explor_rate * kl_cov

        flag_nan = torch.isnan(intr_reward)
        if torch.any(flag_nan).item():
            print('intr_reward_nan')

        return self.scale * intr_reward.unsqueeze(1)

    def compute_deviation_loss(self, obs, skill, beta=0.5):

        res_loss, kl_loss = self.deviation(obs, skill)
        loss = (res_loss + beta * kl_loss).mean()
        return loss, res_loss, kl_loss

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        if self.reward_free:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
            metrics.update(self.update_deviation(next_obs, skill))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, metrics)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            batch = next(replay_iter)
            obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
