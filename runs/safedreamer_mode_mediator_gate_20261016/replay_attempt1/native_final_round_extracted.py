def replay_final(self, latent, mean, std, rng_before, reserve_before):
    num_pi_trajs = 0
    _mean_before = mean
    nj.context().rng = rng_before
    nj.context().reserve = list(reserve_before)
    latent_gaus = {k: jnp.repeat(v, self.num_samples, 0) for (k, v) in latent.items()}
    latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
    latent_gaus['action_mean'] = mean
    latent_gaus['action_std'] = std

    def policy(s, horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        return jnp.clip(current_mean + current_std * jax.random.normal(nj.rng(), (self.num_samples,) + self.act_space.shape), -1, 1)
    traj_gaus = self.wm.imagine(policy, latent_gaus, self.horizon, use_planner=True)
    if num_pi_trajs > 0:
        traj_pi_gaus = {}
        for (k, v) in traj_pi.items():
            traj_pi_gaus[k] = jnp.concatenate([traj_pi[k], traj_gaus[k]], axis=1)
    else:
        traj_pi_gaus = traj_gaus
    (rew, ret, value) = self.ac.critics['extr'].score(traj_pi_gaus)
    traj_rew = ret[0]
    if self.config.use_cost:
        (cost, cost_ret, cost_value) = self.ac.cost_critics['extr'].score(traj_pi_gaus)
        traj_cost = cost.sum(0)
        traj_cost = traj_cost * (1000 / self.horizon)
    else:
        recon = self.wm.heads['decoder'](traj_pi_gaus)
        cost = self.cost_from_recon(recon['observation'].mode())
        traj_cost = cost.sum(0)
    num_safe_traj = jnp.sum(lax.convert_element_type(traj_cost < self.cost_limit, jnp.int32))
    (elite_value, elite_actions) = self.func_with_if_else(num_safe_traj, traj_rew, traj_cost, traj_pi_gaus)
    elite_idxs = jax.lax.top_k(elite_value, self.num_elites)[1]
    (elite_value, elite_actions) = (elite_value[elite_idxs], elite_actions[:, elite_idxs, :])
    _mean = elite_actions.mean(axis=1)
    _std = elite_actions.std(axis=1)
    (mean, std) = (self.momentum * mean + (1 - self.momentum) * _mean, _std)
    _cost_only_ids = jax.lax.top_k(-traj_cost, self.num_elites)[1]
    _cost_only_actions = traj_pi_gaus['action'][:, _cost_only_ids, :]
    _cost_only_mean = self.momentum * _mean_before + (1 - self.momentum) * _cost_only_actions.mean(axis=1)
    _cost_only_std = _cost_only_actions.std(axis=1)
    return {'trajectory': traj_pi_gaus, 'actions': traj_pi_gaus['action'], 'costs': traj_cost, 'returns': traj_rew, 'safe_count': num_safe_traj, 'elite_indices_in_native_filtered_array': elite_idxs, 'mean_after': mean, 'std_after': std, 'action': mean[0], 'always_minimum_cost': {'elite_original_candidate_ids': _cost_only_ids, 'mean_after': _cost_only_mean, 'std_after': _cost_only_std, 'action': _cost_only_mean[0]}, 'rng_after': nj.context().rng, 'reserve_after': tuple(nj.context().reserve)}
