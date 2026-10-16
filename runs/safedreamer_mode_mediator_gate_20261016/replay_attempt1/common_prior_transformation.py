def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
        prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
    if self._classes:
        shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
        prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):
        shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
        prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    (x, deter) = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Linear, **self._kw)(x)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    _audit_key = nj.rng()
    stoch = jax.vmap(lambda row: jaxutils.OneHotDist(logits=row).sample(seed=_audit_key))(stats['logit'].astype(f32))
    prior = {'stoch': stoch, 'deter': deter, **stats}
    _audit_result = cast(prior)
    _audit_result['_audit_prior_key'] = _audit_key
    return _audit_result
