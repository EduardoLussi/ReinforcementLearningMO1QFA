import torch as T

from networks import GenericNetwork

class Agent:
    def __init__(self, alpha=0.001, beta=0.001, input_dims=2, gamma=0.95, n_actions=2,
                 layer1_size=128, layer2_size=128, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs

        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                            layer2_size, n_actions=1)

        # self._default_amp = 0.05021718 # ibmq_lima
        self._default_amp = 0.10223725901141269

        # self._default_amp = 0.69564843
        self._max_amp = self._default_amp + 0.001
        self._min_amp = self._default_amp - 0.001

    def choose_action(self, observation):
        mu, sigma  = self.actor.forward(observation)
        
        print(mu.item(), sigma.item())
        # mu = self._min_amp + (self._max_amp - self._min_amp) * mu
        # sigma = sigma / 5000
        # print(f"Choosing action N({mu.item()*1e2:.4f}e-2, {sigma.item()*1e7:.4f}e-7)")
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        
        action = T.sigmoid(probs)
        action = self._min_amp + (self._max_amp - self._min_amp) * action

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        print("learning...")

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)

        print(f"v_={critic_value_.item()}, v={critic_value.item()}")

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        print(f"delta={delta.item()}")

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        print(f"actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}")

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

        return actor_loss, critic_loss

    def save_models(self):
        print('... saving models ...')
        T.save(self.actor.state_dict(), self.actor.checkpoint_file)
        T.save(self.critic.state_dict(), self.critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_state_dict(T.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(T.load(self.critic.checkpoint_file))