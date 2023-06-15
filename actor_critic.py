import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from networks import ActorCriticNetwork

class Agent:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.gamma = gamma
        self.action = None
        
        self.actor_critic = ActorCriticNetwork()
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

        # default_amp = 0.05021718 # ibmq_lima
        # default_amp = 0.10223725901141269

        self._default_amp = 0.69564843
        self._max_amp = self._default_amp + 0.001
        self._min_amp = self._default_amp - 0.001

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        mean = self._min_amp + (self._max_amp - self._min_amp) * probs.numpy()[0][0]
        std = probs.numpy()[0][1]/10000

        # Create a Normal distribution with the mean and std from probs
        action_dist = tfp.distributions.Normal(mean, std)

        # Sample an action from the distribution
        action = action_dist.sample()

        self.action = action

        return action.numpy()

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            mean = self._min_amp + (self._max_amp - self._min_amp) * probs.numpy()[0][0]
            std = probs.numpy()[0][1]/10000

            # Create a Normal distribution with the mean and std from probs
            action_dist = tfp.distributions.Normal(mean, std)
            log_prob = action_dist.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        
        actor_critic_variables = self.actor_critic.trainable_variables

        gradient = tape.gradient(total_loss, actor_critic_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, actor_critic_variables
        ))

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)