import argparse
import collections
import os
import gym
import numpy as np
import tensorflow as tf
import shutil
import wrappers
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")  # Report only TF errors by default

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=1, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=45, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")


# Just params for model eval freq. and saving ....
parser.add_argument("--evaluate_each", default=300, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")

# Main params ...
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_delay", default=2, type=int, help="delay target policy and target Q")
parser.add_argument("--tau_poliak", default=0.005, type=float, help="Tau for poliak... i.e. (1-tau)*target")

# Noise after sampling
parser.add_argument("--explore_noise", default=0.1, type=float, help=" action + noise ...")

# Noise during sampling (set clip_noise to 0.0 for no noise..)
parser.add_argument("--policy_noise_sd", default=0.2, type=float, help="noise to policy")
parser.add_argument("--policy_noise_clip", default=0.5, type=float, help="clip high policy noise")


# UTILS ......
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def save_model(network, name):
    network.save(name, include_optimizer=False)
    zipf = zipfile.ZipFile(f'{name}.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(f'{name}/', zipf)
    zipf.close()
    shutil.rmtree(name)


def load_model(name):
    with zipfile.ZipFile(name + ".zip", 'r') as zip_ref:
        zip_ref.extractall("./")
    return tf.keras.models.load_model(name)


def load_network(network):
    try:
        # loading the models
        network.policy_actor = load_model("actor_model")
        network.q1_critic = load_model("critic_model")
        network.q2_critic = load_model("critic_2_model")

        network.target_policy_actor = load_model("target_actor_model")
        network.target_q1_critic = load_model("target_critic_model")
        network.target_q2_critic = load_model("target_critic_2_model")
        network.compile()
    except:
        pass


def save_network(network):
    save_model(network.policy_actor, "actor_model")
    save_model(network.q1_critic, "critic_model")
    save_model(network.q2_critic, "critic_2_model")

    save_model(network.target_policy_actor, "target_actor_model")
    save_model(network.target_q1_critic, "target_critic_model")
    save_model(network.target_q2_critic, "target_critic_2_model")


def evaluate_episode(env, args, network, start_evaluation=False):
    rewards, state, done = 0, env.reset(start_evaluation), False
    while not done:
        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
            env.render()

        # TODO: Predict the action using the greedy policy
        action = network.select_action(np.asarray([state]))[0]
        state, reward, done, _ = env.step(action)
        rewards += reward
    return rewards


def set_seeds(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


def print_stats_save_model(all_returns, network, best_so_far, args, env):
    print("Periodic Evaluation ........................................")
    for _ in range(args.evaluate_for):
        r = evaluate_episode(env, args, network, start_evaluation=False)
        all_returns.append(r)
    avg = sum(all_returns[-20:]) / 20
    print("........ Current mean {}-episode return: {}".format(args.evaluate_for, avg))
    if avg > best_so_far:
        best_so_far = avg
        print("Best so far: {}".format(best_so_far))
        save_network(network)
    return all_returns

def test_network(network, env, args):
    while True:
        evaluate_episode(env, args, network, start_evaluation=True)


def warmup_buffer(start_steps, replay_buffer, env, action_dim, max_action):
    while start_steps > 0:
        rewards, state, done = 0, env.reset(), False
        while not done:
            action = np.random.uniform(low=-1*max_action, high=max_action, size=action_dim)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append(Transition(state, action, reward, done, next_state))
            state = next_state
            start_steps -= 1


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TD3 and NETWORKS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class Network:

    def build_compile_Q_critic(self, state_dim, action_dim, learning_rate, i):

        # input for action ... [batch, a] i.e. [64, 4]
        input_actions = tf.keras.layers.Input(action_dim, name='critic_q_{}_input_actions'.format(i))
        # input for state ... [batch, s] i.e. [64, 24]
        input_states = tf.keras.layers.Input(state_dim, name='critic_q_{}_input_states'.format(i))

        # Common hidden layers ...
        hidden_layer = tf.keras.layers.Concatenate(name='critic_q_{}_concatenation'.format(i))([input_states, input_actions])
        hidden_layer = tf.keras.layers.Dense(512, activation=tf.nn.relu, name='critic_q_{}_hidden1_common'.format(i))(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(256, activation=tf.nn.relu, name='critic_q_{}_hidden2_common'.format(i))(hidden_layer)
        output_layer = tf.keras.layers.Dense(1, name='critic_q_{}_output'.format(i))(hidden_layer)  #  1 q value for a \in R^n

        #                           [batch, 24]   [batch, 4]
        q = tf.keras.models.Model([input_states, input_actions], output_layer, name="CRITIC_Q_{}_MODEL".format(i))

        q.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=0.001),
            # loss=tf.keras.losses.MeanSquaredError(),
        )

        return q

    def build_compile_actor(self, args, state_dim, action_dim, max_action):
        input_layer = tf.keras.layers.Input(state_dim)  # ... 24 ...
        hidden_layer = tf.keras.layers.Dense(256, activation=tf.nn.relu)(input_layer)
        hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(hidden_layer)
        output_layer = tf.keras.layers.Dense(action_dim, activation=tf.nn.tanh)(hidden_layer)
        output_layer = tf.multiply(output_layer, max_action)

        policy_actor = tf.keras.Model(input_layer, output_layer)

        policy_actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), )
        return policy_actor

    def compile(self):
        self.policy_actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), )

        self.q1_critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr,  clipnorm=0.001),
            # loss=tf.keras.losses.MeanSquaredError(),
        )

        self.q2_critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr,  clipnorm=0.001),
            # loss=tf.keras.losses.MeanSquaredError(),
        )

    def __init__(self, env, args, state_dim, action_dim, max_action):
        self.tau_poliak = args.tau_poliak_critic
        self.env = env
        self.lr = args.learning_rate
        self.policy_actor = self.build_compile_actor(args, state_dim, action_dim, max_action)
        self.target_policy_actor = tf.keras.models.clone_model(self.policy_actor)

        self.q1_critic = self.build_compile_Q_critic(state_dim, action_dim, args.learning_rate, 1)
        self.target_q1_critic = tf.keras.models.clone_model(self.q1_critic)

        self.q2_critic = self.build_compile_Q_critic(state_dim, action_dim, args.learning_rate, 2)
        self.target_q2_critic = tf.keras.models.clone_model(self.q2_critic)

        self.max_action = 1
        self.policy_noise_sd = args.policy_noise_sd
        self.policy_noise_clip = args.policy_noise_clip

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def select_action(self, states):
        return self.policy_actor(states)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train_critic_Q(self, states, actions, returns):
        # tf.print(states.shape)  # TensorShape([100, 24])
        # tf.print(actions.shape)  # TensorShape([100, 4])
        # tf.print(returns.shape)  # TensorShape([100, 1])
        # tf.print("---------------------")
        with tf.GradientTape() as tape:
            # predict 1
            critic1_values = self.q1_critic([states, actions], training=True)
            # loss 1
            loss1 = tf.reduce_mean(tf.math.square(returns - critic1_values))
        q1_grad = tape.gradient(loss1, self.q1_critic.trainable_variables)
        self.q1_critic.optimizer.apply_gradients(zip(q1_grad, self.q1_critic.trainable_variables))

        with tf.GradientTape() as tape:
            # predict 2
            critic2_values = self.q2_critic([states, actions], training=True)
            # loss 2
            loss2 = tf.reduce_mean(tf.math.square(returns - critic2_values))
        q2_grad = tape.gradient(loss2, self.q2_critic.trainable_variables)
        self.q2_critic.optimizer.apply_gradients(zip(q2_grad, self.q2_critic.trainable_variables))

        return loss1, loss2

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def train_actor(self, states):
        # TODO TD3 - actor updated only using q1
        with tf.GradientTape() as tape:
            critic1_value = self.q1_critic([states, self.policy_actor(states, training=True)], training=True)
            actor_loss = -1 * tf.math.reduce_mean(critic1_value)

        actor_grad = tape.gradient(actor_loss, self.policy_actor.trainable_variables)
        self.policy_actor.optimizer.apply_gradients(zip(actor_grad, self.policy_actor.trainable_variables))
        return actor_loss

    @tf.function
    def polyak_target_update(self):
        for var, target_var in zip(self.policy_actor.trainable_variables, self.target_policy_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.tau_poliak) + var * self.tau_poliak)

        for var, target_var in zip(self.q1_critic.trainable_variables, self.target_q1_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.tau_poliak) + var * self.tau_poliak)

        # TODO TD3 - target Q2 update ...
        for var, target_var in zip(self.q2_critic.trainable_variables, self.target_q2_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.tau_poliak) + var * self.tau_poliak)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        # Predict action + add noise ....
        predicted_actions = self.target_policy_actor(states)
        noise = tf.random.normal(shape=tf.shape(predicted_actions), mean=0.0, stddev=self.policy_noise_sd, dtype=tf.float32)
        noise = tf.clip_by_value(noise, -1*self.policy_noise_clip, self.policy_noise_clip)
        predicted_actions = tf.clip_by_value(predicted_actions + noise, -1*self.max_action, self.max_action)

        # take smaller of two Q
        q1 = self.target_q1_critic([states, predicted_actions])  # model.predict([testAttrX, testImagesX])
        q2 = self.target_q2_critic([states, predicted_actions])
        target_Q = tf.math.minimum(q1, q2)
        return target_Q

    def train(self, replay_buffer, STEPS, args):
        actor_loss, q1_loss, q2_loss = 0, 0, 0

        for s in range(STEPS):
            batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
            #                  (100,)  (100,)
            states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
            rewards = tf.expand_dims(rewards, axis=1)
            gamm = tf.expand_dims((1 - dones) * args.gamma, axis=1)

            # calculate Q and returns (100,1)
            target_Q = self.predict_values(np.asarray(next_states))
            returns = rewards + (gamm * target_Q)
            # train ciric 1 and critic 2
            q1_loss_t, q2_loss_t = self.train_critic_Q(states, actions, returns)
            q1_loss += q1_loss_t
            q2_loss += q2_loss_t

            # TODO TD3: Delayed Updates
            if s % args.target_delay == 0:
                # update critic
                actor_loss += self.train_actor(states)
                # polyak update of target_q1, target_q2, target_critic with
                self.polyak_target_update()

        return actor_loss/STEPS, q1_loss/STEPS, q2_loss/STEPS


def main(env, args):
    # Fix random seeds and number of threads
    set_seeds(args)

    # Construct the network
    state_dim, action_dim, max_action = env.observation_space.shape[0], env.action_space.shape[0], 1
    network = Network(env, args, state_dim, action_dim, max_action)

    # Load model (if available in directory...)
    load_network(network)

    # >>> Test <<<
    if args.recodex:
        test_network(network, env, args)

    # >>> Train <<<
    EPSIODE, all_returns, best_so_far = 0, [], -30

    # Buffer !
    replay_buffer = collections.deque(maxlen=500000)  # max. 500K buffer
    # Preload buffer with 1000 steps performed using uniform random policy
    warmup_buffer(10e3, replay_buffer, env, action_dim, max_action)
    print("Buffer warmed up with {} steps".format(len(replay_buffer)))
    while True:

        # after each "evaluate_each" episodes we evaluate model ...
        for eBeforeEval in range(args.evaluate_each):
            # Train
            state, done, walked_distance, STEP = env.reset(), False,0, 0
            EPSIODE += 1
            while not done:
                # Render from time to time...
                if eBeforeEval and eBeforeEval % 10 == 0:
                    env.render()

                # Select action, and add noise, clip to action range
                action = np.clip(network.select_action(np.asarray([state]))[0] + np.random.normal(0, args.explore_noise, size=action_dim), -1*max_action, max_action)
                # print(action)
                # Perform step and save result to buffer
                next_state, reward, done, _ = env.step(action)
                if -1 < reward < 0:
                    reward *= 10
                if reward > 0:
                    reward *= 10
                # print(reward)
                reward = reward if reward > -90 else -10
                walked_distance += reward
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state
                STEP += 1

            # Train K times at each K-step episode ...
            actor_loss, q1_loss, q2_loss = network.train(replay_buffer, STEP, args)
            print('\rEpisode: {},\tDist.: {:.2f},\tactor_loss: {:.10f},\tc1_loss:{:.10f},\tc2_loss:{:.10f}' .format(EPSIODE, walked_distance, actor_loss, q1_loss, q2_loss, end=""))

        # Perform Periodic Evaluation
        print_stats_save_model(all_returns, network, best_so_far, args, env)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("BipedalWalker-v3"), args.seed)
    main(env, args)
