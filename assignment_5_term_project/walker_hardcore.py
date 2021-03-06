#!/usr/bin/env python3
import argparse
import collections
import os
from time import sleep

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3_289_01")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf


import argparse
import collections
import os

import shutil
import wrappers
import zipfile

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")  # Report only TF errors by default

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=45, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")


# Just params for model eval freq. and saving ....
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=200, type=int, help="Evaluate the given number of episodes.")

# Main params ...
parser.add_argument("--batch_size", default=150, type=int, help="Batch size.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")

parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")  # 0.0005, 0.0003, 0.00025

parser.add_argument("--target_delay", default=2, type=int, help="delay target policy and target Q")
parser.add_argument("--tau_poliak_critic", default=0.005, type=float, help="Tau for poliak... i.e. (1-tau)*target")
parser.add_argument("--tau_poliak_actor", default=0.005, type=float, help="Tau for poliak... i.e. (1-tau)*target")


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
    # return tf.keras.models.load_model(name)


def load_network(network, pre):

    try:
        print(pre + "actor_model")
        # loading the models
        network.policy_actor = load_model(pre + "actor_model")
        print(pre + "actor_model")
        network.q1_critic = load_model(pre + "critic_model")

        network.q2_critic = load_model(pre + "critic_2_model")

        network.target_policy_actor = load_model(pre + "target_actor_model")
        network.target_q1_critic = load_model(pre + "target_critic_model")
        network.target_q2_critic = load_model(pre + "target_critic_2_model")
        network.compile()

        print("successfully loaded")
    except :
        print("error loading")
        pass


def save_network(network, dir):

    prefix = dir + "/" + "experimentalM2_hardcore_"
    save_model(network.policy_actor, prefix + "actor_model")
    save_model(network.q1_critic, prefix + "critic_model")
    save_model(network.q2_critic, prefix + "critic_2_model")
    save_model(network.target_policy_actor, prefix + "target_actor_model")
    save_model(network.target_q1_critic, prefix + "target_critic_model")
    save_model(network.target_q2_critic, prefix + "target_critic_2_model")


def evaluate_episode(env, args, network, start_evaluation=False):
    rewards, (state, _), done = 0, env.reset( start_evaluation ), False
    fall = False
    states = []
    actions = []
    while not done:
        if args.render_each and env.episode % args.render_each == 0:
            env.render()

        action = None
        max_q = None
        best = None
        qs = []
        acts = []

        if isinstance(network, list):
            for i, n in enumerate(network):
                a = n.select_action(np.asarray([state]))[0]
                q = n.get_Q(np.asarray([state]))[0]
                acts.append(a)
                qs.append(q)
                if max_q is None:
                    action = a
                    max_q = q
                    best = i
                elif q > max_q:
                    action = a
                    max_q = q
                    best = i
            p = qs[0]/sum(qs)
            # todo -- do not combine actions for holes !
            # lidar = 0
            # for i in range(14, 24):
            #     lidar =+ state[i]
            # print(lidar/10)


            # print("Best: {}".format(best))
            # Fix fall heuristic
            if qs[1] > 100 or qs[1] > qs[0]:
                # print("OK qV: {} , qM: {}".format(qs[1], qs[0]))
                action = acts[1]
            else:
                # print("NO qV: {} , qM: {}".format(qs[1], qs[0]))
                #if p > 0:
                action = acts[0] * 0.5 + acts[1] *0.5
                action = acts[1]

                # elif qs[1] > qs[0]:
                #         action = acts[1]
                # else:
                #     action = acts[0]
                #action = fix_fall_heurictic(action, states, actions, network)
        else:
            action = network.select_action(np.asarray([state]))[0]

        states.append(state)  # from state
        actions.append(action)  # with action

        state, reward, done, _ = env.step(action)
        # print(state[0])
        if reward == -100:
            fall = True
            print("FAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALLLLLLLLLL")
        rewards += reward
    return rewards, fall


def set_seeds(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


def print_stats_save_model(all_returns, network, best_so_far, args, env):
    print("Periodic Evaluation ........................................")
    fall_count = 0
    ENV = wrappers.EvaluationWrapper(gym.make("BipedalWalkerHardcore-v3"), args.seed)  #
    for _ in range(args.evaluate_for):
        r, fell = evaluate_episode(ENV, args, network, start_evaluation=False)
        fall_count += fell
        all_returns.append(r)
    avg = sum(all_returns[-20:]) / 20
    print("........ Current mean {}-episode return: {}, fall ratio: {}".format(args.evaluate_for, avg, fall_count/args.evaluate_for))
    # if avg > best_so_far:
    #     best_so_far = avg
    #     print("Best so far: {}".format(best_so_far))
    best_so_far += 1
    save_network(network, "{}".format(best_so_far))

    return best_so_far


def modify_reward(reward, done, scale=10, min_clip=-5):
    # Done only if you fell ...
    done = True if done and reward < -10 else False
    reward *= scale
    reward = np.clip(reward, min_clip, 10)
    if done:
        reward = 0

    return reward, done


def test_network(network, env, args):
    fall_cnt = 0
    i = 0
    episode_rewards = 0
    while True:
        episode_reward, fall = evaluate_episode(env, args, network, start_evaluation=True)
        episode_rewards += episode_reward
        fall_cnt += fall
        i += 1
        print(fall)
        # 99 because after 100th test episode it exits...
        if i == 100:
            return args.seed, episode_rewards/100, fall_cnt/100


def warmup_buffer(start_steps, replay_buffer, env, action_dim, max_action, network):
    start_steps = 30*950
    while start_steps > 0:
        rewards, (state, states), done = 0, env.reset(), False
        while not done:
            action = np.clip(network.select_action(np.asarray([state]))[0] + np.random.normal(0, args.explore_noise, size=action_dim), -1 * max_action, max_action)
            next_state, reward, done, _ = env.step(action)
            reward, d = modify_reward(reward, done)
            replay_buffer.append(Transition(state, action, reward, d , next_state))
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

        # Have the network estimate the Advantage function as an intermediate layer
        # action_dim = 1
        # x = tf.keras.layers.Dense(action_dim + 1, activation='linear')(hidden_layer)
        # output_layer = tf.keras.layers.Lambda(lambda i: tf.keras.backend.expand_dims(i[:, 0], -1) + i[:, 1:] - tf.keras.backend.mean(i[:, 1:], keepdims=True),
        #                                       output_shape=(action_dim,))(x)

        output_layer = tf.keras.layers.Dense(1, name='critic_q_{}_output'.format(i))(hidden_layer)  #  1 q value for a \in R^n

        #                           [batch, 24]   [batch, 4]
        q = tf.keras.models.Model([input_states, input_actions], output_layer, name="CRITIC_Q_{}_MODEL".format(i))

        q.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
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

        policy_actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate * 0.1), )
        return policy_actor

    def compile(self):
        self.policy_actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate * 0.1), )

        self.q1_critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            # loss=tf.keras.losses.MeanSquaredError(),
        )

        self.q2_critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            # loss=tf.keras.losses.MeanSquaredError(),
        )

    def __init__(self, env, args, state_dim, action_dim, max_action):
        self.tau_poliak_critic = args.tau_poliak_critic
        self.tau_poliak_actor = args.tau_poliak_actor
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

    @wrappers.typed_np_function(np.float32)
    def get_Q(self, states):
        actions = self.policy_actor(states)
        q1 = self.target_q1_critic([states, actions])
        q2 = self.target_q2_critic([states, actions])
        target_Q = tf.math.minimum(q1, q2)

        return target_Q

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train_critic_Q(self, states, actions, returns):
        # tf.print(states.shape)  # TensorShape([100, 24])
        # tf.print(actions.shape)  # TensorShape([100, 4])
        # tf.print(returns.shape)  # TensorShape([100, 1])
        # tf.print("---------------------")
        #
        # q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        # q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        # value_loss = q1_loss + q2_loss

        with tf.GradientTape(persistent=True) as tape:
            # predict 1
            critic1_values = self.q1_critic([states, actions], training=True)
            # loss 1
            loss1 = 0.5 * tf.reduce_mean(tf.math.square(returns - critic1_values))

            critic2_values = self.q2_critic([states, actions], training=True)

            loss2 = 0.5 * tf.reduce_mean(tf.math.square(returns - critic2_values))

            # value_loss = loss1 + loss2

        q1_grad = tape.gradient(loss1, self.q1_critic.trainable_variables)
        q2_grad = tape.gradient(loss2, self.q2_critic.trainable_variables)
        self.q1_critic.optimizer.apply_gradients(zip(q1_grad, self.q1_critic.trainable_variables))
        self.q2_critic.optimizer.apply_gradients(zip(q2_grad, self.q2_critic.trainable_variables))

        # with tf.GradientTape() as tape:
        #     # predict 2
        #
        #     # loss 2
        #
        # q2_grad = tape.gradient(loss2, self.q2_critic.trainable_variables)
        # self.q2_critic.optimizer.apply_gradients(zip(q2_grad, self.q2_critic.trainable_variables))

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
            target_var.assign(target_var * (1 - self.tau_poliak_actor) + var * self.tau_poliak_actor)

        for var, target_var in zip(self.q1_critic.trainable_variables, self.target_q1_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.tau_poliak_critic) + var * self.tau_poliak_critic)

        # TODO TD3 - target Q2 update ...
        for var, target_var in zip(self.q2_critic.trainable_variables, self.target_q2_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.tau_poliak_critic) + var * self.tau_poliak_critic)

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
        if args.batch_size >= len(replay_buffer):
            return actor_loss, q1_loss, q2_loss

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
    network3 = Network(env, args, state_dim, action_dim, max_action)
    # network3 = Network(env, args, state_dim, action_dim, max_action)

    # Load model (if available in directory...)
    load_network(network3,  "experimentalM3_hardcore_")
    load_network(network, "experimentalM2_hardcore_")

    # >>> Test <<<
    if args.recodex:
        seed, ret, fall = test_network([network3, network], env, args)
        return ret, fall
    # (-0.1, 0.0) .. 0.47
    # (-0.2, -0.1) .. 0.15
    # (-0.3, -0.2) .. 0.05
    # (-0.4, -0.3) .. 0.05
    # 550   0.3
    # 150   0.4
    # >>> Train <<<
    EPSIODE, all_returns, best_so_far = 0, [], 1

    # Buffer !
    replay_buffer = []  # max. 500K buffer
    # Preload buffer with 1000 steps performed using uniform random policy
    warmup_buffer(10e3, replay_buffer, env, action_dim, max_action, network)
    print("Buffer warmed up with {} steps".format(len(replay_buffer)))
    pos_x = None
    while True:
        states = None
        d = None
        regenerated = 0
        regenerated_pos = 0
        # after each "evaluate_each" episodes we evaluate model ...
        for eBeforeEval in range(args.evaluate_each):
            if states is not None and d is True and regenerated < 5:  # replay same failed level up to 5 times
                regenerated += 1
                print("\t[{}] ".format(regenerated), end="")
                (state, states), done, walked_distance, STEP = env.reset( (False, (states, pos_x) ) ), False, 0, 0
                regenerated_pos = pos_x
            else:
                regenerated = 0
                regenerated_pos = 0
                # Train
                (state, states), done, walked_distance, STEP = env.reset(), False,0, 0
            EPSIODE += 1
            ep_steps = 0

            while not done:
                # Render from time to time...
                # if regenerated:
                #     env.render()

                # Select action, and add noise, clip to action range
                action = np.clip(network.select_action(np.asarray([state]))[0] + np.random.normal(0, args.explore_noise, size=action_dim), -1*max_action, max_action)
                # print(action)
                # Perform step and save result to buffer
                next_state, reward, done, _ = env.step(action)
                ep_steps +=1
                pos_x = env.ENV.hull.position.x

                walked_distance += reward if reward >= -1 else 0
                reward, d = modify_reward(reward, done)
                if not regenerated:
                    replay_buffer.append(Transition(state, action, reward, d, next_state))
                # add the hard part again ...
                elif regenerated and pos_x*0.7 <= regenerated_pos <= pos_x*1.2:
                    # print("adding hard part again")
                    replay_buffer.append(Transition(state, action, reward, d, next_state))
                state = next_state
                STEP += 1

            # Train K times at each K-step episode ...
            actor_loss, q1_loss, q2_loss = network.train(replay_buffer, STEP, args)
            if len(replay_buffer) > 500000:  # reduce by one third...
                replay_buffer = replay_buffer[len(replay_buffer)//3:]
            result = "OK" if not d else "NO"
            print('\rEpisode: {} ({} steps [{}]),\tDist.: {:.2f},\tactor_loss: {:.10f},\tc1_loss:{:.10f},\tc2_loss:{:.10f}' .format(EPSIODE, ep_steps, result, walked_distance, actor_loss, q1_loss, q2_loss, end=""))
        # Perform Periodic Evaluation
        best_so_far = print_stats_save_model(all_returns, network, best_so_far, args, env)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    rets = []
    falls = []
    from registration import make as MAKE
    for seed in [42, 116, 554, 2, 0, 45]:
        # Create the environment
        args.seed = seed

        maked = MAKE("BipedalWalkerHardcore-v3") # gym.make("BipedalWalkerHardcore-v3")
        # maked = gym.make("BipedalWalkerHardcore-v3")
        env = wrappers.EvaluationWrapper(maked, args.seed)  #
        ret, fall = main(env, args)
        rets.append(ret)
        falls.append(fall)
    print("---Multi seed eval [2]---")
    print("ret.: {:.3f} +- {:.3f}".format(np.average(rets), np.std(rets)))
    print("fall: {:.3f} +- {:.3f}".format(np.average(falls), np.std(falls)))

# ---Multi seed eval [1]---
# ret.: 221.912 +- 5.923
# fall: 0.348 +- 0.0183

# ---Multi seed eval [2]---
# ret.: 235.561 +- 13.548
# fall: 0.282 +- 0.0483
