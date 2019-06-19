import numpy as np
import gym
import balance_bot

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def callbacks(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():
    env = gym.make("balancebot-v0")
    model =  Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('relu'))
    model.add(Dense(9))
    model.add(Activation('softmax'))
    # print(model.summary())

    memory = SequentialMemory(limit=100000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
        target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=15000, visualize=True, verbose=2, callbacks=None)

    # act = deepq.learn(env,
    #     q_func=model,
    #     lr=1e-3,
    #     max_timesteps=100000,
    #     buffer_size=100000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     print_freq=10,
    #     callback=callback
    # )
    print("Saving model to balance.pkl")
    # After training is done, we save the final weights.
    dqn.save_weights('balance.pkl', overwrite=True)
    print("================================================")
    print('\n')

    #Load the saved weights to dqn
    dqn.load_weights('balance.pkl')
    
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()