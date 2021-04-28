from datetime import datetime
import argparse
import gym
import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path

def rollout(env, args, idx):

    save_root = args.save_root / f'episode_{idx:03d}' 
    (save_root / 'rgb').mkdir(parents=True, exist_ok=False)

    step = 0
    done = False
    obs = env.reset()

    while not done:
        img = env.render(mode='rgb_array')
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(str(save_root / 'rgb' / f'{step:06d}.png'), img)
        for key, item in obs.items():
            path = save_root / key
            if not path.exists():
                path.mkdir(parents=True, exist_ok=False)
            with open(path / f'{step:06d}.png', 'wb') as f:
                np.save(f, item)

        step += 1
        obs = new_obs

        if args.show:
            cv2.imshow('env', img)
            cv2.waitKey(10)


def main(args):
    env = gym.make(args.env)
    for i in tqdm(range(args.num_episodes)):
        tqdm.write(f'running episode {i}')
        rollout(env, args, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--save_root', type=str, default='data')
    parser.add_argument('--env', type=str, default='FetchSlide-v1')
    #parser.add_argument('--env', type=str, default='Skiing-v0')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()

    save_root = Path(args.save_root) / datetime.now().strftime("%Y%m%d_%H%M%S") 
    save_root.mkdir(parents=True, exist_ok=False)
    args.save_root = save_root
    main(args)
