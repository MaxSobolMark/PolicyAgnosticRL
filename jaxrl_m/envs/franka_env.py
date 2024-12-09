import gym
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R



def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler

class FrankaWrapper(gym.Wrapper):
    def __init__(
        self, env, image_size: int = 256, action_std: float = 5, time_limit=500
    ):
        super(FrankaWrapper, self).__init__(env)
        self.env = env
        self.image_size = image_size
        self.action_std = action_std
        self.time_limit = time_limit
        self.step_count = 0

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(self.image_size, self.image_size, 3)
                ),
                "proprio": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(7,)),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * 6 + [0.0]), high=np.array([1.0] * 6 + [1.0])
        )
    
    def get_proprio(self, obs_dict):
        obs_keys = ['eef_pos', 'eef_rot', 'eef_gripper_width']
        raw_obs = []
        for obs_key in obs_keys:
            if obs_key == "eef_rot":
                raw_obs.extend(quat_to_euler(obs_dict[obs_key]))
            else:
                if obs_key == "eef_gripper_width":
                    raw_obs.extend([obs_dict[obs_key]])
                else:
                    raw_obs.extend(obs_dict[obs_key])
        raw_obs = np.array(raw_obs)
        return raw_obs

    def reset(self):
        obs_dict, _ = self.env.reset()
        image = obs_dict["cam0_left"][0]
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC
        )
        proprio = self.get_proprio(obs_dict)
        self.step_count = 0

        # import pdb; pdb.set_trace()
        return {"image": image, "proprio": proprio}, {}

    def step(self, action):
        action = (action[:-1] * self.action_std, (action[-1:] <= 0.5).astype(np.float32))
        obs_dict, reward, done, info = self.env.step(action)

        image = obs_dict["cam0_left"][0]
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC
        )

        proprio = self.get_proprio(obs_dict)

        self.step_count += 1
        done = done or self.step_count >= self.time_limit
        return {"image": image, "proprio": proprio}, reward, done, self.step_count >= self.time_limit, {}
