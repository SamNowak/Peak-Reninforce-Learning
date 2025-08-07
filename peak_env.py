
import gym
import numpy as np
import cv2
import time
import threading
import pyautogui
from gym import spaces
from mss import mss
from collections import defaultdict
import math

class PeakEnv(gym.Env):
    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self,
                 obs_mode='pixels',
                 crop=None,
                 hud_crop=None,
                 dt: float = 0.05):
        super().__init__()
        assert obs_mode in ('pixels','features')
        self.obs_mode = obs_mode
        self.dt = dt  # time between frames

        # Action space
        keys   = spaces.MultiDiscrete([3, 3] + [2]*8 + [2])
        camera = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        scroll = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = spaces.Dict({
            'keys':   keys,
            'camera': camera,
            'scroll': scroll
        })

        # Capture regions
        H, W = 720, 1280
        self.crop     = crop     or {'top': 100, 'left': 320, 'width': W, 'height': H}
        self.hud_crop = hud_crop or {'top': 820, 'left': 50,  'width': 400, 'height': 200}
        self.sct = mss()

        # Observation space
        if obs_mode == 'pixels':
            pixel_space  = spaces.Box(0, 255, (H, W, 3), dtype=np.uint8)
            hud_space    = spaces.Box(
                0, 255,
                (self.hud_crop['height'], self.hud_crop['width'], 3),
                dtype=np.uint8
            )
            sensor_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
            self.observation_space = spaces.Dict({
                'pixels':  pixel_space,
                'hud':      hud_space,
                'sensors':  sensor_space
            })
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

        # Reward weights
        self.alpha       = 1.0
        self.beta        = 0.5
        self.kappa       = 1.0
        self.lambda_     = 1.0
        self.fall_thr    = 20.0
        self.mu          = 2.0
        self.nu          = 0.2
        self.psi         = 0.1
        self.phi         = 0.5

        # Internal state
        self._lock            = threading.Lock()
        self._prev_y          = 0.0
        self._prev_stamina    = 1.0
        self._prev_py         = None
        self.visitation_counts = defaultdict(int)

        # Preload templates & HSV bounds
        self.player_template  = cv2.imread('player_template.png')
        self.summit_template  = cv2.imread('summit_template.png')
        self.stamina_lower    = np.array([40,  50,  50])
        self.stamina_upper    = np.array([80, 255, 255])

        time.sleep(1.0)

    def reset(self):
        pyautogui.press('r')
        time.sleep(self.dt)
        obs = self._get_obs()
        self._prev_y       = self._extract_y(obs['pixels'] if self.obs_mode=='pixels' else None)
        self._prev_stamina = self._get_stamina()
        return obs

    def step(self, action):
        keys, cam, scr = action['keys'], action['camera'], action['scroll']
        mh, mv, jmp, spr, crh, itc, drp, emo, png, clb = keys

        # Discrete inputs
        if mh==0: pyautogui.press('a')
        elif mh==2: pyautogui.press('d')
        if mv==0: pyautogui.press('s')
        elif mv==2: pyautogui.press('w')
        if jmp: pyautogui.press('space')
        if spr: pyautogui.press('shift')
        if crh: pyautogui.press('ctrl')
        if itc: pyautogui.press('e')
        if drp: pyautogui.press('q')
        if emo: pyautogui.press('r')
        if png: pyautogui.press('o')
        if clb:
            pyautogui.mouseDown(button='left'); pyautogui.mouseUp(button='left')

        # Camera and scroll
        dyaw, dpitch = cam
        pyautogui.moveRel(dyaw * 10, dpitch * 10)
        pyautogui.scroll(int(scr[0] * 5))

        time.sleep(self.dt)
        obs = self._get_obs()
        frame   = obs['pixels'] if self.obs_mode=='pixels' else None
        curr_y  = self._extract_y(frame)
        delta_y = curr_y - self._prev_y

        stamina = self._get_stamina()
        damage  = self._get_damage_flag()

        # Action sparsity penalty
        sparsity_count = (mh != 1) + (mv != 1) + jmp + spr + crh + itc + drp + emo + png + clb
        action_sparsity_pen = self.psi * sparsity_count

        # Curiosity reward
        sensors = obs['sensors'] if self.obs_mode=='pixels' else obs
        key = tuple(round(float(x),2) for x in sensors.tolist())
        self.visitation_counts[key] += 1
        curiosity_reward = self.phi / math.sqrt(self.visitation_counts[key])

        # Other penalties & bonuses
        stamina_drop = max(0.0, self._prev_stamina - stamina)
        fall_mag     = max(0.0, -delta_y - self.fall_thr)
        rope_eff     = 1.0 - abs(scr[0])

        reward = ( self.alpha * delta_y
                 + self.beta  * stamina
                 - self.kappa * stamina_drop
                 - self.lambda_ * fall_mag
                 - self.mu     * damage
                 + self.nu     * rope_eff
                 - action_sparsity_pen
                 + curiosity_reward )

        self._prev_y       = curr_y
        self._prev_stamina = stamina

        done = frame is not None and self._check_summit(frame)
        return obs, reward, done, {}

    def render(self, mode='human'):
        if self.obs_mode=='pixels':
            frame = self._get_obs()['pixels']
            if mode=='human':
                cv2.imshow('PEAK', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            elif mode=='rgb_array':
                return frame

    def close(self):
        if self.obs_mode=='pixels':
            cv2.destroyAllWindows()

    # Internal helpers
    def _get_obs(self):
        if self.obs_mode=='pixels':
            with self._lock:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
                hud   = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
            sensors = self._get_sensors()
            return {'pixels': frame, 'hud': hud, 'sensors': sensors}
        else:
            return self._get_sensors()

    def _get_sensors(self):
        px, py, vy = self._extract_player_state()
        stam       = self._get_stamina()
        return np.array([px, py, vy, stam], dtype=np.float32)

    def _extract_player_state(self):
        frame = self._get_obs()['pixels']
        res   = cv2.matchTemplate(frame, self.player_template, cv2.TM_CCOEFF_NORMED)
        _, _, _, (px, py) = cv2.minMaxLoc(res)
        if self._prev_py is not None:
            vy = (py - self._prev_py) / self.dt
        else:
            vy = 0.0
        self._prev_py = py
        return float(px), float(py), float(vy)

    def _get_stamina(self):
        hud = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
        hsv = cv2.cvtColor(hud, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.stamina_lower, self.stamina_upper)
        filled = cv2.countNonZero(mask)
        total  = mask.size
        return float(np.clip(filled / total, 0.0, 1.0))

    def _get_damage_flag(self):
        # Placeholder for damage detection
        return 0

    def _extract_y(self, frame):
        if frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ys   = np.where(gray > 250)[0]
        return float(ys.mean() if len(ys) else self.crop['height'])

    def _check_summit(self, frame):
        res, _, _, _ = cv2.minMaxLoc(
            cv2.matchTemplate(frame, self.summit_template, cv2.TM_CCOEFF_NORMED))
        return bool(res > 0.8)

# Register the environment
from gym.envs.registration import register
register(
    id='Peak-v4',
    entry_point='peak_env:PeakEnv',
    max_episode_steps=1000
)
