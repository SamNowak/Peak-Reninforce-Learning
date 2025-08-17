import gymnasium as gym
import numpy as np
import cv2
import time
import threading
import pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01
from gymnasium import spaces
from mss import mss
from collections import defaultdict, deque
import math
from concurrent.futures import ThreadPoolExecutor
import os

class PeakEnv(gym.Env):
    """Enhanced Peak Environment with optimizations and latest game mechanics"""
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 obs_mode='pixels',
                 crop=None,
                 hud_crop=None,
                 dt: float = 0.05,
                 frame_skip: int = 2,
                 difficulty='easy',
                 output_res=(128, 128),   # <— NEW: downscaled game frame returned to the agent
                 hud_res=(64, 128),       # <— NEW: downscaled HUD returned to the agent
                 grayscale=True,
                 window_title=None,
                 auto_snap_window=True):         # <— NEW: use 1 channel to cut memory
        super().__init__()
        assert obs_mode in ('pixels', 'features')
        self.window_title = window_title
        self.auto_snap_window = auto_snap_window
        self.obs_mode = obs_mode
        self.dt = dt
        self.frame_skip = frame_skip
        self.difficulty = difficulty
        self.output_res = tuple(output_res)
        self.hud_res = tuple(hud_res)
        self.grayscale = bool(grayscale)
        self._obs_channels = 1 if self.grayscale else 3


        # Difficulty settings (for curriculum learning)
        self.difficulty_settings = {
            'easy': {'stamina_mult': 1.5, 'fall_threshold': 30, 'reward_scale': 1.2},
            'medium': {'stamina_mult': 1.0, 'fall_threshold': 20, 'reward_scale': 1.0},
            'hard': {'stamina_mult': 0.8, 'fall_threshold': 15, 'reward_scale': 0.8}
        }
        self.current_settings = self.difficulty_settings[difficulty]

        # Action space
        keys   = spaces.MultiDiscrete([3, 3] + [2]*8 + [2])  # movement + actions
        camera = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        scroll = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = spaces.Dict({'keys': keys, 'camera': camera, 'scroll': scroll})

        # Screen capture regions (high-res capture; we resize before returning to the agent)
        H, W = 768, 1280
        self.crop = {'top': 120 + 32, 'left': 320 + 8, 'width': 1280 - 16, 'height': 768 - 40}
        self.hud_crop = {'top': 120 + 32 + (768 - 40) - 180, 'left': 320 + 8, 'width': 400, 'height': 180}
        self.sct = mss()

        if self.auto_snap_window:
            self._snap_to_window()

        # Frame buffer for performance
        self.frame_buffer = deque(maxlen=4)
        self.frame_skip_counter = 0

        # Observation space (small + optional grayscale)
        if obs_mode == 'pixels':
            out_h, out_w = self.output_res
            hud_h, hud_w = self.hud_res
            pixel_space  = spaces.Box(0, 255, (out_h, out_w, self._obs_channels), dtype=np.uint8)
            hud_space    = spaces.Box(0, 255, (hud_h, hud_w, self._obs_channels), dtype=np.uint8)
            sensor_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
            self.observation_space = spaces.Dict({'pixels': pixel_space, 'hud': hud_space, 'sensors': sensor_space})
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)

        # Adaptive reward weights
        self.alpha = 1.0; self.beta = 0.5; self.kappa = 1.0; self.lambda_ = 1.0
        self.mu = 2.0; self.nu = 0.2; self.psi = 0.05; self.phi = 0.5; self.theta = 0.3

        # Dynamic thresholds
        self.fall_thr = self.current_settings['fall_threshold']
        self.stamina_spam_cooldown = 0.5

        # Internal state
        self._lock = threading.Lock()
        self._prev_y = 0.0; self._current_y = 0.0; self._current_x = 0.0
        self._prev_stamina = 1.0; self._prev_py = None; self._prev_action_hash = None
        self._last_climb_time = 0
        self._estimated_summit_height = 1000.0

        # Tracking
        self.visitation_counts = defaultdict(int)
        self.visited_cells = set()
        self.repetition_count = 0
        self.episode_count = 0
        self.success_count = 0
        self.time_since_checkpoint = 0

        # Async action executor
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Templates and HUD detection settings
        self._load_templates()
        self.stamina_lower = np.array([40, 50, 50])
        self.stamina_upper = np.array([80, 255, 255])

        time.sleep(1.0)

    def _load_templates(self):
        try:
            self.player_template = cv2.imread('player_template.png')
            self.summit_template = cv2.imread('summit_template.png')
            if self.player_template is None or self.summit_template is None:
                print("Warning: Template images not found. Some features disabled.")
                print("Run: python setup_and_test.py --capture")
                self.use_templates = False
            else:
                self.use_templates = True
        except Exception as e:
            print(f"Error loading templates: {e}")
            self.use_templates = False

    def _snap_to_window(self):
        """Find the target window and set crop/hud_crop to its exact client area."""
        # Make coordinates DPI-aware
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

        import win32gui

        hwnd = None

        # Prefer a specific window title if provided
        if getattr(self, "window_title", None):
            try:
                import pygetwindow as gw
                wins = gw.getWindowsWithTitle(self.window_title)
                if wins:
                    w = wins[0]
                    if w.isMinimized:
                        w.restore()
                    hwnd = w._hWnd
            except Exception:
                hwnd = None

        # Fallback: foreground window
        if hwnd is None:
            try:
                hwnd = win32gui.GetForegroundWindow()
            except Exception:
                hwnd = None

        if not hwnd:
            print("[PeakEnv] Could not locate window; keeping previous crop.")
            return

        # Get exact client rect in *screen* coordinates
        try:
            # Client rect in client coords (0,0) to (w,h)
            left_client, top_client, right_client, bottom_client = win32gui.GetClientRect(hwnd)
            # Convert client (0,0) and (w,h) to screen coords
            left, top = win32gui.ClientToScreen(hwnd, (left_client, top_client))
            right, bottom = win32gui.ClientToScreen(hwnd, (right_client, bottom_client))
            width, height = right - left, bottom - top
        except Exception as e:
            print(f"[PeakEnv] GetClientRect/ClientToScreen failed: {e}")
            return

        # Main capture: full client area (whatever size your window is)
        self.crop = {'top': int(top), 'left': int(left), 'width': int(width), 'height': int(height)}

        # HUD region (example: bottom-left portion; tweak fractions if needed)
        hud_w = int(width * 0.30)
        hud_h = int(height * 0.22)
        self.hud_crop = {
            'top': int(top + height - hud_h),
            'left': int(left),
            'width': hud_w,
            'height': hud_h
        }

        print(f"[PeakEnv] Snap window -> crop={self.crop}, hud_crop={self.hud_crop}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pyautogui.press('r')
        time.sleep(self.dt * 2)

        if self.auto_snap_window:
            self._snap_to_window()

        obs = self._get_obs()
        self._prev_y = self._extract_y(obs['pixels'] if self.obs_mode == 'pixels' else None)
        self._current_y = self._prev_y
        self._prev_stamina = self._get_stamina()

        self.visited_cells.clear()
        self.repetition_count = 0
        self.time_since_checkpoint = 0
        self._last_climb_time = 0
        self._prev_py = None
        self._prev_action_hash = None

        if self.episode_count > 0 and self.episode_count % 50 == 0:
            self._adjust_difficulty()
        self.episode_count += 1
        return obs, {'difficulty': self.difficulty}

    def step(self, action):
        keys, cam, scr = action['keys'], action['camera'], action['scroll']

        if len(keys) == 11:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, _ = keys
        else:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb = keys

        current_time = time.time()
        if self._prev_stamina < 0.1 and clb:
            if current_time - self._last_climb_time < self.stamina_spam_cooldown:
                clb = 0
            else:
                self._last_climb_time = current_time

        action_future = self.executor.submit(
            self._execute_actions, mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, cam, scr
        )

        time.sleep(self.dt)
        obs = self._get_obs()

        try:
            action_future.result(timeout=0.1)
        except:
            pass

        frame = obs['pixels'] if self.obs_mode == 'pixels' else None
        curr_y = self._extract_y(frame)
        delta_y = curr_y - self._prev_y
        self._current_y = curr_y

        stamina = self._get_stamina()
        damage = self._get_damage_flag()
        reward = self._calculate_adaptive_reward(delta_y, stamina, damage, action, obs)

        self._prev_y = curr_y
        self._prev_stamina = stamina
        self._prev_action_hash = hash(tuple(keys))
        self.time_since_checkpoint += self.dt

        terminated = False
        if frame is not None and self.use_templates:
            terminated = self._check_summit_raw()  # use raw-sized frame via template matching
            if terminated:
                reward += 100.0
                self.success_count += 1

        truncated = False
        info = {
            'height': curr_y,
            'stamina': stamina,
            'ep_count': self.episode_count,
            'success_rate': self.success_count / max(1, self.episode_count)
        }
        return obs, reward, terminated, truncated, info

    def _execute_actions(self, mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, cam, scr):
        try:
            if mh == 0: pyautogui.press('a')
            elif mh == 2: pyautogui.press('d')
            if mv == 0: pyautogui.press('s')
            elif mv == 2: pyautogui.press('w')
            if jmp: pyautogui.press('space')
            if spr: pyautogui.press('shift')
            if crh: pyautogui.press('ctrl')
            if itc: pyautogui.press('e')
            if drp: pyautogui.press('q')
            if emo: pyautogui.press('r')
            if png: pyautogui.press('o')
            if clb:
                pyautogui.mouseDown(button='left')
                time.sleep(0.05)
                pyautogui.mouseUp(button='left')
            dyaw, dpitch = cam
            pyautogui.moveRel(int(dyaw * 10), int(dpitch * 10))
            pyautogui.scroll(int(scr[0] * 5))
        except:
            pass

    def _calculate_adaptive_reward(self, delta_y, stamina, damage, action, obs):
        keys = action['keys']; scr = action['scroll']
        progress_mult = min(2.0, 1.0 + (self._current_y / self._estimated_summit_height))
        grid_cell = (int(self._current_x / 50), int(self._current_y / 50))
        exploration_bonus = 0
        if grid_cell not in self.visited_cells:
            self.visited_cells.add(grid_cell)
            exploration_bonus = self.phi
        action_hash = hash(tuple(keys))
        if action_hash == self._prev_action_hash:
            self.repetition_count += 1
            repetition_penalty = min(0.5, self.repetition_count * 0.1)
        else:
            self.repetition_count = 0
            repetition_penalty = 0
        efficiency_bonus = 0
        if delta_y > 0 and stamina > 0.3:
            efficiency_bonus = self.theta * min(1.0, delta_y / 10.0)
        stamina_drop = max(0.0, self._prev_stamina - stamina)
        stamina_bonus = self.beta * stamina * self.current_settings['stamina_mult']
        fall_mag = max(0.0, -delta_y - self.fall_thr)
        rope_eff = self.nu * (1.0 - abs(scr[0]))

        if len(keys) == 11:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, _ = keys
        else:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb = keys
        sparsity_count = (mh != 1) + (mv != 1) + jmp + spr + crh + itc + drp + emo + png + clb
        action_sparsity_pen = self.psi * sparsity_count

        sensors = obs['sensors'] if self.obs_mode == 'pixels' else obs
        state_key = tuple(round(float(x), 2) for x in sensors[:4].tolist())
        self.visitation_counts[state_key] += 1
        curiosity_reward = self.phi / math.sqrt(self.visitation_counts[state_key])

        reward = (
            self.alpha * delta_y * progress_mult
            + stamina_bonus
            - self.kappa * stamina_drop
            - self.lambda_ * fall_mag
            - self.mu * damage
            + rope_eff
            - action_sparsity_pen
            - repetition_penalty
            + efficiency_bonus
            + exploration_bonus
            + curiosity_reward
        ) * self.current_settings['reward_scale']
        return reward

    def render(self, mode='human'):
        if self.obs_mode == 'pixels':
            frame = self._get_obs()['pixels']
            if mode == 'human':
                cv2.imshow('PEAK', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return frame

    def close(self):
        if self.obs_mode == 'pixels':
            cv2.destroyAllWindows()
        self.executor.shutdown(wait=False)

    def _resize_and_maybe_gray(self, img, target_hw):
        out_h, out_w = target_hw
        img_small = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            img_small = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
            img_small = img_small[..., None]
        return img_small

    def _get_obs(self):
        """Capture raw frames, then downscale & (optionally) grayscale for the agent."""
        if self.frame_skip_counter % self.frame_skip == 0:
            if self.obs_mode == 'pixels':
                with self._lock:
                    frame_raw = np.array(self.sct.grab(self.crop))[:, :, :3]
                    hud_raw   = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
                    self.frame_buffer.append((frame_raw, hud_raw))
            self.frame_skip_counter = 0
        else:
            self.frame_skip_counter += 1

        if self.obs_mode == 'pixels':
            if len(self.frame_buffer) > 0:
                frame_raw, hud_raw = self.frame_buffer[-1]
            else:
                with self._lock:
                    frame_raw = np.array(self.sct.grab(self.crop))[:, :, :3]
                    hud_raw   = np.array(self.sct.grab(self.hud_crop))[:, :, :3]

            # Return SMALL frames to the agent
            frame_small = self._resize_and_maybe_gray(frame_raw, self.output_res)
            hud_small   = self._resize_and_maybe_gray(hud_raw, self.hud_res)

            sensors = self._get_enhanced_sensors()
            return {'pixels': frame_small, 'hud': hud_small, 'sensors': sensors}
        else:
            return self._get_enhanced_sensors()

    def _get_enhanced_sensors(self):
        px, py, vy = self._extract_player_state()
        stamina = self._get_stamina()
        px_norm = px / self.crop['width'] if self.crop['width'] > 0 else 0
        py_norm = py / self.crop['height'] if self.crop['height'] > 0 else 0
        vy_norm = np.clip(vy / 100.0, -1.0, 1.0)
        grip_quality = self._estimate_grip_quality()
        path_gradient = self._estimate_path_gradient()
        time_norm = min(1.0, self.time_since_checkpoint / 60.0)
        height_progress = min(1.0, self._current_y / self._estimated_summit_height)
        return np.array([
            px_norm, py_norm, vy_norm, stamina,
            grip_quality, path_gradient, time_norm, height_progress
        ], dtype=np.float32)

    def _extract_player_state(self):
        if not self.use_templates:
            return self._extract_player_state_fallback()
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            res = cv2.matchTemplate(gray, cv2.cvtColor(self.player_template, cv2.COLOR_BGR2GRAY),
                                    cv2.TM_CCOEFF_NORMED)
            _, max_val, _, (px, py) = cv2.minMaxLoc(res)
            if max_val < 0.5:
                return self._extract_player_state_fallback()
            if self._prev_py is not None:
                vy = (py - self._prev_py) / (self.dt * max(1, self.frame_skip))
            else:
                vy = 0.0
            self._prev_py = py
            self._current_x = float(px)
            return float(px), float(py), float(vy)
        except Exception:
            return self._extract_player_state_fallback()

    def _extract_player_state_fallback(self):
        return self.crop['width'] / 2, self._current_y, 0.0

    def _get_stamina(self):
        try:
            if len(self.frame_buffer) > 0:
                _, hud = self.frame_buffer[-1]   # use RAW hud for color detection
            else:
                hud = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
            hsv = cv2.cvtColor(hud, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, self.stamina_lower, self.stamina_upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            filled = cv2.countNonZero(mask); total = mask.size
            return float(np.clip(filled / total if total > 0 else 0, 0.0, 1.0))
        except Exception:
            return 0.5

    def _get_damage_flag(self):
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            red_channel = frame[:, :, 0]
            red_mean = np.mean(red_channel)
            if red_mean > 150:
                return 1.0
            return 0.0
        except:
            return 0.0

    def _estimate_grip_quality(self):
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            roi = gray[h//2-50:h//2+50, w//2-50:w//2+50]
            variance = np.var(roi)
            grip = np.clip(variance / 1000.0, 0.0, 1.0)
            return float(grip)
        except:
            return 0.5

    def _estimate_path_gradient(self):
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            h, w = edges.shape
            roi = edges[h//3:2*h//3, w//3:2*w//3]
            lines = cv2.HoughLines(roi, 1, np.pi/180, 50)
            if lines is not None and len(lines) > 0:
                angles = [line[0][1] for line in lines[:5]]
                avg_angle = np.mean(angles)
                gradient = np.sin(avg_angle)
                return float(gradient)
            return 0.0
        except:
            return 0.0

    def _extract_y(self, frame_small):
        if frame_small is None:
            return self._current_y
        try:
            # frame_small is already downscaled (and maybe grayscale)
            if self.grayscale:
                gray = frame_small[..., 0]
            else:
                gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
            _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            ys = np.where(bright > 0)[0]
            if len(ys) > 0:
                return float(np.median(ys))
            else:
                return self._current_y
        except:
            return self._current_y

    def _check_summit_raw(self):
        """Template match on RAW-sized frame (more reliable than tiny frame)."""
        if not self.use_templates:
            return self._current_y < 50
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            summit_gray = cv2.cvtColor(self.summit_template, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, summit_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return bool(max_val > 0.7)
        except:
            return False

    def _adjust_difficulty(self):
        success_rate = self.success_count / max(1, self.episode_count)
        if success_rate > 0.7 and self.difficulty != 'hard':
            self.difficulty = 'medium' if self.difficulty == 'easy' else 'hard'
            print(f"Difficulty increased to: {self.difficulty}")
        elif success_rate < 0.3 and self.difficulty != 'easy':
            self.difficulty = 'medium' if self.difficulty == 'hard' else 'easy'
            print(f"Difficulty decreased to: {self.difficulty}")
        self.current_settings = self.difficulty_settings[self.difficulty]
        self.fall_thr = self.current_settings['fall_threshold']

# Register the enhanced environment
from gymnasium.envs.registration import register

try:
    gym.envs.registry.pop('Peak-v4')
except:
    pass

register(
    id='Peak-v4',
    entry_point='peak_env:PeakEnv',
    max_episode_steps=2000,
    kwargs={
        'obs_mode': 'pixels',
        'difficulty': 'easy',
        'frame_skip': 2,
        'output_res': (128, 128),   # <— small
        'hud_res': (64, 128),       # <— small
        'grayscale': True           # <— 1 channel
    }
)
