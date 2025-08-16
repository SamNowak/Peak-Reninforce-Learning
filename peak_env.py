import gymnasium as gym
import numpy as np
import cv2
import time
import threading
import pyautogui
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
                 difficulty='easy'):
        super().__init__()
        assert obs_mode in ('pixels','features')
        self.obs_mode = obs_mode
        self.dt = dt
        self.frame_skip = frame_skip
        self.difficulty = difficulty

        # Difficulty settings (for curriculum learning)
        self.difficulty_settings = {
            'easy': {'stamina_mult': 1.5, 'fall_threshold': 30, 'reward_scale': 1.2},
            'medium': {'stamina_mult': 1.0, 'fall_threshold': 20, 'reward_scale': 1.0},
            'hard': {'stamina_mult': 0.8, 'fall_threshold': 15, 'reward_scale': 0.8}
        }
        self.current_settings = self.difficulty_settings[difficulty]

        # Action space (unchanged but with better descriptions)
        keys   = spaces.MultiDiscrete([3, 3] + [2]*8 + [2])  # movement + actions
        camera = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        scroll = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = spaces.Dict({
            'keys':   keys,
            'camera': camera,
            'scroll': scroll
        })

        # Capture regions with default values
        H, W = 720, 1280
        self.crop     = crop     or {'top': 100, 'left': 320, 'width': W, 'height': H}
        self.hud_crop = hud_crop or {'top': 820, 'left': 50,  'width': 400, 'height': 200}
        self.sct = mss()

        # Frame buffer for performance
        self.frame_buffer = deque(maxlen=4)
        self.frame_skip_counter = 0

        # Observation space
        if obs_mode == 'pixels':
            pixel_space  = spaces.Box(0, 255, (H, W, 3), dtype=np.uint8)
            hud_space    = spaces.Box(
                0, 255,
                (self.hud_crop['height'], self.hud_crop['width'], 3),
                dtype=np.uint8
            )
            sensor_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)  # Expanded features
            self.observation_space = spaces.Dict({
                'pixels':  pixel_space,
                'hud':      hud_space,
                'sensors':  sensor_space
            })
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)

        # Adaptive reward weights
        self.alpha       = 1.0   # Height progress
        self.beta        = 0.5   # Stamina preservation
        self.kappa       = 1.0   # Stamina loss penalty
        self.lambda_     = 1.0   # Fall penalty
        self.mu          = 2.0   # Damage penalty
        self.nu          = 0.2   # Rope efficiency
        self.psi         = 0.05  # Action sparsity (reduced)
        self.phi         = 0.5   # Exploration bonus
        self.theta       = 0.3   # Efficiency bonus
        
        # Dynamic thresholds
        self.fall_thr    = self.current_settings['fall_threshold']
        self.stamina_spam_cooldown = 0.5  # New: prevent stamina exploit

        # Internal state
        self._lock            = threading.Lock()
        self._prev_y          = 0.0
        self._current_y       = 0.0
        self._current_x       = 0.0
        self._prev_stamina    = 1.0
        self._prev_py         = None
        self._prev_action_hash = None
        self._last_climb_time = 0
        self._estimated_summit_height = 1000.0  # Will be updated
        
        # Tracking for rewards
        self.visitation_counts = defaultdict(int)
        self.visited_cells = set()
        self.repetition_count = 0
        self.episode_count = 0
        self.success_count = 0
        self.time_since_checkpoint = 0
        
        # Async action executor
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Load templates with error handling
        self._load_templates()
        
        # HSV bounds for stamina bar (updated for new UI)
        self.stamina_lower    = np.array([40,  50,  50])
        self.stamina_upper    = np.array([80, 255, 255])

        time.sleep(1.0)

    def _load_templates(self):
        """Load template images with fallback"""
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

    def reset(self, seed=None, options=None):
        """Reset environment with improved state tracking"""
        super().reset(seed=seed)
        
        pyautogui.press('r')
        time.sleep(self.dt * 2)  # Give more time for reset
        
        obs = self._get_obs()
        self._prev_y       = self._extract_y(obs['pixels'] if self.obs_mode=='pixels' else None)
        self._current_y    = self._prev_y
        self._prev_stamina = self._get_stamina()
        
        # Reset tracking
        self.visited_cells.clear()
        self.repetition_count = 0
        self.time_since_checkpoint = 0
        self._last_climb_time = 0
        self._prev_py = None
        self._prev_action_hash = None
        
        # Adjust difficulty if needed
        if self.episode_count > 0 and self.episode_count % 50 == 0:
            self._adjust_difficulty()
        
        self.episode_count += 1
        
        info = {'difficulty': self.difficulty}
        return obs, info

    def step(self, action):
        """Execute action with new mechanics and optimizations"""
        keys, cam, scr = action['keys'], action['camera'], action['scroll']

        # Check the length to handle both old and new formats
        if len(keys) == 11:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, _ = keys  # 11 elements
        else:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb = keys  # 10 elements

        # Check for stamina spam exploit prevention (new in latest patch)
        current_time = time.time()
        if self._prev_stamina < 0.1 and clb:
            if current_time - self._last_climb_time < self.stamina_spam_cooldown:
                clb = 0  # Ignore climb action
            else:
                self._last_climb_time = current_time

        # Execute actions asynchronously for better performance
        action_future = self.executor.submit(
            self._execute_actions,
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, cam, scr
        )

        # Get observation while actions execute
        time.sleep(self.dt)
        obs = self._get_obs()
        
        # Ensure actions completed
        try:
            action_future.result(timeout=0.1)
        except:
            pass  # Continue even if action times out
        
        # Calculate state changes
        frame   = obs['pixels'] if self.obs_mode=='pixels' else None
        curr_y  = self._extract_y(frame)
        delta_y = curr_y - self._prev_y
        self._current_y = curr_y

        stamina = self._get_stamina()
        damage  = self._get_damage_flag()

        # Calculate adaptive reward
        reward = self._calculate_adaptive_reward(
            delta_y, stamina, damage, action, obs
        )

        # Update state
        self._prev_y       = curr_y
        self._prev_stamina = stamina
        self._prev_action_hash = hash(tuple(keys))
        self.time_since_checkpoint += self.dt

        # Check terminal condition
        terminated = False
        if frame is not None and self.use_templates:
            terminated = self._check_summit(frame)
            if terminated:
                reward += 100.0  # Summit bonus
                self.success_count += 1

        # Truncated if episode is too long
        truncated = False
        
        info = {
            'height': curr_y,
            'stamina': stamina,
            'episode': self.episode_count,
            'success_rate': self.success_count / max(1, self.episode_count)
        }

        return obs, reward, terminated, truncated, info

    def _execute_actions(self, mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, cam, scr):
        """Execute keyboard and mouse actions"""
        try:
            # Movement
            if mh == 0: pyautogui.press('a')
            elif mh == 2: pyautogui.press('d')
            if mv == 0: pyautogui.press('s')
            elif mv == 2: pyautogui.press('w')
            
            # Actions
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

            # Camera control
            dyaw, dpitch = cam
            pyautogui.moveRel(int(dyaw * 10), int(dpitch * 10))
            
            # Scroll (rope control)
            pyautogui.scroll(int(scr[0] * 5))
        except:
            pass  # Continue even if action fails

    def _calculate_adaptive_reward(self, delta_y, stamina, damage, action, obs):
        """Enhanced reward calculation with adaptive components"""
        keys = action['keys']
        scr = action['scroll']
        
        # Progress multiplier based on height
        progress_mult = min(2.0, 1.0 + (self._current_y / self._estimated_summit_height))
        
        # Exploration bonus for new areas
        grid_cell = (int(self._current_x / 50), int(self._current_y / 50))
        exploration_bonus = 0
        if grid_cell not in self.visited_cells:
            self.visited_cells.add(grid_cell)
            exploration_bonus = self.phi
        
        # Action repetition penalty
        action_hash = hash(tuple(keys))
        if action_hash == self._prev_action_hash:
            self.repetition_count += 1
            repetition_penalty = min(0.5, self.repetition_count * 0.1)
        else:
            self.repetition_count = 0
            repetition_penalty = 0
        
        # Efficiency bonus for smooth climbing
        efficiency_bonus = 0
        if delta_y > 0 and stamina > 0.3:
            efficiency_bonus = self.theta * min(1.0, delta_y / 10.0)
        
        # Stamina management
        stamina_drop = max(0.0, self._prev_stamina - stamina)
        stamina_bonus = self.beta * stamina * self.current_settings['stamina_mult']
        
        # Fall penalty
        fall_mag = max(0.0, -delta_y - self.fall_thr)
        
        # Rope efficiency
        rope_eff = self.nu * (1.0 - abs(scr[0]))
        
        # Action sparsity (reduced penalty)
        # Handle both 10 and 11 element formats
        if len(keys) == 11:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb, _ = keys  # 11 elements
        else:
            mh, mv, jmp, spr, crh, itc, drp, emo, png, clb = keys  # 10 elements
        sparsity_count = (mh != 1) + (mv != 1) + jmp + spr + crh + itc + drp + emo + png + clb
        action_sparsity_pen = self.psi * sparsity_count
        
        # Curiosity reward using state visitation
        sensors = obs['sensors'] if self.obs_mode == 'pixels' else obs
        state_key = tuple(round(float(x), 2) for x in sensors[:4].tolist())
        self.visitation_counts[state_key] += 1
        curiosity_reward = self.phi / math.sqrt(self.visitation_counts[state_key])
        
        # Combine all reward components
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
        """Render the environment"""
        if self.obs_mode == 'pixels':
            frame = self._get_obs()['pixels']
            if mode == 'human':
                cv2.imshow('PEAK', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return frame

    def close(self):
        """Clean up resources"""
        if self.obs_mode == 'pixels':
            cv2.destroyAllWindows()
        self.executor.shutdown(wait=False)

    def _get_obs(self):
        """Get observation with frame buffering for performance"""
        if self.frame_skip_counter % self.frame_skip == 0:
            if self.obs_mode == 'pixels':
                with self._lock:
                    frame = np.array(self.sct.grab(self.crop))[:, :, :3]
                    hud   = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
                    self.frame_buffer.append((frame, hud))
            self.frame_skip_counter = 0
        else:
            self.frame_skip_counter += 1
        
        if self.obs_mode == 'pixels':
            if len(self.frame_buffer) > 0:
                frame, hud = self.frame_buffer[-1]
            else:
                with self._lock:
                    frame = np.array(self.sct.grab(self.crop))[:, :, :3]
                    hud   = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
            sensors = self._get_enhanced_sensors()
            return {'pixels': frame, 'hud': hud, 'sensors': sensors}
        else:
            return self._get_enhanced_sensors()

    def _get_enhanced_sensors(self):
        """Get enhanced sensor features for better state representation"""
        # Basic state
        px, py, vy = self._extract_player_state()
        stamina = self._get_stamina()
        
        # Normalize positions
        px_norm = px / self.crop['width'] if self.crop['width'] > 0 else 0
        py_norm = py / self.crop['height'] if self.crop['height'] > 0 else 0
        vy_norm = np.clip(vy / 100.0, -1.0, 1.0)
        
        # Additional features
        grip_quality = self._estimate_grip_quality()
        path_gradient = self._estimate_path_gradient()
        time_norm = min(1.0, self.time_since_checkpoint / 60.0)
        height_progress = min(1.0, self._current_y / self._estimated_summit_height)
        
        return np.array([
            px_norm, py_norm, vy_norm, stamina,
            grip_quality, path_gradient, time_norm, height_progress
        ], dtype=np.float32)

    def _extract_player_state(self):
        """Extract player position and velocity using template matching"""
        if not self.use_templates:
            # Fallback: use simple color detection
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
            
            if max_val < 0.5:  # Low confidence
                return self._extract_player_state_fallback()
            
            # Calculate velocity
            if self._prev_py is not None:
                vy = (py - self._prev_py) / (self.dt * max(1, self.frame_skip))
            else:
                vy = 0.0
            
            self._prev_py = py
            self._current_x = float(px)
            
            return float(px), float(py), float(vy)
        except Exception as e:
            return self._extract_player_state_fallback()

    def _extract_player_state_fallback(self):
        """Fallback method using color detection"""
        return self.crop['width'] / 2, self._current_y, 0.0

    def _get_stamina(self):
        """Extract stamina from HUD with improved detection"""
        try:
            if len(self.frame_buffer) > 0:
                _, hud = self.frame_buffer[-1]
            else:
                hud = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
            
            hsv = cv2.cvtColor(hud, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, self.stamina_lower, self.stamina_upper)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            filled = cv2.countNonZero(mask)
            total = mask.size
            
            return float(np.clip(filled / total if total > 0 else 0, 0.0, 1.0))
        except Exception as e:
            return 0.5  # Default middle value

    def _get_damage_flag(self):
        """Detect if player took damage (enhanced detection)"""
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            
            # Check for red flash (damage indicator)
            red_channel = frame[:, :, 0]
            red_mean = np.mean(red_channel)
            
            # Damage typically causes a red tint
            if red_mean > 150:
                return 1.0
            
            return 0.0
        except:
            return 0.0

    def _estimate_grip_quality(self):
        """Estimate surface grip quality based on visual cues"""
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            
            # Simple texture analysis for grip estimation
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Get region around player
            h, w = gray.shape
            roi = gray[h//2-50:h//2+50, w//2-50:w//2+50]
            
            # Calculate texture variance (rough surfaces have higher variance)
            variance = np.var(roi)
            
            # Normalize to 0-1 (higher is better grip)
            grip = np.clip(variance / 1000.0, 0.0, 1.0)
            
            return float(grip)
        except:
            return 0.5

    def _estimate_path_gradient(self):
        """Estimate the slope/gradient of the climbing path"""
        try:
            if len(self.frame_buffer) > 0:
                frame, _ = self.frame_buffer[-1]
            else:
                frame = np.array(self.sct.grab(self.crop))[:, :, :3]
            
            # Edge detection to find climbing surfaces
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Focus on center region
            h, w = edges.shape
            roi = edges[h//3:2*h//3, w//3:2*w//3]
            
            # Detect dominant line angle using Hough transform
            lines = cv2.HoughLines(roi, 1, np.pi/180, 50)
            
            if lines is not None and len(lines) > 0:
                # Get average angle
                angles = [line[0][1] for line in lines[:5]]  # Top 5 lines
                avg_angle = np.mean(angles)
                
                # Convert to gradient (-1 to 1, where 0 is horizontal)
                gradient = np.sin(avg_angle)
                return float(gradient)
            
            return 0.0
        except:
            return 0.0

    def _extract_y(self, frame):
        """Extract vertical position from frame"""
        if frame is None:
            return self._current_y
        
        try:
            # Look for height indicators (white markers, UI elements, etc.)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Find bright regions (potential height markers)
            _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Get vertical positions of bright pixels
            ys = np.where(bright > 0)[0]
            
            if len(ys) > 0:
                # Use median for robustness
                return float(np.median(ys))
            else:
                return self._current_y
        except:
            return self._current_y

    def _check_summit(self, frame):
        """Check if summit is reached"""
        if not self.use_templates:
            # Fallback: check if at top of screen
            return self._current_y < 50
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            summit_gray = cv2.cvtColor(self.summit_template, cv2.COLOR_BGR2GRAY)
            
            res = cv2.matchTemplate(gray, summit_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            return bool(max_val > 0.7)  # Lower threshold for better detection
        except:
            return False

    def _adjust_difficulty(self):
        """Adjust difficulty based on performance (curriculum learning)"""
        success_rate = self.success_count / max(1, self.episode_count)
        
        if success_rate > 0.7 and self.difficulty != 'hard':
            # Increase difficulty
            if self.difficulty == 'easy':
                self.difficulty = 'medium'
            else:
                self.difficulty = 'hard'
            print(f"Difficulty increased to: {self.difficulty}")
        elif success_rate < 0.3 and self.difficulty != 'easy':
            # Decrease difficulty
            if self.difficulty == 'hard':
                self.difficulty = 'medium'
            else:
                self.difficulty = 'easy'
            print(f"Difficulty decreased to: {self.difficulty}")
        
        self.current_settings = self.difficulty_settings[self.difficulty]
        self.fall_thr = self.current_settings['fall_threshold']

    def capture_templates(self):
        """Utility to capture template images while in-game"""
        try:
            import keyboard
        except ImportError:
            print("Please install keyboard: pip install keyboard")
            return
        
        print("=== Template Capture Utility ===")
        print("Make sure Peak game is running and visible")
        
        print("\n1. Position player character in center of screen")
        print("   Press 'p' when ready...")
        keyboard.wait('p')
        
        # Capture player template
        player_region = {
            'top': self.crop['top'] + self.crop['height']//2 - 50,
            'left': self.crop['left'] + self.crop['width']//2 - 40,
            'width': 80,
            'height': 100
        }
        player_img = np.array(self.sct.grab(player_region))[:, :, :3]
        cv2.imwrite('player_template.png', cv2.cvtColor(player_img, cv2.COLOR_RGB2BGR))
        print("✓ Player template saved")
        
        print("\n2. Navigate to summit marker/flag")
        print("   Press 's' when summit is visible...")
        keyboard.wait('s')
        
        # Capture summit template
        summit_region = {
            'top': self.crop['top'] + 100,
            'left': self.crop['left'] + self.crop['width']//2 - 100,
            'width': 200,
            'height': 150
        }
        summit_img = np.array(self.sct.grab(summit_region))[:, :, :3]
        cv2.imwrite('summit_template.png', cv2.cvtColor(summit_img, cv2.COLOR_RGB2BGR))
        print("✓ Summit template saved")
        
        print("\nTemplates captured successfully!")
        print("You can now start training.")

# Register the enhanced environment
from gymnasium.envs.registration import register

# Unregister if exists
try:
    gym.envs.registry.pop('Peak-v4')
except:
    pass

register(
    id='Peak-v4',
    entry_point='peak_env:PeakEnv',
    max_episode_steps=2000,  # Increased for harder difficulties
    kwargs={
        'obs_mode': 'pixels',
        'difficulty': 'easy',  # Start with easy for curriculum learning
        'frame_skip': 2
    }
)