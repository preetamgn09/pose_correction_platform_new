"""
Fast-Response Online Pose Monitoring Engine
Optimized for natural exercise speed with minimal hold times
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import time
from collections import deque

from pose_features import PoseFeatureExtractor
from pose_graph import PoseGraph
from state_discovery import PoseState
from pose_adapters import create_pose_adapter


# ============================================================
# Enhanced Feedback Generator with Priority System
# ============================================================

class EnhancedFeedbackGenerator:
    """Intelligent feedback with prioritization and motivation."""
    
    def __init__(self, history_length: int = 15):
        self.feature_history: List[Dict[str, float]] = []
        self.history_length = history_length
        self.last_feedback_features: List[str] = []
        
    def update_history(self, features: Dict[str, float]) -> None:
        self.feature_history.append(features)
        if len(self.feature_history) > self.history_length:
            self.feature_history.pop(0)
    
    def generate_feedback(
        self,
        deviations: Dict[str, float],
        mean_features: Dict[str, float],
        current_features: Dict[str, float],
        tolerances: Dict[str, float]
    ) -> str:
        """Generate prioritized, actionable feedback."""
        
        if self._is_unstable():
            return "Hold steady - minimize movement"
        
        if not deviations:
            return "Adjust to match target pose"
        
        core_features = ['hip', 'knee', 'elbow', 'torso']
        
        core_devs = {
            k: v for k, v in deviations.items()
            if any(f in k for f in core_features)
        }
        
        if core_devs:
            worst_feature, worst_dev = max(core_devs.items(), key=lambda x: abs(x[1]))
        else:
            worst_feature, worst_dev = max(deviations.items(), key=lambda x: abs(x[1]))
        
        if worst_feature in self.last_feedback_features:
            other_devs = {k: v for k, v in deviations.items() if k != worst_feature}
            if other_devs:
                worst_feature, worst_dev = max(other_devs.items(), key=lambda x: abs(x[1]))
        
        self.last_feedback_features.append(worst_feature)
        if len(self.last_feedback_features) > 3:
            self.last_feedback_features.pop(0)
        
        return self._generate_specific_feedback(worst_feature, worst_dev)
    
    def _generate_specific_feedback(self, feature_name: str, deviation: float) -> str:
        """Generate specific actionable feedback."""
        
        if 'elbow' in feature_name:
            side = 'left' if 'left' in feature_name else 'right'
            if deviation > 0:
                return f"Straighten {side} arm more"
            else:
                return f"Bend {side} elbow deeper"
        elif 'knee' in feature_name:
            side = 'left' if 'left' in feature_name else 'right'
            if deviation > 0:
                return f"Straighten {side} leg"
            else:
                return f"Bend {side} knee more"
        elif 'hip' in feature_name:
            if abs(deviation) > 10:
                if deviation > 0:
                    return "Hinge forward at hips more"
                else:
                    return "Straighten hips, stand taller"
        elif 'shoulder' in feature_name:
            side = 'left' if 'left' in feature_name else 'right'
            return f"Adjust {side} shoulder position"
        elif 'torso' in feature_name:
            if abs(deviation) > 5:
                if deviation > 0:
                    return "Lean torso forward slightly"
                else:
                    return "Bring torso more upright"
        elif 'arm_elevation' in feature_name:
            side = 'left' if 'left' in feature_name else 'right'
            if deviation > 0:
                return f"Raise {side} arm higher"
            else:
                return f"Lower {side} arm"
        
        return "Adjust position to match target"
    
    def _is_unstable(self) -> bool:
        """Check for excessive movement/jitter."""
        if len(self.feature_history) < 5:
            return False
        
        torso_vals = [f.get("torso_angle") for f in self.feature_history if "torso_angle" in f]
        
        if len(torso_vals) < 3:
            return False
        
        return np.std(torso_vals) > 8.0


# ============================================================
# Complexity Analyzer
# ============================================================

class ExerciseComplexityAnalyzer:
    """Analyzes exercise to optimize detection parameters."""
    
    @staticmethod
    def analyze(pose_graph: PoseGraph) -> Dict[str, any]:
        """Analyze graph to determine exercise complexity."""
        
        num_states = len(pose_graph.states)
        hold_durations = [s.min_hold_duration for s in pose_graph.states]
        avg_hold = np.mean(hold_durations)
        
        tolerances = []
        for state in pose_graph.states:
            tolerances.extend(state.feature_tolerances.values())
        avg_tolerance = np.mean(tolerances)
        
        is_dynamic = avg_hold < 1.0
        
        if num_states <= 2 and avg_hold > 2.0:
            level = "SIMPLE"
            tolerance_scale = 0.9
        elif num_states <= 4 and not is_dynamic:
            level = "MODERATE"
            tolerance_scale = 1.0
        else:
            level = "COMPLEX"
            tolerance_scale = 1.15
        
        return {
            'level': level,
            'num_states': num_states,
            'is_dynamic': is_dynamic,
            'avg_hold': avg_hold,
            'tolerance_scale': tolerance_scale
        }


# ============================================================
# Fast-Response Pose Monitor
# ============================================================

class EnhancedPoseMonitor:
    """Fast-response pose monitor optimized for natural exercise speed."""
    
    def __init__(
        self,
        pose_graph: PoseGraph,
        pose_model: str = 'yolo',
        model_name: str = 'yolov8n-pose.pt',
        confidence_threshold: float = 0.7,
        feedback_delay: float = 0.3,
        max_reps: Optional[int] = None,
        enable_adaptive: bool = True,
        speed_mode: str = 'fast'
    ):
        print("Initializing Fast-Response Pose Monitor...")
        
        self.graph = pose_graph
        self.enable_adaptive = enable_adaptive
        self.speed_mode = speed_mode
        
        if enable_adaptive:
            self.complexity = ExerciseComplexityAnalyzer.analyze(pose_graph)
            print(f"  Exercise: {self.complexity['level']}")
            print(f"  Tolerance scale: {self.complexity['tolerance_scale']}x")
        else:
            self.complexity = {'level': 'MODERATE', 'tolerance_scale': 1.0, 'is_dynamic': False}
        
        self.pose_adapter = create_pose_adapter(
            pose_model,
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )
        
        self.feature_extractor = PoseFeatureExtractor(
            confidence_threshold=confidence_threshold
        )
        
        self.feedback_generator = EnhancedFeedbackGenerator()
        
        self.temporal_sequence = self.graph.get_metadata('temporal_sequence')
        if not self.temporal_sequence:
            raise ValueError("Pose graph missing temporal_sequence metadata")
        
        print(f"  States per rep: {len(self.temporal_sequence)}")
        print(f"  Speed mode: {speed_mode.upper()}")
        
        if enable_adaptive:
            self._apply_adaptive_tolerances()
        
        # AGGRESSIVE HOLD TIME REDUCTION
        speed_multipliers = {
            'fast': 0.15,
            'normal': 0.25,
            'strict': 0.40
        }
        
        hold_multiplier = speed_multipliers.get(speed_mode, 0.15)
        
        for state in self.graph.states:
            original_hold = state.min_hold_duration
            state.min_hold_duration = max(0.1, min(0.5, original_hold * hold_multiplier))
            print(f"  State {state.state_id}: {original_hold:.2f}s → {state.min_hold_duration:.2f}s")
        
        self.sequence_index = 0
        self.current_state: Optional[PoseState] = None
        self.current_state_start_time = 0.0
        self.current_state_hold_time = 0.0
        self.total_reps = 0
        self.previous_state_id = None
        
        self.max_reps = max_reps
        self.session_complete = False
        
        self.accuracy_active = False
        self.green_frames = 0
        self.evaluated_frames = 0
        self.quality_active = False
        self.current_quality = 0.0
        
        self.in_transition = False
        self.transition_start_time = 0.0
        self.transition_grace = 0.2
        
        self.current_streak = 0
        self.max_streak = 0
        self.last_was_correct = False
        
        self.state_attempt_counts: Dict[int, int] = {}
        self.state_success_counts: Dict[int, int] = {}
        
        self.last_keypoints = None
        self.last_features = None
        self.last_deviations = None
        
        self.feedback_delay = feedback_delay
        self.last_feedback_time = 0.0
        self.current_feedback = ""
        
        self.frame_times: List[float] = []
        self.session_start_time = 0.0

        self.rep_accuracies: List[float] = [] 
        
        self._reset_to_initial_state()
        
        print("Fast-Response Monitor ready!")
    
    def _apply_adaptive_tolerances(self) -> None:
        """Apply intelligent tolerance scaling."""
        scale = self.complexity['tolerance_scale']
        
        for state in self.graph.states:
            for feature_name, tolerance in state.feature_tolerances.items():
                if any(x in feature_name for x in ['hip', 'knee', 'elbow']):
                    state.feature_tolerances[feature_name] = tolerance * scale
                else:
                    state.feature_tolerances[feature_name] = tolerance * scale * 1.1
        
        print(f"  ✓ Adaptive tolerances applied")
    
    def _reset_to_initial_state(self):
        """Reset to first state."""
        self.sequence_index = 0
        self.current_state = self.graph.get_state(self.temporal_sequence[0])
        self.current_state_start_time = time.time()
        self.current_state_hold_time = 0.0
        self.previous_state_id = None
        self.in_transition = False
    
    def _in_grace_period(self) -> bool:
        """Check if in transition grace period."""
        if not self.in_transition:
            return False
        return (time.time() - self.transition_start_time) < self.transition_grace
    
    def _get_progressive_tolerance(self, state_id: int) -> float:
        """Get progressive tolerance multiplier based on difficulty."""
        attempts = self.state_attempt_counts.get(state_id, 0)
        successes = self.state_success_counts.get(state_id, 0)
        
        if attempts < 5:
            return 1.0
        
        success_rate = successes / attempts if attempts > 0 else 1.0
        
        if success_rate < 0.3:
            return 1.2
        elif success_rate < 0.5:
            return 1.1
        
        return 1.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame with enhancements."""
        start_time = time.time()
        
        if self.session_complete:
            return self._annotate_session_complete(frame), {
                "state_id": self.current_state.state_id,
                "hold_progress": 1.0,
                "is_correct": True,
                "feedback": "Session complete!",
                "reps_completed": self.total_reps,
                "sequence_index": self.sequence_index,
                "session_complete": True,
                "streak": self.current_streak
            }
        
        self.last_keypoints = self.pose_adapter.detect(frame)
        
        features = None
        if self.last_keypoints is not None:
            features = self.feature_extractor.extract_features(self.last_keypoints)
            self.last_features = features
        
        status = self._update_state_machine(features)
        annotated = self._annotate_frame(frame, status)
        
        self.frame_times.append(time.time() - start_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        return annotated, status
    
    def _update_state_machine(self, features: Optional[Dict[str, float]]) -> Dict:
        """Enhanced state machine with adaptive features."""
        current_time = time.time()
        state_id = self.current_state.state_id
        
        if state_id not in self.state_attempt_counts:
            self.state_attempt_counts[state_id] = 0
            self.state_success_counts[state_id] = 0
        
        if not features:
            self.current_feedback = "Step back - full body must be visible"
            self.last_was_correct = False
            self.current_streak = 0
            return self._status(False, 0.0)
        
        self.feedback_generator.update_history(features)
        
        progressive_mult = self._get_progressive_tolerance(state_id)
        original_tolerances = self.current_state.feature_tolerances.copy()
        
        if progressive_mult > 1.0:
            for k in self.current_state.feature_tolerances:
                self.current_state.feature_tolerances[k] *= progressive_mult
        
        if self._in_grace_period():
            for k in self.current_state.feature_tolerances:
                self.current_state.feature_tolerances[k] *= 1.2
        
        matches, deviations = self.current_state.matches(features, strict=False)
        self.last_deviations = deviations
        
        self.current_state.feature_tolerances = original_tolerances
        
        if matches:
            if not self.accuracy_active:
                self.accuracy_active = True
            
            if self.in_transition:
                self.in_transition = False
            
            if not self._in_grace_period():
                self.evaluated_frames += 1
                self.green_frames += 1
                self.state_success_counts[state_id] += 1
            
            if not self.last_was_correct:
                self.current_streak = 1
            else:
                self.current_streak += 1
            
            self.max_streak = max(self.max_streak, self.current_streak)
            self.last_was_correct = True
            
            self.current_state_hold_time = current_time - self.current_state_start_time
            hold_progress = self.current_state_hold_time / self.current_state.min_hold_duration
            
            if hold_progress >= 1.0:
                self._transition_to_next_state()
                self.current_feedback = "Next!"
            else:
                remaining = self.current_state.min_hold_duration - self.current_state_hold_time
                
                if self.current_streak >= 10:
                    self.current_feedback = f"🔥 {remaining:.1f}s"
                elif self.current_streak >= 5:
                    self.current_feedback = f"✨ {remaining:.1f}s"
                else:
                    self.current_feedback = f"Hold {remaining:.1f}s"
            
            return self._status(True, min(1.0, hold_progress))
        
        else:
            self.state_attempt_counts[state_id] += 1
            
            if self.accuracy_active and not self._in_grace_period():
                self.evaluated_frames += 1
            
            self.last_was_correct = False
            self.current_streak = 0
            
            self.current_state_hold_time = 0.0
            self.current_state_start_time = current_time
            
            if current_time - self.last_feedback_time > self.feedback_delay:
                self.current_feedback = self.feedback_generator.generate_feedback(
                    deviations,
                    self.current_state.mean_features,
                    features,
                    self.current_state.feature_tolerances
                )
                self.last_feedback_time = current_time
            
            return self._status(False, 0.0)
    
    def _transition_to_next_state(self):
        """Transition with grace period and accuracy tracking."""
        self.sequence_index += 1
        
        if self.sequence_index >= len(self.temporal_sequence):
            if self.evaluated_frames > 0:
                rep_accuracy = (self.green_frames / self.evaluated_frames) * 100
                self.rep_accuracies.append(rep_accuracy)
            
            self.total_reps += 1
            print(f"  [REP {self.total_reps}] Completed!")
            
            self.sequence_index = 0
            
            if self.max_reps and self.total_reps >= self.max_reps:
                self.session_complete = True
                self._print_summary()
                return
        
        self.current_state = self.graph.get_state(self.temporal_sequence[self.sequence_index])
        self.current_state_start_time = time.time()
        self.current_state_hold_time = 0.0
        
        self.in_transition = True
        self.transition_start_time = time.time()
    
    def _print_summary(self):
        """Print enhanced summary."""
        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        
        duration = time.time() - self.session_start_time
        
        print(f"\nReps: {self.total_reps}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        
        if self.evaluated_frames > 0:
            acc = (self.green_frames / self.evaluated_frames) * 100
            print(f"Accuracy: {acc:.1f}%")
        
        print(f"Max Streak: {self.max_streak} correct frames")
        
        if len(self.frame_times) > 0:
            print(f"Avg FPS: {1.0/np.mean(self.frame_times):.1f}")
        
        print("\n" + "="*60)
    
    def _status(self, correct: bool, progress: float) -> Dict:
        """Create status dictionary."""
        return {
            "state_id": self.current_state.state_id,
            "hold_progress": progress,
            "is_correct": correct,
            "feedback": self.current_feedback,
            "reps_completed": self.total_reps,
            "sequence_index": self.sequence_index,
            "streak": self.current_streak,
            "in_grace": self._in_grace_period()
        }
    
    def _scale_keypoints_to_panel(self, keypoints: np.ndarray, panel_width: int, panel_height: int) -> np.ndarray:
        """Scale and center keypoints to fit panel."""
        if keypoints.shape[1] == 2:
            confidence = np.ones((keypoints.shape[0], 1))
            keypoints = np.hstack([keypoints, confidence])
        
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points) == 0:
            return keypoints
        
        min_x = np.min(valid_points[:, 0])
        max_x = np.max(valid_points[:, 0])
        min_y = np.min(valid_points[:, 1])
        max_y = np.max(valid_points[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return keypoints
        
        padding = 0.15
        scale_x = (panel_width * (1 - 2*padding)) / width
        scale_y = (panel_height * (1 - 2*padding)) / height
        scale = min(scale_x, scale_y)
        
        scaled_width = width * scale
        scaled_height = height * scale
        
        offset_x = (panel_width - scaled_width) / 2
        offset_y = (panel_height - scaled_height) / 2
        
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] = (keypoints[:, 0] - min_x) * scale + offset_x
        scaled_keypoints[:, 1] = (keypoints[:, 1] - min_y) * scale + offset_y
        
        return scaled_keypoints
    
    def _draw_reference_skeleton(self, panel: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw reference skeleton with human-like appearance."""
        panel_h, panel_w = panel.shape[:2]
        drawable_height = panel_h - 80
        drawable_y_offset = 75
        
        scaled_kpts = self._scale_keypoints_to_panel(keypoints, panel_w, drawable_height)
        scaled_kpts[:, 1] += drawable_y_offset
        
        # Draw with custom human-like styling
        return self._draw_humanlike_skeleton(panel, scaled_kpts)
    
    def _draw_humanlike_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw a more human-like skeleton with head, thicker bones, and better colors.
        """
        if keypoints is None:
            return frame
        
        # Define bone groups for different thicknesses and colors
        torso_bones = [
            (5, 6),   # shoulders
            (5, 11),  # left shoulder to hip
            (6, 12),  # right shoulder to hip
            (11, 12), # hips
        ]
        
        arm_bones = [
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
        ]
        
        leg_bones = [
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
        ]
        
        head_bones = [
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            (0, 5), (0, 6),  # nose to shoulders
        ]
        
        # Draw bones with different thicknesses
        # Torso (thickest - main body structure)
        for start_idx, end_idx in torso_bones:
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (100, 200, 100), 5)
        
        # Legs (thick)
        for start_idx, end_idx in leg_bones:
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (120, 180, 120), 4)
        
        # Arms (medium)
        for start_idx, end_idx in arm_bones:
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (140, 160, 140), 3)
        
        # Head connections (thin)
        for start_idx, end_idx in head_bones:
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (150, 150, 150), 2)
        
        # Draw HEAD CIRCLE (face representation)
        # Calculate head center and size from facial keypoints
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        
        if nose[2] > 0.3:
            # Calculate head radius based on eye spacing or default size
            if left_eye[2] > 0.3 and right_eye[2] > 0.3:
                eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])
                head_radius = int(eye_dist * 1.8)
            else:
                head_radius = 25
            
            head_center = (int(nose[0]), int(nose[1] - head_radius * 0.3))
            
            # Draw head as filled circle with outline
            cv2.circle(frame, head_center, head_radius, (180, 160, 140), -1)  # Filled
            cv2.circle(frame, head_center, head_radius, (120, 100, 80), 2)    # Outline
        
        # Draw joints with different colors based on body part
        joint_colors = {
            0: (200, 180, 160),   # nose - face color
            1: (200, 180, 160), 2: (200, 180, 160),  # eyes
            3: (200, 180, 160), 4: (200, 180, 160),  # ears
            5: (100, 200, 100), 6: (100, 200, 100),  # shoulders - green
            7: (80, 160, 80), 8: (80, 160, 80),      # elbows
            9: (60, 140, 60), 10: (60, 140, 60),     # wrists
            11: (100, 200, 100), 12: (100, 200, 100), # hips - green
            13: (80, 160, 80), 14: (80, 160, 80),    # knees
            15: (60, 140, 60), 16: (60, 140, 60),    # ankles
        }
        
        # Draw larger, colored joints
        for i in range(17):
            if keypoints[i][2] > 0.3:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                color = joint_colors.get(i, (100, 100, 100))
                
                # Skip facial keypoints (already part of head circle)
                if i <= 4:
                    continue
                
                # Larger joints for major points
                if i in [5, 6, 11, 12]:  # shoulders and hips
                    radius = 8
                else:
                    radius = 6
                
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)
        
        return frame
    
    def _draw_live_skeleton_with_errors(self, frame: np.ndarray) -> np.ndarray:
        """Draw skeleton with color-coded error highlighting."""
        if self.last_keypoints is None or self.last_deviations is None:
            return frame
        
        frame = self.pose_adapter.draw_skeleton(frame, self.last_keypoints)
        
        joint_map = {
            "left_elbow_angle": 7,
            "right_elbow_angle": 8,
            "left_knee_angle": 13,
            "right_knee_angle": 14,
            "left_hip_angle": 11,
            "right_hip_angle": 12,
            "left_shoulder_angle": 5,
            "right_shoulder_angle": 6,
        }
        
        for feature, deviation in self.last_deviations.items():
            tol = self.current_state.feature_tolerances.get(feature)
            if tol is None or feature not in joint_map:
                continue
            
            idx = joint_map[feature]
            x, y, conf = self.last_keypoints[idx]
            
            if conf < 0.3:
                continue
            
            error_ratio = abs(deviation) / tol
            
            if error_ratio > 1.0:
                color = (0, 0, 255)
                radius = 12
            elif error_ratio > 0.7:
                color = (0, 255, 255)
                radius = 10
            else:
                color = (0, 255, 0)
                radius = 8
            
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
            cv2.circle(frame, (int(x), int(y)), radius+2, (255, 255, 255), 2)
        
        return frame
    
    def _annotate_frame(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """Improved annotation with better layout and no rep performance section."""
        h, w = frame.shape[:2]
        
        # Smaller right panel for better proportions
        panel_width = int(w * 0.22)
        total_width = w + panel_width
        
        canvas = np.zeros((h, total_width, 3), dtype=np.uint8)
        
        # Left side - Live feed with skeleton
        live = frame.copy()
        if self.last_keypoints is not None:
            live = self._draw_live_skeleton_with_errors(live)
        
        canvas[:, :w] = live
        
        # RIGHT PANEL - Reference Pose
        ref_panel = np.ones((h, panel_width, 3), dtype=np.uint8) * 250
        
        # Subtle gradient
        for i in range(100):
            color_val = 250 - int(i * 0.5)
            cv2.line(ref_panel, (0, i), (panel_width, i), (color_val, color_val, color_val), 1)
        
        cv2.putText(ref_panel, "TARGET POSE",
                   (panel_width//2 - 80, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (40, 40, 40), 2)
        
        current_state_id = self.temporal_sequence[self.sequence_index]
        cv2.putText(ref_panel, f"State {current_state_id + 1}",
                   (panel_width//2 - 40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        
        if self.current_state:
            ref_kpts_raw = getattr(self.current_state, "mean_keypoints", None)
            if ref_kpts_raw is not None:
                ref_kpts = np.array(ref_kpts_raw)
                ref_panel = self._draw_reference_skeleton(ref_panel, ref_kpts)
        
        pose_indicator_y = h - 50
        cv2.rectangle(ref_panel, (20, pose_indicator_y - 30), (panel_width - 20, pose_indicator_y + 20), 
                      (240, 240, 240), -1)
        cv2.rectangle(ref_panel, (20, pose_indicator_y - 30), (panel_width - 20, pose_indicator_y + 20), 
                      (200, 200, 200), 2)
        
        pose_text = f"Pose {status['sequence_index']+1}/{len(self.temporal_sequence)}"
        text_size = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (panel_width - text_size[0]) // 2
        cv2.putText(ref_panel, pose_text,
                   (text_x, pose_indicator_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)
        
        canvas[:, w:] = ref_panel
        
        # Separator line
        cv2.line(canvas, (w, 0), (w, h), (180, 180, 180), 2)
        
        # Calculate accuracy
        if self.accuracy_active and self.evaluated_frames > 0:
            current_accuracy = int((self.green_frames / self.evaluated_frames) * 100)
        else:
            current_accuracy = 0
        
        # CIRCULAR ACCURACY METER (positioned in upper left, above streak)
        meter_x = 130
        meter_y = 180
        meter_radius = 70
        
        for i in range(5):
            cv2.circle(canvas, (meter_x, meter_y), meter_radius + 10 - i, 
                      (40 + i*10, 40 + i*10, 40 + i*10), 1)
        
        cv2.circle(canvas, (meter_x, meter_y), meter_radius + 3, (60, 60, 60), -1)
        cv2.circle(canvas, (meter_x, meter_y), meter_radius, (35, 35, 35), -1)
        
        if current_accuracy > 0:
            if current_accuracy >= 85:
                arc_color = (50, 205, 50)
            elif current_accuracy >= 70:
                arc_color = (0, 215, 255)
            else:
                arc_color = (0, 69, 255)
            
            angle = int(current_accuracy * 3.6)
            
            for i in range(10):
                radius = meter_radius - i
                cv2.ellipse(canvas, (meter_x, meter_y), (radius, radius), 
                           -90, 0, angle, arc_color, 2)
        
        cv2.circle(canvas, (meter_x, meter_y), meter_radius - 15, (45, 45, 45), -1)
        
        acc_text = str(current_accuracy)
        text_size = cv2.getTextSize(acc_text, cv2.FONT_HERSHEY_DUPLEX, 2.2, 4)[0]
        text_x = meter_x - text_size[0] // 2
        text_y = meter_y + text_size[1] // 2
        
        cv2.putText(canvas, acc_text, (text_x + 2, text_y + 2),
                   cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 4)
        cv2.putText(canvas, acc_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 4)
        
        label_text = "ACCURACY"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        label_x = meter_x - label_size[0] // 2
        cv2.putText(canvas, label_text, (label_x, meter_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 2)
        
        # STATUS HEADER
        header_height = 90
        
        for i in range(header_height):
            gray_val = 25 + int(i * 0.1)
            cv2.line(canvas, (0, i), (w - 180, i), (gray_val, gray_val, gray_val), 1)
        
        if status["is_correct"]:
            status_text = "CORRECT FORM"
            color = (50, 255, 50)
            icon = "OK"
        else:
            status_text = "ADJUST FORM"
            color = (0, 100, 255)
            icon = "!!"
        
        if status.get("in_grace", False):
            status_text = "TRANSITIONING..."
            color = (0, 255, 255)
            icon = ">>"
        
        cv2.rectangle(canvas, (15, 15), (70, 70), color, 3)
        cv2.rectangle(canvas, (18, 18), (67, 67), (40, 40, 40), -1)
        cv2.putText(canvas, icon, (25, 55),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
        
        cv2.putText(canvas, status_text, (85, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3)
        
        # Reps counter
        info_x = w - 420
        reps_text = f"REPS: {status['reps_completed']}"
        if self.max_reps:
            reps_text += f"/{self.max_reps}"
        
        cv2.rectangle(canvas, (info_x - 10, 15), (info_x + 200, 70), (50, 50, 50), -1)
        cv2.rectangle(canvas, (info_x - 10, 15), (info_x + 200, 70), (100, 100, 100), 2)
        cv2.putText(canvas, reps_text,
                   (info_x + 20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        
        # FEEDBACK BANNER (repositioned to bottom area)
        if status["feedback"]:
            fb_y = h - 60
            text_size = cv2.getTextSize(status["feedback"],
                                       cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
            text_x = (w - text_size[0]) // 2
            
            padding = 25
            cv2.rectangle(canvas,
                         (text_x - padding + 4, fb_y - 42 + 4),
                         (text_x + text_size[0] + padding + 4, fb_y + 15 + 4),
                         (0, 0, 0), -1)
            
            cv2.rectangle(canvas,
                         (text_x - padding, fb_y - 42),
                         (text_x + text_size[0] + padding, fb_y + 15),
                         (40, 40, 40), -1)
            
            border_color = (50, 255, 50) if status["is_correct"] else (0, 215, 255)
            cv2.rectangle(canvas,
                         (text_x - padding, fb_y - 42),
                         (text_x + text_size[0] + padding, fb_y + 15),
                         border_color, 3)
            
            cv2.putText(canvas, status["feedback"], (text_x, fb_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # STATS PANEL (simplified, bottom left)
        stats_x = 30
        stats_y = h - 150
        
        cv2.rectangle(canvas, (stats_x - 10, stats_y - 40), (stats_x + 250, stats_y + 40),
                     (40, 40, 40), -1)
        cv2.rectangle(canvas, (stats_x - 10, stats_y - 40), (stats_x + 250, stats_y + 40),
                     (80, 80, 80), 2)
        
        streak_prefix = "FIRE" if status['streak'] >= 5 else "STREAK"
        streak_text = f"{streak_prefix}: {status['streak']}"
        streak_color = (50, 255, 50) if status['streak'] >= 5 else (200, 200, 200)
        cv2.putText(canvas, streak_text, (stats_x, stats_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, streak_color, 2)
        
        # FPS counter
        if len(self.frame_times) > 0:
            fps = 1.0 / np.mean(self.frame_times)
            cv2.putText(canvas, f"FPS: {fps:.0f}", (w - 100, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return canvas
    
    def _annotate_session_complete(self, frame: np.ndarray) -> np.ndarray:
        """Session complete screen."""
        h, w = frame.shape[:2]
        panel_width = int(w * 0.22)
        total_width = w + panel_width
        
        canvas = np.zeros((h, total_width, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        msgs = [
            "🎉 SESSION COMPLETE! 🎉",
            f"Reps: {self.total_reps}",
            f"Max Streak: {self.max_streak}",
            "",
            "Press 'q' to exit"
        ]
        
        start_y = h // 2 - 100
        for i, msg in enumerate(msgs):
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(canvas, msg, (text_x, start_y + i*45),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        return canvas


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast-Response Pose Monitor')
    parser.add_argument("--graph", required=True, help="Pose graph JSON file")
    parser.add_argument("--model", default="yolo", choices=['yolo', 'mediapipe', 'movenet'])
    parser.add_argument("--max-reps", type=int, default=None, help="Max repetitions")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-adaptive", action='store_true', help="Disable adaptive features")
    parser.add_argument("--speed", default="fast", choices=['fast', 'normal', 'strict'],
                       help="Speed mode: fast (15%% hold), normal (25%%), strict (40%%)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FAST-RESPONSE POSE MONITORING SYSTEM")
    print("="*60 + "\n")
    
    try:
        graph = PoseGraph.load_json(args.graph)
        print(f"✓ Loaded graph: {args.graph}\n")
    except Exception as e:
        print(f"✗ Error loading graph: {e}")
        return 1
    
    try:
        monitor = EnhancedPoseMonitor(
            pose_graph=graph,
            pose_model=args.model,
            max_reps=args.max_reps,
            enable_adaptive=not args.no_adaptive,
            speed_mode=args.speed
        )
    except Exception as e:
        print(f"✗ Error creating monitor: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"✗ Camera {args.camera} failed to open")
        return 1
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera ready\n")
    
    print("="*60)
    print("READY TO START")
    print("="*60)
    print("\n💡 Tips:")
    print("  • Full body must be visible")
    print("  • 2-3 meters from camera")
    print("  • Good lighting")
    print("  • Watch colored joints (green=good, yellow=close, red=off)")
    print("  • Press 'q' to quit")
    print("  • Press 'f' to toggle fullscreen\n")
    
    if args.max_reps:
        print(f"🎯 Goal: {args.max_reps} reps\n")
    
    print("Starting in 3 seconds...\n")
    cv2.waitKey(3000)
    
    monitor.session_start_time = time.time()
    
    window_name = "Fast-Response Pose Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)
    
    fullscreen = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            annotated, status = monitor.process_frame(frame)
            cv2.imshow(window_name, annotated)
            
            if status.get("session_complete", False):
                cv2.waitKey(2000)
                break
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                print("\nℹ️  Stopped by user")
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    except KeyboardInterrupt:
        print("\nℹ️  Interrupted")
    except Exception as e:
        print(f"\nâœ— Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nExiting...\n")
    
    return 0


if __name__ == "__main__":
    exit(main())