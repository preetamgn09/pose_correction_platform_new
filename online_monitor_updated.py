"""
Online Pose Monitoring Engine (FINAL VERSION)
Real-time pose correction and guidance using pre-generated pose graph.

FEATURES:
- TWO-METRIC SYSTEM: Accuracy (binary correctness) + Quality (per-state refinement)
- MULTI-MODEL SUPPORT: YOLOv8, MediaPipe, MoveNet via unified adapter interface
- SEMANTIC STATE TRACKING: Intelligent rep counting with transition gating
- PER-STATE QUALITY: Motivating, recoverable quality scores that reset per pose
- OPTIMIZED INFERENCE: Single model call per frame with caching
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import time

from pose_features import PoseFeatureExtractor
from pose_graph import PoseGraph
from state_discovery import PoseState
from pose_adapters import create_pose_adapter, PoseModelAdapter


class BiomechanicalFeedbackGenerator:
    """
    Coach-style biomechanical feedback generator.

    Philosophy:
    - Think in movement patterns, not joint angles
    - Prefer posture + stability before limb refinement
    - Use coordination and temporal trends
    - Beginner-friendly, calm instructions
    
    FIXED: All methods now safely handle missing features
    """

    def __init__(self, history_length: int = 15):
        self.feature_history: List[Dict[str, float]] = []
        self.history_length = history_length

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
        """Generate realistic instructor-style feedback."""

        # --- 1. Stability check (temporal) ---
        if self._is_unstable(current_features):
            return "Pause and stabilize your body before adjusting further"

        # --- 2. Posture / torso priority ---
        torso_msg = self._torso_feedback(current_features, mean_features, tolerances)
        if torso_msg:
            return torso_msg

        # --- 3. Lower body chain (hips + knees together) ---
        lower_body_msg = self._lower_body_feedback(current_features, mean_features, tolerances)
        if lower_body_msg:
            return lower_body_msg

        # --- 4. Upper body coordination (arms + shoulders) ---
        upper_body_msg = self._upper_body_feedback(current_features, mean_features, tolerances)
        if upper_body_msg:
            return upper_body_msg

        # --- 5. Minimal positive reinforcement ---
        return "Good. Hold this shape steadily"

    def _is_unstable(self, features: Dict[str, float]) -> bool:
        """Check for temporal instability in torso angle."""
        if len(self.feature_history) < 5:
            return False

        torso_vals = [f.get("torso_angle") for f in self.feature_history if "torso_angle" in f]
        
        if len(torso_vals) < 3:
            return False
        
        return np.std(torso_vals) > 6.0

    def _torso_feedback(self, current, target, tol) -> Optional[str]:
        """Provide feedback on torso alignment."""
        if "torso_angle" not in current or "torso_angle" not in target or "torso_angle" not in tol:
            return None
        
        delta = current["torso_angle"] - target["torso_angle"]
        if abs(delta) > tol["torso_angle"]:
            if delta > 0:
                return "Lift your chest and stay more upright"
            else:
                return "Avoid leaning too far back; stack your torso over your hips"
        return None

    def _lower_body_feedback(self, current, target, tol) -> Optional[str]:
        """Provide feedback on lower body (hips and knees)."""
        has_left_knee = all(k in d for d in [current, target, tol] for k in ["left_knee_angle"])
        has_right_knee = all(k in d for d in [current, target, tol] for k in ["right_knee_angle"])
        
        if has_left_knee or has_right_knee:
            left_knee = current.get("left_knee_angle", 0) - target.get("left_knee_angle", 0)
            right_knee = current.get("right_knee_angle", 0) - target.get("right_knee_angle", 0)
            
            left_tol = tol.get("left_knee_angle", float('inf'))
            right_tol = tol.get("right_knee_angle", float('inf'))
            
            if abs(left_knee) > left_tol or abs(right_knee) > right_tol:
                return "Keep your knees soft and aligned over your feet"
        
        has_hip_angles = all(k in d for d in [current, target, tol] for k in ["left_hip_angle", "right_hip_angle"])
        
        if has_hip_angles:
            hip_depth = (
                current["left_hip_angle"] + current["right_hip_angle"]
            ) / 2 - (
                target["left_hip_angle"] + target["right_hip_angle"]
            ) / 2
            
            hip_tol = (tol["left_hip_angle"] + tol["right_hip_angle"]) / 2
            
            if abs(hip_depth) > hip_tol:
                if hip_depth < 0:
                    return "Sink a little deeper through your hips"
                else:
                    return "Lift slightly out of the hips to reduce strain"
        
        return None

    def _upper_body_feedback(self, current, target, tol) -> Optional[str]:
        """Provide feedback on upper body (arms and shoulders)."""
        has_arms = all(k in d for d in [current, target, tol] for k in ["left_arm_elevation", "right_arm_elevation"])
        
        if has_arms:
            left_arm = current["left_arm_elevation"] - target["left_arm_elevation"]
            right_arm = current["right_arm_elevation"] - target["right_arm_elevation"]

            if abs(left_arm - right_arm) > 0.15:
                return "Balance both arms evenly"

            left_tol = tol["left_arm_elevation"]
            if abs(left_arm) > left_tol:
                if left_arm < 0:
                    return "Reach your arms a bit higher"
                else:
                    return "Soften your arms slightly"
        
        has_shoulders = all(k in d for d in [current, target, tol] for k in ["left_shoulder_angle", "right_shoulder_angle"])
        
        if has_shoulders:
            shoulder_delta = (
                current["left_shoulder_angle"] + current["right_shoulder_angle"]
            ) / 2 - (
                target["left_shoulder_angle"] + target["right_shoulder_angle"]
            ) / 2
            
            shoulder_tol = (tol["left_shoulder_angle"] + tol["right_shoulder_angle"]) / 2

            if abs(shoulder_delta) > shoulder_tol:
                return "Relax your shoulders away from your ears"

        return None


class PoseMonitor:
    """
    Real-time pose monitoring system with TWO-METRIC EVALUATION.
    
    KEY FEATURES:
    - METRIC 1 - ACCURACY: Binary correctness (GREEN/RED)
    - METRIC 2 - QUALITY: Per-state refinement within GREEN poses
    - MULTI-MODEL: Supports YOLOv8, MediaPipe, MoveNet
    - Metrics are INDEPENDENT and never mixed
    - Accuracy paused during transitions
    - Quality resets per state and is motivating/recoverable
    """
    
    def __init__(
        self,
        pose_graph: PoseGraph,
        pose_model: str = 'yolo',
        model_name: str = 'yolov8n-pose.pt',
        confidence_threshold: float = 0.5,
        feedback_delay: float = 0.5,
        max_reps: Optional[int] = None
    ):
        """
        Args:
            pose_graph: Pre-generated PoseGraph object
            pose_model: Pose estimation model type ('yolo', 'mediapipe', 'movenet')
            model_name: Model-specific name/path
            confidence_threshold: Minimum keypoint confidence
            feedback_delay: Minimum time between feedback messages (seconds)
            max_reps: Maximum repetitions before auto-stop (None = unlimited)
        """
        print(f"Initializing Pose Monitor...")
        
        self.graph = pose_graph
        
        # Create pose model adapter (unified interface for all models)
        # This keeps the pose graph and all downstream logic model-agnostic
        self.pose_adapter = create_pose_adapter(
            pose_model,
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )
        
        self.feature_extractor = PoseFeatureExtractor(
            confidence_threshold=confidence_threshold
        )
        self.feedback_generator = BiomechanicalFeedbackGenerator()
        
        # TEMPORAL SEQUENCE EXECUTION
        self.temporal_sequence = self.graph.get_metadata('temporal_sequence')
        
        if not self.temporal_sequence:
            raise ValueError(
                "Pose graph missing 'temporal_sequence' metadata. "
                "Please regenerate the graph with updated offline_graph_generator.py"
            )
        
        print(f"  Loaded temporal sequence: {len(self.temporal_sequence)} states per rep")
        
        # State machine variables
        self.sequence_index: int = 0
        self.current_state: Optional[PoseState] = None
        self.current_state_start_time: float = 0.0
        self.current_state_hold_time: float = 0.0
        self.total_reps: int = 0
        
        # SEMANTIC STATE TRACKING
        self.previous_state_id: Optional[int] = None
        
        # MAX REPS SUPPORT
        self.max_reps = max_reps
        self.session_complete = False
        
        if self.max_reps:
            print(f"  Max reps: {self.max_reps}")
        else:
            print(f"  Max reps: unlimited")
        
        # ═══════════════════════════════════════════════════════════════
        # TWO-METRIC EVALUATION SYSTEM
        # ═══════════════════════════════════════════════════════════════
        # METRIC 1: ACCURACY - Binary correctness consistency
        # METRIC 2: QUALITY - Per-state refinement within correct poses
        # These metrics are INDEPENDENT and NEVER mixed mathematically
        
        # ACCURACY TRACKING (Percent-in-Tolerance)
        # Measures: "How consistently did I perform the pose correctly?"
        # - Binary: GREEN (correct) or RED (incorrect)
        # - No deviation calculation, no penalties, no time dependency
        self.accuracy_active: bool = False      # Activates after first GREEN
        self.green_frames: int = 0              # Count of GREEN frames
        self.evaluated_frames: int = 0          # Count of evaluated frames
        
        # QUALITY TRACKING (Per-State Refinement)
        # Measures: "How well did I perform when I was correct?"
        # - Computed ONLY when pose is GREEN
        # - RESETS when state changes (per-state metric)
        # - Uses short sliding window for stability within state
        # - Does NOT accumulate across states
        self.quality_active: bool = False           # Activates when accuracy activates
        
        # PER-STATE QUALITY (resets on state change)
        self.current_state_quality_buffer: List[float] = []  # Sliding window for current state
        self.quality_window_size: int = 10          # Frames to smooth (motivating, not punitive)
        self.current_quality: float = 0.0           # Current smoothed state quality
        self.current_frame_quality: float = 0.0     # Instantaneous single-frame quality
        
        # COMPLETED STATE QUALITIES (for final session average)
        self.completed_state_qualities: List[float] = []  # One score per completed state
        
        # TRANSITION GATING
        # Pauses accuracy tracking during pose transitions
        self.in_transition: bool = False        # True during pose transitions
        
        # PER-FRAME KEYPOINT CACHE
        # CRITICAL: adapter.detect() is expensive and must run ONCE per frame
        # All downstream consumers reuse this cached result
        self.last_keypoints: Optional[np.ndarray] = None
        
        # Feedback control
        self.feedback_delay = feedback_delay
        self.last_feedback_time: float = 0.0
        self.current_feedback: str = ""
        
        # Performance tracking
        self.frame_times: List[float] = []
        self.session_start_time: float = 0.0
        
        # Initialize to first state in temporal sequence
        self._reset_to_initial_state()
        
        print(f"  Loaded graph with {len(self.graph.states)} unique states")
        print(f"  Starting at State {self.current_state.state_id}")
        print(f"  Execution mode: DETERMINISTIC (follows temporal sequence)")
        print(f"  Metrics: Two-metric system (Accuracy + Quality)")
        print(f"    - Accuracy: Binary correctness (GREEN/RED)")
        print(f"    - Quality: Per-state refinement (GREEN-only, recoverable)")
        print("Pose Monitor ready!")
    
    def _reset_to_initial_state(self) -> None:
        """
        Reset state machine to initial state.
        
        DETERMINISTIC: Uses temporal_sequence[0], not graph.get_initial_state()
        SEMANTIC: Resets previous_state_id to None for new rep
        """
        self.sequence_index = 0
        initial_state_id = self.temporal_sequence[0]
        self.current_state = self.graph.get_state(initial_state_id)
        self.current_state_start_time = time.time()
        self.current_state_hold_time = 0.0
        
        # SEMANTIC STATE TRACKING: Reset previous state to detect first transition
        self.previous_state_id = None
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single video frame.
        
        TWO-METRIC SYSTEM:
        - ACCURACY: Counts GREEN vs RED frames (binary)
        - QUALITY: Measures per-state refinement within GREEN frames
        
        PERFORMANCE: Pose inference runs ONCE per frame and is cached
        to avoid redundant model calls in visualization.
        
        Args:
            frame: Input BGR image from camera
        
        Returns:
            (annotated_frame, status_dict)
        """
        start_time = time.time()
        
        # Check if session is complete
        if self.session_complete:
            annotated = self._annotate_session_complete(frame)
            return annotated, {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': 1.0,
                'is_correct': True,
                'feedback': "Session complete!",
                'reps_completed': self.total_reps,
                'session_complete': True
            }
        
        # CRITICAL: Run pose inference ONCE per frame and cache result
        # This prevents redundant adapter.detect() calls in visualization
        # Performance impact: ~2x speedup for inference-bound scenarios
        self.last_keypoints = self.pose_adapter.detect(frame)
        
        # Extract features from cached keypoints
        features = None
        if self.last_keypoints is not None:
            features = self.feature_extractor.extract_features(self.last_keypoints)
        
        # Update state machine (handles both metrics internally)
        status = self._update_state_machine(features)
        
        # Annotate frame with visual feedback (uses cached keypoints, no re-inference)
        annotated_frame = self._annotate_frame(frame, status)
        
        # Track performance
        self.frame_times.append(time.time() - start_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        return annotated_frame, status
    
    def _update_state_machine(self, features: Optional[Dict[str, float]]) -> Dict:
        """
        Update state machine based on current features.
        
        TWO-METRIC SYSTEM IMPLEMENTATION:
        
        METRIC 1 - ACCURACY (Binary Correctness):
        - Activates on first GREEN frame (counts that frame)
        - GREEN frame → increment both green_frames and evaluated_frames
        - RED frame → increment only evaluated_frames
        - PAUSED during transitions (in_transition gate)
        
        METRIC 2 - QUALITY (Per-State Refinement):
        - Computed ONLY when pose is GREEN
        - Uses sliding window for stability
        - Resets when leaving GREEN (recoverable)
        - Resets when changing state (per-state)
        
        Returns:
            Status dictionary with current state info
        """
        current_time = time.time()
        
        if features is None or not features:
            # No pose detected - no metric updates
            self.current_feedback = "No pose detected. Step into view of the camera."
            return {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': 0.0,
                'is_correct': False,
                'feedback': self.current_feedback,
                'reps_completed': self.total_reps,
                'sequence_index': self.sequence_index
            }
        
        # Update feedback history
        self.feedback_generator.update_history(features)
        
        # Check if current pose matches current state
        matches, deviations = self.current_state.matches(features, strict=False)
        
        if matches:
            # ═══════════════════════════════════════════════════════════════
            # POSE IS GREEN (WITHIN TOLERANCE)
            # ═══════════════════════════════════════════════════════════════
            
            # METRIC 1: ACCURACY (Binary Correctness)
            # Activate on first GREEN and COUNT this first frame
            if not self.accuracy_active:
                self.accuracy_active = True
                self.quality_active = True
                print("  [METRICS] Activated: First GREEN pose detected")
            
            # Clear transition gate when pose is GREEN
            if self.in_transition:
                self.in_transition = False
                print("  [TRANSITION] Completed, accuracy resumed")
            
            # Count GREEN frames ONLY if not in transition
            if not self.in_transition:
                self.evaluated_frames += 1
                self.green_frames += 1
            
            # METRIC 2: QUALITY (Per-State Refinement)
            # Compute quality ONLY when pose is GREEN
            # Quality is PER-STATE and resets on state change
            if self.quality_active:
                frame_quality = self._compute_quality_score(
                    features,
                    self.current_state.mean_features,
                    self.current_state.feature_tolerances
                )
                
                # Store instantaneous frame quality for display
                self.current_frame_quality = frame_quality
                
                # Add to current state's sliding window
                self.current_state_quality_buffer.append(frame_quality)
                
                # Keep only recent frames (sliding window for stability)
                if len(self.current_state_quality_buffer) > self.quality_window_size:
                    self.current_state_quality_buffer.pop(0)
                
                # Current quality = mean of sliding window (motivating, recoverable)
                # This allows quality to INCREASE when user corrects posture
                # and NOT decay during a correct hold
                self.current_quality = np.mean(self.current_state_quality_buffer)
            
            # Accumulate hold time
            self.current_state_hold_time = current_time - self.current_state_start_time
            hold_progress = self.current_state_hold_time / self.current_state.min_hold_duration
            
            if hold_progress >= 1.0:
                # State completed, transition to next
                self._transition_to_next_state_deterministic()
                self.current_feedback = "Excellent! Moving to next pose..."
            else:
                # Still holding - provide positive reinforcement
                remaining = self.current_state.min_hold_duration - self.current_state_hold_time
                self.current_feedback = f"Hold steady... {remaining:.1f}s remaining"
            
            return {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': min(hold_progress, 1.0),
                'is_correct': True,
                'feedback': self.current_feedback,
                'reps_completed': self.total_reps,
                'sequence_index': self.sequence_index
            }
        else:
            # ═══════════════════════════════════════════════════════════════
            # POSE IS RED (OUTSIDE TOLERANCE)
            # ═══════════════════════════════════════════════════════════════
            
            # METRIC 1: ACCURACY
            # Count as incorrect frame (if accuracy active AND not in transition)
            if self.accuracy_active and not self.in_transition:
                self.evaluated_frames += 1
                # Do NOT increment green_frames
            
            # METRIC 2: QUALITY
            # FIX 1: Reset quality buffer when leaving GREEN
            # This ensures recoverability - user gets fresh start when returning to GREEN
            if len(self.current_state_quality_buffer) > 0:
                self.current_state_quality_buffer = []
                self.current_quality = 0.0
                self.current_frame_quality = 0.0
            
            # Reset hold time
            self.current_state_hold_time = 0.0
            self.current_state_start_time = current_time
            
            # Generate enhanced feedback (rate-limited)
            if current_time - self.last_feedback_time > self.feedback_delay:
                self.current_feedback = self.feedback_generator.generate_feedback(
                    deviations=deviations,
                    mean_features=self.current_state.mean_features,
                    current_features=features,
                    tolerances=self.current_state.feature_tolerances
                )
                self.last_feedback_time = current_time
            
            return {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': 0.0,
                'is_correct': False,
                'feedback': self.current_feedback,
                'reps_completed': self.total_reps,
                'sequence_index': self.sequence_index
            }
    
    def _compute_quality_score(
        self,
        observed_features: Dict[str, float],
        reference_features: Dict[str, float],
        tolerances: Dict[str, float]
    ) -> float:
        """
        Compute quality score for current GREEN frame.
        
        QUALITY METRIC:
        - Measures refinement within correct pose (GREEN only)
        - Uses normalized angle deviations
        - Does NOT affect accuracy
        - Range: 0-100 (100 = perfect match to reference)
        
        Formula:
            For each feature:
                normalized_error = abs(observed - reference) / tolerance
            
            mean_error = mean(normalized_errors)
            quality_score = max(0, 100 - (mean_error * 100))
        
        Args:
            observed_features: Current user pose features
            reference_features: Target pose features
            tolerances: Acceptable deviations
        
        Returns:
            Quality score [0, 100]
        """
        normalized_errors = []
        
        for feature_name in reference_features.keys():
            # Skip missing features
            if feature_name not in observed_features:
                continue
            
            if feature_name not in tolerances:
                continue
            
            tolerance = tolerances[feature_name]
            
            # Skip zero tolerance to avoid division by zero
            if tolerance == 0:
                continue
            
            observed_val = observed_features[feature_name]
            reference_val = reference_features[feature_name]
            
            # Compute normalized error (0 = perfect, 1 = at tolerance boundary)
            normalized_error = abs(observed_val - reference_val) / tolerance
            normalized_errors.append(normalized_error)
        
        # If no valid features, return perfect quality
        if len(normalized_errors) == 0:
            return 100.0
        
        # Compute mean normalized error
        mean_error = np.mean(normalized_errors)
        
        # Convert to quality score (100 = perfect, 0 = at/beyond tolerance)
        quality_score = max(0.0, 100.0 - (mean_error * 100.0))
        
        # Clamp to [0, 100]
        quality_score = np.clip(quality_score, 0.0, 100.0)
        
        return quality_score
    
    def _transition_to_next_state_deterministic(self) -> None:
        """
        Transition to the next state using TEMPORAL SEQUENCE.
        
        CRITICAL SEMANTIC STATE HANDLING:
        - Only updates previous_state_id when state_id ACTUALLY CHANGES
        - This prevents double-counting when first and last states are the same
        
        ACCURACY GATING:
        - Pauses accuracy tracking during transitions
        
        QUALITY FINALIZATION:
        - Finalizes per-state quality before transition
        - Resets quality buffer for next state
        """
        # Advance sequence index
        self.sequence_index += 1
        
        # Check if we completed one full repetition
        if self.sequence_index >= len(self.temporal_sequence):
            # One rep completed
            self.total_reps += 1
            
            # Check if we reached max_reps
            if self.max_reps and self.total_reps >= self.max_reps:
                # FIX 2: Finalize last state quality before session completion
                # The last pose state must contribute to session average
                if self.quality_active and len(self.current_state_quality_buffer) > 0:
                    state_quality = np.mean(self.current_state_quality_buffer)
                    self.completed_state_qualities.append(state_quality)
                    print(f"  [QUALITY] Final state {self.current_state.state_id} completed with quality: {state_quality:.1f}%")
                    self.current_state_quality_buffer = []
                
                # Session complete!
                self.session_complete = True
                self._print_session_summary()
                return
            
            # Loop back to start for next rep
            self.sequence_index = 0
        
        # Get next state from temporal sequence (DETERMINISTIC)
        next_state_id = self.temporal_sequence[self.sequence_index]

        # SEMANTIC FIX: Do NOT reset if pose did not change
        if next_state_id == self.current_state.state_id:
            # Same semantic pose – continue holding
            return

        next_state = self.graph.get_state(next_state_id)

        if next_state is None:
            raise RuntimeError(f"Invalid state_id {next_state_id} in temporal_sequence")

        # ═══════════════════════════════════════════════════════════════
        # FINALIZE PER-STATE QUALITY BEFORE TRANSITION
        # ═══════════════════════════════════════════════════════════════
        # Quality is PER-STATE: finalize current state's quality
        # and reset buffer for next state
        if self.quality_active and len(self.current_state_quality_buffer) > 0:
            # Finalize this state's quality (mean of sliding window)
            state_quality = np.mean(self.current_state_quality_buffer)
            self.completed_state_qualities.append(state_quality)
            print(f"  [QUALITY] State {self.current_state.state_id} completed with quality: {state_quality:.1f}%")
            
            # Reset quality buffer for next state
            # This ensures quality is PER-STATE and recoverable
            self.current_state_quality_buffer = []
            self.current_quality = 0.0
            self.current_frame_quality = 0.0
        
        # ═══════════════════════════════════════════════════════════════
        # PAUSE ACCURACY DURING TRANSITION
        # ═══════════════════════════════════════════════════════════════
        # Set transition gate to pause accuracy tracking
        self.in_transition = True
        print(f"  [TRANSITION] Moving to State {next_state_id}, accuracy paused")
        
        # Transition ONLY when semantic pose changes
        self.previous_state_id = self.current_state.state_id
        self.current_state = next_state
        self.current_state_start_time = time.time()
        self.current_state_hold_time = 0.0
    
    def _print_session_summary(self) -> None:
        """
        Print final session statistics when max_reps reached.
        
        TWO-METRIC SYSTEM: Shows both accuracy and quality separately.
        """
        print("\n" + "="*60)
        print("SESSION COMPLETE!")
        print("="*60)
        
        session_duration = time.time() - self.session_start_time
        
        print(f"\nSession Statistics:")
        print(f"  Repetitions Completed: {self.total_reps}")
        print(f"  Total Session Duration: {session_duration:.1f}s ({session_duration/60:.1f} min)")
        
        # Compute final metrics
        if self.evaluated_frames > 0:
            accuracy = (self.green_frames / self.evaluated_frames) * 100
        else:
            accuracy = 100.0
        
        # Per-state quality: average of all completed state qualities
        if len(self.completed_state_qualities) > 0:
            quality = np.mean(self.completed_state_qualities)
        else:
            quality = 0.0
        
        print(f"\nPose Quality Metrics (Two-Metric System):")
        print(f"\n  METRIC 1: ACCURACY (Binary Correctness)")
        print(f"    Total Frames Evaluated: {self.evaluated_frames}")
        print(f"    GREEN Frames (Correct): {self.green_frames}")
        print(f"    RED Frames (Incorrect): {self.evaluated_frames - self.green_frames}")
        print(f"    Accuracy Score: {accuracy:.1f}%")
        
        print(f"\n  METRIC 2: QUALITY (Per-State Refinement)")
        print(f"    States Completed: {len(self.completed_state_qualities)}")
        print(f"    Average Quality Score: {quality:.1f}%")
        
        print(f"\n  INTERPRETATION:")
        print(f"    Accuracy tells you how consistently you stayed within tolerance")
        print(f"    Quality tells you how well you refined your poses when correct")
        
        # Performance stats
        if len(self.frame_times) > 0:
            avg_fps = 1.0 / np.mean(self.frame_times)
            print(f"\nPerformance:")
            print(f"  Average FPS: {avg_fps:.1f}")
        
        print("\n" + "="*60)
    
    def _annotate_frame(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """
        Annotate frame with visual feedback.
        
        CRITICAL: Uses CACHED keypoints (self.last_keypoints)
        Does NOT call adapter.detect() again.
        
        Args:
            frame: Input frame
            status: Status dictionary from _update_state_machine
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw skeleton using cached keypoints
        if self.last_keypoints is not None:
            annotated = self.pose_adapter.draw_skeleton(annotated, self.last_keypoints)
        
        # Determine status color
        if status['is_correct']:
            status_color = (0, 255, 0)  # GREEN
            status_text = "CORRECT"
        else:
            status_color = (0, 0, 255)  # RED
            status_text = "ADJUST"
        
        # Draw status banner
        cv2.rectangle(annotated, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            status_text,
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            status_color,
            3
        )
        
        # Draw state info
        state_text = f"State {status['state_id']} ({status['sequence_index'] + 1}/{len(self.temporal_sequence)})"
        cv2.putText(
            annotated,
            state_text,
            (w - 300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw reps counter
        reps_text = f"Reps: {status['reps_completed']}"
        if self.max_reps:
            reps_text += f"/{self.max_reps}"
        cv2.putText(
            annotated,
            reps_text,
            (w - 300, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw hold progress bar (only when correct)
        if status['is_correct']:
            bar_width = 200
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = 90
            
            # Background
            cv2.rectangle(
                annotated,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Progress
            progress_width = int(bar_width * status['hold_progress'])
            cv2.rectangle(
                annotated,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                (0, 255, 0),
                -1
            )
            
            # Border
            cv2.rectangle(
                annotated,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2
            )
        
        # Draw metrics panel (bottom left)
        panel_y = h - 150
        
        # Metrics background
        cv2.rectangle(annotated, (0, panel_y), (400, h), (0, 0, 0), -1)
        
        # METRIC 1: ACCURACY
        if self.accuracy_active:
            if self.evaluated_frames > 0:
                accuracy = (self.green_frames / self.evaluated_frames) * 100
            else:
                accuracy = 100.0
            
            accuracy_text = f"Accuracy: {accuracy:.1f}%"
            accuracy_detail = f"({self.green_frames}/{self.evaluated_frames} frames)"
        else:
            accuracy_text = "Accuracy: --"
            accuracy_detail = "(waiting for first pose)"
        
        cv2.putText(
            annotated,
            accuracy_text,
            (20, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(
            annotated,
            accuracy_detail,
            (20, panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        # METRIC 2: QUALITY
        if self.quality_active and status['is_correct']:
            # Show both current state quality and instantaneous frame quality
            quality_text = f"Quality: {self.current_quality:.1f}%"
            quality_detail = f"(now: {self.current_frame_quality:.1f}%)"
        else:
            quality_text = "Quality: --"
            quality_detail = "(GREEN poses only)"
        
        cv2.putText(
            annotated,
            quality_text,
            (20, panel_y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(
            annotated,
            quality_detail,
            (20, panel_y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        # Draw feedback text (center bottom)
        if status['feedback']:
            feedback_lines = self._wrap_text(status['feedback'], 60)
            feedback_y = h - 180
            
            for i, line in enumerate(feedback_lines):
                text_size = cv2.getTextSize(
                    line,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )[0]
                text_x = (w - text_size[0]) // 2
                
                # Background for readability
                cv2.rectangle(
                    annotated,
                    (text_x - 10, feedback_y + i*30 - 25),
                    (text_x + text_size[0] + 10, feedback_y + i*30 + 5),
                    (0, 0, 0),
                    -1
                )
                
                cv2.putText(
                    annotated,
                    line,
                    (text_x, feedback_y + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
        
        return annotated
    
    def _annotate_session_complete(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotate frame when session is complete.
        
        Args:
            frame: Input frame
        
        Returns:
            Annotated frame with completion message
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw skeleton using cached keypoints if available
        if self.last_keypoints is not None:
            annotated = self.pose_adapter.draw_skeleton(annotated, self.last_keypoints)
        
        # Semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        # Completion message
        messages = [
            "SESSION COMPLETE!",
            f"Reps Completed: {self.total_reps}",
            "",
            "Great work!",
            "Press 'q' to exit"
        ]
        
        start_y = h // 2 - 100
        for i, msg in enumerate(messages):
            text_size = cv2.getTextSize(
                msg,
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                2
            )[0]
            text_x = (w - text_size[0]) // 2
            
            cv2.putText(
                annotated,
                msg,
                (text_x, start_y + i*50),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        
        return annotated
    
    def _wrap_text(self, text: str, max_length: int) -> List[str]:
        """
        Wrap text into multiple lines.
        
        Args:
            text: Input text
            max_length: Maximum characters per line
        
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def start_session(self) -> None:
        """Mark the start of a monitoring session."""
        self.session_start_time = time.time()
    
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.frame_times) == 0:
            return {'fps': 0.0, 'avg_frame_time': 0.0}
        
        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return {
            'fps': fps,
            'avg_frame_time': avg_frame_time * 1000,  # Convert to ms
            'frames_processed': len(self.frame_times)
        }


def main():
    """
    Example usage of PoseMonitor with live camera feed.
    
    Usage:
        python online_monitor.py --graph <path_to_graph.pkl> --model yolo --max-reps 5
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time pose monitoring')
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to pose graph pickle file')
    parser.add_argument('--model', type=str, default='yolo',
                        choices=['yolo', 'mediapipe', 'movenet'],
                        help='Pose estimation model')
    parser.add_argument('--model-name', type=str, default='yolov8n-pose.pt',
                        help='Model-specific name/path')
    parser.add_argument('--max-reps', type=int, default=None,
                        help='Maximum repetitions (None = unlimited)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    args = parser.parse_args()
    
    # Load pose graph
    print(f"Loading pose graph from {args.graph}...")
    graph = PoseGraph.load(args.graph)
    
    # Create monitor
    monitor = PoseMonitor(
        pose_graph=graph,
        pose_model=args.model,
        model_name=args.model_name,
        max_reps=args.max_reps
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print("\nStarting pose monitoring...")
    print("Press 'q' to quit")
    monitor.start_session()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            annotated_frame, status = monitor.process_frame(frame)
            
            # Display
            cv2.imshow('Pose Monitor', annotated_frame)
            
            # Check for exit or session complete
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            if status.get('session_complete', False):
                # Keep displaying completion screen until user quits
                continue
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats if not already printed
        if not monitor.session_complete:
            print("\nSession interrupted by user")


if __name__ == '__main__':
    main()