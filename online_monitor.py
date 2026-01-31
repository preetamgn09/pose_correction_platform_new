"""
Online Pose Monitoring Engine
Real-time pose correction and guidance using pre-generated pose graph.
Provides visual feedback and corrective instructions.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Dict, List, Tuple
import time

from pose_features import PoseFeatureExtractor
from pose_graph import PoseGraph
from state_discovery import PoseState


class PoseMonitor:
    """
    Real-time pose monitoring system using state machine approach.
    Tracks user progress through pose sequence and provides feedback.
    """
    
    def __init__(
        self,
        pose_graph: PoseGraph,
        model_name: str = 'yolov8n-pose.pt',
        confidence_threshold: float = 0.5,
        feedback_delay: float = 0.5
    ):
        """
        Args:
            pose_graph: Pre-generated PoseGraph object
            model_name: YOLOv8 pose model name
            confidence_threshold: Minimum keypoint confidence
            feedback_delay: Minimum time between feedback messages (seconds)
        """
        print(f"Initializing Pose Monitor...")
        
        self.graph = pose_graph
        self.model = YOLO(model_name)
        self.feature_extractor = PoseFeatureExtractor(
            confidence_threshold=confidence_threshold
        )
        
        # State machine variables
        self.current_state: Optional[PoseState] = None
        self.current_state_start_time: float = 0.0
        self.current_state_hold_time: float = 0.0
        self.completed_states: List[int] = []
        self.total_reps: int = 0
        
        # Feedback control
        self.feedback_delay = feedback_delay
        self.last_feedback_time: float = 0.0
        self.current_feedback: str = ""
        
        # Performance tracking
        self.frame_times: List[float] = []
        
        # Initialize to first state
        self._reset_to_initial_state()
        
        print(f"  Loaded graph with {len(self.graph.states)} states")
        print(f"  Starting at State {self.current_state.state_id}")
        print("Pose Monitor ready!")
    
    def _reset_to_initial_state(self) -> None:
        """Reset state machine to initial state."""
        self.current_state = self.graph.get_initial_state()
        self.current_state_start_time = time.time()
        self.current_state_hold_time = 0.0
        self.completed_states = []
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single video frame.
        
        Args:
            frame: Input BGR image from camera
        
        Returns:
            (annotated_frame, status_dict)
        """
        start_time = time.time()
        
        # Run pose estimation
        results = self.model(frame, verbose=False)
        
        # Extract features
        features = self._extract_features_from_result(results[0])
        
        # Update state machine
        status = self._update_state_machine(features)
        
        # Annotate frame with visual feedback
        annotated_frame = self._annotate_frame(frame, results[0], status)
        
        # Track performance
        self.frame_times.append(time.time() - start_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        return annotated_frame, status
    
    def _extract_features_from_result(self, result) -> Optional[Dict[str, float]]:
        """Extract pose features from YOLO result."""
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        
        # Get first detected person
        keypoints = result.keypoints.data[0].cpu().numpy()
        
        # Extract features
        features = self.feature_extractor.extract_features(keypoints)
        
        return features
    
    def _update_state_machine(self, features: Optional[Dict[str, float]]) -> Dict:
        """
        Update state machine based on current features.
        
        Returns:
            Status dictionary with current state info
        """
        current_time = time.time()
        
        if features is None:
            # No pose detected
            self.current_feedback = "No pose detected. Please stand in view."
            return {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': 0.0,
                'is_correct': False,
                'feedback': self.current_feedback,
                'reps_completed': self.total_reps
            }
        
        # Check if current pose matches current state
        matches, deviations = self.current_state.matches(features, strict=False)
        
        if matches:
            # Pose is correct, accumulate hold time
            self.current_state_hold_time = current_time - self.current_state_start_time
            hold_progress = self.current_state_hold_time / self.current_state.min_hold_duration
            
            if hold_progress >= 1.0:
                # State completed, transition to next
                self._transition_to_next_state()
                self.current_feedback = "Good! Moving to next pose..."
            else:
                # Still holding
                self.current_feedback = f"Hold this pose... {hold_progress*100:.0f}%"
            
            return {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': min(hold_progress, 1.0),
                'is_correct': True,
                'feedback': self.current_feedback,
                'reps_completed': self.total_reps
            }
        else:
            # Pose is incorrect, provide corrective feedback
            self.current_state_hold_time = 0.0
            self.current_state_start_time = current_time
            
            # Generate feedback (rate-limited)
            if current_time - self.last_feedback_time > self.feedback_delay:
                self.current_feedback = self._generate_corrective_feedback(deviations)
                self.last_feedback_time = current_time
            
            return {
                'state_id': self.current_state.state_id,
                'state_name': f"State {self.current_state.state_id}",
                'hold_progress': 0.0,
                'is_correct': False,
                'feedback': self.current_feedback,
                'reps_completed': self.total_reps
            }
    
    def _transition_to_next_state(self) -> None:
        """Transition to the next state in the sequence."""
        # Mark current state as completed
        if self.current_state.state_id not in self.completed_states:
            self.completed_states.append(self.current_state.state_id)
        
        # Get next state
        next_state_ids = self.graph.get_next_states(self.current_state.state_id)
        
        if not next_state_ids:
            # No next state (end of sequence)
            self._reset_to_initial_state()
            self.total_reps += 1
            return
        
        # Transition to next state (take first valid transition)
        next_state_id = next_state_ids[0]
        next_state = self.graph.get_state(next_state_id)
        
        if next_state is None:
            # Invalid state, reset
            self._reset_to_initial_state()
            return
        
        # Check if we completed full sequence (back to start)
        if next_state.state_id == self.graph.get_initial_state().state_id:
            self.total_reps += 1
            self.completed_states = []
        
        self.current_state = next_state
        self.current_state_start_time = time.time()
        self.current_state_hold_time = 0.0
    
    def _generate_corrective_feedback(self, deviations: Dict[str, float]) -> str:
        """
        Generate human-readable corrective feedback based on feature deviations.
        
        Args:
            deviations: Dictionary of feature_name -> deviation_amount
        
        Returns:
            Corrective feedback string
        """
        # Find features with largest deviations (above tolerance)
        problem_features = []
        for feature_name, deviation in deviations.items():
            tolerance = self.current_state.feature_tolerances[feature_name]
            if deviation > tolerance:
                problem_features.append((feature_name, deviation, tolerance))
        
        if not problem_features:
            return "Adjust your pose slightly"
        
        # Sort by severity (deviation / tolerance ratio)
        problem_features.sort(key=lambda x: x[1] / x[2], reverse=True)
        
        # Generate feedback for top issue
        feature_name, deviation, tolerance = problem_features[0]
        
        # Map feature names to user-friendly corrections
        feedback_map = {
            'left_knee_angle': "Bend your left knee more",
            'right_knee_angle': "Bend your right knee more",
            'left_elbow_angle': "Adjust your left elbow angle",
            'right_elbow_angle': "Adjust your right elbow angle",
            'left_hip_angle': "Adjust your left hip position",
            'right_hip_angle': "Adjust your right hip position",
            'left_shoulder_angle': "Adjust your left shoulder",
            'right_shoulder_angle': "Adjust your right shoulder",
            'torso_angle': "Straighten your torso",
            'left_arm_elevation': "Adjust your left arm height",
            'right_arm_elevation': "Adjust your right arm height",
            'left_leg_spread': "Adjust your left leg position",
            'right_leg_spread': "Adjust your right leg position",
            'body_center_y': "Adjust your body height"
        }
        
        return feedback_map.get(feature_name, "Adjust your pose")
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        result,
        status: Dict
    ) -> np.ndarray:
        """
        Add visual annotations to frame.
        
        Args:
            frame: Input BGR image
            result: YOLO result object
            status: Status dictionary from state machine
        
        Returns:
            Annotated BGR image
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw skeleton if pose detected
        if result.keypoints is not None and len(result.keypoints) > 0:
            annotated = result.plot()
        
        # Create status overlay
        overlay = annotated.copy()
        
        # Status panel background
        panel_height = 180
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Status text
        y_offset = 40
        line_height = 35
        
        # State info
        state_text = f"State: {status['state_name']}"
        color = (0, 255, 0) if status['is_correct'] else (0, 165, 255)
        cv2.putText(annotated, state_text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y_offset += line_height
        
        # Progress bar
        progress = status['hold_progress']
        bar_width = 400
        bar_height = 30
        bar_x = 20
        bar_y = y_offset
        
        # Background bar
        cv2.rectangle(annotated, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress)
        bar_color = (0, 255, 0) if status['is_correct'] else (0, 165, 255)
        cv2.rectangle(annotated, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
        
        # Progress percentage
        progress_text = f"{progress*100:.0f}%"
        cv2.putText(annotated, progress_text, (bar_x + bar_width + 20, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += bar_height + 20
        
        # Feedback text
        feedback = status['feedback']
        cv2.putText(annotated, feedback, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += line_height
        
        # Reps counter
        reps_text = f"Reps: {status['reps_completed']}"
        cv2.putText(annotated, reps_text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FPS counter (top right)
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (w - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated
    
    def run_camera(self, camera_id: int = 0) -> None:
        """
        Run monitoring with live camera feed.
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        print(f"\n{'='*60}")
        print("STARTING POSE MONITORING")
        print(f"{'='*60}")
        print(f"Camera ID: {camera_id}")
        print("Press 'q' to quit, 'r' to reset")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                annotated_frame, status = self.process_frame(frame)
                
                # Display
                cv2.imshow('Yoga Pose Monitor', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("\nResetting to initial state...")
                    self._reset_to_initial_state()
                    self.total_reps = 0
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nMonitoring stopped.")


def main():
    """Example usage of online pose monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time pose monitoring and correction'
    )
    parser.add_argument(
        'graph_path',
        type=str,
        help='Path to pose graph JSON file'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n-pose.pt',
        help='YOLOv8 pose model (default: yolov8n-pose.pt)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Keypoint confidence threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load pose graph
        print(f"Loading pose graph from: {args.graph_path}")
        graph = PoseGraph.load_json(args.graph_path)
        graph.print_summary()
        
        # Create monitor
        monitor = PoseMonitor(
            pose_graph=graph,
            model_name=args.model,
            confidence_threshold=args.confidence
        )
        
        # Run monitoring
        monitor.run_camera(camera_id=args.camera)
        
        print("\nSUCCESS: Monitoring completed!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())