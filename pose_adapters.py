"""
Pose Model Adapter Interface
Provides a unified interface for multiple pose estimation models.

Each adapter maps model-specific outputs to a canonical COCO-17 skeleton format,
ensuring the pose graph and downstream logic remain model-agnostic.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import cv2


class PoseModelAdapter(ABC):
    """
    Abstract interface for pose estimation models.
    
    All adapters must output keypoints in the canonical COCO-17 format:
    [nose, left_eye, right_eye, left_ear, right_ear,
     left_shoulder, right_shoulder, left_elbow, right_elbow,
     left_wrist, right_wrist, left_hip, right_hip,
     left_knee, right_knee, left_ankle, right_ankle]
    
    This ensures the pose graph and all downstream logic work
    identically regardless of the underlying model.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose keypoints in frame.
        
        Args:
            frame: BGR image from camera
        
        Returns:
            Keypoints array of shape (17, 3) in format [x, y, confidence]
            or None if no pose detected.
            
            Coordinates are in pixel space relative to frame dimensions.
            Confidence is in range [0, 1].
        """
        pass
    def draw_skeleton(self, frame: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw skeleton visualization on frame.
        
        Args:
            frame: BGR image
            keypoints: Keypoints array of shape (17, 3) in format [x, y, confidence]
        
        Returns:
            Annotated frame
        """
        # Default implementation: no-op, returns frame as-is
        return frame


class YOLOv8PoseAdapter(PoseModelAdapter):
    """
    Adapter for YOLOv8 Pose model.
    
    YOLOv8 already outputs COCO-17 format, so this is mostly a thin wrapper
    that extracts the primary person and ensures consistent output format.
    """
    
    def __init__(self, model_name: str = 'yolov8n-pose.pt'):
        """
        Args:
            model_name: YOLOv8 pose model name
        """
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        print(f"  [ADAPTER] Loaded YOLOv8 Pose: {model_name}")
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose using YOLOv8.
        
        YOLOv8 outputs COCO-17 format directly, so we just need to:
        1. Run inference
        2. Select primary person (largest bbox)
        3. Return keypoints in canonical format
        """
        results = self.model(frame, verbose=False)
        
        if results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        
        # Select primary person (largest bounding box)
        num_detections = len(results[0].keypoints)
        
        if num_detections == 0:
            return None
        
        if num_detections == 1:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            return keypoints  # Already in COCO-17 format [17, 3]
        
        # Multiple people - select largest bbox
        boxes = results[0].boxes.data.cpu().numpy()
        max_area = 0
        primary_idx = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                primary_idx = i
        
        keypoints = results[0].keypoints.data[primary_idx].cpu().numpy()
        return keypoints  # Shape: (17, 3)
    def draw_skeleton(self, frame: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw COCO-17 skeleton on frame.
        
        Uses cached keypoints (no inference).
        Draws circles for joints and lines for bones.
        """
        if keypoints is None:
            return frame
        
        # COCO-17 skeleton connections (bone pairs)
        skeleton = [
            (0, 1), (0, 2),      # nose to eyes
            (1, 3), (2, 4),      # eyes to ears
            (0, 5), (0, 6),      # nose to shoulders
            (5, 6),              # shoulders
            (5, 7), (7, 9),      # left arm
            (6, 8), (8, 10),     # right arm
            (5, 11), (6, 12),    # shoulders to hips
            (11, 12),            # hips
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
        ]
        
        # Draw bones (lines between joints)
        for start_idx, end_idx in skeleton:
            if (keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw joints (circles at keypoints)
        for i in range(17):
            if keypoints[i][2] > 0.5:  # Only draw if confidence > 0.5
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        return frame


class MediaPipePoseAdapter(PoseModelAdapter):
    """
    Adapter for MediaPipe Pose model.
    
    MediaPipe outputs 33 landmarks, but we only need the COCO-17 subset.
    Maps MediaPipe landmark indices to canonical COCO-17 format.
    """
    
    # MediaPipe landmark indices mapping to COCO-17
    # https://google.github.io/mediapipe/solutions/pose.html
    MEDIAPIPE_TO_COCO = {
        0: 0,   # nose
        2: 1,   # left_eye (inner)
        5: 2,   # right_eye (inner)
        7: 3,   # left_ear
        8: 4,   # right_ear
        11: 5,  # left_shoulder
        12: 6,  # right_shoulder
        13: 7,  # left_elbow
        14: 8,  # right_elbow
        15: 9,  # left_wrist
        16: 10, # right_wrist
        23: 11, # left_hip
        24: 12, # right_hip
        25: 13, # left_knee
        26: 14, # right_knee
        27: 15, # left_ankle
        28: 16, # right_ankle
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum detection confidence
        """
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold
            )
            print(f"  [ADAPTER] Loaded MediaPipe Pose")
        except ImportError:
            raise ImportError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose using MediaPipe.
        
        MediaPipe workflow:
        1. Convert BGR to RGB
        2. Run inference
        3. Extract 33 landmarks
        4. Map to COCO-17 subset
        5. Convert visibility to confidence
        6. Denormalize coordinates to pixel space
        """
        # MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None
        
        h, w = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark
        
        # Initialize COCO-17 keypoints array
        keypoints = np.zeros((17, 3), dtype=np.float32)
        
        # Map MediaPipe landmarks to COCO-17
        for mp_idx, coco_idx in self.MEDIAPIPE_TO_COCO.items():
            lm = landmarks[mp_idx]
            
            # Denormalize coordinates to pixel space
            x = lm.x * w
            y = lm.y * h
            
            # MediaPipe provides 'visibility' (0-1), use as confidence
            # Clamp to [0, 1] in case of numerical issues
            confidence = np.clip(lm.visibility, 0.0, 1.0)
            
            keypoints[coco_idx] = [x, y, confidence]
        
        return keypoints  # Shape: (17, 3)


class MoveNetPoseAdapter(PoseModelAdapter):
    """
    Adapter for MoveNet (TensorFlow Lite) model.
    
    MoveNet outputs 17 keypoints but in a different order than COCO.
    Maps MoveNet indices to canonical COCO-17 format.
    """
    
    # MoveNet keypoint order to COCO-17 mapping
    # https://www.tensorflow.org/hub/tutorials/movenet
    MOVENET_TO_COCO = {
        0: 0,   # nose
        1: 1,   # left_eye
        2: 2,   # right_eye
        3: 3,   # left_ear
        4: 4,   # right_ear
        5: 5,   # left_shoulder
        6: 6,   # right_shoulder
        7: 7,   # left_elbow
        8: 8,   # right_elbow
        9: 9,   # left_wrist
        10: 10, # right_wrist
        11: 11, # left_hip
        12: 12, # right_hip
        13: 13, # left_knee
        14: 14, # right_knee
        15: 15, # left_ankle
        16: 16, # right_ankle
    }
    
    def __init__(self, model_path: str = 'movenet_thunder.tflite'):
        """
        Args:
            model_path: Path to MoveNet TFLite model
        """
        try:
            import tensorflow as tf
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get expected input size
            self.input_size = self.input_details[0]['shape'][1]
            
            print(f"  [ADAPTER] Loaded MoveNet: {model_path}")
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MoveNet model: {e}")
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose using MoveNet.
        
        MoveNet workflow:
        1. Resize frame to model input size (e.g., 256x256)
        2. Run inference
        3. Extract 17 keypoints with scores
        4. Map to COCO-17 order (MoveNet already uses COCO order)
        5. Denormalize coordinates to original frame size
        """
        h, w = frame.shape[:2]
        
        # Prepare input: resize and normalize
        input_image = cv2.resize(frame, (self.input_size, self.input_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.uint8)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        
        # Get output: shape (1, 1, 17, 3) -> [y, x, score]
        keypoints_with_scores = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0, 0]
        
        # Initialize COCO-17 keypoints array
        keypoints = np.zeros((17, 3), dtype=np.float32)
        
        # MoveNet outputs: [y_norm, x_norm, score]
        # We need: [x_pixel, y_pixel, confidence]
        for movenet_idx, coco_idx in self.MOVENET_TO_COCO.items():
            y_norm, x_norm, score = keypoints_with_scores[movenet_idx]
            
            # Denormalize to original frame size
            x = x_norm * w
            y = y_norm * h
            confidence = score
            
            keypoints[coco_idx] = [x, y, confidence]
        
        # Check if any keypoint detected with reasonable confidence
        if np.max(keypoints[:, 2]) < 0.1:
            return None
        
        return keypoints  # Shape: (17, 3)


def create_pose_adapter(model_type: str, **kwargs) -> PoseModelAdapter:
    """
    Factory function to create appropriate pose adapter.
    
    Args:
        model_type: One of 'yolo', 'mediapipe', 'movenet'
        **kwargs: Model-specific parameters
    
    Returns:
        Appropriate PoseModelAdapter instance
    """
    model_type = model_type.lower()
    
    if model_type == 'yolo' or model_type == 'yolov8':
        model_name = kwargs.get('model_name', 'yolov8n-pose.pt')
        return YOLOv8PoseAdapter(model_name=model_name)
    
    elif model_type == 'mediapipe':
        confidence = kwargs.get('confidence_threshold', 0.5)
        return MediaPipePoseAdapter(confidence_threshold=confidence)
    
    elif model_type == 'movenet':
        model_path = kwargs.get('model_path', 'movenet_thunder.tflite')
        return MoveNetPoseAdapter(model_path=model_path)
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: 'yolo', 'mediapipe', 'movenet'"
        )