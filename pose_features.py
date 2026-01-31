"""
Pose Feature Extraction Module
Converts raw COCO keypoints into normalized, scale-invariant features
suitable for pose comparison and state detection.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class PoseFeatureExtractor:
    """
    Extracts normalized geometric features from COCO 17-keypoint skeleton.
    
    COCO Keypoint Indices:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum confidence for valid keypoint
        """
        self.conf_thresh = confidence_threshold
        
        # Define keypoint indices for readability
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
    
    def extract_features(self, keypoints: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract normalized pose features from raw keypoints.
        
        Args:
            keypoints: Array of shape (17, 3) where each row is [x, y, confidence]
        
        Returns:
            Dictionary of feature names to values, or None if insufficient confidence
        """
        if not self._validate_keypoints(keypoints):
            return None
        
        # Compute body scale for normalization (hip-shoulder distance)
        body_scale = self._compute_body_scale(keypoints)
        if body_scale < 10:  # Sanity check: too small means bad detection
            return None
        
        features = {}
        
        # Extract joint angles
        features.update(self._compute_joint_angles(keypoints))
        
        # Extract torso orientation
        features['torso_angle'] = self._compute_torso_angle(keypoints)
        
        # Extract arm elevations (normalized height differences)
        left_arm_elev = self._compute_arm_elevation(
            keypoints, self.LEFT_WRIST, self.LEFT_SHOULDER, body_scale
        )
        right_arm_elev = self._compute_arm_elevation(
            keypoints, self.RIGHT_WRIST, self.RIGHT_SHOULDER, body_scale
        )
        
        if left_arm_elev is not None:
            features['left_arm_elevation'] = left_arm_elev
        if right_arm_elev is not None:
            features['right_arm_elevation'] = right_arm_elev
        
        # Extract leg spread (normalized horizontal hip-ankle distance)
        features.update(self._compute_leg_spread(keypoints, body_scale))
        
        # Extract body center height (for standing/sitting/ground poses)
        features['body_center_y'] = self._compute_body_center_y(keypoints, body_scale)
        
        return features
    
    def _validate_keypoints(self, keypoints: np.ndarray) -> bool:
        """Check if enough keypoints have sufficient confidence."""
        if keypoints.shape != (17, 3):
            return False
        
        # Critical keypoints for pose analysis
        critical_indices = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_KNEE, self.RIGHT_KNEE
        ]
        
        valid_count = sum(
            keypoints[idx, 2] >= self.conf_thresh for idx in critical_indices
        )
        
        return valid_count >= 4  # At least 4/6 critical points must be visible
    
    def _compute_body_scale(self, keypoints: np.ndarray) -> float:
        """Compute characteristic body scale (hip-to-shoulder distance)."""
        left_shoulder = keypoints[self.LEFT_SHOULDER, :2]
        right_shoulder = keypoints[self.RIGHT_SHOULDER, :2]
        left_hip = keypoints[self.LEFT_HIP, :2]
        right_hip = keypoints[self.RIGHT_HIP, :2]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        return np.linalg.norm(shoulder_center - hip_center)
    
    def _compute_joint_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Compute all relevant joint angles."""
        angles = {}
        
        # Left elbow angle
        left_elbow_angle = self._compute_angle(
            keypoints[self.LEFT_SHOULDER, :2],
            keypoints[self.LEFT_ELBOW, :2],
            keypoints[self.LEFT_WRIST, :2],
            keypoints[[self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST], 2]
        )
        if left_elbow_angle is not None:
            angles['left_elbow_angle'] = left_elbow_angle
        
        # Right elbow angle
        right_elbow_angle = self._compute_angle(
            keypoints[self.RIGHT_SHOULDER, :2],
            keypoints[self.RIGHT_ELBOW, :2],
            keypoints[self.RIGHT_WRIST, :2],
            keypoints[[self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST], 2]
        )
        if right_elbow_angle is not None:
            angles['right_elbow_angle'] = right_elbow_angle
        
        # Left knee angle
        left_knee_angle = self._compute_angle(
            keypoints[self.LEFT_HIP, :2],
            keypoints[self.LEFT_KNEE, :2],
            keypoints[self.LEFT_ANKLE, :2],
            keypoints[[self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE], 2]
        )
        if left_knee_angle is not None:
            angles['left_knee_angle'] = left_knee_angle
        
        # Right knee angle
        right_knee_angle = self._compute_angle(
            keypoints[self.RIGHT_HIP, :2],
            keypoints[self.RIGHT_KNEE, :2],
            keypoints[self.RIGHT_ANKLE, :2],
            keypoints[[self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE], 2]
        )
        if right_knee_angle is not None:
            angles['right_knee_angle'] = right_knee_angle
        
        # Left shoulder angle (for arm position relative to torso)
        left_shoulder_angle = self._compute_angle(
            keypoints[self.LEFT_HIP, :2],
            keypoints[self.LEFT_SHOULDER, :2],
            keypoints[self.LEFT_ELBOW, :2],
            keypoints[[self.LEFT_HIP, self.LEFT_SHOULDER, self.LEFT_ELBOW], 2]
        )
        if left_shoulder_angle is not None:
            angles['left_shoulder_angle'] = left_shoulder_angle
        
        # Right shoulder angle
        right_shoulder_angle = self._compute_angle(
            keypoints[self.RIGHT_HIP, :2],
            keypoints[self.RIGHT_SHOULDER, :2],
            keypoints[self.RIGHT_ELBOW, :2],
            keypoints[[self.RIGHT_HIP, self.RIGHT_SHOULDER, self.RIGHT_ELBOW], 2]
        )
        if right_shoulder_angle is not None:
            angles['right_shoulder_angle'] = right_shoulder_angle
        
        # Left hip angle (thigh relative to torso)
        left_hip_angle = self._compute_angle(
            keypoints[self.LEFT_SHOULDER, :2],
            keypoints[self.LEFT_HIP, :2],
            keypoints[self.LEFT_KNEE, :2],
            keypoints[[self.LEFT_SHOULDER, self.LEFT_HIP, self.LEFT_KNEE], 2]
        )
        if left_hip_angle is not None:
            angles['left_hip_angle'] = left_hip_angle
        
        # Right hip angle
        right_hip_angle = self._compute_angle(
            keypoints[self.RIGHT_SHOULDER, :2],
            keypoints[self.RIGHT_HIP, :2],
            keypoints[self.RIGHT_KNEE, :2],
            keypoints[[self.RIGHT_SHOULDER, self.RIGHT_HIP, self.RIGHT_KNEE], 2]
        )
        if right_hip_angle is not None:
            angles['right_hip_angle'] = right_hip_angle
        
        return angles
    
    def _compute_angle(
        self, 
        p1: np.ndarray, 
        p2: np.ndarray, 
        p3: np.ndarray,
        confidences: np.ndarray
    ) -> Optional[float]:
        """
        Compute angle at p2 formed by p1-p2-p3.
        Returns angle in degrees [0, 180].
        """
        if np.any(confidences < self.conf_thresh):
            return None
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return None
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Compute angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)
    
    def _compute_torso_angle(self, keypoints: np.ndarray) -> float:
        """
        Compute torso orientation angle relative to vertical.
        Returns angle in degrees [-90, 90] where 0 is upright.
        """
        left_shoulder = keypoints[self.LEFT_SHOULDER, :2]
        right_shoulder = keypoints[self.RIGHT_SHOULDER, :2]
        left_hip = keypoints[self.LEFT_HIP, :2]
        right_hip = keypoints[self.RIGHT_HIP, :2]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Vector from hip to shoulder
        torso_vec = shoulder_center - hip_center
        
        # Angle from vertical (positive y-axis points down in image coords)
        angle_rad = np.arctan2(torso_vec[0], -torso_vec[1])
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)
    
    def _compute_arm_elevation(
        self, 
        keypoints: np.ndarray, 
        wrist_idx: int, 
        shoulder_idx: int,
        body_scale: float
    ) -> Optional[float]:
        """
        Compute normalized vertical distance between wrist and shoulder.
        Positive = wrist above shoulder, Negative = wrist below shoulder.
        """
        if (keypoints[wrist_idx, 2] < self.conf_thresh or 
            keypoints[shoulder_idx, 2] < self.conf_thresh):
            return None
        
        wrist_y = keypoints[wrist_idx, 1]
        shoulder_y = keypoints[shoulder_idx, 1]
        
        # Normalize by body scale (negative because y increases downward)
        elevation = -(wrist_y - shoulder_y) / body_scale
        
        return float(elevation)
    
    def _compute_leg_spread(
        self, 
        keypoints: np.ndarray, 
        body_scale: float
    ) -> Dict[str, float]:
        """Compute normalized horizontal distances for leg positioning."""
        spread = {}
        
        # Hip-to-ankle horizontal distance (for wide stance detection)
        if (keypoints[self.LEFT_HIP, 2] >= self.conf_thresh and
            keypoints[self.LEFT_ANKLE, 2] >= self.conf_thresh):
            left_spread = abs(
                keypoints[self.LEFT_ANKLE, 0] - keypoints[self.LEFT_HIP, 0]
            ) / body_scale
            spread['left_leg_spread'] = float(left_spread)
        
        if (keypoints[self.RIGHT_HIP, 2] >= self.conf_thresh and
            keypoints[self.RIGHT_ANKLE, 2] >= self.conf_thresh):
            right_spread = abs(
                keypoints[self.RIGHT_ANKLE, 0] - keypoints[self.RIGHT_HIP, 0]
            ) / body_scale
            spread['right_leg_spread'] = float(right_spread)
        
        return spread
    
    def _compute_body_center_y(
        self, 
        keypoints: np.ndarray, 
        body_scale: float
    ) -> float:
        """
        Compute normalized vertical position of body center.
        Useful for distinguishing standing/sitting/lying poses.
        """
        left_hip = keypoints[self.LEFT_HIP, :2]
        right_hip = keypoints[self.RIGHT_HIP, :2]
        hip_center = (left_hip + right_hip) / 2
        
        # Normalize relative to body scale (arbitrary reference)
        # Lower values = higher in frame, higher values = lower in frame
        return float(hip_center[1] / body_scale)