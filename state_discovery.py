"""
Automatic Pose State Discovery Module (WITH TEMPORAL SEQUENCE TRACKING)
Segments a video into distinct pose states using temporal stability
and feature-space clustering without manual labeling.

MODIFICATIONS:
- Now returns temporal sequence of discovered segments
- Each segment tracks its temporal position for transition building
- Keypoint storage preserved
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque


class PoseState:
    """Represents a discovered pose state with mean features and tolerances."""
    
    def __init__(
        self, 
        state_id: int,
        mean_features: Dict[str, float],
        feature_tolerances: Dict[str, float],
        min_hold_duration: float,
        mean_keypoints: Optional[np.ndarray] = None
    ):
        """
        Args:
            state_id: Unique identifier for this state
            mean_features: Mean value for each feature
            feature_tolerances: Acceptable deviation for each feature
            min_hold_duration: Minimum time (seconds) to hold this pose
            mean_keypoints: Mean keypoint positions (17, 2) for visualization
        """
        self.state_id = state_id
        self.mean_features = mean_features
        self.feature_tolerances = feature_tolerances
        self.min_hold_duration = min_hold_duration
        self.mean_keypoints = mean_keypoints
    
    def matches(self, features: Dict[str, float], strict: bool = False) -> Tuple[bool, Dict[str, float]]:
        """
        Check if given features match this state within tolerances.
        
        Args:
            features: Feature dictionary to check
            strict: If True, all features must match. If False, allow partial match
        
        Returns:
            (matches: bool, deviations: Dict[str, float])
        """
        deviations = {}
        matches_count = 0
        total_count = 0
        
        for feature_name, mean_val in self.mean_features.items():
            if feature_name not in features:
                if strict:
                    return False, {}
                continue
            
            total_count += 1
            tolerance = self.feature_tolerances[feature_name]
            deviation = abs(features[feature_name] - mean_val)
            deviations[feature_name] = deviation
            
            if deviation <= tolerance:
                matches_count += 1
        
        if total_count == 0:
            return False, {}
        
        # For strict mode, all features must match
        if strict:
            return matches_count == total_count, deviations
        
        # For non-strict mode, require at least 70% of features to match
        match_ratio = matches_count / total_count
        return match_ratio >= 0.6, deviations
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'state_id': self.state_id,
            'mean_features': self.mean_features,
            'feature_tolerances': self.feature_tolerances,
            'min_hold_duration': self.min_hold_duration
        }
        
        # Include mean keypoints if available
        if self.mean_keypoints is not None:
            # Store only x, y coordinates (drop confidence column)
            result['mean_keypoints'] = self.mean_keypoints[:, :2].tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PoseState':
        """Create PoseState from dictionary."""
        mean_keypoints = None
        if 'mean_keypoints' in data:
            mean_keypoints = np.array(data['mean_keypoints'])
        
        return cls(
            state_id=data['state_id'],
            mean_features=data['mean_features'],
            feature_tolerances=data['feature_tolerances'],
            min_hold_duration=data['min_hold_duration'],
            mean_keypoints=mean_keypoints
        )


class SegmentInfo:
    """
    Represents a discovered stable segment with its temporal context.
    
    This class encapsulates a contiguous frame range that represents a stable pose,
    along with the computed mean features for that segment. It serves as an 
    intermediate representation before state merging.
    """
    
    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        mean_features: Dict[str, float],
        segment_index: int
    ):
        """
        Args:
            start_frame: First frame index of this segment
            end_frame: Last frame index of this segment (inclusive)
            mean_features: Computed mean features across all frames in segment
            segment_index: Temporal order index (0, 1, 2, ...) in video
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.mean_features = mean_features
        self.segment_index = segment_index
        self.assigned_state_id: Optional[int] = None  # Set during merging
    
    def duration_frames(self) -> int:
        """Get duration of this segment in frames."""
        return self.end_frame - self.start_frame + 1


class StateDiscovery:
    """
    Discovers pose states from a sequence of pose features using
    temporal stability and feature-space distance thresholds.
    
    NOW WITH TEMPORAL SEQUENCE TRACKING for non-linear graph construction.
    """
    
    def __init__(
        self,
        stability_threshold: float = 0.15,
        min_hold_frames: int = 15,
        smoothing_window: int = 5,
        transition_buffer: int = 3,
        tolerance_multiplier: float = 2.0
    ):
        """
        Args:
            stability_threshold: Maximum normalized feature distance for same state
            min_hold_frames: Minimum frames a pose must be held to be a state
            smoothing_window: Number of frames for rolling average smoothing
            transition_buffer: Frames to skip during state transitions
            tolerance_multiplier: Multiplier for std to set feature tolerances
        """
        self.stability_threshold = stability_threshold
        self.min_hold_frames = min_hold_frames
        self.smoothing_window = smoothing_window
        self.transition_buffer = transition_buffer
        self.tolerance_multiplier = tolerance_multiplier
    
    def discover_states(
        self, 
        feature_sequence: List[Dict[str, float]],
        fps: float,
        keypoint_sequence: Optional[List[Optional[np.ndarray]]] = None
    ) -> List[PoseState]:
        """
        Automatically discover pose states from feature sequence.
        
        BACKWARD COMPATIBILITY: Returns List[PoseState] as before, but
        internally now tracks temporal sequence for use by graph generator.
        
        Args:
            feature_sequence: List of feature dictionaries (one per frame)
            fps: Video frame rate
            keypoint_sequence: List of keypoint arrays (17, 3) per frame (optional)
        
        Returns:
            List of discovered PoseState objects (ordered by first appearance)
        """
        if len(feature_sequence) < self.min_hold_frames:
            return []
        
        # Step 1: Apply temporal smoothing
        smoothed_features = self._smooth_features(feature_sequence)
        
        # Step 2: Detect stable segments (returns temporal sequence)
        segment_info_list = self._detect_stable_segments_with_tracking(smoothed_features)
        
        # Step 3: Create states with temporal tracking
        states, segment_to_state_mapping = self._create_states_from_segments_with_merging(
            segment_info_list,
            smoothed_features,
            fps,
            keypoint_sequence
        )
        
        # Store mapping for use by graph generator
        # This allows the generator to reconstruct the temporal transition sequence
        self._last_segment_info_list = segment_info_list
        self._last_segment_to_state = segment_to_state_mapping
        
        return states
    
    def get_temporal_sequence(self) -> List[int]:
        """
        Get the temporal sequence of state IDs as they appeared in the video.
        
        This method should be called after discover_states() to retrieve
        the actual order of states for building non-linear transitions.
        
        Returns:
            List of state IDs in temporal order (e.g., [0, 1, 2, 0, 1, 3])
        """
        if not hasattr(self, '_last_segment_to_state'):
            return []
        
        # Build temporal sequence from segment assignments
        temporal_sequence = []
        for segment_info in self._last_segment_info_list:
            state_id = self._last_segment_to_state[segment_info.segment_index]
            temporal_sequence.append(state_id)
        
        return temporal_sequence
    
    def _smooth_features(
        self, 
        feature_sequence: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Apply rolling average smoothing to feature sequence."""
        if len(feature_sequence) < self.smoothing_window:
            return feature_sequence
        
        # Get all feature names (use first valid frame as template)
        feature_names = set()
        for features in feature_sequence:
            if features:
                feature_names.update(features.keys())
        feature_names = list(feature_names)
        
        smoothed = []
        window = deque(maxlen=self.smoothing_window)
        
        for features in feature_sequence:
            if not features:
                smoothed.append({})
                continue
            
            window.append(features)
            
            # Compute rolling average for each feature
            smoothed_frame = {}
            for fname in feature_names:
                values = [f[fname] for f in window if fname in f]
                if values:
                    smoothed_frame[fname] = np.mean(values)
            
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    def _detect_stable_segments_with_tracking(
        self, 
        feature_sequence: List[Dict[str, float]]
    ) -> List[SegmentInfo]:
        """
        Detect segments where pose remains stable and track temporal order.
        
        KEY CHANGE: Returns List[SegmentInfo] instead of List[Tuple[int, int]]
        Each SegmentInfo includes frame range, mean features, and temporal index.
        
        Returns:
            List of SegmentInfo objects in temporal order
        """
        segments = []
        segment_index = 0
        current_start = 0
        current_reference = None
        
        i = 0
        while i < len(feature_sequence):
            features = feature_sequence[i]
            
            if not features:
                # Invalid frame, reset
                if current_reference is not None and (i - current_start) >= self.min_hold_frames:
                    # Compute mean features for this segment
                    segment_features = feature_sequence[current_start:i]
                    mean_features = self._compute_mean_features(segment_features)
                    
                    segment_info = SegmentInfo(
                        start_frame=current_start,
                        end_frame=i - 1,
                        mean_features=mean_features,
                        segment_index=segment_index
                    )
                    segments.append(segment_info)
                    segment_index += 1
                
                current_reference = None
                current_start = i + 1
                i += 1
                continue
            
            if current_reference is None:
                # Start new segment
                current_reference = features
                current_start = i
                i += 1
                continue
            
            # Check if current frame is similar to reference
            distance = self._compute_feature_distance(features, current_reference)
            
            if distance <= self.stability_threshold:
                # Still in same stable state
                i += 1
            else:
                # State changed
                if (i - current_start) >= self.min_hold_frames:
                    # Compute mean features for this segment
                    segment_features = feature_sequence[current_start:i]
                    mean_features = self._compute_mean_features(segment_features)
                    
                    segment_info = SegmentInfo(
                        start_frame=current_start,
                        end_frame=i - 1,
                        mean_features=mean_features,
                        segment_index=segment_index
                    )
                    segments.append(segment_info)
                    segment_index += 1
                
                # Skip transition buffer frames
                current_reference = None
                i += self.transition_buffer
                current_start = i
        
        # Handle last segment
        if current_reference is not None and (len(feature_sequence) - current_start) >= self.min_hold_frames:
            segment_features = feature_sequence[current_start:len(feature_sequence)]
            mean_features = self._compute_mean_features(segment_features)
            
            segment_info = SegmentInfo(
                start_frame=current_start,
                end_frame=len(feature_sequence) - 1,
                mean_features=mean_features,
                segment_index=segment_index
            )
            segments.append(segment_info)
        
        return segments
    
    def _compute_feature_distance(
        self, 
        features1: Dict[str, float], 
        features2: Dict[str, float]
    ) -> float:
        """
        Compute normalized distance between two feature sets.
        Uses weighted Euclidean distance with angle normalization.
        """
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return float('inf')
        
        distances = []
        for fname in common_features:
            val1 = features1[fname]
            val2 = features2[fname]
            
            # Special handling for angles (normalize to [-180, 180])
            if 'angle' in fname:
                diff = abs(val1 - val2)
                # Handle angle wraparound
                if diff > 180:
                    diff = 360 - diff
                # Normalize angle differences to [0, 1] range
                distances.append(diff / 180.0)
            else:
                # For non-angle features (elevations, spreads), use absolute difference
                # These are already normalized by body scale
                distances.append(abs(val1 - val2))
        
        # Return mean distance
        return np.mean(distances)
    
    def _create_states_from_segments_with_merging(
        self,
        segment_info_list: List[SegmentInfo],
        feature_sequence: List[Dict[str, float]],
        fps: float,
        keypoint_sequence: Optional[List[Optional[np.ndarray]]] = None
    ) -> Tuple[List[PoseState], Dict[int, int]]:
        """
        Create PoseState objects from segments with intelligent merging.
        
        CRITICAL LOGIC: This is where state merging happens.
        - Iterate through segments in temporal order
        - For each segment, compare against existing canonical states
        - If similar enough → assign to existing state
        - If not similar → create new state
        
        Args:
            segment_info_list: List of SegmentInfo in temporal order
            feature_sequence: Full feature sequence for computing tolerances
            fps: Frames per second
            keypoint_sequence: Optional keypoint sequence
        
        Returns:
            (states: List[PoseState], segment_to_state: Dict[segment_idx -> state_id])
        """
        if not segment_info_list:
            return [], {}
        
        # Canonical states: stores (state_id, mean_features, frame_ranges, keypoint_frames)
        canonical_states: List[Tuple[int, Dict[str, float], List[Tuple[int, int]], List[Tuple[int, int]]]] = []
        segment_to_state_mapping: Dict[int, int] = {}
        
        next_state_id = 0
        
        # Process each segment in temporal order
        for segment_info in segment_info_list:
            # Try to match this segment to an existing state
            matched_state_id = self._find_matching_state(
                segment_info.mean_features,
                canonical_states
            )
            
            if matched_state_id is not None:
                # Merge with existing state
                segment_to_state_mapping[segment_info.segment_index] = matched_state_id
                
                # Add this segment's frame range to the matched state
                for i, (state_id, mean_feat, frame_ranges, kpt_ranges) in enumerate(canonical_states):
                    if state_id == matched_state_id:
                        frame_ranges.append((segment_info.start_frame, segment_info.end_frame))
                        kpt_ranges.append((segment_info.start_frame, segment_info.end_frame))
                        break
            else:
                # Create new state
                new_state_id = next_state_id
                segment_to_state_mapping[segment_info.segment_index] = new_state_id
                
                canonical_states.append((
                    new_state_id,
                    segment_info.mean_features,
                    [(segment_info.start_frame, segment_info.end_frame)],  # frame ranges
                    [(segment_info.start_frame, segment_info.end_frame)]   # keypoint ranges
                ))
                next_state_id += 1
        
        # Now build final PoseState objects from canonical states
        final_states = []
        
        for state_id, _, frame_ranges, kpt_ranges in canonical_states:
            # Recompute mean features across ALL frames assigned to this state
            all_features = []
            for start, end in frame_ranges:
                all_features.extend(feature_sequence[start:end+1])
            
            merged_mean_features = self._compute_mean_features(all_features)
            
            # Compute tolerances
            tolerances = self._compute_feature_tolerances_from_frames(
                frame_ranges,
                feature_sequence,
                merged_mean_features
            )
            
            # Compute hold duration (median duration)
            durations = [(end - start + 1) / fps for start, end in frame_ranges]
            min_hold_duration = np.median(durations) * 0.6
            
            # Compute mean keypoints if available
            mean_keypoints = None
            if keypoint_sequence is not None:
                mean_keypoints = self._compute_mean_keypoints(kpt_ranges, keypoint_sequence)
            
            pose_state = PoseState(
                state_id=state_id,
                mean_features=merged_mean_features,
                feature_tolerances=tolerances,
                min_hold_duration=float(min_hold_duration),
                mean_keypoints=mean_keypoints
            )
            final_states.append(pose_state)
        
        # Sort states by ID for consistency
        final_states.sort(key=lambda s: s.state_id)
        
        return final_states, segment_to_state_mapping
    
    def _find_matching_state(
        self,
        segment_features: Dict[str, float],
        canonical_states: List[Tuple[int, Dict[str, float], List, List]]
    ) -> Optional[int]:
        """
        Find if segment features match any existing canonical state.
        
        DESIGN DECISION: Use stability_threshold for matching.
        We use the SAME threshold used for segment detection because:
        - If two frame ranges were stable relative to this threshold
        - And their mean features are within this threshold
        - Then they represent the same pose state
        
        Args:
            segment_features: Mean features of candidate segment
            canonical_states: List of (state_id, mean_features, frame_ranges, kpt_ranges)
        
        Returns:
            state_id if match found, None otherwise
        """
        for state_id, state_mean_features, _, _ in canonical_states:
            distance = self._compute_feature_distance(segment_features, state_mean_features)
            
            # Use stability threshold as merge threshold
            # Rationale: If poses are stable within this threshold during detection,
            # they should be merged within the same threshold
            if distance <= self.stability_threshold:
                return state_id
        
        return None
    
    def _compute_mean_features(
        self, 
        feature_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute mean value for each feature across all frames."""
        if not feature_list:
            return {}
        
        # Get all feature names
        all_features = set()
        for features in feature_list:
            all_features.update(features.keys())
        
        mean_features = {}
        for fname in all_features:
            values = [f[fname] for f in feature_list if fname in f]
            if values:
                mean_features[fname] = float(np.mean(values))
        
        return mean_features
    
    def _compute_mean_keypoints(
        self,
        frame_ranges: List[Tuple[int, int]],
        keypoint_sequence: List[Optional[np.ndarray]]
    ) -> Optional[np.ndarray]:
        """
        Compute mean keypoint positions across all frames in the state.
        
        Args:
            frame_ranges: List of (start, end) frame indices for this state
            keypoint_sequence: Full sequence of keypoints (17, 3) per frame
        
        Returns:
            Mean keypoints array (17, 3) or None if insufficient data
        """
        all_keypoints = []
        
        # Collect all keypoints from all frame ranges
        for start, end in frame_ranges:
            for frame_idx in range(start, end + 1):
                if frame_idx < len(keypoint_sequence):
                    kpts = keypoint_sequence[frame_idx]
                    if kpts is not None:
                        all_keypoints.append(kpts)
        
        if not all_keypoints:
            return None
        
        # Stack and compute mean
        keypoints_array = np.array(all_keypoints)  # Shape: (N, 17, 3)
        mean_keypoints = np.mean(keypoints_array, axis=0)  # Shape: (17, 3)
        
        return mean_keypoints
    
    def _compute_feature_tolerances_from_frames(
        self,
        frame_ranges: List[Tuple[int, int]],
        feature_sequence: List[Dict[str, float]],
        mean_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute acceptable tolerance for each feature based on variance.
        """
        # Collect all features from all frame ranges
        all_features = []
        for start, end in frame_ranges:
            all_features.extend(feature_sequence[start:end+1])
        
        tolerances = {}
        for fname, mean_val in mean_features.items():
            values = [f[fname] for f in all_features if fname in f]
            
            if len(values) > 1:
                std = np.std(values)
                # Use multiple of std as tolerance, with minimum threshold
                if 'angle' in fname:
                    # Angles: minimum 10 degrees tolerance
                    tolerance = max(std * self.tolerance_multiplier, 10.0)
                else:
                    # Other features: minimum 0.1 tolerance (normalized scale)
                    tolerance = max(std * self.tolerance_multiplier, 0.1)
            else:
                # Default tolerances if not enough variance data
                if 'angle' in fname:
                    tolerance = 15.0
                else:
                    tolerance = 0.15
            
            tolerances[fname] = float(tolerance)
        
        return tolerances