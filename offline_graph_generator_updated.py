"""
Offline Pose Graph Generator (WITH STATE MERGING AND NON-LINEAR TRANSITIONS)
Processes a reference video to automatically extract pose states
and build a pose graph for online monitoring.

MODIFICATIONS:
- Uses new StateDiscovery temporal tracking
- Builds non-linear graph from actual video flow
- Replaces build_linear_sequence with build_from_temporal_sequence
- States are merged when similar poses appear multiple times
- STORES TEMPORAL SEQUENCE IN METADATA for deterministic replay
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple
import os
from datetime import datetime

from pose_features import PoseFeatureExtractor
from state_discovery import StateDiscovery
from pose_graph import PoseGraph


class OfflineGraphGenerator:
    """
    Processes reference video to generate pose graph.
    
    Pipeline:
    1. Load video and extract frames
    2. Run YOLOv8 pose estimation on each frame
    3. SELECT SINGLE PRIMARY SUBJECT (best quality + prominence)
    4. Extract normalized features from keypoints
    5. Discover stable pose states using temporal analysis
    6. MERGE IDENTICAL/SIMILAR STATES (NEW)
    7. Build directed graph from ACTUAL VIDEO TRANSITIONS (NEW)
    8. STORE TEMPORAL SEQUENCE in metadata for deterministic replay
    9. Save graph to JSON (WITH KEYPOINTS)
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n-pose.pt',
        confidence_threshold: float = 0.6,
        stability_threshold: float = 0.12,
        min_hold_frames: int = 18,
        smoothing_window: int = 7,
        transition_buffer: int = 3,
        tolerance_multiplier: float = 2.0
    ):
        """
        Args:
            model_name: YOLOv8 pose model name
            confidence_threshold: Minimum keypoint confidence
            stability_threshold: Maximum distance for same state (ALSO USED FOR MERGING)
            min_hold_frames: Minimum frames to qualify as a state
            smoothing_window: Frames for rolling average
            transition_buffer: Frames to skip during transitions
            tolerance_multiplier: Multiplier for feature tolerance
        """
        print(f"Loading YOLOv8 pose model: {model_name}")
        self.model = YOLO(model_name)
        
        self.feature_extractor = PoseFeatureExtractor(
            confidence_threshold=confidence_threshold
        )
        
        # StateDiscovery now handles both segment detection AND state merging
        # The stability_threshold is used for BOTH:
        # - Detecting when a pose changes (segment boundaries)
        # - Merging similar poses into the same state
        self.state_discovery = StateDiscovery(
            stability_threshold=stability_threshold,
            min_hold_frames=min_hold_frames,
            smoothing_window=smoothing_window,
            transition_buffer=transition_buffer,
            tolerance_multiplier=tolerance_multiplier
        )
    
    def generate_from_video(
        self,
        video_path: str,
        output_json_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 1
    ) -> PoseGraph:
        """
        Generate pose graph from reference video.
        
        NOW BUILDS NON-LINEAR GRAPH based on actual video transitions.
        STORES TEMPORAL SEQUENCE for deterministic online replay.
        
        Args:
            video_path: Path to reference video file
            output_json_path: Path for output JSON file
            max_frames: Maximum frames to process (None = all)
            skip_frames: Process every Nth frame (1 = all frames)
        
        Returns:
            Generated PoseGraph object with non-linear transitions
        """
        print(f"\n{'='*60}")
        print(f"GENERATING POSE GRAPH FROM VIDEO")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Output: {output_json_path}")
        
        # Step 1: Load video and get metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        
        # Adjust FPS for skipped frames
        effective_fps = fps / skip_frames
        
        # Step 2: Extract pose features AND keypoints from all frames
        print(f"\nExtracting pose features (single-person mode)...")
        feature_sequence, keypoint_sequence = self._extract_features_from_video(
            cap, max_frames, skip_frames
        )
        cap.release()
        
        valid_frames = sum(1 for f in feature_sequence if f)
        print(f"  Processed: {len(feature_sequence)} frames")
        print(f"  Valid poses: {valid_frames} frames ({valid_frames/len(feature_sequence)*100:.1f}%)")
        
        if valid_frames < 10:
            raise ValueError("Too few valid poses detected. Check video quality or camera angle.")
        
        # Step 3: Discover pose states WITH STATE MERGING
        # The StateDiscovery now:
        # - Detects stable segments
        # - Merges similar segments into the same state
        # - Tracks the temporal sequence of states
        print(f"\nDiscovering pose states (with automatic merging)...")
        states = self.state_discovery.discover_states(
            feature_sequence, 
            fps=effective_fps,
            keypoint_sequence=keypoint_sequence
        )
        
        print(f"  Discovered: {len(states)} distinct pose states")
        
        if len(states) == 0:
            raise ValueError("No pose states discovered. Try adjusting parameters or video.")
        
        # Step 4: Get temporal sequence of states
        # This is the ACTUAL order states appeared in the video
        # Example: [0, 1, 2, 0, 1, 3] means:
        #   - State 0 appeared first
        #   - Then state 1
        #   - Then state 2
        #   - Then back to state 0 (cycle!)
        #   - Then state 1 again
        #   - Then new state 3
        temporal_sequence = self.state_discovery.get_temporal_sequence()
        
        print(f"\nTemporal sequence of states:")
        print(f"  {temporal_sequence}")
        print(f"  Length: {len(temporal_sequence)} state occurrences")
        
        # Check if states were actually merged
        num_segments = len(temporal_sequence)
        num_unique_states = len(states)
        if num_segments > num_unique_states:
            print(f"  ✓ State merging successful: {num_segments} segments → {num_unique_states} unique states")
        else:
            print(f"  ⚠ No merging occurred (all segments are unique poses)")
        
        # Step 5: Build pose graph from temporal sequence
        # This is where NON-LINEAR structure is created
        print(f"\nBuilding pose graph from temporal transitions...")
        graph = PoseGraph()
        graph.build_from_temporal_sequence(states, temporal_sequence)
        
        # Get graph statistics to show structure
        stats = graph.get_graph_statistics()
        print(f"  Graph type: {'Linear' if stats['is_linear'] else 'Non-Linear'}")
        print(f"  Has cycles: {'Yes' if stats['has_cycles'] else 'No'}")
        print(f"  Total transitions: {stats['num_transitions']}")
        
        # Step 6: Add metadata INCLUDING TEMPORAL SEQUENCE
        # CRITICAL: Store temporal_sequence for deterministic online replay
        # The online monitor will use this to follow the exact reference video order
        graph.set_metadata('video_path', os.path.basename(video_path))
        graph.set_metadata('fps', float(effective_fps))
        graph.set_metadata('original_fps', float(fps))
        graph.set_metadata('skip_frames', skip_frames)
        graph.set_metadata('total_frames', len(feature_sequence))
        graph.set_metadata('valid_frames', valid_frames)
        graph.set_metadata('num_states', len(states))
        graph.set_metadata('num_segments', len(temporal_sequence))
        graph.set_metadata('created_at', datetime.now().isoformat())
        graph.set_metadata('stability_threshold', self.state_discovery.stability_threshold)
        
        # STORE TEMPORAL SEQUENCE - this is the execution order for online monitor
        graph.set_metadata('temporal_sequence', temporal_sequence)
        
        print(f"  ✓ Temporal sequence stored in metadata for deterministic replay")
        
        # Step 7: Validate graph
        is_valid, errors = graph.validate()
        if not is_valid:
            print("\nWARNING: Graph validation failed:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  Graph validation: PASSED")
        
        # Step 8: Save to JSON
        graph.save_json(output_json_path)
        
        # Print summary
        graph.print_summary()
        
        print(f"\n{'='*60}")
        print(f"GRAPH GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nKey Features:")
        print(f"  • States are merged when similar poses repeat")
        print(f"  • Graph reflects actual video flow (not forced linear)")
        print(f"  • Temporal sequence stored for deterministic replay")
        print(f"  • Online monitor will follow exact reference video order")
        print(f"{'='*60}\n")
        
        return graph
    
    def _extract_features_from_video(
        self,
        cap: cv2.VideoCapture,
        max_frames: Optional[int],
        skip_frames: int
    ) -> Tuple[List[dict], List[Optional[np.ndarray]]]:
        """
        Extract pose features AND keypoints from all video frames.
        ENFORCES SINGLE-PERSON DETECTION.
        
        Returns:
            Tuple of (feature_sequence, keypoint_sequence)
            - feature_sequence: List of feature dictionaries
            - keypoint_sequence: List of keypoint arrays (17, 3) or None
        """
        feature_sequence = []
        keypoint_sequence = []
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            # Check max frames limit
            if max_frames and processed_count >= max_frames:
                break
            
            # Run pose estimation
            results = self.model(frame, verbose=False)
            
            # Extract features AND keypoints from SINGLE primary subject
            features, keypoints = self._extract_features_from_result(results[0])
            feature_sequence.append(features)
            keypoint_sequence.append(keypoints)
            
            processed_count += 1
            
            # Progress indicator
            if processed_count % 30 == 0:
                print(f"  Processed: {processed_count} frames", end='\r')
            
            frame_idx += 1
        
        print(f"  Processed: {processed_count} frames")
        return feature_sequence, keypoint_sequence
    
    def _select_primary_person(self, result) -> Optional[np.ndarray]:
        """
        Select the primary person from multi-person detections.
        
        STRATEGY: Highest mean keypoint confidence with bbox size as tiebreaker.
        
        Args:
            result: YOLO result object with keypoints and boxes
        
        Returns:
            Keypoints array for primary person, or None if no valid detection
        """
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        
        num_detections = len(result.keypoints)
        
        if num_detections == 0:
            return None
        
        # Single person - use directly
        if num_detections == 1:
            return result.keypoints.data[0].cpu().numpy()
        
        # Multiple people detected - select by keypoint quality + bbox size
        keypoints_list = result.keypoints.data.cpu().numpy()
        
        # Calculate scores for each detection
        best_score = -1
        best_idx = 0
        
        for i in range(num_detections):
            kpts = keypoints_list[i]
            
            # Mean confidence of all keypoints
            mean_confidence = kpts[:, 2].mean()
            
            # Bbox area (if available)
            bbox_area = 1.0
            if result.boxes is not None and len(result.boxes) > i:
                box = result.boxes.data[i].cpu().numpy()
                x1, y1, x2, y2 = box[:4]
                bbox_area = (x2 - x1) * (y2 - y1)
                # Normalize bbox area
                bbox_area = min(bbox_area / (1920 * 1080), 1.0)
            
            # Combined score: 70% confidence + 30% bbox size
            score = 0.7 * mean_confidence + 0.3 * bbox_area
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return keypoints_list[best_idx]
    
    def _extract_features_from_result(self, result) -> Tuple[dict, Optional[np.ndarray]]:
        """
        Extract pose features AND keypoints from YOLO result.
        ENFORCES SINGLE-PERSON SELECTION.
        
        Args:
            result: YOLO result object
        
        Returns:
            Tuple of (features_dict, keypoints_array)
            - features_dict: Feature dictionary (empty if no valid pose)
            - keypoints_array: Raw keypoints (17, 3) or None
        """
        # Select primary person (best quality + prominence)
        keypoints = self._select_primary_person(result)
        
        if keypoints is None:
            return {}, None
        
        # Extract features using feature extractor
        features = self.feature_extractor.extract_features(keypoints)
        
        # Return both features AND raw keypoints
        return (features if features else {}, keypoints)


def main():
    """Example usage of offline graph generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate pose graph from reference video (with state merging and temporal sequence)'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to reference video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pose_graph.json',
        help='Output JSON file path (default: pose_graph.json)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n-pose.pt',
        help='YOLOv8 pose model (default: yolov8n-pose.pt)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process (default: all)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1)'
    )
    parser.add_argument(
        '--stability',
        type=float,
        default=0.12,
        help='Stability threshold for state detection AND merging (default: 0.12)'
    )
    parser.add_argument(
        '--min-hold',
        type=int,
        default=18,
        help='Minimum frames to hold pose (default: 18)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.6,
        help='Keypoint confidence threshold (default: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = OfflineGraphGenerator(
        model_name=args.model,
        confidence_threshold=args.confidence,
        stability_threshold=args.stability,
        min_hold_frames=args.min_hold
    )
    
    # Generate graph
    try:
        graph = generator.generate_from_video(
            video_path=args.video_path,
            output_json_path=args.output,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )
        print("SUCCESS: Pose graph generated successfully!")
        print(f"\nThe graph now:")
        print(f"  ✓ Merges repeated poses into the same state")
        print(f"  ✓ Reflects actual video transitions (non-linear)")
        print(f"  ✓ Stores temporal sequence for deterministic replay")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())