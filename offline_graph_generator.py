"""
Offline Pose Graph Generator
Processes a reference video to automatically extract pose states
and build a pose graph for online monitoring.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
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
    3. Extract normalized features from keypoints
    4. Discover stable pose states using temporal analysis
    5. Build directed graph of state transitions
    6. Save graph to JSON
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n-pose.pt',
        confidence_threshold: float = 0.5,
        stability_threshold: float = 0.15,
        min_hold_frames: int = 15,
        smoothing_window: int = 5,
        transition_buffer: int = 3,
        tolerance_multiplier: float = 2.0
    ):
        """
        Args:
            model_name: YOLOv8 pose model name
            confidence_threshold: Minimum keypoint confidence
            stability_threshold: Maximum distance for same state
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
        
        Args:
            video_path: Path to reference video file
            output_json_path: Path for output JSON file
            max_frames: Maximum frames to process (None = all)
            skip_frames: Process every Nth frame (1 = all frames)
        
        Returns:
            Generated PoseGraph object
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
        
        # Step 2: Extract pose features from all frames
        print(f"\nExtracting pose features...")
        feature_sequence = self._extract_features_from_video(
            cap, max_frames, skip_frames
        )
        cap.release()
        
        valid_frames = sum(1 for f in feature_sequence if f)
        print(f"  Processed: {len(feature_sequence)} frames")
        print(f"  Valid poses: {valid_frames} frames ({valid_frames/len(feature_sequence)*100:.1f}%)")
        
        if valid_frames < 10:
            raise ValueError("Too few valid poses detected. Check video quality or camera angle.")
        
        # Step 3: Discover pose states
        print(f"\nDiscovering pose states...")
        states = self.state_discovery.discover_states(feature_sequence, effective_fps)
        
        print(f"  Discovered: {len(states)} distinct pose states")
        
        if len(states) == 0:
            raise ValueError("No pose states discovered. Try adjusting parameters or video.")
        
        # Step 4: Build pose graph
        print(f"\nBuilding pose graph...")
        graph = PoseGraph()
        graph.build_linear_sequence(states)
        
        # Add metadata
        graph.set_metadata('video_path', os.path.basename(video_path))
        graph.set_metadata('fps', float(effective_fps))
        graph.set_metadata('original_fps', float(fps))
        graph.set_metadata('skip_frames', skip_frames)
        graph.set_metadata('total_frames', len(feature_sequence))
        graph.set_metadata('valid_frames', valid_frames)
        graph.set_metadata('num_states', len(states))
        graph.set_metadata('created_at', datetime.now().isoformat())
        
        # Step 5: Validate graph
        is_valid, errors = graph.validate()
        if not is_valid:
            print("\nWARNING: Graph validation failed:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  Graph validation: PASSED")
        
        # Step 6: Save to JSON
        graph.save_json(output_json_path)
        
        # Print summary
        graph.print_summary()
        
        print(f"\n{'='*60}")
        print(f"GRAPH GENERATION COMPLETE")
        print(f"{'='*60}\n")
        
        return graph
    
    def _extract_features_from_video(
        self,
        cap: cv2.VideoCapture,
        max_frames: Optional[int],
        skip_frames: int
    ) -> List[dict]:
        """
        Extract pose features from all video frames.
        
        Returns:
            List of feature dictionaries (empty dict if detection failed)
        """
        feature_sequence = []
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
            
            # Extract features from detection
            features = self._extract_features_from_result(results[0])
            feature_sequence.append(features)
            
            processed_count += 1
            
            # Progress indicator
            if processed_count % 30 == 0:
                print(f"  Processed: {processed_count} frames", end='\r')
            
            frame_idx += 1
        
        print(f"  Processed: {processed_count} frames")
        return feature_sequence
    
    def _extract_features_from_result(self, result) -> dict:
        """
        Extract pose features from YOLO result.
        
        Args:
            result: YOLO result object
        
        Returns:
            Feature dictionary (empty if no valid pose detected)
        """
        if result.keypoints is None or len(result.keypoints) == 0:
            return {}
        
        # Get first detected person (assume single person in reference video)
        keypoints = result.keypoints.data[0].cpu().numpy()
        
        # Extract features using feature extractor
        features = self.feature_extractor.extract_features(keypoints)
        
        return features if features else {}


def main():
    """Example usage of offline graph generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate pose graph from reference video'
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
        default=0.15,
        help='Stability threshold for state detection (default: 0.15)'
    )
    parser.add_argument(
        '--min-hold',
        type=int,
        default=15,
        help='Minimum frames to hold pose (default: 15)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Keypoint confidence threshold (default: 0.5)'
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
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())