"""
Pose Graph Visualization with Real Stick Figure Nodes
Visualizes pose state graphs using ACTUAL stored keypoints from the JSON.

NO RECONSTRUCTION - Uses exact mean keypoint positions stored during graph generation.

Usage:
    python visualize_pose_graph.py pose_graph.json
    python visualize_pose_graph.py pose_graph.json --layout spring --save output.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional


# COCO 17-keypoint skeleton connections (bone definitions)
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    # Torso
    (5, 6),   # shoulders
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    # Left arm
    (5, 7), (7, 9),  # shoulder -> elbow -> wrist
    # Right arm
    (6, 8), (8, 10),  # shoulder -> elbow -> wrist
    # Left leg
    (11, 13), (13, 15),  # hip -> knee -> ankle
    # Right leg
    (12, 14), (14, 16),  # hip -> knee -> ankle
]


class PoseGraphVisualizer:
    """
    Visualizes pose state transition graphs with stick figure nodes.
    Uses REAL keypoints stored in the JSON (no reconstruction).
    """
    
    def __init__(self, graph_json_path: str):
        """
        Args:
            graph_json_path: Path to pose graph JSON file
        """
        self.graph_path = graph_json_path
        self.graph_data = self._load_graph()
        self.nx_graph = self._build_networkx_graph()
    
    def _load_graph(self) -> Dict:
        """Load and validate pose graph JSON."""
        with open(self.graph_path, 'r') as f:
            data = json.load(f)
        
        if 'states' not in data or 'transitions' not in data:
            raise ValueError("Invalid pose graph format")
        
        print(f"Loaded pose graph: {self.graph_path}")
        print(f"  States: {len(data['states'])}")
        
        # Check if keypoints are present
        has_keypoints = any('mean_keypoints' in state for state in data['states'])
        if has_keypoints:
            print(f"  ✓ Keypoints found - will render actual poses")
        else:
            print(f"  ✗ No keypoints found - regenerate graph with updated code")
            raise ValueError(
                "This pose graph does not contain keypoint data.\n"
                "Please regenerate using the updated offline_graph_generator.py"
            )
        
        return data
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from pose graph data."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for state in self.graph_data['states']:
            state_id = state['state_id']
            hold_duration = state['min_hold_duration']
            mean_keypoints = state.get('mean_keypoints', None)
            
            G.add_node(
                state_id,
                hold_duration=hold_duration,
                mean_keypoints=mean_keypoints
            )
        
        # Add edges (transitions)
        transitions = self.graph_data['transitions']
        for from_id_str, to_ids in transitions.items():
            from_id = int(from_id_str)
            for to_id in to_ids:
                G.add_edge(from_id, to_id)
        
        return G
    
    def visualize(
        self,
        layout: str = 'circular',
        save_path: str = None,
        figsize: Tuple[int, int] = (16, 10),
        show: bool = True
    ):
        """
        Visualize the pose graph with stick figure nodes.
        
        Args:
            layout: Layout algorithm ('circular', 'spring', 'hierarchical', 'shell')
            save_path: If provided, save figure to this path
            figsize: Figure size (width, height)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#f5f5f5')
        
        # Choose layout
        if layout == 'circular':
            pos = nx.circular_layout(self.nx_graph, scale=2.0)
        elif layout == 'spring':
            pos = nx.spring_layout(self.nx_graph, k=3, iterations=50, seed=42, scale=2.0)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout()
        elif layout == 'shell':
            pos = nx.shell_layout(self.nx_graph, scale=2.0)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        # Draw edges first (so they appear behind nodes)
        self._draw_edges(ax, pos)
        
        # Draw stick figure nodes using REAL keypoints
        self._draw_stick_figure_nodes(ax, pos)
        
        # Add title and metadata
        metadata = self.graph_data.get('metadata', {})
        video_name = metadata.get('video_path', 'Unknown')
        num_states = len(self.graph_data['states'])
        
        title = f"Pose State Transition Graph\n"
        title += f"Video: {video_name} | States: {num_states} | Layout: {layout}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.axis('equal')
        ax.axis('off')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved visualization to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig, ax
    
    def _hierarchical_layout(self) -> Dict:
        """Create hierarchical layout (left-to-right flow)."""
        pos = {}
        num_nodes = len(self.nx_graph.nodes())
        
        for i, node in enumerate(sorted(self.nx_graph.nodes())):
            x = i / max(num_nodes - 1, 1) * 4.0  # Scale for spacing
            y = 0.5
            pos[node] = (x, y)
        
        return pos
    
    def _draw_edges(self, ax, pos):
        """Draw directed edges between states."""
        for edge in self.nx_graph.edges():
            start_node, end_node = edge
            start_pos = pos[start_node]
            end_pos = pos[end_node]
            
            # Calculate arrow direction
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            
            # Offset from node center (to avoid overlapping with stick figure)
            node_radius = 0.4
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                
                arrow_start = (start_pos[0] + dx_norm * node_radius, 
                              start_pos[1] + dy_norm * node_radius)
                arrow_end = (end_pos[0] - dx_norm * node_radius,
                            end_pos[1] - dy_norm * node_radius)
                
                ax.annotate('',
                           xy=arrow_end,
                           xytext=arrow_start,
                           arrowprops=dict(
                               arrowstyle='->',
                               lw=2.5,
                               color='#555555',
                               connectionstyle='arc3,rad=0.1',
                               alpha=0.7
                           ))
    
    def _draw_stick_figure_nodes(self, ax, pos):
        """Draw stick figure skeleton at each node position using REAL keypoints."""
        for node in self.nx_graph.nodes():
            node_pos = pos[node]
            mean_keypoints = self.nx_graph.nodes[node]['mean_keypoints']
            hold_duration = self.nx_graph.nodes[node]['hold_duration']
            
            if mean_keypoints is None:
                # Fallback: draw simple circle if no keypoints
                circle = plt.Circle(
                    node_pos, 
                    0.3, 
                    color='lightgray',
                    ec='#333333',
                    linewidth=2,
                    zorder=5
                )
                ax.add_patch(circle)
                ax.text(node_pos[0], node_pos[1], f"S{node}", 
                       ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                # Use REAL stored keypoints
                keypoints = np.array(mean_keypoints)  # Shape: (17, 2)
                
                # Normalize and scale skeleton to fit in node
                keypoints_scaled = self._normalize_skeleton(keypoints)
                
                # Draw background circle
                circle = plt.Circle(
                    node_pos, 
                    0.38, 
                    color='white',
                    ec='#333333',
                    linewidth=2.5,
                    zorder=5,
                    alpha=0.95
                )
                ax.add_patch(circle)
                
                # Draw skeleton using REAL keypoints
                self._draw_skeleton(ax, keypoints_scaled, node_pos, scale=0.32)
            
            # Add state label below node
            ax.text(
                node_pos[0], 
                node_pos[1] - 0.55, 
                f"State {node}\n{hold_duration:.1f}s",
                ha='center',
                va='top',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='#333333', linewidth=1.5, alpha=0.9)
            )
    
    def _normalize_skeleton(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize skeleton to fit in unit square centered at origin.
        
        APPROACH:
        - Center the skeleton at origin (0, 0)
        - FLIP Y-axis (image coords have Y pointing down, plot has Y pointing up)
        - Scale so largest dimension fits in [-0.5, 0.5]
        - This makes all poses uniform size regardless of camera distance
        
        Args:
            keypoints: Array of shape (17, 2) with [x, y] positions
        
        Returns:
            Normalized keypoints in range approximately [-0.5, 0.5]
        """
        # Remove invalid keypoints (all zeros or NaN)
        valid_mask = ~(np.all(keypoints == 0, axis=1) | np.any(np.isnan(keypoints), axis=1))
        
        if not np.any(valid_mask):
            return keypoints
        
        valid_kpts = keypoints[valid_mask]
        
        # Center at origin
        center = valid_kpts.mean(axis=0)
        keypoints_centered = keypoints - center
        
        # FLIP Y-axis to correct upside-down orientation
        # Image coords: Y increases downward
        # Plot coords: Y increases upward
        keypoints_centered[:, 1] = -keypoints_centered[:, 1]
        
        # Scale to fit in unit square
        # Use max extent across both x and y
        max_extent_x = np.abs(valid_kpts[:, 0] - center[0]).max()
        max_extent_y = np.abs(valid_kpts[:, 1] - center[1]).max()
        max_extent = max(max_extent_x, max_extent_y)
        
        if max_extent > 1e-6:
            # Scale to fit with some padding (2.5x gives nice margins)
            keypoints_scaled = keypoints_centered / (max_extent * 2.5)
        else:
            keypoints_scaled = keypoints_centered
        
        return keypoints_scaled
    
    def _draw_skeleton(
        self, 
        ax, 
        keypoints: np.ndarray, 
        center_pos: Tuple[float, float],
        scale: float = 0.3
    ):
        """
        Draw stick figure skeleton using REAL keypoint positions.
        
        Args:
            ax: Matplotlib axis
            keypoints: Normalized keypoints (17, 2) in range [-0.5, 0.5]
            center_pos: Center position (x, y) on the plot
            scale: Scaling factor to fit in node circle
        """
        # Transform keypoints to plot coordinates
        kpts_plot = keypoints * scale + np.array(center_pos)
        
        # Draw bones (connections) first
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            start_pos = kpts_plot[start_idx]
            end_pos = kpts_plot[end_idx]
            
            # Skip if either point is at origin (invalid keypoint)
            if np.allclose(keypoints[start_idx], 0) or np.allclose(keypoints[end_idx], 0):
                continue
            
            # Skip if either point is NaN
            if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
                continue
            
            # Draw bone as thick line
            ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                color='#4CAF50',  # Green
                linewidth=3.0,
                solid_capstyle='round',
                zorder=6,
                alpha=0.9
            )
        
        # Draw joints (keypoints) on top
        for i, kpt in enumerate(kpts_plot):
            # Skip invalid keypoints
            if np.allclose(keypoints[i], 0) or np.any(np.isnan(kpt)):
                continue
            
            # Draw joint as circle with border
            ax.plot(
                kpt[0], kpt[1],
                'o',
                color='white',
                markersize=5,
                markeredgecolor='#2196F3',  # Blue
                markeredgewidth=2,
                zorder=7
            )
    
    def print_statistics(self):
        """Print graph statistics."""
        print("\n" + "="*60)
        print("POSE GRAPH STATISTICS")
        print("="*60)
        
        num_nodes = self.nx_graph.number_of_nodes()
        num_edges = self.nx_graph.number_of_edges()
        
        print(f"\nNodes (States): {num_nodes}")
        print(f"Edges (Transitions): {num_edges}")
        
        # Hold durations
        durations = [
            data['hold_duration'] 
            for _, data in self.nx_graph.nodes(data=True)
        ]
        print(f"\nHold Durations:")
        print(f"  Min: {min(durations):.2f}s")
        print(f"  Max: {max(durations):.2f}s")
        print(f"  Mean: {sum(durations)/len(durations):.2f}s")
        print(f"  Total: {sum(durations):.2f}s")
        
        # Check keypoint availability
        has_keypoints = sum(
            1 for _, data in self.nx_graph.nodes(data=True) 
            if data['mean_keypoints'] is not None
        )
        print(f"\nKeypoint Data:")
        print(f"  States with keypoints: {has_keypoints}/{num_nodes}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Command-line interface for pose graph visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize pose state transition graph with real stick figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_pose_graph.py pose_graph.json
  python visualize_pose_graph.py pose_graph.json --layout spring
  python visualize_pose_graph.py pose_graph.json --layout hierarchical --save viz.png
  python visualize_pose_graph.py pose_graph.json --stats-only

NOTE: The pose graph must contain 'mean_keypoints' data.
If you get an error, regenerate the graph using the updated offline_graph_generator.py
        """
    )
    
    parser.add_argument(
        'graph_path',
        type=str,
        help='Path to pose graph JSON file'
    )
    
    parser.add_argument(
        '--layout',
        type=str,
        default='circular',
        choices=['circular', 'spring', 'hierarchical', 'shell'],
        help='Graph layout algorithm (default: circular)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save visualization to file (e.g., graph.png)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (useful with --save)'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Print statistics only, do not visualize'
    )
    
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[16, 10],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 16 10)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = PoseGraphVisualizer(args.graph_path)
        
        # Print statistics
        visualizer.print_statistics()
        
        # Visualize if not stats-only
        if not args.stats_only:
            visualizer.visualize(
                layout=args.layout,
                save_path=args.save,
                figsize=tuple(args.figsize),
                show=not args.no_show
            )
        
        print("SUCCESS: Visualization complete!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())