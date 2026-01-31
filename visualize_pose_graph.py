"""
Pose Graph Visualization Tool
Visualizes the directed state graph from a pose graph JSON file.

Usage:
    python visualize_pose_graph.py pose_graph.json
    python visualize_pose_graph.py pose_graph.json --layout spring
    python visualize_pose_graph.py pose_graph.json --save output.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple


class PoseGraphVisualizer:
    """
    Visualizes pose state transition graphs using NetworkX and Matplotlib.
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
        print(f"  Metadata: {data.get('metadata', {})}")
        
        return data
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from pose graph data."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for state in self.graph_data['states']:
            state_id = state['state_id']
            hold_duration = state['min_hold_duration']
            num_features = len(state['mean_features'])
            
            G.add_node(
                state_id,
                hold_duration=hold_duration,
                num_features=num_features
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
        figsize: Tuple[int, int] = (12, 8),
        show: bool = True
    ):
        """
        Visualize the pose graph.
        
        Args:
            layout: Layout algorithm ('circular', 'spring', 'hierarchical', 'shell')
            save_path: If provided, save figure to this path
            figsize: Figure size (width, height)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'circular':
            pos = nx.circular_layout(self.nx_graph)
        elif layout == 'spring':
            pos = nx.spring_layout(self.nx_graph, k=2, iterations=50, seed=42)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout()
        elif layout == 'shell':
            pos = nx.shell_layout(self.nx_graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        # Draw nodes
        node_colors = self._get_node_colors()
        node_sizes = self._get_node_sizes()
        
        nx.draw_networkx_nodes(
            self.nx_graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with arrows
        nx.draw_networkx_edges(
            self.nx_graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw labels
        labels = self._get_node_labels()
        nx.draw_networkx_labels(
            self.nx_graph,
            pos,
            labels,
            font_size=10,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        # Add title and metadata
        metadata = self.graph_data.get('metadata', {})
        video_name = metadata.get('video_path', 'Unknown')
        num_states = len(self.graph_data['states'])
        
        title = f"Pose State Transition Graph\n"
        title += f"Video: {video_name} | States: {num_states} | Layout: {layout}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        self._add_legend(ax)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig, ax
    
    def _hierarchical_layout(self) -> Dict:
        """
        Create hierarchical layout (left-to-right flow).
        Assumes linear or near-linear state progression.
        """
        pos = {}
        num_nodes = len(self.nx_graph.nodes())
        
        # Simple left-to-right layout
        for i, node in enumerate(sorted(self.nx_graph.nodes())):
            x = i / max(num_nodes - 1, 1)  # Normalize to [0, 1]
            y = 0.5  # Center vertically
            pos[node] = (x, y)
        
        return pos
    
    def _get_node_colors(self) -> List[str]:
        """
        Assign colors based on node position in sequence.
        Start state = green, end state = red, middle = gradient.
        """
        nodes = sorted(self.nx_graph.nodes())
        num_nodes = len(nodes)
        
        colors = []
        for i, node in enumerate(nodes):
            if i == 0:
                # Start state - green
                colors.append('#2ecc71')
            elif i == num_nodes - 1:
                # End state - red
                colors.append('#e74c3c')
            else:
                # Middle states - blue gradient
                intensity = 0.4 + 0.6 * (i / num_nodes)
                colors.append(f'#{int(52*intensity):02x}{int(152*intensity):02x}{int(219*intensity):02x}')
        
        return colors
    
    def _get_node_sizes(self) -> List[int]:
        """
        Node size proportional to hold duration.
        Longer holds = larger nodes.
        """
        sizes = []
        max_duration = max(
            data['hold_duration'] 
            for _, data in self.nx_graph.nodes(data=True)
        )
        
        for node in sorted(self.nx_graph.nodes()):
            hold_duration = self.nx_graph.nodes[node]['hold_duration']
            # Scale to range [800, 2000]
            size = 800 + (hold_duration / max_duration) * 1200
            sizes.append(size)
        
        return sizes
    
    def _get_node_labels(self) -> Dict[int, str]:
        """
        Create node labels showing state ID and hold duration.
        """
        labels = {}
        for node in self.nx_graph.nodes():
            hold_duration = self.nx_graph.nodes[node]['hold_duration']
            labels[node] = f"S{node}\n{hold_duration:.1f}s"
        
        return labels
    
    def _add_legend(self, ax):
        """Add legend explaining visualization."""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Start State'),
            Patch(facecolor='#3498db', label='Middle States'),
            Patch(facecolor='#e74c3c', label='End State'),
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9
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
        
        # Connectivity
        if nx.is_strongly_connected(self.nx_graph):
            print(f"\nConnectivity: Strongly connected (all states reachable)")
        elif nx.is_weakly_connected(self.nx_graph):
            print(f"\nConnectivity: Weakly connected")
        else:
            print(f"\nConnectivity: Disconnected")
        
        # Degree distribution
        in_degrees = dict(self.nx_graph.in_degree())
        out_degrees = dict(self.nx_graph.out_degree())
        
        print(f"\nDegree Distribution:")
        print(f"  States with no incoming edges: {sum(1 for d in in_degrees.values() if d == 0)}")
        print(f"  States with no outgoing edges: {sum(1 for d in out_degrees.values() if d == 0)}")
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.nx_graph))
            if cycles:
                print(f"\nCycles detected: {len(cycles)}")
                print(f"  (Indicates looping sequences)")
            else:
                print(f"\nNo cycles (Linear sequence)")
        except:
            pass
        
        print("\n" + "="*60 + "\n")


def main():
    """Command-line interface for pose graph visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize pose state transition graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_pose_graph.py pose_graph.json
  python visualize_pose_graph.py pose_graph.json --layout spring
  python visualize_pose_graph.py pose_graph.json --layout hierarchical --save viz.png
  python visualize_pose_graph.py pose_graph.json --stats-only
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
        default=[12, 8],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 12 8)'
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