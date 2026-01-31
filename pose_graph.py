"""
Pose Graph Module (UPDATED FOR NON-LINEAR GRAPHS)
Manages the directed graph of pose states and their transitions.
Provides serialization/deserialization and validation logic.

MODIFICATIONS:
- Removed forced linear sequence construction
- Added incremental transition building based on actual video flow
- Supports cycles, branches, and merges in the graph
"""

import json
from typing import List, Dict, Optional, Tuple, Set
from state_discovery import PoseState


class PoseGraph:
    """
    Directed graph representing valid pose state transitions.
    Each node is a PoseState, edges represent valid transitions.
    
    NOW SUPPORTS NON-LINEAR GRAPHS:
    - States can have multiple incoming and outgoing edges
    - Cycles are allowed (e.g., A → B → C → A)
    - Branches are allowed (e.g., A → B or A → C)
    """
    
    def __init__(self):
        """Initialize empty pose graph."""
        self.states: List[PoseState] = []
        self.transitions: Dict[int, List[int]] = {}  # state_id -> [next_state_ids]
        self.metadata: Dict = {}
    
    def add_state(self, state: PoseState) -> None:
        """Add a pose state to the graph."""
        self.states.append(state)
        if state.state_id not in self.transitions:
            self.transitions[state.state_id] = []
    
    def add_transition(self, from_state_id: int, to_state_id: int) -> None:
        """
        Add a directed edge between two states.
        
        UPDATED: Now checks for duplicate edges to avoid redundant transitions.
        
        Args:
            from_state_id: Source state ID
            to_state_id: Destination state ID
        """
        if from_state_id not in self.transitions:
            self.transitions[from_state_id] = []
        
        # Avoid duplicate edges
        if to_state_id not in self.transitions[from_state_id]:
            self.transitions[from_state_id].append(to_state_id)
    
    def build_from_temporal_sequence(
        self,
        states: List[PoseState],
        temporal_sequence: List[int]
    ) -> None:
        """
        Build graph from states and their temporal sequence.
        
        This is the NEW way to construct the graph, replacing build_linear_sequence().
        
        ALGORITHM:
        1. Add all states to the graph
        2. Walk through temporal sequence
        3. For each consecutive pair (state_i, state_j), add transition i → j
        4. Automatically handles cycles and branches
        
        Example:
            states = [State0, State1, State2]
            temporal_sequence = [0, 1, 2, 0, 1]
            
            Results in edges:
            0 → 1 (appears twice, but stored once due to deduplication)
            1 → 2
            2 → 0
            
        Args:
            states: List of PoseState objects
            temporal_sequence: List of state IDs in temporal order (e.g., [0, 1, 2, 0, 1, 3])
        """
        # Clear existing graph
        self.states = []
        self.transitions = {}
        
        # Add all states
        for state in states:
            self.add_state(state)
        
        # Build transitions from temporal sequence
        if len(temporal_sequence) < 2:
            # Edge case: single state or empty sequence
            # No transitions to add
            return
        
        for i in range(len(temporal_sequence) - 1):
            from_state_id = temporal_sequence[i]
            to_state_id = temporal_sequence[i + 1]
            
            # Add transition (deduplication handled by add_transition)
            self.add_transition(from_state_id, to_state_id)
        
        # Optional: Add transition from last state back to first for looping
        # This depends on whether the exercise is meant to loop
        # For now, we'll add it to support repetitive exercises
        if temporal_sequence:
            last_state_id = temporal_sequence[-1]
            first_state_id = temporal_sequence[0]
            
            # Only add if it creates a meaningful cycle (not self-loop)
            if last_state_id != first_state_id:
                self.add_transition(last_state_id, first_state_id)
    
    def build_linear_sequence(self, states: List[PoseState]) -> None:
        """
        Build a linear sequence graph from ordered states.
        Each state transitions to the next, with last transitioning to first (loop).
        
        DEPRECATED: Use build_from_temporal_sequence() instead for true video flow.
        This method is kept for backward compatibility.
        
        Args:
            states: Ordered list of PoseState objects
        """
        self.states = states
        self.transitions = {}
        
        for i, state in enumerate(states):
            self.transitions[state.state_id] = []
            
            # Add transition to next state
            if i < len(states) - 1:
                next_state_id = states[i + 1].state_id
                self.add_transition(state.state_id, next_state_id)
            else:
                # Last state can transition back to first (for looping sequences)
                first_state_id = states[0].state_id
                self.add_transition(state.state_id, first_state_id)
    
    def get_state(self, state_id: int) -> Optional[PoseState]:
        """Retrieve a state by ID."""
        for state in self.states:
            if state.state_id == state_id:
                return state
        return None
    
    def get_next_states(self, current_state_id: int) -> List[int]:
        """Get list of valid next state IDs from current state."""
        return self.transitions.get(current_state_id, [])
    
    def is_valid_transition(self, from_state_id: int, to_state_id: int) -> bool:
        """Check if transition from one state to another is valid."""
        return to_state_id in self.transitions.get(from_state_id, [])
    
    def get_initial_state(self) -> Optional[PoseState]:
        """Get the first state in the sequence."""
        if self.states:
            return self.states[0]
        return None
    
    def get_final_state(self) -> Optional[PoseState]:
        """Get the last state in the sequence."""
        if self.states:
            return self.states[-1]
        return None
    
    def get_incoming_transitions(self, state_id: int) -> List[int]:
        """
        Get list of state IDs that can transition TO the given state.
        
        NEW METHOD: Useful for analyzing graph structure.
        
        Args:
            state_id: Target state ID
        
        Returns:
            List of state IDs that have transitions to state_id
        """
        incoming = []
        for from_id, to_ids in self.transitions.items():
            if state_id in to_ids:
                incoming.append(from_id)
        return incoming
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """
        Compute graph statistics for analysis.
        
        NEW METHOD: Provides insight into graph structure.
        
        Returns:
            Dictionary with graph metrics
        """
        if not self.states:
            return {
                'num_states': 0,
                'num_transitions': 0,
                'is_linear': True,
                'has_cycles': False,
                'max_outgoing': 0,
                'max_incoming': 0
            }
        
        num_transitions = sum(len(to_ids) for to_ids in self.transitions.values())
        
        # Check if graph is linear (each state has exactly one outgoing edge)
        is_linear = all(len(to_ids) == 1 for to_ids in self.transitions.values())
        
        # Detect cycles (simple check: any state transitions to a lower ID)
        has_cycles = False
        for from_id, to_ids in self.transitions.items():
            for to_id in to_ids:
                if to_id <= from_id:
                    has_cycles = True
                    break
            if has_cycles:
                break
        
        # Max branching factor
        max_outgoing = max((len(to_ids) for to_ids in self.transitions.values()), default=0)
        
        # Max incoming transitions
        incoming_counts = [len(self.get_incoming_transitions(s.state_id)) for s in self.states]
        max_incoming = max(incoming_counts, default=0)
        
        return {
            'num_states': len(self.states),
            'num_transitions': num_transitions,
            'is_linear': is_linear,
            'has_cycles': has_cycles,
            'max_outgoing': max_outgoing,
            'max_incoming': max_incoming,
            'avg_transitions_per_state': num_transitions / len(self.states) if self.states else 0
        }
    
    def set_metadata(self, key: str, value) -> None:
        """Set metadata field (e.g., video name, fps, date created)."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default=None):
        """Get metadata field."""
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict:
        """
        Convert graph to dictionary for JSON serialization.
        
        Returns:
            Dictionary with complete graph structure
        """
        return {
            'version': '1.1',  # Incremented version for non-linear support
            'metadata': self.metadata,
            'states': [state.to_dict() for state in self.states],
            'transitions': self.transitions
        }
    
    def save_json(self, filepath: str) -> None:
        """
        Save graph to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Pose graph saved to: {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PoseGraph':
        """
        Load graph from dictionary.
        
        Args:
            data: Dictionary with graph structure
        
        Returns:
            PoseGraph instance
        """
        graph = cls()
        
        # Load metadata
        graph.metadata = data.get('metadata', {})
        
        # Load states
        for state_data in data.get('states', []):
            state = PoseState.from_dict(state_data)
            graph.add_state(state)
        
        # Load transitions
        transitions = data.get('transitions', {})
        for from_id_str, to_ids in transitions.items():
            from_id = int(from_id_str)
            graph.transitions[from_id] = to_ids
        
        return graph
    
    @classmethod
    def load_json(cls, filepath: str) -> 'PoseGraph':
        """
        Load graph from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            PoseGraph instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Pose graph loaded from: {filepath}")
        return cls.from_dict(data)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate graph structure and consistency.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if graph has states
        if not self.states:
            errors.append("Graph has no states")
            return False, errors
        
        # Check for duplicate state IDs
        state_ids = [s.state_id for s in self.states]
        if len(state_ids) != len(set(state_ids)):
            errors.append("Duplicate state IDs found")
        
        # Check if all transition references are valid
        valid_state_ids = set(state_ids)
        
        for from_id, to_ids in self.transitions.items():
            if from_id not in valid_state_ids:
                errors.append(f"Transition references non-existent source state: {from_id}")
            
            for to_id in to_ids:
                if to_id not in valid_state_ids:
                    errors.append(f"Transition to non-existent state: {to_id}")
        
        # Check if all states have valid features
        for state in self.states:
            if not state.mean_features:
                errors.append(f"State {state.state_id} has no features")
            if not state.feature_tolerances:
                errors.append(f"State {state.state_id} has no tolerances")
        
        # Check for isolated states (no outgoing transitions)
        for state in self.states:
            if state.state_id not in self.transitions or not self.transitions[state.state_id]:
                errors.append(f"State {state.state_id} has no outgoing transitions (isolated)")
        
        return len(errors) == 0, errors
    @classmethod
    def load(cls, filepath: str) -> 'PoseGraph':
        """
        Alias for load_json() to support alternative API usage.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            PoseGraph instance
        """
        return cls.load_json(filepath)
    def print_summary(self) -> None:
        """Print human-readable summary of the graph."""
        print("\n" + "="*60)
        print("POSE GRAPH SUMMARY")
        print("="*60)
        
        print(f"\nMetadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
        
        # Print graph statistics
        stats = self.get_graph_statistics()
        print(f"\nGraph Structure:")
        print(f"  Total States: {stats['num_states']}")
        print(f"  Total Transitions: {stats['num_transitions']}")
        print(f"  Graph Type: {'Linear' if stats['is_linear'] else 'Non-Linear'}")
        print(f"  Has Cycles: {'Yes' if stats['has_cycles'] else 'No'}")
        print(f"  Max Outgoing Transitions: {stats['max_outgoing']}")
        print(f"  Max Incoming Transitions: {stats['max_incoming']}")
        print(f"  Avg Transitions per State: {stats['avg_transitions_per_state']:.2f}")
        
        print(f"\nState Details:")
        
        for i, state in enumerate(self.states):
            print(f"\n  State {state.state_id}:")
            print(f"    Min Hold Duration: {state.min_hold_duration:.2f}s")
            print(f"    Features: {len(state.mean_features)}")
            
            # Show sample features
            sample_features = list(state.mean_features.items())[:3]
            for fname, fval in sample_features:
                tolerance = state.feature_tolerances.get(fname, 0)
                print(f"      {fname}: {fval:.2f} ± {tolerance:.2f}")
            
            if len(state.mean_features) > 3:
                print(f"      ... and {len(state.mean_features) - 3} more features")
            
            # Show transitions
            next_states = self.get_next_states(state.state_id)
            if next_states:
                print(f"    Next States: {next_states}")
            
            # Show incoming transitions (useful for non-linear graphs)
            incoming_states = self.get_incoming_transitions(state.state_id)
            if len(incoming_states) > 1 or (incoming_states and not stats['is_linear']):
                print(f"    Previous States: {incoming_states}")
        
        print("\n" + "="*60 + "\n")