#!/usr/bin/env python3
"""
Scalable RTL Component Reusability Analyzer
Optimized for large-scale designs with hybrid analysis modes.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
import hashlib
import random
from dataclasses import dataclass
import json

# ==================== CONFIGURATION ====================
class AnalysisConfig:
    """Configuration for scalable analysis."""
    
    # Sampling parameters (to handle huge graphs)
    MAX_NODES_FULL_ANALYSIS = 5000  # Full analysis up to 5k nodes
    SAMPLE_RATIO = 0.3  # Sample 30% of nodes for huge graphs
    MAX_MOTIF_SAMPLES = 500  # Max motifs to sample
    
    # Analysis modes
    USE_VVP_IF_AVAILABLE = True  # Try iverilog first
    FALLBACK_TO_REGEX = True  # Use regex if VVP fails
    
    # Edit distance timeout
    EDIT_DISTANCE_TIMEOUT = 60  # seconds
    
    # Parallel processing
    USE_PARALLEL = False  # Set to True if multiprocessing available

@dataclass
class AnalysisMode:
    """Track which analysis mode was used."""
    project: str
    mode: str  # 'vvp', 'regex', 'hybrid'
    success: bool
    nodes: int
    edges: int

class IVerilogVVPExtractor:
    """Extract gate-level netlist using iverilog compilation."""
    
    def __init__(self):
        self.iverilog_available = self._check_iverilog()
    
    def _check_iverilog(self) -> bool:
        """Check if iverilog is installed."""
        try:
            subprocess.run(['iverilog', '-V'], 
                         capture_output=True, 
                         timeout=5)
            return True
        except:
            return False
    
    def compile_to_vvp(self, rtl_files: List[Path], 
                       output_vvp: Path) -> bool:
        """
        Compile RTL files to VVP netlist.
        Returns True if successful.
        """
        if not self.iverilog_available:
            return False
        
        try:
            # Create file list
            file_list = [str(f.absolute()) for f in rtl_files]
            
            # Try to compile
            cmd = ['iverilog', '-o', str(output_vvp)] + file_list
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
                text=True
            )
            
            return result.returncode == 0 and output_vvp.exists()
            
        except Exception as e:
            print(f"    ⚠ iverilog compilation failed: {e}")
            return False
    
    def extract_graph_from_vvp(self, vvp_file: Path) -> nx.DiGraph:
        """Extract connectivity graph from VVP file."""
        G = nx.DiGraph()
        
        try:
            with open(vvp_file, 'r') as f:
                content = f.read()
            
            # Parse VVP format (simplified - actual parsing is complex)
            # This is a basic example - full VVP parsing needs more work
            
            # Look for gate instantiations in VVP
            # Format: L_0x... .functor AND 1, L_0x..., L_0x..., C4<0>, C4<0>;
            
            functor_pattern = r'L_(\w+)\s+\.functor\s+(\w+)'
            
            for match in re.finditer(functor_pattern, content):
                node_id = match.group(1)
                gate_type = match.group(2)
                
                G.add_node(node_id, gate_type=gate_type)
            
            # Extract connections (simplified)
            # Actual VVP format is more complex
            
            return G
            
        except Exception as e:
            print(f"    ⚠ VVP parsing failed: {e}")
            return nx.DiGraph()

class OptimizedGraphAnalyzer:
    """Memory-efficient graph analysis with sampling."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def sample_graph(self, G: nx.DiGraph, sample_ratio: float = 0.3) -> nx.DiGraph:
        """Sample a subgraph for analysis."""
        if len(G.nodes) <= self.config.MAX_NODES_FULL_ANALYSIS:
            return G
        
        print(f"    → Sampling {sample_ratio*100:.0f}% of {len(G.nodes)} nodes for scalability")
        
        # Sample nodes
        num_samples = int(len(G.nodes) * sample_ratio)
        sampled_nodes = random.sample(list(G.nodes), num_samples)
        
        # Create subgraph
        return G.subgraph(sampled_nodes).copy()
    
    def compute_graph_features(self, G: nx.DiGraph) -> Dict:
        """Extract graph features efficiently."""
        
        if len(G.nodes) == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'avg_degree': 0,
                'density': 0,
                'clustering': 0
            }
        
        degrees = [d for n, d in G.degree()]
        
        # For very large graphs, approximate clustering
        if len(G.nodes) > 1000:
            # Sample 100 nodes for clustering
            sample_nodes = random.sample(list(G.nodes), min(100, len(G.nodes)))
            clustering = nx.average_clustering(G.subgraph(sample_nodes))
        else:
            try:
                clustering = nx.average_clustering(G)
            except:
                clustering = 0
        
        return {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': np.max(degrees) if degrees else 0,
            'density': nx.density(G),
            'clustering': clustering
        }
    
    def find_common_patterns(self, G1: nx.DiGraph, G2: nx.DiGraph,
                           k: int = 3) -> int:
        """
        Find common k-node patterns efficiently using sampling.
        Returns count of similar patterns.
        """
        
        # Sample subgraphs
        G1_sample = self.sample_graph(G1, 0.3)
        G2_sample = self.sample_graph(G2, 0.3)
        
        # Extract degree sequences as pattern signatures
        def get_degree_patterns(G, k):
            patterns = []
            nodes = list(G.nodes)
            
            # Sample random k-node subsets
            num_samples = min(self.config.MAX_MOTIF_SAMPLES, 
                            len(nodes) // k)
            
            for _ in range(num_samples):
                if len(nodes) >= k:
                    subset = random.sample(nodes, k)
                    subgraph = G.subgraph(subset)
                    
                    # Pattern: sorted degree sequence
                    pattern = tuple(sorted([subgraph.degree(n) for n in subset]))
                    patterns.append(pattern)
            
            return Counter(patterns)
        
        patterns1 = get_degree_patterns(G1_sample, k)
        patterns2 = get_degree_patterns(G2_sample, k)
        
        # Count common patterns
        common_count = sum((patterns1 & patterns2).values())
        
        return common_count
    
    def compute_structural_similarity(self, G1: nx.DiGraph, 
                                     G2: nx.DiGraph) -> float:
        """
        Fast structural similarity without expensive isomorphism.
        Returns similarity score 0-100.
        """
        
        # Feature-based similarity
        feat1 = self.compute_graph_features(G1)
        feat2 = self.compute_graph_features(G2)
        
        # Compare features
        similarities = []
        
        for key in feat1.keys():
            v1, v2 = feat1[key], feat2[key]
            if v1 + v2 > 0:
                sim = 1 - abs(v1 - v2) / (v1 + v2)
                similarities.append(sim)
        
        # Common patterns
        common_patterns = self.find_common_patterns(G1, G2, k=3)
        max_patterns = min(100, max(len(G1.nodes), len(G2.nodes)) // 3)
        pattern_sim = min(common_patterns / max_patterns, 1.0) if max_patterns > 0 else 0
        
        similarities.append(pattern_sim)
        
        return np.mean(similarities) * 100

class RegexBasedExtractor:
    """Fallback regex-based extraction (from original implementation)."""
    
    def __init__(self):
        self.gate_patterns = {
            'and': r'&(?!&)',
            'or': r'\|(?!\|)',
            'xor': r'\^',
            'not': r'~',
            'add': r'\+',
            'mul': r'\*',
            'mux': r'\?',
            'dff': r'\balways.*posedge',
        }
    
    def remove_comments(self, code: str) -> str:
        """Remove comments."""
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        return code
    
    def extract_gate_counts(self, code: str) -> Dict[str, int]:
        """Count gates."""
        clean_code = self.remove_comments(code)
        return {gate: len(re.findall(pattern, clean_code)) 
                for gate, pattern in self.gate_patterns.items()}
    
    def build_lightweight_graph(self, code: str, 
                               max_assignments: int = 5000) -> nx.DiGraph:
        """
        Build lightweight connectivity graph.
        Limits assignments to prevent memory explosion.
        """
        clean_code = self.remove_comments(code)
        G = nx.DiGraph()
        
        assign_pattern = r'(\w+)\s*[<]?=\s*([^;]+);'
        
        count = 0
        for match in re.finditer(assign_pattern, clean_code):
            if count >= max_assignments:
                print(f"    → Limited to {max_assignments} assignments for memory efficiency")
                break
            
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            
            # Extract signals
            rhs_signals = re.findall(r'\b[a-zA-Z_]\w*\b', rhs)
            
            # Simple edges
            for sig in rhs_signals[:5]:  # Limit RHS signals
                if sig not in ['if', 'else', 'case', 'begin', 'end']:
                    G.add_edge(sig, lhs)
            
            count += 1
        
        return G

class ScalableReusabilityAnalyzer:
    """Main analyzer with hybrid modes and scalability."""
    
    def __init__(self, base_directory: str, config: AnalysisConfig = None):
        self.base_dir = Path(base_directory)
        self.config = config or AnalysisConfig()
        
        self.vvp_extractor = IVerilogVVPExtractor()
        self.regex_extractor = RegexBasedExtractor()
        self.graph_analyzer = OptimizedGraphAnalyzer(self.config)
        
        self.projects = {}
        self.analysis_modes = []
    
    def read_rtl_files(self, directory: Path) -> Tuple[str, List[Path]]:
        """Read RTL files and return code + file paths."""
        rtl_code = ""
        rtl_files = []
        
        for ext in ['*.v', '*.sv']:
            for filepath in directory.rglob(ext):
                try:
                    rtl_files.append(filepath)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        rtl_code += f.read() + "\n"
                except:
                    pass
        
        return rtl_code, rtl_files
    
    def analyze_project_hybrid(self, project_dir: Path, 
                               project_name: str) -> Optional[Dict]:
        """
        Hybrid analysis: Try VVP first, fallback to regex.
        """
        
        print(f"  Mode: Hybrid (VVP → Regex fallback)")
        
        rtl_code, rtl_files = self.read_rtl_files(project_dir)
        
        if not rtl_code:
            return None
        
        print(f"  → Found {len(rtl_files)} RTL files")
        
        # Try VVP first
        conn_graph = None
        mode = 'failed'
        
        if self.config.USE_VVP_IF_AVAILABLE and self.vvp_extractor.iverilog_available:
            print(f"  → Attempting iverilog compilation...")
            
            with tempfile.NamedTemporaryFile(suffix='.vvp', delete=False) as tmp:
                vvp_path = Path(tmp.name)
            
            if self.vvp_extractor.compile_to_vvp(rtl_files, vvp_path):
                print(f"  ✓ Compilation successful, extracting from VVP")
                conn_graph = self.vvp_extractor.extract_graph_from_vvp(vvp_path)
                mode = 'vvp'
            else:
                print(f"  ✗ Compilation failed")
            
            # Cleanup
            try:
                vvp_path.unlink()
            except:
                pass
        
        # Fallback to regex
        if conn_graph is None or len(conn_graph.nodes) == 0:
            if self.config.FALLBACK_TO_REGEX:
                print(f"  → Using regex-based extraction")
                conn_graph = self.regex_extractor.build_lightweight_graph(rtl_code)
                mode = 'regex'
            else:
                return None
        
        if conn_graph is None or len(conn_graph.nodes) == 0:
            return None
        
        print(f"  → Built graph: {len(conn_graph.nodes)} nodes, {len(conn_graph.edges)} edges")
        
        # Sample if too large
        if len(conn_graph.nodes) > self.config.MAX_NODES_FULL_ANALYSIS:
            conn_graph = self.graph_analyzer.sample_graph(conn_graph, 
                                                         self.config.SAMPLE_RATIO)
            mode += '_sampled'
        
        # Extract features
        graph_features = self.graph_analyzer.compute_graph_features(conn_graph)
        gate_counts = self.regex_extractor.extract_gate_counts(rtl_code)
        
        # Record analysis mode
        self.analysis_modes.append(AnalysisMode(
            project=project_name,
            mode=mode,
            success=True,
            nodes=len(conn_graph.nodes),
            edges=len(conn_graph.edges)
        ))
        
        return {
            'conn_graph': conn_graph,
            'graph_features': graph_features,
            'gate_counts': gate_counts,
            'mode': mode
        }
    
    def discover_projects(self) -> List[Path]:
        """Auto-discover projects."""
        if not self.base_dir.exists():
            return []
        
        projects = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                has_rtl = any(item.rglob('*.v')) or any(item.rglob('*.sv'))
                if has_rtl:
                    projects.append(item)
        
        return projects
    
    def analyze_projects(self, project_names: List[str] = None):
        """Analyze all projects."""
        
        if project_names is None:
            project_dirs = self.discover_projects()
        else:
            project_dirs = [self.base_dir / name for name in project_names 
                          if (self.base_dir / name).exists()]
        
        if not project_dirs:
            print("No projects found!")
            return
        
        print(f"\n{'='*80}")
        print(f"SCALABLE COMPONENT REUSABILITY ANALYSIS")
        print(f"{'='*80}")
        print(f"Config: Max nodes for full analysis = {self.config.MAX_NODES_FULL_ANALYSIS}")
        print(f"        Sample ratio for large graphs = {self.config.SAMPLE_RATIO}")
        print(f"        VVP mode = {self.config.USE_VVP_IF_AVAILABLE}")
        print(f"        iverilog available = {self.vvp_extractor.iverilog_available}")
        print(f"{'='*80}\n")
        
        for project_dir in project_dirs:
            project_name = project_dir.name
            print(f"Analyzing: {project_name}")
            
            result = self.analyze_project_hybrid(project_dir, project_name)
            
            if result:
                self.projects[project_name] = result
                print(f"  ✓ Success (mode: {result['mode']})\n")
            else:
                print(f"  ✗ Failed\n")
    
    def compute_reusability_matrix(self) -> pd.DataFrame:
        """Compute pairwise reusability."""
        
        if not self.projects:
            return pd.DataFrame()
        
        project_names = list(self.projects.keys())
        n = len(project_names)
        reusability = np.zeros((n, n))
        
        print(f"\n{'='*80}")
        print("COMPUTING REUSABILITY SCORES")
        print(f"{'='*80}\n")
        
        for i, proj1 in enumerate(project_names):
            for j, proj2 in enumerate(project_names):
                if i == j:
                    reusability[i, j] = 100.0
                elif i < j:
                    print(f"Comparing {proj1} ↔ {proj2}...")
                    
                    G1 = self.projects[proj1]['conn_graph']
                    G2 = self.projects[proj2]['conn_graph']
                    
                    # Fast structural similarity
                    score = self.graph_analyzer.compute_structural_similarity(G1, G2)
                    
                    reusability[i, j] = score
                    reusability[j, i] = score
                    
                    print(f"  → Score: {score:.1f}%\n")
        
        return pd.DataFrame(reusability, index=project_names, columns=project_names)
    
    def print_results(self):
        """Print results."""
        
        if not self.projects:
            print("\n❌ No projects analyzed")
            return
        
        # Analysis modes summary
        print(f"\n{'='*80}")
        print("ANALYSIS MODES USED")
        print(f"{'='*80}\n")
        
        for mode_info in self.analysis_modes:
            print(f"{mode_info.project}:")
            print(f"  Mode: {mode_info.mode}")
            print(f"  Nodes: {mode_info.nodes}, Edges: {mode_info.edges}\n")
        
        # Compute reusability
        reuse_matrix = self.compute_reusability_matrix()
        
        print(f"\n{'='*80}")
        print("REUSABILITY MATRIX")
        print(f"{'='*80}\n")
        print(reuse_matrix.round(1).to_string())
    
    def export_results(self, output_dir: str = "./output"):
        """Export results."""
        
        if not self.projects:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        reuse_matrix = self.compute_reusability_matrix()
        reuse_matrix.to_csv(output_path / "reusability_matrix.csv")
        
        # Export analysis modes
        modes_data = []
        for mode_info in self.analysis_modes:
            modes_data.append({
                'project': mode_info.project,
                'mode': mode_info.mode,
                'nodes': mode_info.nodes,
                'edges': mode_info.edges
            })
        
        pd.DataFrame(modes_data).to_csv(output_path / "analysis_modes.csv", index=False)
        
        print(f"\n✓ Results exported to {output_path}/\n")

def main():
    """Main execution."""
    
    # Configuration
    BASE_DIRECTORY = "./verilog_proj"
    
    PROJECT_NAMES = [
        "CNN-Accelerator-Based-on-Eyeriss-v2",
        "NeuroSpector",
        "Tiny_LeViT_Hardware_Accelerator",
        "rt",
        "tiny-tpu"
    ]
    
    # Configure analysis
    config = AnalysisConfig()
    config.MAX_NODES_FULL_ANALYSIS = 5000  # Adjust based on your RAM
    config.SAMPLE_RATIO = 0.2  # Sample 20% for huge graphs
    config.USE_VVP_IF_AVAILABLE = True  # Try iverilog
    config.FALLBACK_TO_REGEX = True  # Fallback if VVP fails
    
    # Run analysis
    analyzer = ScalableReusabilityAnalyzer(BASE_DIRECTORY, config)
    analyzer.analyze_projects(PROJECT_NAMES)
    
    if analyzer.projects:
        analyzer.print_results()
        analyzer.export_results()
    
    print("\n✓ Analysis complete\n")

if __name__ == "__main__":
    main()
