"""
BRITE configuration generator

Generates deterministic BRITE configuration files from YAML templates.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class BRITEConfigGenerator:
    """Generate BRITE configuration files from templates"""
    
    DEFAULT_CONFIG = {
        'topology_type': 'AS',
        'n_nodes': 100,
        'geo_bounds': [1000, 1000],
        'model': 'Barabasi-Albert',
        'degree_alpha': 2.2,
        'bw_dist': 'Uniform',
        'bw_min': 1,
        'bw_max': 100,
        'growth_type': 'Incremental',
        'pref_conn': 'On',
        'seed': 42
    }
    
    def __init__(self, template_path: Path = None):
        """
        Args:
            template_path: Path to YAML template file
        """
        self.template_path = template_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        if template_path and template_path.exists():
            with open(template_path) as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
    
    def generate(self, output_path: Path, **kwargs) -> Path:
        """
        Generate BRITE configuration file
        
        Args:
            output_path: Where to write the .conf file
            **kwargs: Override template parameters
            
        Returns:
            Path to generated config file
        """
        # Override with any provided parameters
        config = self.config.copy()
        config.update(kwargs)
        
        # Generate BRITE config format
        conf_content = self._format_brite_config(config)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(conf_content)
            
        return output_path
    
    def _format_brite_config(self, config: Dict[str, Any]) -> str:
        """Format configuration as BRITE .conf file"""
        lines = []
        
        # Topology section
        lines.append(f"BriteConfig")
        lines.append("")
        lines.append(f"BeginModel")
        lines.append(f"\tName = {config['topology_type']}_Topology")
        lines.append(f"\tn = {config['n_nodes']}")
        lines.append(f"\tk = -1")  # Not used in BA model
        lines.append(f"\tLS = {config['geo_bounds'][0]}")
        lines.append(f"\tHS = {config['geo_bounds'][1]}")
        lines.append(f"\tNodePlacement = Random")
        lines.append(f"\tm = 2")  # New links per node
        lines.append(f"\tGrowthType = {config['growth_type']}")
        lines.append(f"\tPreferentialConnectivity = {config['pref_conn']}")
        lines.append(f"\talpha = {config['degree_alpha']}")
        lines.append(f"\tbeta = 0.2")  # Distance factor
        lines.append(f"\tgamma = -1")  # Not used
        lines.append(f"\tBWDist = {config['bw_dist']}")
        lines.append(f"\tBWMin = {config['bw_min']}")
        lines.append(f"\tBWMax = {config['bw_max']}")
        lines.append(f"EndModel")
        lines.append("")
        lines.append(f"BeginOutput")
        lines.append(f"\tOutputType = BRITE")
        lines.append(f"EndOutput")
        
        return '\n'.join(lines)
    
    def generate_batch(self, base_path: Path, configurations: list) -> list:
        """
        Generate multiple configuration files
        
        Args:
            base_path: Base directory for output
            configurations: List of configuration dictionaries
            
        Returns:
            List of paths to generated config files
        """
        paths = []
        for i, config in enumerate(configurations):
            output_path = base_path / f"config_{i}.conf"
            path = self.generate(output_path, **config)
            paths.append(path)
        return paths


# Reuse the existing BRITEWrapper functionality
try:
    from src.topology.brite_wrapper import BRITEWrapper
    
    class BRITERunner(BRITEWrapper):
        """Extended BRITE runner with parallel execution support"""
        
        def run_parallel(self, config_files: list, output_dir: Path, n_jobs: int = -1):
            """
            Run multiple BRITE instances in parallel
            
            Args:
                config_files: List of configuration file paths
                output_dir: Directory for output files
                n_jobs: Number of parallel jobs (-1 for all CPUs)
            """
            from joblib import Parallel, delayed
            import multiprocessing
            
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            def run_single(config_path, output_dir):
                output_name = config_path.stem
                return self.generate_topology(
                    n_nodes=None,  # Read from config
                    model_type=None,  # Read from config
                    output_dir=output_dir,
                    output_name=output_name,
                    config_file=str(config_path)
                )
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single)(cfg, output_dir) 
                for cfg in config_files
            )
            
            return results
            
except ImportError:
    # Fallback implementation without dependency on src/
    import subprocess
    
    class BRITERunner:
        """Minimal BRITE runner implementation"""
        
        def __init__(self, brite_path: Path = None):
            self.brite_path = brite_path or Path("external/brite")
            
        def run_parallel(self, config_files: list, output_dir: Path, n_jobs: int = -1):
            """Run BRITE with GNU Parallel"""
            import tempfile
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a list of commands
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for cfg in config_files:
                    cmd = f"cd {self.brite_path} && java -cp Java/:. Main.Brite {cfg} {output_dir}/{cfg.stem}"
                    f.write(cmd + '\n')
                cmd_file = f.name
            
            # Run with GNU parallel
            subprocess.run([
                'parallel', '-j', str(n_jobs), '--progress'
            ], stdin=open(cmd_file), check=True)
            
            # Clean up
            Path(cmd_file).unlink()
            
            # Return output files
            return list(output_dir.glob("*.brite"))