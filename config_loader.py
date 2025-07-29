# config_loader.py
import toml
import os
from dataclasses import dataclass
from typing import List

@dataclass
class GlobalConfig:
    basedir: str
    observer_coords: List[float]
    output_dir: str

@dataclass 
class Mode1Config:
    mcmc_start: int
    mcmc_end: int
    mass_cut: float
    radius_cut: float
    eps: float
    min_samples: int

@dataclass
class Mode2Config:
    target_snapshot: int
    min_cluster_size: int

@dataclass
class Mode3Config:
    radius_inner: float
    radius_outer: float
    figure_dpi: int

@dataclass
class Config:
    global_config: GlobalConfig
    mode1: Mode1Config
    mode2: Mode2Config
    mode3: Mode3Config

def load_config(config_path: str = "config.toml") -> Config:
    with open(config_path, 'r') as f:
        data = toml.load(f)
    
    global_config = GlobalConfig(
        basedir=str(data['global']['basedir']),
        observer_coords=[float(x) for x in data['global']['observer_coords']],
        output_dir=str(data['global']['output_dir'])
    )
    
    mode1_config = Mode1Config(
        mcmc_start=int(data['mode1']['mcmc_start']),
        mcmc_end=int(data['mode1']['mcmc_end']),
        mass_cut=float(data['mode1']['mass_cut']),
        radius_cut=float(data['mode1']['radius_cut']),
        eps=float(data['mode1']['eps']),
        min_samples=int(data['mode1']['min_samples'])
    )
    
    mode2_config = Mode2Config(
        target_snapshot=int(data['mode2']['target_snapshot']),
        min_cluster_size=int(data['mode2']['min_cluster_size'])
    )
    
    mode3_config = Mode3Config(
        radius_inner=float(data['mode3']['radius_inner']),
        radius_outer=float(data['mode3']['radius_outer']),
        figure_dpi=int(data['mode3']['figure_dpi'])
    )
    
    return Config(
        global_config=global_config,
        mode1=mode1_config,
        mode2=mode2_config,
        mode3=mode3_config
    )
