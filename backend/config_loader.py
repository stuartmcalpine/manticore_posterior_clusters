import toml
import os
from dataclasses import dataclass
from typing import List

@dataclass
class GlobalConfig:
    basedir: str
    observer_coords: List[float]
    output_dir: str
    boxsize: float

@dataclass 
class Mode1Config:
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    eps: float
    min_samples: int

@dataclass
class Mode2Config:
    target_snapshot: int
    min_cluster_size: int

@dataclass
class Mode3Config:
    basedir: str
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    num_samplings: int
    eps: float
    min_samples: int

@dataclass
class Mode4Config:
    basedir: str
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    target_snapshot: int
    observer_coords: List[float]

@dataclass
class Config:
    global_config: GlobalConfig
    mode1: Mode1Config
    mode2: Mode2Config
    mode3: Mode3Config
    mode4: Mode4Config

def load_config(config_path: str = "config.toml") -> Config:
    with open(config_path, 'r') as f:
        data = toml.load(f)
    
    global_config = GlobalConfig(
        basedir=str(data['global']['basedir']),
        observer_coords=[float(x) for x in data['global']['observer_coords']],
        output_dir=str(data['global']['output_dir']),
        boxsize=float(data['global']['boxsize'])
    )
    
    mode1_config = Mode1Config(
        mcmc_start=int(data['mode1']['mcmc_start']),
        mcmc_end=int(data['mode1']['mcmc_end']),
        m200_mass_cut=float(data['mode1']['m200_mass_cut']),
        radius_cut=float(data['mode1']['radius_cut']),
        eps=float(data['mode1']['eps']),
        min_samples=int(data['mode1']['min_samples'])
    )
    
    mode2_config = Mode2Config(
        target_snapshot=int(data['mode2']['target_snapshot']),
        min_cluster_size=int(data['mode2']['min_cluster_size'])
    )
    
    mode3_config = Mode3Config(
        basedir=str(data['mode3']['basedir']),
        mcmc_start=int(data['mode3']['mcmc_start']),
        mcmc_end=int(data['mode3']['mcmc_end']),
        m200_mass_cut=float(data['mode3']['m200_mass_cut']),
        radius_cut=float(data['mode3']['radius_cut']),
        num_samplings=int(data['mode3']['num_samplings']),
        eps=float(data['mode3']['eps']),
        min_samples=int(data['mode3']['min_samples'])
    )
    
    mode4_config = Mode4Config(
        basedir=str(data['mode4']['basedir']),
        mcmc_start=int(data['mode4']['mcmc_start']),
        mcmc_end=int(data['mode4']['mcmc_end']),
        m200_mass_cut=float(data['mode4']['m200_mass_cut']),
        radius_cut=float(data['mode4']['radius_cut']),
        target_snapshot=int(data['mode4']['target_snapshot']),
        observer_coords=[float(x) for x in data['mode4']['observer_coords']]
    )
    
    return Config(
        global_config=global_config,
        mode1=mode1_config,
        mode2=mode2_config,
        mode3=mode3_config,
        mode4=mode4_config
    )
