"""Synthetic game generator (ground truth + CV observation stream)."""
from .game import SimNoise, SimResult, default_zones, simulate_game
from .demo_data import DemoSetup, demo_setup, load_demo

__all__ = ["SimNoise", "SimResult", "default_zones", "simulate_game", "DemoSetup", "demo_setup", "load_demo"]
