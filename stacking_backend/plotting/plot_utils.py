import numpy as np
import matplotlib.pyplot as plt

class PlotUtils:
    """Common plotting utilities and formatting"""
    
    @staticmethod
    def calculate_color_limits(data, percentile_range=(5, 95)):
        """Calculate color limits for image display"""
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            finite_data = data[finite_mask]
            vmin, vmax = np.percentile(finite_data, percentile_range)
        else:
            vmin, vmax = -1e-6, 1e-6
        return vmin, vmax
    
    @staticmethod
    def add_circle_apertures(ax, center=(0, 0), inner_radius=None, outer_radius=None):
        """Add aperture circles to plot"""
        if inner_radius is not None:
            circle_inner = plt.Circle(center, inner_radius, fill=False,
                                     color='white', linewidth=2, linestyle='-', 
                                     label='Inner aperture')
            ax.add_patch(circle_inner)
        
        if outer_radius is not None:
            circle_outer = plt.Circle(center, outer_radius, fill=False,
                                     color='white', linewidth=2, linestyle='--', 
                                     label='Outer aperture')
            ax.add_patch(circle_outer)
    
    @staticmethod
    def format_patch_plot(ax, patch_size_deg, title=None, show_grid=True):
        """Standard formatting for patch plots"""
        extent = [-patch_size_deg/2, patch_size_deg/2, -patch_size_deg/2, patch_size_deg/2]
        
        ax.set_xlabel('ΔRA [degrees]', fontsize=12)
        ax.set_ylabel('ΔDec [degrees]', fontsize=12)
        if title:
            ax.set_title(title, fontsize=13)
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        return extent
    
    @staticmethod
    def add_statistics_text(ax, data, position=(0.02, 0.98)):
        """Add statistics text box to plot"""
        finite_mask = np.isfinite(data)
        if np.any(finite_mask):
            finite_data = data[finite_mask]
            stats_text = f'Statistics:\n'
            stats_text += f'Mean: {np.mean(finite_data):.2e}\n'
            stats_text += f'Std: {np.std(finite_data):.2e}\n'
            stats_text += f'Min: {np.min(finite_data):.2e}\n'
            stats_text += f'Max: {np.max(finite_data):.2e}\n'
            stats_text += f'Valid pixels: {len(finite_data)}/{len(data.flatten())}'
            
            ax.text(position[0], position[1], stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
