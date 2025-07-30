import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from .plot_utils import PlotUtils

class BasicPlotter:
    """Basic plotting functionality for patches and profiles"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def plot_patch(self, ra, dec, patch_size_deg=10.0, npix=256, 
                   title=None, cmap='RdBu_r', percentile_range=(5, 95),
                   show_center=True, show_grid=True):
        """Plot y-map patch and mask side by side"""
        
        # Convert RA/Dec to Galactic coordinates
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        gal_coord = coord.galactic
        gal_lon = gal_coord.l.deg
        gal_lat = gal_coord.b.deg
        
        # Extract patch
        patch_data, mask_patch = self.patch_extractor.extract_patch(
            center_coords=(gal_lon, gal_lat),
            patch_size_deg=patch_size_deg,
            npix=npix
        )
        
        # Create masked version for display
        y_display = patch_data.copy()
        if mask_patch is not None:
            y_display[~mask_patch] = np.nan
        
        # Calculate color limits
        vmin, vmax = PlotUtils.calculate_color_limits(y_display, percentile_range)
        
        # Create dual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        
        # Plot extent
        extent = PlotUtils.format_patch_plot(ax1, patch_size_deg, 'Y-Parameter Map', show_grid)
        PlotUtils.format_patch_plot(ax2, patch_size_deg, 
                                   f'Mask ({np.mean(mask_patch)*100:.1f}% valid)' if mask_patch is not None else 'Mask (100% valid)', 
                                   show_grid)
        
        # Left plot: Y-map
        im1 = ax1.imshow(y_display, extent=extent, origin='lower',
                        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        
        if show_center:
            ax1.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
        
        # Right plot: Mask
        if mask_patch is not None:
            im2 = ax2.imshow(mask_patch.astype(float), extent=extent, origin='lower',
                            cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
        else:
            im2 = ax2.imshow(np.ones_like(patch_data), extent=extent, origin='lower',
                            cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
        
        if show_center:
            ax2.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
        
        # Colorbars
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Y-parameter')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Mask (1=valid, 0=masked)')
        
        # Overall title
        if title is None:
            title = f'RA={ra:.3f}°, Dec={dec:.3f}°'
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2), patch_data, mask_patch
    
    def plot_stacked_patch(self, stacked_patch, patch_size_deg, title=None,
                          cmap='RdBu_r', percentile_range=(5, 95),
                          show_apertures=True, inner_radius_deg=None, outer_radius_deg=None):
        """Plot a pre-computed stacked patch"""
        
        if stacked_patch is None:
            print("❌ No stacked patch data provided!")
            return None, None
        
        npix = stacked_patch.shape[0]
        
        # Calculate color limits
        vmin, vmax = PlotUtils.calculate_color_limits(stacked_patch, percentile_range)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        
        # Format plot
        extent = PlotUtils.format_patch_plot(ax, patch_size_deg, 
                                           title or f'Stacked Y-Map ({npix}×{npix} pixels)')
        
        # Main image
        im = ax.imshow(stacked_patch, extent=extent, origin='lower',
                      cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Center marker
        ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, label='Center')
        
        # Show apertures if requested
        if show_apertures:
            PlotUtils.add_circle_apertures(ax, (0, 0), inner_radius_deg, outer_radius_deg)
        
        ax.legend(loc='upper right', fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Y-parameter', fontsize=11)
        
        # Statistics text
        PlotUtils.add_statistics_text(ax, stacked_patch)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_radial_profile(self, radii, profile, errors=None, title=None,
                           ylabel='Y-parameter', show_zero_line=True,
                           color='blue', marker='o'):
        """Plot a radial profile"""
        
        if radii is None or profile is None:
            print("❌ No profile data provided!")
            return None, None
        
        # Filter out NaN values
        finite_mask = np.isfinite(profile) & np.isfinite(radii)
        if errors is not None:
            finite_mask = finite_mask & np.isfinite(errors)
        
        if not np.any(finite_mask):
            print("❌ No valid profile data!")
            return None, None
        
        radii_plot = radii[finite_mask]
        profile_plot = profile[finite_mask]
        errors_plot = errors[finite_mask] if errors is not None else None
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Plot profile
        if errors_plot is not None:
            ax.errorbar(radii_plot, profile_plot, yerr=errors_plot,
                       fmt=marker, color=color, markersize=6, linewidth=2,
                       capsize=3, label='Radial profile')
        else:
            ax.plot(radii_plot, profile_plot, marker=marker, color=color,
                   markersize=6, linewidth=2, label='Radial profile')
        
        # Zero line
        if show_zero_line:
            ax.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Labels and formatting
        ax.set_xlabel('Radius [degrees]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title or 'Radial Profile', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add statistics text
        if len(profile_plot) > 0:
            central_value = profile_plot[0]
            stats_text = f'Central value: {central_value:.2e}\nProfile points: {len(profile_plot)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
