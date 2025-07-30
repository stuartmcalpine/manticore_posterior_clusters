#!/usr/bin/env python3
"""
PR4 Y-Map Plotter - Simple dual plot (y-map + mask)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

class PR4YMapPlotter:
    """Simple plotter for PR4 NILC y-map patches"""
    
    def __init__(self, analyzer=None):
        """Initialize plotter with PR4YMapAnalyzer object"""
        self.analyzer = analyzer
    
    def plot_patch(self, ra, dec, patch_size_deg=10.0, npix=256, 
                   title=None, cmap='RdBu_r', percentile_range=(5, 95),
                   show_center=True, show_grid=True):
        """Plot y-map patch and mask side by side"""
        
        if self.analyzer is None:
            raise ValueError("No analyzer provided! Pass PR4YMapAnalyzer to constructor.")
        
        # Convert RA/Dec to Galactic coordinates
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        gal_coord = coord.galactic
        gal_lon = gal_coord.l.deg
        gal_lat = gal_coord.b.deg
        
        # Extract patch using the analyzer
        patch_data, mask_patch = self.analyzer.extract_patch(
            center_coords=(gal_lon, gal_lat),
            patch_size_deg=patch_size_deg,
            npix=npix
        )
        
        # Create masked version for y-map display
        y_display = patch_data.copy()
        if mask_patch is not None:
            y_display[~mask_patch] = np.nan
        
        # Calculate color limits for y-map
        finite_mask = np.isfinite(y_display)
        if np.any(finite_mask):
            finite_data = y_display[finite_mask]
            vmin, vmax = np.percentile(finite_data, percentile_range)
        else:
            vmin, vmax = -1e-6, 1e-6
        
        # Create dual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        
        # Plot extent
        extent = [-patch_size_deg/2, patch_size_deg/2, 
                 -patch_size_deg/2, patch_size_deg/2]
        
        # Left plot: Y-map
        im1 = ax1.imshow(y_display, extent=extent, origin='lower',
                        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        
        if show_center:
            ax1.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
        ax1.set_xlabel('ΔRA [degrees]')
        ax1.set_ylabel('ΔDec [degrees]')
        ax1.set_title('Y-Parameter Map')
        if show_grid:
            ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal')
        
        # Colorbar for y-map
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Y-parameter')
        
        # Right plot: Mask
        if mask_patch is not None:
            im2 = ax2.imshow(mask_patch.astype(float), extent=extent, origin='lower',
                            cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            mask_coverage = np.mean(mask_patch) * 100
            mask_title = f'Mask ({mask_coverage:.1f}% valid)'
        else:
            # No mask - show all ones
            im2 = ax2.imshow(np.ones_like(patch_data), extent=extent, origin='lower',
                            cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            mask_title = 'Mask (100% valid)'
        
        if show_center:
            ax2.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
        ax2.set_xlabel('ΔRA [degrees]')
        ax2.set_ylabel('ΔDec [degrees]')
        ax2.set_title(mask_title)
        if show_grid:
            ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal')
        
        # Colorbar for mask
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
        finite_mask = np.isfinite(stacked_patch)
        if np.any(finite_mask):
            finite_data = stacked_patch[finite_mask]
            vmin, vmax = np.percentile(finite_data, percentile_range)
        else:
            vmin, vmax = -1e-6, 1e-6

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        # Plot extent
        extent = [-patch_size_deg/2, patch_size_deg/2,
                 -patch_size_deg/2, patch_size_deg/2]

        # Main image
        im = ax.imshow(stacked_patch, extent=extent, origin='lower',
                      cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        # Center marker
        ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, label='Center')

        # Show apertures if requested
        if show_apertures and inner_radius_deg is not None and outer_radius_deg is not None:
            circle_inner = plt.Circle((0, 0), inner_radius_deg, fill=False,
                                     color='white', linewidth=2, linestyle='-', label='Inner aperture')
            circle_outer = plt.Circle((0, 0), outer_radius_deg, fill=False,
                                     color='white', linewidth=2, linestyle='--', label='Outer aperture')
            ax.add_patch(circle_inner)
            ax.add_patch(circle_outer)

        # Labels and formatting
        ax.set_xlabel('ΔRA [degrees]', fontsize=12)
        ax.set_ylabel('ΔDec [degrees]', fontsize=12)

        if title is None:
            title = f'Stacked Y-Map ({npix}×{npix} pixels)'
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Y-parameter', fontsize=11)

        # Statistics text
        if np.any(finite_mask):
            finite_data = stacked_patch[finite_mask]
            stats_text = f'Stacked Statistics:\n'
            stats_text += f'Mean: {np.mean(finite_data):.2e}\n'
            stats_text += f'Std: {np.std(finite_data):.2e}\n'
            stats_text += f'Min: {np.min(finite_data):.2e}\n'
            stats_text += f'Max: {np.max(finite_data):.2e}\n'
            stats_text += f'Valid pixels: {len(finite_data)}/{npix**2}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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

        if title is None:
            title = 'Radial Profile'
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add some statistics text
        if len(profile_plot) > 0:
            central_value = profile_plot[0] if len(profile_plot) > 0 else np.nan
            stats_text = f'Central value: {central_value:.2e}\n'
            stats_text += f'Profile points: {len(profile_plot)}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_analysis_summary(self, results, title=None, cmap='RdBu_r'):
        """Multi-panel summary plot of cluster analysis results"""

        if not results.get('success', False):
            print("❌ Analysis results indicate failure!")
            return None, None

        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data from results
        stacked_patch = results.get('stacked_patch')
        patch_size_deg = results.get('patch_size_deg', 15.0)
        radii = results.get('profile_radii')
        profile = results.get('profile_mean')
        profile_errors = results.get('profile_errors')
        inner_radius = results.get('inner_radius_deg')
        outer_radius = results.get('outer_radius_deg')

        # 1. Stacked patch (top-left)
        ax = axes[0, 0]
        if stacked_patch is not None:
            npix = stacked_patch.shape[0]
            extent = [-patch_size_deg/2, patch_size_deg/2,
                     -patch_size_deg/2, patch_size_deg/2]

            # Calculate color limits
            finite_mask = np.isfinite(stacked_patch)
            if np.any(finite_mask):
                finite_data = stacked_patch[finite_mask]
                vmin, vmax = np.percentile(finite_data, [5, 95])
            else:
                vmin, vmax = -1e-6, 1e-6

            im = ax.imshow(stacked_patch, extent=extent, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

            # Center and apertures
            ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
            if inner_radius and outer_radius:
                circle_inner = plt.Circle((0, 0), inner_radius, fill=False,
                                         color='white', linewidth=2, linestyle='-')
                circle_outer = plt.Circle((0, 0), outer_radius, fill=False,
                                         color='white', linewidth=2, linestyle='--')
                ax.add_patch(circle_inner)
                ax.add_patch(circle_outer)

            plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xlabel('ΔRA [degrees]')
        ax.set_ylabel('ΔDec [degrees]')
        ax.set_title('Stacked Y-Map')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 2. Radial profile (top-right)
        ax = axes[0, 1]
        if radii is not None and profile is not None:
            finite_mask = np.isfinite(profile) & np.isfinite(radii)
            if profile_errors is not None:
                finite_mask = finite_mask & np.isfinite(profile_errors)

            if np.any(finite_mask):
                radii_plot = radii[finite_mask]
                profile_plot = profile[finite_mask]
                errors_plot = profile_errors[finite_mask] if profile_errors is not None else None

                if errors_plot is not None:
                    ax.errorbar(radii_plot, profile_plot, yerr=errors_plot,
                               fmt='o-', color='blue', markersize=4, linewidth=2, capsize=3)
                else:
                    ax.plot(radii_plot, profile_plot, 'o-', color='blue',
                           markersize=4, linewidth=2)

        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Radius [degrees]')
        ax.set_ylabel('Y-parameter')
        ax.set_title('Radial Profile')
        ax.grid(True, alpha=0.3)

        # 3. Key measurements (bottom-left)
        ax = axes[1, 0]
        ax.axis('off')

        # Compile key statistics
        measurements_text = f"""
    CLUSTER ANALYSIS RESULTS

    Signal Measurement:
    • Δy = {results.get('mean_delta_y', np.nan):.2e}
    • Error = {results.get('error_mean', np.nan):.2e}
    • Significance = {results.get('significance', np.nan):.1f}σ

    Aperture Details:
    • Inner radius = {results.get('inner_radius_deg', np.nan):.2f}°
    • Outer radius = {results.get('outer_radius_deg', np.nan):.2f}°
    • Inner mean = {results.get('inner_mean', np.nan):.2e}
    • Outer mean = {results.get('outer_mean', np.nan):.2e}

    Sample Information:
    • Valid patches = {results.get('n_measurements', 0)}
    • Input coordinates = {results.get('n_input_coords', 0)}
    • Rejection rate = {results.get('n_rejected', 0)}/{results.get('n_input_coords', 0)}
    • Patch size = {results.get('patch_size_deg', 15.0)}° × {results.get('patch_size_deg', 15.0)}°
    """

        ax.text(0.05, 0.95, measurements_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 4. Sample quality metrics (bottom-right)
        ax = axes[1, 1]

        # Bar chart of rejection reasons
        rejection_stats = results.get('rejection_stats', {})
        if rejection_stats:
            reasons = list(rejection_stats.keys())
            counts = list(rejection_stats.values())
            total_rejected = sum(counts)

            if total_rejected > 0:
                bars = ax.bar(range(len(reasons)), counts, alpha=0.7,
                             color=['red', 'orange', 'yellow'][:len(reasons)])
                ax.set_xticks(range(len(reasons)))
                ax.set_xticklabels([r.replace('_', ' ').title() for r in reasons], rotation=45)
                ax.set_ylabel('Number Rejected')
                ax.set_title('Rejection Statistics')

                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                               str(count), ha='center', va='bottom', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Rejections\n✅ Perfect Sample!',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                ax.set_xticks([])
                ax.set_yticks([])

        ax.grid(True, alpha=0.3)

        # Overall title
        if title is None:
            significance = results.get('significance', 0)
            n_patches = results.get('n_measurements', 0)
            title = f'Cluster Analysis Summary (σ={significance:.1f}, N={n_patches})'

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        return fig, axes

    def plot_mass_range_profiles(self, results_dict, title=None, colors=None, figsize=(12, 8)):
        """
        Plot radial profiles for different mass ranges on the same plot

        Parameters:
        -----------
        results_dict : dict
            Dictionary from run_mass_range_analysis()
        title : str, optional
            Plot title
        colors : list, optional
            Colors for each mass bin
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Set up colors
        if colors is None:
            colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot each mass range
        for i, (label, results) in enumerate(results_dict.items()):
            if not results.get('success', False):
                continue

            radii = results.get('profile_radii')
            profile = results.get('profile_mean')
            errors = results.get('profile_errors')

            if radii is None or profile is None:
                continue

            # Filter out NaN values
            finite_mask = np.isfinite(profile) & np.isfinite(radii)
            if errors is not None:
                finite_mask = finite_mask & np.isfinite(errors)

            if not np.any(finite_mask):
                continue

            radii_plot = radii[finite_mask]
            profile_plot = profile[finite_mask]
            errors_plot = errors[finite_mask] if errors is not None else None

            # Get color and create label
            color = colors[i % len(colors)]
            n_clusters = results['n_measurements']
            significance = results['significance']

            if label == 'Random':
                plot_label = f"{label} (N={n_clusters}, σ={significance:.1f})"
                linestyle = '--'
                alpha = 0.7
            else:
                plot_label = f"{label} M☉ (N={n_clusters}, σ={significance:.1f})"
                linestyle = '-'
                alpha = 1.0

            # Plot with error bars
            if errors_plot is not None:
                ax.errorbar(radii_plot, profile_plot, yerr=errors_plot,
                           fmt='o-', color=color, markersize=4, linewidth=2,
                           capsize=3, label=plot_label, linestyle=linestyle, alpha=alpha)
            else:
                ax.plot(radii_plot, profile_plot, 'o-', color=color,
                       markersize=4, linewidth=2, label=plot_label,
                       linestyle=linestyle, alpha=alpha)

        # Formatting
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.set_xlabel('Radius [degrees]', fontsize=12)
        ax.set_ylabel('Y-parameter', fontsize=12)

        if title is None:
            title = 'Cluster Radial Profiles by Mass Range'
        ax.set_title(title, fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        # Add summary statistics
        n_mass_bins = len([k for k in results_dict.keys() if k != 'Random'])
        stats_text = f'Mass bins analyzed: {n_mass_bins}'
        if 'Random' in results_dict:
            stats_text += f'\nRandom control included'

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_mass_scaling(self, results_dict, title=None, figsize=(10, 7)):
        """
        Plot significance and signal strength vs mass

        Parameters:
        -----------
        results_dict : dict
            Dictionary from run_mass_range_analysis()
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Extract data (exclude random)
        mass_centers = []
        significances = []
        delta_ys = []
        delta_y_errors = []
        n_clusters = []
        labels = []

        for label, results in results_dict.items():
            if label == 'Random' or not results.get('success', False):
                continue

            mass_centers.append(results['mass_center'])
            significances.append(results['significance'])
            delta_ys.append(results['mean_delta_y'])
            delta_y_errors.append(results['error_mean'])
            n_clusters.append(results['n_measurements'])
            labels.append(label)

        if not mass_centers:
            print("❌ No valid mass data to plot!")
            return None, None

        # Convert to arrays
        mass_centers = np.array(mass_centers)
        significances = np.array(significances)
        delta_ys = np.array(delta_ys)
        delta_y_errors = np.array(delta_y_errors)
        n_clusters = np.array(n_clusters)

        # Create dual-axis plot
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Plot significance
        color1 = 'tab:blue'
        ax1.set_xlabel('Cluster Mass [M☉]', fontsize=12)
        ax1.set_ylabel('Detection Significance [σ]', color=color1, fontsize=12)
        ax1.errorbar(mass_centers, significances, fmt='o-', color=color1,
                    markersize=8, linewidth=2, capsize=3, label='Significance')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # Add significance threshold lines
        ax1.axhline(3, color='red', linestyle='--', alpha=0.5, label='3σ threshold')
        ax1.axhline(5, color='red', linestyle=':', alpha=0.5, label='5σ threshold')

        # Plot Δy on second axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Δy Signal Strength', color=color2, fontsize=12)
        ax2.errorbar(mass_centers, delta_ys, yerr=delta_y_errors, fmt='s-', color=color2,
                    markersize=6, linewidth=2, capsize=3, alpha=0.7, label='Δy')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add sample size annotations
        for i, (mass, sig, n) in enumerate(zip(mass_centers, significances, n_clusters)):
            ax1.annotate(f'N={n}', (mass, sig), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, alpha=0.7)

        # Formatting
        if title is None:
            title = 'Cluster Detection vs Mass'
        fig.suptitle(title, fontsize=14)

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.show()

        return fig, (ax1, ax2)

    def plot_mass_summary_grid(self, results_dict, figsize=(16, 12)):
        """
        Create a comprehensive summary plot with multiple panels
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Radial profiles (top-left)
        ax = axes[0, 0]
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'gray']

        for i, (label, results) in enumerate(results_dict.items()):
            if not results.get('success', False):
                continue

            radii = results.get('profile_radii')
            profile = results.get('profile_mean')

            if radii is None or profile is None:
                continue

            finite_mask = np.isfinite(profile) & np.isfinite(radii)
            if not np.any(finite_mask):
                continue

            color = colors[i % len(colors)]
            linestyle = '--' if label == 'Random' else '-'
            alpha = 0.7 if label == 'Random' else 1.0

            ax.plot(radii[finite_mask], profile[finite_mask],
                   color=color, linestyle=linestyle, alpha=alpha,
                   linewidth=2, label=f"{label} ({results['significance']:.1f}σ)")

        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Radius [degrees]')
        ax.set_ylabel('Y-parameter')
        ax.set_title('Radial Profiles by Mass')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Mass scaling (top-right)
        ax = axes[0, 1]

        # Extract non-random data
        mass_data = [(label, results) for label, results in results_dict.items()
                     if label != 'Random' and results.get('success', False)]

        if mass_data:
            masses = [results['mass_center'] for _, results in mass_data]
            sigs = [results['significance'] for _, results in mass_data]
            ns = [results['n_measurements'] for _, results in mass_data]

            ax.scatter(masses, sigs, s=[n*3 for n in ns], alpha=0.7, c=range(len(masses)), cmap='viridis')
            ax.set_xscale('log')
            ax.axhline(3, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Mass [M☉]')
            ax.set_ylabel('Significance [σ]')
            ax.set_title('Detection vs Mass')
            ax.grid(True, alpha=0.3)

            # Add annotations
            for mass, sig, n, (label, _) in zip(masses, sigs, ns, mass_data):
                ax.annotate(f'N={n}', (mass, sig), xytext=(3, 3),
                           textcoords='offset points', fontsize=7)

        # 3. Sample sizes (bottom-left)
        ax = axes[1, 0]

        labels = []
        sample_sizes = []
        colors_bar = []

        for i, (label, results) in enumerate(results_dict.items()):
            if results.get('success', False):
                labels.append(label)
                sample_sizes.append(results['n_measurements'])
                colors_bar.append(colors[i % len(colors)])

        if labels:
            bars = ax.bar(labels, sample_sizes, color=colors_bar, alpha=0.7)
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Sample Sizes')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, size in zip(bars, sample_sizes):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       str(size), ha='center', va='bottom', fontsize=9)

        # 4. Significance summary (bottom-right)
        ax = axes[1, 1]

        if labels and len(labels) > 1:
            significances = [results['significance'] for label, results in results_dict.items()
                            if results.get('success', False)]

            bars = ax.bar(labels, significances, color=colors_bar, alpha=0.7)
            ax.axhline(3, color='red', linestyle='--', alpha=0.5, label='3σ')
            ax.axhline(5, color='red', linestyle=':', alpha=0.5, label='5σ')
            ax.set_ylabel('Significance [σ]')
            ax.set_title('Detection Significance')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

            # Add value labels
            for bar, sig in zip(bars, significances):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{sig:.1f}σ', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

        return fig, axes

#!/usr/bin/env python3
"""
PR4 Y-Map Plotter - Simple dual plot (y-map + mask)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

class PR4YMapPlotter:
    """Simple plotter for PR4 NILC y-map patches"""

    def __init__(self, analyzer=None):
        """Initialize plotter with PR4YMapAnalyzer object"""
        self.analyzer = analyzer

    def plot_patch(self, ra, dec, patch_size_deg=10.0, npix=256,
                   title=None, cmap='RdBu_r', percentile_range=(5, 95),
                   show_center=True, show_grid=True):
        """Plot y-map patch and mask side by side"""

        if self.analyzer is None:
            raise ValueError("No analyzer provided! Pass PR4YMapAnalyzer to constructor.")

        # Convert RA/Dec to Galactic coordinates
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        gal_coord = coord.galactic
        gal_lon = gal_coord.l.deg
        gal_lat = gal_coord.b.deg

        # Extract patch using the analyzer
        patch_data, mask_patch = self.analyzer.extract_patch(
            center_coords=(gal_lon, gal_lat),
            patch_size_deg=patch_size_deg,
            npix=npix
        )

        # Create masked version for y-map display
        y_display = patch_data.copy()
        if mask_patch is not None:
            y_display[~mask_patch] = np.nan

        # Calculate color limits for y-map
        finite_mask = np.isfinite(y_display)
        if np.any(finite_mask):
            finite_data = y_display[finite_mask]
            vmin, vmax = np.percentile(finite_data, percentile_range)
        else:
            vmin, vmax = -1e-6, 1e-6

        # Create dual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

        # Plot extent
        extent = [-patch_size_deg/2, patch_size_deg/2,
                 -patch_size_deg/2, patch_size_deg/2]

        # Left plot: Y-map
        im1 = ax1.imshow(y_display, extent=extent, origin='lower',
                        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        if show_center:
            ax1.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
        ax1.set_xlabel('ΔRA [degrees]')
        ax1.set_ylabel('ΔDec [degrees]')
        ax1.set_title('Y-Parameter Map')
        if show_grid:
            ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal')

        # Colorbar for y-map
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Y-parameter')

        # Right plot: Mask
        if mask_patch is not None:
            im2 = ax2.imshow(mask_patch.astype(float), extent=extent, origin='lower',
                            cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            mask_coverage = np.mean(mask_patch) * 100
            mask_title = f'Mask ({mask_coverage:.1f}% valid)'
        else:
            # No mask - show all ones
            im2 = ax2.imshow(np.ones_like(patch_data), extent=extent, origin='lower',
                            cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            mask_title = 'Mask (100% valid)'

        if show_center:
            ax2.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
        ax2.set_xlabel('ΔRA [degrees]')
        ax2.set_ylabel('ΔDec [degrees]')
        ax2.set_title(mask_title)
        if show_grid:
            ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal')

        # Colorbar for mask
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
        finite_mask = np.isfinite(stacked_patch)
        if np.any(finite_mask):
            finite_data = stacked_patch[finite_mask]
            vmin, vmax = np.percentile(finite_data, percentile_range)
        else:
            vmin, vmax = -1e-6, 1e-6

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        # Plot extent
        extent = [-patch_size_deg/2, patch_size_deg/2,
                 -patch_size_deg/2, patch_size_deg/2]

        # Main image
        im = ax.imshow(stacked_patch, extent=extent, origin='lower',
                      cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        # Center marker
        ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, label='Center')

        # Show apertures if requested
        if show_apertures and inner_radius_deg is not None and outer_radius_deg is not None:
            circle_inner = plt.Circle((0, 0), inner_radius_deg, fill=False,
                                     color='white', linewidth=2, linestyle='-', label='Inner aperture')
            circle_outer = plt.Circle((0, 0), outer_radius_deg, fill=False,
                                     color='white', linewidth=2, linestyle='--', label='Outer aperture')
            ax.add_patch(circle_inner)
            ax.add_patch(circle_outer)

        # Labels and formatting
        ax.set_xlabel('ΔRA [degrees]', fontsize=12)
        ax.set_ylabel('ΔDec [degrees]', fontsize=12)

        if title is None:
            title = f'Stacked Y-Map ({npix}×{npix} pixels)'
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Y-parameter', fontsize=11)

        # Statistics text
        if np.any(finite_mask):
            finite_data = stacked_patch[finite_mask]
            stats_text = f'Stacked Statistics:\n'
            stats_text += f'Mean: {np.mean(finite_data):.2e}\n'
            stats_text += f'Std: {np.std(finite_data):.2e}\n'
            stats_text += f'Min: {np.min(finite_data):.2e}\n'
            stats_text += f'Max: {np.max(finite_data):.2e}\n'
            stats_text += f'Valid pixels: {len(finite_data)}/{npix**2}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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

        if title is None:
            title = 'Radial Profile'
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add some statistics text
        if len(profile_plot) > 0:
            central_value = profile_plot[0] if len(profile_plot) > 0 else np.nan
            stats_text = f'Central value: {central_value:.2e}\n'
            stats_text += f'Profile points: {len(profile_plot)}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_analysis_summary(self, results, title=None, cmap='RdBu_r'):
        """Multi-panel summary plot of cluster analysis results"""

        if not results.get('success', False):
            print("❌ Analysis results indicate failure!")
            return None, None

        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data from results
        stacked_patch = results.get('stacked_patch')
        patch_size_deg = results.get('patch_size_deg', 15.0)
        radii = results.get('profile_radii')
        profile = results.get('profile_mean')
        profile_errors = results.get('profile_errors')
        inner_radius = results.get('inner_radius_deg')
        outer_radius = results.get('outer_radius_deg')

        # 1. Stacked patch (top-left)
        ax = axes[0, 0]
        if stacked_patch is not None:
            npix = stacked_patch.shape[0]
            extent = [-patch_size_deg/2, patch_size_deg/2,
                     -patch_size_deg/2, patch_size_deg/2]

            # Calculate color limits
            finite_mask = np.isfinite(stacked_patch)
            if np.any(finite_mask):
                finite_data = stacked_patch[finite_mask]
                vmin, vmax = np.percentile(finite_data, [5, 95])
            else:
                vmin, vmax = -1e-6, 1e-6

            im = ax.imshow(stacked_patch, extent=extent, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

            # Center and apertures
            ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
            if inner_radius and outer_radius:
                circle_inner = plt.Circle((0, 0), inner_radius, fill=False,
                                         color='white', linewidth=2, linestyle='-')
                circle_outer = plt.Circle((0, 0), outer_radius, fill=False,
                                         color='white', linewidth=2, linestyle='--')
                ax.add_patch(circle_inner)
                ax.add_patch(circle_outer)

            plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xlabel('ΔRA [degrees]')
        ax.set_ylabel('ΔDec [degrees]')
        ax.set_title('Stacked Y-Map')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 2. Radial profile (top-right)
        ax = axes[0, 1]
        if radii is not None and profile is not None:
            finite_mask = np.isfinite(profile) & np.isfinite(radii)
            if profile_errors is not None:
                finite_mask = finite_mask & np.isfinite(profile_errors)

            if np.any(finite_mask):
                radii_plot = radii[finite_mask]
                profile_plot = profile[finite_mask]
                errors_plot = profile_errors[finite_mask] if profile_errors is not None else None

                if errors_plot is not None:
                    ax.errorbar(radii_plot, profile_plot, yerr=errors_plot,
                               fmt='o-', color='blue', markersize=4, linewidth=2, capsize=3)
                else:
                    ax.plot(radii_plot, profile_plot, 'o-', color='blue',
                           markersize=4, linewidth=2)

        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Radius [degrees]')
        ax.set_ylabel('Y-parameter')
        ax.set_title('Radial Profile')
        ax.grid(True, alpha=0.3)

        # 3. Key measurements (bottom-left)
        ax = axes[1, 0]
        ax.axis('off')

        # Compile key statistics
        measurements_text = f"""
    CLUSTER ANALYSIS RESULTS

    Signal Measurement:
    • Δy = {results.get('mean_delta_y', np.nan):.2e}
    • Error = {results.get('error_mean', np.nan):.2e}
    • Significance = {results.get('significance', np.nan):.1f}σ

    Aperture Details:
    • Inner radius = {results.get('inner_radius_deg', np.nan):.2f}°
    • Outer radius = {results.get('outer_radius_deg', np.nan):.2f}°
    • Inner mean = {results.get('inner_mean', np.nan):.2e}
    • Outer mean = {results.get('outer_mean', np.nan):.2e}

    Sample Information:
    • Valid patches = {results.get('n_measurements', 0)}
    • Input coordinates = {results.get('n_input_coords', 0)}
    • Rejection rate = {results.get('n_rejected', 0)}/{results.get('n_input_coords', 0)}
    • Patch size = {results.get('patch_size_deg', 15.0)}° × {results.get('patch_size_deg', 15.0)}°
    """

        ax.text(0.05, 0.95, measurements_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 4. Sample quality metrics (bottom-right)
        ax = axes[1, 1]

        # Bar chart of rejection reasons
        rejection_stats = results.get('rejection_stats', {})
        if rejection_stats:
            reasons = list(rejection_stats.keys())
            counts = list(rejection_stats.values())
            total_rejected = sum(counts)

            if total_rejected > 0:
                bars = ax.bar(range(len(reasons)), counts, alpha=0.7,
                             color=['red', 'orange', 'yellow'][:len(reasons)])
                ax.set_xticks(range(len(reasons)))
                ax.set_xticklabels([r.replace('_', ' ').title() for r in reasons], rotation=45)
                ax.set_ylabel('Number Rejected')
                ax.set_title('Rejection Statistics')

                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                               str(count), ha='center', va='bottom', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Rejections\n✅ Perfect Sample!',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                ax.set_xticks([])
                ax.set_yticks([])

        ax.grid(True, alpha=0.3)

        # Overall title
        if title is None:
            significance = results.get('significance', 0)
            n_patches = results.get('n_measurements', 0)
            title = f'Cluster Analysis Summary (σ={significance:.1f}, N={n_patches})'

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        return fig, axes

    def plot_mass_range_profiles(self, results_dict, title=None, colors=None, figsize=(12, 8)):
        """
        Plot radial profiles for different mass ranges on the same plot

        Parameters:
        -----------
        results_dict : dict
            Dictionary from run_mass_range_analysis()
        title : str, optional
            Plot title
        colors : list, optional
            Colors for each mass bin
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Set up colors
        if colors is None:
            colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot each mass range
        for i, (label, results) in enumerate(results_dict.items()):
            if not results.get('success', False):
                continue

            radii = results.get('profile_radii')
            profile = results.get('profile_mean')
            errors = results.get('profile_errors')

            if radii is None or profile is None:
                continue

            # Filter out NaN values
            finite_mask = np.isfinite(profile) & np.isfinite(radii)
            if errors is not None:
                finite_mask = finite_mask & np.isfinite(errors)

            if not np.any(finite_mask):
                continue

            radii_plot = radii[finite_mask]
            profile_plot = profile[finite_mask]
            errors_plot = errors[finite_mask] if errors is not None else None

            # Get color and create label
            color = colors[i % len(colors)]
            n_clusters = results['n_measurements']
            significance = results['significance']

            if label == 'Random':
                plot_label = f"{label} (N={n_clusters}, σ={significance:.1f})"
                linestyle = '--'
                alpha = 0.7
            else:
                plot_label = f"{label} M☉ (N={n_clusters}, σ={significance:.1f})"
                linestyle = '-'
                alpha = 1.0

            # Plot with error bars
            if errors_plot is not None:
                ax.errorbar(radii_plot, profile_plot, yerr=errors_plot,
                           fmt='o-', color=color, markersize=4, linewidth=2,
                           capsize=3, label=plot_label, linestyle=linestyle, alpha=alpha)
            else:
                ax.plot(radii_plot, profile_plot, 'o-', color=color,
                       markersize=4, linewidth=2, label=plot_label,
                       linestyle=linestyle, alpha=alpha)

        # Formatting
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.set_xlabel('Radius [degrees]', fontsize=12)
        ax.set_ylabel('Y-parameter', fontsize=12)

        if title is None:
            title = 'Cluster Radial Profiles by Mass Range'
        ax.set_title(title, fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        # Add summary statistics
        n_mass_bins = len([k for k in results_dict.keys() if k != 'Random'])
        stats_text = f'Mass bins analyzed: {n_mass_bins}'
        if 'Random' in results_dict:
            stats_text += f'\nRandom control included'

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_mass_scaling(self, results_dict, title=None, figsize=(10, 7)):
        """
        Plot significance and signal strength vs mass

        Parameters:
        -----------
        results_dict : dict
            Dictionary from run_mass_range_analysis()
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Extract data (exclude random)
        mass_centers = []
        significances = []
        delta_ys = []
        delta_y_errors = []
        n_clusters = []
        labels = []

        for label, results in results_dict.items():
            if label == 'Random' or not results.get('success', False):
                continue

            mass_centers.append(results['mass_center'])
            significances.append(results['significance'])
            delta_ys.append(results['mean_delta_y'])
            delta_y_errors.append(results['error_mean'])
            n_clusters.append(results['n_measurements'])
            labels.append(label)

        if not mass_centers:
            print("❌ No valid mass data to plot!")
            return None, None

        # Convert to arrays
        mass_centers = np.array(mass_centers)
        significances = np.array(significances)
        delta_ys = np.array(delta_ys)
        delta_y_errors = np.array(delta_y_errors)
        n_clusters = np.array(n_clusters)

        # Create dual-axis plot
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Plot significance
        color1 = 'tab:blue'
        ax1.set_xlabel('Cluster Mass [M☉]', fontsize=12)
        ax1.set_ylabel('Detection Significance [σ]', color=color1, fontsize=12)
        ax1.errorbar(mass_centers, significances, fmt='o-', color=color1,
                    markersize=8, linewidth=2, capsize=3, label='Significance')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # Add significance threshold lines
        ax1.axhline(3, color='red', linestyle='--', alpha=0.5, label='3σ threshold')
        ax1.axhline(5, color='red', linestyle=':', alpha=0.5, label='5σ threshold')

        # Plot Δy on second axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Δy Signal Strength', color=color2, fontsize=12)
        ax2.errorbar(mass_centers, delta_ys, yerr=delta_y_errors, fmt='s-', color=color2,
                    markersize=6, linewidth=2, capsize=3, alpha=0.7, label='Δy')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add sample size annotations
        for i, (mass, sig, n) in enumerate(zip(mass_centers, significances, n_clusters)):
            ax1.annotate(f'N={n}', (mass, sig), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, alpha=0.7)

        # Formatting
        if title is None:
            title = 'Cluster Detection vs Mass'
        fig.suptitle(title, fontsize=14)

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.show()

        return fig, (ax1, ax2)

    def plot_mass_summary_grid(self, results_dict, figsize=(16, 12)):
        """
        Create a comprehensive summary plot with multiple panels
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Radial profiles (top-left)
        ax = axes[0, 0]
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'gray']

        for i, (label, results) in enumerate(results_dict.items()):
            if not results.get('success', False):
                continue

            radii = results.get('profile_radii')
            profile = results.get('profile_mean')

            if radii is None or profile is None:
                continue

            finite_mask = np.isfinite(profile) & np.isfinite(radii)
            if not np.any(finite_mask):
                continue

            color = colors[i % len(colors)]
            linestyle = '--' if label == 'Random' else '-'
            alpha = 0.7 if label == 'Random' else 1.0

            ax.plot(radii[finite_mask], profile[finite_mask],
                   color=color, linestyle=linestyle, alpha=alpha,
                   linewidth=2, label=f"{label} ({results['significance']:.1f}σ)")

        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Radius [degrees]')
        ax.set_ylabel('Y-parameter')
        ax.set_title('Radial Profiles by Mass')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Mass scaling (top-right)
        ax = axes[0, 1]

        # Extract non-random data
        mass_data = [(label, results) for label, results in results_dict.items()
                     if label != 'Random' and results.get('success', False)]

        if mass_data:
            masses = [results['mass_center'] for _, results in mass_data]
            sigs = [results['significance'] for _, results in mass_data]
            ns = [results['n_measurements'] for _, results in mass_data]

            ax.scatter(masses, sigs, s=[n*3 for n in ns], alpha=0.7, c=range(len(masses)), cmap='viridis')
            ax.set_xscale('log')
            ax.axhline(3, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Mass [M☉]')
            ax.set_ylabel('Significance [σ]')
            ax.set_title('Detection vs Mass')
            ax.grid(True, alpha=0.3)

            # Add annotations
            for mass, sig, n, (label, _) in zip(masses, sigs, ns, mass_data):
                ax.annotate(f'N={n}', (mass, sig), xytext=(3, 3),
                           textcoords='offset points', fontsize=7)

        # 3. Sample sizes (bottom-left)
        ax = axes[1, 0]

        labels = []
        sample_sizes = []
        colors_bar = []

        for i, (label, results) in enumerate(results_dict.items()):
            if results.get('success', False):
                labels.append(label)
                sample_sizes.append(results['n_measurements'])
                colors_bar.append(colors[i % len(colors)])

        if labels:
            bars = ax.bar(labels, sample_sizes, color=colors_bar, alpha=0.7)
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Sample Sizes')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, size in zip(bars, sample_sizes):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       str(size), ha='center', va='bottom', fontsize=9)

        # 4. Significance summary (bottom-right)
        ax = axes[1, 1]

        if labels and len(labels) > 1:
            significances = [results['significance'] for label, results in results_dict.items()
                            if results.get('success', False)]

            bars = ax.bar(labels, significances, color=colors_bar, alpha=0.7)
            ax.axhline(3, color='red', linestyle='--', alpha=0.5, label='3σ')
            ax.axhline(5, color='red', linestyle=':', alpha=0.5, label='5σ')
            ax.set_ylabel('Significance [σ]')
            ax.set_title('Detection Significance')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

            # Add value labels
            for bar, sig in zip(bars, significances):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                       f'{sig:.1f}σ', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

        return fig, axes

    def plot_y_mass_scaling(self, results_dict, title=None, figsize=(12, 8),
                            show_theory=True, theory_params=None, fit_scaling=True):
        """
        Plot scaling relations between Compton-y signal and halo mass

        Parameters:
        -----------
        results_dict : dict
            Dictionary from run_mass_range_analysis()
        title : str, optional
            Plot title
        figsize : tuple
            Figure size
        show_theory : bool
            Whether to show theoretical scaling predictions
        theory_params : dict, optional
            Parameters for theoretical scaling (see below for defaults)
        fit_scaling : bool
            Whether to fit a power law to the data

        Theory parameters dict can contain:
        - 'Y_star': Normalization (default: 1e-4 arcmin^2)
        - 'M_star': Pivot mass (default: 3e14 M_sun)
        - 'alpha': Mass slope (default: 1.79 from Planck)
        - 'z_ref': Reference redshift (default: 0.0)
        - 'E_z': E(z) factor (default: 1.0 for z~0)
        """

        if not results_dict:
            print("❌ No results to plot!")
            return None, None

        # Default theory parameters - adjusted for dimensionless y-parameter
        # NOTE: Converting from Y500 (arcmin^2) to dimensionless y-parameter
        # Rough conversion: Y500 ≈ y_avg × solid_angle
        # For R500 ~ 1° → solid_angle ~ 3000 arcmin^2
        # So y_avg ~ Y500 / 3000 → Y_star_dimensionless ~ 1e-4 / 3000 ~ 3e-8
        default_theory = {
            'Y_star': 3e-8,  # Dimensionless y-parameter at pivot mass
            'M_star': 3e14,  # M_sun
            'alpha': 1.79,   # Mass slope
            'z_ref': 0.0,    # Reference redshift
            'E_z': 1.0       # E(z) = H(z)/H_0 factor
        }

        if theory_params is None:
            theory_params = default_theory
        else:
            # Update defaults with provided values
            theory_params = {**default_theory, **theory_params}

        # Extract observational data (exclude random)
        masses = []
        y_signals = []
        y_errors = []
        n_clusters = []
        labels = []
        redshifts = []

        for label, results in results_dict.items():
            if label == 'Random' or not results.get('success', False):
                continue

            mass_center = results['mass_center']
            delta_y = results['mean_delta_y']
            error = results['error_mean']
            n_clust = results['n_measurements']
            z_mean = results.get('mean_redshift', 0.02)  # Default z~0.02 for local

            masses.append(mass_center)
            y_signals.append(abs(delta_y))  # Take absolute value for log plot
            y_errors.append(error)
            n_clusters.append(n_clust)
            labels.append(label)
            redshifts.append(z_mean)

        if not masses:
            print("❌ No valid mass data to plot!")
            return None, None

        # Convert to arrays
        masses = np.array(masses)
        y_signals = np.array(y_signals)
        y_errors = np.array(y_errors)
        n_clusters = np.array(n_clusters)
        redshifts = np.array(redshifts)

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot theoretical scaling if requested
        if show_theory:
            mass_theory = np.logspace(13.5, 15.5, 100)  # 10^13.5 to 10^15.5 M_sun

            # Theoretical Y-M relation: Y = Y_star * (M/M_star)^alpha * E(z)^beta
            # For simplicity, assume beta ~ 2/3 and E(z) ~ 1 for low-z
            y_theory = (theory_params['Y_star'] *
                       (mass_theory / theory_params['M_star'])**theory_params['alpha'] *
                       theory_params['E_z']**(2/3))

            ax.plot(mass_theory, y_theory, 'k--', linewidth=2, alpha=0.7,
                    label=f'Theory (α={theory_params["alpha"]:.1f}, adjusted for Δy)')

            # Show uncertainty band (±20% typical for theory)
            ax.fill_between(mass_theory, y_theory*0.8, y_theory*1.2,
                           color='gray', alpha=0.2, label='Theory ±20%')

        # Plot observational data
        colors = plt.cm.viridis(np.linspace(0, 1, len(masses)))

        for i, (mass, y_sig, y_err, n, label) in enumerate(zip(masses, y_signals, y_errors, n_clusters, labels)):
            # Size proportional to number of clusters
            size = 50 + n * 2

            ax.errorbar(mass, y_sig, yerr=y_err, fmt='o', color=colors[i],
                       markersize=np.sqrt(size/10), linewidth=2, capsize=4,
                       label=f'{label} M☉ (N={n})', alpha=0.8)

            # Add text annotation
            ax.annotate(f'{n}', (mass, y_sig), xytext=(3, 3),
                       textcoords='offset points', fontsize=8,
                       ha='left', va='bottom', alpha=0.7)

        # Fit power law to data if requested
        if fit_scaling and len(masses) >= 3:
            try:
                # Fit in log space: log(Y) = log(A) + alpha * log(M/M_pivot)
                log_masses = np.log10(masses / theory_params['M_star'])
                log_y = np.log10(y_signals)
                weights = 1.0 / (y_errors / y_signals / np.log(10))  # Convert to log space

                # Weighted linear fit
                coeffs = np.polyfit(log_masses, log_y, 1, w=weights)
                fitted_alpha = coeffs[0]
                log_norm = coeffs[1]

                # Plot fitted relation
                mass_fit = np.logspace(13.5, 15.5, 100)
                log_mass_fit = np.log10(mass_fit / theory_params['M_star'])
                log_y_fit = log_norm + fitted_alpha * log_mass_fit
                y_fit = 10**log_y_fit

                ax.plot(mass_fit, y_fit, 'r-', linewidth=2, alpha=0.8,
                       label=f'Fitted: α={fitted_alpha:.2f}±?')

                # Calculate chi-squared
                log_y_pred = log_norm + fitted_alpha * log_masses
                chi2 = np.sum(weights * (log_y - log_y_pred)**2)
                dof = len(masses) - 2

                print(f"📊 Scaling Fit Results:")
                print(f"   Fitted slope: α = {fitted_alpha:.2f}")
                print(f"   Theory slope: α = {theory_params['alpha']:.2f}")
                print(f"   χ²/dof = {chi2:.1f}/{dof} = {chi2/dof:.2f}")

            except Exception as e:
                print(f"⚠️  Fitting failed: {e}")

        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Halo Mass [M☉]', fontsize=13)
        ax.set_ylabel('|Δy| Signal Strength (dimensionless)', fontsize=13)

        if title is None:
            title = 'Compton-y vs Halo Mass Scaling Relation'
        ax.set_title(title, fontsize=14)

        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='best')

        # Add mass range indicators
        ax.axvline(1e14, color='gray', linestyle=':', alpha=0.5, label='Group/Cluster transition')

        # Set reasonable axis limits
        if len(masses) > 0:
            mass_min, mass_max = masses.min(), masses.max()
            mass_range = mass_max / mass_min
            ax.set_xlim(mass_min / 2, mass_max * 2)

            y_min, y_max = y_signals.min(), y_signals.max()
            ax.set_ylim(y_min / 3, y_max * 3)

        # Add information box
        info_text = f"""
    Scaling Analysis:
    • Mass range: {masses.min():.1e} - {masses.max():.1e} M☉
    • Total clusters: {n_clusters.sum()}
    • Mass bins: {len(masses)}
    • Redshift: z < 0.05
    """

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plt.tight_layout()
        plt.show()

        # Return fit parameters if fitting was done
        fit_results = None
        if fit_scaling and len(masses) >= 3:
            try:
                fit_results = {
                    'fitted_slope': fitted_alpha,
                    'theory_slope': theory_params['alpha'],
                    'chi2': chi2,
                    'dof': dof,
                    'masses': masses,
                    'y_signals': y_signals,
                    'y_errors': y_errors
                }
            except:
                pass

        return fig, ax, fit_results
