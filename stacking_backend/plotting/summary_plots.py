import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import PlotUtils

class SummaryPlotter:
    """Summary and diagnostic plotting functionality"""
    
    @staticmethod
    def plot_analysis_summary(results, title=None, cmap='RdBu_r'):
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
        
        # 1. Stacked patch (top-left)
        SummaryPlotter._plot_stacked_patch_panel(axes[0, 0], stacked_patch, patch_size_deg, cmap, results)
        
        # 2. Radial profile (top-right)
        SummaryPlotter._plot_radial_profile_panel(axes[0, 1], radii, profile, profile_errors)
        
        # 3. Key measurements (bottom-left)
        SummaryPlotter._plot_measurements_panel(axes[1, 0], results)
        
        # 4. Sample quality metrics (bottom-right)
        SummaryPlotter._plot_quality_panel(axes[1, 1], results)
        
        # Overall title
        if title is None:
            significance = results.get('significance', 0)
            n_patches = results.get('n_measurements', 0)
            title = f'Cluster Analysis Summary (σ={significance:.1f}, N={n_patches})'
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    @staticmethod
    def _plot_stacked_patch_panel(ax, stacked_patch, patch_size_deg, cmap, results):
        """Plot stacked patch panel"""
        if stacked_patch is not None:
            extent = PlotUtils.format_patch_plot(ax, patch_size_deg, 'Stacked Y-Map')
            
            # Calculate color limits
            vmin, vmax = PlotUtils.calculate_color_limits(stacked_patch, [5, 95])
            
            im = ax.imshow(stacked_patch, extent=extent, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Center and apertures
            ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
            
            inner_radius = results.get('r200_median', 0) * results.get('inner_r200_factor', 1)
            outer_radius = results.get('r200_median', 0) * results.get('outer_r200_factor', 3)
            
            if inner_radius and outer_radius:
                PlotUtils.add_circle_apertures(ax, (0, 0), inner_radius, outer_radius)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    @staticmethod
    def _plot_radial_profile_panel(ax, radii, profile, profile_errors):
        """Plot radial profile panel"""
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
    
    @staticmethod
    def _plot_measurements_panel(ax, results):
        """Plot key measurements panel"""
        ax.axis('off')
        
        measurements_text = f"""
    CLUSTER ANALYSIS RESULTS

    Signal Measurement:
    • Δy = {results.get('mean_delta_y', np.nan):.2e}
    • Error = {results.get('error_mean', np.nan):.2e}
    • Significance = {results.get('significance', np.nan):.1f}σ

    Aperture Details:
    • R200 median = {results.get('r200_median', np.nan):.3f}°
    • Inner factor = {results.get('inner_r200_factor', np.nan):.1f}
    • Outer factor = {results.get('outer_r200_factor', np.nan):.1f}

    Sample Information:
    • Valid patches = {results.get('n_measurements', 0)}
    • Input coordinates = {results.get('n_input_coords', 0)}
    • Rejection rate = {results.get('n_rejected', 0)}/{results.get('n_input_coords', 0)}
    • Patch size = {results.get('patch_size_deg', 15.0)}° × {results.get('patch_size_deg', 15.0)}°
    """
        
        ax.text(0.05, 0.95, measurements_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    @staticmethod
    def _plot_quality_panel(ax, results):
        """Plot sample quality metrics panel"""
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
    
    @staticmethod
    def plot_mass_range_profiles(results_dict, title=None, colors=None, figsize=(12, 8)):
        """Plot radial profiles for different mass ranges"""
        
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
            
            if 'random' in label.lower():
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
        ax.set_title(title or 'Cluster Radial Profiles by Mass Range', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        # Add summary statistics
        n_mass_bins = len([k for k in results_dict.keys() if 'random' not in k.lower()])
        stats_text = f'Mass bins analyzed: {n_mass_bins}'
        if any('random' in k.lower() for k in results_dict.keys()):
            stats_text += f'\nRandom control included'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
