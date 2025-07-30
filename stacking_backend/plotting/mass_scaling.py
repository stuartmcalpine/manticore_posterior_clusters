import numpy as np
import matplotlib.pyplot as plt

class MassScalingPlotter:
    """Plotting functionality for mass scaling relations"""
    
    @staticmethod
    def plot_y_mass_scaling(results_dict, title=None, figsize=(12, 8),
                           show_theory=True, theory_params=None, fit_scaling=True):
        """Plot scaling relations between Compton-y signal and halo mass"""
        
        if not results_dict:
            print("❌ No results to plot!")
            return None, None
        
        # Default theory parameters
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
            theory_params = {**default_theory, **theory_params}
        
        # Extract observational data
        masses, y_signals, y_errors, n_clusters, labels = MassScalingPlotter._extract_mass_data(results_dict)
        
        if not masses:
            print("❌ No valid mass data to plot!")
            return None, None
        
        # Convert to arrays
        masses = np.array(masses)
        y_signals = np.array(y_signals)
        y_errors = np.array(y_errors)
        n_clusters = np.array(n_clusters)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot theoretical scaling if requested
        if show_theory:
            MassScalingPlotter._plot_theory_scaling(ax, theory_params)
        
        # Plot observational data
        MassScalingPlotter._plot_observational_data(ax, masses, y_signals, y_errors, n_clusters, labels)
        
        # Fit power law to data if requested
        fit_results = None
        if fit_scaling and len(masses) >= 3:
            fit_results = MassScalingPlotter._fit_and_plot_scaling(ax, masses, y_signals, y_errors, theory_params)
        
        # Formatting
        MassScalingPlotter._format_scaling_plot(ax, masses, y_signals, n_clusters, title)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, fit_results
    
    @staticmethod
    def _extract_mass_data(results_dict):
        """Extract mass scaling data from results dictionary"""
        masses = []
        y_signals = []
        y_errors = []
        n_clusters = []
        labels = []
        
        for label, results in results_dict.items():
            if 'random' in label.lower() or not results.get('success', False):
                continue
            
            mass_center = results['mass_center']
            delta_y = results['mean_delta_y']
            error = results['error_mean']
            n_clust = results['n_measurements']
            
            masses.append(mass_center)
            y_signals.append(abs(delta_y))  # Take absolute value for log plot
            y_errors.append(error)
            n_clusters.append(n_clust)
            labels.append(label)
        
        return masses, y_signals, y_errors, n_clusters, labels
    
    @staticmethod
    def _plot_theory_scaling(ax, theory_params):
        """Plot theoretical scaling relation"""
        mass_theory = np.logspace(13.5, 15.5, 100)
        
        # Theoretical Y-M relation
        y_theory = (theory_params['Y_star'] *
                   (mass_theory / theory_params['M_star'])**theory_params['alpha'] *
                   theory_params['E_z']**(2/3))
        
        ax.plot(mass_theory, y_theory, 'k--', linewidth=2, alpha=0.7,
                label=f'Theory (α={theory_params["alpha"]:.1f}, adjusted for Δy)')
        
        # Show uncertainty band
        ax.fill_between(mass_theory, y_theory*0.8, y_theory*1.2,
                       color='gray', alpha=0.2, label='Theory ±20%')
    
    @staticmethod
    def _plot_observational_data(ax, masses, y_signals, y_errors, n_clusters, labels):
        """Plot observational data points"""
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
    
    @staticmethod
    def _fit_and_plot_scaling(ax, masses, y_signals, y_errors, theory_params):
        """Fit and plot power law scaling"""
        try:
            # Fit in log space
            log_masses = np.log10(masses / theory_params['M_star'])
            log_y = np.log10(y_signals)
            weights = 1.0 / (y_errors / y_signals / np.log(10))
            
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
            
            return {
                'fitted_slope': fitted_alpha,
                'theory_slope': theory_params['alpha'],
                'chi2': chi2,
                'dof': dof,
                'masses': masses,
                'y_signals': y_signals,
                'y_errors': y_errors
            }
            
        except Exception as e:
            print(f"⚠️  Fitting failed: {e}")
            return None
    
    @staticmethod
    def _format_scaling_plot(ax, masses, y_signals, n_clusters, title):
        """Format the scaling plot"""
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Halo Mass [M☉]', fontsize=13)
        ax.set_ylabel('|Δy| Signal Strength (dimensionless)', fontsize=13)
        ax.set_title(title or 'Compton-y vs Halo Mass Scaling Relation', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='best')
        
        # Add mass range indicators
        ax.axvline(1e14, color='gray', linestyle=':', alpha=0.5, label='Group/Cluster transition')
        
        # Set reasonable axis limits
        if len(masses) > 0:
            mass_min, mass_max = masses.min(), masses.max()
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
