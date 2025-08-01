# stacking_backend/plotting/mass_scaling.py
import numpy as np
import matplotlib.pyplot as plt

class MassScalingPlotter:
    """Plotting functionality for mass scaling relations"""
    
    @staticmethod
    def plot_y_mass_scaling(results_dict, title=None, figsize=(12, 8),
                           show_theory=True, theory_params=None, fit_scaling=True):
        """Plot Y500-M500 scaling relations for multiple datasets"""
        
        if not results_dict:
            print("❌ No results to plot!")
            return None, None, None
        
        # Default theory parameters for Y500-M500 relation
        # Based on Planck 2013 results: Y500 ∝ (M500/3e14)^1.79
        default_theory = {
            'Y_star': 1.0e-4,    # Y500 at pivot mass in 10^-4 arcmin^2 units
            'M_star': 3e14,      # Pivot mass in M_sun
            'alpha': 1.79,       # Mass slope from Planck
            'z_ref': 0.0,        # Reference redshift
            'E_z': 1.0           # E(z) factor
        }
        
        if theory_params is None:
            theory_params = default_theory
        else:
            theory_params = {**default_theory, **theory_params}
        
        # Dataset styles
        dataset_styles = {
            'manticore': {'color': 'blue', 'marker': 'o', 'label': 'Manticore'},
            'erosita': {'color': 'red', 'marker': 's', 'label': 'eROSITA'},
            'mcxc': {'color': 'green', 'marker': '^', 'label': 'MCXC'}
        }
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot theoretical scaling if requested
        if show_theory:
            mass_theory = np.logspace(13.5, 15.5, 100)
            y500_theory = (theory_params['Y_star'] *
                          (mass_theory / theory_params['M_star'])**theory_params['alpha'] *
                          theory_params['E_z']**(2/3))
            
            ax.plot(mass_theory, y500_theory, 'k--', linewidth=2, alpha=0.7,
                    label=f'Theory (α={theory_params["alpha"]:.1f})')
            ax.fill_between(mass_theory, y500_theory*0.8, y500_theory*1.2,
                           color='gray', alpha=0.2, label='Theory ±20%')
        
        # Plot each dataset
        all_fit_results = {}
        
        for dataset_name, dataset_results in results_dict.items():
            if dataset_name not in dataset_styles:
                continue
                
            style = dataset_styles[dataset_name]
            
            # Extract data for this dataset
            masses, y500_values, y500_errors, n_clusters, labels = [], [], [], [], []
            
            for mass_label, results in dataset_results.items():
                if not results.get('success', False):
                    continue
                
                # Calculate mass_center if not present
                if 'mass_center' in results:
                    mass_center = results['mass_center']
                elif 'mass_range' in results:
                    mass_min, mass_max = results['mass_range']
                    mass_center = np.sqrt(mass_min * mass_max)
                else:
                    # Try to parse from mass_label
                    try:
                        parts = mass_label.split('-')
                        mass_min = float(parts[0].replace('e+', 'e'))
                        mass_max = float(parts[1].replace('e+', 'e'))
                        mass_center = np.sqrt(mass_min * mass_max)
                    except:
                        print(f"Warning: Cannot determine mass for {dataset_name} {mass_label}")
                        continue
                
                # Extract Y500 values from individual measurements
                individual_results = results.get('individual_results', [])
                if not individual_results:
                    print(f"Warning: No individual results for {dataset_name} {mass_label}")
                    continue
                
                # Calculate mean Y500 and error
                y500_individual = [res['y500'] for res in individual_results if 'y500' in res]
                if not y500_individual:
                    print(f"Warning: No Y500 values for {dataset_name} {mass_label}")
                    continue
                
                y500_individual = np.array(y500_individual)
                # Convert to 10^-4 arcmin^2 units for literature comparison
                y500_individual_scaled = y500_individual * 1e4
                
                mean_y500 = np.mean(y500_individual_scaled)
                error_y500 = np.std(y500_individual_scaled) / np.sqrt(len(y500_individual_scaled))
                
                masses.append(mass_center)
                y500_values.append(mean_y500)
                y500_errors.append(error_y500)
                n_clusters.append(len(y500_individual))
                labels.append(mass_label)
            
            if not masses:
                continue
                
            masses = np.array(masses)
            y500_values = np.array(y500_values)
            y500_errors = np.array(y500_errors)
            n_clusters = np.array(n_clusters)
            
            # Plot data points
            for i, (mass, y500, y500_err, n) in enumerate(zip(masses, y500_values, y500_errors, n_clusters)):
                size = 50 + n * 2
                ax.errorbar(mass, y500, yerr=y500_err, fmt=style['marker'], 
                           color=style['color'], markersize=np.sqrt(size/10), 
                           linewidth=2, capsize=4, alpha=0.8,
                           label=style['label'] if i == 0 else "")
                
                ax.annotate(f'{n}', (mass, y500), xytext=(3, 3),
                           textcoords='offset points', fontsize=8,
                           ha='left', va='bottom', alpha=0.7)
            
            # Fit scaling relation for this dataset
            if fit_scaling and len(masses) >= 3:
                try:
                    log_masses = np.log10(masses / theory_params['M_star'])
                    log_y500 = np.log10(y500_values)
                    weights = 1.0 / (y500_errors / y500_values / np.log(10))
                    
                    coeffs = np.polyfit(log_masses, log_y500, 1, w=weights)
                    fitted_alpha = coeffs[0]
                    log_norm = coeffs[1]
                    
                    # Plot fitted relation
                    mass_fit = np.logspace(13.5, 15.5, 100)
                    log_mass_fit = np.log10(mass_fit / theory_params['M_star'])
                    log_y500_fit = log_norm + fitted_alpha * log_mass_fit
                    y500_fit = 10**log_y500_fit
                    
                    ax.plot(mass_fit, y500_fit, '-', color=style['color'], 
                           linewidth=2, alpha=0.6,
                           label=f'{style["label"]} fit: α={fitted_alpha:.2f}')
                    
                    # Calculate chi-squared
                    log_y500_pred = log_norm + fitted_alpha * log_masses
                    chi2 = np.sum(weights * (log_y500 - log_y500_pred)**2)
                    dof = len(masses) - 2
                    
                    all_fit_results[dataset_name] = {
                        'fitted_slope': fitted_alpha,
                        'theory_slope': theory_params['alpha'],
                        'chi2': chi2,
                        'dof': dof,
                        'masses': masses,
                        'y500_values': y500_values,
                        'y500_errors': y500_errors
                    }
                    
                except Exception as e:
                    print(f"⚠️ Fitting failed for {dataset_name}: {e}")
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('M₅₀₀ [M☉]', fontsize=13)
        ax.set_ylabel('Y₅₀₀ [10⁻⁴ arcmin²]', fontsize=13)
        ax.set_title(title or 'Y₅₀₀-M₅₀₀ Scaling Relation: Multi-Dataset Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='best')
        ax.axvline(1e14, color='gray', linestyle=':', alpha=0.5, label='Group/Cluster transition')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, all_fit_results
