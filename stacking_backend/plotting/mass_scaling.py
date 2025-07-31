# stacking_backend/plotting/mass_scaling.py
import numpy as np
import matplotlib.pyplot as plt

class MassScalingPlotter:
    """Plotting functionality for mass scaling relations"""
    
    @staticmethod
    def plot_y_mass_scaling(results_dict, title=None, figsize=(12, 8),
                           show_theory=True, theory_params=None, fit_scaling=True):
        """Plot scaling relations between Compton-y signal and halo mass for multiple datasets"""
        
        if not results_dict:
            print("❌ No results to plot!")
            return None, None, None
        
        # Default theory parameters
        default_theory = {
            'Y_star': 3e-8,
            'M_star': 3e14,
            'alpha': 1.79,
            'z_ref': 0.0,
            'E_z': 1.0
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
            y_theory = (theory_params['Y_star'] *
                       (mass_theory / theory_params['M_star'])**theory_params['alpha'] *
                       theory_params['E_z']**(2/3))
            
            ax.plot(mass_theory, y_theory, 'k--', linewidth=2, alpha=0.7,
                    label=f'Theory (α={theory_params["alpha"]:.1f})')
            ax.fill_between(mass_theory, y_theory*0.8, y_theory*1.2,
                           color='gray', alpha=0.2, label='Theory ±20%')
        
        # Plot each dataset
        all_fit_results = {}
        
        for dataset_name, dataset_results in results_dict.items():
            if dataset_name not in dataset_styles:
                continue
                
            style = dataset_styles[dataset_name]
            
            # Extract data for this dataset
            masses, y_signals, y_errors, n_clusters, labels = [], [], [], [], []
            
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
                    
                masses.append(mass_center)
                y_signals.append(abs(results['mean_delta_y']))
                y_errors.append(results['error_mean'])
                n_clusters.append(results['n_measurements'])
                labels.append(mass_label)
            
            if not masses:
                continue
                
            masses = np.array(masses)
            y_signals = np.array(y_signals)
            y_errors = np.array(y_errors)
            n_clusters = np.array(n_clusters)
            
            # Plot data points
            for i, (mass, y_sig, y_err, n) in enumerate(zip(masses, y_signals, y_errors, n_clusters)):
                size = 50 + n * 2
                ax.errorbar(mass, y_sig, yerr=y_err, fmt=style['marker'], 
                           color=style['color'], markersize=np.sqrt(size/10), 
                           linewidth=2, capsize=4, alpha=0.8,
                           label=style['label'] if i == 0 else "")
                
                ax.annotate(f'{n}', (mass, y_sig), xytext=(3, 3),
                           textcoords='offset points', fontsize=8,
                           ha='left', va='bottom', alpha=0.7)
            
            # Fit scaling relation for this dataset
            if fit_scaling and len(masses) >= 3:
                try:
                    log_masses = np.log10(masses / theory_params['M_star'])
                    log_y = np.log10(y_signals)
                    weights = 1.0 / (y_errors / y_signals / np.log(10))
                    
                    coeffs = np.polyfit(log_masses, log_y, 1, w=weights)
                    fitted_alpha = coeffs[0]
                    log_norm = coeffs[1]
                    
                    # Plot fitted relation
                    mass_fit = np.logspace(13.5, 15.5, 100)
                    log_mass_fit = np.log10(mass_fit / theory_params['M_star'])
                    log_y_fit = log_norm + fitted_alpha * log_mass_fit
                    y_fit = 10**log_y_fit
                    
                    ax.plot(mass_fit, y_fit, '-', color=style['color'], 
                           linewidth=2, alpha=0.6,
                           label=f'{style["label"]} fit: α={fitted_alpha:.2f}')
                    
                    # Calculate chi-squared
                    log_y_pred = log_norm + fitted_alpha * log_masses
                    chi2 = np.sum(weights * (log_y - log_y_pred)**2)
                    dof = len(masses) - 2
                    
                    all_fit_results[dataset_name] = {
                        'fitted_slope': fitted_alpha,
                        'theory_slope': theory_params['alpha'],
                        'chi2': chi2,
                        'dof': dof,
                        'masses': masses,
                        'y_signals': y_signals,
                        'y_errors': y_errors
                    }
                    
                except Exception as e:
                    print(f"⚠️ Fitting failed for {dataset_name}: {e}")
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Halo Mass [M☉]', fontsize=13)
        ax.set_ylabel('|Δy| Signal Strength (dimensionless)', fontsize=13)
        ax.set_title(title or 'Compton-y vs Halo Mass Scaling: Multi-Dataset Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='best')
        ax.axvline(1e14, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, all_fit_results
