import numpy as np

class StatisticsCalculator:
    """Statistical calculations for cluster analysis"""
    
    @staticmethod
    def calculate_sample_statistics(measurements):
        """Calculate sample statistics from measurements"""
        measurements = np.array(measurements)
        
        mean_val = np.mean(measurements)
        std_val = np.std(measurements)
        error_val = std_val / np.sqrt(len(measurements))
        significance = mean_val / error_val if error_val > 0 else 0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'error': error_val,
            'significance': significance,
            'n_samples': len(measurements)
        }
    
    @staticmethod
    def calculate_r200_statistics(r200_values):
        """Calculate R200 statistics"""
        r200_values = np.array(r200_values)
        
        return {
            'median': np.median(r200_values),
            'mean': np.mean(r200_values),
            'std': np.std(r200_values),
            'min': np.min(r200_values),
            'max': np.max(r200_values),
            'range': (np.min(r200_values), np.max(r200_values))
        }
    
    @staticmethod
    def calculate_weighted_mean(values, errors):
        """Calculate weighted mean and error"""
        values = np.array(values)
        errors = np.array(errors)
        
        weights = 1.0 / errors**2
        weighted_mean = np.sum(weights * values) / np.sum(weights)
        weighted_error = 1.0 / np.sqrt(np.sum(weights))
        
        return weighted_mean, weighted_error
    
    @staticmethod
    def calculate_significance(signal, error):
        """Calculate detection significance"""
        return signal / error if error > 0 else 0
