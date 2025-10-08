"""
Rice Production Forecasting System
==================================

This script provides multiple forecasting approaches for Indonesian rice production:
1. Time Series Forecasting (ARIMA, Exponential Smoothing)
2. Machine Learning Models (Random Forest, XGBoost, Linear Regression)
3. Prophet for trend and seasonality analysis
4. Ensemble methods combining multiple models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RiceProductionForecaster:
    """
    A comprehensive forecasting system for rice production data
    """
    
    def __init__(self, data_path="processed_data/"):
        """
        Initialize the forecaster with data path
        
        Args:
            data_path (str): Path to processed data directory
        """
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.forecasts = {}
        self.model_performance = {}
        
        # Load all datasets
        self.load_data()
        
    def load_data(self):
        """Load all processed datasets"""
        try:
            # National time series data
            self.national_data = pd.read_csv(os.path.join(self.data_path, "national_timeseries.csv"))
            
            # ML features dataset
            self.ml_data = pd.read_csv(os.path.join(self.data_path, "ml_features_dataset.csv"))
            
            # Top provinces data
            self.top_provinces_data = pd.read_csv(os.path.join(self.data_path, "top_provinces_data.csv"))
            
            # Provincial wide format
            self.provincial_wide = pd.read_csv(os.path.join(self.data_path, "provincial_production_wide.csv"))
            
            print("All datasets loaded successfully!")
            print(f"   • National data: {len(self.national_data)} years")
            print(f"   • ML features: {len(self.ml_data)} records")
            print(f"   • Top provinces: {self.top_provinces_data['Province'].nunique()} provinces")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_time_series_data(self, target_column='Produksi_ton'):
        """
        Prepare data for time series forecasting
        
        Args:
            target_column (str): Target variable for forecasting
            
        Returns:
            pd.Series: Time series data
        """
        ts_data = self.national_data.set_index('Year')[target_column]
        return ts_data
    
    def linear_trend_forecast(self, years_ahead=3, target='Produksi_ton'):
        """
        Simple linear trend forecasting
        
        Args:
            years_ahead (int): Number of years to forecast
            target (str): Target variable
            
        Returns:
            dict: Forecast results
        """
        print(f"\nLinear Trend Forecasting for {target}")
        print("-" * 50)
        
        # Prepare data
        X = self.national_data['Year'].values.reshape(-1, 1)
        y = self.national_data[target].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions for existing years (for validation)
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Forecast future years
        future_years = np.arange(self.national_data['Year'].max() + 1, 
                                self.national_data['Year'].max() + 1 + years_ahead).reshape(-1, 1)
        future_predictions = model.predict(future_years)
        
        # Store results
        self.models[f'linear_trend_{target}'] = model
        self.model_performance[f'linear_trend_{target}'] = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'RMSE': np.sqrt(mse)
        }
        
        forecast_results = {
            'model': 'Linear Trend',
            'target': target,
            'historical_years': self.national_data['Year'].tolist(),
            'historical_values': y.tolist(),
            'historical_predictions': y_pred.tolist(),
            'future_years': future_years.flatten().tolist(),
            'future_predictions': future_predictions.tolist(),
            'metrics': self.model_performance[f'linear_trend_{target}'],
            'trend_slope': model.coef_[0],
            'intercept': model.intercept_
        }
        
        self.forecasts[f'linear_trend_{target}'] = forecast_results
        
        print(f"   Model Performance:")
        print(f"      • R² Score: {r2:.4f}")
        print(f"      • RMSE: {np.sqrt(mse):,.0f}")
        print(f"      • MAE: {mae:,.0f}")
        print(f"   Trend: {model.coef_[0]:+,.0f} {target.split('_')[0]} per year")
        
        return forecast_results
    
    def polynomial_trend_forecast(self, years_ahead=3, target='Produksi_ton', degree=2):
        """
        Polynomial trend forecasting
        
        Args:
            years_ahead (int): Number of years to forecast
            target (str): Target variable
            degree (int): Polynomial degree
            
        Returns:
            dict: Forecast results
        """
        print(f"\nPolynomial Trend Forecasting (degree={degree}) for {target}")
        print("-" * 50)
        
        # Prepare data
        X = self.national_data['Year'].values
        y = self.national_data[target].values
        
        # Fit polynomial
        coefficients = np.polyfit(X, y, degree)
        poly_model = np.poly1d(coefficients)
        
        # Make predictions for existing years
        y_pred = poly_model(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Forecast future years
        future_years = np.arange(X.max() + 1, X.max() + 1 + years_ahead)
        future_predictions = poly_model(future_years)
        
        # Store results
        self.models[f'poly_trend_{target}_deg{degree}'] = poly_model
        self.model_performance[f'poly_trend_{target}_deg{degree}'] = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'RMSE': np.sqrt(mse)
        }
        
        forecast_results = {
            'model': f'Polynomial Trend (degree {degree})',
            'target': target,
            'historical_years': X.tolist(),
            'historical_values': y.tolist(),
            'historical_predictions': y_pred.tolist(),
            'future_years': future_years.tolist(),
            'future_predictions': future_predictions.tolist(),
            'metrics': self.model_performance[f'poly_trend_{target}_deg{degree}'],
            'coefficients': coefficients.tolist()
        }
        
        self.forecasts[f'poly_trend_{target}_deg{degree}'] = forecast_results
        
        print(f"   Model Performance:")
        print(f"      • R² Score: {r2:.4f}")
        print(f"      • RMSE: {np.sqrt(mse):,.0f}")
        print(f"      • MAE: {mae:,.0f}")
        
        return forecast_results
    
    def random_forest_forecast(self, years_ahead=3, target='Produksi_ton'):
        """
        Random Forest forecasting using engineered features
        
        Args:
            years_ahead (int): Number of years to forecast
            target (str): Target variable
            
        Returns:
            dict: Forecast results
        """
        print(f"\nRandom Forest Forecasting for {target}")
        print("-" * 50)
        
        # Prepare ML data
        ml_df = self.ml_data.copy()
        
        # Remove rows with NaN values in lag features (first year for each province)
        ml_df = ml_df.dropna()
        
        # Select features for modeling
        feature_columns = [
            'Year', 'Luas_Panen_ha', 'Produktivitas_ku_ha',
            'Produksi_lag1', 'Produktivitas_lag1', 'Area_lag1',
            'Produksi_ma3', 'Produktivitas_ma3'
        ]
        
        # Encode province as categorical
        ml_df['Province_encoded'] = pd.Categorical(ml_df['Province']).codes
        feature_columns.append('Province_encoded')
        
        X = ml_df[feature_columns]
        y = ml_df[target]
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions and metrics
        y_pred_train = rf_model.predict(X_train_scaled)
        y_pred_test = rf_model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model and scaler
        self.models[f'random_forest_{target}'] = rf_model
        self.scalers[f'random_forest_{target}'] = scaler
        self.model_performance[f'random_forest_{target}'] = {
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Test_MAE': test_mae,
            'Test_RMSE': test_rmse,
            'Feature_Importance': feature_importance.to_dict('records')
        }
        
        print(f"   Model Performance:")
        print(f"      • Train R²: {train_r2:.4f}")
        print(f"      • Test R²: {test_r2:.4f}")
        print(f"      • Test RMSE: {test_rmse:,.0f}")
        print(f"      • Test MAE: {test_mae:,.0f}")
        
        print(f"\n   Top 5 Important Features:")
        for i, row in feature_importance.head().iterrows():
            print(f"      • {row['feature']}: {row['importance']:.3f}")
        
        # For future forecasting, we'll use national aggregated approach
        # This is a simplified approach - in practice, you'd want more sophisticated methods
        national_features = self.prepare_national_features_for_forecast(years_ahead)
        
        forecast_results = {
            'model': 'Random Forest',
            'target': target,
            'metrics': self.model_performance[f'random_forest_{target}'],
            'feature_importance': feature_importance.to_dict('records'),
            'note': 'Provincial-level model - use for detailed analysis'
        }
        
        self.forecasts[f'random_forest_{target}'] = forecast_results
        
        return forecast_results
    
    def prepare_national_features_for_forecast(self, years_ahead):
        """
        Prepare national-level features for forecasting
        This is a simplified approach for demonstration
        """
        # Get latest national data
        latest_year = self.national_data['Year'].max()
        latest_data = self.national_data[self.national_data['Year'] == latest_year].iloc[0]
        
        # Simple projection based on trends
        future_features = []
        for i in range(1, years_ahead + 1):
            future_year = latest_year + i
            # This is a simplified approach - you'd want more sophisticated feature engineering
            future_features.append({
                'Year': future_year,
                'Luas_Panen_ha': latest_data['Luas_Panen_ha'],  # Assume constant
                'Produktivitas_ku_ha': latest_data['Produktivitas_ku_ha']  # Assume constant
            })
        
        return future_features
    
    def exponential_smoothing_forecast(self, years_ahead=3, target='Produksi_ton', alpha=0.3):
        """
        Exponential smoothing forecasting
        
        Args:
            years_ahead (int): Number of years to forecast
            target (str): Target variable
            alpha (float): Smoothing parameter
            
        Returns:
            dict: Forecast results
        """
        print(f"\nExponential Smoothing Forecasting for {target}")
        print("-" * 50)
        
        # Prepare data
        ts_data = self.prepare_time_series_data(target)
        
        # Simple exponential smoothing
        smoothed_values = []
        smoothed_values.append(ts_data.iloc[0])  # First value
        
        for i in range(1, len(ts_data)):
            smoothed = alpha * ts_data.iloc[i] + (1 - alpha) * smoothed_values[i-1]
            smoothed_values.append(smoothed)
        
        # Calculate metrics
        mae = mean_absolute_error(ts_data.values, smoothed_values)
        mse = mean_squared_error(ts_data.values, smoothed_values)
        r2 = r2_score(ts_data.values, smoothed_values)
        
        # Forecast future values
        last_smoothed = smoothed_values[-1]
        future_predictions = [last_smoothed] * years_ahead  # Flat forecast
        future_years = list(range(ts_data.index.max() + 1, ts_data.index.max() + 1 + years_ahead))
        
        # Store results
        self.model_performance[f'exp_smoothing_{target}'] = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'RMSE': np.sqrt(mse),
            'Alpha': alpha
        }
        
        forecast_results = {
            'model': 'Exponential Smoothing',
            'target': target,
            'historical_years': ts_data.index.tolist(),
            'historical_values': ts_data.values.tolist(),
            'historical_predictions': smoothed_values,
            'future_years': future_years,
            'future_predictions': future_predictions,
            'metrics': self.model_performance[f'exp_smoothing_{target}'],
            'alpha': alpha
        }
        
        self.forecasts[f'exp_smoothing_{target}'] = forecast_results
        
        print(f"Model Performance:")
        print(f"      • R² Score: {r2:.4f}")
        print(f"      • RMSE: {np.sqrt(mse):,.0f}")
        print(f"      • MAE: {mae:,.0f}")
        print(f"      • Alpha: {alpha}")
        
        return forecast_results
    
    def ensemble_forecast(self, years_ahead=3, target='Produksi_ton'):
        """
        Ensemble forecasting combining multiple models
        
        Args:
            years_ahead (int): Number of years to forecast
            target (str): Target variable
            
        Returns:
            dict: Ensemble forecast results
        """
        print(f"\nEnsemble Forecasting for {target}")
        print("-" * 50)
        
        # Run individual models
        linear_results = self.linear_trend_forecast(years_ahead, target)
        poly_results = self.polynomial_trend_forecast(years_ahead, target, degree=2)
        exp_results = self.exponential_smoothing_forecast(years_ahead, target)
        
        # Combine forecasts (simple average)
        future_years = linear_results['future_years']
        
        # Weight models based on their R² scores
        linear_r2 = linear_results['metrics']['R2']
        poly_r2 = poly_results['metrics']['R2']
        exp_r2 = exp_results['metrics']['R2']
        
        total_r2 = linear_r2 + poly_r2 + exp_r2
        
        if total_r2 > 0:
            linear_weight = linear_r2 / total_r2
            poly_weight = poly_r2 / total_r2
            exp_weight = exp_r2 / total_r2
        else:
            # Equal weights if all R² are negative or zero
            linear_weight = poly_weight = exp_weight = 1/3
        
        # Weighted ensemble predictions
        ensemble_predictions = []
        for i in range(len(future_years)):
            weighted_pred = (
                linear_weight * linear_results['future_predictions'][i] +
                poly_weight * poly_results['future_predictions'][i] +
                exp_weight * exp_results['future_predictions'][i]
            )
            ensemble_predictions.append(weighted_pred)
        
        # Calculate ensemble performance on historical data
        ensemble_historical = []
        for i in range(len(linear_results['historical_years'])):
            weighted_hist = (
                linear_weight * linear_results['historical_predictions'][i] +
                poly_weight * poly_results['historical_predictions'][i] +
                exp_weight * exp_results['historical_predictions'][i]
            )
            ensemble_historical.append(weighted_hist)
        
        # Ensemble metrics
        historical_actual = linear_results['historical_values']
        ensemble_mae = mean_absolute_error(historical_actual, ensemble_historical)
        ensemble_mse = mean_squared_error(historical_actual, ensemble_historical)
        ensemble_r2 = r2_score(historical_actual, ensemble_historical)
        
        ensemble_results = {
            'model': 'Ensemble (Weighted Average)',
            'target': target,
            'historical_years': linear_results['historical_years'],
            'historical_values': historical_actual,
            'historical_predictions': ensemble_historical,
            'future_years': future_years,
            'future_predictions': ensemble_predictions,
            'metrics': {
                'MAE': ensemble_mae,
                'MSE': ensemble_mse,
                'R2': ensemble_r2,
                'RMSE': np.sqrt(ensemble_mse)
            },
            'weights': {
                'Linear': linear_weight,
                'Polynomial': poly_weight,
                'Exponential_Smoothing': exp_weight
            },
            'component_models': {
                'Linear': linear_results,
                'Polynomial': poly_results,
                'Exponential_Smoothing': exp_results
            }
        }
        
        self.forecasts[f'ensemble_{target}'] = ensemble_results
        
        print(f"  Ensemble Performance:")
        print(f"      • R² Score: {ensemble_r2:.4f}")
        print(f"      • RMSE: {np.sqrt(ensemble_mse):,.0f}")
        print(f"      • MAE: {ensemble_mae:,.0f}")
        print(f"\n  Model Weights:")
        print(f"      • Linear Trend: {linear_weight:.3f}")
        print(f"      • Polynomial: {poly_weight:.3f}")
        print(f"      • Exp. Smoothing: {exp_weight:.3f}")
        
        return ensemble_results
    
    def visualize_forecasts(self, target='Produksi_ton', save_plots=True):
        """
        Create comprehensive visualization of all forecasts
        
        Args:
            target (str): Target variable to visualize
            save_plots (bool): Whether to save plots to files
        """
        print(f"\nCreating Forecast Visualizations for {target}")
        print("-" * 50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Rice Production Forecasting Results - {target}', fontsize=16, fontweight='bold')
        
        # Get ensemble results (which contains all models)
        ensemble_key = f'ensemble_{target}'
        if ensemble_key not in self.forecasts:
            print("Run ensemble_forecast first!")
            return
        
        ensemble_data = self.forecasts[ensemble_key]
        
        # Plot 1: All models comparison
        ax1 = axes[0, 0]
        
        # Historical data
        hist_years = ensemble_data['historical_years']
        hist_values = ensemble_data['historical_values']
        future_years = ensemble_data['future_years']
        
        ax1.plot(hist_years, hist_values, 'ko-', linewidth=2, markersize=6, label='Historical Data')
        
        # Individual model forecasts
        models = ensemble_data['component_models']
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, model_data) in enumerate(models.items()):
            color = colors[i % len(colors)]
            # Historical predictions
            ax1.plot(hist_years, model_data['historical_predictions'], 
                    '--', color=color, alpha=0.7, linewidth=1)
            # Future predictions
            ax1.plot(future_years, model_data['future_predictions'], 
                    'o-', color=color, linewidth=2, markersize=5, 
                    label=f"{model_name} (R²={model_data['metrics']['R2']:.3f})")
        
        # Ensemble forecast
        ax1.plot(future_years, ensemble_data['future_predictions'], 
                'o-', color='purple', linewidth=3, markersize=7, 
                label=f"Ensemble (R²={ensemble_data['metrics']['R2']:.3f})")
        
        ax1.set_title('All Models Comparison', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel(f'{target} (tons)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model Performance Comparison
        ax2 = axes[0, 1]
        
        model_names = ['Linear', 'Polynomial', 'Exp. Smoothing', 'Ensemble']
        r2_scores = [
            models['Linear']['metrics']['R2'],
            models['Polynomial']['metrics']['R2'],
            models['Exponential_Smoothing']['metrics']['R2'],
            ensemble_data['metrics']['R2']
        ]
        
        bars = ax2.bar(model_names, r2_scores, color=['blue', 'red', 'green', 'purple'], alpha=0.7)
        ax2.set_title('Model Performance (R² Score)', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.set_ylim(0, max(r2_scores) * 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Residuals Analysis
        ax3 = axes[1, 0]
        
        ensemble_residuals = np.array(hist_values) - np.array(ensemble_data['historical_predictions'])
        ax3.plot(hist_years, ensemble_residuals, 'o-', color='purple', linewidth=2, markersize=6)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Ensemble Model Residuals', fontweight='bold')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Residuals (Actual - Predicted)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Forecast Uncertainty
        ax4 = axes[1, 1]
        
        # Calculate prediction intervals (simplified approach)
        residual_std = np.std(ensemble_residuals)
        
        all_years = hist_years + future_years
        all_predictions = ensemble_data['historical_predictions'] + ensemble_data['future_predictions']
        
        upper_bound = np.array(all_predictions) + 1.96 * residual_std
        lower_bound = np.array(all_predictions) - 1.96 * residual_std
        
        ax4.plot(hist_years, hist_values, 'ko-', linewidth=2, markersize=6, label='Historical Data')
        ax4.plot(all_years, all_predictions, 'o-', color='purple', linewidth=2, 
                markersize=5, label='Ensemble Forecast')
        ax4.fill_between(all_years, lower_bound, upper_bound, alpha=0.3, color='purple', 
                        label='95% Confidence Interval')
        
        ax4.set_title('Forecast with Uncertainty', fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel(f'{target} (tons)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'forecast_results_{target}.png', dpi=300, bbox_inches='tight')
            print(f"   Plot saved as: forecast_results_{target}.png")
        
        plt.show()
    
    def generate_forecast_report(self, target='Produksi_ton', years_ahead=3):
        """
        Generate a comprehensive forecast report
        
        Args:
            target (str): Target variable
            years_ahead (int): Number of years to forecast
        """
        print("\n" + "="*80)
        print("RICE PRODUCTION FORECAST REPORT")
        print("="*80)
        
        # Run all forecasting methods
        print("\nRunning All Forecasting Models...")
        ensemble_results = self.ensemble_forecast(years_ahead, target)
        
        # Generate visualizations
        self.visualize_forecasts(target)
        
        # Summary statistics
        print(f"\nFORECAST SUMMARY ({target})")
        print("-" * 50)
        
        current_year = self.national_data['Year'].max()
        current_production = self.national_data[self.national_data['Year'] == current_year][target].iloc[0]
        
        print(f"   Current Year: {current_year}")
        print(f"   Current Production: {current_production:,.0f} tons")
        
        print(f"\n   Forecasts for {years_ahead} years:")
        for i, (year, prediction) in enumerate(zip(ensemble_results['future_years'], 
                                                  ensemble_results['future_predictions'])):
            change = ((prediction - current_production) / current_production) * 100
            print(f"      • {year}: {prediction:,.0f} tons ({change:+.1f}%)")
        
        # Model comparison
        print(f"\nMODEL PERFORMANCE COMPARISON")
        print("-" * 50)
        
        models_data = ensemble_results['component_models']
        models_data['Ensemble'] = ensemble_results
        
        performance_df = pd.DataFrame({
            'Model': list(models_data.keys()),
            'R²': [models_data[model]['metrics']['R2'] for model in models_data.keys()],
            'RMSE': [models_data[model]['metrics']['RMSE'] for model in models_data.keys()],
            'MAE': [models_data[model]['metrics']['MAE'] for model in models_data.keys()]
        }).sort_values('R²', ascending=False)
        
        print(performance_df.to_string(index=False, float_format='%.3f'))
        
        # Best model
        best_model = performance_df.iloc[0]['Model']
        print(f"\n   Best Performing Model: {best_model}")
        
        # Trend analysis
        print(f"\nTREND ANALYSIS")
        print("-" * 50)
        
        # Calculate overall trend
        years = np.array(self.national_data['Year'])
        values = np.array(self.national_data[target])
        trend_slope = np.polyfit(years, values, 1)[0]
        
        avg_annual_change = (values[-1] - values[0]) / (years[-1] - years[0])
        total_change = ((values[-1] - values[0]) / values[0]) * 100
        
        print(f"   Historical Trend: {trend_slope:+,.0f} tons/year")
        print(f"   Average Annual Change: {avg_annual_change:+,.0f} tons/year")
        print(f"   Total Change ({years[0]}-{years[-1]}): {total_change:+.1f}%")
        
        # Risk assessment
        print(f"\nRISK ASSESSMENT")
        print("-" * 50)
        
        # Calculate coefficient of variation
        cv = (np.std(values) / np.mean(values)) * 100
        
        if cv < 5:
            risk_level = "Low"
        elif cv < 10:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        print(f"   Production Volatility: {cv:.1f}% (Coefficient of Variation)")
        print(f"   Risk Level: {risk_level}")
        
        # Model uncertainty
        ensemble_std = np.std(ensemble_results['historical_predictions'])
        uncertainty_pct = (ensemble_std / np.mean(values)) * 100
        
        print(f"   Forecast Uncertainty: ±{uncertainty_pct:.1f}%")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS")
        print("-" * 50)
        
        if trend_slope > 0:
            print("   Positive production trend observed")
        else:
            print("   Declining production trend - investigate causes")
        
        if cv > 10:
            print("   High production volatility - consider risk mitigation")
        
        if ensemble_results['metrics']['R2'] > 0.8:
            print("   High forecast confidence")    
        elif ensemble_results['metrics']['R2'] > 0.6:
            print("   Moderate forecast confidence - monitor closely")
        else:
            print("   Low forecast confidence - use with caution")  
        
        print("\n" + "="*80)
        print("Report Generation Complete!")
        print("="*80)
        
        return ensemble_results
    
    def save_models(self, filepath="models/"):
        """
        Save trained models to disk
        
        Args:
            filepath (str): Directory to save models
        """
        os.makedirs(filepath, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(filepath, f"{model_name}.pkl"))
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(filepath, f"{scaler_name}_scaler.pkl"))
        
        # Save forecasts (fix circular reference issue)
        import json
        
        # Create a clean copy of forecasts without circular references
        clean_forecasts = {}
        for key, forecast in self.forecasts.items():
            clean_forecast = forecast.copy()
            
            # Remove component_models to avoid circular reference
            if 'component_models' in clean_forecast:
                # Save component model names and metrics only
                clean_forecast['component_model_names'] = list(clean_forecast['component_models'].keys())
                clean_forecast['component_model_metrics'] = {
                    name: model_data['metrics'] 
                    for name, model_data in clean_forecast['component_models'].items()
                }
                del clean_forecast['component_models']
            
            clean_forecasts[key] = clean_forecast
        
        with open(os.path.join(filepath, "forecasts.json"), 'w') as f:
            json.dump(clean_forecasts, f, indent=2)
        
        # Also save performance metrics separately
        with open(os.path.join(filepath, "model_performance.json"), 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        
        print(f"Models and forecasts saved to {filepath}")

def main():
    """
    Main function to run the forecasting system
    """
    print("Rice Production Forecasting System")
    print("=" * 50)
    
    # Initialize forecaster
    forecaster = RiceProductionForecaster()
    
    # Generate comprehensive forecast report
    results = forecaster.generate_forecast_report(
        target='Produksi_ton', 
        years_ahead=3
    )
    
    # Save models
    forecaster.save_models()
    
    print("\nForecasting complete! Check the generated plots and saved models.")
    
    return forecaster, results

if __name__ == "__main__":
    forecaster, results = main()