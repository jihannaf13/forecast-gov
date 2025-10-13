import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Rice Production Monitoring Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .alert-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        # Load historical data
        national_data = pd.read_csv('processed_data/national_timeseries.csv')
        provincial_data = pd.read_csv('processed_data/complete_cleaned_data.csv')
        top_provinces_data = pd.read_csv('processed_data/top_provinces_data.csv')
        
        # Load forecast data
        with open('models/forecasts.json', 'r') as f:
            forecasts = json.load(f)
        
        with open('models/model_performance.json', 'r') as f:
            model_performance = json.load(f)
        
        return national_data, provincial_data, top_provinces_data, forecasts, model_performance
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

class RiceProductionMonitor:
    def __init__(self):
        # Load data using the cached function
        data = load_data()
        if data[0] is not None:
            self.national_data, self.provincial_data, self.top_provinces_data, self.forecasts, self.model_performance = data
            # Identify the best performing model
            self.best_model_key = self.get_best_model()
        else:
            st.error("Failed to load data. Please check if all data files exist.")
            st.stop()
    
    def get_best_model(self):
        """Identify the best performing model based on R² score"""
        best_r2 = -1
        best_model = 'ensemble_Produksi_ton'  # Default fallback
        
        # Check individual models
        for model_key, performance in self.model_performance.items():
            if 'R2' in performance and performance['R2'] > best_r2:
                best_r2 = performance['R2']
                best_model = model_key
        
        # Check ensemble model if available
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        if ensemble_forecast and 'metrics' in ensemble_forecast:
            ensemble_r2 = ensemble_forecast['metrics'].get('R2', 0)
            if ensemble_r2 > best_r2:
                best_r2 = ensemble_r2
                best_model = 'ensemble_Produksi_ton'
        
        return best_model
    
    def get_best_forecast_data(self):
        """Get forecast data from the best performing model"""
        if self.best_model_key in self.forecasts:
            return self.forecasts[self.best_model_key]
        elif self.best_model_key in self.model_performance:
            # For individual models, we need to find the corresponding forecast
            for forecast_key in self.forecasts:
                if self.best_model_key in forecast_key:
                    return self.forecasts[forecast_key]
        
        # Fallback to ensemble
        return self.forecasts.get('ensemble_Produksi_ton', {})
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        st.markdown('<div class="main-header">Rice Production Monitoring Dashboard</div>', unsafe_allow_html=True)
        
        # Get latest data
        latest_year = self.national_data['Year'].max()
        latest_production = self.national_data[self.national_data['Year'] == latest_year]['Produksi_ton'].iloc[0]
        latest_area = self.national_data[self.national_data['Year'] == latest_year]['Luas_Panen_ha'].iloc[0]
        latest_productivity = self.national_data[self.national_data['Year'] == latest_year]['Produktivitas_ku_ha'].iloc[0]
        
        # Calculate year-over-year changes
        prev_year_data = self.national_data[self.national_data['Year'] == latest_year - 1]
        if not prev_year_data.empty:
            prev_production = prev_year_data['Produksi_ton'].iloc[0]
            production_change = ((latest_production - prev_production) / prev_production) * 100
        else:
            production_change = 0
        
        # Get forecast for next year using best model
        best_forecast = self.get_best_forecast_data()
        next_year_forecast = None
        if best_forecast and 'future_predictions' in best_forecast:
            next_year_forecast = best_forecast['future_predictions'][0]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Production (2024)",
                value=f"{latest_production/1e6:.1f}M tons",
                delta=f"{production_change:+.1f}% YoY"
            )
        
        with col2:
            st.metric(
                label="Harvested Area",
                value=f"{latest_area/1e6:.1f}M ha",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Productivity",
                value=f"{latest_productivity:.1f} ku/ha",
                delta=None
            )
        
        with col4:
            if next_year_forecast:
                forecast_change = ((next_year_forecast - latest_production) / latest_production) * 100
                # Get best model name for display
                best_model_name = self.get_model_display_name(self.best_model_key)
                st.metric(
                    label=f"2025 Forecast ({best_model_name})",
                    value=f"{next_year_forecast/1e6:.1f}M tons",
                    delta=f"{forecast_change:+.1f}% vs 2024"
                )
            else:
                st.metric(
                    label="2025 Forecast",
                    value="N/A",
                    delta=None
                )
    
    def get_model_display_name(self, model_key):
        """Get a user-friendly display name for the model"""
        if 'linear_trend' in model_key:
            return 'Linear'
        elif 'poly_trend' in model_key:
            return 'Polynomial'
        elif 'exp_smoothing' in model_key:
            return 'Exp. Smoothing'
        elif 'ensemble' in model_key:
            return 'Ensemble'
        elif 'random_forest' in model_key:
            return 'Random Forest'
        else:
            return 'Best Model'
    
    def create_production_trends(self):
        """Create production trends visualization"""
        st.subheader("Production Trends & Forecasts")
        
        # Show best model information
        best_forecast = self.get_best_forecast_data()
        best_model_name = self.get_model_display_name(self.best_model_key)
        
        # Get best model performance
        if self.best_model_key in self.model_performance:
            best_r2 = self.model_performance[self.best_model_key]['R2']
        elif best_forecast and 'metrics' in best_forecast:
            best_r2 = best_forecast['metrics']['R2']
        else:
            best_r2 = 0
        
        st.info(f"Using **{best_model_name}** model (R² = {best_r2:.3f}) - automatically selected as the best performing model")
        
        # Create comprehensive 4-chart analysis
        if best_forecast:
            self.create_comprehensive_forecast_analysis(best_forecast, best_model_name)
        
        # Show model performance comparison
        self.show_model_comparison()
    
    def create_comprehensive_forecast_analysis(self, best_forecast, best_model_name):
        """Create comprehensive 4-chart forecast analysis similar to fix_forecast_gaps.py"""
        
        # Get data
        hist_years = best_forecast.get('historical_years', [])
        hist_values = best_forecast.get('historical_values', [])
        future_years = best_forecast.get('future_years', [])
        future_predictions = best_forecast.get('future_predictions', [])
        
        if not all([hist_years, hist_values, future_years, future_predictions]):
            st.error("Insufficient forecast data for comprehensive analysis")
            return
        
        # Create 2x2 subplot layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Chart 1: Continuous Production Trend with Transition
            self.create_continuous_trend_chart(hist_years, hist_values, future_years, future_predictions, best_model_name)
            
            # Chart 3: Year-over-Year Growth Rate
            self.create_growth_rate_chart(hist_years, hist_values, future_years, future_predictions)
        
        with col2:
            # Chart 2: Smooth Trend Analysis
            self.create_smooth_trend_chart(hist_years, hist_values, future_years, future_predictions)
            
            # Chart 4: Forecast with Confidence Intervals
            self.create_confidence_interval_chart(best_forecast, hist_years, hist_values, future_years, future_predictions)
    
    def create_continuous_trend_chart(self, hist_years, hist_values, future_years, future_predictions, model_name):
        """Create continuous production trend chart with transition line"""
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_years,
            y=hist_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8, color='white', line=dict(color='#2E8B57', width=2))
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=future_years,
            y=future_predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8, color='white', line=dict(color='#FF6B6B', width=2))
        ))
        
        # Transition line
        transition_x = [hist_years[-1], future_years[0]]
        transition_y = [hist_values[-1], future_predictions[0]]
        fig.add_trace(go.Scatter(
            x=transition_x,
            y=transition_y,
            mode='lines',
            name='Transition',
            line=dict(color='gray', width=2, dash='dash'),
            opacity=0.7
        ))
        
        # Add vertical line to separate historical from forecast
        fig.add_vline(
            x=max(hist_years), 
            line_dash="dot", 
            line_color="gray",
            opacity=0.5,
            annotation_text="Forecast Start",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=f'Continuous Production Trend - {model_name} Model',
            xaxis_title='Year',
            yaxis_title='Production (tons)',
            height=400,
            hovermode='x unified',
            yaxis=dict(tickformat='.2s')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_smooth_trend_chart(self, hist_years, hist_values, future_years, future_predictions):
        """Create smooth trend analysis chart"""
        
        # Combine all data
        all_years = hist_years + future_years
        all_values = hist_values + future_predictions
        
        # Create smooth trend line using polynomial fit
        z = np.polyfit(all_years, all_values, 2)
        p = np.poly1d(z)
        smooth_years = np.linspace(min(all_years), max(all_years), 100)
        smooth_values = p(smooth_years)
        
        fig = go.Figure()
        
        # Smooth trend line
        fig.add_trace(go.Scatter(
            x=smooth_years,
            y=smooth_values,
            mode='lines',
            name='Smooth Trend',
            line=dict(color='green', width=3),
            opacity=0.7
        ))
        
        # Historical points
        fig.add_trace(go.Scatter(
            x=hist_years,
            y=hist_values,
            mode='markers',
            name='Historical',
            marker=dict(size=10, color='#2E8B57', opacity=0.8)
        ))
        
        # Forecast points
        fig.add_trace(go.Scatter(
            x=future_years,
            y=future_predictions,
            mode='markers',
            name='Forecast',
            marker=dict(size=10, color='#FF6B6B', opacity=0.8)
        ))
        
        fig.update_layout(
            title='Smooth Trend Analysis',
            xaxis_title='Year',
            yaxis_title='Production (tons)',
            height=400,
            yaxis=dict(tickformat='.2s')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_growth_rate_chart(self, hist_years, hist_values, future_years, future_predictions):
        """Create year-over-year growth rate chart"""
        
        # Combine all data
        all_years = hist_years + future_years
        all_values = hist_values + future_predictions
        
        # Calculate growth rates
        growth_rates = []
        growth_years = []
        colors = []
        
        for i in range(1, len(all_values)):
            growth_rate = ((all_values[i] - all_values[i-1]) / all_values[i-1]) * 100
            growth_rates.append(growth_rate)
            growth_years.append(all_years[i])
            # Color coding: blue for historical, red for forecast
            colors.append('#2E8B57' if all_years[i] <= max(hist_years) else '#FF6B6B')
        
        fig = go.Figure()
        
        # Growth rate bars
        fig.add_trace(go.Bar(
            x=growth_years,
            y=growth_rates,
            name='Growth Rate',
            marker_color=colors,
            opacity=0.7,
            text=[f'{rate:.1f}%' for rate in growth_rates],
            textposition='outside'
        ))
        
        # Add average line
        avg_growth = np.mean(growth_rates)
        fig.add_hline(
            y=avg_growth,
            line_dash="dash",
            line_color="green",
            opacity=0.7,
            annotation_text=f"Average: {avg_growth:.1f}%"
        )
        
        # Add zero line
        fig.add_hline(y=0, line_color="black", opacity=0.5)
        
        fig.update_layout(
            title='Year-over-Year Growth Rate',
            xaxis_title='Year',
            yaxis_title='Growth Rate (%)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_confidence_interval_chart(self, best_forecast, hist_years, hist_values, future_years, future_predictions):
        """Create forecast with confidence intervals chart"""
        
        # Calculate confidence intervals
        if 'historical_predictions' in best_forecast:
            hist_predictions = best_forecast['historical_predictions']
            residuals = np.array(hist_values) - np.array(hist_predictions)
            residual_std = np.std(residuals)
        else:
            # Fallback: use 5% of mean as standard deviation
            residual_std = np.mean(hist_values) * 0.05
        
        # Create confidence intervals for forecast (95% confidence)
        upper_bound = np.array(future_predictions) + 1.96 * residual_std
        lower_bound = np.array(future_predictions) - 1.96 * residual_std
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_years,
            y=hist_values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=6)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_years,
            y=future_predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=6)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_years + future_years[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            hoverinfo="skip"
        ))
        
        # Connect historical to forecast smoothly
        fig.add_trace(go.Scatter(
            x=[hist_years[-1], future_years[0]],
            y=[hist_values[-1], future_predictions[0]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            opacity=0.7,
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            title='Forecast with Confidence Intervals',
            xaxis_title='Year',
            yaxis_title='Production (tons)',
            height=400,
            yaxis=dict(tickformat='.2s')
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_model_comparison(self):
        """Show comparison of all available models"""
        st.markdown("### Model Performance Comparison")
        
        # Prepare model comparison data
        model_data = []
        
        # Add individual models
        for model_key, performance in self.model_performance.items():
            model_name = self.get_model_display_name(model_key)
            model_data.append({
                'Model': model_name,
                'R² Score': performance.get('R2', 0),
                'RMSE': performance.get('RMSE', 0),
                'MAE': performance.get('MAE', 0),
                'Best': model_key == self.best_model_key
            })
        
        # Add ensemble model if available
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        if ensemble_forecast and 'metrics' in ensemble_forecast:
            metrics = ensemble_forecast['metrics']
            model_data.append({
                'Model': 'Ensemble',
                'R² Score': metrics.get('R2', 0),
                'RMSE': metrics.get('RMSE', 0),
                'MAE': metrics.get('MAE', 0),
                'Best': self.best_model_key == 'ensemble_Produksi_ton'
            })
        
        # Sort by R² score
        model_data.sort(key=lambda x: x['R² Score'], reverse=True)
        
        # Create comparison chart
        df_comparison = pd.DataFrame(model_data)
        
        fig = px.bar(
            df_comparison,
            x='Model',
            y='R² Score',
            title='Model Performance Comparison (R² Score)',
            color='Best',
            color_discrete_map={True: '#FF6B6B', False: '#87CEEB'}
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="R² Score"
        )
        
        # Add value labels on bars
        for i, row in df_comparison.iterrows():
            fig.add_annotation(
                x=row['Model'],
                y=row['R² Score'] + 0.01,
                text=f"{row['R² Score']:.3f}",
                showarrow=False,
                font=dict(size=10, color='black')
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed comparison table and trend analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Detailed Model Metrics")
            display_df = df_comparison.copy()
            display_df['R² Score'] = display_df['R² Score'].round(3)
            display_df['RMSE'] = display_df['RMSE'].round(0).astype(int)
            display_df['MAE'] = display_df['MAE'].round(0).astype(int)
            display_df['Status'] = display_df['Best'].apply(lambda x: '⭐ Best' if x else '')
            display_df = display_df.drop('Best', axis=1)
            st.dataframe(display_df, use_container_width=True)
        
        with col2:
            st.markdown("#### Trend Analysis")
            # Calculate trend
            years = self.national_data['Year'].values
            production = self.national_data['Produksi_ton'].values
            trend_slope = np.polyfit(years, production, 1)[0]
            
            if trend_slope < 0:
                st.markdown(f"**Historical Trend:** Declining ({trend_slope/1e6:.2f}M tons/year)")
                st.markdown("**Status:** ⚠️ Requires attention")
            else:
                st.markdown(f"**Historical Trend:** Growing ({trend_slope/1e6:.2f}M tons/year)")
                st.markdown("**Status:** ✅ Positive trend")
            
            # Show best model performance
            best_forecast = self.get_best_forecast_data()
            best_model_name = self.get_model_display_name(self.best_model_key)
            
            if best_forecast and 'metrics' in best_forecast:
                metrics = best_forecast['metrics']
                st.markdown(f"**Best Model:** {best_model_name}")
                st.markdown(f"**R² Score:** {metrics.get('R2', 0):.3f}")
                st.markdown(f"**RMSE:** {metrics.get('RMSE', 0):,.0f} tons")
                st.markdown(f"**MAE:** {metrics.get('MAE', 0):,.0f} tons")
            elif self.best_model_key in self.model_performance:
                metrics = self.model_performance[self.best_model_key]
                st.markdown(f"**Best Model:** {best_model_name}")
                st.markdown(f"**R² Score:** {metrics.get('R2', 0):.3f}")
                st.markdown(f"**RMSE:** {metrics.get('RMSE', 0):,.0f} tons")
                st.markdown(f"**MAE:** {metrics.get('MAE', 0):,.0f} tons")

    def create_provincial_analysis(self):
        """Create provincial analysis"""
        st.subheader("Provincial Analysis")
        
        # Get latest year data by province
        latest_year = self.provincial_data['Year'].max()
        latest_provincial = self.provincial_data[self.provincial_data['Year'] == latest_year].copy()
        
        # Top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 Producers (2024)")
            top_producers = latest_provincial.nlargest(10, 'Produksi_ton')[['Province', 'Produksi_ton']]
            top_producers['Produksi_ton'] = top_producers['Produksi_ton'] / 1e6  # Convert to millions
            
            fig_top = px.bar(
                top_producers, 
                x='Produksi_ton', 
                y='Province',
                orientation='h',
                title='Top 10 Rice Producing Provinces',
                color='Produksi_ton',
                color_continuous_scale='Greens'
            )
            fig_top.update_layout(
                height=400, 
                showlegend=False,
                xaxis_title="Production (Million tons)"
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Provincial trends over time
            st.markdown("#### Provincial Trends")
            
            # Get top 5 provinces for trend analysis
            top_5_provinces = latest_provincial.nlargest(5, 'Produksi_ton')['Province'].tolist()
            trend_data = self.provincial_data[self.provincial_data['Province'].isin(top_5_provinces)]
            
            fig_trends = px.line(
                trend_data,
                x='Year',
                y='Produksi_ton',
                color='Province',
                title='Production Trends - Top 5 Provinces'
            )
            fig_trends.update_layout(
                height=400,
                yaxis=dict(tickformat='.0s')
            )
            st.plotly_chart(fig_trends, use_container_width=True)
    
    def create_alerts_system(self):
        """Create alerts and monitoring system"""
        st.subheader("Alerts & Monitoring")
        
        alerts = []
        
        # Check production decline
        recent_years = self.national_data.tail(3)
        if len(recent_years) >= 2:
            latest_production = recent_years.iloc[-1]['Produksi_ton']
            prev_production = recent_years.iloc[-2]['Produksi_ton']
            decline_rate = ((latest_production - prev_production) / prev_production) * 100
            
            if decline_rate < -2:
                alerts.append({
                    'type': 'danger',
                    'title': 'Significant Production Decline',
                    'message': f'Production declined by {abs(decline_rate):.1f}% from previous year'
                })
            elif decline_rate < 0:
                alerts.append({
                    'type': 'warning',
                    'title': 'Production Decline',
                    'message': f'Production declined by {abs(decline_rate):.1f}% from previous year'
                })
        
        # Check forecast reliability
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        if ensemble_forecast and 'metrics' in ensemble_forecast:
            r2_score = ensemble_forecast['metrics'].get('R2', 0)
            if r2_score < 0.5:
                alerts.append({
                    'type': 'warning',
                    'title': 'Low Model Accuracy',
                    'message': f'Model R² score is {r2_score:.3f}, consider model improvement'
                })
        
        # Check area reduction
        if len(recent_years) >= 2:
            latest_area = recent_years.iloc[-1]['Luas_Panen_ha']
            prev_area = recent_years.iloc[-2]['Luas_Panen_ha']
            area_change = ((latest_area - prev_area) / prev_area) * 100
            
            if area_change < -3:
                alerts.append({
                    'type': 'danger',
                    'title': 'Significant Area Reduction',
                    'message': f'Harvested area reduced by {abs(area_change):.1f}% from previous year'
                })
        
        # Display alerts
        if alerts:
            for alert in alerts:
                alert_class = f"alert-{alert['type']}"
                st.markdown(f"""
                <div class="alert-box {alert_class}">
                    <strong>{alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>All Systems Normal</strong><br>
                No critical alerts detected in the monitoring system.
            </div>
            """, unsafe_allow_html=True)
    
    def create_recommendations(self):
        """Create recommendations section"""
        st.subheader("Recommendations")
        
        # Analyze trends and provide recommendations
        latest_data = self.national_data.iloc[-1]
        prev_data = self.national_data.iloc[-2] if len(self.national_data) > 1 else None
        
        recommendations = []
        
        if prev_data is not None:
            # Production trend
            prod_change = ((latest_data['Produksi_ton'] - prev_data['Produksi_ton']) / prev_data['Produksi_ton']) * 100
            area_change = ((latest_data['Luas_Panen_ha'] - prev_data['Luas_Panen_ha']) / prev_data['Luas_Panen_ha']) * 100
            productivity_change = ((latest_data['Produktivitas_ku_ha'] - prev_data['Produktivitas_ku_ha']) / prev_data['Produktivitas_ku_ha']) * 100
            
            if prod_change < 0:
                recommendations.append("**Focus on Production Recovery**: Implement targeted interventions to reverse declining production trends")
            
            if area_change < -2:
                recommendations.append("**Address Area Reduction**: Investigate causes of harvested area decline and implement land conservation policies")
            
            if productivity_change < 0:
                recommendations.append("**Improve Productivity**: Invest in agricultural technology, better seeds, and farmer training programs")
            
            if productivity_change > 0 and area_change < 0:
                recommendations.append("**Optimize Land Use**: While productivity is improving, focus on maintaining or expanding cultivated areas")
        
        # Model-based recommendations
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        if ensemble_forecast and 'metrics' in ensemble_forecast:
            r2_score = ensemble_forecast['metrics'].get('R2', 0)
            if r2_score < 0.7:
                recommendations.append("**Enhance Forecasting**: Consider incorporating additional variables (weather, policy, economic factors) to improve model accuracy")
        
        # General recommendations
        recommendations.extend([
            "**Continuous Monitoring**: Implement real-time data collection systems for better tracking",
            "**Stakeholder Engagement**: Collaborate with farmers, local governments, and agricultural experts",
            "**Climate Adaptation**: Develop climate-resilient rice varieties and farming practices",
            "**Investment Planning**: Allocate resources based on data-driven insights and forecasts"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

        # Rice Allocation Plan (Fair Distribution)
        st.subheader("Rice Allocation Plan (Fair Distribution)")
        try:
            # Ensure we have at least two years of provincial data
            years = sorted(self.provincial_data['Year'].dropna().unique())
            if len(years) < 2:
                st.info("Insufficient provincial time series (need at least 2 years) to propose inter-province allocation.")
                return
            
            latest_year = years[-1]
            prev_year = years[-2]
            
            latest_df = self.provincial_data[self.provincial_data['Year'] == latest_year][['Province', 'Produksi_ton']].rename(columns={'Produksi_ton': 'latest_production'})
            prev_df = self.provincial_data[self.provincial_data['Year'] == prev_year][['Province', 'Produksi_ton']].rename(columns={'Produksi_ton': 'prev_production'})
            
            # Merge only provinces present in both years to avoid NaN issues
            yoy = latest_df.merge(prev_df, on='Province', how='inner')
            yoy['change_ton'] = yoy['latest_production'] - yoy['prev_production']
            
            # Deficits: negative change (needs rice), Surpluses: positive change (can provide rice)
            deficits = yoy[yoy['change_ton'] < 0].copy()
            surpluses = yoy[yoy['change_ton'] > 0].copy()
            
            if deficits.empty:
                st.info("No provinces show a year-over-year production decline. No deficit-based allocation needed.")
                return
            if surpluses.empty:
                st.info("No provinces show a year-over-year production increase. No surplus available for allocation.")
                return
            
            deficits['deficit_ton'] = (-deficits['change_ton']).astype(float)
            surpluses['surplus_ton'] = (surpluses['change_ton']).astype(float)
            
            total_deficit = deficits['deficit_ton'].sum()
            total_surplus = surpluses['surplus_ton'].sum()
            available_for_allocation = min(total_surplus, total_deficit)
            
            # Fair allocation: proportional to each province's share of total deficit
            deficits['allocation_share'] = deficits['deficit_ton'] / total_deficit
            deficits['allocation_ton'] = (deficits['allocation_share'] * available_for_allocation).round(0)
            deficits['allocation_share_pct'] = (deficits['allocation_share'] * 100).round(2)
            
            # Recommended outflows from surplus provinces: proportional to surplus share, capped by available_for_allocation
            surpluses['outflow_share'] = surpluses['surplus_ton'] / total_surplus
            surpluses['outflow_ton'] = (surpluses['outflow_share'] * available_for_allocation).round(0)
            surpluses['outflow_share_pct'] = (surpluses['outflow_share'] * 100).round(2)
            
            # Display summary
            st.markdown(f"• Latest year: {latest_year}, Previous year: {prev_year}")
            st.markdown(f"• Total deficit: {int(total_deficit):,} tons")
            st.markdown(f"• Total surplus: {int(total_surplus):,} tons")
            st.markdown(f"• Proposed allocation volume: {int(available_for_allocation):,} tons (capped by min(surplus, deficit))")
            
            # Show allocation to deficit provinces
            alloc_view = deficits[['Province', 'prev_production', 'latest_production', 'deficit_ton', 'allocation_ton', 'allocation_share_pct']].sort_values('allocation_ton', ascending=False)
            alloc_view.columns = ['Province', 'Prev Prod (ton)', 'Latest Prod (ton)', 'Deficit (ton)', 'Allocation (ton)', 'Allocation Share (%)']
            st.markdown("Recommended Allocation to Provinces (by need):")
            st.dataframe(
                alloc_view.style.format({
                    'Prev Prod (ton)': '{:,.0f}',
                    'Latest Prod (ton)': '{:,.0f}',
                    'Deficit (ton)': '{:,.0f}',
                    'Allocation (ton)': '{:,.0f}',
                    'Allocation Share (%)': '{:.2f}%'
                }),
                width='stretch'
            )
            
            # Show recommended outflows from surplus provinces
            outflow_view = surpluses[['Province', 'prev_production', 'latest_production', 'surplus_ton', 'outflow_ton', 'outflow_share_pct']].sort_values('outflow_ton', ascending=False)
            outflow_view.columns = ['Province', 'Prev Prod (ton)', 'Latest Prod (ton)', 'Surplus (ton)', 'Recommended Outflow (ton)', 'Outflow Share (%)']
            st.markdown("Recommended Outflows from Surplus Provinces:")
            st.dataframe(
                outflow_view.style.format({
                    'Prev Prod (ton)': '{:,.0f}',
                    'Latest Prod (ton)': '{:,.0f}',
                    'Surplus (ton)': '{:,.0f}',
                    'Recommended Outflow (ton)': '{:,.0f}',
                    'Outflow Share (%)': '{:.2f}%'
                }),
                width='stretch'
            )
            
            # Guidance text
            st.markdown(
                "- Allocation is calculated fairly, proportional to each deficit province’s share of total deficit, "
                "and outflows are proportional to surplus capacity. "
                "Adjust logistically for transportation costs, storage, and regional priorities."
            )
        except Exception as e:
            st.warning(f"Could not compute fair allocation due to an error: {e}")
            
    def create_developer_dashboard(self):
        """Developer view: model metrics, forecast internals, and raw data access"""
        st.header("Developer View")
        
        dev_tab1, dev_tab2, dev_tab3 = st.tabs(["Model & Metrics", "Forecast Internals", "Data"])
        
        with dev_tab1:
            st.subheader("Model Performance")
            # Build a performance dataframe
            perf_rows = []
            for model_key, metrics in self.model_performance.items():
                perf_rows.append({
                    'Model': model_key,
                    'R²': metrics.get('R2', 0.0),
                    'RMSE (ton)': metrics.get('RMSE', 0.0),
                    'MAE (ton)': metrics.get('MAE', 0.0),
                })
            if perf_rows:
                perf_df = pd.DataFrame(perf_rows).sort_values('R²', ascending=False)
                st.dataframe(
                    perf_df.style.format({
                        'R²': '{:.3f}',
                        'RMSE (ton)': '{:,.0f}',
                        'MAE (ton)': '{:,.0f}',
                    }),
                    width='stretch'
                )
            else:
                st.info("No model performance data available.")
            
            # Ensemble summary if present
            ensemble = self.forecasts.get('ensemble_Produksi_ton', {})
            if ensemble and 'metrics' in ensemble:
                em = ensemble['metrics']
                st.markdown(f"**Ensemble R²:** {em.get('R2', 0):.3f}")
                st.markdown(f"**Ensemble RMSE:** {em.get('RMSE', 0):,.0f} tons")
                st.markdown(f"**Ensemble MAE:** {em.get('MAE', 0):,.0f} tons")
        
        with dev_tab2:
            st.subheader("Forecast Visuals & Internals")
            best_model_name = getattr(self, 'best_model_key', 'Ensemble')
            best_forecast = self.get_best_forecast_data()
            if best_forecast:
                # Comprehensive charts
                self.create_comprehensive_forecast_analysis(best_forecast, best_model_name)
            else:
                st.warning("Best forecast data not available.")
            
            st.markdown("Forecast Results (JSON)")
            st.json(self.forecasts)
        
        with dev_tab3:
            st.subheader("Raw Data")
            data_choice = st.selectbox("Select dataset:", 
                                       ["National Timeseries", "Provincial Data", "Forecast Results"])
            if data_choice == "National Timeseries":
                st.dataframe(self.national_data, width='stretch')
            elif data_choice == "Provincial Data":
                st.dataframe(self.provincial_data, width='stretch')
            else:
                st.json(self.forecasts)

def main():
    """Main Streamlit application"""
    
    # Initialize the monitor
    monitor = RiceProductionMonitor()
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Sidebar information
    st.sidebar.markdown("### Data Overview")
    st.sidebar.info(f"""
    **Data Period:** 2018-2024
    **Provinces:** 39
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)
    
    st.sidebar.markdown("### Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 min)", value=False)
    show_raw_data = st.sidebar.checkbox("Show raw data", value=False)
    
    # Role-based view selector
    view_mode = st.sidebar.radio("View Mode", ["Executive", "Developer"], index=0)
    
    if view_mode == "Executive":
        # Executive view: KPIs, trends, alerts, recommendations
        monitor.create_overview_metrics()  # Render header + KPI cards only once
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Provincial", "Alerts", "Recommendations"])
        with tab1:
            monitor.create_production_trends()
        with tab2:
            monitor.create_provincial_analysis()
        with tab3:
            monitor.create_alerts_system()
        with tab4:
            monitor.create_recommendations()
    else:
        # Developer view
        monitor.create_developer_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Rice Production Monitoring Dashboard | Built with Streamlit | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()