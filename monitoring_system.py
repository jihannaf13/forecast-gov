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
        else:
            st.error("Failed to load data. Please check if all data files exist.")
            st.stop()
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        st.markdown('<div class="main-header">🌾 Rice Production Monitoring Dashboard</div>', unsafe_allow_html=True)
        
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
        
        # Get forecast for next year
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        next_year_forecast = None
        if ensemble_forecast and 'future_predictions' in ensemble_forecast:
            next_year_forecast = ensemble_forecast['future_predictions'][0]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Current Production (2024)",
                value=f"{latest_production/1e6:.1f}M tons",
                delta=f"{production_change:+.1f}% YoY"
            )
        
        with col2:
            st.metric(
                label="🌾 Harvested Area",
                value=f"{latest_area/1e6:.1f}M ha",
                delta=None
            )
        
        with col3:
            st.metric(
                label="⚡ Productivity",
                value=f"{latest_productivity:.1f} ku/ha",
                delta=None
            )
        
        with col4:
            if next_year_forecast:
                forecast_change = ((next_year_forecast - latest_production) / latest_production) * 100
                st.metric(
                    label="🔮 2025 Forecast",
                    value=f"{next_year_forecast/1e6:.1f}M tons",
                    delta=f"{forecast_change:+.1f}% vs 2024"
                )
            else:
                st.metric(
                    label="🔮 2025 Forecast",
                    value="N/A",
                    delta=None
                )
    
    def create_production_trends(self):
        """Create production trends visualization"""
        st.subheader("📈 Production Trends & Forecasts")
        
        # Prepare data for visualization
        historical_data = self.national_data.copy()
        
        # Get ensemble forecast data
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        
        if ensemble_forecast:
            # Create forecast dataframe
            forecast_years = ensemble_forecast.get('future_years', [])
            forecast_values = ensemble_forecast.get('future_predictions', [])
            
            if forecast_years and forecast_values:
                forecast_df = pd.DataFrame({
                    'Year': forecast_years,
                    'Produksi_ton': forecast_values,
                    'Type': 'Forecast'
                })
                
                # Add type column to historical data
                historical_data['Type'] = 'Historical'
                
                # Combine data
                combined_data = pd.concat([
                    historical_data[['Year', 'Produksi_ton', 'Type']],
                    forecast_df
                ], ignore_index=True)
                
                # Create the plot
                fig = px.line(
                    combined_data, 
                    x='Year', 
                    y='Produksi_ton',
                    color='Type',
                    title='Rice Production: Historical Data vs Forecasts',
                    color_discrete_map={'Historical': '#2E8B57', 'Forecast': '#FF6B6B'}
                )
                
                # Add markers
                fig.update_traces(mode='lines+markers', marker_size=8)
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Production (tons)",
                    hovermode='x unified',
                    height=500,
                    yaxis=dict(tickformat='.0s')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Show model performance
        col1, col2 = st.columns(2)
        
        with col1:
            if ensemble_forecast and 'metrics' in ensemble_forecast:
                metrics = ensemble_forecast['metrics']
                st.markdown("### 🎯 Model Performance")
                st.markdown(f"**R² Score:** {metrics.get('R2', 0):.3f}")
                st.markdown(f"**RMSE:** {metrics.get('RMSE', 0):,.0f} tons")
                st.markdown(f"**MAE:** {metrics.get('MAE', 0):,.0f} tons")
        
        with col2:
            # Calculate trend
            years = self.national_data['Year'].values
            production = self.national_data['Produksi_ton'].values
            trend_slope = np.polyfit(years, production, 1)[0]
            
            st.markdown("### 📊 Trend Analysis")
            if trend_slope < 0:
                st.markdown(f"**Trend:** ⬇️ Declining ({trend_slope/1e6:.2f}M tons/year)")
                st.markdown("**Status:** ⚠️ Requires attention")
            else:
                st.markdown(f"**Trend:** ⬆️ Growing ({trend_slope/1e6:.2f}M tons/year)")
                st.markdown("**Status:** ✅ Positive trend")
    
    def create_provincial_analysis(self):
        """Create provincial analysis"""
        st.subheader("🗺️ Provincial Analysis")
        
        # Get latest year data by province
        latest_year = self.provincial_data['Year'].max()
        latest_provincial = self.provincial_data[self.provincial_data['Year'] == latest_year].copy()
        
        # Top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top 10 Producers (2024)")
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
            st.markdown("#### 📈 Provincial Trends")
            
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
        st.subheader("🚨 Alerts & Monitoring")
        
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
                    'title': '🚨 Significant Production Decline',
                    'message': f'Production declined by {abs(decline_rate):.1f}% from previous year'
                })
            elif decline_rate < 0:
                alerts.append({
                    'type': 'warning',
                    'title': '⚠️ Production Decline',
                    'message': f'Production declined by {abs(decline_rate):.1f}% from previous year'
                })
        
        # Check forecast reliability
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        if ensemble_forecast and 'metrics' in ensemble_forecast:
            r2_score = ensemble_forecast['metrics'].get('R2', 0)
            if r2_score < 0.5:
                alerts.append({
                    'type': 'warning',
                    'title': '📊 Low Model Accuracy',
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
                    'title': '🌾 Significant Area Reduction',
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
                <strong>✅ All Systems Normal</strong><br>
                No critical alerts detected in the monitoring system.
            </div>
            """, unsafe_allow_html=True)
    
    def create_recommendations(self):
        """Create recommendations section"""
        st.subheader("💡 Recommendations")
        
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
                recommendations.append("🎯 **Focus on Production Recovery**: Implement targeted interventions to reverse declining production trends")
            
            if area_change < -2:
                recommendations.append("🌾 **Address Area Reduction**: Investigate causes of harvested area decline and implement land conservation policies")
            
            if productivity_change < 0:
                recommendations.append("⚡ **Improve Productivity**: Invest in agricultural technology, better seeds, and farmer training programs")
            
            if productivity_change > 0 and area_change < 0:
                recommendations.append("📊 **Optimize Land Use**: While productivity is improving, focus on maintaining or expanding cultivated areas")
        
        # Model-based recommendations
        ensemble_forecast = self.forecasts.get('ensemble_Produksi_ton', {})
        if ensemble_forecast and 'metrics' in ensemble_forecast:
            r2_score = ensemble_forecast['metrics'].get('R2', 0)
            if r2_score < 0.7:
                recommendations.append("🔧 **Enhance Forecasting**: Consider incorporating additional variables (weather, policy, economic factors) to improve model accuracy")
        
        # General recommendations
        recommendations.extend([
            "📈 **Continuous Monitoring**: Implement real-time data collection systems for better tracking",
            "🤝 **Stakeholder Engagement**: Collaborate with farmers, local governments, and agricultural experts",
            "🌡️ **Climate Adaptation**: Develop climate-resilient rice varieties and farming practices",
            "💰 **Investment Planning**: Allocate resources based on data-driven insights and forecasts"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

def main():
    """Main Streamlit application"""
    
    # Initialize the monitor
    monitor = RiceProductionMonitor()
    
    # Sidebar
    st.sidebar.title("🌾 Navigation")
    st.sidebar.markdown("---")
    
    # Sidebar information
    st.sidebar.markdown("### 📊 Data Overview")
    st.sidebar.info(f"""
    **Data Period:** 2018-2024
    **Provinces:** 39
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)
    
    st.sidebar.markdown("### 🔧 Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 min)", value=False)
    show_raw_data = st.sidebar.checkbox("Show raw data", value=False)
    
    # Main content
    monitor.create_overview_metrics()
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Provincial", "🚨 Alerts", "💡 Recommendations"])
    
    with tab1:
        monitor.create_production_trends()
    
    with tab2:
        monitor.create_provincial_analysis()
    
    with tab3:
        monitor.create_alerts_system()
    
    with tab4:
        monitor.create_recommendations()
    
    # Show raw data if requested
    if show_raw_data:
        st.markdown("---")
        st.subheader("📋 Raw Data")
        
        data_choice = st.selectbox("Select dataset:", 
                                 ["National Timeseries", "Provincial Data", "Forecast Results"])
        
        if data_choice == "National Timeseries":
            st.dataframe(monitor.national_data)
        elif data_choice == "Provincial Data":
            st.dataframe(monitor.provincial_data)
        elif data_choice == "Forecast Results":
            st.json(monitor.forecasts)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        🌾 Rice Production Monitoring Dashboard | Built with Streamlit | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()