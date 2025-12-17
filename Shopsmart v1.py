import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D 

# --- 1. CONFIGURATION ---
plt.switch_backend('Agg') 
# Set a dark background fit for burgundy
plt.rcParams['figure.facecolor'] = '#2A0A0A'
plt.rcParams['axes.facecolor'] = '#2A0A0A'
plt.rcParams['text.color'] = '#E8D0D0'
plt.rcParams['axes.labelcolor'] = '#E8D0D0'
plt.rcParams['xtick.color'] = '#E8D0D0'
plt.rcParams['ytick.color'] = '#E8D0D0'
plt.rcParams['grid.color'] = '#8C3A3A'
plt.rcParams['grid.alpha'] = 0.3

# --- 2. DATA ENGINE ---
def generate_data(n_customers, business_type):
    n = int(n_customers)
    if business_type == "Retail (High Vol)":
        avg_spend = 50; freq_scale = 30; spend_dist = 1.5
    elif business_type == "Luxury (High Value)":
        avg_spend = 1200; freq_scale = 180; spend_dist = 0.8
    else: # SaaS
        avg_spend = 150; freq_scale = 45; spend_dist = 5.0
        
    days = np.random.exponential(freq_scale, n).astype(int)
    base_date = np.datetime64('today')
    dates = base_date - days.astype('timedelta64[D]')
    spend = np.random.gamma(shape=spend_dist, scale=avg_spend/spend_dist, size=n)
    
    return pd.DataFrame({
        'CustomerID': np.arange(1000, 1000 + n),
        'InvoiceDate': dates,
        'TotalSpend': spend,
        'InvoiceNo': np.arange(n)
    })

def get_data(uploaded_file, n_customers, business_type):
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str): df = pd.read_csv(uploaded_file)
            else: df = pd.read_csv(uploaded_file.name)
                
            cols = {c.lower().strip().replace('_', ''): c for c in df.columns}
            mapping = {}
            for t in ['invoicedate', 'date']: 
                if t in cols: mapping[cols[t]] = 'InvoiceDate'
            for t in ['totalspend', 'amount', 'sales']: 
                if t in cols: mapping[cols[t]] = 'TotalSpend'
            
            df.rename(columns=mapping, inplace=True)
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df.dropna(subset=['InvoiceDate'], inplace=True)
            
            if 'TotalSpend' not in df.columns: df['TotalSpend'] = np.random.uniform(10, 100, len(df))
            if 'CustomerID' not in df.columns: df['CustomerID'] = np.arange(len(df))
            if 'InvoiceNo' not in df.columns: df['InvoiceNo'] = range(len(df))
            return df
        except Exception as e:
            print(f"CSV Error: {e}")
            return pd.DataFrame()
    else:
        return generate_data(n_customers, business_type)

# --- 3. ANALYTICS CORE ---
def analyze_business(csv_file, business_type, n_cust, k_clusters, churn_days, budget, channel, discount):
    try:
        # A. LOAD
        df = get_data(csv_file, n_cust, business_type)
        if df.empty: return "<h3>Error: Could not load data.</h3>", "", None, None, None, None, None

        # B. RFM
        snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot - x.max()).days,
            'InvoiceNo': 'count',
            'TotalSpend': 'sum'
        }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'}).reset_index()

        # C. CLUSTERING
        X = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=int(k_clusters), random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_avg = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False).index
        names = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Iron', 'Lead']
        mapping = {old: names[i] if i < len(names) else f'Tier {i}' for i, old in enumerate(cluster_avg)}
        rfm['Segment'] = rfm['Cluster'].map(mapping)

        # D. CHURN
        rfm['Churned'] = (rfm['Recency'] > churn_days).astype(int)
        churn_rate = rfm['Churned'].mean()
        
        if len(rfm['Churned'].unique()) > 1:
            model = LogisticRegression(max_iter=200)
            X_m = rfm[['Frequency', 'Monetary', 'Recency']]
            y_m = rfm['Churned']
            model.fit(X_m, y_m)

        # E. ROI
        at_risk = rfm[rfm['Churned'] == 1]['Monetary'].sum()
        
        eff = 0.18 if "Social" in channel else (0.35 if "Influencer" in channel else 0.10)
        cost_factor = 1.0 if "Social" in channel else (2.5 if "Influencer" in channel else 0.2)
            
        conversion = min(0.50, eff * (1 + (discount/20)))
        saved = at_risk * conversion
        real_cost = budget * cost_factor + (saved * (discount/100))
        net = saved - real_cost
        roi = (net / real_cost) * 100 if real_cost > 0 else 0

        # --- F. OUTPUTS ---
        c_col = "#FF4D4D" if churn_rate > 0.25 else "#D4AF37"
        r_col = "#D4AF37" if net > 0 else "#FF4D4D"
        
        html = f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">TOTAL REVENUE</div>
                <div class="kpi-val">${rfm['Monetary'].sum():,.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">CHURN RISK</div>
                <div class="kpi-val" style="color:{c_col}">{churn_rate*100:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">ROI ({channel.split()[0]})</div>
                <div class="kpi-val" style="color:{r_col}">{roi:.0f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">NET PROFIT</div>
                <div class="kpi-val" style="color:{r_col}">${net:,.0f}</div>
            </div>
        </div>
        """

        status_text = "PROFITABLE" if net > 0 else "LOSING MONEY"
        strategy = f"""
        ### **Strategy Engine**
        * **Marketing Status:** The **{channel}** campaign is **{status_text}**.
        * **Financial Impact:** Projecting **${saved:,.0f}** in saved revenue vs **${real_cost:,.0f}** cost.
        """

        # 3. PLOTS
        plot_df = rfm.sample(n=min(2000, len(rfm)), random_state=42)
        warm_palette = 'inferno'

        # Plot 1: 3D
        fig1 = plt.figure(figsize=(10, 8))
        ax = fig1.add_subplot(111, projection='3d')
        sc = ax.scatter(plot_df['Recency'], plot_df['Frequency'], plot_df['Monetary'], 
                        c=pd.factorize(plot_df['Segment'])[0], cmap=warm_palette, s=60, alpha=0.8)
        ax.set_xlabel('Recency'); ax.set_ylabel('Frequency'); ax.set_zlabel('Monetary')
        ax.set_title('3D Customer Universe', pad=20)
        ax.xaxis.set_pane_color((0.2, 0.05, 0.05, 1.0))
        ax.yaxis.set_pane_color((0.2, 0.05, 0.05, 1.0))
        ax.zaxis.set_pane_color((0.2, 0.05, 0.05, 1.0))

        # Plot 2: ROI
        budgets = np.linspace(500, budget*3, 20)
        profits = [(at_risk * conversion) - (b * cost_factor + (at_risk * conversion * (discount/100))) for b in budgets]
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        ax2.plot(budgets, profits, color='#D4AF37', linewidth=3) # Gold line
        ax2.axhline(0, color='#FF4D4D', linestyle='--'); ax2.axvline(budget, color='#E8D0D0', linestyle=':', label="Current")
        plt.title("Profit vs Budget Curve"); plt.legend()

        # Plot 3: Dist
        fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(plot_df['Recency'], ax=ax3a, color='#C07A7A', kde=True); ax3a.set_title("Recency")
        sns.histplot(plot_df['Frequency'], ax=ax3b, color='#D4AF37', kde=True); ax3b.set_title("Frequency")
        sns.histplot(plot_df['Monetary'], ax=ax3c, color='#8C3A3A', kde=True); ax3c.set_title("Monetary"); ax3c.set_yscale('log')

        csv_path = "ShopSmart_Data.csv"
        rfm.to_csv(csv_path, index=False)

        return html, strategy, fig1, fig2, fig3, rfm.head(50), csv_path

    except Exception as e:
        return f"<h3>System Error: {str(e)}</h3>", "Please check settings.", None, None, None, None, None

# --- 4. BURGUNDY GLASS UI ---
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

/* Main Background */
body, .gradio-container { 
    background: linear-gradient(135deg, #2A0A0A 0%, #3C1414 100%) !important; 
    font-family: 'Inter', sans-serif !important; 
    color: #E8D0D0 !important;
}

/* Updated Headline */
h1 { 
    color: #E8D0D0 !important; 
    text-shadow: 0 0 20px rgba(212, 175, 55, 0.5); /* Gold Glow */
    font-size: 3.5rem !important; 
    margin: 0 !important; 
    font-weight: 800 !important;
    letter-spacing: -2px;
}

/* Glass Panels */
.glass-panel {
    background: rgba(60, 10, 10, 0.65) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(140, 60, 60, 0.3) !important;
    border-radius: 16px !important;
    padding: 25px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
}

.input-card, .marketing-card {
    background: rgba(80, 30, 30, 0.5) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(160, 80, 80, 0.4) !important;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}
.marketing-card { border-color: #D4AF37 !important; }

.input-header { 
    color: #D4AF37; 
    font-size: 0.9rem; font-weight: 700; text-transform: uppercase; 
    margin-bottom: 12px; border-bottom: 1px solid rgba(140, 60, 60, 0.5); padding-bottom: 5px; 
}

/* KPI GRID & CARDS */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin-bottom: 20px; }

.kpi-card { 
    background: rgba(70, 20, 20, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(140, 60, 60, 0.4) !important;
    padding: 15px; border-radius: 12px; text-align: center; 
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #D4AF37 !important; }
.kpi-label { color: #D4AF37; font-size: 0.75rem; font-weight: 700; margin-bottom: 5px; }
.kpi-val { color: #F8F0F0; font-size: 1.6rem; font-weight: 800; }

/* FLARE FOR TABS */
.tab-nav button {
    font-weight: bold !important;
    font-size: 1.1rem !important;
    color: #C07A7A !important; 
    transition: all 0.3s ease !important;
}
.tab-nav button:hover {
    color: #F8F0F0 !important;
    text-shadow: 0 0 8px rgba(255, 255, 255, 0.5);
}
.tab-nav button.selected {
    color: #D4AF37 !important; 
    border-bottom: 3px solid #D4AF37 !important; 
    background: linear-gradient(0deg, rgba(212, 175, 55, 0.1) 0%, rgba(0,0,0,0) 100%);
    text-shadow: 0 0 10px rgba(212, 175, 55, 0.6);
}

/* RUN BUTTON */
#run-btn { 
    background: linear-gradient(to right, #8C2B2B, #B33030) !important; 
    border: 1px solid #C07A7A !important;
    color: #F8F0F0 !important; font-weight: 800; margin-top: 10px; 
    box-shadow: 0 0 10px rgba(140, 43, 43, 0.5);
}
#run-btn:hover { background: linear-gradient(to right, #B33030, #D43535) !important; box-shadow: 0 0 20px rgba(179, 48, 48, 0.8); }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate")) as app:
    
    with gr.Row(elem_classes="header-container"):
        # Updated Headline
        gr.Markdown("# ShopSmart")

    with gr.Row():
        # --- LEFT PANELS (Glass) ---
        with gr.Column(scale=1, elem_classes="glass-panel"):
            
            with gr.Group(elem_classes="input-card"):
                gr.Markdown("<div class='input-header'>Source & Config</div>")
                file_up = gr.File(label="Upload CSV", type="filepath")
                biz_type = gr.Dropdown(["Retail (High Vol)", "Luxury (High Value)", "SaaS (Subscription)"], value="Retail (High Vol)", label="Industry Type")
                n_cust = gr.Slider(500, 10000, 2000, 500, label="Simulation Size")

            with gr.Group(elem_classes="input-card"):
                gr.Markdown("<div class='input-header'>AI Tuning</div>")
                k_val = gr.Slider(2, 6, 4, 1, label="Segment Count")
                churn_val = gr.Slider(30, 180, 60, 10, label="Churn Days")

            with gr.Group(elem_classes="marketing-card"):
                gr.Markdown("<div class='input-header' style='color:#D4AF37;'>Marketing Action</div>")
                channel = gr.Dropdown(["Email Drip (Low Cost)", "Social Ads (Medium)", "Influencer (High Cost)"], value="Social Ads (Medium)", label="Channel")
                budget = gr.Slider(500, 20000, 5000, 500, label="Budget")
                discount = gr.Slider(0, 50, 15, 5, label="Discount")
            
            run_btn = gr.Button("INITIALIZE ANALYTICS", elem_id="run-btn", size="lg")

        # --- RIGHT DASHBOARD (Glass) ---
        with gr.Column(scale=3, elem_classes="glass-panel"):
            
            out_html = gr.HTML()
            out_strat = gr.Markdown()
            
            with gr.Tabs():
                with gr.TabItem("3D Cluster Map"): plot1 = gr.Plot()
                with gr.TabItem("Profit Simulation"): plot2 = gr.Plot()
                with gr.TabItem("Metric Stats"): plot3 = gr.Plot()
                with gr.TabItem("Database Export"): 
                    df_out = gr.Dataframe(interactive=False)
                    dl_btn = gr.File(label="Download Full Report")

    run_btn.click(
        fn=analyze_business,
        inputs=[file_up, biz_type, n_cust, k_val, churn_val, budget, channel, discount],
        outputs=[out_html, out_strat, plot1, plot2, plot3, df_out, dl_btn]
    )

if __name__ == "__main__":
    app.launch()