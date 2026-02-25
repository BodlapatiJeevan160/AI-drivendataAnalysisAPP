import streamlit as st
import traceback
import datetime

# 1. CONFIGURATION (MUST BE FIRST)
st.set_page_config(
    page_title="AI-Driven Automated Data Analytics Platform",
    layout="wide",
    page_icon="📊"
)

# 2. SAFE IMPORTS & SETUP
try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from io import BytesIO
    import yfinance as yf
    
    # ML Libraries
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error
    
except ImportError as e:
    st.error("❌ Critical Library Missing")
    st.write(f"Error details: {e}")
    st.info("Please install required packages: pip install streamlit pandas numpy plotly openpyxl yfinance scikit-learn")
    st.stop()

# 3. HELPER FUNCTIONS (Logic Layer)

@st.cache_data(show_spinner=False)
def load_data(file):
    """Safely loads CSV or Excel files."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def clean_data_logic(df):
    """Performs basic auto-cleaning: drops duplicates, handles missing values."""
    df_clean = df.copy()
    
    # Drop Duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dupes_removed = initial_rows - len(df_clean)
    
    # Fill Missing Numeric with Mean
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
    # Fill Missing Categorical with "Unknown"
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna("Unknown")
        
    return df_clean, dupes_removed

# --- HEALTHCARE SPECIFIC HELPERS ---

def clean_healthcare_data(df):
    """Specialized cleaning for healthcare data."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")
            
    return df

def train_healthcare_model(df, target_col):
    """Trains a classifier and returns metrics + model."""
    try:
        df_encoded = df.copy()
        label_encoders = {}
        
        for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if len(np.unique(y)) == 2:
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = np.max(model.predict_proba(X), axis=1)
            
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "feature_importance": importances,
            "risk_scores": probs,
            "model": model,
            "X": X
        }
        
    except Exception as e:
        st.error(f"ML Training Error: {e}")
        return None

def generate_health_insights(df, target_col, importances):
    insights = []
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
        top_val = df[target_col].mode()[0]
        pct = (df[target_col] == top_val).mean() * 100
        insights.append(f"Most common outcome for **{target_col}** is **{top_val}** ({pct:.1f}% of cases).")
        
    top_feature = importances.iloc[0]
    insights.append(f"The most critical factor influencing **{target_col}** is **{top_feature['Feature']}** (Importance: {top_feature['Importance']:.2f}).")
    
    return insights

def generate_health_decisions(importances):
    decisions = []
    top_feats = importances.head(3)['Feature'].tolist()
    decisions.append(f"🛡️ **Risk Screening**: Prioritize screening based on top risk factors: **{', '.join(top_feats)}**.")
    decisions.append(f"📊 **Resource Allocation**: Allocate more monitoring resources to patients showing abnormalities in **{top_feats[0]}**.")
    decisions.append(f"🤖 **AI Triage**: Use the computed Risk Score to prioritize high-risk patients.")
    return decisions

# --- SALES & BUSINESS HELPERS ---

def detect_sales_columns(df):
    """Auto-detects Date, Sales, Product, and Region columns."""
    cols = [c.lower() for c in df.columns]
    
    date_col = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'period', 'day'])), None)
    sales_col = next((c for c in df.columns if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'profit', 'turnover'])), None)
    prod_col = next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'sku', 'category'])), None)
    region_col = next((c for c in df.columns if any(x in c.lower() for x in ['region', 'country', 'city', 'state', 'location', 'store'])), None)
    
    return date_col, sales_col, prod_col, region_col

def clean_sales_data(df, date_col, sales_col):
    """Prepares sales data for forecasting."""
    df = df.copy()
    
    # Drop Unnamed
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Convert Date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    
    # Convert Sales to Numeric
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
    df = df.dropna(subset=[sales_col])
    
    return df

def generate_sales_forecast(df, date_col, sales_col, days=30):
    """Aggregates daily sales and predicts next 30 days."""
    try:
        # Aggregate by day
        daily_sales = df.groupby(date_col)[sales_col].sum().reset_index()
        daily_sales = daily_sales.set_index(date_col).asfreq('D').fillna(0).reset_index()
        
        # Prepare for ML
        daily_sales['Ordinal_Date'] = daily_sales[date_col].map(pd.Timestamp.toordinal)
        X = daily_sales[['Ordinal_Date']]
        y = daily_sales[sales_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Future
        last_date = daily_sales[date_col].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days + 1)]
        future_ordinals = [[d.toordinal()] for d in future_dates]
        
        predictions = model.predict(future_ordinals)
        predictions = [max(0, p) for p in predictions] # No negative sales
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': predictions})
        return daily_sales, forecast_df, model.coef_[0]
        
    except Exception as e:
        return None, None, None

def generate_business_suggestions(daily_sales, sales_col, slope, top_prod, top_region):
    """Generates business recommendations."""
    recs = []
    
    # Trend Analysis
    recent_trend = "growing" if slope > 0 else "declining"
    recs.append(f"📈 **Trend Strategy**: Sales are {recent_trend} (Slope: {slope:.2f}). {'Increase inventory & marketing.' if slope > 0 else 'Investigate root causes & run promotions.'}")
    
    # Volatility
    volatility = daily_sales[sales_col].std() / daily_sales[sales_col].mean()
    if volatility > 0.5:
        recs.append(f"⚡ **Stability**: High sales volatility detected. Consider stabilizing revenue streams with subscriptions or long-term contracts.")
    
    # Top Performers
    if top_prod:
        recs.append(f"🏆 **Product Focus**: Double down on **{top_prod}**, as it is the top revenue driver.")
    if top_region:
        recs.append(f"🌍 **Expansion**: **{top_region}** is the strongest market. replicate its success strategy in other regions.")
        
    # Seasonality (Simple)
    last_month_sales = daily_sales.set_index(daily_sales.columns[0]).last('30D')[sales_col].sum()
    prev_month_sales = daily_sales.set_index(daily_sales.columns[0]).last('60D').first('30D')[sales_col].sum()
    
    if last_month_sales > prev_month_sales:
        recs.append("🔥 **Momentum**: Sales increased month-over-month. Capitalize on current market sentiment.")
    else:
        recs.append("📉 **Retention**: Sales dipped recently. Activate customer retention campaigns.")
        
    return recs

# --- ANALYTICS INSIGHTS LOGIC (GENERAL) ---

def generate_auto_insights(df):
    """Generates rule-based textual insights from the dataframe."""
    insights = []
    insights.append(f"Dataset contains **{len(df)} rows** and **{len(df.columns)} columns**.")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            insights.append(f"Column **{col}** is highly skewed ({skew:.2f}).")
    return insights

def generate_auto_suggestions(df):
    """Suggests charts based on data types."""
    suggestions = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 1: suggestions.append(f"📊 **Histogram**: Use for **{num_cols[0]}**.")
    if len(num_cols) >= 2: suggestions.append(f"📉 **Scatter Plot**: Plot **{num_cols[0]} vs {num_cols[1]}**.")
    return suggestions

def generate_auto_decisions(df):
    """Simulates business decisions."""
    decisions = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        target = num_cols[-1]
        decisions.append(f"**KPI Monitoring**: Monitor **{target}** closely.")
    return decisions

def generate_ai_response(query, df):
    """Safe, Rule-Based AI Response Generator."""
    query = query.lower()
    response_text = "I can help you visualize data! Try asking for a **summary** or **distribution**."
    chart = None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if "summary" in query: response_text = "Here is the statistical summary."
    elif "distribution" in query and len(numeric_cols) > 0:
        chart = px.histogram(df, x=numeric_cols[0])
        response_text = f"Distribution of {numeric_cols[0]}."
    return response_text, chart

# --- FINANCIAL MARKET LOGIC (STOCKS & CRYPTO) ---

def get_market_data(ticker, period="6mo", mode="stock"):
    try:
        ticker = ticker.strip().upper()
        if mode == "crypto" and not (ticker.endswith("-USD") or ticker.endswith("USD")):
            search_ticker = f"{ticker}-USD"
        else:
            search_ticker = ticker
        stock = yf.Ticker(search_ticker)
        df = stock.history(period=period)
        if df.empty and mode == "crypto":
             stock = yf.Ticker(ticker)
             df = stock.history(period=period)
             search_ticker = ticker
        if df.empty: return None, None
        return df, search_ticker
    except: return None, None

def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def train_predict_market(df):
    df_train = df.copy().dropna()
    df_train['Date_Ordinal'] = df_train.index.map(pd.Timestamp.toordinal)
    X = df_train[['Date_Ordinal']]
    y = df_train['Close']
    model = LinearRegression()
    model.fit(X, y)
    last_date = df_train.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
    future_ordinals = [[d.toordinal()] for d in future_dates]
    predictions = model.predict(future_ordinals)
    return future_dates, predictions, model.coef_[0]

def get_market_recommendation(df, preds):
    last_row = df.iloc[-1]
    score = 0
    reasons = []
    
    # Simple logic for brevity
    if preds[-1] > last_row['Close']: score += 1; reasons.append("Forecast is Bullish.")
    else: score -= 1; reasons.append("Forecast is Bearish.")
    
    if last_row['RSI'] < 30: score += 1; reasons.append("RSI Oversold.")
    elif last_row['RSI'] > 70: score -= 1; reasons.append("RSI Overbought.")
    
    if score > 0: return {"Rec": "BUY 🟢", "Reasons": reasons, "Buy Zone": "Now", "Sell Zone": "High", "Stop Loss": "Low"}
    elif score < 0: return {"Rec": "SELL 🔴", "Reasons": reasons, "Buy Zone": "Low", "Sell Zone": "Now", "Stop Loss": "High"}
    return {"Rec": "HOLD 🟡", "Reasons": reasons, "Buy Zone": "Support", "Sell Zone": "Resistance", "Stop Loss": "Mid"}

# --- DATA SCIENTIST MODE HELPERS ---

def ds_preprocess_data(df):
    """
    Intelligent Data Cleaning & Feature Engineering for DS Mode.
    """
    df_ds = df.copy()
    df_ds.drop_duplicates(inplace=True)
    df_ds = df_ds.loc[:, (df_ds != df_ds.iloc[0]).any()]
    df_ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df_ds.select_dtypes(include=[np.number]).columns
    for col in num_cols: df_ds[col] = df_ds[col].fillna(df_ds[col].median())
    cat_cols = df_ds.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if not df_ds[col].mode().empty: df_ds[col] = df_ds[col].fillna(df_ds[col].mode()[0])
    for col in cat_cols:
        if df_ds[col].nunique() > 0.9 * len(df_ds): df_ds.drop(columns=[col], inplace=True)
    return df_ds

def ds_feature_engineering(df, target_col):
    """Encodes and Scales features for modeling."""
    df_eng = df.copy()
    cat_cols = df_eng.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_eng[col] = le.fit_transform(df_eng[col].astype(str))
    num_cols = df_eng.select_dtypes(include=[np.number]).columns
    features_to_scale = [c for c in num_cols if c != target_col]
    if features_to_scale:
        scaler = StandardScaler()
        df_eng[features_to_scale] = scaler.fit_transform(df_eng[features_to_scale])
    return df_eng

def run_data_scientist_mode(df):
    st.header("🔬 Data Scientist Studio")
    columns = df.columns.tolist()
    target_col = st.selectbox("Select Target Variable to Predict", columns)
    
    if target_col:
        with st.expander("📊 Step 1: Automated Data Profiling", expanded=False):
            st.write(df.describe())
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                st.plotly_chart(px.imshow(num_df.corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

        st.subheader("2. Data Preparation")
        apply_clean = st.checkbox("Apply Intelligent Cleaning & Feature Engineering", value=True)
        if apply_clean:
            df_ready = ds_feature_engineering(ds_preprocess_data(df), target_col)
            st.success("✅ Data Cleaned, Encoded & Scaled")
        else:
            df_ready = df.dropna()
        
        if st.button("🚀 Train Auto-ML Models"):
            try:
                if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
                    problem_type = "Regression"
                else:
                    problem_type = "Classification"
                
                st.info(f"Detected Problem Type: **{problem_type}**")
                X = df_ready.drop(columns=[target_col])
                y = df_ready[target_col]
                
                if problem_type == "Classification" and y.dtype == 'object':
                     le = LabelEncoder()
                     y = le.fit_transform(y.astype(str))
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                models = {}
                results = {}
                
                with st.spinner("Training Models..."):
                    if problem_type == "Regression":
                        models["Linear Regression"] = LinearRegression()
                        models["Random Forest"] = RandomForestRegressor(n_estimators=50, random_state=42)
                        models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=50, random_state=42)
                        for name, clf in models.items():
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            results[name] = {"R2": r2_score(y_test, y_pred), "model": clf}
                        res_df = pd.DataFrame(results).T[['R2']].sort_values(by='R2', ascending=False)
                        st.dataframe(res_df)
                        best_model_name = res_df.index[0]
                        st.success(f"Best Model: **{best_model_name}**")
                        
                        best_model = results[best_model_name]["model"]
                        y_best_pred = best_model.predict(X_test)
                        fig_res = px.scatter(x=y_test, y=y_best_pred, labels={'x':'Actual', 'y':'Predicted'}, title=f"Actual vs Predicted ({best_model_name})")
                        fig_res.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
                        st.plotly_chart(fig_res, use_container_width=True)

                    else: # Classification
                        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
                        models["Random Forest"] = RandomForestClassifier(n_estimators=50, random_state=42)
                        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=50, random_state=42)
                        for name, clf in models.items():
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            results[name] = {"Accuracy": accuracy_score(y_test, y_pred), "model": clf}
                        res_df = pd.DataFrame(results).T[['Accuracy']].sort_values(by='Accuracy', ascending=False)
                        st.dataframe(res_df)
                        best_model_name = res_df.index[0]
                        st.success(f"Best Model: **{best_model_name}**")
                        
                        best_model = results[best_model_name]["model"]
                        if hasattr(best_model, 'feature_importances_'):
                            imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_}).sort_values(by='Importance', ascending=False)
                            st.plotly_chart(px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importance"))

            except Exception as e:
                st.error(f"Training Failed: {e}")

# --- AI DATA COPILOT (UPGRADED LOGIC) ---

def build_dataset_context(df):
    """Generates a text summary of the dataset for the AI."""
    context = []
    context.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    context.append(f"Columns: {', '.join(df.columns.tolist())}")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        context.append(f"Numeric Columns: {', '.join(num_cols)}")
        context.append("Key Statistics:")
        stats = df[num_cols].describe().T[['mean', 'min', 'max']].to_dict()
        for col, val in list(stats.items())[:5]: # Limit to 5 cols to save context
            context.append(f"- {col}: Mean={val['mean']:.2f}, Min={val['min']:.2f}, Max={val['max']:.2f}")
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        context.append(f"Missing Values: {missing.to_dict()}")
        
    return "\n".join(context)

def detect_user_intent(query):
    """Determines the user's intent based on natural language."""
    q = query.lower()
    
    # 1. Data View
    if any(word in q for word in ["show data", "display data", "see data", "view data", "print data", "dataset"]):
        return "show_data"
    
    # 2. Data Structure / Metadata
    if any(word in q for word in ["what data", "type of data", "data present", "columns", "column names", "structure", "names of data", "fields", "features"]):
        return "describe_data"
        
    # 3. Dashboard / Visualization
    if any(word in q for word in ["dashboard", "visualize", "charts", "graphs", "plot", "analysis view"]):
        return "create_dashboard"
        
    # 4. Statistics
    if any(word in q for word in ["statistics", "summary", "average", "mean", "describe"]):
        return "statistics"
        
    # 5. Insights
    if any(word in q for word in ["insight", "explain", "analysis", "tell me about data"]):
        return "insights"
        
    # 6. Correlation
    if "correlation" in q:
        return "correlation"
        
    return "chat"

def safe_numeric_df(df):
    """Safely return numeric dataframe or None."""
    num = df.select_dtypes(include=["number"])
    return num if not num.empty else None

def run_ai_copilot(df):
    """Runs the interactive AI Data Copilot interface."""
    st.header("🤖 AI Data Copilot")
    st.caption("Your personal data analyst. Ask questions about the uploaded dataset.")
    
    # 1. Initialize History with Type support
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = [
            {"role": "assistant", "content": "Hello! I am ready to analyze your data. Try asking: 'Show data', 'Give insights', or 'Create dashboard'.", "type": "text"}
        ]
        
    # 2. Display History
    for msg in st.session_state["copilot_messages"]:
        with st.chat_message(msg["role"]):
            msg_type = msg.get("type", "text")
            if msg_type == "text":
                st.write(msg["content"])
            elif msg_type == "data":
                st.dataframe(msg["content"])
            elif msg_type == "plot":
                st.plotly_chart(msg["content"], use_container_width=True)
            elif msg_type == "dashboard":
                # Dashboard is a list of figures
                for fig in msg["content"]:
                    st.plotly_chart(fig, use_container_width=True)
        
    # 3. Handle Input
    if prompt := st.chat_input("Ask a question (e.g., 'What insights can you find?')"):
        st.session_state["copilot_messages"].append({"role": "user", "content": prompt, "type": "text"})
        st.chat_message("user").write(prompt)
        
        # 4. Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response_payloads = []
                try:
                    intent = detect_user_intent(prompt)
                    
                    if intent == "describe_data":
                        info = {
                            "Rows": df.shape[0],
                            "Columns": df.shape[1],
                            "Column Names": list(df.columns),
                            "Missing Values": df.isna().sum().sum()
                        }
                        response_payloads.append({"role": "assistant", "content": "### Dataset Overview", "type": "text"})
                        response_payloads.append({"role": "assistant", "content": info, "type": "data"})
                        response_payloads.append({"role": "assistant", "content": "This is the structural overview of your data.", "type": "text"})

                    elif intent == "show_data":
                        response_payloads.append({"role": "assistant", "content": df.head(100), "type": "data"})
                        response_payloads.append({"role": "assistant", "content": "Here is a preview of the first 100 rows.", "type": "text"})

                    elif intent == "statistics":
                        num_df = safe_numeric_df(df)
                        if num_df is None:
                             response_payloads.append({"role": "assistant", "content": "No numeric columns available for statistics.", "type": "text"})
                        else:
                            response_payloads.append({"role": "assistant", "content": num_df.describe(), "type": "data"})
                            response_payloads.append({"role": "assistant", "content": "Here are the descriptive statistics.", "type": "text"})

                    elif intent == "correlation":
                        num_df = safe_numeric_df(df)
                        if num_df is None or num_df.shape[1] < 2:
                             response_payloads.append({"role": "assistant", "content": "Need at least two numeric columns for correlation.", "type": "text"})
                        else:
                            fig = px.imshow(num_df.corr(), text_auto=True, title="Correlation Matrix")
                            response_payloads.append({"role": "assistant", "content": fig, "type": "plot"})

                    elif intent == "create_dashboard":
                        figs = []
                        num_df = safe_numeric_df(df)
                        
                        # 1. Histogram
                        if num_df is not None:
                            col1 = num_df.columns[0]
                            figs.append(px.histogram(df, x=col1, title=f"Distribution of {col1}"))
                        
                        # 2. Bar Chart (Categorical)
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns
                        if len(cat_cols) > 0:
                            counts = df[cat_cols[0]].value_counts().head(10)
                            figs.append(px.bar(x=counts.index, y=counts.values, title=f"Top {cat_cols[0]} Counts"))
                            
                        # 3. Line Chart
                        if num_df is not None:
                            col_line = num_df.columns[0]
                            figs.append(px.line(df.reset_index(), y=col_line, title=f"Trend of {col_line}"))

                        if figs:
                            response_payloads.append({"role": "assistant", "content": figs, "type": "dashboard"})
                            response_payloads.append({"role": "assistant", "content": "I have generated a dashboard based on your data.", "type": "text"})
                        else:
                            response_payloads.append({"role": "assistant", "content": "Could not auto-generate charts. Ensure numeric/categorical data exists.", "type": "text"})

                    elif intent == "insights":
                        insights = []
                        num_df = safe_numeric_df(df)
                        
                        if num_df is not None:
                            # Variance
                            try:
                                var_col = num_df.var().idxmax()
                                insights.append(f"• **Highest Variance:** '{var_col}' has the widest range of values.")
                            except: pass
                            
                            # Correlation
                            if num_df.shape[1] > 1:
                                corr = num_df.corr().abs().unstack().sort_values(ascending=False)
                                top_corr = corr[corr < 1.0].head(1)
                                if not top_corr.empty:
                                    v1, v2 = top_corr.index[0]
                                    insights.append(f"• **Strongest Correlation:** Between '{v1}' and '{v2}' ({top_corr.iloc[0]:.2f}).")
                        
                        # Missing
                        missing = df.isnull().sum().sum()
                        if missing > 0:
                            insights.append(f"• **Data Quality:** There are {missing} missing values to address.")
                        else:
                            insights.append("• **Data Quality:** The dataset is complete.")
                            
                        insights.append(f"• **Size:** {df.shape[0]} records analyzed.")
                        
                        txt = "### 💡 AI Insights\n" + "\n".join(insights)
                        response_payloads.append({"role": "assistant", "content": txt, "type": "text"})

                    else: # Default Chat
                        help_text = """
                        I can help you explore your dataset. Try asking:
                        - **Show data**
                        - **What columns are present?**
                        - **Create dashboard**
                        - **Give insights**
                        """
                        response_payloads.append({"role": "assistant", "content": help_text, "type": "text"})

                except Exception as e:
                    response_payloads.append({"role": "assistant", "content": f"I understood your intent but encountered an error: {e}", "type": "text"})

                # Render and Save
                for p in response_payloads:
                    st.session_state["copilot_messages"].append(p)
                    if p["type"] == "text":
                        st.write(p["content"])
                    elif p["type"] == "data":
                        st.dataframe(p["content"])
                    elif p["type"] == "plot":
                        st.plotly_chart(p["content"], use_container_width=True)
                    elif p["type"] == "dashboard":
                        for f in p["content"]:
                            st.plotly_chart(f, use_container_width=True)


# 4. MAIN APPLICATION LOGIC
def main():
    with st.sidebar:
        st.header("⚙️ Data Controls")
        
        app_mode = st.radio(
            "Select Operation Mode",
            ["Full Analytics", "Data Cleaning Only", "Stock Market Prediction", "Crypto Market Prediction", "Healthcare Analytics & Risk Prediction", "Sales Forecasting & Business Dashboard", "Data Scientist Mode", "AI Data Copilot"]
        )
        
        uploaded_file = None
        # Allow upload in all modes except pure market prediction (which fetches data)
        # Actually, let's allow upload everywhere for consistency, 
        # but Market modes ignore it if they use Ticker.
        if app_mode not in ["Stock Market Prediction", "Crypto Market Prediction"]:
            st.header("📂 Upload Data")
            uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
            st.header("🛠️ Settings")
            clean_data_opt = st.checkbox("Auto-Clean Data", value=True)

    st.title("AI-Driven Automated Data Analytics Platform")
    
    # ---------------------------------------------------------
    # MODE 1 & 2: STOCK / CRYPTO
    # ---------------------------------------------------------
    if app_mode in ["Stock Market Prediction", "Crypto Market Prediction"]:
        is_crypto = app_mode == "Crypto Market Prediction"
        mode_icon = "🪙" if is_crypto else "📈"
        default_ticker = "BTC" if is_crypto else "AAPL"
        
        st.subheader(f"{mode_icon} {app_mode}")
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_input(f"Enter {'Crypto' if is_crypto else 'Stock'} Ticker", value=default_ticker)
        with col2:
            st.write(""); st.write("")
            fetch_btn = st.button("Analyze Market")
            
        if fetch_btn and ticker_input:
            with st.spinner("Fetching data..."):
                try:
                    mode_param = "crypto" if is_crypto else "stock"
                    df_market, clean_ticker = get_market_data(ticker_input, mode=mode_param)
                    if df_market is not None:
                        df_market = calculate_technical_indicators(df_market)
                        last_row = df_market.iloc[-1]
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Price", f"{last_row['Close']:,.2f}")
                        m2.metric("RSI", f"{last_row['RSI']:.2f}")
                        m3.metric("Volume", f"{last_row['Volume']:,}")
                        
                        st.subheader("Charts")
                        fig = go.Figure(data=[go.Candlestick(x=df_market.index, open=df_market['Open'], high=df_market['High'], low=df_market['Low'], close=df_market['Close'])])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Forecast (7 Days)")
                        dates, preds, slope = train_predict_market(df_market)
                        pred_df = pd.DataFrame({"Date": dates, "Price": preds})
                        st.line_chart(pred_df.set_index("Date"))
                        
                        rec = get_market_recommendation(df_market, preds)
                        st.success(f"Recommendation: {rec['Rec']}")
                        for r in rec['Reasons']: st.write(f"- {r}")
                    else: st.error("Ticker not found.")
                except Exception as e: st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # MODE 3: DATA CLEANING ONLY
    # ---------------------------------------------------------
    elif app_mode == "Data Cleaning Only":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                if clean_data_opt: df, _ = clean_data_logic(df)
                st.write("### Cleaned Data")
                st.dataframe(df.head())
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "cleaned.csv", "text/csv")
        else: st.info("Upload data.")

    # ---------------------------------------------------------
    # MODE 4: FULL ANALYTICS
    # ---------------------------------------------------------
    elif app_mode == "Full Analytics":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                if clean_data_opt: df, _ = clean_data_logic(df)
                t1, t2, t3 = st.tabs(["Overview", "Visuals", "Insights"])
                with t1: st.dataframe(df.head()); st.write(df.describe())
                with t2:
                    num_cols = df.select_dtypes(include=np.number).columns
                    if len(num_cols) > 0: st.plotly_chart(px.histogram(df, x=num_cols[0]))
                with t3:
                    for i in generate_auto_insights(df): st.write(f"- {i}")
        else: st.info("Upload data.")

    # ---------------------------------------------------------
    # MODE 5: HEALTHCARE ANALYTICS
    # ---------------------------------------------------------
    elif app_mode == "Healthcare Analytics & Risk Prediction":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                if clean_data_opt: df = clean_healthcare_data(df)
                st.subheader("🏥 Healthcare Risk Engine")
                st.dataframe(df.head())
                
                target = st.selectbox("Select Target (Diagnosis/Risk)", df.columns)
                if st.button("Train Model"):
                    res = train_healthcare_model(df, target)
                    if res:
                        st.metric("Accuracy", f"{res['accuracy']:.2%}")
                        st.plotly_chart(px.bar(res['feature_importance'].head(10), x='Importance', y='Feature', orientation='h'))
                        st.warning("Disclaimer: Educational use only.")
        else: st.info("Upload Healthcare Data.")

    # ---------------------------------------------------------
    # MODE 6: SALES FORECASTING & DASHBOARD
    # ---------------------------------------------------------
    elif app_mode == "Sales Forecasting & Business Dashboard":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.subheader("💼 Business Intelligence Dashboard")
                
                # Auto-Detect Columns
                date_col, sales_col, prod_col, reg_col = detect_sales_columns(df)
                
                # Column mapping UI
                c1, c2, c3, c4 = st.columns(4)
                with c1: date_col = st.selectbox("Date Column", df.columns, index=df.columns.get_loc(date_col) if date_col else 0)
                with c2: sales_col = st.selectbox("Sales Column", df.columns, index=df.columns.get_loc(sales_col) if sales_col else 0)
                with c3: prod_col = st.selectbox("Product Column (Opt)", ["None"] + list(df.columns), index=df.columns.get_loc(prod_col)+1 if prod_col else 0)
                with c4: reg_col = st.selectbox("Region Column (Opt)", ["None"] + list(df.columns), index=df.columns.get_loc(reg_col)+1 if reg_col else 0)

                if date_col and sales_col:
                    # Cleaning for Sales
                    if clean_data_opt:
                        df_sales = clean_sales_data(df, date_col, sales_col)
                    else:
                        df_sales = df.copy()

                    # KPIs
                    total_sales = df_sales[sales_col].sum()
                    avg_daily = df_sales.groupby(date_col)[sales_col].sum().mean()
                    
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Revenue", f"${total_sales:,.0f}")
                    k2.metric("Avg Daily Sales", f"${avg_daily:,.0f}")
                    
                    if prod_col != "None":
                        top_p = df_sales.groupby(prod_col)[sales_col].sum().idxmax()
                        k3.metric("Top Product", top_p)
                    else: k3.metric("Top Product", "N/A")
                        
                    if reg_col != "None":
                        top_r = df_sales.groupby(reg_col)[sales_col].sum().idxmax()
                        k4.metric("Top Region", top_r)
                    else: k4.metric("Top Region", "N/A")

                    # Tabs
                    tab_dash, tab_fc, tab_rec = st.tabs(["📉 Dashboard", "🔮 Forecast", "💡 Recommendations"])
                    
                    with tab_dash:
                        # Trend
                        daily_trend = df_sales.groupby(date_col)[sales_col].sum().reset_index()
                        fig_trend = px.line(daily_trend, x=date_col, y=sales_col, title="Sales Trend Over Time")
                        st.plotly_chart(fig_trend, use_container_width=True)
                        
                        r1, r2 = st.columns(2)
                        with r1:
                            # Monthly
                            df_sales['Month'] = df_sales[date_col].dt.month_name()
                            df_sales['Year'] = df_sales[date_col].dt.year
                            # Sort by month order needed, simplified here
                            monthly = df_sales.groupby('Month')[sales_col].sum().reset_index()
                            fig_mon = px.bar(monthly, x='Month', y=sales_col, title="Monthly Sales")
                            st.plotly_chart(fig_mon, use_container_width=True)
                            
                        with r2:
                            # Heatmap
                            heatmap_data = df_sales.groupby(['Year', 'Month'])[sales_col].sum().reset_index()
                            heatmap_pivot = heatmap_data.pivot(index='Year', columns='Month', values=sales_col)
                            fig_heat = px.imshow(heatmap_pivot, text_auto=True, title="Year-Month Sales Heatmap")
                            st.plotly_chart(fig_heat, use_container_width=True)

                        # Product/Region breakdown
                        if prod_col != "None":
                            top_prods = df_sales.groupby(prod_col)[sales_col].sum().nlargest(10).reset_index()
                            st.plotly_chart(px.bar(top_prods, x=prod_col, y=sales_col, title="Top 10 Products by Revenue"), use_container_width=True)

                    with tab_fc:
                        st.subheader("30-Day Revenue Forecast")
                        daily_agg, fc_df, slope = generate_sales_forecast(df_sales, date_col, sales_col)
                        
                        if fc_df is not None:
                            # Plot History + Forecast
                            fig_fc = go.Figure()
                            fig_fc.add_trace(go.Scatter(x=daily_agg[date_col], y=daily_agg[sales_col], name='Historical'))
                            fig_fc.add_trace(go.Scatter(x=fc_df['Date'], y=fc_df['Predicted_Sales'], name='Forecast', line=dict(color='purple', dash='dash')))
                            fig_fc.update_layout(title="Sales Projection")
                            st.plotly_chart(fig_fc, use_container_width=True)
                            
                            st.dataframe(fc_df)
                        else:
                            st.error("Forecast failed. Ensure sufficient numeric data and dates.")

                    with tab_rec:
                        st.subheader("Automated Business Suggestions")
                        if fc_df is not None:
                            # Pass data for suggestions
                            top_p_val = df_sales.groupby(prod_col)[sales_col].sum().idxmax() if prod_col != "None" else None
                            top_r_val = df_sales.groupby(reg_col)[sales_col].sum().idxmax() if reg_col != "None" else None
                            
                            recs = generate_business_suggestions(daily_agg, sales_col, slope, top_p_val, top_r_val)
                            for r in recs:
                                st.info(r)
                            
                            st.download_button("📥 Download Business Report", "\n".join(recs), "business_report.txt")
                        else:
                            st.warning("Run forecast to generate suggestions.")

                else:
                    st.warning("Could not auto-detect Date and Sales columns. Please select them manually above.")
        else:
            st.info("Upload Sales Data (CSV/Excel).")

    # ---------------------------------------------------------
    # MODE 7: DATA SCIENTIST MODE
    # ---------------------------------------------------------
    elif app_mode == "Data Scientist Mode":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                if clean_data_opt:
                    df, _ = clean_data_logic(df)
                
                run_data_scientist_mode(df)
        else:
            st.info("Upload Data for Data Science Workflow.")

    # ---------------------------------------------------------
    # MODE 8: AI DATA COPILOT (UPGRADED)
    # ---------------------------------------------------------
    elif app_mode == "AI Data Copilot":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                if clean_data_opt:
                    df, _ = clean_data_logic(df)
                run_ai_copilot(df)
        else:
            st.info("Upload Data to Chat with it.")

# 5. ENTRY POINT
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical Application Error")
        st.code(traceback.format_exc())