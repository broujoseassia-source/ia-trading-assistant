# 📦 Installation requise : pip install pandas ta yfinance scikit-learn streamlit
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration de la page Streamlit ---
st.set_page_config(layout="wide", page_title="Assistant IA - Pocket Option Pro")

# --- 1. Chargement des données OHLCV ---
@st.cache_data(ttl=60)
def load_data(symbol='EURUSD=X', period='7d', interval='5m'):
    """Charge les données OHLCV à partir de Yahoo Finance."""
    try:
        df = yf.download(tickers=symbol, period=period, interval=interval)
        if df.empty:
            st.error(f"Aucune donnée trouvée pour {symbol} avec l'intervalle {interval}.")
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

# --- 2. Calcul des indicateurs techniques AMÉLIORÉS ---
def add_indicators(df):
    """Ajoute un ensemble plus riche d'indicateurs techniques."""
    if df.empty:
        return df

    # Indicateurs de Tendance
    df['ema_14'] = ta.trend.EMAIndicator(df['Close'], window=14).ema_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    
    # Indicateurs de Momentum
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd_diff'] = ta.trend.MACD(df['Close']).macd_diff()
    df['stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    
    # Indicateurs de Volatilité
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    # Indicateurs de Volume (si disponible)
    df['volume_sma'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    
    df.dropna(inplace=True)
    return df

# --- 3. Préparation des données pour l’IA ---
def prepare_data(df):
    """Prépare les données pour l'entraînement du modèle IA."""
    if df.empty:
        return None, None, None, None

    # Cible : 1 = CALL (le prix monte), 0 = PUT (le prix descend ou stagne)
    # Nous allons prédire le mouvement du prix sur la prochaine bougie
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Caractéristiques (Features)
    features = [
        'ema_14', 'sma_50', 'rsi', 'macd_diff', 'stoch_k', 
        'bb_high', 'bb_low', 'atr', 'volume_sma'
    ]
    
    # S'assurer que toutes les colonnes existent après le nettoyage
    features = [f for f in features if f in df.columns]
    
    X = df[features].iloc[:-1] # Exclure la dernière ligne car la cible est NaN
    y = df['target'].iloc[:-1]
    
    if len(X) < 2:
        st.warning("Pas assez de données pour l'entraînement du modèle après l'ajout des indicateurs.")
        return None, None, None, None

    # Séparation des données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test

# --- 4. Entraînement et prédiction ---
@st.cache_resource
def train_model(X_train, y_train):
    """Entraîne le modèle Random Forest."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def get_predictions(model, X_test, y_test):
    """Calcule les prédictions et la précision."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy

# --- 5. Visualisation Plotly (Graphique en chandeliers) ---
def plot_candlestick(df):
    """Crée un graphique en chandeliers interactif avec les indicateurs."""
    
    # Créer des sous-graphiques : 1 pour le prix/BB/EMA, 1 pour le MACD, 1 pour le RSI
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2])

    # Graphique 1 : Prix (Chandeliers)
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Prix'), row=1, col=1)

    # Bandes de Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], line=dict(color='orange', width=1), name='BB Haut'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], line=dict(color='orange', width=1), name='BB Bas'), row=1, col=1)
    
    # EMA 14
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_14'], line=dict(color='blue', width=1), name='EMA 14'), row=1, col=1)

    # Graphique 2 : MACD
    colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
    fig.add_trace(go.Bar(x=df.index, y=df['macd_diff'], name='MACD Diff', marker_color=colors), row=2, col=1)

    # Graphique 3 : RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), line=dict(color='red', width=0.5, dash='dash'), name='RSI 70', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), line=dict(color='green', width=0.5, dash='dash'), name='RSI 30', showlegend=False), row=3, col=1)

    # Mise en page
    fig.update_layout(
        title='Analyse Technique et Chandeliers',
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_dark"
    )
    
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig

# --- 6. Fonction de Backtesting ---
def run_backtest(df, model, initial_balance=1000, trade_amount=10, payout_rate=0.8):
    """Fonction de backtesting simplifiée (sera développée en phase 4)."""
    st.subheader("⚙️ Backtesting (Simulation de Trading)")
    
    # 1. Générer les signaux sur l'ensemble des données
    features = [
        'ema_14', 'sma_50', 'rsi', 'macd_diff', 'stoch_k', 
        'bb_high', 'bb_low', 'atr', 'volume_sma'
    ]
    
    # Assurez-vous que les données sont prêtes pour la prédiction
    df_test = df.copy().dropna()
    
    if df_test.empty:
        st.warning("Pas assez de données pour le backtesting après nettoyage.")
        return

    # Prédiction des signaux (1=CALL, 0=PUT)
    df_test['Signal'] = model.predict(df_test[features])
    
    # Décalage du signal d'une période pour simuler l'entrée au début de la bougie suivante
    df_test['Trade_Signal'] = df_test['Signal'].shift(1)
    
    # Définir le résultat du trade (simplifié pour une option binaire à expiration d'une bougie)
    # Un CALL (1) gagne si le prix de clôture est supérieur au prix d'ouverture de la bougie du trade
    # Un PUT (0) gagne si le prix de clôture est inférieur au prix d'ouverture de la bougie du trade
    
    # Résultat réel de la bougie du trade (1 si Close > Open, 0 sinon)
    df_test['Actual_Result'] = np.where(df_test['Close'] > df_test['Open'], 1, 0)
    
    # Déterminer si le trade a été gagnant (Win=1, Loss=0)
    # Win si (Signal=CALL et Actual=UP) OU (Signal=PUT et Actual=DOWN)
    df_test['Win'] = np.where(
        ((df_test['Trade_Signal'] == 1) & (df_test['Actual_Result'] == 1)) | 
        ((df_test['Trade_Signal'] == 0) & (df_test['Actual_Result'] == 0)), 
        1, 
        0
    )
    
    # Ne considérer que les trades où un signal a été donné
    df_trades = df_test.dropna(subset=['Trade_Signal']).copy()
    
    # 2. Calculer la performance financière
    df_trades['Profit'] = np.where(df_trades['Win'] == 1, trade_amount * payout_rate, -trade_amount)
    
    # 3. Calculer le solde du compte
    df_trades['Balance'] = initial_balance + df_trades['Profit'].cumsum()
    
    # 4. Afficher les résultats
    total_trades = len(df_trades)
    winning_trades = df_trades['Win'].sum()
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    final_balance = df_trades['Balance'].iloc[-1] if not df_trades.empty else initial_balance
    net_profit = final_balance - initial_balance
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Solde Final", f"{final_balance:.2f} $")
    col_b.metric("Taux de Gain", f"{win_rate:.2f} %")
    col_c.metric("Profit Net", f"{net_profit:.2f} $")
    
    st.subheader("Évolution du Solde")
    st.line_chart(df_trades['Balance'])
    
    st.subheader("Historique des Trades (5 derniers)")
    st.dataframe(df_trades[['Close', 'Trade_Signal', 'Actual_Result', 'Win', 'Profit', 'Balance']].tail(5))


# --- 7. Interface Streamlit ---

# Barre latérale pour les contrôles
with st.sidebar:
    st.title("Paramètres de l'Assistant")
    
    # Contrôles de l'actif et de l'intervalle
    symbol = st.selectbox("Choisir l'Actif", ['EURUSD=X', 'BTC-USD', 'ETH-USD', 'GBPUSD=X', 'USDJPY=X'])
    interval = st.selectbox("Time Frame (Intervalle)", ['1m', '5m', '15m', '30m', '1h'])
    period_map = {'1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d', '1h': '60d'}
    period = period_map.get(interval, '60d')
    
    # Mode de fonctionnement
    st.subheader("Mode de Fonctionnement")
    mode_simulation = st.toggle("Mode Simulation", value=True)
    mode_reel = st.toggle("Mode Réel (Désactivé)", value=False, disabled=True)
    
    # Bouton de rafraîchissement
    if st.button("Rafraîchir les Données et le Modèle"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

# Chargement et préparation des données
df = load_data(symbol=symbol, period=period, interval=interval)

if not df.empty:
    df = add_indicators(df)
    
    # Préparation des données pour l'entraînement
    X_train, X_test, y_train, y_test = prepare_data(df)

    if X_train is not None:
        # Entraînement du modèle
        model = train_model(X_train, y_train)
        
        # Évaluation du modèle
        predictions, accuracy = get_predictions(model, X_test, y_test)

        # --- Colonnes principales de l'interface ---
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.header(f"Analyse Technique pour {symbol} ({interval})")
            # Affichage du graphique interactif
            st.plotly_chart(plot_candlestick(df), use_container_width=True)

        with col2:
            st.header("Assistant IA")
            
            # Affichage du signal IA
            st.subheader("Signal de Trading Actuel")
            
            # Prédiction sur la dernière bougie disponible
            latest_features = df[[
                'ema_14', 'sma_50', 'rsi', 'macd_diff', 'stoch_k', 
                'bb_high', 'bb_low', 'atr', 'volume_sma'
            ]].iloc[-1].to_frame().T.fillna(0) # Préparer la dernière ligne pour la prédiction
            
            signal = model.predict(latest_features)[0]
            signal_proba = model.predict_proba(latest_features)[0]
            
            if signal == 1:
                st.success(f"**CALL** (Achat) - Probabilité: {signal_proba[1]*100:.2f}%")
            else:
                st.error(f"**PUT** (Vente) - Probabilité: {signal_proba[0]*100:.2f}%")
                
            # Métriques de performance
            st.subheader("Performance du Modèle")
            st.metric("Précision (Test Set)", f"{accuracy*100:.2f}%")
            
            # Affichage des indicateurs clés
            st.subheader("Indicateurs Clés (Dernière Bougie)")
            latest = df.iloc[-1]
            st.metric("Prix de Clôture", f"{latest['Close']:.5f}")
            st.metric("RSI", f"{latest['rsi']:.2f}")
            st.metric("MACD Diff", f"{latest['macd_diff']:.4f}")
            st.metric("ATR (Volatilité)", f"{latest['atr']:.4f}")
            
            # Affichage de l'importance des caractéristiques
            st.subheader("Importance des Caractéristiques")
            feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            st.bar_chart(feature_importances)
            
            # Backtesting (Appel de la fonction)
            if mode_simulation:
                st.subheader("Paramètres de Simulation")
                initial_balance = st.number_input("Solde Initial ($)", value=1000, min_value=100)
                trade_amount = st.number_input("Montant par Trade ($)", value=10, min_value=1)
                payout_rate = st.slider("Taux de Paiement (ex: 0.8 pour 80%)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                
                run_backtest(df.copy(), model, initial_balance, trade_amount, payout_rate)
            
    else:
        st.warning("Veuillez ajuster les paramètres (Actif/Intervalle) ou attendre que plus de données soient disponibles.")

else:
    st.info("Veuillez sélectionner un actif et un intervalle pour commencer l'analyse.")
