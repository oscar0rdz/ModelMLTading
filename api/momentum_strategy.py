import pandas as pd
from api.binance_connector import get_historical_data
from app.models import Signal
from tortoise.transactions import in_transaction
import numpy as np

async def momentum_strategy(symbol: str, interval: str = '1h', limit: int = 1000):
    # Obtener datos históricos
    df = get_historical_data(symbol, interval, limit)
    
    # **Eliminar NaT en 'timestamp'**
    df = df.dropna(subset=['timestamp'])

    # Verificar si hay valores NaN en columnas clave
    df = df.dropna(subset=['close', 'high', 'low', 'volume'])

    # Verificación de las primeras filas para asegurar datos limpios
    print(f"Primeros 5 valores de 'timestamp' después de eliminar NaT:")
    print(df['timestamp'].head())

    # Calcular indicadores técnicos en la temporalidad base
    df['EMA_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA_23'] = df['close'].ewm(span=23, adjust=False).mean()
    df['MACD'] = df['EMA_8'] - df['EMA_23']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calcular RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calcular ADX
    df['TR'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ])
    df['DM_plus'] = np.where(
        (df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
        df['high'] - df['high'].shift(),
        0
    )
    df['DM_plus'] = np.where(df['DM_plus'] < 0, 0, df['DM_plus'])

    df['DM_minus'] = np.where(
        (df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
        df['low'].shift() - df['low'],
        0
    )
    df['DM_minus'] = np.where(df['DM_minus'] < 0, 0, df['DM_minus'])

    # Suavizar TR, DM+ y DM-
    df['TR_smooth'] = df['TR'].rolling(window=14).sum()
    df['DM_plus_smooth'] = df['DM_plus'].rolling(window=14).sum()
    df['DM_minus_smooth'] = df['DM_minus'].rolling(window=14).sum()

    # Calcular DI+ y DI-
    df['DI_plus'] = 100 * (df['DM_plus_smooth'] / df['TR_smooth'])
    df['DI_minus'] = 100 * (df['DM_minus_smooth'] / df['TR_smooth'])

    # Calcular DX y asegurar que no hay valores NaN
    df['DX'] = 100 * (abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus']))
    df['DX'].fillna(0, inplace=True)  # Reemplazar NaN en 'DX' con 0

    # Calcular ADX
    df['ADX'] = df['DX'].rolling(window=14).mean()

    # Calcular Volumen Promedio
    df['Volume_EMA'] = df['volume'].ewm(span=20, adjust=False).mean()

    # Obtener datos en temporalidad superior (4h)
    df_higher = get_historical_data(symbol, '4h', limit)
    df_higher['timestamp'] = pd.to_datetime(df_higher['timestamp'], unit='ms', utc=True)

    # **Eliminar NaT en 'timestamp' en el DataFrame superior**
    df_higher = df_higher.dropna(subset=['timestamp'])

    # Calcular EMAs en temporalidad superior
    df_higher['EMA_50'] = df_higher['close'].ewm(span=50, adjust=False).mean()
    df_higher['EMA_200'] = df_higher['close'].ewm(span=200, adjust=False).mean()

    # Alinear temporalidades
    df_higher.set_index('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    df = df.join(df_higher[['EMA_50', 'EMA_200']], how='left', rsuffix='_higher')
    df.reset_index(inplace=True)

    df['Higher_Trend'] = np.where(df['EMA_50'] > df['EMA_200'], 'bullish', 'bearish')

    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    # Generar señales
    df['signal'] = 0
    df.loc[
        (df['EMA_8'] > df['EMA_23']) &
        (df['RSI'] < 70) &
        (df['ADX'] > 25) &
        (df['Higher_Trend'] == 'bullish') &
        (df['volume'] > df['Volume_EMA']),
        'signal'
    ] = 1

    df.loc[
        (df['EMA_8'] < df['EMA_23']) &
        (df['RSI'] > 30) &
        (df['ADX'] > 25) &
        (df['Higher_Trend'] == 'bearish') &
        (df['volume'] > df['Volume_EMA']),
        'signal'
    ] = -1

    # Guardar señales en la base de datos
    async with in_transaction():
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            await Signal.create(
                symbol=symbol,
                close=row['close'],
                ema_8=row['EMA_8'],
                ema_23=row['EMA_23'],
                macd=row['MACD'],
                signal_line=row['Signal_Line'],
                rsi=row['RSI'],
                adx=row['ADX'],
                volume=row['volume'],
                higher_trend=row['Higher_Trend'],
                signal=row['signal'],
                timestamp=timestamp,
                interval=interval
            )

    return df
