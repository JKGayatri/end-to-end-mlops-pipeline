import numpy as np

def create_features(df):

    # Total Area
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + \
                    df.get('1stFlrSF', 0) + \
                    df.get('2ndFlrSF', 0)

    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']

    # Bathrooms
    df['TotalBathrooms'] = (
        df.get('FullBath', 0) +
        0.5 * df.get('HalfBath', 0) +
        df.get('BsmtFullBath', 0) +
        0.5 * df.get('BsmtHalfBath', 0)
    )

    # Porch
    df['TotalPorchSF'] = (
        df.get('OpenPorchSF', 0) +
        df.get('EnclosedPorch', 0) +
        df.get('3SsnPorch', 0) +
        df.get('ScreenPorch', 0)
    )

    # Binary features
    df['HasGarage'] = df.get('GarageArea', 0).apply(lambda x: 1 if x > 0 else 0)
    df['HasFireplace'] = df.get('Fireplaces', 0).apply(lambda x: 1 if x > 0 else 0)

    return df