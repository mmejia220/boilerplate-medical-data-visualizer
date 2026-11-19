
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar los datos
df = pd.read_csv('medical_examination.csv')

# 2. Calcular el sobrepeso a partir del IMC
df['overweight'] = (
    df['weight'] / (df['height'] / 100) ** 2 > 25
).astype(int)

# 3. Normalizar colesterol y glucosa
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Gráfico categórico
def draw_cat_plot():

    # 5. Transformar los datos a formato largo
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=[
            'cholesterol', 'gluc', 'smoke',
            'alco', 'active', 'overweight'
        ]
    )

    # 6. Agrupar y contar los registros
    df_cat = (
        df_cat.groupby(
            ['cardio', 'variable', 'value']
        )
        .size()
        .reset_index(name='total')
    )

    # 7. Crear el gráfico categórico
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        order=[
            'active', 'alco', 'cholesterol',
            'gluc', 'overweight', 'smoke'
        ]
    )

    # 8. Obtener la figura
    fig = g.fig

    # 9. No modificar
    fig.savefig('catplot.png')
    return fig


# 10. Mapa de calor
def draw_heat_map():

    # 11. Filtrar registros inconsistentes
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Matriz de correlación
    corr = df_heat.corr()

    # 13. Máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar la figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Crear el mapa de calor
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16. No modificar
    fig.savefig('heatmap.png')
    return fig

