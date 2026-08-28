import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from streamlit_option_menu import option_menu
from scipy.optimize import newton
import numpy_financial as npf
import missingno as msno
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sucker rod pump optimizer",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- ESTILO CSS PERSONALIZADO ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    /* OCULTAR SOLO EL MENÚ SUPERIOR */
    header {
    display: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN LATERAL ---
with st.sidebar:
    st.image(
        "company.png",
        width=280  # Tamaño incrementado
    )
    st.markdown(
        """
        <div style="
            color:#f39c12;
            text-align:center;
            font-size:30px;
            font-weight:700;
            margin-bottom:12px;
        ">
            Sucker rod pump optimizer
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        """
        <h4 style="
            color:#f39c12;
            text-align:left;
            margin:10px 0px 5px 0px;
        ">
            Options
        </h4>
        """,
        unsafe_allow_html=True
    )
    selected = option_menu(
        menu_title=None,
        options=["Inicio",
                 "Análisis de datos",
                 "Validación técnica",
                 "Análisis Nodal",
                 "Dimensionamiento de equipos",
                 "Validación económica"],
        icons=["house",
               "database",
               "shield-check",
               "graph-up-arrow",
               "sliders",
               "cash-coin"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0e1117"},
            "icon": {"color": "#f39c12", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "15px", "text-align": "left",
                         "margin": "0px"},
            "nav-link-selected": {"background-color": "#262730"},
        }
    )

#funciones ------

# ==========================================
# 1. FUNCIONES DE DEPURACIÓN Y GRÁFICOS
# ==========================================


def Depuracion_produccion(df):
  df.columns = df.columns.str.strip().str.lower()
  df.columns = (
      df.columns.str.replace("@", "", regex=False)
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
  )

  df.columns = df.columns.str.replace(
      "tiempo_prueba hours", "t_prueba h", regex=False
  )
  df.columns = df.columns.str.replace(
      "tiempo de prueba hours", "t_prueba h", regex=False
  )
  df.columns = df.columns.str.replace(
      "presion de intake psi", "pip psi", regex=False
  )
  df.columns = df.columns.str.replace(
      "presion de fondo fluyente psi", "pwf psi", regex=False
  )
  df.columns = df.columns.str.replace(
      "presion estatica del reservorio psi", "p_reservorio psi", regex=False
  )
  df.columns = df.columns.str.replace(
      "gravedad api del petroleo", "api", regex=False
  )
  df.columns = df.columns.str.replace(
      "indice de productividad bbl/psi", "ip bbl/d/psi", regex=False
  )
  df.columns = df.columns.str.replace("presion", "p", regex=False)

  df.columns = df.columns.str.replace(
      "prueba de produccion petroleo a 24 horas bbl/d",
      "produccion de petroleo bbl",
      regex=False,
  )
  df.columns = df.columns.str.replace(
      "prueba de produccion petróleo a 24 horas bbl/d",
      "produccion de petroleo bbl",
      regex=False,
  )
  df.columns = df.columns.str.replace(
      "prueba de produccion agua a 24 horas bbl/d",
      "produccion de agua bbl",
      regex=False,
  )
  df.columns = df.columns.str.replace(
      "prueba de producción agua a 24 horas bbl/d",
      "produccion de agua bbl",
      regex=False,
  )
  df.columns = df.columns.str.replace(
      "prueba de produccion gas a 24 horas mcf/d",
      "produccion de gas mcf",
      regex=False,
  )
  df.columns = df.columns.str.replace(
      "prueba de producción gas a 24 horas mcf/d",
      "produccion de gas mcf",
      regex=False,
  )

  df.columns = df.columns.str.strip()
  df["date"] = pd.to_datetime(df["date"])
  df = df.sort_values("date").reset_index(drop=True)

  col_petro = [c for c in df.columns if "petroleo" in c or "petróleo" in c][0]
  df = df[df[col_petro] > 0]
  return df


def grafico_nulos(df, nombre_pozo):
  fig, ax = plt.subplots(figsize=(10, 5))
  msno.matrix(df, ax=ax, color=(0.1, 0.4, 0.6), fontsize=10)
  ax.set_title(
      f"Mapa de Valores Faltantes (Matriz) - Pozo {nombre_pozo}",
      fontsize=13,
      fontweight="bold",
  )
  return fig


def grafico_linea_liquidos_plotly(df, nombre_pozo):
  col_petro = [c for c in df.columns if "petróleo" in c or "petroleo" in c][0]
  col_agua = [c for c in df.columns if "agua" in c][0]

  df_melted = df.melt(
      id_vars=["date"],
      value_vars=[col_petro, col_agua],
      var_name="Fluido",
      value_name="Caudal (bbl/d)",
  )
  df_melted["Fluido"] = df_melted["Fluido"].replace(
      {col_petro: "Petróleo (bbl/d)", col_agua: "Agua (bbl/d)"}
  )

  fig = px.line(
      df_melted,
      x="date",
      y="Caudal (bbl/d)",
      color="Fluido",
      markers=True,
      title=f"Comportamiento Histórico de Líquidos - Pozo {nombre_pozo}",
      color_discrete_map={
          "Petróleo (bbl/d)": "green",
          "Agua (bbl/d)": "blue",
      },
  )
  fig.update_layout(
      xaxis_title="Fecha de Prueba",
      yaxis_title="Caudal (bbl/d)",
      legend_title="Fluidos",
      template="plotly_white",
  )
  return fig


def grafico_linea_gas_plotly(df, nombre_pozo):
  col_gas = [c for c in df.columns if "gas" in c][0]

  fig = px.line(
      df,
      x="date",
      y=col_gas,
      markers=True,
      title=f"Comportamiento Histórico de Gas - Pozo {nombre_pozo}",
      color_discrete_sequence=["orange"],
  )
  fig.update_traces(name="Gas (Mcf/d)", showlegend=True)
  fig.update_layout(
      xaxis_title="Fecha de Prueba",
      yaxis_title="Caudal de Gas (Mcf/d)",
      legend_title="Fluidos",
      template="plotly_white",
  )
  return fig

def graf_boxplots_fluidos_individuales(df, nombre_pozo):
  col_petro = [c for c in df.columns if "petroleo" in c or "petróleo" in c][0]
  col_agua = [c for c in df.columns if "agua" in c][0]
  col_gas = [c for c in df.columns if "gas" in c][0]

  pozo_formateado = (
      nombre_pozo if str(nombre_pozo).startswith("SCH-") else f"SCH-{nombre_pozo}"
  )

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  sns.boxplot(
      data=df, y=col_petro, color="tab:green", width=0.4, ax=axes[0]
  )
  axes[0].set_title(
      f"Distribución de Petróleo\nPozo {pozo_formateado}",
      fontsize=11,
      fontweight="bold",
  )
  axes[0].set_ylabel("Caudal (bbl/d)")
  axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

  sns.boxplot(data=df, y=col_agua, color="tab:blue", width=0.4, ax=axes[1])
  axes[1].set_title(
      f"Distribución de Agua\nPozo {pozo_formateado}",
      fontsize=11,
      fontweight="bold",
  )
  axes[1].set_ylabel("Caudal (bbl/d)")
  axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

  sns.boxplot(data=df, y=col_gas, color="tab:orange", width=0.4, ax=axes[2])
  axes[2].set_title(
      f"Distribución de Gas\nPozo {pozo_formateado}",
      fontsize=11,
      fontweight="bold",
  )
  axes[2].set_ylabel("Caudal (Mcf/d)")
  axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")

  plt.tight_layout()
  return fig


def heat_map_PAG(df, nombre_pozo):
  col_petro = [c for c in df.columns if "petróleo" in c or "petroleo" in c][0]
  col_agua = [c for c in df.columns if "agua" in c][0]
  col_gas = [c for c in df.columns if "gas" in c][0]

  df_fluidos = df[[col_petro, col_agua, col_gas]].copy()
  df_fluidos.columns = ["Petróleo (bbl/d)", "Agua (bbl/d)", "Gas (Mcf/d)"]
  matriz_corr = df_fluidos.corr()

  fig, ax = plt.subplots(figsize=(6, 5))
  sns.heatmap(
      matriz_corr,
      annot=True,
      fmt=".2f",
      cmap="coolwarm",
      vmin=-1,
      vmax=1,
      linewidths=1,
      ax=ax,
      cbar_kws={"label": "Coeficiente de Correlación"},
  )
  ax.set_title(
      f"Correlación entre Fluidos\nPozo {nombre_pozo}",
      fontsize=12,
      fontweight="bold",
      pad=15,
  )
  plt.tight_layout()
  return fig

def obtener_metricas_descriptivas(df):
  # Seleccionamos únicamente las columnas de tipo numérico para evitar conflictos con la fecha
  df_numerico = df.select_dtypes(include=["number"])
  return df_numerico.describe().T

# ==========================================
# 2. FUNCIONES DE VALIDACION TECNICA
# ==========================================

def funcion_validacion_petro(
    Pozo,
    Prof,
    QF,
    QP,
    T,
    API,
    prof_punzados,
    Nvl_Dina,
    Incl,
    DLS,
    BSW,
    u,
    GOR,
):
  # Definición de límites y condiciones de fallo para cada variable
  falla_prof = not (8000 <= Prof <= 12500)
  falla_qp = QP > 500
  falla_t = T > 500
  falla_api = API <= 8
  falla_punzados = not (7800 <= prof_punzados <= 11500)
  falla_nvl = not (3500 <= Nvl_Dina <= 9000)
  falla_incl = Incl > 35
  falla_dls = DLS >= 8
  falla_qf = not (50 <= QF <= 700)
  falla_bsw = BSW > 95
  falla_u = u > 600
  falla_gor = GOR > 1200

  # Lista de fallos para verificar si pasa todas las reglas
  fallos = [
      falla_prof,
      falla_qp,
      falla_t,
      falla_api,
      falla_punzados,
      falla_nvl,
      falla_incl,
      falla_dls,
      falla_qf,
      falla_bsw,
      falla_u,
      falla_gor,
  ]

  factibilidad = 'no' if any(fallos) else 'si'
  pozo_formateado = (
      f'SCH-{Pozo}' if not str(Pozo).startswith('') else str(Pozo)
  )
  n_filas = len(fallos)

  datos_validacion = pd.DataFrame({
      'Pozo': [pozo_formateado] * n_filas,
      'Variable': [
          'Profundidad',
          'Caudal de petróleo',
          'Temperatura',
          'Grado API',
          'Profundidad de punzados',
          'Nivel dinámico',
          'Inclinación',
          'Dog Leg Severity',
          'Caudal de fluido',
          'BSW',
          'Viscosidad',
          'GOR',
      ],
      'Valor Ingresado': [
          Prof,
          QP,
          T,
          API,
          prof_punzados,
          Nvl_Dina,
          Incl,
          DLS,
          QF,
          BSW,
          u,
          GOR,
      ],
      'Parámetro Delimitante (Límite)': [
          '8000 - 12500 ft',
          '<= 500 bbl/d',
          '<= 500 °F',
          '>= 9 API',
          '7800 - 11500 ft',
          '3500 - 9000 ft',
          '<= 35 °',
          '<= 8 °/100ft',
          '50 - 700 bbl/d',
          '<= 95 %',
          '<= 600 cp',
          '<= 1200 scf/bbl',
      ],
      'Cumple': ['no' if f else 'si' for f in fallos],
      'Factible Bombeo Mecánico': [factibilidad] * n_filas,
  })

  return datos_validacion


def formatear_con_unidades(row):
    """
    Función auxiliar para agregar unidades y formatear a 3 decimales
    los valores ingresados en la tabla de resultados.
    """
    var = row['Variable']
    val = row['Valor Ingresado']

    if var in ['Profundidad', 'Profundidad de punzados', 'Nivel dinámico']:
        return f"{val:.1f} ft"
    elif var in ['Caudal de petróleo', 'Caudal de fluido']:
        return f"{val:.1f} bbl/d"
    elif var == 'Temperatura':
        return f"{val:.1f} °F"
    elif var == 'Grado API':
        return f"{val:.1f} °API"
    elif var == 'BSW':
        return f"{val:.1f} %"
    elif var == 'Viscosidad':
        return f"{val:.1f} cp"
    elif var == 'GOR':
        return f"{val:.1f} scf/bbl"
    elif var in ['Inclinación', 'Dog Leg Severity']:
        return f"{val:.1f} °"
    else:
        try:
            return f"{val:.1f}"
        except:
            return str(val)

def color_cumple(val):
    """
    Función para aplicar color condicional a la columna 'Cumple'.
    """
    return (
        'background-color: #d4edda; color: #155724;'
        if val == 'si'
        else 'background-color: #f8d7da; color: #721c24'
    )

def Nivel_Dinamico(API, BSW, SGh2o, Pr, Hperf):
  SGoil = 141.5 / (API + 131.5)
  Por_BSW = BSW / 100
  SGfluido = SGoil * (1 - Por_BSW) + SGh2o * Por_BSW
  Gfluido = 0.433 * SGfluido
  N_static = Pr / Gfluido
  N_dinamic = Hperf - N_static
  return N_dinamic

# ==========================================
# 4. FUNCIONES DE CÁLCULO DE DIMENSIONAMIENTO
# ==========================================


def elasticdad_frecuencia(Varilla, D, L):
  varilla_dim = Varilla.copy()
  if Varilla["No"] == 44:
    Er = 0.000001990
    Fo = 1
    varilla_dim["Size_in"] = ["1/2"]
    varilla_dim["long_ft"] = [L]
  elif Varilla["No"] == 54:
    G_54 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "5/8", "1/2"],
        [1.06, 0.908, 0.00000668, 1.138, 0.446, 0.554],
        [1.25, 0.929, 0.00000633, 1.140, 0.495, 0.505],
        [1.50, 0.957, 0.00000584, 1.137, 0.565, 0.434],
        [1.75, 0.990, 0.00000525, 1.122, 0.646, 0.354],
        [2.00, 1.027, 0.00000460, 1.095, 0.737, 0.263],
        [2.25, 1.067, 0.00000391, 1.061, 0.834, 0.166],
        [2.50, 1.108, 0.00000318, 1.023, 1.000, 0.000],
    ])
    fila = G_54.index[G_54[0] == D][0]
    Er = G_54.iloc[fila, 2]
    Fo = G_54.iloc[fila, 3]
    varilla_dim["Size_in"] = G_54.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_54.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 55:
    Er = 0.000001270
    Fo = 1
    varilla_dim["Size_in"] = ["5/8"]
    varilla_dim["long_ft"] = [L]
  elif Varilla["No"] == 64:
    G_64 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "3/4", "5/8", "1/2"],
        [1.06, 0.908, 0.000001382, 1.229, 0.333, 0.332, 0.335],
        [1.25, 0.929, 0.000001319, 1.215, 0.372, 0.359, 0.269],
        [1.50, 0.957, 0.000001232, 1.184, 0.423, 0.404, 0.173],
        [1.75, 0.990, 0.000001141, 1.145, 0.474, 0.452, 0.074],
    ])
    fila = G_64.index[G_64[0] == D][0]
    Er = G_64.iloc[fila, 2]
    Fo = G_64.iloc[fila, 3]
    varilla_dim["Size_in"] = G_64.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_64.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 65:
    G_65 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "3/4", "5/8"],
        [1.06, 0.908, 0.000001138, 1.098, 0.344, 0.656],
        [1.25, 0.929, 0.000001127, 1.104, 0.373, 0.627],
        [1.50, 0.957, 0.000001110, 1.110, 0.418, 0.582],
        [1.75, 0.990, 0.000001090, 1.114, 0.469, 0.531],
        [2.00, 1.027, 0.000001070, 1.114, 0.520, 0.480],
        [2.25, 1.067, 0.000001045, 1.110, 0.584, 0.416],
        [2.50, 1.108, 0.000001018, 1.099, 0.652, 0.348],
        [2.75, 1.067, 0.000000990, 1.082, 0.725, 0.275],
        [3.25, 1.108, 0.000000930, 1.037, 0.881, 0.119],
    ])
    fila = G_65.index[G_65[0] == D][0]
    Er = G_65.iloc[fila, 2]
    Fo = G_65.iloc[fila, 3]
    varilla_dim["Size_in"] = G_65.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_65.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 66:
    Er = 0.000000883
    Fo = 1
    varilla_dim["Size_in"] = ["3/4"]
    varilla_dim["long_ft"] = [L]
  elif Varilla["No"] == 75:
    G_75 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "7/8", "3/4", "5/8"],
        [1.06, 1.566, 0.000000997, 1.191, 0.270, 0.274, 0.456],
        [1.25, 1.604, 0.000000973, 1.193, 0.294, 0.298, 0.408],
        [1.50, 1.664, 0.000000935, 1.189, 0.333, 0.333, 0.334],
        [1.75, 1.732, 0.000000892, 1.174, 0.378, 0.371, 0.251],
        [2.00, 1.803, 0.000000847, 1.151, 0.424, 0.413, 0.163],
        [2.25, 1.875, 0.000000801, 1.121, 0.469, 0.458, 0.073],
    ])
    fila = G_75.index[G_75[0] == D][0]
    Er = G_75.iloc[fila, 2]
    Fo = G_75.iloc[fila, 3]
    varilla_dim["Size_in"] = G_75.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_75.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 76:
    G_76 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "7/8", "3/4"],
        [1.06, 1.566, 0.000000816, 1.072, 0.285, 0.715],
        [1.25, 1.604, 0.000000812, 1.077, 0.306, 0.694],
        [1.50, 1.664, 0.000000804, 1.082, 0.338, 0.662],
        [1.75, 1.732, 0.000000795, 1.088, 0.375, 0.625],
        [2.00, 1.803, 0.000000785, 1.093, 0.417, 0.583],
        [2.25, 1.875, 0.000000774, 1.096, 0.465, 0.535],
        [2.50, 1.664, 0.000000764, 1.097, 0.502, 0.492],
        [2.75, 1.732, 0.000000751, 1.094, 0.565, 0.435],
        [3.25, 1.803, 0.000000722, 1.078, 0.687, 0.313],
        [3.50, 1.875, 0.000000609, 1.047, 0.823, 0.177],
    ])
    fila = G_76.index[G_76[0] == D][0]
    Er = G_76.iloc[fila, 2]
    Fo = G_76.iloc[fila, 3]
    varilla_dim["Size_in"] = G_76.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (np.round(L * G_76.iloc[fila, 4:], 3)).tolist()
  elif Varilla["No"] == 77:
    Er = 0.000000649
    Fo = 1
    varilla_dim["Size_in"] = ["7/8"]
    varilla_dim["long_ft"] = [L]
  elif Varilla["No"] == 85:
    G_85 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1", "7/8", "3/4", "5/8"],
        [1.06, 0.908, 0.000000873, 1.261, 0.222, 0.224, 0.224, 0.332],
        [1.25, 0.929, 0.000000841, 1.253, 0.239, 0.242, 0.243, 0.276],
        [1.50, 0.957, 0.000000791, 1.232, 0.267, 0.274, 0.268, 0.192],
        [1.75, 0.990, 0.000000738, 1.201, 0.296, 0.304, 0.295, 0.105],
    ])
    fila = G_85.index[G_85[0] == D][0]
    Er = G_85.iloc[fila, 2]
    Fo = G_85.iloc[fila, 3]
    varilla_dim["Size_in"] = G_85.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_85.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 86:
    G_86 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1", "7/8", "3/4"],
        [1.06, 1.566, 0.000000742, 1.151, 0.226, 0.230, 0.544],
        [1.25, 1.604, 0.000000732, 1.156, 0.243, 0.245, 0.512],
        [1.50, 1.664, 0.000000717, 1.162, 0.267, 0.270, 0.463],
        [1.75, 1.732, 0.000000699, 1.164, 0.294, 0.300, 0.406],
        [2.00, 1.803, 0.000000679, 1.161, 0.329, 0.332, 0.339],
        [2.25, 1.875, 0.000000656, 1.153, 0.369, 0.360, 0.271],
        [2.50, 1.664, 0.000000633, 1.138, 0.406, 0.397, 0.197],
        [2.75, 1.875, 0.000000610, 1.119, 0.445, 0.433, 0.122],
    ])
    fila = G_86.index[G_86[0] == D][0]
    Er = G_86.iloc[fila, 2]
    Fo = G_86.iloc[fila, 3]
    varilla_dim["Size_in"] = G_86.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_86.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 87:
    G_87 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1", "7/8"],
        [1.06, 1.566, 0.000000612, 1.055, 0.243, 0.757],
        [1.25, 1.604, 0.000000610, 1.058, 0.257, 0.743],
        [1.50, 1.664, 0.000000607, 1.062, 0.277, 0.723],
        [1.75, 1.732, 0.000000603, 1.066, 0.303, 0.697],
        [2.00, 1.803, 0.000000598, 1.071, 0.332, 0.668],
        [2.25, 1.875, 0.000000594, 1.075, 0.364, 0.636],
        [2.50, 1.664, 0.000000588, 1.079, 0.399, 0.601],
        [2.75, 1.732, 0.000000582, 1.082, 0.439, 0.561],
        [3.25, 1.803, 0.000000570, 1.084, 0.516, 0.484],
        [3.75, 1.875, 0.000000556, 1.078, 0.612, 0.388],
        [4.75, 1.875, 0.000000522, 1.038, 0.836, 0.164],
    ])
    fila = G_87.index[G_87[0] == D][0]
    Er = G_87.iloc[fila, 2]
    Fo = G_87.iloc[fila, 3]
    varilla_dim["Size_in"] = G_87.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_87.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 88:
    Er = 0.000000497
    Fo = 1
    varilla_dim["Size_in"] = ["1"]
    varilla_dim["long_ft"] = [L]
  elif Varilla["No"] == 96:
    G_96 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1-1/8", "1", "7/8", "3/4"],
        [1.06, 1.566, 0.000000670, 1.222, 0.191, 0.191, 0.195, 0.423],
        [1.25, 1.604, 0.000000655, 1.224, 0.205, 0.205, 0.207, 0.383],
        [1.50, 1.664, 0.000000633, 1.223, 0.224, 0.225, 0.228, 0.323],
        [1.75, 1.732, 0.000000606, 1.213, 0.248, 0.251, 0.251, 0.251],
        [2.00, 1.803, 0.000000578, 1.196, 0.271, 0.279, 0.274, 0.176],
        [2.25, 1.875, 0.000000549, 1.172, 0.269, 0.307, 0.298, 0.098],
    ])
    fila = G_96.index[G_96[0] == D][0]
    Er = G_96.iloc[fila, 2]
    Fo = G_96.iloc[fila, 3]
    varilla_dim["Size_in"] = G_96.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_96.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 97:
    G_97 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1-1/8", "1", "7/8"],
        [1.06, 0.908, 0.000000568, 1.120, 0.196, 0.200, 0.604],
        [1.25, 0.929, 0.000000563, 1.124, 0.208, 0.212, 0.580],
        [1.50, 0.957, 0.000000556, 1.131, 0.225, 0.230, 0.545],
        [1.75, 0.990, 0.000000548, 1.137, 0.245, 0.250, 0.505],
        [2.00, 1.027, 0.000000538, 1.141, 0.268, 0.274, 0.458],
        [2.25, 1.067, 0.000000528, 1.143, 0.294, 0.302, 0.404],
        [2.50, 1.108, 0.000000515, 1.141, 0.325, 0.331, 0.344],
        [2.75, 1.067, 0.000000503, 1.135, 0.361, 0.353, 0.286],
        [3.25, 1.108, 0.000000475, 1.111, 0.429, 0.419, 0.152],
    ])
    fila = G_97.index[G_97[0] == D][0]
    Er = G_97.iloc[fila, 2]
    Fo = G_97.iloc[fila, 3]
    varilla_dim["Size_in"] = G_97.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_97.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 98:
    G_98 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1-1/8", "1"],
        [1.06, 1.566, 0.000000475, 1.043, 0.212, 0.788],
        [1.25, 1.604, 0.000000474, 1.045, 0.222, 0.778],
        [1.50, 1.664, 0.000000472, 1.048, 0.238, 0.762],
        [1.75, 1.732, 0.000000470, 1.051, 0.257, 0.743],
        [2.00, 1.803, 0.000000468, 1.055, 0.277, 0.723],
        [2.25, 1.875, 0.000000465, 1.058, 0.301, 0.699],
        [2.50, 1.664, 0.000000463, 1.062, 0.327, 0.673],
        [2.75, 1.732, 0.000000460, 1.066, 0.356, 0.644],
        [3.25, 1.803, 0.000000453, 1.071, 0.422, 0.578],
        [3.75, 1.875, 0.000000445, 1.074, 0.497, 0.503],
        [4.75, 1.875, 0.000000428, 1.064, 0.657, 0.343],
    ])
    fila = G_98.index[G_98[0] == D][0]
    Er = G_98.iloc[fila, 2]
    Fo = G_98.iloc[fila, 3]
    varilla_dim["Size_in"] = G_98.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_98.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 99:
    Er = 0.000000393
    Fo = 1
    varilla_dim["Size_in"] = ["1-1/8"]
    varilla_dim["long_ft"] = [L]
  elif Varilla["No"] == 107:
    G_107 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1-1/4", "1-1/8", "1", "7/8"],
        [1.06, 1.566, 0.000000524, 1.184, 0.169, 0.168, 0.171, 0.491],
        [1.25, 1.604, 0.000000517, 1.189, 0.179, 0.178, 0.180, 0.463],
        [1.50, 1.664, 0.000000506, 1.195, 0.194, 0.192, 0.195, 0.419],
        [1.75, 1.732, 0.000000494, 1.197, 0.210, 0.210, 0.212, 0.368],
        [2.00, 1.803, 0.000000480, 1.195, 0.227, 0.228, 0.231, 0.314],
        [2.25, 1.875, 0.000000464, 1.187, 0.250, 0.250, 0.250, 0.250],
        [2.50, 1.664, 0.000000447, 1.174, 0.269, 0.277, 0.271, 0.182],
        [2.75, 1.875, 0.000000430, 1.156, 0.292, 0.302, 0.293, 0.113],
    ])
    fila = G_107.index[G_107[0] == D][0]
    Er = G_107.iloc[fila, 2]
    Fo = G_107.iloc[fila, 3]
    varilla_dim["Size_in"] = G_107.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_107.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 108:
    G_108 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1-1/4", "1-1/8", "1"],
        [1.06, 1.566, 0.000000447, 1.097, 0.173, 0.178, 0.649],
        [1.25, 1.604, 0.000000445, 1.101, 0.182, 0.186, 0.632],
        [1.50, 1.664, 0.000000441, 1.106, 0.194, 0.199, 0.607],
        [1.75, 1.732, 0.000000437, 1.111, 0.209, 0.214, 0.577],
        [2.00, 1.803, 0.000000432, 1.117, 0.226, 0.230, 0.544],
        [2.25, 1.875, 0.000000427, 1.121, 0.245, 0.250, 0.505],
        [2.50, 1.664, 0.000000421, 1.124, 0.265, 0.272, 0.463],
        [2.75, 1.732, 0.000000415, 1.126, 0.287, 0.297, 0.416],
        [3.25, 1.803, 0.000000400, 1.123, 0.345, 0.339, 0.316],
        [3.75, 1.875, 0.000000383, 1.108, 0.406, 0.395, 0.199],
    ])
    fila = G_108.index[G_108[0] == D][0]
    Er = G_108.iloc[fila, 2]
    Fo = G_108.iloc[fila, 3]
    varilla_dim["Size_in"] = G_108.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_108.iloc[fila, 4:]).tolist()
  elif Varilla["No"] == 109:
    G_109 = pd.DataFrame([
        ["Dia", "Wr", "Er", "Fr", "1-1/4", "1-1/8"],
        [1.06, 1.566, 0.000000378, 1.035, 0.189, 0.811],
        [1.25, 1.604, 0.000000378, 1.036, 0.196, 0.804],
        [1.50, 1.664, 0.000000377, 1.038, 0.207, 0.793],
        [1.75, 1.732, 0.000000376, 1.040, 0.221, 0.779],
        [2.00, 1.803, 0.000000375, 1.043, 0.237, 0.763],
        [2.25, 1.875, 0.000000374, 1.046, 0.254, 0.746],
        [2.50, 1.664, 0.000000372, 1.048, 0.272, 0.728],
        [2.75, 1.732, 0.000000371, 1.051, 0.294, 0.706],
        [3.25, 1.803, 0.000000367, 1.057, 0.342, 0.658],
        [3.75, 1.875, 0.000000363, 1.063, 0.399, 0.601],
        [4.75, 1.875, 0.000000354, 1.066, 0.515, 0.485],
    ])
    fila = G_109.index[G_109[0] == D][0]
    Er = G_109.iloc[fila, 2]
    Fo = G_109.iloc[fila, 3]
    varilla_dim["Size_in"] = G_109.iloc[0, 4:].tolist()
    varilla_dim["long_ft"] = (L * G_109.iloc[fila, 4:]).tolist()
  return Er, Fo, varilla_dim


def Calculo_Wr(Varilla, G):
  Rod_Size = pd.DataFrame([
      ["1/2", 0.196, 0.726, 0.000001990],
      ["5/8", 0.307, 1.135, 0.000001270],
      ["3/4", 0.442, 1.634, 0.000000883],
      ["7/8", 0.601, 2.224, 0.000000649],
      ["1", 0.785, 2.904, 0.000000497],
      ["1 1/8", 0.994, 3.676, 0.000000393],
      ["1 1/4", 1.227, 4.538, 0.00000318],
  ])
  Weight_air = []
  for gros in Varilla["Size_in"]:
    fila = Rod_Size.index[Rod_Size[0] == gros][0]
    Wa_gros = float(Rod_Size.iloc[fila, 2])
    Weight_air.append(Wa_gros)
  result = []
  for num in range(len(Weight_air)):
    l_g = Varilla["long_ft"][num] * Weight_air[num]
    result.append(float(l_g))
  Wr = sum(result)
  Wrf = Wr * (1 - 0.128 * G)
  return Wr, Wrf


def Nivel_Dinamico_SGfluido(API, BSW, Gh2o, Pr, Hperf):
  SGoil = 141.5 / (API + 131.5)
  Por_BSW = BSW / 100
  SGfluido = SGoil * (1 - Por_BSW) + Gh2o * Por_BSW
  Gfluido = 0.433 * SGfluido
  N_static = Pr / Gfluido
  N_dinamic = Hperf - N_static
  return N_dinamic, SGfluido


def adimensionales(G, D, H, Er, L, S, N, Fc):
  Fo = 0.34 * G * D**2 * H
  Fo_Skr = (Fo * Er * L) / S
  N_No = (N * L) / 245000
  C_N_No = N_No / Fc
  return Fo_Skr, N_No, C_N_No


def simulador_completo_api_11l(
    n_no: float,
    n_no_prima: float,
    fo_skr: float,
    N=None,
    w_rf=None,
    s=None,
    Er=None,
    L=None,
):
  X_stroke = n_no_prima
  X_loads = n_no
  Y = fo_skr

  def evaluar_polinomio(c, x_val):
    return (
        c[0]
        + c[1] * x_val
        + c[2] * Y
        + c[3] * (x_val**2)
        + c[4] * (Y**2)
        + c[5] * x_val * Y
        + c[6] * (x_val**3)
        + c[7] * (Y**3)
        + c[8] * (x_val**2) * Y
        + c[9] * x_val * (Y**2)
    )

  coef_sp_s = [
      1.0003,
      -0.0543,
      -1.0211,
      0.6412,
      0.1874,
      -0.2105,
      -0.3211,
      -0.0452,
      0.5123,
      -0.3114,
  ]
  coef_f1 = [
      1.0021,
      0.1254,
      1.0423,
      0.8122,
      -0.1543,
      0.3121,
      -0.2104,
      0.0312,
      0.6421,
      -0.1245,
  ]
  coef_torque = [
      0.0512,
      0.2412,
      0.4124,
      0.5211,
      0.0841,
      0.6123,
      -0.1874,
      -0.0121,
      0.4102,
      -0.0954,
  ]
  coef_f2 = [
      1.0014,
      -0.3842,
      -0.9241,
      -0.4211,
      0.1142,
      0.2843,
      0.1241,
      -0.0211,
      -0.3214,
      0.0841,
  ]
  coef_f3 = [
      0.0054,
      0.8241,
      0.1142,
      0.2411,
      0.0342,
      1.1241,
      -0.1142,
      -0.0051,
      0.2142,
      -0.0412,
  ]

  resultados = {
      "Sp/S": round(evaluar_polinomio(coef_sp_s, X_stroke), 4),
      "F1/SKr (Carga Máx Adim)": round(
          evaluar_polinomio(coef_f1, X_loads) - 0.9, 4
      ),
      "F2/SKr (Carga Mín Adim)": round(
          evaluar_polinomio(coef_f2, X_loads) - 0.505, 4
      ),
      "2T/S^2Kr (Torque Adim)": round(
          evaluar_polinomio(coef_torque, X_loads), 4
      ),
      "F3/SKr (Potencia Adim)": round(
          evaluar_polinomio(coef_f3, X_loads), 4
      ),
  }

  if all(v is not None for v in [N, w_rf, s, Er, L]):
    skr = s * 1 / (Er * L)
    resultados["Wmax (lb)"] = round(
        w_rf + (skr * resultados["F1/SKr (Carga Máx Adim)"]), 2
    )
    resultados["Wmin (lb)"] = round(
        w_rf - (skr * resultados["F2/SKr (Carga Mín Adim)"]), 2
    )

  return resultados


def calcular_ta_ajuste(
    n_no_prima: float, fo_skr: float, wrf: float, S: float, Er: float, L: float
) -> tuple[float, float]:
  m0, m1, m2, m3, m4, m5, m6, m7, m8, m9 = [
      0.0125,
      0.4521,
      -0.1245,
      1.8412,
      0.0512,
      1.2145,
      -0.8412,
      -0.0124,
      2.1045,
      -0.4123,
  ]
  Xv, Yv = n_no_prima, fo_skr
  x_porcentaje = (
      m0
      + m1 * Xv
      + m2 * Yv
      + m3 * Xv**2
      + m4 * Yv**2
      + m5 * Xv * Yv
      + m6 * Xv**3
      + m7 * Yv**3
      + m8 * Xv**2 * Yv
      + m9 * Xv * Yv**2
  )
  ta = 1.0 + (x_porcentaje * ((((wrf * Er * L) / S) - 0.3) / 10))
  return round(ta, 4), round(x_porcentaje, 2)


def Calcular_PD(S, N, D, Sp_S):
  Sp_pre = S
  Ep = 1
  PDp = 0.1166 * Sp_pre * N * D**2 * Ep
  Sp = Sp_S * S
  PD = 0.1166 * Sp * N * D**2 * Ep
  return PDp, PD


def Calculo_PT_PRHP_HPnp(T2_S2Kr, F3_SKr, S, Er, L, Ta, N, Motor):
  PT = T2_S2Kr * (S / (Er * L)) * (S / 2) * Ta
  PRHP = F3_SKr * (S / (Er * L)) * S * N * 0.00000253
  CLF = 1.375 if Motor == "NEMA D" else 1.897
  HPnp = (PRHP / 0.63) * CLF
  return PT, PRHP, HPnp

# ==========================================
# 5. FUNCIONES DE VALIDACION ECONOMICA
# ==========================================

def gasto_energia(HP_motor):
  Kw_re = HP_motor * 0.7457 * 0.8 / 0.91
  Costo_E = np.round(Kw_re * 720 * 0.09, 3)
  return Costo_E


def tabla_VAN(n, Q, price_bbl, Inv_inicial, cost_bbl, Gastos_mensuales, i, Dec):
  tiempo = np.arange(1, n + 1)

  # Cálculo de producción con declinación (Alineado exactamente a 'n' meses)
  produccion_diaria = [Q]  # El mes 1 arranca con Q
  for mes in range(2, n + 1):
    produccion_anterior = produccion_diaria[-1]
    produccion_actual = produccion_anterior - (produccion_anterior * Dec)
    produccion_diaria.append(produccion_actual)

  Prod = np.array(produccion_diaria)
  Prod_m = Prod * 30.4
  Ing_ventas = Prod_m * price_bbl

  # Cálculo de egresos iniciales
  gasto_I = 0
  for valor in Inv_inicial.values():
    gasto_I += valor

  gasto_fijo_mensual = sum(Gastos_mensuales.values())

  costos = np.array([gasto_I + Gastos_mensuales["renta"]] + [
      (cost_bbl * q) + gasto_fijo_mensual for q in Prod_m
  ])

  Ing_ventas_total = np.insert(Ing_ventas, 0, 0.0)
  FNC = Ing_ventas_total - costos

  # Cálculo del FNC actualizado
  FNC_act = []
  for term in range(len(FNC)):
    if FNC[term] <= 0:
      FNC_act.append(float(np.round(FNC[term], 3)))
    else:
      FNC_act.append(float(np.round(FNC[term] / ((1 + i) ** term), 3)))

  # Sumatoria acumulada del FNC actualizado
  sum_FN_act = []
  for u in range(len(FNC_act)):
    if u == 0:
      sum_FN_act.append(float(np.round(FNC_act[0], 3)))
    else:
      valor_acumulado = FNC_act[u] + sum_FN_act[u - 1]
      sum_FN_act.append(float(np.round(valor_acumulado, 3)))

  datos_VAN = pd.DataFrame({
      "Tiempo (meses)": tiempo,
      "Recuperacion produccion (bbld)": np.round(Prod, 2),
      "Produccion por mes (bblm)": np.round(Prod_m, 2),
      "Ingreso por ventas (USD)": np.round(Ing_ventas, 2),
      "Egresos (USD)": np.round(costos[1:], 2),
      "Flujo neto de caja (USD)": np.round(FNC[1:], 2),
      "FNC actualizado (USD)": np.round(FNC_act[1:], 2),
      "sumatoria FNC act (USD)": np.round(sum_FN_act[1:], 2),
  })
  return datos_VAN

def calcular_tir(fnc_vector):
  """Función robusta para calcular la TIR usando numpy_financial."""
  try:
    tir_val = npf.irr(fnc_vector)
    if np.isnan(tir_val) or np.isinf(tir_val):
      return np.nan
    return tir_val
  except Exception:
    return np.nan


def VAN_TIR_BC_PB_comparativa(tabla, tabla_R, i, n, Pozo, Inv_inicial):

  def obtener_payback_exacto(df):
    fnc_vector = df["Flujo neto de caja (USD)"].values
    acumulado = df["sumatoria FNC act (USD)"].values
    t_arr = np.insert(df["Tiempo (meses)"].values, 0, 0)
    f_arr = np.insert(acumulado, 0, 0.0)

    for idx in range(len(f_arr) - 1):
      if f_arr[idx] <= 0 and f_arr[idx + 1] > 0:
        x1, x2 = t_arr[idx], t_arr[idx + 1]
        y1, y2 = f_arr[idx], f_arr[idx + 1]
        if y2 != y1:
          return x1 - y1 * (x2 - x1) / (y2 - y1)
        return x2
    return np.nan

  # --- INVERSIÓN INICIAL REAL Y TOTAL CONFIGURADA POR EL USUARIO ---
  inversion_inicial = sum(Inv_inicial.values())

  def procesar_indicadores(df_tabla):
    fnc_neto = df_tabla["Flujo neto de caja (USD)"].values
    ingresos = df_tabla["Ingreso por ventas (USD)"].values

    # Vector para la TIR: Mes 0 con la inversión total negativa, seguido de los flujos netos
    vector_tir = np.insert(fnc_neto, 0, -inversion_inicial)

    # 1. VAN (Suma de FNC actualizados)
    van = sum(df_tabla["FNC actualizado (USD)"])

    # 2. TIR
    tir_val = calcular_tir(vector_tir)
    TIR_porcentaje = np.round(tir_val * 100, 2) if not np.isnan(tir_val) else np.nan

    # 3. Relación Beneficio/Costo (B/C) = Beneficios Actualizados / Inversión Inicial
    beneficios_actualizados = sum(
        [ing / ((1 + i) ** t) for t, ing in enumerate(ingresos, start=1)]
    )
    relacion_bc = (
        np.round(beneficios_actualizados / inversion_inicial, 2)
        if inversion_inicial > 0
        else 0
    )

    # 4. Ganancia Neta y ROI (%) = (Ganancia Neta Total / Inversión Inicial) * 100
    ganancia_neta = sum(fnc_neto)
    roi = (
        np.round((ganancia_neta / inversion_inicial) * 100, 2)
        if inversion_inicial > 0
        else 0
    )

    # 5. Payback exacto
    payback = obtener_payback_exacto(df_tabla)

    # 6. Factibilidad lógica
    factibilidad = (
        "si"
        if (
            van > 0
            and not np.isnan(tir_val)
            and tir_val > i
            and relacion_bc > 1
        )
        else "no"
    )

    return van, TIR_porcentaje, relacion_bc, factibilidad, payback, roi

  # Procesar Caso 1 (Propios)
  van_1, TIR_1, relacion_bc_1, factibilidad_1, payback_1, ROI_1 = procesar_indicadores(tabla)

  # Procesar Caso 2 (Rentados)
  van_2, TIR_2, relacion_bc_2, factibilidad_2, payback_2, ROI_2 = procesar_indicadores(tabla_R)

  datos_economicos = pd.DataFrame({
      "Pozo": [Pozo, Pozo],
      "Propiedad": ["Equipos Propios", "Equipos Rentados"],
      "VAN (USD)": [van_1, van_2],
      "TIR (%)": [TIR_1, TIR_2],
      "B/C (USD)": [relacion_bc_1, relacion_bc_2],
      "Factibilidad": [factibilidad_1, factibilidad_2],
      "Payback (meses)": [
          np.round(payback_1, 2) if not np.isnan(payback_1) else np.nan,
          np.round(payback_2, 2) if not np.isnan(payback_2) else np.nan,
      ],
      "ROI (%)": [ROI_1, ROI_2],
  })
  return datos_economicos

def VAN_TIR_BC_PB_unico(tabla, i, n, Pozo, Inv_inicial, tipo_equipo="Equipo Propio"):
  """Calcula los indicadores económicos (VAN, TIR, B/C, Payback, ROI, Factibilidad)

  para un solo escenario (modo único o personalizado).
  """

  def obtener_payback_exacto(df):
    fnc_vector = df["Flujo neto de caja (USD)"].values
    acumulado = df["sumatoria FNC act (USD)"].values
    t_arr = np.insert(df["Tiempo (meses)"].values, 0, 0)
    f_arr = np.insert(acumulado, 0, 0.0)

    for idx in range(len(f_arr) - 1):
      if f_arr[idx] <= 0 and f_arr[idx + 1] > 0:
        x1, x2 = t_arr[idx], t_arr[idx + 1]
        y1, y2 = f_arr[idx], f_arr[idx + 1]
        if y2 != y1:
          return x1 - y1 * (x2 - x1) / (y2 - y1)
        return x2
    return np.nan

  # Inversión inicial total
  inversion_inicial = sum(Inv_inicial.values())

  fnc_neto = tabla["Flujo neto de caja (USD)"].values
  ingresos = tabla["Ingreso por ventas (USD)"].values

  # Vector para la TIR: Mes 0 con la inversión total negativa, seguido de los flujos netos
  vector_tir = np.insert(fnc_neto, 0, -inversion_inicial)

  # 1. VAN (Suma de FNC actualizados)
  van = sum(tabla["FNC actualizado (USD)"])

  # 2. TIR
  tir_val = calcular_tir(vector_tir)
  TIR_porcentaje = np.round(tir_val * 100, 2) if not np.isnan(tir_val) else np.nan

  # 3. Relación Beneficio/Costo (B/C)
  beneficios_actualizados = sum(
      [ing / ((1 + i) ** t) for t, ing in enumerate(ingresos, start=1)]
  )
  relacion_bc = (
      np.round(beneficios_actualizados / inversion_inicial, 2)
      if inversion_inicial > 0
      else 0
  )

  # 4. Ganancia Neta y ROI (%)
  ganancia_neta = sum(fnc_neto)
  roi = (
      np.round((ganancia_neta / inversion_inicial) * 100, 2)
      if inversion_inicial > 0
      else 0
  )

  # 5. Payback exacto
  payback = obtener_payback_exacto(tabla)

  # 6. Factibilidad lógica
  factibilidad = (
      "si"
      if (van > 0 and not np.isnan(tir_val) and tir_val > i and relacion_bc > 1)
      else "no"
  )

  # Retornamos un DataFrame de una sola fila (o un formato de tabla limpia para Streamlit)
  datos_economicos_unico = pd.DataFrame({
      "Pozo": [Pozo],
      "Configuración": [tipo_equipo],
      "VAN (USD)": [van],
      "TIR (%)": [TIR_porcentaje],
      "B/C (USD)": [relacion_bc],
      "Factibilidad": [factibilidad],
      "Payback (meses)": [
          np.round(payback, 2) if not np.isnan(payback) else np.nan
      ],
      "ROI (%)": [roi],
  })

  return datos_economicos_unico

def generar_grafico_fnc(tabla_base, tabla_renta=None, modo="unico", pozo="Pozo"):
  fig = go.Figure()

  if modo == "unico":
    fig.add_trace(
        go.Scatter(
            x=tabla_base["Tiempo (meses)"],
            y=tabla_base["sumatoria FNC act (USD)"],
            mode="lines+markers",
            name="FNC Acumulado Actualizado",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.update_layout(
        title=f"Evolución del Flujo Neto de Caja Acumulado - {pozo}",
        xaxis_title="Tiempo (meses)",
        yaxis_title="FNC Acumulado Actualizado (USD)",
        template="plotly_white",
    )
  else:
    fig.add_trace(
        go.Scatter(
            x=tabla_base["Tiempo (meses)"],
            y=tabla_base["sumatoria FNC act (USD)"],
            mode="lines+markers",
            name="Equipos Propios",
            line=dict(color="#2ca02c", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tabla_renta["Tiempo (meses)"],
            y=tabla_renta["sumatoria FNC act (USD)"],
            mode="lines+markers",
            name="Equipos Rentados",
            line=dict(color="#ff7f0e", width=3),
        )
    )
    fig.update_layout(
        title=f"Comparativa FNC Acumulado (Propios vs Rentados) - {pozo}",
        xaxis_title="Tiempo (meses)",
        yaxis_title="FNC Acumulado Actualizado (USD)",
        template="plotly_white",
    )

  return fig

# ==========================================
# PROGRAMA DEL APLICATIVO
# ==========================================

# ==========================================
#  --- SECCIÓN: INICIO ---
# ==========================================

if selected == "Inicio":

    # --- HERO SECTION ---
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0e1117, #1f2933);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 30px;
    ">
        <h1 style="color:#f39c12; font-size:42px; margin-bottom:10px;">
            Sucker Rod Pump Optimizer
        </h1>
        <p style="color:#d1d5db; font-size:18px; max-width:900px;">
            Solución digital interactiva para el diagnóstico, simulación y optimización integral 
            de sistemas de levantamiento artificial por bombeo mecánico.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- IMAGEN DEBAJO DEL HERO ---
    st.image(
        "campo_srp.png",
        use_container_width=True
    )

    # --- MÉTRICAS RESUMEN ---
    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Módulos Activos", "6")
    col2.metric("🛢️ Campo", "Sacha - Bloque 60")
    col3.metric("📈 Enfoque", "Ingeniería de Producción")

    st.markdown("---")

    # --- TARJETAS DE MÓDULOS ---
    st.markdown("## 🔎 Módulos Disponibles - Flujo de trabajo recomendado")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("""
        <div class="stMetric">
        <h4>📊 Análisis de datos</h4>
        <p style="font-size:14px;">
        Visualización histórica de producción de petróleo, agua y gas por pozo.
        Identificación de tendencias y comportamiento productivo.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stMetric">
        <h4>📋 Validación técnica</h4>
        <p style="font-size:14px;">
        Análisis de sensibilidad operacional y validación de parámetros de campo para 
        el soporte en la toma de decisiones de levantamiento artificial.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stMetric">
        <h4>🎯 Análisis nodal</h4>
        <p style="font-size:14px;">
        Evaluación del índice de productividad e integración de curvas IPR–VLP para el 
        análisis del punto de operación, estimación del AOF y el nivel dinámico, 
        con el fin de determinar la configuración óptima de los equipos de fondo.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="stMetric">
        <h4>⚙️ Dimensionamiento de equipos</h4>
        <p style="font-size:14px;">
        Análisis matemático para la selección, dimensionamiento y optimización técnica y
        energética de los equipos de subsuelo y superficie en el sistema de 
        bombeo mecánico.
          de bombeo mecánico.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="stMetric">
        <h4>💵 Validación económica</h4>
        <p style="font-size:14px;">
        Evaluación de la viabilidad económica de la implementación del bombeo mecánico 
        mediante indicadores financieros (VAN, TIR y Payback) para maximizar la 
        rentabilidad del proyecto.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        > 💡 Este procedimiento simula la metodología de ingeniería aplicada en campo para 
        el diseño y optimización de proyectos de bombeo mecánico.
        """)

    st.markdown("---")

    # --- FLUJO RECOMENDADO ---
    st.markdown("## 🧭 Informacion")

    # --- INFORMACIÓN DEL CAMPO SACHA ---
    col_img, col_txt = st.columns([1, 1])

    with col_img:
        st.image("sacha.png", use_container_width=True)

    with col_txt:
        st.markdown("## INFORMACIÓN CAMPO SACHA (ECUADOR)")
        st.markdown(
            "El **Campo Sacha** (Bloque 60), conocido como la *'Joya de la Corona'* de"
            " la producción petrolera en la cuenca Oriente de Ecuador, destaca por"
            " sus formaciones productivas de areniscas de alta permeabilidad como los"
            " complejos **T-Basal y Hollín**. Debido a su madurez operativa y"
            " declinación natural de presión, la implementación masiva de sistemas"
            " de levantamiento artificial —como el bombeo mecánico (SRP)— resulta"
            " fundamental para optimizar la extracción de crudo y maximizar el factor"
            " de recobro económico del yacimiento."
        )

    st.markdown("---")

    st.markdown("""
    <h2 style="font-size:36px; color:#f39c12;">
        ¿Qué hace un ingeniero en producción?
    </h2>
    """, unsafe_allow_html=True)

    main_col1, main_col2 = st.columns([2, 2], gap="large")

    # ---------------- INPUTS ----------------
    with main_col1:
        with st.container(border=True):
            st.markdown("# QUE SON LOS BALANCINES PETROLEROS")
            st.markdown("## rompiendo mitos, conoce los balancines petroleros"
                        ", la herramienta maestra de la industria petrolera mundial")
            # --- VIDEO LOCAL ---
            st.video("https://www.youtube.com/watch?v=Oxjzj6n636I")
    with main_col2:
        with st.container(border=True):
            st.markdown("# QUE ES LA INGERNERIA PETROLERA")
            st.markdown("## Conoce los diferentes campos y oportunidades que ofrece la ingeneria petrolera")
            # --- VIDEO LOCAL ---
            st.video("petroleos.mp4")

    # --- MENSAJE FINAL ---
    st.info(
        "📌 Esta herramienta está diseñada para análisis técnico, soporte a decisiones "
        "operacionales y entrenamiento en ingeniería de producción."
    )
    st.info(
        "📌 Este aplicativo fue desarrollado por Marco Aspiazu Gonzalez (Sanyu) Estudiante de la carrera de petroleos en la ESPOL."
    )

# ==========================================
# 1. SECCIÓN DE ANALISIS DE DATOS
# ==========================================
if selected == "Análisis de datos":
  st.title("🛢️ Análisis Exploratorio y Depuración de Pozos")

  st.markdown(
      "Selecciona un pozo predeterminado del Campo Sacha o carga tu propio"
      " archivo Excel para realizar la limpieza y análisis exploratorio."
  )

  origen_datos = st.radio(
      "Selecciona la fuente de datos:",
      [
          "Pozos Predeterminados (Campo Sacha)",
          "Subir Archivo Personalizado (Excel)",
      ],
      horizontal=True,
  )

  df_pozo = None
  nombre_pozo = ""

  if origen_datos == "Pozos Predeterminados (Campo Sacha)":
    pozos_disponibles = ["156", "133R1", "132", "217", "464"]
    pozo_seleccionado = st.selectbox(
        "Elige el Pozo del Campo Sacha:", pozos_disponibles
    )
    nombre_pozo = f"SCH-{pozo_seleccionado}"

    archivo_predeterminado = os.path.join(
        "Data", "HISTORIALES (con Pwf, Pws).xlsx"
    )

    if os.path.exists(archivo_predeterminado):
      try:
        df_raw = pd.read_excel(
            archivo_predeterminado, sheet_name=pozo_seleccionado
        )
        df_pozo = Depuracion_produccion(df_raw)
      except Exception as e:
        st.error(f"Error al leer la hoja del pozo: {e}")
    else:
      st.warning(
          f"No se encontró el archivo en la ruta: {archivo_predeterminado}"
      )

  else:
    archivo_subido = st.file_uploader(
        "Carga tu archivo Excel de historiales", type=["xlsx"]
    )
    if archivo_subido is not None:
      excel_file = pd.ExcelFile(archivo_subido)
      hojas = excel_file.sheet_names
      hoja_elegida = st.selectbox("Selecciona la hoja del pozo:", hojas)
      nombre_pozo = st.text_input(
          "Nombre o identificador del pozo:", hoja_elegida
      )

      try:
        df_raw = pd.read_excel(archivo_subido, sheet_name=hoja_elegida)
        df_pozo = Depuracion_produccion(df_raw)
      except Exception as e:
        st.error(f"Error al procesar el archivo subido: {e}")

  # ==========================================
  # VISUALIZACIÓN Y BOTONES DE ACCIÓN
  # ==========================================
  if df_pozo is not None:
    st.success(
        f"¡Datos cargados y depurados exitosamente para el pozo {nombre_pozo}!"
    )

    with st.expander("🔍 Ver DataFrame Depurado"):
      st.dataframe(df_pozo, use_container_width=True)

    st.markdown("### 📊 Generar Gráficos de Análisis Exploratorio")
    col1, col2 = st.columns(2)

    with col1:
      btn_nulos = st.button(
          "🗺️ Mapa de Valores Faltantes", use_container_width=True
      )
      btn_lineas = st.button(
          "📈 Historial de Producción", use_container_width=True
      )

    with col2:
      btn_box = st.button(
          "📦 Diagramas de Caja (Boxplots)", use_container_width=True
      )
      btn_corr = st.button("🔥 Mapa de Calor de Correlación", use_container_width=True)

      # NUEVO BOTÓN DE RESUMEN ESTADÍSTICO
      btn_stats = st.button(
          "📋 Ver Resumen Estadístico (Describe)", use_container_width=True
      )

    if btn_nulos:
      st.markdown(f"#### Mapa de Valores Faltantes - Pozo {nombre_pozo}")
      fig = grafico_nulos(df_pozo, nombre_pozo)
      st.pyplot(fig)
      plt.close(fig)

    if btn_lineas:
        st.markdown(f"#### Historial de Producción - Pozo {nombre_pozo}")

        # Gráfico 1: Líquidos (Petróleo y Agua)
        fig_liq = grafico_linea_liquidos_plotly(df_pozo, nombre_pozo)
        st.plotly_chart(fig_liq, use_container_width=True)

        # Gráfico 2: Gas individual
        fig_gas = grafico_linea_gas_plotly(df_pozo, nombre_pozo)
        st.plotly_chart(fig_gas, use_container_width=True)

    if btn_box:
      st.markdown(
          f"#### Distribución Estadística de Fluidos - Pozo {nombre_pozo}"
      )
      fig = graf_boxplots_fluidos_individuales(df_pozo, nombre_pozo)
      st.pyplot(fig)
      plt.close(fig)

    if btn_corr:
      st.markdown(f"#### Matriz de Correlación - Pozo {nombre_pozo}")
      fig = heat_map_PAG(df_pozo, nombre_pozo)
      st.pyplot(fig)
      plt.close(fig)

    if btn_stats:
        st.markdown(f"#### 📋 Resumen Estadístico - Pozo {nombre_pozo}")
        df_stats = obtener_metricas_descriptivas(df_pozo)
        st.dataframe(df_stats, use_container_width=True)

  else:
    st.info(
        "👆 Selecciona un pozo de la lista o carga un archivo para comenzar con"
        " el análisis."
    )


# ==========================================
# 2. LÓGICA DE LA INTERFAZ (Sección Validación Técnica)
# ==========================================

# Simulación de la selección de navegación (si usas option_menu u otro)
#selected = 'Validación técnica'

if selected == 'Validación técnica':
    st.markdown('## 📋 Validación Técnica y Operacional')
    st.markdown(
        'Ingrese los parámetros del pozo para evaluar la factibilidad técnica de'
        ' la implementación del sistema de levantamiento artificial por bombeo'
        ' mecánico.'
    )

    st.subheader('⚙️ Parámetros del Pozo (Campo Sacha)')

    # Dividimos las entradas en 3 columnas equilibradas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📍 Ubicación y Geometría")
        pozo = st.text_input('Nombre / Número de Pozo', value='SCH - 213', key='input_pozo')
        prof = st.number_input(
            'Profundidad Total (ft)', min_value=1000.0, max_value=20000.0,
            value=10000.0, key='input_prof'
        )
        prof_punzados = st.number_input(
            "Profundidad de Punzados (ft)", value=9500.0, key="input_prof_punzados"
        )
        t = st.number_input('Temperatura (°F)', value=180.0, key='input_t')

    with col2:
        st.markdown("##### 🧪 Propiedades de Fluidos")

        tipo_entrada_api = st.radio(
            "¿Cómo desea ingresar el crudo?",
            ["Grado API", "SG del Crudo"],
            horizontal=True,
            key="tipo_entrada_api_radio"
        )

        if tipo_entrada_api == "Grado API":
            api = st.number_input(
                "Gravedad API",
                min_value=0.0,
                max_value=60.0,
                value=22.0,
                key="input_api_directo"
            )
        else:
            sg_oil = st.number_input(
                "Gravedad Específica (SGoil)",
                min_value=0.5,
                max_value=1.1,
                value=0.92,
                key="input_sg_oil"
            )
            api = (141.5 / sg_oil) - 131.5
            st.caption(f"📌 API calculado: **{api:.2f}°**")

        bsw = st.number_input("BSW (%)", min_value=0.0, max_value=100.0, value=40.0,
                              key="input_bsw")
        u = st.number_input("Viscosidad del Crudo (cp)", value=25.0,
                            key="input_viscosidad")
        gor = st.number_input("GOR (scf/bbl)", value=300.0, key="input_gor")

    with col3:
        st.markdown("##### ⚙️ Operación y Dinámica")
        qf = st.number_input('Caudal Total de Fluido (bbl/d)', value=250.0,
                             key='input_qf')
        qp = st.number_input('Caudal de Petróleo (bbl/d)', value=150.0, key='input_qp')
        incl = st.number_input("Inclinación (°)", value=15.0, key="input_incl")
        dls = st.number_input("Dog Leg Severity (°/100ft)", value=3.0, key="input_dls")

    # --- Bloque inferior para Nivel Dinámico ---
    st.markdown("---")
    st.markdown("##### 🌊 Evaluación del Nivel Dinámico")

    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        modo_nivel = st.radio(
            "¿Cómo desea definir el Nivel Dinámico?",
            ["Ingresar valor directo", "Calcular teóricamente (Analítico)"],
            horizontal=True,
            key="modo_nivel_radio"
        )

    with col_d2:
        if modo_nivel == "Ingresar valor directo":
            nvl_dina = st.number_input(
                "Nivel Dinámico (ft)",
                min_value=0.0,
                value=5000.0,
                key="input_nivel_directo"
            )
            sg_h2o = 1.0
            pr = 1500.0
        else:
            subc1, subc2 = st.columns(2)
            with subc1:
                pr = st.number_input(
                    "Presión de Yacimiento Pr (psi)",
                    min_value=0.0,
                    value=1500.0,
                    key="input_pr"
                )
            with subc2:
                sg_h2o = st.number_input(
                    "SG del Agua (SGh2o)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    key="input_sg_h2o"
                )

            nvl_dina = Nivel_Dinamico(
                API=api, BSW=bsw, SGh2o=sg_h2o, Pr=pr, Hperf=prof_punzados
            )
            st.success(
                f"💡 Nivel dinámico calculado teóricamente: **{nvl_dina:.2f} ft**")

    # Botón de ejecución general
    st.markdown("---")
    ejecutar_validacion = st.button("🔍 Ejecutar Validación Técnica", type="primary",
                                    use_container_width=True)

    if ejecutar_validacion:
        if modo_nivel != "Ingresar valor directo":
            nvl_dina = Nivel_Dinamico(
                API=api, BSW=bsw, SGh2o=sg_h2o, Pr=pr, Hperf=prof_punzados
            )

        # Llamada directa a tu función principal de validación petrolera
        df_resultados = funcion_validacion_petro(
            Pozo=pozo,
            Prof=prof,
            QF=qf,
            QP=qp,
            T=t,
            API=api,
            prof_punzados=prof_punzados,
            Nvl_Dina=nvl_dina,
            Incl=incl,
            DLS=dls,
            BSW=bsw,
            u=u,
            GOR=gor,
        )

        es_factible = df_resultados['Factible Bombeo Mecánico'].iloc[0] == 'si'

        st.markdown('---')
        st.subheader('📊 Resultados de la Evaluación')

        if es_factible:
            st.success(
                '✅ **Resultado:** El pozo cumple con los parámetros técnicos'
                ' establecidos. **Es FACTIBLE** implementar el sistema de bombeo'
                ' mecánico.'
            )
        else:
            st.error(
                '⚠️ **Resultado:** Uno o varios parámetros superan los límites'
                ' recomendados. Se requiere revisión de sensibilidad o ajustes en el'
                ' diseño.'
            )

        # Copia para formatear visualmente a 3 decimales con unidades sin alterar los datos puros
        df_visualizacion = df_resultados.copy()
        df_visualizacion['Valor Ingresado'] = df_visualizacion.apply(
            formatear_con_unidades, axis=1)

        st.dataframe(
            df_visualizacion.style.map(color_cumple, subset=['Cumple']),
            use_container_width=True,
            hide_index=True,  # Elimina la columna de índices de la izquierda
        )


# ==========================================
# 4. SECCIÓN DE DIMENSIONAMIENTO DE EQUIPOS
# ==========================================

if selected == "Dimensionamiento de equipos":
    st.markdown("## ⚙️ Módulo de Dimensionamiento - Bombeo Mecánico (SRP)")

    st.subheader("1. Parámetros Operativos y del Yacimiento")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pr = st.number_input("Presión de Reservorio (Pr) [psi]", value=446.0)
        Hperf = st.number_input(
            "Profundidad de Punzados (Hperf) [ft]", value=9443.0
        )
    with col2:
        API = st.number_input("Gravedad API", value=22.3)
        BSW = st.number_input("Corte de Agua (BSW) [%]", value=13.43)
    with col3:
        Gh2o = st.number_input("Gravedad Específica del Agua", value=1.1)
        D = st.number_input("Diámetro del Pistón (D) [in]", value=1.25)

    col4, col5, col6 = st.columns(3)
    with col4:
        N = st.number_input("Velocidad de Bombeo (N) [spm]", value=7.0)
    with col5:
        S = st.number_input("Carrera de Barra Pulida (S) [in]", value=144.0)
    with col6:
        Motor = st.selectbox("Tipo de Motor", ["NEMA D", "NEMA C"])

    # --- CÁLCULO / INGRESO DE NIVEL DINÁMICO ---
    st.subheader("2. Nivel Dinámico de Fluido")
    calc_dinamico = st.checkbox(
        "Calcular nivel dinámico automáticamente mediante correlación",
        value=True,
    )

    H_calc, G_calc = Nivel_Dinamico_SGfluido(API, BSW, Gh2o, Pr, Hperf)

    if calc_dinamico:
        H = H_calc
        st.info(
            f"💡 **Nivel dinámico calculado:** {np.round(H, 2)} ft (Gravedad"
            f" específica del fluido G: {np.round(G_calc, 2)})"
        )
    else:
        H = st.number_input(
            "Ingrese el Nivel Dinámico (H) [ft]",
            value=float(np.round(H_calc, 2)),
        )

    # --- OPCIONES DE PROFUNDIDAD DE BOMBA ---
    st.subheader("3. Profundidad de la Bomba (L)")
    opcion_l = st.radio(
        "Seleccione método de ubicación de bomba:",
        [
            "Opción 1: Recomendada (Por encima de punzados)",
            "Opción 2: Personalizada",
        ],
    )

    if "Recomendada" in opcion_l:
        L_default = Hperf - 500
        L = st.number_input(
            "Profundidad de bomba recomendada (Hperf - 500 ft)",
            value=float(L_default),
        )
    else:
        L = st.number_input(
            "Ingrese la profundidad personalizada de la bomba [ft]",
            value=float(Hperf - 500)
        )

    # --- VALIDACIONES DE PROFUNDIDAD DE BOMBA ---
    bomba_valida = True
    if L < H:
        st.error(
            "❌ **Error:** ¡Usted no puede poner la bomba a esa altura porque está"
            " encima del nivel dinámico!"
        )
        bomba_valida = False

    if L >= Hperf:
        st.warning(
            "⚠️ **Advertencia:** Usted puede poner la bomba al nivel o por debajo de"
            " los punzados, pero se arriesga a problemas de arenamiento."
        )

    # --- SELECCIÓN DE VARILLA (GRADO Y NÚMERO CONDICIONADO) ---
    st.subheader("4. Selección de Varilla de Succesión")
    grado_opciones = {
        "C": "Bajas cargas",
        "H": "Cargas medianas",
        "D": "Cargas pesadas",
        "K": "Resistente a la corrosión",
        "KD": "Cargas pesadas y resistente a la corrosión",
    }
    grado_sel = st.selectbox(
        "Grado de la Varilla",
        options=list(grado_opciones.keys()),
        format_func=lambda x: f"{x} - {grado_opciones[x]}",
    )

    # Condicionar el número de varilla (No) según la profundidad L (¡Ahora se actualiza al instante!)
    if L <= 4000:
        no_opciones = [44, 54, 55, 65, 66, 76, 87, 88, 98, 99, 109]
    elif 4000 < L <= 7000:
        no_opciones = [54, 64, 65, 75, 76, 86, 87, 97, 98, 108, 109]
    elif 7000 < L <= 12000:
        no_opciones = [64, 75, 85, 86, 96, 97, 107, 108]
    else:
        no_opciones = [85, 96, 107]

    no_sel = st.selectbox(
        "Número de Ensamblaje de Varilla (No)", options=no_opciones
    )
    Varilla_dict = {"No": no_sel, "Grado": grado_sel}

    # --- OPCIONES DE VISTA DE RESULTADOS ---
    st.subheader("5. Formato de Resultados")
    tipo_resultado = st.radio(
        "Seleccione cómo desea ver los resultados:",
        [
            "Opción 1: DataFrames de Varillas y Dimensiones",
            "Opción 2: Todos los parámetros calculados + DataFrames",
        ],
    )

    st.markdown("---")
    btn_calcular = st.button("🚀 Calcular Dimensionamiento", type="primary")

    if btn_calcular:
        if not bomba_valida:
            st.error(
                "⚠️ Por favor, corrija la profundidad de la bomba antes de calcular."
            )
        else:
            # Ejecución de cálculos
            G = G_calc
            Er, Fc, varilla_dim = elasticdad_frecuencia(Varilla_dict, D, L)
            Fo_Skr, N_No, C_N_No = adimensionales(G, D, H, Er, L, S, N, Fc)
            Wr, Wrf = Calculo_Wr(varilla_dim, G)
            resultados = simulador_completo_api_11l(
                N_No, C_N_No, Fo_Skr, N, Wrf, S, Er, L
            )

            Sp_S = resultados["Sp/S"]
            F1_Skr = resultados["F1/SKr (Carga Máx Adim)"]
            F2_Skr = resultados["F2/SKr (Carga Mín Adim)"]
            T2_S2Kr = resultados["2T/S^2Kr (Torque Adim)"]
            F3_SKr = resultados["F3/SKr (Potencia Adim)"]
            PPRL = resultados.get("Wmax (lb)", 0)
            MPRL = resultados.get("Wmin (lb)", 0)

            ta, x_val = calcular_ta_ajuste(C_N_No, Fo_Skr, Wrf, S, Er, L)
            PDp, PD = Calcular_PD(S, N, D, Sp_S)
            PT, PRHP, HPnp = Calculo_PT_PRHP_HPnp(
                T2_S2Kr, F3_SKr, S, Er, L, ta, N, Motor
            )

            # Dataframes requeridos
            df_varillas = pd.DataFrame({
                "No": [Varilla_dict["No"]] * len(varilla_dim["Size_in"]),
                "Grado": [Varilla_dict["Grado"]] * len(varilla_dim["Size_in"]),
                "Size (in)": varilla_dim["Size_in"],
                "Lon_ft": varilla_dim["long_ft"],
            })

            df_dimensiones = pd.DataFrame({
                "PD": [np.round(PD, 3)],
                "PPRL": [np.round(PPRL, 3)],
                "MPRL": [np.round(MPRL, 3)],
                "PT": [np.round(PT, 3)],
                "PRHP": [np.round(PRHP, 3)],
                "HPnp": [np.round(HPnp, 3)],
            })

            st.success("✅ ¡Cálculos realizados con éxito!")

            if "Opción 1" in tipo_resultado:
                st.markdown("### 📋 1. Configuración de Varillas")
                st.dataframe(df_varillas, use_container_width=True)
                st.markdown("### 📋 2. Resultados de Dimensionamiento (Dimensiones)")
                st.dataframe(df_dimensiones, use_container_width=True)

            else:
                st.markdown("### 📊 Listado Completo de Parámetros Calculados")
                parametros_completos = pd.DataFrame({
                    "Parámetro": [
                        "Fo/Skr",
                        "N/No",
                        "N/N'o (C_N_No)",
                        "PDpreliminar (PDp)",
                        "SP/S",
                        "PD",
                        "Wrf",
                        "F1/Skr",
                        "PPRL",
                        "F2/Skr",
                        "MPRL",
                        "Ta",
                        "2T/S2Kr",
                        "PT",
                        "PRHP",
                        "HPnp",
                    ],
                    "Valor": [
                        round(Fo_Skr, 4),
                        round(N_No, 4),
                        round(C_N_No, 4),
                        round(PDp, 3),
                        round(Sp_S, 4),
                        round(PD, 3),
                        round(Wrf, 2),
                        round(F1_Skr, 4),
                        round(PPRL, 2),
                        round(F2_Skr, 4),
                        round(MPRL, 2),
                        round(ta, 4),
                        round(T2_S2Kr, 4),
                        round(PT, 2),
                        round(PRHP, 3),
                        round(HPnp, 3),
                    ],
                })
                st.dataframe(parametros_completos, use_container_width=True)

                st.markdown("### 📋 DataFrames Detallados (Varillas y Dimensiones)")
                st.dataframe(df_varillas, use_container_width=True)
                st.dataframe(df_dimensiones, use_container_width=True)



# ==========================================
# 5. SECCIÓN DE VALIDACION ECONOMICA
# ==========================================
if selected == "Validación económica":
  st.title("💰 Evaluación Económica de Pozos Petroleros")
  st.markdown(
      "Módulo interactivo para el análisis de flujos de caja, indicadores"
      " financieros y evaluación comparativa."
  )
  st.divider()

  # --- DATOS GENERALES DEL POZO ---
  st.subheader("📌 Datos Generales del Pozo y Producción")
  col_p1, col_p2, col_p3 = st.columns(3)
  with col_p1:
    pozo = st.text_input("Nombre / Identificador del Pozo", value="SCH-132")
    price_bbl = st.number_input(
        "Precio de venta del barril (USD)", value=93.12, step=0.1
    )
  with col_p2:
    Q = st.number_input("Caudal / Producción inicial (BBPD)", value=32.0)
    cost_bbl = st.number_input(
        "Costo operativo por barril (USD)", value=7.00, step=0.1
    )
  with col_p3:
    Dec = st.number_input(
        "Declinación mensual (Fracción)", value=0.017, format="%.4f", step=0.001
    )
    hp_motor = st.number_input(
        "Potencia del motor del equipo (HP)", value=87.89, step=0.1
    )

  col_p4, col_p5 = st.columns(2)
  with col_p4:
    i = st.number_input(
        "Tasa de descuento mensual (i)", value=0.01, step=0.005, format="%.3f"
    )
  with col_p5:
    n = st.slider(
        "Tiempo de análisis (meses)", min_value=6, max_value=60, value=12
    )

  st.divider()

  # --- 1. ALCANCE DEL ANÁLISIS (AHORA ES EL PRIMERO) ---
  st.subheader("📈 1. Alcance del Análisis")
  tipo_analisis = st.selectbox(
      "Seleccione el tipo de evaluación:",
      ["Análisis Comparativo (Caso Propios vs Renta)", "Caso Único (Personalizado)"],
  )

  st.divider()

  # --- 2. CONFIGURACIÓN DE GASTOS MENSUALES (DINÁMICO SEGÚN EL ALCANCE) ---
  st.subheader("🛠️ 2. Configuración de Gastos Mensuales")

  costo_energia = gasto_energia(hp_motor)
  quimicos_mensuales = st.number_input(
      "Costo de Químicos Mensuales (USD)", value=4000.0
  )

  if tipo_analisis == "Análisis Comparativo (Caso Propios vs Renta)":
    st.info(
        "ℹ️ Para el análisis comparativo, ingrese ambos valores: la tarifa de"
        " renta y el costo de mantenimiento."
    )
    col_gc1, col_gc2 = st.columns(2)
    with col_gc1:
      renta_mensual = st.number_input(
          "Tarifa de Renta Mensual - Caso Rentado (USD)", value=18000.0
      )
    with col_gc2:
      mantenimiento_mensual = st.number_input(
          "Costo de Mantenimiento - Caso Propios (USD)", value=9000.0
      )
  else:
    modo_operativo = st.radio(
        "Seleccione el esquema operativo principal:",
        [
            "Equipos Propios (Con Mantenimiento)",
            "Equipos Rentados (Con Tarifa Renta)",
        ],
        horizontal=True,
    )
    if modo_operativo.startswith("Equipos Propios"):
      renta_mensual = 0.0
      mantenimiento_mensual = st.number_input(
          "Costo de Mantenimiento Mensual (USD)", value=9000.0
      )
    else:
      renta_mensual = st.number_input(
          "Tarifa de Renta Mensual (USD)", value=18000.0
      )
      mantenimiento_mensual = 0.0

  Gastos_mensuales = {
      "renta": renta_mensual,
      "energia": costo_energia,
      "mantenimiento": mantenimiento_mensual,
      "Quimicos": quimicos_mensuales,
  }

  st.divider()

  # --- 3. CONFIGURACIÓN DE INVERSIÓN INICIAL ---
  st.subheader("⚙️ 3. Configuración de Inversión Inicial")
  tipo_inversion = st.radio(
      "¿Cómo desea ingresar la Inversión Inicial?",
      ["Desglose Detallado de Componentes", "Valor Total Directo"],
      horizontal=True,
  )

  if tipo_inversion == "Valor Total Directo":
    inv_total_input = st.number_input(
        "Inversión Inicial Total (USD)", value=428078.63, step=1000.0
    )
    Inv_inicial = {"Inversion_Total": inv_total_input}
  else:
    st.markdown("Ingrese los valores por componente:")
    c1, c2 = st.columns(2)
    with c1:
      mov_T = st.number_input("Movilización de torre", value=10056.0)
      work_T = st.number_input("Trabajo en torre", value=202608.0)
      adecuacion = st.number_input("Adecuación de la zona", value=5000.0)
      eq_fon = st.number_input("Equipo de fondo", value=90000.0)
      control_P = st.number_input("Control de pozo", value=20172.93)
    with c2:
      S_T = st.number_input("Supervisión y transporte", value=2859.0)
      Quim = st.number_input("Químicos iniciales", value=385.5)
      wireline = st.number_input("Wireline", value=1150.0)
      r_elec = st.number_input("Registro electromagnético", value=13431.48)
      conting = st.number_input("Contingencias", value=80405.72)

    Inv_inicial = {
        "mov_T": mov_T,
        "work_T": work_T,
        "adecuacion": adecuacion,
        "S_T": S_T,
        "Quim": Quim,
        "Eq_sup": 0,
        "Eq_fon": eq_fon,
        "wireline": wireline,
        "r_elec": r_elec,
        "packer": 0,
        "Ev_pump": 2010,
        "Control_P": control_P,
        "conting": conting,
    }

  st.divider()

  # --- EJECUCIÓN DE LA SIMULACIÓN ---
  if st.button(
          "🚀 Ejecutar Evaluación Económica", type="primary", use_container_width=True
  ):
      if tipo_analisis == "Caso Único (Personalizado)":
          tabla_resultado = tabla_VAN(
              n, Q, price_bbl, Inv_inicial, cost_bbl, Gastos_mensuales, i, Dec
          )
          st.success(f"¡Evaluación completada con éxito para el pozo: {pozo}!")

          # --- AQUÍ CALCULAS Y MUESTRAS LOS INDICADORES DEL CASO ÚNICO ---
          df_resultado_unico = VAN_TIR_BC_PB_unico(
              tabla_resultado, i, n, pozo, Inv_inicial, tipo_equipo="Equipo Propio"
          )
          st.markdown("### 📊 Indicadores Económicos - Caso Único")
          st.dataframe(df_resultado_unico.style.format(precision=2),
                       use_container_width=True)
          # -------------------------------------------------------------

          st.markdown("### 📋 Tabla de Flujo de Caja")
          st.dataframe(tabla_resultado, use_container_width=True)

          # Gráfica Plotly para Caso Único
          st.markdown("### 📈 Gráfica de Comportamiento Financiero")
          fig_unico = generar_grafico_fnc(
              tabla_base=tabla_resultado, modo="unico", pozo=pozo
          )
          st.plotly_chart(fig_unico, use_container_width=True)

      else:
          # Para el caso base (equipos propios): renta = 0, se usa el mantenimiento ingresado
          gastos_base = Gastos_mensuales.copy()
          gastos_base["renta"] = 0

          # Para el caso renta: se usa la tarifa de renta ingresada, mantenimiento = 0
          gastos_renta = Gastos_mensuales.copy()
          gastos_renta["renta"] = renta_mensual
          gastos_renta["mantenimiento"] = 0

          tabla_base = tabla_VAN(
              n, Q, price_bbl, Inv_inicial, cost_bbl, gastos_base, i, Dec
          )
          tabla_renta = tabla_VAN(
              n, Q, price_bbl, Inv_inicial, cost_bbl, gastos_renta, i, Dec
          )

          df_comparativo = VAN_TIR_BC_PB_comparativa(
              tabla_base, tabla_renta, i, n, pozo, Inv_inicial
          )

          st.success(
              f"¡Análisis comparativo generado con éxito para el pozo {pozo}!"
          )
          st.markdown("### 📊 Tabla Comparativa de Indicadores")
          st.dataframe(df_comparativo.style.format(precision=2),
                       use_container_width=True)

          # Gráfica Plotly Comparativa
          st.markdown("### 📈 Gráfica Comparativa de FNC Acumulado")
          fig_comp = generar_grafico_fnc(
              tabla_base=tabla_base,
              tabla_renta=tabla_renta,
              modo="comparativo",
              pozo=pozo,
          )
          st.plotly_chart(fig_comp, use_container_width=True)

          t1, t2 = st.tabs(["Detalle Equipos Propios", "Detalle Equipos Rentados"])
          with t1:
              st.dataframe(tabla_base, use_container_width=True)
          with t2:
              st.dataframe(tabla_renta, use_container_width=True)

