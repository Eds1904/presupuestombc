import os
import io
import sys
import json
import logging
from typing import Optional, Tuple, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime

# ======== LOGGING / CHECKPOINTS ========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stderr,
)

def log(msg: str):
    """Manda log a stderr y lo guarda para mostrar en la UI."""
    try:
        logging.info(msg)
        st.session_state.setdefault("_logs", []).append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")
    except Exception:
        pass

# Mostrar siempre algo rápido para evitar “pantalla en blanco”
st.set_page_config(page_title="📊 Gestión de Flujos", layout="wide")
st.title("📊 Gestión de Flujos Financieros")
st.caption("✅ checkpoint: front levantó")
st.set_option("client.showErrorDetails", True)
log("UI inicial pintada.")

# ============ CONFIG ============
FILE_ESTIMADOS = "estimados.csv"
FILE_CERTIFICACIONES = "certificaciones.csv"
FILE_PROVEEDORES = "proveedores.csv"
FILE_VALOR = "valor.csv"
PREFS_FILE = "ui_prefs.json"

CSV_DATE_FMT = "%m-%Y"   # guardamos/mostramos MM-YYYY
TIPOS = ["CAPEX", "OPEX"]
CATEGORIAS = ["Licencia", "Infraestructura", "Despliegue", "Otros"]
MESES_ES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
MES_ABR_ES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]  # abreviado para ejes

# ============ HELPERS: IO / PREFS ============
def load_prefs(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            prefs = json.load(f)
        log(f"Preferencias cargadas desde {path}.")
        return prefs
    except Exception as e:
        log(f"Sin preferencias previas ({e}).")
        return {}

def save_prefs(path: str, prefs: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
        log(f"Preferencias guardadas en {path}.")
    except Exception as e:
        log(f"ERROR guardando preferencias: {e}")

# ====== PERSISTENCIA EN GOOGLE DRIVE (CSV) con fallback local ======
# Detectar si hay secrets (evita crash local)
try:
    HAS_SECRETS = len(getattr(st, "secrets", {})) > 0
except Exception:
    HAS_SECRETS = False

USE_DRIVE = False
_DRIVE_IMPORT_ERROR = None

if HAS_SECRETS:
    try:
        USE_DRIVE = ("gcp" in st.secrets) and ("drive" in st.secrets) and bool(st.secrets["drive"].get("folder_id"))
    except Exception:
        USE_DRIVE = False

if USE_DRIVE:
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
        from google.oauth2.service_account import Credentials
        log("Dependencias de Google API importadas correctamente.")
    except Exception as e:
        _DRIVE_IMPORT_ERROR = str(e)
        USE_DRIVE = False
        log(f"ERROR importando Google API libs, deshabilito Drive: {e}")

@st.cache_resource(show_spinner=False)
def _drive_service():
    """Crea el servicio de Drive (cacheado)."""
    creds_info = dict(st.secrets["gcp"])
    scopes = ["https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("drive", "v3", credentials=credentials)

def _find_file(service, name, parent_id):
    """Busca archivo por nombre exacto en carpeta."""
    q_name = name.replace("'", "")
    q = f"name = '{q_name}' and '{parent_id}' in parents and trashed = false"
    r = service.files().list(q=q, fields="files(id, name)", pageSize=1).execute()
    files = r.get("files", [])
    return files[0]["id"] if files else None

def load_drive_csv(name: str, cols):
    service = _drive_service()
    folder_id = st.secrets["drive"]["folder_id"]
    file_id = _find_file(service, name, folder_id)
    if not file_id:
        log(f"[Drive] No existe {name}, devuelvo DF vacío con columnas esperadas.")
        return pd.DataFrame(columns=cols)

    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)

    # Evitar loops infinitos (máx 50 chunks con reintentos)
    done = False
    for _ in range(50):
        status, done = downloader.next_chunk(num_retries=2)
        if done:
            break
    if not done:
        raise RuntimeError(f"Descarga de Drive no concluyó para {name} (timeout de chunks).")

    buf.seek(0)
    df = pd.read_csv(buf)
    df = normalize_columns(df)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0 if c in ("Monto","Valor") else ""
    if "Fecha" in df.columns:
        df["Fecha"] = df["Fecha"].apply(try_parse_month)
    log(f"[Drive] {name} cargado. Filas: {len(df)}")
    return df[cols]

def save_drive_csv(df: pd.DataFrame, name: str):
    service = _drive_service()
    folder_id = st.secrets["drive"]["folder_id"]
    file_id = _find_file(service, name, folder_id)

    out = df.copy()
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.strftime(CSV_DATE_FMT)
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    buf.seek(0)

    media = MediaIoBaseUpload(buf, mimetype="text/csv", resumable=True)
    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        meta = {"name": name, "parents": [folder_id], "mimeType": "text/csv"}
        service.files().create(body=meta, media_body=media, fields="id").execute()
    log(f"[Drive] {name} guardado. Filas: {len(df)}")

def upload_to_drive_from_filelike(file, target_name: str):
    """Sube un archivo CSV (file-like) a la carpeta, reemplazando si existe."""
    service = _drive_service()
    folder_id = st.secrets["drive"]["folder_id"]
    file_id = _find_file(service, target_name, folder_id)
    data = file.read()
    buf = io.BytesIO(data)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="text/csv", resumable=True)
    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        meta = {"name": target_name, "parents": [folder_id], "mimeType": "text/csv"}
        service.files().create(body=meta, media_body=media, fields="id").execute()
    log(f"[Drive] Subida/replace de {target_name} OK.")

# ============ HELPERS: FECHAS / CSV ============
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("descripción","descripcion"): mapping[c] = "Descripcion"
        elif low == "comentario": mapping[c] = "Comentario"
        elif low == "proveedor": mapping[c] = "Proveedor"
        elif low == "monto": mapping[c] = "Monto"
        elif low == "valor": mapping[c] = "Valor"
        elif low == "fecha": mapping[c] = "Fecha"
        elif low == "cuit": mapping[c] = "CUIT"
        elif low == "categoria": mapping[c] = "Categoria"
        elif low == "tipo": mapping[c] = "Tipo"
        else: mapping[c] = c.strip()
    return df.rename(columns=mapping)

def try_parse_month(x):
    """Devuelve primer día del mes como Timestamp."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime, date)):
        return pd.to_datetime(f"{x.year}-{x.month:02d}-01")
    s = str(x).strip()
    for fmt in [CSV_DATE_FMT, "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"]:
        try:
            dt = datetime.strptime(s, fmt)
            return pd.to_datetime(f"{dt.year}-{dt.month:02d}-01")
        except Exception:
            pass
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(dt): return pd.NaT
    return pd.to_datetime(f"{dt.year}-{dt.month:02d}-01")

def load_data(file, cols):
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            df = normalize_columns(df)
            for c in cols:
                if c not in df.columns:
                    df[c] = 0.0 if c in ("Monto","Valor") else ""
            if "Fecha" in df.columns:
                df["Fecha"] = df["Fecha"].apply(try_parse_month)
            log(f"[Local] {file} cargado. Filas: {len(df)}")
            return df[cols]
        except Exception as e:
            log(f"ERROR leyendo {file}: {e}. Devuelvo DF vacío.")
            return pd.DataFrame(columns=cols)
    else:
        log(f"[Local] {file} no existe, DF vacío.")
        return pd.DataFrame(columns=cols)

def save_data(df, file):
    try:
        out = df.copy()
        if "Fecha" in out.columns:
            out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.strftime(CSV_DATE_FMT)
        out.to_csv(file, index=False)
        log(f"[Local] {file} guardado. Filas: {len(df)}")
    except Exception as e:
        log(f"ERROR guardando {file}: {e}")
        raise

# Funciones unificadas para usar SIEMPRE:
def LOAD(file, cols):
    if USE_DRIVE:
        return load_drive_csv(file, cols)
    else:
        return load_data(file, cols)

def SAVE(df, file):
    if USE_DRIVE:
        return save_drive_csv(df, file)
    else:
        return save_data(df, file)

def SAVE_SAFE(df, file, label: str = ""):
    try:
        SAVE(df, file)
        st.success("✅ Cambios guardados.")
        log(f"Guardado OK: {file}")
    except Exception as e:
        st.error(f"❌ Error guardando {label or file}: {e}")
        log(f"ERROR guardando {file}: {e}")

# ============ VALIDACIONES ============
def validar_cuit(cuit: str) -> bool:
    """CUIT 11 dígitos con checksum. Vacío = válido."""
    if not cuit: return True
    num = "".join([c for c in str(cuit) if c.isdigit()])
    if len(num) != 11: return False
    mult = [5,4,3,2,7,6,5,4,3,2]
    s = sum(int(num[i])*mult[i] for i in range(10))
    dv = 11 - (s % 11)
    if dv == 11: dv = 0
    if dv == 10: dv = 9
    return dv == int(num[-1])

def validar_obligatorios(campos: dict) -> bool:
    ok = True
    for nombre, valor in campos.items():
        low = nombre.lower()
        if low in ("monto","valor"):
            try:
                if valor is None or float(valor) <= 0:
                    st.error(f"⚠️ '{nombre}' es obligatorio y > 0.")
                    ok = False
            except Exception:
                st.error(f"⚠️ '{nombre}' debe ser numérico.")
                ok = False
        elif low == "fecha":
            if pd.isna(pd.to_datetime(valor, errors="coerce")):
                st.error("⚠️ Debés seleccionar una fecha (Mes/Año).")
                ok = False
        else:
            if not str(valor).strip():
                st.error(f"⚠️ '{nombre}' es obligatorio.")
                ok = False
    return ok

# ============ PICKER MES/AÑO ============
def month_year_picker(label, key_prefix, default: date = None):
    if default is None: default = date.today()
    c1, c2 = st.columns(2)
    mes_nombre = c1.selectbox(f"{label} - Mes", MESES_ES, index=default.month-1, key=f"{key_prefix}_mes")
    anio = int(c2.number_input(f"{label} - Año", min_value=1990, max_value=2100, value=default.year, step=1, key=f"{key_prefix}_anio"))
    mes = MESES_ES.index(mes_nombre) + 1
    return pd.to_datetime(f"{anio}-{mes:02d}-01")

# ============ TABLA EDITABLE ============
def show_table_editable(df, file, cols, select_options: Dict[str, list] = None):
    if df.empty:
        st.info("No hay registros cargados todavía.")
        return df

    select_options = select_options or {}
    view = df.copy()
    if "Fecha" in view.columns:
        view["Fecha"] = pd.to_datetime(view["Fecha"], errors="coerce").dt.strftime(CSV_DATE_FMT)
    view["Eliminar"] = False

    # Column config con SelectboxColumn cuando corresponda
    col_config = {}
    if "Proveedor" in view.columns and "Proveedor" in select_options:
        col_config["Proveedor"] = st.column_config.SelectboxColumn(
            "Proveedor", options=select_options["Proveedor"], required=True
        )
    if "Tipo" in view.columns and "Tipo" in select_options:
        col_config["Tipo"] = st.column_config.SelectboxColumn(
            "Tipo", options=select_options["Tipo"], required=True
        )
    if "Categoria" in view.columns and "Categoria" in select_options:
        col_config["Categoria"] = st.column_config.SelectboxColumn(
            "Categoria", options=select_options["Categoria"], required=True
        )

    st.write("✏️ Editá valores; tildá **Eliminar** para borrar filas. Luego guardá cambios.")
    edited = st.data_editor(
        view,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor_{file}",
        column_config=col_config
    )

    keep = edited[edited["Eliminar"] == False].drop(columns=["Eliminar"])
    eliminadas = len(keep) < len(df)

    c1, c2, c3 = st.columns([1,1,1])
    confirm = True
    if eliminadas:
        c2.warning("⚠️ Detectamos filas marcadas para eliminar.")
        confirm = c2.checkbox("Confirmo eliminaciones", value=False, key=f"confirm_{file}")

    if c1.button("💾 Guardar cambios", key=f"save_{file}"):
        if eliminadas and not confirm:
            st.warning("Marcá la casilla de confirmación para guardar con eliminaciones.")
            return df
        if "Fecha" in keep.columns:
            keep["Fecha"] = keep["Fecha"].apply(try_parse_month)
        for col in cols:
            if col not in keep.columns:
                keep[col] = 0.0 if col in ("Monto","Valor") else ""

        # Seguridad: Proveedor/Tipo/Categoria pertenecen a listas válidas si existen
        if "Proveedor" in keep.columns and "Proveedor" in select_options:
            keep["Proveedor"] = keep["Proveedor"].where(keep["Proveedor"].isin(select_options["Proveedor"]), "")
        if "Tipo" in keep.columns and "Tipo" in select_options:
            keep["Tipo"] = keep["Tipo"].where(keep["Tipo"].isin(select_options["Tipo"]), "")
        if "Categoria" in keep.columns and "Categoria" in select_options:
            keep["Categoria"] = keep["Categoria"].where(keep["Categoria"].isin(select_options["Categoria"]), "")

        keep = keep[cols]
        SAVE_SAFE(keep, file, label=os.path.basename(file))
        return keep

    c3.download_button(
        "⬇️ Descargar CSV",
        data=df_to_bytes(df),
        file_name=os.path.basename(file),
        mime="text/csv",
        use_container_width=True
    )
    return df

def df_to_bytes(df):
    tmp = df.copy()
    if "Fecha" in tmp.columns:
        tmp["Fecha"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.strftime(CSV_DATE_FMT)
    return tmp.to_csv(index=False).encode("utf-8")

# ============ AGREGACIÓN / ORDEN / EJE COMPLETO ============
def periodo_label(fecha: pd.Series, freq: str) -> pd.Series:
    f = pd.to_datetime(fecha, errors="coerce")
    if freq == "M":
        return f.apply(lambda x: f"{MES_ABR_ES[x.month-1]} {x.year}" if pd.notna(x) else None)
    elif freq == "Q":
        return f.apply(lambda x: f"T{x.quarter} {x.year}" if pd.notna(x) else None)
    else:
        return f.dt.year.astype("Int64").astype(str)

def build_period_skeleton(freq: str, fmin: pd.Timestamp, fmax: pd.Timestamp) -> pd.DataFrame:
    if pd.isna(fmin) or pd.isna(fmax):
        return pd.DataFrame(columns=["__key__","Periodo"])
    if freq == "M":
        pr = pd.period_range(fmin.to_period("M").start_time, fmax.to_period("M").start_time, freq="M")
        keys = [p.year*100 + p.month for p in pr]
        labels = [f"{MES_ABR_ES[p.month-1]} {p.year}" for p in pr]
    elif freq == "Q":
        pr = pd.period_range(fmin.to_period("Q").start_time, fmax.to_period("Q").start_time, freq="Q")
        keys = [p.year*10 + p.quarter for p in pr]
        labels = [f"T{p.quarter} {p.year}" for p in pr]
    else:  # "Y"
        pr = pd.period_range(fmin.to_period("Y").start_time, fmax.to_period("Y").start_time, freq="Y")
        keys = [p.year for p in pr]
        labels = [f"{p.year}" for p in pr]
    return pd.DataFrame({"__key__": keys, "Periodo": labels})

def agregar_por_periodo(df: pd.DataFrame, value_col: str, freq: str, fmin: pd.Timestamp, fmax: pd.Timestamp, acumulado=False):
    sk = build_period_skeleton(freq, fmin, fmax)
    if sk.empty:
        return pd.DataFrame(columns=["Periodo", value_col]), []

    if df.empty:
        out = sk.copy()
        out[value_col] = 0.0
    else:
        d = df.copy()
        d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
        d = d.dropna(subset=["Fecha"])
        if freq == "M":
            d["__key__"] = d["Fecha"].dt.year*100 + d["Fecha"].dt.month
        elif freq == "Q":
            d["__key__"] = d["Fecha"].dt.year*10 + d["Fecha"].dt.quarter
        else:
            d["__key__"] = d["Fecha"].dt.year
        agg = d.groupby("__key__", as_index=False)[value_col].sum()
        out = sk.merge(agg, on="__key__", how="left").fillna({value_col: 0.0})

    if acumulado:
        out[value_col] = out[value_col].cumsum()

    out = out.sort_values("__key__")
    ordered = out["Periodo"].tolist()
    out["Periodo"] = pd.Categorical(out["Periodo"], categories=ordered, ordered=True)
    return out[["Periodo", value_col]], ordered

def comparar_plan_real(est_df, cer_df, freq):
    fmins, fmaxs = [], []
    for d in (est_df, cer_df):
        if not d.empty:
            f = pd.to_datetime(d["Fecha"], errors="coerce")
            if not f.dropna().empty:
                fmins.append(f.min())
                fmaxs.append(f.max())
    if not fmins:
        return pd.DataFrame(columns=["Periodo","Estimado","Certificado","Variacion","Cum_Estimado","Cum_Certificado"]), []

    fmin = min(fmins)
    fmax = max(fmaxs)

    est, _ = agregar_por_periodo(est_df, "Monto", freq, fmin, fmax, False)
    cer, ordered = agregar_por_periodo(cer_df, "Monto", freq, fmin, fmax, False)
    est = est.rename(columns={"Monto":"Estimado"})
    cer = cer.rename(columns={"Monto":"Certificado"})

    comp = pd.merge(est, cer, on="Periodo", how="outer")
    for col in ["Estimado", "Certificado"]:
        if col in comp.columns:
            comp[col] = comp[col].fillna(0.0)

    comp["Periodo"] = pd.Categorical(comp["Periodo"], categories=ordered, ordered=True)
    comp = comp.sort_values("Periodo")
    comp["Variacion"] = comp["Certificado"] - comp["Estimado"]
    comp["Cum_Estimado"] = comp["Estimado"].cumsum()
    comp["Cum_Certificado"] = comp["Certificado"].cumsum()
    return comp, ordered

# ============ PROYECCIÓN REAL ACUMULADO ============
def agregar_proyeccion(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega:
      - Cum_Proyectado: hasta último período con Certificado>0 sigue Cum_Certificado;
        luego suma Estimado futuros (proyección).
      - Cum_Certificado_Masked: igual a Cum_Certificado hasta último real; NaN a futuro.
      - Estimado_Proy: misma serie Estimado.
      - Real_Proy_Periodo: contribución por período (real hasta último real; luego estimado).
    """
    if comp_df.empty:
        return comp_df.copy()

    df = comp_df.copy()
    cer = df["Certificado"].fillna(0.0).values
    est = df["Estimado"].fillna(0.0).values

    real_idx = max([i for i, v in enumerate(cer) if v > 0.0], default=-1)

    cum_proj = []
    cum_real_mask = []
    real_proy_periodo = []

    for i in range(len(df)):
        # máscara del real acumulado
        if i <= real_idx:
            cum_real_mask.append(float(df["Cum_Certificado"].iloc[i]))
        else:
            cum_real_mask.append(float("nan"))

        # proyección acumulada
        if i <= real_idx:
            cum_proj.append(float(df["Cum_Certificado"].iloc[i]))
            real_proy_periodo.append(float(cer[i]))
        else:
            prev = cum_proj[i-1] if i > 0 else 0.0
            cum_proj.append(prev + float(est[i]))
            real_proy_periodo.append(float(est[i]))

    df["Cum_Proyectado"] = cum_proj
    df["Cum_Certificado_Masked"] = cum_real_mask
    df["Estimado_Proy"] = df["Estimado"]
    df["Real_Proy_Periodo"] = real_proy_periodo

    # Variaciones
    df["Variacion_Proy_Periodo"] = df["Real_Proy_Periodo"] - df["Estimado_Proy"]
    df["Variacion_Proy_Acum"] = df["Cum_Proyectado"] - df["Cum_Estimado"]
    return df

# ============ FIG HELPERS ============
def apply_colors_and_axes(
    fig: go.Figure,
    color_map: Dict[str, str],
    opacity_map: Dict[str, float],
    yscale: str = "linear",
    yrange: Optional[Tuple[float, float]] = None,
    barmode: Optional[str] = None,
):
    for tr in fig.data:
        name = getattr(tr, "name", "")
        if name in color_map and color_map[name]:
            if hasattr(tr, "marker"):
                tr.marker.color = color_map[name]
            if hasattr(tr, "line"):
                tr.line.color = color_map[name]
        if name in opacity_map and opacity_map[name] is not None:
            tr.opacity = opacity_map[name]

    fig.update_yaxes(type=yscale, tickformat=",.0f" if yscale == "linear" else None)
    if yrange is not None:
        fig.update_yaxes(range=list(yrange))
    if barmode:
        fig.update_layout(barmode=barmode)
    return fig

def fig_bar_categ(df, x, y, title, ordered):
    fig = px.bar(df, x=x, y=y, title=title, text_auto=True)
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ordered, tickangle=-15, title=None)
    fig.update_yaxes(title=None, tickformat=",.0f")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig

def fig_line_categ(df, x, y_list, title, ordered):
    fig = px.line(df, x=x, y=y_list, title=title)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ordered, tickangle=-15, title=None)
    fig.update_yaxes(title=None, tickformat=",.0f")
    fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ============ KPI 'AL DÍA DE HOY' ============
def label_hoy(freq: str) -> str:
    hoy = pd.Timestamp.today()
    s = pd.Series([hoy])
    return periodo_label(s, freq).iloc[0]

def diferencia_hoy(comp_df: pd.DataFrame, ord_list: list, freq: str):
    if comp_df.empty:
        return 0.0, 0.0, 0.0
    hoy_lbl = label_hoy(freq)
    if hoy_lbl in ord_list:
        idx_hoy = ord_list.index(hoy_lbl)
    else:
        idx_hoy = len(ord_list) - 1
        if idx_hoy < 0:
            return 0.0, 0.0, 0.0
    hasta_hoy = set(ord_list[:idx_hoy + 1])
    comp_hoy = comp_df[comp_df["Periodo"].astype(str).isin(hasta_hoy)]
    if comp_hoy.empty:
        return 0.0, 0.0, 0.0
    cum_est_hoy = float(comp_hoy["Cum_Estimado"].iloc[-1])
    cum_real_hoy = float(comp_hoy["Cum_Certificado"].iloc[-1])
    delta_hoy = cum_real_hoy - cum_est_hoy
    return cum_est_hoy, cum_real_hoy, delta_hoy

# ============ MATRIZ TIPO PLANILLA ============
def matriz_planilla(est_df, cer_df, freq, ordered_periods):
    """
    Devuelve (wide_df, tooltips_df):
     - wide_df: filas MultiIndex (Proveedor, Serie ['Estimado','Real']), columnas Periodos ordenados
     - tooltips_df: mismo shape con strings (comentarios/descripciones concatenadas)
    """
    def _prep(df_in, value_col, serie_name):
        if df_in.empty:
            return pd.DataFrame(columns=["Proveedor","Periodo",value_col,"Comentarios","Serie"])
        d = df_in.copy()
        d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
        d = d.dropna(subset=["Fecha"])
        d["Periodo"] = periodo_label(d["Fecha"], freq)
        # Comentarios: combinamos Comentario + Descripcion si existen
        com = []
        for _, r in d.iterrows():
            bits = []
            if "Comentario" in d.columns and isinstance(r.get("Comentario",""), str) and str(r["Comentario"]).strip():
                bits.append(str(r["Comentario"]).strip())
            if "Descripcion" in d.columns and isinstance(r.get("Descripcion",""), str) and str(r["Descripcion"]).strip():
                bits.append(str(r["Descripcion"]).strip())
            com.append(" | ".join(bits))
        d["Comentarios"] = com
        grp_val = d.groupby(["Proveedor","Periodo"], as_index=False)[value_col].sum()
        grp_com = d.groupby(["Proveedor","Periodo"])["Comentarios"].apply(
            lambda s: " || ".join([x for x in s if x])
        ).reset_index()
        g = pd.merge(grp_val, grp_com, on=["Proveedor","Periodo"], how="left")
        g = g.rename(columns={value_col: "Monto"})
        g["Comentarios"] = g["Comentarios"].fillna("")
        g["Serie"] = serie_name
        return g

    e = _prep(est_df, "Monto", "Estimado")
    r = _prep(cer_df, "Monto", "Real")
    base = pd.concat([e, r], ignore_index=True) if not e.empty or not r.empty else pd.DataFrame(columns=["Proveedor","Periodo","Monto","Comentarios","Serie"])

    if base.empty:
        idx = pd.MultiIndex.from_tuples([], names=["Proveedor","Serie"])
        wide = pd.DataFrame(index=idx, columns=ordered_periods).astype(float)
        tips = wide.copy()
        return wide, tips

    base["Periodo"] = pd.Categorical(base["Periodo"], categories=ordered_periods, ordered=True)

    wide = base.pivot_table(index=["Proveedor","Serie"], columns="Periodo", values="Monto", aggfunc="sum", fill_value=0.0)
    tips = base.pivot_table(index=["Proveedor","Serie"], columns="Periodo", values="Comentarios", aggfunc=lambda s: " || ".join([x for x in s if x]), fill_value="")

    # ordenar filas por total Real desc (solo donde exista "Real")
    ordered_rows = []
    try:
        if "Real" in wide.index.get_level_values("Serie"):
            tot_real = wide.xs("Real", level="Serie", drop_level=False).sum(axis=1)
            order_idx = tot_real.sort_values(ascending=False).index.get_level_values("Proveedor").unique().tolist()
            for prov in order_idx:
                for serie in ["Estimado","Real"]:
                    tup = (prov, serie)
                    if tup in wide.index:
                        ordered_rows.append(tup)
    except Exception:
        pass

    # agregar cualquier fila que no haya quedado incluida
    for idxr in wide.index:
        if idxr not in ordered_rows:
            ordered_rows.append(idxr)

    # Reindex seguro
    ordered_rows_mi = pd.MultiIndex.from_tuples(ordered_rows, names=wide.index.names)
    wide = wide.reindex(ordered_rows_mi)
    tips = tips.reindex(ordered_rows_mi).fillna("")

    # Alinear columnas exactamente a ordered_periods
    wide = wide.reindex(columns=ordered_periods)
    tips = tips.reindex(columns=ordered_periods).fillna("")

    return wide, tips

# ====== SIDEBAR: ESTADO / DIAGNÓSTICO ======
with st.sidebar.expander("🔐 Estado de almacenamiento", expanded=False):
    if USE_DRIVE:
        st.success("Persistencia: Google Drive (API).")
        st.caption("Los CSV viven en la carpeta de Drive configurada en secrets.")
    else:
        if HAS_SECRETS and _DRIVE_IMPORT_ERROR:
            st.warning(f"Secrets presentes, pero falló importar Google API: {_DRIVE_IMPORT_ERROR}. Usando CSV locales.")
        elif HAS_SECRETS:
            st.warning("Secrets presentes pero Drive deshabilitado (revisá [drive.folder_id]). Usando CSV locales.")
        else:
            st.info("Sin secrets: modo local (CSV en disco).")

with st.sidebar.expander("🧪 Diagnóstico (logs)", expanded=False):
    logs_text = "\n".join(st.session_state.get("_logs", [])) or "Sin logs aún."
    st.code(logs_text, language="text")
    st.caption("Los logs también se envían a los Logs del servicio (stderr).")

# ====== SUBIDA CSV ======
def handle_csv_uploads():
    st.sidebar.markdown("### 📤 Cargar / Reemplazar CSV")
    st.sidebar.caption("Acepta: estimados.csv, certificaciones.csv, proveedores.csv, valor.csv")
    uploads = st.sidebar.file_uploader(
        "Arrastrá o seleccioná los CSV",
        type=["csv"], accept_multiple_files=True, key="uploader_csv"
    )
    if not uploads:
        return
    name_map = {
        "estimados.csv": FILE_ESTIMADOS,
        "certificaciones.csv": FILE_CERTIFICACIONES,
        "proveedores.csv": FILE_PROVEEDORES,
        "valor.csv": FILE_VALOR,
    }
    touched = False
    for up in uploads:
        base = up.name.strip().lower()
        if base in name_map:
            target = name_map[base]
            try:
                if USE_DRIVE:
                    upload_to_drive_from_filelike(up, target)
                else:
                    with open(target, "wb") as f:
                        f.write(up.getbuffer())
                st.sidebar.success(f"Reemplazado: {base}")
                log(f"Upload OK: {base} -> {target}")
                touched = True
            except Exception as e:
                st.sidebar.error(f"❌ Error subiendo {base}: {e}")
                log(f"ERROR upload {base}: {e}")
        else:
            st.sidebar.warning(f"Ignorado {up.name} (nombre no reconocido).")
            log(f"Ignorado upload {up.name}")
    if touched:
        st.sidebar.info("Datos actualizados desde los CSV subidos.")
        st.rerun()

handle_csv_uploads()

# ============ CARGA INICIAL (robusta) ============
def SAFE_LOAD(file, cols, label):
    try:
        log(f"Cargando {label} desde {'Drive' if USE_DRIVE else 'local'}")
        df = LOAD(file, cols)
        return df
    except Exception as e:
        st.warning(f"No se pudo cargar {label}: {e}. Se usa tabla vacía.")
        log(f"ERROR al cargar {label}: {e}")
        return pd.DataFrame(columns=cols)

estimados = SAFE_LOAD(FILE_ESTIMADOS, ["Fecha","Monto","Proveedor","Descripcion","Comentario","Tipo","Categoria"], "Estimados")
certificaciones = SAFE_LOAD(FILE_CERTIFICACIONES, ["Fecha","Monto","Proveedor","Comentario","Tipo","Categoria"], "Certificaciones")
proveedores = SAFE_LOAD(FILE_PROVEEDORES, ["Proveedor","CUIT","Descripcion"], "Proveedores")
valor = SAFE_LOAD(FILE_VALOR, ["Fecha","Valor","Comentario"], "Valor")

# ============ SIDEBAR ============
st.sidebar.title("📌 Menú")
menu = st.sidebar.radio("Ir a", ["Carga Estimado","Carga Certificación","Carga Proveedor","Captura de Valor","Reportes"], index=0)

# ============ FORM: ESTIMADO ============
if menu == "Carga Estimado":
    st.subheader("➕ Cargar Estimado")
    with st.form("form_estimado", clear_on_submit=False):
        fecha = month_year_picker("Fecha", "est", date.today())
        c1,c2,c3 = st.columns([1,1,2])
        with c1:
            monto = st.number_input("Monto [$]", min_value=0.0, step=100.0, format="%.2f", key="est_monto")
        with c2:
            tipo = st.radio("Tipo", options=TIPOS, horizontal=True, key="est_tipo")
        with c3:
            proveedor = st.selectbox("Proveedor", [""] + (proveedores["Proveedor"].dropna().unique().tolist()
                                   if not proveedores.empty else []), key="est_prov")
        c4,c5 = st.columns([1,1])
        with c4:
            categoria = st.selectbox("Categoria", CATEGORIAS, key="est_cat")
        with c5:
            descripcion = st.text_input("Descripcion (opcional)", key="est_desc")
        comentario = st.text_input("Comentario (opcional)", key="est_com")
        submitted = st.form_submit_button("Guardar Estimado")
    if submitted:
        if validar_obligatorios({"Fecha":fecha,"Monto":monto,"Proveedor":proveedor,"Tipo":tipo,"Categoria":categoria}):
            nuevo = pd.DataFrame([{
                "Fecha": fecha, "Monto": float(monto), "Proveedor": proveedor,
                "Descripcion": (st.session_state.get("est_desc") or "").strip(),
                "Comentario": (st.session_state.get("est_com") or "").strip(),
                "Tipo": tipo, "Categoria": categoria
            }])
            estimados = pd.concat([estimados, nuevo], ignore_index=True)
            SAVE_SAFE(estimados, FILE_ESTIMADOS, label="Estimados")
            st.rerun()
    st.divider(); st.subheader("📋 Estimados cargados")
    proveedor_opts = proveedores["Proveedor"].dropna().unique().tolist() if not proveedores.empty else []
    estimados = show_table_editable(
        estimados,
        FILE_ESTIMADOS,
        ["Fecha","Monto","Proveedor","Descripcion","Comentario","Tipo","Categoria"],
        select_options={
            "Proveedor": proveedor_opts,
            "Tipo": TIPOS,
            "Categoria": CATEGORIAS
        }
    )

# ============ FORM: CERTIFICACIÓN ============
elif menu == "Carga Certificación":
    st.subheader("📝 Cargar Certificación")
    with st.form("form_cert", clear_on_submit=False):
        fecha = month_year_picker("Fecha", "cer", date.today())
        c1,c2,c3 = st.columns([1,1,2])
        with c1:
            monto = st.number_input("Monto certificado [$]", min_value=0.0, step=100.0, format="%.2f", key="cer_monto")
        with c2:
            tipo = st.radio("Tipo", options=TIPOS, horizontal=True, key="cer_tipo")
        with c3:
            proveedor = st.selectbox("Proveedor", [""] + (proveedores["Proveedor"].dropna().unique().tolist()
                                   if not proveedores.empty else []), key="cer_prov")
        c4,c5 = st.columns([1,1])
        with c4:
            categoria = st.selectbox("Categoria", CATEGORIAS, key="cer_cat")
        with c5:
            comentario = st.text_area("Comentario (opcional)", key="cer_com")
        submitted = st.form_submit_button("Guardar Certificación")
    if submitted:
        if validar_obligatorios({"Fecha":fecha,"Monto":monto,"Proveedor":proveedor,"Tipo":tipo,"Categoria":categoria}):
            nuevo = pd.DataFrame([{
                "Fecha": fecha, "Monto": float(monto), "Proveedor": proveedor,
                "Comentario": (st.session_state.get("cer_com") or "").strip(),
                "Tipo": tipo, "Categoria": categoria
            }])
            certificaciones = pd.concat([certificaciones, nuevo], ignore_index=True)
            SAVE_SAFE(certificaciones, FILE_CERTIFICACIONES, label="Certificaciones")
            st.rerun()
    st.divider(); st.subheader("📋 Certificaciones cargadas")
    proveedor_opts = proveedores["Proveedor"].dropna().unique().tolist() if not proveedores.empty else []
    certificaciones = show_table_editable(
        certificaciones,
        FILE_CERTIFICACIONES,
        ["Fecha","Monto","Proveedor","Comentario","Tipo","Categoria"],
        select_options={
            "Proveedor": proveedor_opts,
            "Tipo": TIPOS,
            "Categoria": CATEGORIAS
        }
    )

# ============ FORM: PROVEEDOR ============
elif menu == "Carga Proveedor":
    st.subheader("🏢 Alta de Proveedores")
    with st.form("form_prov", clear_on_submit=False):
        c1,c2 = st.columns([2,1])
        prov = c1.text_input("Proveedor", key="prov_nom")
        cuit = c2.text_input("CUIT (opcional)", key="prov_cuit")
        desc = st.text_input("Descripcion (opcional)", key="prov_desc")
        submitted = st.form_submit_button("Guardar Proveedor")
    if submitted:
        if validar_obligatorios({"Proveedor":prov}):
            if cuit and not validar_cuit(cuit):
                st.warning("⚠️ El CUIT no parece válido (checksum). Se guarda igual por ser opcional.")
            nuevo = pd.DataFrame([{"Proveedor":prov.strip(),"CUIT":(cuit or "").strip(),"Descripcion":(desc or "").strip()}])
            proveedores = pd.concat([proveedores, nuevo], ignore_index=True)
            SAVE_SAFE(proveedores, FILE_PROVEEDORES, label="Proveedores")
            st.rerun()
    st.divider(); st.subheader("📋 Proveedores cargados")
    proveedores = show_table_editable(proveedores, FILE_PROVEEDORES, ["Proveedor","CUIT","Descripcion"])

# ============ FORM: CAPTURA DE VALOR ============
elif menu == "Captura de Valor":
    st.subheader("💡 Captura de Valor")
    with st.form("form_val", clear_on_submit=False):
        fecha = month_year_picker("Fecha", "val", date.today())
        c1,c2 = st.columns([1,1])
        val_monto = c1.number_input("Valor [$]", min_value=0.0, step=100.0, format="%.2f", key="val_monto")
        comentario = c2.text_area("Comentario (opcional)", key="val_com")
        submitted = st.form_submit_button("Guardar Captura de Valor")
    if submitted:
        if validar_obligatorios({"Fecha":fecha,"Valor":val_monto}):
            nuevo = pd.DataFrame([{"Fecha":fecha,"Valor":float(val_monto),"Comentario":(st.session_state.get("val_com") or "").strip()}])
            valor = pd.concat([valor, nuevo], ignore_index=True)
            SAVE_SAFE(valor, FILE_VALOR, label="Valor")
            st.rerun()
    st.divider(); st.subheader("📋 Valores cargados")
    valor = show_table_editable(valor, FILE_VALOR, ["Fecha","Valor","Comentario"])

# ============ REPORTES ============
elif menu == "Reportes":
    st.subheader("📊 Reportes Ejecutivos")

    # Filtros superiores
    freq_map = {"Mensual":"M","Trimestral":"Q","Anual":"Y"}
    colA, colB, colC = st.columns([1,2,2])
    agg_label = colA.selectbox("Agregación", ["Mensual","Trimestral","Anual"], index=0)
    freq = freq_map[agg_label]
    provs = sorted(list(set(
        (estimados["Proveedor"].dropna().unique().tolist() if not estimados.empty else []) +
        (certificaciones["Proveedor"].dropna().unique().tolist() if not certificaciones.empty else [])
    )))
    sel_provs = colB.multiselect("Proveedor (opcional)", provs, default=[])
    colD, colE = st.columns([1,1])
    sel_tipos = colD.multiselect("Tipo", TIPOS, default=TIPOS)
    sel_cats  = colE.multiselect("Categoria", CATEGORIAS, default=CATEGORIAS)

    def aplicar_filtros(d, usar_tipo_cat=False):
        if d.empty: return d
        x = d.copy()
        if sel_provs and "Proveedor" in x: x = x[x["Proveedor"].isin(sel_provs)]
        if usar_tipo_cat:
            if sel_tipos: x = x[x["Tipo"].isin(sel_tipos)]
            if sel_cats:  x = x[x["Categoria"].isin(sel_cats)]
        return x

    est_f = aplicar_filtros(estimados, True)
    cer_f = aplicar_filtros(certificaciones, True)
    val_f = aplicar_filtros(valor, False)

    # Filtro de fecha (Desde/Hasta)
    def _minmax_fecha(*dfs):
        fechas = []
        for d in dfs:
            if not d.empty and "Fecha" in d.columns:
                f = pd.to_datetime(d["Fecha"], errors="coerce")
                f = f.dropna()
                if not f.empty:
                    fechas.append((f.min(), f.max()))
        if not fechas:
            hoy = pd.Timestamp.today().normalize().replace(day=1)
            return hoy, hoy
        mins = [a for a,_ in fechas]
        maxs = [b for _,b in fechas]
        return min(mins), max(maxs)

    fmin_all, fmax_all = _minmax_fecha(est_f, cer_f, val_f)
    colF, colG = st.columns(2)
    desde = month_year_picker("Desde", "rep_desde", fmin_all.date())
    hasta = month_year_picker("Hasta", "rep_hasta", fmax_all.date())
    desde = pd.to_datetime(desde).to_period("M").start_time
    hasta = pd.to_datetime(hasta).to_period("M").start_time
    if desde > hasta:
        st.warning("⚠️ 'Desde' es posterior a 'Hasta'. Invertimos el rango para continuar.")
        desde, hasta = hasta, desde

    def _filtrar_rango(d):
        if d.empty or "Fecha" not in d.columns: return d
        f = pd.to_datetime(d["Fecha"], errors="coerce").dt.to_period("M").dt.start_time
        m = (f >= desde) & (f <= hasta)
        return d[m]

    est_f = _filtrar_rango(est_f)
    cer_f = _filtrar_rango(cer_f)
    val_f = _filtrar_rango(val_f)

    # ===== 🎨 Apariencia y escalas (EXPANDER, con guardar/aplicar) =====
    default_prefs = {
        "col_est": "#1f77b4", "col_cer": "#ff7f0e", "col_var": "#2ca02c",
        "col_cum_est": "#9467bd", "col_cum_cer": "#d62728",
        "col_cum_proj": "#17becf",
        "bar_mode": "group", "yscale": "linear",
        "set_range": False, "y_min": 0.0, "y_max": 0.0,
        "op_est": 0.9, "op_cer": 0.9, "op_var": 0.9,
    }
    if "ui_prefs" not in st.session_state:
        prefs_disk = load_prefs(PREFS_FILE)
        st.session_state.ui_prefs = {**default_prefs, **prefs_disk}
    ui = st.session_state.ui_prefs

    with st.expander("🎨 Apariencia y escalas", expanded=False):
        c1, c2, c3 = st.columns(3)
        col_est = c1.color_picker("Color Estimado", ui["col_est"])
        col_cer = c2.color_picker("Color Certificado (Real)", ui["col_cer"])
        col_var = c3.color_picker("Color Variación (Real/Proy - Estimado)", ui["col_var"])

        c4, c5, c6 = st.columns(3)
        col_cum_est = c4.color_picker("Color Curva Real - Plan (Est.)", ui["col_cum_est"])
        col_cum_cer = c5.color_picker("Color Curva Real - Plan (Real)", ui["col_cum_cer"])
        bar_mode = c6.selectbox("Modo de barras", ["group","stack"], index=0 if ui["bar_mode"]=="group" else 1)

        c7, c8, c9 = st.columns(3)  # <- FIX: c7 (no 'googlc7')
        yscale = c7.selectbox("Escala Y", ["linear","log"], index=0 if ui["yscale"]=="linear" else 1)
        y_min = float(c8.number_input("Y mínimo", value=float(ui["y_min"]), step=1000.0))
        y_max = float(c9.number_input("Y máximo", value=float(ui["y_max"]), step=1000.0))

        set_range = st.checkbox("Fijar rango Y manual", value=ui["set_range"])
        yrange = (y_min, y_max) if set_range and y_max > 0 else None

        d1, d2, d3 = st.columns(3)
        op_est = d1.slider("Opacidad Estimado", 0.1, 1.0, float(ui["op_est"]), 0.05)
        op_cer = d2.slider("Opacidad Certificado", 0.1, 1.0, float(ui["op_cer"]), 0.05)
        op_var = d3.slider("Opacidad Variación", 0.1, 1.0, float(ui["op_var"]), 0.05)

        b1, b2, _ = st.columns([1,1,2])
        if b1.button("💾 Guardar"):
            st.session_state.ui_prefs = {
                "col_est": col_est, "col_cer": col_cer, "col_var": col_var,
                "col_cum_est": col_cum_est, "col_cum_cer": col_cum_cer,
                "bar_mode": bar_mode, "yscale": yscale,
                "set_range": set_range, "y_min": y_min, "y_max": y_max,
                "op_est": op_est, "op_cer": op_cer, "op_var": op_var,
            }
            save_prefs(PREFS_FILE, st.session_state.ui_prefs)
            st.success("Preferencias guardadas.")
        if b2.button("🔄 Aplicar"):
            st.session_state.ui_prefs = {
                "col_est": col_est, "col_cer": col_cer, "col_var": col_var,
                "col_cum_est": col_cum_est, "col_cum_cer": col_cum_cer,
                "bar_mode": bar_mode, "yscale": yscale,
                "set_range": set_range, "y_min": y_min, "y_max": y_max,
                "op_est": op_est, "op_cer": op_cer, "op_var": op_var,
            }
            st.rerun()

    # mapas para apply_colors_and_axes
    color_map_common = {
        "Estimado": st.session_state.ui_prefs["col_est"],
        "Certificado": st.session_state.ui_prefs["col_cer"],
        "Variacion": st.session_state.ui_prefs["col_var"],
        "Cum_Estimado": st.session_state.ui_prefs["col_cum_est"],
        "Cum_Certificado": st.session_state.ui_prefs["col_cum_cer"],
        "Estimado acumulado": st.session_state.ui_prefs["col_cum_est"],
        "Real acumulado": st.session_state.ui_prefs["col_cum_cer"],
        "Proyectado": st.session_state.ui_prefs["col_cum_cer"],
    }
    opacity_map_common = {
        "Estimado": st.session_state.ui_prefs["op_est"],
        "Certificado": st.session_state.ui_prefs["op_cer"],
        "Variacion": st.session_state.ui_prefs["op_var"],
        "Cum_Estimado": 0.95,
        "Cum_Certificado": 0.95,
        "Estimado acumulado": 0.95,
        "Real acumulado": 0.95,
        "Proyectado": 0.95,
    }
    bar_mode = st.session_state.ui_prefs["bar_mode"]
    yscale   = st.session_state.ui_prefs["yscale"]
    yrange = (st.session_state.ui_prefs["y_min"], st.session_state.ui_prefs["y_max"]) if st.session_state.ui_prefs["set_range"] and st.session_state.ui_prefs["y_max"] > 0 else None

    # ===== KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    total_est = est_f["Monto"].sum() if not est_f.empty else 0
    total_cer = cer_f["Monto"].sum() if not cer_f.empty else 0
    capex_real = cer_f[cer_f["Tipo"]=="CAPEX"]["Monto"].sum() if not cer_f.empty else 0
    opex_real  = cer_f[cer_f["Tipo"]=="OPEX"]["Monto"].sum() if not cer_f.empty else 0
    capex_plan = est_f[est_f["Tipo"]=="CAPEX"]["Monto"].sum() if not est_f.empty else 0
    captura    = val_f["Valor"].sum() if not val_f.empty else 0
    delta = total_cer - total_est

    k1.metric("Estimado total", f"${total_est:,.0f}")
    k2.metric("Certificado total", f"${total_cer:,.0f}", delta=f"{delta:,.0f}")
    k3.metric("CAPEX (Real)", f"${capex_real:,.0f}")
    k4.metric("OPEX (Real)", f"${opex_real:,.0f}")
    k5.metric("CAPEX (Plan)", f"${capex_plan:,.0f}")
    k6.metric("Captura de valor", f"${captura:,.0f}")

    st.divider()

    # ===== Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Resumen", "⏱️ Flujo temporal", "🏢 Por proveedor", "📋 Detalle", "🧮 Planilla estilo Excel"])

    # ---------- TAB 1: Resumen ----------
    with tab1:
        comp, ord_comp = comparar_plan_real(est_f, cer_f, freq)
        if comp.empty:
            st.info("No hay datos con los filtros aplicados.")
        else:
            # Proyección de real acumulado hacia futuro con Estimados
            comp_proj = agregar_proyeccion(comp)

            modo = st.radio(
                "Visualización",
                ["Barras (Plan vs Real)", "Líneas acumuladas (Real vs Proyectado)"],
                horizontal=True,
                index=0,
                help="En 'Líneas' se muestra el Real hasta su última fecha y el Proyectado (línea punteada) usando los Estimados futuros."
            )

            if modo.startswith("Barras"):
                long = comp.melt(id_vars="Periodo", value_vars=["Estimado","Certificado"],
                                 var_name="Serie", value_name="Monto")
                fig = px.bar(long, x="Periodo", y="Monto", color="Serie",
                             barmode=bar_mode, title=f"Plan vs Real por {agg_label}")
                fig.update_xaxes(type="category", categoryorder="array",
                                 categoryarray=ord_comp, tickangle=-15)
                fig.update_yaxes(tickformat=",.0f", title=None)
                fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20))
                fig = apply_colors_and_axes(fig, color_map_common, opacity_map_common,
                                            yscale=yscale, yrange=yrange, barmode=bar_mode)
                st.plotly_chart(fig, use_container_width=True)

                # Variación: hasta hoy vs estimado
                cum_est_hoy, cum_real_hoy, delta_hoy = diferencia_hoy(comp, ord_comp, freq)
                cH1, cH2, cH3 = st.columns(3)
                cH1.metric("Estimado acumulado (hoy)", f"${cum_est_hoy:,.0f}")
                cH2.metric("Real acumulado (hoy)", f"${cum_real_hoy:,.0f}", delta=f"{(cum_real_hoy - cum_est_hoy):,.0f}")
                cH3.metric("Diferencia al día de hoy (Real - Plan)", f"{delta_hoy:,.0f}")

            else:
                # Curva acumulada: Real (solo hasta última fecha real), Proyectado (real+estimados futuros), y Estimado
                dfp = comp_proj.copy()
                figS = fig_line_categ(
                    dfp, "Periodo", ["Cum_Estimado","Cum_Certificado_Masked","Cum_Proyectado"],
                    f"Curva Real - Plan (acumulada) — {agg_label}", ord_comp
                )
                # Renombrar para leyenda y hover
                rename_map = {
                    "Cum_Estimado": "Estimado acumulado",
                    "Cum_Certificado_Masked": "Real acumulado",
                    "Cum_Proyectado": "Proyectado"
                }
                figS.for_each_trace(
                    lambda t: t.update(
                        name=rename_map.get(t.name, t.name),
                        hovertemplate="%{x}<br>%{y:,.0f}<extra>%{fullData.name}</extra>"
                    )
                )
                # Líneas: Proyectado punteado
                for tr in figS.data:
                    if getattr(tr, "name", "") == "Proyectado":
                        if hasattr(tr, "line"):
                            tr.line.dash = "dot"

                figS = apply_colors_and_axes(figS, color_map_common, opacity_map_common,
                                             yscale=yscale, yrange=yrange)
                st.plotly_chart(figS, use_container_width=True)

                # Indicadores
                cum_est_hoy, cum_real_hoy, delta_hoy = diferencia_hoy(comp, ord_comp, freq)
                delta_final_proj = float(dfp["Cum_Proyectado"].iloc[-1] - dfp["Cum_Estimado"].iloc[-1])
                cH1, cH2, cH3 = st.columns(3)
                cH1.metric("Estimado acumulado (hoy)", f"${cum_est_hoy:,.0f}")
                cH2.metric("Real acumulado (hoy)", f"${cum_real_hoy:,.0f}", delta=f"{(cum_real_hoy - cum_est_hoy):,.0f}")
                cH3.metric("Δ a fin de rango (Proyectado - Plan)", f"{delta_final_proj:,.0f}")

    # ---------- TAB 2: Flujo temporal ----------
    with tab2:
        if not est_f.empty:
            fmin_e = pd.to_datetime(est_f["Fecha"], errors="coerce").min()
            fmax_e = pd.to_datetime(est_f["Fecha"], errors="coerce").max()
            est_agg, ord_e = agregar_por_periodo(est_f, "Monto", freq, fmin_e, fmax_e, False)
            st.subheader(f"Estimados ({agg_label})")
            fig_e = fig_bar_categ(est_agg, "Periodo", "Monto", f"Evolución de Estimados ({agg_label})", ord_e)
            if fig_e.data: fig_e.data[0].name = "Estimado"
            fig_e = apply_colors_and_axes(fig_e, color_map_common, opacity_map_common,
                                          yscale=yscale, yrange=yrange, barmode=bar_mode)
            st.plotly_chart(fig_e, use_container_width=True)
        else:
            st.info("Sin datos de Estimados.")

        if not cer_f.empty:
            fmin_c = pd.to_datetime(cer_f["Fecha"], errors="coerce").min()
            fmax_c = pd.to_datetime(cer_f["Fecha"], errors="coerce").max()
            cer_agg, ord_c = agregar_por_periodo(cer_f, "Monto", freq, fmin_c, fmax_c, False)
            st.subheader(f"Certificaciones ({agg_label})")
            fig_c = fig_bar_categ(cer_agg, "Periodo", "Monto", f"Evolución de Certificaciones ({agg_label})", ord_c)
            if fig_c.data: fig_c.data[0].name = "Certificado"
            fig_c = apply_colors_and_axes(fig_c, color_map_common, opacity_map_common,
                                          yscale=yscale, yrange=yrange, barmode=bar_mode)
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Sin datos de Certificaciones.")

        comp2, ord_comp2 = comparar_plan_real(est_f, cer_f, freq)
        if not comp2.empty:
            st.subheader("Curva Real - Plan (acumulada) con proyección")
            dfp2 = agregar_proyeccion(comp2)
            fig_curva = fig_line_categ(dfp2, "Periodo",
                                       ["Cum_Estimado","Cum_Certificado_Masked","Cum_Proyectado"],
                                       f"Curva Real - Plan — {agg_label}", ord_comp2)
            rename_map = {
                "Cum_Estimado": "Estimado acumulado",
                "Cum_Certificado_Masked": "Real acumulado",
                "Cum_Proyectado": "Proyectado"
            }
            fig_curva.for_each_trace(
                lambda t: t.update(
                    name=rename_map.get(t.name, t.name),
                    hovertemplate="%{x}<br>%{y:,.0f}<extra>%{fullData.name}</extra>"
                )
            )
            for tr in fig_curva.data:
                if getattr(tr, "name", "") == "Proyectado":
                    if hasattr(tr, "line"):
                        tr.line.dash = "dot"
            fig_curva = apply_colors_and_axes(fig_curva, color_map_common, opacity_map_common,
                                              yscale=yscale, yrange=yrange)
            st.plotly_chart(fig_curva, use_container_width=True)

    # ---------- TAB 3: Por proveedor ----------
    with tab3:
        if est_f.empty and cer_f.empty:
            st.info("No hay datos para mostrar por proveedor.")
        else:
            est_p = est_f.groupby(["Proveedor","Tipo","Categoria"], as_index=False)["Monto"].sum().rename(columns={"Monto":"Estimado"})
            cer_p = cer_f.groupby(["Proveedor","Tipo","Categoria"], as_index=False)["Monto"].sum().rename(columns={"Monto":"Certificado"})
            prov = pd.merge(est_p, cer_p, on=["Proveedor","Tipo","Categoria"], how="outer").fillna(0.0)
            prov["Variacion"] = prov["Certificado"] - prov["Estimado"]
            prov = prov.sort_values("Certificado", ascending=False)

            top_n = st.slider("Top proveedores por certificado", 3, 20, 10)
            fig_top = fig_bar_categ(prov.head(top_n), "Proveedor", "Certificado",
                                    f"Top {top_n} Proveedores (Real)", prov["Proveedor"].tolist())
            if fig_top.data: fig_top.data[0].name = "Certificado"
            fig_top = apply_colors_and_axes(fig_top, color_map_common, opacity_map_common,
                                            yscale=yscale, yrange=yrange, barmode=bar_mode)
            st.plotly_chart(fig_top, use_container_width=True)

            with st.expander("Desglose por Tipo y Categoria"):
                st.write("**Por Tipo**")
                st.dataframe(prov.groupby("Tipo", as_index=False)[["Estimado","Certificado"]].sum(), use_container_width=True)
                st.write("**Por Categoria**")
                st.dataframe(prov.groupby("Categoria", as_index=False)[["Estimado","Certificado"]].sum(), use_container_width=True)

            st.markdown("#### Tabla completa por proveedor")
            st.dataframe(prov, use_container_width=True)
            st.download_button("⬇️ Descargar tabla proveedores", data=prov.to_csv(index=False).encode("utf-8"),
                               file_name="reporte_proveedores.csv", mime="text/csv")

    # ---------- TAB 4: Detalle ----------
    with tab4:
        comp, ord_comp = comparar_plan_real(est_f, cer_f, freq)
        if comp.empty:
            st.info("No hay datos para el detalle.")
        else:
            if not val_f.empty:
                fmin_v = pd.to_datetime(val_f["Fecha"], errors="coerce").min()
                fmax_v = pd.to_datetime(val_f["Fecha"], errors="coerce").max()
                v_agg, _ = agregar_por_periodo(val_f, "Valor", freq, fmin_v, fmax_v, False)
                comp = pd.merge(comp, v_agg, on="Periodo", how="left")
                if "Valor" in comp.columns:
                    comp["Valor"] = comp["Valor"].fillna(0.0)
            st.dataframe(comp, use_container_width=True)
            st.download_button("⬇️ Descargar detalle", data=comp.to_csv(index=False).encode("utf-8"),
                               file_name="detalle_periodos.csv", mime="text/csv")

    # ---------- TAB 5: Planilla estilo Excel ----------
    with tab5:
        comp2, ord_comp2 = comparar_plan_real(est_f, cer_f, freq)
        if not comp2.empty:
            wide, tips = matriz_planilla(est_f, cer_f, freq, ord_comp2)
            st.caption("Filas: Proveedor + Serie (Estimado/Real). Columnas: períodos. Pasá el mouse para ver comentarios.")
            # Formato de columnas numéricas
            fmt = {col: "{:,.0f}" for col in wide.columns} if not wide.empty else {}
            # Tooltips con pandas Styler (requiere jinja2); fallback si no está
            try:
                tips_clean = tips.replace("", pd.NA) if isinstance(tips, pd.DataFrame) else tips
                styled = wide.style.format(fmt).set_tooltips(tips_clean)
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.info("Mostrando planilla sin tooltips (agregá 'jinja2' en requirements.txt para ver comentarios al pasar el mouse).")
                st.dataframe(wide, use_container_width=True)

            st.download_button(
                "⬇️ Descargar planilla (CSV)",
                data=wide.to_csv().encode("utf-8"),
                file_name="planilla_proveedor_periodos.csv",
                mime="text/csv"
            )
        else:
            st.info("No hay datos para la planilla con los filtros aplicados.")


                                      


   


   

