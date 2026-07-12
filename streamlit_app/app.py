import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from streamlit_app.ui_utils import (API_URL, CLASS_NAMES, FIELD_LABELS, GROUPS, MALIGNANT,
                                    PATIENT_FICHA, VALUE_LABELS, api_online, call_predict,
                                    categorical_options, is_numeric, label_value, load_catalog,
                                    malignancy_score, risk_level)

st.set_page_config(page_title="Apoyo Diagnóstico Dermatológico · ORT", layout="wide")

st.markdown("""<style>
.stApp { background:#ffffff; color:#1a1a1a;
         font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
section[data-testid="stSidebar"] { background:#f5f7f9; border-right:1px solid #cfd6dd; }
* { border-radius:0 !important; }
.apptop{background:#1f3a5f;color:#ffffff;padding:12px 18px;margin-bottom:14px;}
.apptop .t{font-size:1.15rem;font-weight:700;letter-spacing:.2px;}
.apptop .s{font-size:0.78rem;opacity:.8;margin-top:2px;}
.ctxbar{display:flex;gap:0;border:1px solid #cfd6dd;border-left:3px solid #1f3a5f;
        background:#eef1f4;margin-bottom:12px;}
.ctxbar .cell{padding:6px 16px;border-right:1px solid #d7dde3;font-size:0.85rem;}
.ctxbar .cell .l{color:#5a6672;font-size:0.7rem;text-transform:uppercase;letter-spacing:.4px;}
.ctxbar .cell .v{font-weight:600;}
.sec{font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
     color:#1f3a5f;border-bottom:2px solid #1f3a5f;padding-bottom:3px;margin:6px 0 10px;}
table.grid{border-collapse:collapse;width:100%;font-size:0.85rem;}
table.grid td,table.grid th{border:1px solid #cfd6dd;padding:5px 10px;text-align:left;}
table.grid td.k{background:#f0f3f6;font-weight:600;width:48%;color:#33414f;}
table.grid th{background:#1f3a5f;color:#fff;font-weight:600;font-size:0.78rem;
              text-transform:uppercase;letter-spacing:.3px;}
table.grid td.n{text-align:right;font-variant-numeric:tabular-nums;}
.bar{background:#dfe4ea;height:12px;width:100%;}
.bar>span{display:block;height:12px;background:#34617f;}
.note{font-size:0.75rem;color:#5a6672;margin-top:16px;border-top:1px solid #cfd6dd;padding-top:8px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_catalog():
    return load_catalog()


try:
    catalog = get_catalog()
    options = categorical_options(catalog)
except Exception:
    catalog, options = None, {}


def notna_value(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def fmt(field, v):
    v = notna_value(v)
    if v is None:
        return "—"
    if is_numeric(field):
        try:
            fv = float(v)
            return str(int(fv)) if fv == int(fv) else f"{fv:g}"
        except (TypeError, ValueError):
            return str(v)
    return label_value(v)


def render_field(field, case, key, disabled):
    label = FIELD_LABELS.get(field, field)
    val = notna_value(case.get(field)) if case else None
    if is_numeric(field):
        default = float(val) if val is not None else 0.0
        return st.number_input(label, value=default, step=1.0, key=key, disabled=disabled)
    opts = options.get(field, [])
    if opts:
        choices = [None] + opts
        current = str(val) if val is not None else None
        idx = choices.index(current) if current in choices else 0
        return st.selectbox(label, choices, index=idx, key=key, disabled=disabled,
                            format_func=lambda v: "(sin dato)" if v is None else VALUE_LABELS.get(v, v))
    text = st.text_input(label, value="" if val is None else str(val), key=key, disabled=disabled)
    return text or None


def context_bar(case):
    if case is None:
        cells = [("Registro", "Nueva ficha"), ("Estado", "Sin datos cargados")]
    else:
        cells = [
            ("ID Paciente", str(case.get("patient_id"))),
            ("Edad", fmt("age", case.get("age"))),
            ("Sexo", fmt("gender", case.get("gender"))),
            ("Fototipo", fmt("fitspatrick", case.get("fitspatrick"))),
        ]
    inner = "".join(f'<div class="cell"><div class="l">{l}</div><div class="v">{v}</div></div>'
                    for l, v in cells)
    return f'<div class="ctxbar">{inner}</div>'


def render_ficha(case):
    rows = "".join(
        f"<tr><td class='k'>{FIELD_LABELS[f]}</td><td>{fmt(f, case.get(f))}</td></tr>"
        for f in PATIENT_FICHA)
    st.markdown(f"<table class='grid'>{rows}</table>", unsafe_allow_html=True)


def render_differential(probs):
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    rows = ""
    for n, (k, v) in enumerate(items):
        pct = v * 100
        nat = "Maligna" if k in MALIGNANT else "Benigna"
        natcol = "#9b2226" if k in MALIGNANT else "#2e5d34"
        weight = "700" if n == 0 else "400"
        rows += (f"<tr style='font-weight:{weight};'>"
                 f"<td>{CLASS_NAMES.get(k, k)}</td><td>{k}</td>"
                 f"<td style='color:{natcol};'>{nat}</td>"
                 f"<td class='n'>{pct:.1f}%</td>"
                 f"<td style='width:200px;'><div class='bar'><span style='width:{pct:.1f}%;'></span></div></td></tr>")
    st.markdown(
        "<table class='grid'><tr><th>Diagnóstico</th><th>Código</th><th>Naturaleza</th>"
        f"<th>Probabilidad</th><th>Distribución</th></tr>{rows}</table>",
        unsafe_allow_html=True)


def render_results(res, case):
    probs, pred = res["probabilidades"], res["prediccion"]
    mal = malignancy_score(probs)
    level, color = risk_level(mal)
    natcol = "#9b2226" if pred in MALIGNANT else "#2e5d34"

    st.markdown('<div class="sec">Resultado del análisis</div>', unsafe_allow_html=True)
    summary = [
        ("Diagnóstico más probable", f"{CLASS_NAMES.get(pred, pred)} ({pred})"),
        ("Confianza", f"{probs[pred]*100:.1f}%"),
        ("Naturaleza", f"<span style='color:{natcol};font-weight:600;'>"
                       f"{'Maligno' if pred in MALIGNANT else 'Benigno'}</span>"),
        ("Probabilidad de malignidad (BCC+MEL+SCC)",
         f"{mal*100:.0f}% · <span style='color:{color};font-weight:600;'>Nivel {level}</span>"),
    ]
    if case is not None and notna_value(case.get("diagnostic")):
        truth = case["diagnostic"]
        ok = truth == pred
        tone = "#2e5d34" if ok else "#9b2226"
        summary.append(("Diagnóstico confirmado (biopsia)",
                        f"{CLASS_NAMES.get(truth, truth)} ({truth}) — "
                        f"<span style='color:{tone};font-weight:600;'>"
                        f"{'coincide con el modelo' if ok else 'no coincide con el modelo'}</span>"))
    st.markdown(
        "<table class='grid'>" +
        "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in summary) +
        "</table>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Diagnóstico diferencial</div>', unsafe_allow_html=True)
    render_differential(probs)


# ---- Barra lateral: búsqueda de paciente ----
with st.sidebar:
    st.markdown('<div class="sec">Directorio de pacientes</div>', unsafe_allow_html=True)
    origen = st.radio("Origen de la ficha", ["Paciente registrado", "Cargar nueva ficha"],
                      label_visibility="collapsed")
    patient_rows = None
    if origen == "Paciente registrado":
        if catalog is not None:
            pats = catalog.drop_duplicates("patient_id")
            sel_pid = st.selectbox("Paciente (conjunto de test)", pats["patient_id"].tolist())
            patient_rows = catalog[catalog["patient_id"] == sel_pid].reset_index(drop=True)
            st.caption("Identificadores anonimizados del dataset PAD-UFES-20.")
        else:
            st.info("Catálogo sin datos. Utilice 'Cargar nueva ficha'.")
    st.divider()
    dot = "#2e5d34" if api_online() else "#9b2226"
    estado = "En línea" if api_online() else "Sin conexión"
    st.markdown(f"<div style='font-size:0.8rem;color:#5a6672;'>Servicio de inferencia<br>"
                f"<span style='color:{dot};'>●</span> {estado} · {API_URL}</div>",
                unsafe_allow_html=True)

# ---- Encabezado ----
st.markdown(
    '<div class="apptop"><div class="t">Apoyo Diagnóstico de Lesiones Cutáneas</div>'
    '<div class="s">Simulación académica — Obligatorio Machine Learning en Producción · '
    'Universidad ORT · Modelo multimodal sobre dataset PAD-UFES-20</div></div>',
    unsafe_allow_html=True)

case = None
if patient_rows is not None:
    st.markdown(context_bar(patient_rows.iloc[0].to_dict()), unsafe_allow_html=True)
    with st.expander("Antecedentes del paciente", expanded=False):
        render_ficha(patient_rows.iloc[0].to_dict())
    n = len(patient_rows)
    idx = 0
    if n > 1:
        idx = st.selectbox(
            f"Lesión registrada ({n} en total)", range(n),
            format_func=lambda i: f"{patient_rows.iloc[i]['region']} — {patient_rows.iloc[i]['img_id']}")
    case = patient_rows.iloc[idx].to_dict()
else:
    st.markdown(context_bar(None), unsafe_allow_html=True)

case_id = case["img_id"] if case else "manual"

if st.session_state.get("current_case") != case_id:
    st.session_state.current_case = case_id
    st.session_state.edit = case is None
edit_mode = st.session_state.edit

col_img, col_form = st.columns([1, 1.3])

with col_img:
    st.markdown('<div class="sec">Imagen clínica</div>', unsafe_allow_html=True)
    up_label = "Cargar nueva imagen" if case is not None else "Cargar imagen de la lesión"
    uploaded = st.file_uploader(up_label, type=["png", "jpg", "jpeg"], key=f"up_{case_id}")
    image_bytes, filename = None, None
    if uploaded is not None:
        image_bytes, filename = uploaded.getvalue(), uploaded.name
    elif case is not None:
        image_bytes = Path(case["img_path"]).read_bytes()
        filename = case["img_id"]
    if image_bytes:
        st.image(image_bytes, use_container_width=True)
    else:
        st.info("Cargue una imagen para realizar el análisis.")

with col_form:
    head, btn = st.columns([3, 1])
    head.markdown('<div class="sec">Ficha clínica</div>', unsafe_allow_html=True)
    if case is not None and not edit_mode:
        if btn.button("Editar", use_container_width=True):
            st.session_state.edit = True
            edit_mode = True
    elif case is not None and edit_mode:
        btn.caption("Edición habilitada")

    clinical = {}
    for group, fields in GROUPS.items():
        with st.expander(group, expanded=(group == "Lesión")):
            cols = st.columns(2)
            for i, field in enumerate(fields):
                with cols[i % 2]:
                    clinical[field] = render_field(field, case, f"{field}_{case_id}", not edit_mode)
    analizar = st.button("Analizar lesión", type="primary", use_container_width=True)

if analizar:
    if not image_bytes:
        st.error("Se requiere una imagen para realizar el análisis.")
    else:
        try:
            with st.spinner("Procesando análisis…"):
                res = call_predict(image_bytes, filename or "lesion.png", clinical)
            render_results(res, case)
        except Exception as e:
            st.error(f"No se pudo contactar el servicio de inferencia ({API_URL}). Detalle: {e}")

st.markdown(
    '<div class="note">Herramienta de carácter académico. No constituye un dispositivo médico '
    'ni reemplaza el juicio clínico profesional.</div>', unsafe_allow_html=True)
