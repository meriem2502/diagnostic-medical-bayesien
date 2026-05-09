import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

# =====================================
# CONFIGURATION DE LA PAGE
# =====================================

st.set_page_config(
    page_title="Diagnostic Médical Bayésien",
    page_icon="🩺",
    layout="centered"
)

# =====================================
# TITRE
# =====================================

st.title("🩺 Diagnostic Médical Bayésien")
st.markdown("### Système intelligent basé sur un réseau bayésien")

st.write(
    "Cette application estime la probabilité qu'un patient ait la grippe "
    "à partir des symptômes sélectionnés."
)

# =====================================
# CRÉATION DU RÉSEAU BAYÉSIEN
# =====================================

model = DiscreteBayesianNetwork([
    ('Grippe', 'Fièvre'),
    ('Grippe', 'Toux'),
    ('Grippe', 'Fatigue')
])

# =====================================
# TABLES DE PROBABILITÉS (CPT)
# =====================================

# Probabilité de grippe
cpd_grippe = TabularCPD(
    variable='Grippe',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={
        'Grippe': ['Non', 'Oui']
    }
)

# Fièvre selon grippe
cpd_fievre = TabularCPD(
    variable='Fièvre',
    variable_card=2,
    values=[
        [0.9, 0.2],
        [0.1, 0.8]
    ],
    evidence=['Grippe'],
    evidence_card=[2],
    state_names={
        'Fièvre': ['Non', 'Oui'],
        'Grippe': ['Non', 'Oui']
    }
)

# Toux selon grippe
cpd_toux = TabularCPD(
    variable='Toux',
    variable_card=2,
    values=[
        [0.8, 0.3],
        [0.2, 0.7]
    ],
    evidence=['Grippe'],
    evidence_card=[2],
    state_names={
        'Toux': ['Non', 'Oui'],
        'Grippe': ['Non', 'Oui']
    }
)

# Fatigue selon grippe
cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.7, 0.2],
        [0.3, 0.8]
    ],
    evidence=['Grippe'],
    evidence_card=[2],
    state_names={
        'Fatigue': ['Non', 'Oui'],
        'Grippe': ['Non', 'Oui']
    }
)

# Ajouter les CPT au modèle
model.add_cpds(
    cpd_grippe,
    cpd_fievre,
    cpd_toux,
    cpd_fatigue
)

# Vérification du modèle
if model.check_model():
    inference = VariableElimination(model)
else:
    st.error("Erreur dans le modèle bayésien")

# =====================================
# INTERFACE UTILISATEUR
# =====================================

st.markdown("---")
st.subheader("🧾 Sélection des symptômes")

col1, col2, col3 = st.columns(3)

with col1:
    fievre = st.checkbox("Fièvre")

with col2:
    toux = st.checkbox("Toux")

with col3:
    fatigue = st.checkbox("Fatigue")

st.markdown("---")

# =====================================
# DIAGNOSTIC
# =====================================

if st.button("🔍 Diagnostiquer"):

    evidence = {
        'Fièvre': 'Oui' if fievre else 'Non',
        'Toux': 'Oui' if toux else 'Non',
        'Fatigue': 'Oui' if fatigue else 'Non'
    }

    result = inference.query(
        variables=['Grippe'],
        evidence=evidence
    )

    prob_non = result.values[0] * 100
    prob_oui = result.values[1] * 100

    st.subheader("📊 Résultat du diagnostic")

    st.success(f"Probabilité de grippe : {prob_oui:.2f}%")
    st.info(f"Probabilité sans grippe : {prob_non:.2f}%")

    # =====================================
    # INTERPRÉTATION
    # =====================================

    if prob_oui >= 70:
        st.warning("⚠️ Forte probabilité de grippe")
    elif prob_oui >= 40:
        st.warning("🟡 Probabilité moyenne de grippe")
    else:
        st.success("🟢 Faible probabilité de grippe")

    # =====================================
    # AFFICHAGE GRAPHIQUE
    # =====================================

    labels = ['Grippe', 'Pas Grippe']
    values = [prob_oui, prob_non]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct='%1.1f%%')

    st.pyplot(fig)

# =====================================
# INFORMATIONS
# =====================================

st.markdown("---")

with st.expander("ℹ️ Informations sur le projet"):
    st.write("""
    ### Réseau Bayésien

    Ce projet utilise un réseau bayésien pour représenter les relations
    probabilistes entre une maladie et plusieurs symptômes.

    ### Symptômes utilisés
    - Fièvre
    - Toux
    - Fatigue

    ### Méthode utilisée
    - Inférence probabiliste
    - Probabilités conditionnelles
    - Réseau bayésien avec pgmpy

    ### Technologies
    - Python
    - Streamlit
    - pgmpy
    - Matplotlib
    """)

# =====================================
# PIED DE PAGE
# =====================================

st.markdown("---")
st.caption("Projet IA - Diagnostic médical avec réseau bayésien")


