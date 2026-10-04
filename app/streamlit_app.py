from __future__ import annotations

import streamlit as st

import page_map
import page_overview
import page_reposition
import page_results
import page_zone

st.set_page_config(page_title="SurgeMap", page_icon="🚕", layout="wide")

pages = [
    st.Page(page_overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(page_map.render, title="Demand forecast map", icon="🗺️", url_path="map"),
    st.Page(page_zone.render, title="Zone explorer", icon="📈", url_path="zone"),
    st.Page(page_results.render, title="Forecast accuracy", icon="📊", url_path="accuracy"),
    st.Page(page_reposition.render, title="Repositioning", icon="🚕", url_path="repositioning"),
]
st.navigation(pages).run()
