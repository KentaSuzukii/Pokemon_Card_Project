# Scripts/run_streamlit.py

import os
import sys
import streamlit.web.cli as stcli

port = os.environ.get("PORT", "8501")
sys.argv = [
    "streamlit", "run", "Streamlit_app/app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.enableCORS=false"
]

sys.exit(stcli.main())
