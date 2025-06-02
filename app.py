import app as st
from starcraft_predictor import load_scorer

st.title("Starcraft Predictor")


replay_scorer = load_scorer()
uploaded_file = st.file_uploader("Upload a replay file", type=["SC2Replay"])
if uploaded_file is not None:
    figure = replay_scorer.score_replay(uploaded_file)
    st.pyplot(figure)
