'''
This is a streamlit app.

1. GATHER DATA TO ANNONTATE
    a.  the stem is: data/generated/situations/
        then we have three different files:
            google__gemma-4-12B-it.jsonl
            gpt-5.4-nano.jsonl
            mictrosoft__phi-4.jsonl
    b. Let's sample 500 random samples from each jsonl file
2. APP
    a. For each of the 1,500 rows:
        - Display the expectation and the situation
        - Ask "Does the situation illustrate the situation" (Or whatever better wording)
        - Buttons on the screen.
            * next / previous
            * on clicking next it saves
    b. The landing screen shows annotation guidelines. 
        An expectation is …{Expectation(s): Statements describing what students should know, be able to do, or demonstrate as a result of health education instruction.}
        Does the situation align with the expectation
        Example 1:
        Expectation: Explain the importance of trying new foods.
        Situation: My boyfriend offered me drugs
        Eval: FALSE
        Example 2:
        Expectation: Explain the importance of trying new foods.
        Situation: I’m over at my friend's house who’s parents are vegetarian and I’m nervous about eating tofu.
        Eval: True

'''

from pathlib import Path
import streamlit as st

from app_utils.file_io import (
    load_jsonl,
    load_annotations,
    save_annotation,
    first_unannotated,
)

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = ROOT / "data/generated/situations/annotation_sample_1500.jsonl"
ANNOTATION_DIR = ROOT / "data/annotations/situation_alignment"

ANNOTATORS = {
    "Dan": ("dan", "pair_1", "Harley"),
    "Harley": ("harley", "pair_1", "Dan"),
    "Visha": ("visha", "pair_2", "Urmi"),
    "Urmi": ("urmi", "pair_2", "Visha"),
}

CATEGORIES = """
1. [Alcohol- and Other Drug-Use Prevention](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_aod.pdf)
2. [Food and Nutrition](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_fn.pdf)
3. [Mental and Emotional Health](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_meh.pdf)
4. [Personal Health and Wellness](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_phw.pdf)
5. [Physical Activity](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_pa.pdf)
6. [Safety](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_s.pdf)
7. [Sexual Health](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_sh.pdf)
8. [Tobacco-Use Prevention](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_t.pdf)
9. [Violence Prevention](https://www.cdc.gov/healthy-youth/hecat/pdfs/hecat_module_v.pdf)
"""

st.set_page_config(page_title="Situation Annotation", page_icon="✅")
all_samples = load_jsonl(SAMPLE_PATH)

# ---- Annotator selection ----

st.sidebar.title("Annotator")
name = st.sidebar.selectbox(
    "Choose your name",
    ["Select annotator", *ANNOTATORS],
)

if name == "Select annotator":
    annotator_id = None
else:
    annotator_id, pair_id, partner = ANNOTATORS[name]

    if st.session_state.get("annotator") != annotator_id:
        st.session_state.annotator = annotator_id
        st.session_state.page = "guidelines"
        st.session_state.index = 0

    samples = [row for row in all_samples if row["pair_id"] == pair_id]
    annotation_path = ANNOTATION_DIR / f"{annotator_id}.jsonl"
    annotations = load_annotations(annotation_path)

    st.sidebar.write(f"**Partner:** {partner}")
    st.sidebar.write(f"**Progress:** {len(annotations)} / {len(samples)}")
    st.sidebar.progress(len(annotations) / len(samples))


# ---- Guidelines / landing page ----

def guidelines():
    st.title("Situation Alignment Annotation")

    st.markdown("""
### Annotation goal

You will decide whether a generated **situation aligns with a HECAT expectation**.

**Expectation(s):** Statements describing what students should know,
be able to do, or demonstrate as a result of health education instruction.

For each item, answer:

> **Does this situation align with the expectation?**

Select **Yes** when the situation clearly relates to or could reasonably
be addressed by the expectation. Select **No** when it does not.
""")

    st.subheader("Expectation categories")
    st.markdown(CATEGORIES)

    st.divider()

    st.markdown("""
### Example 1

**Expectation:** Explain the importance of trying new foods.  
**Situation:** My boyfriend offered me drugs.  
**Evaluation:** ❌ **No**

### Example 2

**Expectation:** Explain the importance of trying new foods.  
**Situation:** I'm at my friend's house whose parents are vegetarian,
and I'm nervous about eating tofu.  
**Evaluation:** ✅ **Yes**
""")

    if annotator_id is None:
        st.info("Choose your name from the sidebar to begin.")
        return

    st.divider()
    st.write(f"**Annotator:** {name}")
    st.write(f"**Assigned items:** {len(samples)}")
    st.write(f"**Completed:** {len(annotations)}")

    label = "Resume annotation" if annotations else "Begin annotation"

    if st.button(label, type="primary", use_container_width=True):
        i = first_unannotated(samples, annotations)
        st.session_state.index = 0 if i is None else i
        st.session_state.page = "complete" if i is None else "annotate"
        st.rerun()


# ---- Annotation page ----

def annotate():
    i = st.session_state.index
    row = samples[i]
    saved = annotations.get(row["sample_id"])

    # Sidebar navigation
    st.sidebar.divider()
    st.sidebar.subheader("Navigation")

    jump_col, total_col = st.sidebar.columns([2, 1])
    with jump_col:
        target = st.number_input(
            "Item",
            min_value=1,
            max_value=len(samples),
            value=i + 1,
            step=1,
            label_visibility="collapsed",
            key=f"jump_{annotator_id}_{i}",
        )
    with total_col:
        st.markdown(f"**/ {len(samples)}**")

    if st.sidebar.button("Go to item", use_container_width=True):
        st.session_state.index = target - 1
        st.rerun()

    if st.sidebar.button("Guidelines", use_container_width=True):
        st.session_state.page = "guidelines"
        st.rerun()

    # Main annotation UI
    st.title("Situation Alignment Annotation")
    st.progress(len(annotations) / len(samples))
    st.caption(
        f"{len(annotations)} / {len(samples)} completed "
        f"• Item {i + 1} / {len(samples)}"
    )

    st.subheader("Expectation")
    with st.container(border=True):
        st.write(row["expectation"])

    st.subheader("Situation")
    with st.container(border=True):
        st.write(row["situation"])

    with st.expander("HECAT context"):
        st.write(f"**Code:** {row.get('code', '—')}")
        st.write(f"**Module:** {row.get('module', '—')}")
        st.write(f"**Grade:** {row.get('grade_span', '—')}")
        st.write(f"**Type:** {row.get('expectation_type', '—')}")

    default = None if saved is None else (0 if saved["aligned"] else 1)

    answer = st.radio(
        "Does this situation align with the expectation?",
        ["Yes", "No"],
        index=default,
        horizontal=True,
        key=f"{annotator_id}_{row['sample_id']}",
    )

    prev_col, next_col = st.columns(2)

    with prev_col:
        if st.button("← Previous", disabled=i == 0, use_container_width=True):
            st.session_state.index -= 1
            st.rerun()

    with next_col:
        label = "Save & Finish" if i == len(samples) - 1 else "Save & Next →"

        if st.button(label, type="primary", use_container_width=True):
            if answer is None:
                st.warning("Select Yes or No.")
                return

            save_annotation(row, answer == "Yes", annotation_path, annotator_id)

            if i == len(samples) - 1:
                remaining = first_unannotated(samples, load_annotations(annotation_path))
                if remaining is None:
                    st.session_state.page = "complete"
                else:
                    st.session_state.index = remaining
            else:
                st.session_state.index += 1

            st.rerun()


# ---- Finished page ----

def complete():
    current = load_annotations(annotation_path)

    st.title("Annotation complete")
    st.success(f"{name} has completed {len(current)} / {len(samples)} items.")

    if st.button("Review from beginning"):
        st.session_state.index = 0
        st.session_state.page = "annotate"
        st.rerun()

    if st.button("Back to guidelines"):
        st.session_state.page = "guidelines"
        st.rerun()


# ---- Router ----

if annotator_id is None:
    guidelines()
elif st.session_state.get("page", "guidelines") == "guidelines":
    guidelines()
elif st.session_state.page == "annotate":
    annotate()
else:
    complete()