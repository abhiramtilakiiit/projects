# Experimental Data Dictionary

This directory contains the experimental records collected during the study:
**"Enhancing Music-Induced Emotions: The Role of Extramusical Contextual Narratives"** ($N = 50$).

---

## 1. Directory Layout

- **`raw/`**:
  - `survey_responses_raw.txt`: Original unedited tab/comma survey export containing all 52 collected entries including test pilots.
  - `survey_responses_raw.xlsx`: Original spreadsheet format of survey submissions.
- **`processed/`**:
  - `experiment_data_clean.csv`: Cleaned dataset ($N = 50$) used for all statistical models and hypothesis testing.

---

## 2. Data Cleaning Protocol

As documented during data curation:
1. **Pilot entries filtered**: Removed initial test/debug responses ($ID \le 5$) submitted prior to standardized protocol execution.
2. **Invalid response validation**: Checked and removed entries with incorrect adjective inputs or incomplete ratings.
3. **Localhost port standardization**: Ensured the condition port ($8001$–$8006$) was accurately extracted from the participant's logged IP/URL string.

---

## 3. Schema & Variables

The cleaned dataset `experiment_data_clean.csv` contains 97 fields structured as follows:

### Metadata & Demographics (Columns 0–3)
- `ID`: Unique participant numeric identifier.
- `Start time`: Timestamp when the participant began the survey.
- `Completion time`: Timestamp when the survey was completed (~20–25 minutes duration).
- `Email`: Participant email field (anonymized as `anonymous`).

### Baseline Affect Battery: PANAS Set 1 (Columns 4–23)
Self-report mood rating on 20 standard PANAS adjectives prior to music exposure, scored on a **5-point Likert scale**:
- `1`: *Very slightly or not at all*
- `2`: *A little*
- `3`: *Moderately*
- `4`: *Quite a bit*
- `5`: *Extremely*

**10 Positive Affect (PA) Adjectives**:  
`attentive`, `inspired`, `enthusiastic`, `excited`, `interested`, `active`, `proud`, `determined`, `strong`, `alert`

**10 Negative Affect (NA) Adjectives**:  
`nervous`, `guilty`, `upset`, `jittery`, `irritable`, `ashamed`, `hostile`, `afraid`, `distressed`, `scared`

### Word Recall Task (Column 24)
- `List as many adjectives as possible...`: Free-text field where participants had 2 minutes immediately following musical listening to recall as many adjectives from the initial PANAS battery as possible.

### Picture Judgment Task (Columns 25–74)
Ratings of 25 prototypical facial expressions (generated via morphing algorithms; Vanger et al., 1998). Participants rated each face on two dimensions using a **7-point Likert scale**:
- `Happy`, `Happy2`, ..., `Happy25`
- `Upset2`, `Upset3`, ..., `Upset26`

**Scale mapping**:
- `1`: *Not at all* / *Very slightly or not at all*
- `2`: *Slightly*
- `3`: *Somewhat*
- `4`: *Moderately*
- `5`: *Quite a bit* / *Quite*
- `6`: *Really*
- `7`: *Extremely*

### Post-Exposure Affect Battery: PANAS Set 2 (Columns 75–94)
Second administration of the 20 PANAS adjectives following music listening to measure affective shift ($\Delta Z$). Columns are labeled with suffix `2` (e.g., `attentive2`, ..., `alert2`, with `upset27`).

### Qualitative & Assignment Fields (Columns 95–96)
- `Describe what kind of thoughts and impressions went through your mind while listening to the music:`: Qualitative open-ended text response.
- `Please put the IP address you were provided below\n`: The host/port address providing the stimulus condition (ports `8001` through `8006`).
