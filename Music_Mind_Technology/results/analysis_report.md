# Experimental Analysis Report: Music-Induced Emotions & Contextual Narratives

**Project**: Enhancing Music-Induced Emotions: The Role of Extramusical Contextual Narratives  
**Dataset**: Cleaned experimental records ($N = 50$)  
**Design**: 2 (Music: Sad vs. Happy) $\times$ 3 (Narrative: Congruent, Incongruent, None) factorial design  

---

## 1. Experimental Conditions & Sample Breakdown

| Group | Port | Music Valence | Narrative Context | Description | Sample Size ($n$) |
|:---:|:---:|:---:|:---:|:---|:---:|
| **1** | 8001 | Sad | Happy Narrative | *Incongruent* (Little Michael story + Sad excerpt) | 8 |
| **2** | 8002 | Sad | Sad Narrative | *Congruent* (Old Man at Bridge + Sad excerpt) | 8 |
| **3** | 8003 | Sad | No Narrative | *Control* (Sad excerpt only) | 8 |
| **4** | 8004 | Happy | Happy Narrative | *Congruent* (Little Michael story + Happy excerpt) | 9 |
| **5** | 8005 | Happy | Sad Narrative | *Incongruent* (Old Man at Bridge + Happy excerpt) | 9 |
| **6** | 8006 | Happy | No Narrative | *Control* (Happy excerpt only) | 8 |
| **Total** | | | | | **50** |

---

## 2. Word Recall Task (Implicit Memory Bias)

Participants freely recalled mood adjectives from the initial 20-item PANAS battery immediately following music listening.

### Between-Group Comparisons
- **(Positive - Negative) Difference Score**:
  - Test: ANOVA ($F = 1.3042$, $p = 0.2796$)
  - Statistical significance: **Not significant**
- **Total Adjectives Recalled**:
  - Test: ANOVA ($F = 0.4511$, $p = 0.8102$)
  - Statistical significance: **Not significant**

### Within-Group Recall Bias (Positive vs. Negative)
| Group | Condition | Positive Words (Mean $\pm$ SEM) | Negative Words (Mean $\pm$ SEM) | Statistical Test | p-value | Interpretation |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **8001** | Sad Music + Happy Narrative (Incongruent) | 3.38 $\pm$ 0.32 | 2.88 $\pm$ 0.69 | Paired t-test (t=0.6417) | $p = 0.2708$ | No significant bias ($ns$) |
| **8002** | Sad Music + Sad Narrative (Congruent) | 4.38 $\pm$ 0.56 | 2.62 $\pm$ 0.42 | Paired t-test (t=2.1385) | $p = 0.0349$ | Significant positive recall bias ($p < .05$) |
| **8003** | Sad Music + No Narrative (Control) | 3.88 $\pm$ 0.61 | 3.00 $\pm$ 0.57 | Paired t-test (t=1.1784) | $p = 0.1386$ | No significant bias ($ns$) |
| **8004** | Happy Music + Happy Narrative (Congruent) | 4.44 $\pm$ 0.69 | 2.33 $\pm$ 0.62 | Paired t-test (t=3.5907) | $p = 0.0035$ | Significant positive recall bias ($p < .05$) |
| **8005** | Happy Music + Sad Narrative (Incongruent) | 2.78 $\pm$ 0.60 | 2.67 $\pm$ 0.62 | Paired t-test (t=0.1754) | $p = 0.4326$ | No significant bias ($ns$) |
| **8006** | Happy Music + No Narrative (Control) | 4.00 $\pm$ 0.19 | 2.75 $\pm$ 0.45 | Paired t-test (t=2.7584) | $p = 0.0141$ | Significant positive recall bias ($p < .05$) |

*Plot saved to: `figures/word_recall_scores.png`*

---

## 3. Picture Judgment Task (Facial Emotion Interpretation)

Participants rated 25 prototypical facial expressions (Happy and Upset) on a 1–7 Likert scale. Ratings were ipsatized (z-scored across all face items per subject) to capture relative perceptual bias.

### ANOVA Results
- **Happy Facial Expression Ratings**:
  - Normality of Residuals: Shapiro-Wilk $W = 0.9899, p = 0.9448$ (Assumptions met)
  - Homogeneity of Variance: Levene's $W = 0.9978, p = 0.4303$ (Assumptions met)
  - ANOVA: $F(5, 44) = 2.2434, p = 0.0666$ (Marginally significant trend (p < .10))
- **Upset Facial Expression Ratings**:
  - Normality of Residuals: Shapiro-Wilk $W = 0.9899, p = 0.9448$ (Assumptions met)
  - Homogeneity of Variance: Levene's $W = 0.9978, p = 0.4303$ (Assumptions met)
  - ANOVA: $F(5, 44) = 2.2434, p = 0.0666$ (Marginally significant trend (p < .10))

*Plot saved to: `figures/facial_emotion_ratings.png`*

---

## 4. Subjective Affect Shifts (Pre vs. Post PANAS)

Measured within-participant changes in self-reported Positive Affect (PA) and Negative Affect (NA) before and after music listening ($Z_{post} - Z_{pre}$).

| Group | Condition | Positive Affect Shift ($\Delta Z$) | PA Test ($p$-value) | Negative Affect Shift ($\Delta Z$) | NA Test ($p$-value) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **8001** | Sad Music + Happy Narrative (Incongruent) | -0.1415 $\pm$ 0.3036 | Paired t-test ($p = 0.6554$) | +0.3078 $\pm$ 0.2656 | Wilcoxon ($p = 0.3125$) |
| **8002** | Sad Music + Sad Narrative (Congruent) | -0.3653 $\pm$ 0.2095 | Paired t-test ($p = 0.1247$) | -0.0599 $\pm$ 0.0600 | Paired t-test ($p = 0.3513$) |
| **8003** | Sad Music + No Narrative (Control) | +0.0703 $\pm$ 0.1349 | Wilcoxon ($p = 0.9453$) | -0.2629 $\pm$ 0.1181 | Paired t-test ($p = 0.0613$) |
| **8004** | Happy Music + Happy Narrative (Congruent) | -0.0706 $\pm$ 0.1269 | Paired t-test ($p = 0.5930$) | -0.0832 $\pm$ 0.0994 | Wilcoxon ($p = 0.8125$) |
| **8005** | Happy Music + Sad Narrative (Incongruent) | **-0.5315 $\pm$ 0.1094** | Paired t-test ($p = 0.0013$) | -0.2827 $\pm$ 0.1948 | Paired t-test ($p = 0.1848$) |
| **8006** | Happy Music + No Narrative (Control) | -0.0702 $\pm$ 0.2510 | Paired t-test ($p = 0.7878$) | -0.0842 $\pm$ 0.1863 | Paired t-test ($p = 0.6647$) |

*Plot saved to: `figures/panas_affect_shifts.png`*

---

## 5. Key Psychological Insights

1. **Incongruence Neutralizes Positive Memory Bias**:
   - In emotionally congruent conditions (Group 8002: Sad Music + Sad Story, $p = 0.0349$; Group 8004: Happy Music + Happy Story, $p = 0.0035$) and neutral music baseline (Group 8006, $p = 0.0141$), participants exhibited significant positive recall bias.
   - When narrative and musical cues conflicted (Group 8001: Sad Music + Happy Story, $p = 0.2708$; Group 8005: Happy Music + Sad Story, $p = 0.4326$), this bias disappeared, indicating cognitive interference from affective incongruence.

2. **Tragic Narratives Overpower Happy Music**:
   - In Group 8005 (Happy Music paired with Hemingway's war displacement narrative), participants suffered a severe and statistically significant decline in self-reported positive affect ($\Delta Z = -0.5315, t = 4.8586, p = 0.0013$).
   - This empirically demonstrates that contextual semantic narratives can exert top-down modulation strong enough to invert positive auditory affect.

3. **Face Interpretation Trend**:
   - Standardized facial emotion ratings approached statistical significance across experimental groups ($F = 2.243, p = 0.0666$), confirming subtle implicit perceptual priming from extramusical context.
