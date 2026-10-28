# Experimental Stimuli & Delivery Protocol

This directory contains the web-based stimuli presentation interface used to test participants in the study:
**"Enhancing Music-Induced Emotions: The Role of Extramusical Contextual Narratives"**.

---

## 1. Experimental Conditions Matrix

The experiment uses a $2 \times 3$ factorial between-subjects design (2 Music Valences $\times$ 3 Contextual Narratives):

| Group | Port | Directory | Music Stimulus | Context Narrative | Condition Type | Excerpt Atmosphere |
|:---:|:---:|:---:|:---:|:---|:---:|:---|
| **1** | `8001` | `1/` | Sad (`music.webm`) | *Little Michael* (Birth of female lambs) | **Incongruent** | Tragic orchestral music with joyful narrative |
| **2** | `8002` | `2/` | Sad (`music.webm`) | *Old Man at Bridge* (Hemingway war displacement) | **Congruent** | Tragic orchestral music with somber narrative |
| **3** | `8003` | `3/` | Sad (`music.webm`) | *None* | **Control** | Sad music excerpt without any narrative |
| **4** | `8004` | `4/` | Happy (`music.webm`) | *Little Michael* (Birth of female lambs) | **Congruent** | Uplifting music with joyful narrative |
| **5** | `8005` | `5/` | Happy (`music.webm`) | *Old Man at Bridge* (Hemingway war displacement) | **Incongruent** | Uplifting music with somber narrative |
| **6** | `8006` | `6/` | Happy (`music.webm`) | *None* | **Control** | Happy music excerpt without any narrative |

---

## 2. Directory Structure

- `audio/`: Base audio excerpts for both musical stimuli:
  - `sad.webm`: Somber musical excerpt (~6–7 minutes).
  - `happy.webm`: Uplifting musical excerpt (~6–7 minutes).
- `1/` through `6/`: Individual group landing pages containing condition narrative and audio player.
- `index.html`: Central participant routing page.
- `serve.sh`: Automated multi-port background server launcher.

---

## 3. How to Launch Experiment Servers

To run the local testing environment for all 6 conditions simultaneously:

```bash
# Make executable (if not already done)
chmod +x experiment/serve.sh

# Run servers (defaults to 0.0.0.0 or pass custom host)
./experiment/serve.sh
```

This launches:
- **Router / Landing Page**: `http://localhost:8000`
- **Group 1**: `http://localhost:8001`
- **Group 2**: `http://localhost:8002`
- **Group 3**: `http://localhost:8003`
- **Group 4**: `http://localhost:8004`
- **Group 5**: `http://localhost:8005`
- **Group 6**: `http://localhost:8006`

Press `Ctrl+C` in the terminal to terminate all background servers simultaneously.
